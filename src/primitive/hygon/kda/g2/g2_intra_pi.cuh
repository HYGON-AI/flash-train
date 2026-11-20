// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// G2 S4-a: chunk-intra backward (FLA chunk_kda_bwd_kernel_intra role) in P3's pi layout.
//
// CTA = (chunk, head), 4 waves, wave a owns rows [16a, 16a+16) -- the P3 division, and all
// 128 columns (no K split): the loop runs over 8 column tiles kt of 16 columns.  Output
// C-fragments use the pi trick: the B lane with hardware column n reads source column
// pi(n), so lane l = 16q + m holds row m, columns 16kt + 4q + e (4 CONSECUTIVE columns)
// -> every global read/write of dq/dk/dg/g/q/k is a 16/8-byte vector op.
//
// fp32 MMAC 16x16x8: A lane (m,q) holds A[m][2q+i], B lane (n,q) holds B[2q+i][n].  The
// contraction index is PERMUTED: hardware k-slot (q, i) of step hh maps to token
// j = 16b + 4q + 2hh + i.  Then an A-fragment is 4 consecutive dA entries of row m (one
// 16-byte load; in S4-b exactly P3's C-pi dAkk registers) and a B-fragment is 4
// consecutive j of one column (one ds_read_b128 from a column-major table).
//
// Numerics.  Every exponent is a difference of g inside one chunk, referenced to the MID
// row M(b) = 16b + 8 of a 16-token block (never the retracted full-chunk 2^{-g} split):
//   d_j = g_j - g_M(b)   (rounded ONCE per element; |d| <= 8 * 7.21)
//   tables (per kt, each wave writes its own 16 rows = block b):
//     TA[j][c] = k_j 2^{-d_j},  TQ[j][c] = q_j 2^{d_j},  TK[j][c] = beta_j k_j 2^{d_j}
//   part A (j <= i, blocks b <= a): B = TA * 2^{g_M(a) - g_M(b)}   (factor 1 for b = a)
//                                   C *= 2^{d_i}
//   part B (j >= i, blocks b >= a): B = TQ|TK * 2^{g_M(b) - g_M(a)} (factor 1 for b = a)
//                                   C *= 2^{-d_i}
// The diagonal block is therefore FLA's safe-gate diagonal term exactly (same mid-row
// pieces; for i = j the rounded exponents cancel), and the C scale reuses the table's
// two exps (the lane's own row i is a row of its own block).  Off-diagonal blocks share
// the accumulator; the extra factor 2^{g_M(a) - g_M(b)} <= 1 may flush for far blocks,
// whose final contribution is then < 2^-68 of the diagonal term.
//
// Determinism: every sum has a fixed order (MMAC chains per hh, db over kt then xsum16).
// In-place use (o_* == input) is allowed: each element is read and written by one lane.
#pragma once
#include "g2_common.h"

namespace g2 {
namespace ipi {

typedef float f2v __attribute__((ext_vector_type(2)));
__device__ __forceinline__ f4 mm8(f2v a, f2v b, f4 c) {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__)
    return __builtin_amdgcn_mmac_f32_16x16x8f32(a, b, c);
#elif defined(__gfx938__) && __has_builtin(__builtin_hcu_mmac_16x16x8_f32_lit_lts)
    // gfx938's native MMAC builtin with lit = lts = false: the same operation as gfx936's
    // v_mmac (bit-identical); the generic amdgcn builtin lowers with lit / lts set on gfx938.
    return __builtin_hcu_mmac_16x16x8_f32_lit_lts(a, b, c, false, false);
#else
    __builtin_trap();  // gfx936-only matrix instruction: the KDA Primitives reject other archs
    return c;
#endif
}
__device__ __forceinline__ float ex2(float x) { return __builtin_amdgcn_exp2f(x); }
// 2^(a-b) with the rounding error of the fp32 difference folded back in (TwoSum): the
// off-diagonal block factor spans up to 3 blocks of decay, where fl(a-b) alone loses
// ~ulp(|a-b|) of the exponent (measured: 1.1x FLA's oracle error on the real case).
__device__ __forceinline__ float ex2_diff(float a, float b) {
    const float s = a - b, bp = s - a, ap = s - bp;
    const float t = (a - ap) + (-b - bp);
    return __builtin_amdgcn_exp2f(s) * (1.f + t * 0.69314718f);
}
__device__ __forceinline__ f2v lo2(f4 v) { return f2v{v[0], v[1]}; }
__device__ __forceinline__ f2v hi2(f4 v) { return f2v{v[2], v[3]}; }

constexpr int TS = 68;                        // table row stride (floats): 64 j + 4 pad
// tables are indexed by HARDWARE B column n (row n holds source column pi(n)); with
// TS = 68 a b128 read by 8 consecutive lanes covers 32 distinct banks
struct Smem {
    float TA[2][16 * TS], TQ[2][16 * TS], TK[2][16 * TS];
    float gM[2][4][16];                       // [buf][block][n]: g[16b+8][pi(n)]
};

struct IntraPiParams {
    const bf16 *q, *k;
    const float *g, *beta, *dAqk, *dAkk, *dq, *dk, *db, *dg;
    float *o_dq, *o_dk, *o_db, *o_dg;
    int H, NT;
};

template <int A>
__device__ __forceinline__ void body(const IntraPiParams& p, Smem& sm) {
    const int tid = threadIdx.x, l = tid & 63, m = l & 15, q = l >> 4;
    const int i = blockIdx.x, hh = blockIdx.y, H = p.H;
    const int R = H * kD, RA = H * kBT;
    const long tok0 = (long)i * kBT;
    const size_t cb = g2_rm_tok(tok0, hh, H, kD), cbA = g2_rm_tok(tok0, hh, H, kBT);
    const int row = 16 * A + m;
    const int ro = row * R + 4 * q;
    const bf16 *q_ = p.q + cb, *k_ = p.k + cb;
    const float *g_ = p.g + cb, *dq_ = p.dq + cb, *dk_ = p.dk + cb, *dg_ = p.dg + cb;
    float *odq = p.o_dq + cb, *odk = p.o_dk + cb, *odg = p.o_dg + cb;
    const float* aq_ = p.dAqk + cbA;
    const float* ak_ = p.dAkk + cbA;
    const float beta_i = p.beta[(tok0 + row) * H + hh];

    // A fragments, kt-invariant.  part A: row `row`, j = 16b + 4q + {0..3}; part B:
    // transposed, dA[16b + 4q + x][row].  Diagonal blocks masked like FLA
    // (j <= i part A, j >= i part B, both local within the block).
    f4 aqA[A + 1], akA[A + 1], aqB[4 - A], akB[4 - A];
    #pragma unroll
    for (int b = 0; b <= A; ++b) {
        aqA[b] = ld16f(aq_ + row * RA + 16 * b + 4 * q);
        akA[b] = ld16f(ak_ + row * RA + 16 * b + 4 * q);
        if (b == A) {
            #pragma unroll
            for (int x = 0; x < 4; ++x)
                if (4 * q + x > m) { aqA[b][x] = 0.f; akA[b][x] = 0.f; }
        }
    }
    #pragma unroll
    for (int b = A; b < 4; ++b) {
        #pragma unroll
        for (int x = 0; x < 4; ++x) {
            const int j = 16 * b + 4 * q + x;
            const bool keep = b > A || 4 * q + x >= m;
            aqB[b - A][x] = keep ? aq_[j * RA + row] : 0.f;
            akB[b - A][x] = keep ? ak_[j * RA + row] : 0.f;
        }
    }

    float dbacc = 0.f;
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {
        const int buf = kt & 1, kc = 16 * kt;
        const f4 gv = ld16f(g_ + ro + kc);
        const v4bh kk = ld8(k_ + ro + kc), qq = ld8(q_ + ro + kc);
        // own-row exps, shared by the tables and the C scales: ep = 2^{d}, en = 2^{-d}
        f4 ep, en;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float d = gv[e] - __shfl(gv[e], 16 * q + 8, 64);
            ep[e] = ex2(d);
            en[e] = ex2(-d);
        }
        // stage this wave's rows; table row for source column 4q+e is pi(4q+e) = 4e+q
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float kf = bf2f(kk[e]), qf = bf2f(qq[e]);
            const int o = (4 * e + q) * TS + row;
            sm.TA[buf][o] = kf * en[e];
            sm.TQ[buf][o] = qf * ep[e];
            sm.TK[buf][o] = kf * beta_i * ep[e];
        }
        if (m == 8) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm.gM[buf][A][4 * e + q] = gv[e];
        }
        lds_barrier();

        const float gm_n = sm.gM[buf][A][m];               // g[M(A)][pi(m)] of this B lane
        f4 cq0 = {0.f, 0.f, 0.f, 0.f}, cq1 = cq0, ck0 = cq0, ck1 = cq0, ct0 = cq0, ct1 = cq0;
        #pragma unroll
        for (int b = 0; b <= A; ++b) {
            f4 t = *reinterpret_cast<const f4*>(&sm.TA[buf][m * TS + 16 * b + 4 * q]);
            if (b < A) t *= ex2_diff(gm_n, sm.gM[buf][b][m]);
            cq0 = mm8(lo2(aqA[b]), lo2(t), cq0);
            ck0 = mm8(lo2(akA[b]), lo2(t), ck0);
            cq1 = mm8(hi2(aqA[b]), hi2(t), cq1);
            ck1 = mm8(hi2(akA[b]), hi2(t), ck1);
        }
        #pragma unroll
        for (int b = A; b < 4; ++b) {
            f4 tq = *reinterpret_cast<const f4*>(&sm.TQ[buf][m * TS + 16 * b + 4 * q]);
            f4 tk = *reinterpret_cast<const f4*>(&sm.TK[buf][m * TS + 16 * b + 4 * q]);
            if (b > A) {
                const float s = ex2_diff(sm.gM[buf][b][m], gm_n);
                tq *= s;
                tk *= s;
            }
            ct0 = mm8(lo2(aqB[b - A]), lo2(tq), ct0);
            ct1 = mm8(lo2(akB[b - A]), lo2(tk), ct1);
            ct0 = mm8(hi2(aqB[b - A]), hi2(tq), ct0);
            ct1 = mm8(hi2(akB[b - A]), hi2(tk), ct1);
        }

        const f4 dqi = ld16f(dq_ + ro + kc), dki = ld16f(dk_ + ro + kc), dgi = ld16f(dg_ + ro + kc);
        f4 oq, ok, og, dbp;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float kf = bf2f(kk[e]), qf = bf2f(qq[e]);
            const float dq2 = (cq0[e] + cq1[e]) * ep[e];
            const float dk2 = (ck0[e] + ck1[e]) * ep[e];
            const float dkt = (ct0[e] + ct1[e]) * en[e];
            dbp[e] = dk2 * kf;
            const float dk2b = dk2 * beta_i;
            oq[e] = dqi[e] + dq2;
            ok[e] = dki[e] + dk2b + dkt;
            og[e] = dgi[e] + qf * dq2 + (dk2b - dkt) * kf;
        }
        dbacc += (dbp[0] + dbp[1]) + (dbp[2] + dbp[3]);
        *reinterpret_cast<f4*>(odq + ro + kc) = oq;
        *reinterpret_cast<f4*>(odk + ro + kc) = ok;
        *reinterpret_cast<f4*>(odg + ro + kc) = og;
    }
    dbacc = xsum16(dbacc);
    if (q == 0) {
        const size_t t = (size_t)(tok0 + row) * H + hh;
        p.o_db[t] = p.db[t] + dbacc;
    }
}

__global__ void __launch_bounds__(256) g2_intra_pi_kernel(IntraPiParams p) {
    __shared__ __align__(16) Smem sm;
    // wave-uniform (SGPR) branch: one instantiation per row block; every instantiation
    // executes the same 8 CTA barriers
    switch (__builtin_amdgcn_readfirstlane(threadIdx.x >> 6)) {
        case 0: body<0>(p, sm); break;
        case 1: body<1>(p, sm); break;
        case 2: body<2>(p, sm); break;
        default: body<3>(p, sm); break;
    }
}

}  // namespace ipi

extern "C" int g2_intra_pi(const void* q, const void* k, const void* g, const void* beta,
                           const void* dAqk, const void* dAkk, const void* dq,
                           const void* dk, const void* db, const void* dg,
                           void* o_dq, void* o_dk, void* o_db, void* o_dg,
                           int B, int T, int H, hipStream_t s) {
    if (T % kBT || B != 1) return -1;
    ipi::IntraPiParams p{(const bf16*)q, (const bf16*)k, (const float*)g, (const float*)beta,
                         (const float*)dAqk, (const float*)dAkk, (const float*)dq,
                         (const float*)dk, (const float*)db, (const float*)dg,
                         (float*)o_dq, (float*)o_dk, (float*)o_db, (float*)o_dg,
                         H, T / kBT};
    hipLaunchKernelGGL(ipi::g2_intra_pi_kernel, dim3(T / kBT, H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}

}  // namespace g2
