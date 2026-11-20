// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// S3-12: HIP fwd_intra -- ONE kernel per (chunk, head) replacing FLA's
// sub_chunk + inter_solve_fused + recompute_w_u and the five forward layout
// conversions.  Operates on PREPROCESSED inputs (l2normed qn/kn, chunk-cumsum
// g, post-sigmoid beta -- F0 stays FLA this round; folding it in is next).
//
// Math (fla chunk_intra.py, safe-gate tier):
//   diag 16x16 blocks, reference = the sub-block's MIDDLE row (offset 8):
//     Aqk_d = (q*2^(g-gm)) @ (k*2^(gm-g))^T * scale      (col <= row, else 0)
//     L_d   = (k*2^(g-gm)) @ (k*2^(gm-g))^T * beta[row]  (strict lower)
//     Ai_s  : in-place recursion Ai[r] = -L[r] + sum_{j<r} (-L[r][j]) Ai[j],
//             rows 0/1 keep the plain -L init, diagonal += 1 at the end
//   off-diagonal blocks (rb > cb), reference = the row block's FIRST row:
//     Aqk_o = (q*2^(g-gr)) @ (k*2^(gr-g))^T * scale
//     Akk_o = same with k on the row side, scaled by beta of the row
//   merged inverse (layered; raw off blocks overwritten with finals last):
//     L1: Ai10 = -(Ai11@Akk10)@Ai00   Ai21 = -(Ai22@Akk21)@Ai11
//         Ai32 = -(Ai33@Akk32)@Ai22                       (scratch 10/11/12)
//     L2: Ai20 = -Ai22@(Akk20@Ai00 + Akk21@Ai10)   Ai31 = -Ai33@(Akk31@Ai11 + Akk32@Ai21)
//     L3: Ai30 = -Ai33@(Akk30@Ai00 + Akk31@Ai10 + Akk32@Ai20)
//   w = Ai @ bf16(k*beta*2^g) -> Wn (A form);  u = Ai @ bf16(v*beta) -> Un (C)
//   QgA = bf16(q*2^g) A form;  Kn = bf16(k*2^(gn-g)) B form (gn = chunk-last g)
//
// Numerics: fp32 MMAC for A construction/merge, bf16 MMAC for w/u; the S3-12
// adjudication governs (fp32-truth <= FLA x1.10, pairwise <= 5e-3), bitwise
// NOT required.  Aqk/Akk upper triangles are written as EXACT ZEROS.  All
// global addressing goes through g2_rm_tok / g2_nat_blk (REVIEW_S3-12 rule 1).
#include "g2_common.h"

namespace g2 {

#ifndef G2_F32_MMAC32_DEFINED       // shared with g2_bwd_intra.cuh (same TU)
#define G2_F32_MMAC32_DEFINED
typedef float f2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ f4 mmac32(f2 a, f2 b, f4 c) {
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
#endif

struct FwdIParams {
    const short* qn; const short* kn; const short* v; const float* g; const float* beta;
    short* Wn; short* Kn; short* Un; short* QgA; short* AqkA;
    short* Aqk; short* Akk;
    float scale; int H, NT;
    const int* ctok; const int* clen;   // varlen phase 2 (fwd_intra2 only; null = dense)
    int stop;   // phase-timing: 1 after pair MMACs, 2 after solve, 3 after
                // merge+publish, 4 after QgA/Kn emits, 0 = full
};

constexpr int kPK2 = 132, kPA2 = 20;

__device__ __forceinline__ int s_off(int rb, int cb) { return 4 + rb * (rb - 1) / 2 + cb; }

__global__ void __launch_bounds__(256) g2_fwd_intra_kernel(FwdIParams p) {
    const int i = blockIdx.x, hh = blockIdx.y, tid = threadIdx.x;
    const int wv = tid >> 6, l = threadIdx.x & 63;
    const int m = l & 15, q4 = l >> 4;
    const size_t nblk = g2_nat_blk(i, hh, p.H, p.NT, false, 8192);
    const size_t nblk64 = g2_nat_blk(i, hh, p.H, p.NT, false, 4096);

    __shared__ __align__(16) short qk[2][64 * kPK2];    // q,k tiles (reused below)
    __shared__ __align__(16) float Ab[16][16 * kPA2];   // 0..3 diag Ai, 4..9 off raw
                                                        // -> final, 10..12 L1 results,
                                                        // 13..15 scratch.  L2's Ai31
                                                        // must NOT land in slot 8
                                                        // (raw Akk31) before L3 reads it
    for (int t = tid; t < 2048; t += 256) {
        const int r = t >> 5, cq = (t & 31) << 2;
        const size_t o = g2_rm_tok(i * kBT + r, hh, p.H, kK) + cq;
        *reinterpret_cast<v4bh*>(qk[0] + (size_t)r * kPK2 + cq) =
            *reinterpret_cast<const v4bh*>(p.qn + o);
        *reinterpret_cast<v4bh*>(qk[1] + (size_t)r * kPK2 + cq) =
            *reinterpret_cast<const v4bh*>(p.kn + o);
    }
    __syncthreads();

    auto gv = [&](int r, int c) { return p.g[g2_rm_tok(i * kBT + r, hh, p.H, kK) + c]; };
    auto qf = [&](int r, int c) { return bf2f(qk[0][(size_t)r * kPK2 + c]); };
    auto kf = [&](int r, int c) { return bf2f(qk[1][(size_t)r * kPK2 + c]); };
    auto bstore = [](short* dst, size_t off, float x) {
        dst[off] = (short)__bfloat16_as_ushort(__float2bfloat16(x));
    };

    auto pair_mmac = [&](int rb, int cb, int mode) {   // 0: rows q, 1: rows k
        const int rref = 16 * rb + (rb == cb ? 8 : 0);
        f4 acc = f4{0.f, 0.f, 0.f, 0.f};
        for (int kt = 0; kt < 16; ++kt) {           // 16 tiles of k=8 (16x16x8!)
            const int c0 = 8 * kt + 2 * q4;
            const float gr0 = gv(rref, c0), gr1 = gv(rref, c0 + 1);
            const int rr = 16 * rb + m, cc = 16 * cb + m;
            const f2 a2 = f2{(mode ? kf(rr, c0) : qf(rr, c0)) * exp2f(gv(rr, c0) - gr0),
                             (mode ? kf(rr, c0 + 1) : qf(rr, c0 + 1)) * exp2f(gv(rr, c0 + 1) - gr1)};
            const f2 b2 = f2{kf(cc, c0) * exp2f(gr0 - gv(cc, c0)),
                             kf(cc, c0 + 1) * exp2f(gr1 - gv(cc, c0 + 1))};
            acc = mmac32(a2, b2, acc);
        }
        return acc;
    };
    auto emit_Aqk = [&](int rb, int cb, const f4& acc, bool mask_diag) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = 16 * rb + m, j = 16 * cb + q4 + 4 * e;
            const float val = (!mask_diag || j <= t) ? acc[e] * p.scale : 0.f;
            bstore(p.Aqk, g2_rm_tok(i * kBT + t, hh, p.H, kBT) + j, val);
            bstore(p.AqkA, nat16(nblk64 + (size_t)(cb + 4 * rb) * 256 + 4 * m + q4 + 64 * e), val);
        }
    };
    auto emit_Akk = [&](int slot, int rb, int cb) {
        // C-fragment columns are q4 + 4e, NOT q4 + e
        const float x[4] = {Ab[slot][m * kPA2 + q4], Ab[slot][m * kPA2 + q4 + 4],
                            Ab[slot][m * kPA2 + q4 + 8], Ab[slot][m * kPA2 + q4 + 12]};
        #pragma unroll
        for (int e = 0; e < 4; ++e)
            bstore(p.Akk, g2_rm_tok(i * kBT + 16 * rb + m, hh, p.H, kBT) + 16 * cb + q4 + 4 * e,
                   x[e]);
    };

    // ---- phase 1: warp w owns row-block w: diag L + Aqk emits + raw off Akk
    {
        const float bm = p.beta[g2_rm_tok(i * kBT + 16 * wv + m, hh, p.H, 1)];
        f4 acc = pair_mmac(wv, wv, 1);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {          // init = -L (FLA's b_Ai = -b_Akk)
            const int j = q4 + 4 * e;
            Ab[wv][m * kPA2 + j] = (j < m) ? -acc[e] * bm : 0.f;
        }
        acc = pair_mmac(wv, wv, 0);
        emit_Aqk(wv, wv, acc, true);
        for (int cb = 0; cb < wv; ++cb) {
            acc = pair_mmac(wv, cb, 0);
            emit_Aqk(wv, cb, acc, false);
            acc = pair_mmac(wv, cb, 1);
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                Ab[s_off(wv, cb)][m * kPA2 + q4 + 4 * e] = acc[e] * bm;
        }
        // REVIEW_S3-13: the upper-triangle blocks are NEVER computed -- they must
        // still be WRITTEN as exact zeros (row-major Aqk/Akk AND the AqkA native),
        // or torch.empty garbage leaks (the 24 + 64*(3-r) per-band residue and the
        // "NaN" entries were uninitialized bf16 bit patterns, not computed NaNs).
        for (int cb = wv + 1; cb < 4; ++cb) {
            emit_Aqk(wv, cb, f4{0.f, 0.f, 0.f, 0.f}, false);
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                bstore(p.Akk, g2_rm_tok(i * kBT + 16 * wv + m, hh, p.H, kBT)
                       + 16 * cb + q4 + 4 * e, 0.f);
        }
    }
    __syncthreads();

    if (p.stop == 1) return;
    // ---- phase 2: forward substitution, all-lanes/registers version.  Lane m
    // of warp wv owns row m in registers; step j broadcasts row j elementwise
    // via __shfl_sync (register data -- no LDS ordering hazard) and every lane
    // with m > j does one vector FMA.  16 steps replace the old 14-iteration
    // single-lane serial recursion + CTA barrier (measured 2.45ms -> target
    // ~0.1ms).  Rows 0/1 keep the plain -L init per FLA.
    {
        float row[16], a[16];
        #pragma unroll
        for (int c = 0; c < 16; ++c) { row[c] = Ab[wv][m * kPA2 + c]; a[c] = row[c]; }
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            float bcast[16];
            #pragma unroll
            for (int c = 0; c < 16; ++c)
                bcast[c] = __shfl_sync(0xffffffffffffffffull, row[c], j, 16);
            const float aj = (m > j) ? a[j] : 0.f;     // RAW -L[m][j] (FLA reads
            if (m > j)                                  // the untouched row)
                #pragma unroll
                for (int c = 0; c < 16; ++c) row[c] += aj * bcast[c];
        }
        row[m] += 1.f;                                  // diagonal of I
        #pragma unroll
        for (int c = 0; c < 16; ++c) Ab[wv][m * kPA2 + c] = row[c];
    }
    emit_Akk(wv, wv, wv);
    __syncthreads();

    if (p.stop == 2) return;
    // ---- phase 3: merged inverse.  dot16(X, Y) reads operands from Ab slots;
    // chained products stage their intermediate in a scratch slot first.
    auto dot16 = [&](int X, int Y) {
        f4 acc = f4{0.f, 0.f, 0.f, 0.f};
        for (int h = 0; h < 2; ++h) {                  // k bases 8h + 2q4 (NOT 4h!)
            const f2 a2 = f2{Ab[X][m * kPA2 + 8 * h + 2 * q4],
                             Ab[X][m * kPA2 + 8 * h + 2 * q4 + 1]};
            const f2 b2 = f2{Ab[Y][(8 * h + 2 * q4) * kPA2 + m],
                             Ab[Y][(8 * h + 2 * q4 + 1) * kPA2 + m]};
            acc = mmac32(a2, b2, acc);
        }
        return acc;
    };
    auto neg16 = [](f4 c) { return f4{-c[0], -c[1], -c[2], -c[3]}; };
    auto add16 = [](f4 a, f4 b) { return f4{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]}; };
    auto store16 = [&](int slot, f4 c) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) Ab[slot][m * kPA2 + q4 + 4 * e] = c[e];
    };
    auto copy16 = [&](int dst, int src) {
        #pragma unroll
        for (int e = 0; e < 4; ++e)
            Ab[dst][m * kPA2 + q4 + 4 * e] = Ab[src][m * kPA2 + q4 + 4 * e];
    };
    // Merge with FULL __syncthreads between every step (REVIEW_S3-14: the
    // intra-warp __syncwarp staging raced -- plain __syncwarp carries no LDS
    // memory ordering, so the compiler hoisted LDS reads across it; outputs
    // were nondeterministic run-to-run.  Work is warp-guarded, barriers are
    // NOT -- every thread reaches every barrier.)
    #define MERGE_STEP(W, WORK) { if (wv == (W)) { WORK; } __syncthreads(); }
    MERGE_STEP(0, store16(13, dot16(1, s_off(1, 0))));   // Ai11 @ Akk10
    MERGE_STEP(0, store16(10, neg16(dot16(13, 0))));     // Ai10
    MERGE_STEP(1, store16(13, dot16(2, s_off(2, 1))));   // Ai22 @ Akk21
    MERGE_STEP(1, store16(11, neg16(dot16(13, 1))));     // Ai21
    MERGE_STEP(2, store16(13, dot16(3, s_off(3, 2))));   // Ai33 @ Akk32
    MERGE_STEP(2, store16(12, neg16(dot16(13, 2))));     // Ai32
    MERGE_STEP(1, store16(13, add16(dot16(s_off(2, 0), 0), dot16(s_off(2, 1), 10))));
    MERGE_STEP(1, store16(5, neg16(dot16(2, 13))));      // Ai20
    MERGE_STEP(2, store16(14, add16(dot16(s_off(3, 1), 1), dot16(s_off(3, 2), 11))));
    MERGE_STEP(2, store16(15, neg16(dot16(3, 14))));     // Ai31 -> scratch (slot 8
                                                         // still holds raw Akk31)
    MERGE_STEP(3, store16(13, add16(dot16(s_off(3, 0), 0), dot16(s_off(3, 1), 10))));
    MERGE_STEP(3, store16(14, dot16(s_off(3, 2), 5)));   // Akk32 @ Ai20
    if (wv == 3) {                                       // sum of all three terms
        const f4 s01 = f4{Ab[13][m * kPA2 + q4], Ab[13][m * kPA2 + q4 + 4],      // fragment
                          Ab[13][m * kPA2 + q4 + 8], Ab[13][m * kPA2 + q4 + 12]};  // cols q4+4e
        const f4 t2 = f4{Ab[14][m * kPA2 + q4], Ab[14][m * kPA2 + q4 + 4],
                         Ab[14][m * kPA2 + q4 + 8], Ab[14][m * kPA2 + q4 + 12]};
        store16(14, add16(s01, t2));
    }
    __syncthreads();
    MERGE_STEP(3, store16(7, neg16(dot16(3, 14))));      // Ai30
    #undef MERGE_STEP
    __syncthreads();
    copy16(8, 15);                                       // publish Ai31 (raw 8 consumed)
    copy16(4, 10);                                      // publish L1 results
    copy16(6, 11);
    copy16(9, 12);
    emit_Akk(4, 1, 0);
    emit_Akk(5, 2, 0);
    emit_Akk(6, 2, 1);
    emit_Akk(7, 3, 0);
    emit_Akk(8, 3, 1);
    emit_Akk(9, 3, 2);

    if (p.stop == 3) return;
    // ---- phase 4: QgA / Kn from the q/k tiles (g and gn from global)
    for (int t = tid; t < 2048; t += 256) {
        const int row = t >> 5, cq = (t & 31) << 2;
        const size_t go = g2_rm_tok(i * kBT + row, hh, p.H, kK) + cq;
        const f4 gf = *reinterpret_cast<const f4*>(p.g + go);
        v4bh qg, kg;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            qg[j] = bf_rne(qf(row, cq + j) * exp2f(gf[j]));
            kg[j] = bf_rne(kf(row, cq + j) * exp2f(gv(63, cq + j) - gf[j]));
        }
        {   // QgA: A form -- the source quad is an e-run, contiguous
            const int d0 = row >> 4, d1 = cq >> 4, q = (cq & 15) >> 2;
            *reinterpret_cast<v4bh*>(p.QgA + nat16(nblk + (size_t)(4 * (row & 15) + 64 * q
                + 256 * d1 + 2048 * d0))) = qg;
        }
        {   // Kn: B form -- the source quad is an m-run, stride-4 scalars
            const int d1 = row >> 4, q = (row >> 2) & 3, e = row & 3;
            const int d0 = cq >> 4, mm = cq & 15;
            const size_t ib = nblk + e + 4 * mm + 64 * q + 256 * d1 + 1024 * d0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) p.Kn[nat16(ib + 4 * j)] = kg[j];
        }
    }
    __syncthreads();

    // ---- phase 5/6: w = Ai @ kb, u = Ai @ vb (bf16 MMAC, kbT/vbT staged in
    // the freed qk[0] memory; A fragments cast on the fly from the Ab tiles)
    short* T2 = qk[0];
    auto wu_phase = [&](const short* x, bool use_g, short* outA, short* outC) {
        for (int t = tid; t < 2048; t += 256) {
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t o = g2_rm_tok(i * kBT + j, hh, p.H, kK) + vq;
            const v4bh xv = *reinterpret_cast<const v4bh*>(x + o);
            const float b = p.beta[g2_rm_tok(i * kBT + j, hh, p.H, 1)];
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                const float val = use_g
                    ? bf2f(xv[d]) * b * exp2f(p.g[o + d]) : bf2f(xv[d]) * b;
                T2[(size_t)(vq + d) * 68 + j] = bf_rne(val);
            }
        }
        __syncthreads();
        f4 acc[8];
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) acc[vt] = f4{0.f, 0.f, 0.f, 0.f};
        for (int js = 0; js < 4; ++js) {
            const int slot = (wv == js) ? wv : s_off(wv, js); (void)slot;
            // swap-test: read the A fragment from the GLOBAL row-major Akk exactly
            // like the verified standalone g2_wu (isolates tile-read vs staging)
            const v4bh af = *reinterpret_cast<const v4bh*>(
                p.Akk + g2_rm_tok(i * kBT + 16 * wv + (l & 15), hh, p.H, kBT)
                + 16 * js + (l >> 4) * 4);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {
                const size_t lo = (size_t)(16 * vt + m) * 68 + 16 * js + q4 * 4;
                acc[vt] = mmac(af, *reinterpret_cast<const v4bh*>(T2 + lo), acc[vt]);
            }
        }
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) {
            short we[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) we[e] = bf_rne(acc[vt][e]);
            if (outC)   // C form: the C fragment layout itself
                *reinterpret_cast<v4bh*>(outC + nat16(nblk + (size_t)(vt + 8 * wv) * 256 + l * 4)) =
                    *reinterpret_cast<const v4bh*>(we);
            if (outA) { // A form: stride-64 scalars
                const size_t ia = nblk + (size_t)(q4 + 4 * m + 256 * (vt + 8 * wv));
                #pragma unroll
                for (int e = 0; e < 4; ++e) outA[nat16(ia + 64 * e)] = we[e];
            }
        }
        __syncthreads();
    };
    if (p.stop == 4) return;
    wu_phase(p.kn, true, p.Wn, nullptr);
    wu_phase(p.v, false, nullptr, p.Un);
}

}  // namespace g2

extern "C" int g2_fwd_intra(const void* qn, const void* kn, const void* v, const void* g,
                            const void* beta, void* Wn, void* Kn, void* Un, void* QgA,
                            void* AqkA, void* Aqk, void* Akk, double scale,
                            int B, int T, int H, hipStream_t s) {
    if (B != 1 || T % 64) return -1;
    int stop = 0;
    if (const char* e = g2::g2_getenv("G2_FI_STOP")) stop = atoi(e);
    g2::FwdIParams p{(const short*)qn, (const short*)kn, (const short*)v, (const float*)g,
                     (const float*)beta, (short*)Wn, (short*)Kn, (short*)Un, (short*)QgA,
                     (short*)AqkA, (short*)Aqk, (short*)Akk, (float)scale, H, T / 64,
                     nullptr, nullptr, stop};
    hipLaunchKernelGGL(g2::g2_fwd_intra_kernel, dim3(T / 64, H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}
