// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// G2 engine P3 (chunk-parallel backward, FLA chunk_kda_bwd_wy_dqkg_fused role), gfx936.
//
// Verbatim extraction of p3_wy12_kernel:
//   deepseek/handoff_20260925_G2_integration/src/prototypes/p3_wy.hip lines 2650-2892
//   sha256(260f2315...) -- body unchanged; the two prototype-only indirections were
//   resolved into named parameters:
//     * WyParams::q carried the B-native kg and WyParams::v the B-native v, with the
//       row-major originals smuggled through the trailing qrow/vrow arguments.  They are
//       q_kgb / v_vb / q_row / v_row here.
//     * the redundant q/v aliases are gone.
//
// Block order is (NT, H) -- chunk-major, head-minor -- unlike P1/P2's (B, H, NT).
//
// Output-column permutation: a B-fragment lane with column index n reads source column
// pi(n) = 4*(n&3) + (n>>2), so the hardware C-fragment element e of lane l sits at
// n = q+4e (q = l>>4) and holds source column 4q+e: every lane owns 4 CONSECUTIVE
// columns, which makes the elementwise epilogue vectorised and lets a C-fragment serve
// directly as the A-fragment of the next GEMM.
#include "g2_common.h"

// P4 feasibility knobs (defaults = production codegen, unchanged):
//   G2_P3_WMIN : minimum waves/EU requested from the compiler (2 caps VGPRs at 128)
//   G2_S4_STOP : p3_wy_s4 ends after stage n (1 prologue, 2 pass Q, 3 pass K, 4 pass W,
//                5 dA stage) and dumps its live accumulators into p.dA -- per-stage
//                VGPR attribution only, never built into the .so
#ifndef G2_P3_WMIN
#define G2_P3_WMIN 1
#endif
#ifndef G2_S4_STOP
#define G2_S4_STOP 0
#endif
#ifndef G2_P4_EXP        // timing-only ablation bits (WRONG math when non-zero; bench only)
#define G2_P4_EXP 0
#endif
// P4' restructure stage of p3_wy_s4 (0 = S4 layout; 1 = pass Q merged into the tail loop;
// 2 = pass K merged too, h.dh from the in-register B-fragments; 3 = the dg last-row column
// sums become per-wave LDS partials summed in fixed order -- no atomics left, dg bitwise
// deterministic; 4 = P4'': the loop's h/dh B tiles staged once per CTA in LDS, tile kt+2
// prefetched during iteration kt, pass Q moved ahead of the table barrier)
#ifndef G2_P4_STEP
#define G2_P4_STEP 4
#endif

namespace g2 {

struct WyParams {
    // row-major A-side operands
    const bf16 *q_kgb, *k, *v_vb, *v_new, *A, *h, *do_, *dh, *dv;
    const float *g, *beta;
    // row-major outputs
    float *dq, *dk, *dg, *db, *dA;
    bf16* dv2;
    float scale;
    int T, H, NT;
    // fragment-native (B-native) aliases of q's and v's slots
    const bf16* q_row;              // row-major q (kg is the B-native tensor in q_kgb's slot)
    const bf16* v_row;              // row-major v (v_vb's slot holds the B-native v)
};

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(G2_P3_WMIN, 3))) p3_wy12_kernel(WyParams p) {
    __shared__ __align__(16) bf16 S[kD * kPT];
    __shared__ float dgk1[kD], dgk2[kD];
    const int tid = threadIdx.x, r = tid >> 6, l = tid & 63, m = l & 15, q = l >> 4, pm = pi16(m);
    const int i = blockIdx.x, hh = blockIdx.y, H = p.H;
    const int tr = 16 * r + m;
    const size_t rs = (size_t)H * kD, rsA = (size_t)H * kBT;
    const size_t cb = (size_t)i * kBT * rs + (size_t)hh * kD;          // chunk row 0, this head
    const size_t cbA = (size_t)i * kBT * rsA + (size_t)hh * kBT;
    const int R = H * kD;                                               // row stride (elements, fits int)
    const int RA = H * kBT;
    const bf16 *q_ = p.q_row + cb, *k_ = p.k + cb, *v_ = p.v_row + cb, *vn_ = p.v_new + cb, *do_ = p.do_ + cb, *dv_ = p.dv + cb;
    const float* g_ = p.g + cb;
    const bf16* A_ = p.A + cbA;
    const float* beta_ = p.beta + (size_t)i * kBT * H + hh;
    float *dq_ = p.dq + cb, *dk_ = p.dk + cb, *dg_ = p.dg + cb, *dA_ = p.dA + cbA;
    bf16* dv2_ = p.dv2 + cb;
    // fragment-native intermediates: VnA/DvA [BH-chunk][r4][vt8][lane][4] (A: m=t, d=v);
    // HnB/DhB [BH-chunk][kt8][vt8][lane][4] (B: n=k via pi, d=v) -> one 512 B block per wave load
    const size_t cA = ((size_t)i * H + hh) * 4 * 8 * 256, cB = ((size_t)i * H + hh) * 8 * 8 * 256;
    const bf16* hb = p.h + cB;
    const bf16* dhb = p.dh + cB;
    const bf16* vnN = p.v_new + cA;
    const bf16* dvN = p.dv + cA;
    const bf16* doN = p.do_ + cA;                                 // dO, A-native
    const bf16* vBN = p.v_vb + ((size_t)i * H + hh) * 4 * 8 * 256;   // v, B-native [st4][vt8] (n=s via pi, d=v)
    const bf16* kgB = p.q_kgb + ((size_t)i * H + hh) * 4 * 8 * 256;   // PROTOTYPE: kg=bf16(k*2^g), B-native [st4][kt8], in q's slot
    auto aN = [&](const bf16* base, int vt) { return ld8(base + ((r * 8 + vt) * 64 + l) * 4); };
    const int ro = tr * R + 4 * q;                                      // this lane's row, 4q offset
    auto ex2 = [](float x) { return __builtin_amdgcn_exp2f(x); };

    if (tid < kD) { dgk1[tid] = 0.f; dgk2[tid] = 0.f; }
    const float beta_t = beta_[tr * H];
    v4bh aa[4];                                                         // A^T row tr = column tr of A
    #pragma unroll
    for (int st = 0; st < 4; ++st)
        #pragma unroll
        for (int e = 0; e < 4; ++e) aa[st][e] = reinterpret_cast<const short*>(A_)[(16 * st + 4 * q + e) * RA + tr];
    v4bh dva[8];
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt) dva[vt] = aN(dvN, vt);
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) reinterpret_cast<short*>(S)[(16 * vt + 4 * q + e) * kPT + tr] = dva[vt][e];
    lds_barrier();                                                      // dvT
    float db = 0.f;
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt) {
        f4 acc = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st) acc = mmac(aa[st], ld8(S + (16 * vt + pm) * kPT + 16 * st + 4 * q), acc);
        const v4bh vv = ld8(v_ + ro + 16 * vt);
        v4bh o;
        #pragma unroll
        for (int e = 0; e < 4; ++e) { o[e] = bfr(acc[e] * beta_t); db += acc[e] * bf2f(vv[e]); }
        *reinterpret_cast<v4bh*>(dv2_ + ro + 16 * vt) = o;
    }
    // dA1 = dv . v^T
    f4 dA[4];
    #pragma unroll
    for (int st = 0; st < 4; ++st) dA[st] = f4{0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt)
        #pragma unroll
        for (int st = 0; st < 4; ++st)
            dA[st] = mmac(dva[vt], ld8(vBN + ((st * 8 + vt) * 64 + l) * 4), dA[st]);
    lds_barrier();                                                      // dvT dead -> dwT

    // three passes: acc[kt] = sum_vt A[vt] . B(kt, vt), B rows 16kt+pm of a [K][V] state block
    v4bh af[8], bf0[8], bf1[8];
    // pass Q: dq final, dg <- q*dq
    {
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) af[vt] = aN(doN, vt);
        const bf16* st_ = (hb);
        auto loadB = [&](int kt, v4bh* bf) {
            const bf16* pb = st_ + (kt * 8 * 64 + l) * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) bf[vt] = ld8(pb + vt * 256);
        };
        auto body = [&](int kt, v4bh* bf) {
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(af[vt], bf[vt], a4[vt & 3]);
            const f4 acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
                     const int kc = 16 * kt;         const f4 gv = ld16f(g_ + ro + kc);         const v4bh qq = ld8(q_ + ro + kc);         f4 dqf, dg;         
#pragma unroll
         for (int e = 0; e < 4; ++e) { dqf[e] = acc[e] * ex2(gv[e]) * p.scale; dg[e] = bf2f(qq[e]) * dqf[e]; }         *reinterpret_cast<f4*>(dq_ + ro + kc) = dqf;         *reinterpret_cast<f4*>(dg_ + ro + kc) = dg;     
        };
        loadB(0, bf0);
        #pragma unroll 1
        for (int kt = 0; kt < 8; kt += 2) {
            loadB(kt + 1, bf1);
            body(kt, bf0);
            loadB(kt < 6 ? kt + 2 : kt + 1, bf0);
            body(kt + 1, bf1);
        }
    }
    // pass K: dk_pre, dg -= k*dk_pre, column sums; dgk1 partial over v-tiles {2r, 2r+1}
    {
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) af[vt] = aN(vnN, vt);
        const bf16* st_ = (dhb);
        auto loadB = [&](int kt, v4bh* bf) {
            const bf16* pb = st_ + (kt * 8 * 64 + l) * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) bf[vt] = ld8(pb + vt * 256);
        };
        auto body = [&](int kt, v4bh* bf) {
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(af[vt], bf[vt], a4[vt & 3]);
            const f4 acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
                     const int kc = 16 * kt;         const f4 gv = ld16f(g_ + ro + kc), gn = ld16f(g_ + (kBT - 1) * R + 4 * q + kc);         const v4bh kk = ld8(k_ + ro + kc);         f4 dkp, dg = ld16f(dg_ + ro + kc);         float kd[4];         
#pragma unroll
         for (int e = 0; e < 4; ++e) {             dkp[e] = acc[e] * ex2(gn[e] - gv[e]);             kd[e] = bf2f(kk[e]) * dkp[e];             dg[e] -= kd[e];         }         *reinterpret_cast<f4*>(dk_ + ro + kc) = dkp;         *reinterpret_cast<f4*>(dg_ + ro + kc) = dg;         
#pragma unroll
         for (int e = 0; e < 4; ++e) { const float c = msum(kd[e]); if (m == 0) atomicAdd(&dgk2[kc + 4 * q + e], c); }         float hd = 0.f;         
#pragma unroll
         for (int j = 0; j < 2; ++j) {             const int off = ((kt * 8 + 2 * r + j) * 64 + l) * 4;             const v4bh h0 = ld8(hb + off), d0 = ld8(dhb + off);             
#pragma unroll
             for (int e = 0; e < 4; ++e) hd += bf2f(h0[e]) * bf2f(d0[e]);         }         hd = xsum16(hd);         if (q == 0) atomicAdd(&dgk1[16 * kt + pm], hd);     
        };
        loadB(0, bf0);
        #pragma unroll 1
        for (int kt = 0; kt < 8; kt += 2) {
            loadB(kt + 1, bf1);
            body(kt, bf0);
            loadB(kt < 6 ? kt + 2 : kt + 1, bf0);
            body(kt + 1, bf1);
        }
    }
    // pass W: -dw -> dwT, dA += (-dw) kg^T
    {
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) af[vt] = aN(dvN, vt);
        const bf16* st_ = (hb);
        auto loadB = [&](int kt, v4bh* bf) {
            const bf16* pb = st_ + (kt * 8 * 64 + l) * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) bf[vt] = ld8(pb + vt * 256);
        };
        auto body = [&](int kt, v4bh* bf) {
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(af[vt], bf[vt], a4[vt & 3]);
            const f4 acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
                     const int kc = 16 * kt;         v4bh dwn;         
#pragma unroll
         for (int e = 0; e < 4; ++e) {             dwn[e] = bfr(-acc[e]);             reinterpret_cast<short*>(S)[(kc + 4 * q + e) * kPT + tr] = dwn[e];         }         
#pragma unroll
         for (int st = 0; st < 4; ++st) {             const v4bh kgb = ld8(kgB + ((st * 8 + kt) * 64 + l) * 4); dA[st] = mmac(dwn, kgb, dA[st]);         }     
        };
        loadB(0, bf0);
        #pragma unroll 1
        for (int kt = 0; kt < 8; kt += 2) {
            loadB(kt + 1, bf1);
            body(kt, bf0);
            loadB(kt < 6 ? kt + 2 : kt + 1, bf0);
            body(kt + 1, bf1);
        }
    }
    lds_barrier();                                                      // dwT, dgk1, dgk2 complete
    // dkgb = A^T (-dw): dk, dg, db epilogue
    #pragma unroll 2
    for (int kt = 0; kt < 8; ++kt) {
        f4 acc = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st) acc = mmac(aa[st], ld8(S + (16 * kt + pm) * kPT + 16 * st + 4 * q), acc);
        const int kc = 16 * kt;
        const f4 gv = ld16f(g_ + ro + kc), gn = ld16f(g_ + (kBT - 1) * R + 4 * q + kc);
        const v4bh kk = ld8(k_ + ro + kc);
        f4 dk = ld16f(dk_ + ro + kc), dg = ld16f(dg_ + ro + kc);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float eg = ex2(gv[e]);
            const float kg = bf2f(kk[e]) * eg;
            dk[e] += acc[e] * (eg * beta_t);
            dg[e] += kg * acc[e] * beta_t;
            db += acc[e] * kg;
            if (tr == kBT - 1) dg[e] += dgk1[kc + 4 * q + e] * ex2(gn[e]) + dgk2[kc + 4 * q + e];
        }
        *reinterpret_cast<f4*>(dk_ + ro + kc) = dk;
        *reinterpret_cast<f4*>(dg_ + ro + kc) = dg;
    }
    db = xsum16(db);
    if (q == 0) p.db[(size_t)(i * kBT + tr) * H + hh] = db;
    // dA = mask(-A^T ((mask(dA) * beta_s) A^T_stored...))
    v4bh da[4];
    #pragma unroll
    for (int st = 0; st < 4; ++st)
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int s = 16 * st + 4 * q + e;
            da[st][e] = bfr(tr > s ? dA[st][e] * beta_[s * H] : 0.f);
        }
    f4 xc[4];
    #pragma unroll
    for (int ut = 0; ut < 4; ++ut) {
        xc[ut] = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st) xc[ut] = mmac(da[st], ld8(A_ + (16 * ut + pm) * RA + 16 * st + 4 * q), xc[ut]);
    }
    lds_barrier();                                                      // dwT reads done -> XT
    #pragma unroll
    for (int ut = 0; ut < 4; ++ut)
        #pragma unroll
        for (int e = 0; e < 4; ++e) reinterpret_cast<short*>(S)[(16 * ut + 4 * q + e) * kPT + tr] = bfr(xc[ut][e]);
    lds_barrier();
    #pragma unroll
    for (int ut = 0; ut < 4; ++ut) {
        f4 acc = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st) acc = mmac(aa[st], ld8(S + (16 * ut + pm) * kPT + 16 * st + 4 * q), acc);
        f4 o;
        #pragma unroll
        for (int e = 0; e < 4; ++e) { const int u = 16 * ut + 4 * q + e; o[e] = tr > u ? -acc[e] : 0.f; }
        *reinterpret_cast<f4*>(dA_ + tr * RA + 16 * ut + 4 * q) = o;
    }
}

extern "C" int g2_wy(const void* q_kgb, const void* k, const void* v_vb, const void* v_new, const void* g,
                     const void* beta, const void* A, const void* h, const void* do_, const void* dh,
                     const void* dv, void* dq, void* dk, void* dv2, void* dg, void* db, void* dA,
                     float scale, int T, int H, int NT, const void* q_row, const void* v_row,
                     hipStream_t s) {
    if (T % kBT || NT != T / kBT) return -1;
    WyParams p{(const bf16*)q_kgb, (const bf16*)k, (const bf16*)v_vb, (const bf16*)v_new,
               (const bf16*)A, (const bf16*)h, (const bf16*)do_, (const bf16*)dh, (const bf16*)dv,
               (const float*)g, (const float*)beta,
               (float*)dq, (float*)dk, (float*)dg, (float*)db, (float*)dA, (bf16*)dv2,
               scale, T, H, NT, (const bf16*)q_row, (const bf16*)v_row};
    hipLaunchKernelGGL(p3_wy12_kernel, dim3(NT, H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}

}  // namespace g2

// ============================================================================
// S4-b: P3 + chunk-intra backward fused (replaces p3_wy12 + FLA chunk_kda_bwd_intra).
//
// Same CTA (chunk, head) and wave rows as p3_wy12; the intra body is g2_intra_pi's.
// Order changes vs p3_wy12:
//   * pass Q writes dq_P3 only (no dg); pass K writes dk_K = dk_pre and dg_K = -k*dk_pre
//     (no read-modify-write of dg);
//   * the dA stage runs BEFORE the dk/dg epilogue (XT gets its own LDS region because
//     S still holds dwT); dAkk never leaves the CTA: own-row part = the C-pi registers
//     (exactly the intra A-fragments), transposed part via LDS;
//   * the p3 epilogue (A^T(-dw)) and the intra terms share ONE kt loop that reads
//     dq_P3/dk_K/dg_K back (same lane wrote them) and writes the final dq/dk/dg once;
//     db is written once (P3 part + intra part).
#include "g2_intra_pi.cuh"

namespace g2 {

struct WyS4Params {
    const bf16 *q_kgb, *k, *v_vb, *v_new, *A, *h, *do_, *dh, *dv;
    const float *g, *beta, *dAqk;
    float *dq, *dk, *dg, *db, *dA;                 // dA: optional debug copy of dAkk
    bf16* dv2;
    float scale;
    int T, H, NT;
    const bf16 *q_row, *v_row;
};

struct S4Smem {
    bf16 S[kD * kPT];                              // dvT -> dwT (live through the tail)
    float dgk1[kD], dgk2[kD];
    float cs1[2][4][16], cs2[2][4][16];            // P4' step 3: per-wave column partials
    // P4'' (ST >= 4): the kt loop's h / dh B-fragment tiles, staged once per CTA (they do
    // not depend on the wave's row block) -- [buffer = kt & 1][h, dh][2048], same element
    // order as global (vt*256 + lane*4 + e), so a wave's ds_read_b64 per lane is contiguous
    bf16 stg[2][2][8 * 256];
    union {
        struct { bf16 XT[kBT * kPT]; float dAT[kBT * ipi::TS]; } a;   // dA stage
        ipi::Smem t;                                                // intra tables
    } u;
};

template <int A, int ST>
__device__ __forceinline__ void s4_tail(const WyS4Params& p, S4Smem& sm, const v4bh (&aa)[4],
                                        const f4 (&dak)[4], float db, float beta_t) {
    using namespace ipi;
    const int tid = threadIdx.x, l = tid & 63, m = l & 15, q = l >> 4, pm = pi16(m);
    const int i = blockIdx.x, hh = blockIdx.y, H = p.H;
    const int R = H * kD, RA = H * kBT;
    const long tok0 = (long)i * kBT;
    const size_t cb = g2_rm_tok(tok0, hh, H, kD), cbA = g2_rm_tok(tok0, hh, H, kBT);
    const int row = 16 * A + m;
    const int ro = row * R + 4 * q;
    const bf16 *q_ = p.q_row + cb, *k_ = p.k + cb;
    const float* g_ = p.g + cb;
    float *dq_ = p.dq + cb, *dk_ = p.dk + cb, *dg_ = p.dg + cb;
    const float* aq_ = p.dAqk + cbA;

    // A fragments (see g2_intra_pi.cuh): dAkk own row = the dA-stage registers (already
    // zero for j >= i), transposed rows from the LDS copy; dAqk from global, FLA masks
    f4 aqA[A + 1], aqB[4 - A], akB[4 - A];
    #pragma unroll
    for (int b = 0; b <= A; ++b) {
        aqA[b] = ld16f(aq_ + row * RA + 16 * b + 4 * q);
        if (b == A) {
            #pragma unroll
            for (int x = 0; x < 4; ++x)
                if (4 * q + x > m) aqA[b][x] = 0.f;
        }
    }
    #pragma unroll
    for (int b = A; b < 4; ++b) {
        akB[b - A] = *reinterpret_cast<const f4*>(&sm.u.a.dAT[row * TS + 16 * b + 4 * q]);
        #pragma unroll
        for (int x = 0; x < 4; ++x) {
            const int j = 16 * b + 4 * q + x;
            const bool keep = b > A || 4 * q + x >= m;
            aqB[b - A][x] = keep ? aq_[j * RA + row] : 0.f;
            if (!keep) akB[b - A][x] = 0.f;
        }
    }
    // P4' step 1 (ST >= 1): pass Q lives in this loop -- dO A-fragments stay resident, the
    // h B-fragments of tile kt are loaded per iteration; dq_P3 never goes to memory.
    // P4' step 2 (ST >= 2): pass K too -- v_new A-fragments resident, dh B-fragments per tile.
    // (Arrays of an inactive step are never read, so they cost no registers.)
    const size_t cQ = (size_t)i * H + hh;
    const bf16* hb = p.h + cQ * 8 * 8 * 256;
    const bf16* dhb = p.dh + cQ * 8 * 8 * 256;
    v4bh ado[8], avn[8];
    if constexpr (ST >= 1) {
        const bf16* doN = p.do_ + cQ * 4 * 8 * 256;
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) ado[vt] = ld8(doN + ((A * 8 + vt) * 64 + l) * 4);
    }
    if constexpr (ST >= 2) {
        const bf16* vnN = p.v_new + cQ * 4 * 8 * 256;
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) avn[vt] = ld8(vnN + ((A * 8 + vt) * 64 + l) * 4);
    }
    // P4'' (ST >= 4): tile kt of h / dh is 2048 contiguous bf16 at hb/dhb + kt*2048; the
    // CTA copies it into sm.stg with one 16-byte load per thread per tensor.  Tiles 0 and
    // 1 go in before the barrier below; tile kt+2 is loaded at the top of iteration kt and
    // stored into buffer kt&1 AFTER iteration kt's table barrier.  Every read of the
    // staging buffers happens BEFORE that barrier (pass Q is moved ahead of it), so:
    //   RAW: tile kt+2 is written before barrier(kt+1) and read after it (iteration kt+2);
    //   WAR: the writer of buffer kt&1 has passed barrier(kt), i.e. every wave has finished
    //        its pre-barrier reads of tile kt.
    // No barrier is added; the loop keeps exactly one CTA barrier per iteration.
    const int4* hg4 = reinterpret_cast<const int4*>(hb);
    const int4* dhg4 = reinterpret_cast<const int4*>(dhb);
    int4* stg4 = reinterpret_cast<int4*>(&sm.stg[0][0][0]);
    if constexpr (ST >= 4) {
        #pragma unroll
        for (int t0 = 0; t0 < 2; ++t0) {
            stg4[(t0 * 2 + 0) * 256 + tid] = hg4[t0 * 256 + tid];
            stg4[(t0 * 2 + 1) * 256 + tid] = dhg4[t0 * 256 + tid];
        }
    }
    lds_barrier();                                 // dAT reads done -> tables alias it

    float dbacc = 0.f;
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {
        const int buf = kt & 1, kc = 16 * kt;
        v4bh hB[8], dhB[8];
        int4 pfh, pfd;                             // ST >= 4: tile kt+2 in flight
        const bf16* sh = &sm.stg[buf][0][0];
        const bf16* sdh = &sm.stg[buf][1][0];
        if constexpr (ST >= 4) {
            if (kt + 2 < 8) {
                pfh = hg4[(kt + 2) * 256 + tid];
                pfd = dhg4[(kt + 2) * 256 + tid];
            }
        }
        // B-fragment accessors: registers (ST 1-3) or the staged tile (ST >= 4)
        auto HB = [&](int vt) -> v4bh {
            if constexpr (ST >= 4) return ld8(sh + vt * 256 + l * 4);
            else return hB[vt];
        };
        auto DHB = [&](int vt) -> v4bh {
            if constexpr (ST >= 4) return ld8(sdh + vt * 256 + l * 4);
            else return dhB[vt];
        };
        if constexpr (ST >= 1 && ST < 4) {
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) hB[vt] = ld8(hb + (kt * 8 * 64 + l) * 4 + vt * 256);
        }
        if constexpr (ST >= 2 && ST < 4) {
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) dhB[vt] = ld8(dhb + (kt * 8 * 64 + l) * 4 + vt * 256);
        }
#if G2_P4_EXP & 256     // timing ablation (ST >= 4 bench only): no g/k/q row loads
        const f4 gv = f4{-0.01f * kt, -0.02f, -0.03f, -0.04f};
        const v4bh kk = {(short)(0x3c00 + kt), 0x3c00, 0x3c00, 0x3c00}, qq = kk;
#else
        const f4 gv = ld16f(g_ + ro + kc);
        const v4bh kk = ld8(k_ + ro + kc), qq = ld8(q_ + ro + kc);
#endif
        // gn: loaded here only when pass K needs it; steps 0/1 load it late (short live
        // range -- loading it here pushed the step-0 reference to 256 VGPR + 7 spills)
        f4 gn;
#if G2_P4_EXP & 256
        if constexpr (ST >= 2) gn = gv * 2.f;
#else
        if constexpr (ST >= 2) gn = ld16f(g_ + (kBT - 1) * R + 4 * q + kc);
#endif
        // pass K tile kt (same expressions as the former pass).  Its column sums must be
        // complete before row 63 consumes them below: they are issued before the table
        // barrier of this iteration.
        f4 dkp, dgK;
        if constexpr (ST >= 2) {
        {
            #pragma clang fp contract(off)
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
#if !(G2_P4_EXP & 32)    // timing ablation: no pass-K MMACs
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(avn[vt], DHB(vt), a4[vt & 3]);
#endif
            const f4 acck = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            float kd[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                dkp[e] = acck[e] * ex2(gn[e] - gv[e]);
                kd[e] = bf2f(kk[e]) * dkp[e];
                dgK[e] = -kd[e];
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float c = msum(kd[e]);
                if (m == 0) {
                    if constexpr (ST >= 3) sm.cs2[buf][A][4 * q + e] = c;   // fixed-order sum below
                    else atomicAdd(&sm.dgk2[kc + 4 * q + e], c);
                }
            }
        }
        {
            // h.dh over this wave's v-tiles {2A, 2A+1}: both B-fragments are already here
            // (default contraction, as in the former pass K)
            float hd = 0.f;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const v4bh h0 = HB(2 * A + j), d0 = DHB(2 * A + j);
                #pragma unroll
                for (int e = 0; e < 4; ++e) hd += bf2f(h0[e]) * bf2f(d0[e]);
            }
            hd = xsum16(hd);
            if (q == 0) {
                if constexpr (ST >= 3) sm.cs1[buf][A][pm] = hd;
                else atomicAdd(&sm.dgk1[16 * kt + pm], hd);
            }
        }
        }
        // ST >= 4: pass Q runs here, BEFORE the table barrier, so that every read of the
        // staged tile precedes the barrier (see the WAR argument above).  Same expression
        // and contraction as the late pass Q of ST 1-3.
        f4 dqiE;
        if constexpr (ST >= 4) {
            #pragma clang fp contract(off)
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
#if !(G2_P4_EXP & 64)    // timing ablation: no pass-Q MMACs
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(ado[vt], HB(vt), a4[vt & 3]);
#endif
            const f4 accq = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            #pragma unroll
            for (int e = 0; e < 4; ++e) dqiE[e] = accq[e] * ex2(gv[e]) * p.scale;
        }
        f4 ep, en;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float d = gv[e] - __shfl(gv[e], 16 * q + 8, 64);
            ep[e] = ex2(d);
            en[e] = ex2(-d);
        }
        constexpr bool kIntra = !(G2_P4_EXP & 4);   // bit 2: timing ablation, no intra terms
        if (kIntra) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float kf = bf2f(kk[e]), qf = bf2f(qq[e]);
            const int o = (4 * e + q) * TS + row;
            sm.u.t.TA[buf][o] = kf * en[e];
            sm.u.t.TQ[buf][o] = qf * ep[e];
            sm.u.t.TK[buf][o] = kf * beta_t * ep[e];
        }
        if (m == 8) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) sm.u.t.gM[buf][A][4 * e + q] = gv[e];
        }
        lds_barrier();
        }
        if constexpr (ST >= 4) {                   // tile kt+2 -> the buffer tile kt just freed
            if (kt + 2 < 8) {
                stg4[(buf * 2 + 0) * 256 + tid] = pfh;
                stg4[(buf * 2 + 1) * 256 + tid] = pfd;
            }
        }

        // p3 epilogue product: dkgb = A^T (-dw), tile kt (dwT in S)
        f4 acc = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st)
            acc = mmac(aa[st], ld8(sm.S + (kc + pm) * kPT + 16 * st + 4 * q), acc);

        f4 cq0 = {0.f, 0.f, 0.f, 0.f}, cq1 = cq0, ck0 = cq0, ck1 = cq0, ct0 = cq0, ct1 = cq0;
        if (kIntra) {
        const float gm_n = sm.u.t.gM[buf][A][m];
        #pragma unroll
        for (int b = 0; b <= A; ++b) {
            f4 t = *reinterpret_cast<const f4*>(&sm.u.t.TA[buf][m * TS + 16 * b + 4 * q]);
            if (b < A) t *= ex2_diff(gm_n, sm.u.t.gM[buf][b][m]);
            cq0 = mm8(lo2(aqA[b]), lo2(t), cq0);
            ck0 = mm8(lo2(dak[b]), lo2(t), ck0);
            cq1 = mm8(hi2(aqA[b]), hi2(t), cq1);
            ck1 = mm8(hi2(dak[b]), hi2(t), ck1);
        }
        #pragma unroll
        for (int b = A; b < 4; ++b) {
            f4 tq = *reinterpret_cast<const f4*>(&sm.u.t.TQ[buf][m * TS + 16 * b + 4 * q]);
            f4 tk = *reinterpret_cast<const f4*>(&sm.u.t.TK[buf][m * TS + 16 * b + 4 * q]);
            if (b > A) {
                const float s = ex2_diff(sm.u.t.gM[buf][b][m], gm_n);
                tq *= s;
                tk *= s;
            }
            ct0 = mm8(lo2(aqB[b - A]), lo2(tq), ct0);
            ct1 = mm8(lo2(akB[b - A]), lo2(tk), ct1);
            ct0 = mm8(hi2(aqB[b - A]), hi2(tq), ct0);
            ct1 = mm8(hi2(akB[b - A]), hi2(tk), ct1);
        }
        }

        if constexpr (ST < 2) gn = ld16f(g_ + (kBT - 1) * R + 4 * q + kc);
#if G2_P4_EXP & 8       // timing ablation: no read-back of the pass-Q/K partials
        const f4 dqi = gv * 0.5f, dki = gv * 0.25f, dgi = gv * 0.125f;
#else
        f4 dqi, dki, dgi;
        if constexpr (ST >= 4) {
            dqi = dqiE;
        } else if constexpr (ST >= 1) {            // pass Q tile kt, same expression as before
            // dq_P3 used to round-trip through memory, which kept (acc*2^g)*scale a separate
            // rounding; without contract(off) it fuses into an FMA with "+ dq2" below
            // (measured: dq checksum moved) -- keep the validated rounding bit for bit
            #pragma clang fp contract(off)
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(ado[vt], hB[vt], a4[vt & 3]);
            const f4 accq = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            #pragma unroll
            for (int e = 0; e < 4; ++e) dqi[e] = accq[e] * ex2(gv[e]) * p.scale;
        } else {
            dqi = ld16f(dq_ + ro + kc);
        }
        if constexpr (ST >= 2) {                   // pass K tile, never stored
            dki = dkp;
            dgi = dgK;
        } else {
            dki = ld16f(dk_ + ro + kc);
            dgi = ld16f(dg_ + ro + kc);
        }
#endif
        f4 oq, ok, og, dbp;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            // No FMA formation in the final combination: which products the compiler fused
            // depended on the surrounding code (P4' step 2 moved dk/dg of wave 3 by up to
            // 77 fp32 ulp, rel 1.8e-8, with identical source expressions).  Plain mul+add
            // makes the result a function of the source alone.
            #pragma clang fp contract(off)
            const float kf = bf2f(kk[e]), qf = bf2f(qq[e]);
            // p3 part (p3_wy12 epilogue), then the intra increments
            const float eg = ex2(gv[e]);
            const float kg = kf * eg;
            float dk = dki[e] + acc[e] * (eg * beta_t);
            float dg = dgi[e] + kg * acc[e] * beta_t;
            db += acc[e] * kg;
            if (row == kBT - 1) {
                if constexpr (ST >= 3) {           // column sums over the 4 waves, fixed order
                    const int c = 4 * q + e;
                    const float s1 = ((sm.cs1[buf][0][c] + sm.cs1[buf][1][c]) + sm.cs1[buf][2][c]) + sm.cs1[buf][3][c];
                    const float s2 = ((sm.cs2[buf][0][c] + sm.cs2[buf][1][c]) + sm.cs2[buf][2][c]) + sm.cs2[buf][3][c];
                    dg += s1 * ex2(gn[e]) + s2;
                } else {
                    dg += sm.dgk1[kc + 4 * q + e] * ex2(gn[e]) + sm.dgk2[kc + 4 * q + e];
                }
            }
            const float dq2 = (cq0[e] + cq1[e]) * ep[e];
            const float dk2 = (ck0[e] + ck1[e]) * ep[e];
            const float dkt = (ct0[e] + ct1[e]) * en[e];
            dbp[e] = dk2 * kf;
            const float dk2b = dk2 * beta_t;
            const float dqf = dqi[e] + dq2;
            oq[e] = dqf;
            ok[e] = dk + dk2b + dkt;
            og[e] = dg + qf * dqf + (dk2b - dkt) * kf;
        }
        dbacc += (dbp[0] + dbp[1]) + (dbp[2] + dbp[3]);
        // bit 128 (timing ablation): the stores stay in the code but are skipped at run time
        if (!(G2_P4_EXP & 128) || p.scale < 0.f) {
            *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;
            *reinterpret_cast<f4*>(dg_ + ro + kc) = og;
        }
    }
    db = xsum16(db) + xsum16(dbacc);
    if (q == 0) p.db[(size_t)(tok0 + row) * H + hh] = db;
}

// Three instantiations of the same body (g2_wy_s4_kernel.inc; see the note there on why
// it is textually included instead of being a function template):
//   p3_wy_s4_kernel      production, ST = G2_P4_STEP (default 4 = P4'': staged B tiles)
//   p3_wy_s4_p4p_kernel  P4' (ST 3), kept for same-process A/B (KDA_G2_P4=1)
//   p3_wy_s4_ref_kernel  S4 layout (ST 0), fallback / reference (KDA_G2_P4=0)
// With the tail's contract(off), ST 3 and ST 4 agree bit for bit on every output; ST 0
// agrees on dq/dk/dv2/db and on dg outside chunk row 63.
#define G2_S4_KNAME p3_wy_s4_kernel
#define G2_S4_KST G2_P4_STEP
#define G2_S4_KWMIN G2_P3_WMIN
#include "g2_wy_s4_kernel.inc"
#undef G2_S4_KNAME
#undef G2_S4_KST
#undef G2_S4_KWMIN
#define G2_S4_KNAME p3_wy_s4_p4p_kernel
#define G2_S4_KST 3
#define G2_S4_KWMIN 1
#include "g2_wy_s4_kernel.inc"
#undef G2_S4_KNAME
#undef G2_S4_KST
#undef G2_S4_KWMIN
#define G2_S4_KNAME p3_wy_s4_ref_kernel
#define G2_S4_KST 0
#define G2_S4_KWMIN 1
#include "g2_wy_s4_kernel.inc"
#undef G2_S4_KNAME
#undef G2_S4_KST
#undef G2_S4_KWMIN

// variant: 0 = S4 reference, 1 = P4', 2 = production (P4'')
static int wy_s4_launch(int variant, const void* q_kgb, const void* k, const void* v_vb,
                        const void* v_new, const void* g, const void* beta, const void* A,
                        const void* h, const void* do_, const void* dh, const void* dv,
                        const void* dAqk, void* dq, void* dk, void* dv2, void* dg, void* db,
                        void* dA, float scale, int T, int H, int NT, const void* q_row,
                        const void* v_row, hipStream_t s) {
    if (T % kBT || NT != T / kBT) return -1;
    WyS4Params p{(const bf16*)q_kgb, (const bf16*)k, (const bf16*)v_vb, (const bf16*)v_new,
                 (const bf16*)A, (const bf16*)h, (const bf16*)do_, (const bf16*)dh, (const bf16*)dv,
                 (const float*)g, (const float*)beta, (const float*)dAqk,
                 (float*)dq, (float*)dk, (float*)dg, (float*)db, (float*)dA, (bf16*)dv2,
                 scale, T, H, NT, (const bf16*)q_row, (const bf16*)v_row};
    if (variant == 0) hipLaunchKernelGGL(p3_wy_s4_ref_kernel, dim3(NT, H), dim3(256), 0, s, p);
    else if (variant == 1) hipLaunchKernelGGL(p3_wy_s4_p4p_kernel, dim3(NT, H), dim3(256), 0, s, p);
    else hipLaunchKernelGGL(p3_wy_s4_kernel, dim3(NT, H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}

extern "C" int g2_wy_s4(const void* q_kgb, const void* k, const void* v_vb, const void* v_new,
                        const void* g, const void* beta, const void* A, const void* h,
                        const void* do_, const void* dh, const void* dv, const void* dAqk,
                        void* dq, void* dk, void* dv2, void* dg, void* db, void* dA,
                        float scale, int T, int H, int NT, const void* q_row, const void* v_row,
                        hipStream_t s) {
    return wy_s4_launch(2, q_kgb, k, v_vb, v_new, g, beta, A, h, do_, dh, dv, dAqk, dq, dk,
                        dv2, dg, db, dA, scale, T, H, NT, q_row, v_row, s);
}

extern "C" int g2_wy_s4_ref(const void* q_kgb, const void* k, const void* v_vb, const void* v_new,
                            const void* g, const void* beta, const void* A, const void* h,
                            const void* do_, const void* dh, const void* dv, const void* dAqk,
                            void* dq, void* dk, void* dv2, void* dg, void* db, void* dA,
                            float scale, int T, int H, int NT, const void* q_row,
                            const void* v_row, hipStream_t s) {
    return wy_s4_launch(0, q_kgb, k, v_vb, v_new, g, beta, A, h, do_, dh, dv, dAqk, dq, dk,
                        dv2, dg, db, dA, scale, T, H, NT, q_row, v_row, s);
}

extern "C" int g2_wy_s4_variant(int variant, const void* q_kgb, const void* k, const void* v_vb,
                                const void* v_new, const void* g, const void* beta, const void* A,
                                const void* h, const void* do_, const void* dh, const void* dv,
                                const void* dAqk, void* dq, void* dk, void* dv2, void* dg,
                                void* db, void* dA, float scale, int T, int H, int NT,
                                const void* q_row, const void* v_row, hipStream_t s) {
    return wy_s4_launch(variant, q_kgb, k, v_vb, v_new, g, beta, A, h, do_, dh, dv, dAqk, dq, dk,
                        dv2, dg, db, dA, scale, T, H, NT, q_row, v_row, s);
}

// ================================================================ W1: s4 split (P4''')
// KDA_G2_P4=3: the fused p3_wy_s4 split into two kernels so each fits 32 KB LDS and the
// scheduler co-residents TWO CTAs per CU (the fused kernel is LDS-bound at 62.5 KB, 1
// CTA/CU).  Segment A = prologue + pass W + dA stage (LDS 26,112 B); segment B = the kt
// loop (s4_tail structure: staged h/dh tiles prefetched one iteration ahead, single-
// buffered intra tables + a second CTA barrier for the table WAR; LDS 30,720 B).
// Handoff through global: dwT (bf16 [NT*H][128][64]), dak/dAkk (f32, the former debug
// output, now mandatory), db lane partials (f32 [NT*H][256]) -- all lossless round
// trips, every arithmetic expression copied verbatim, so all five outputs are bit for
// bit equal to p3_wy_s4 (ST=4); verified by checksum (w1_bench) and torch-level gates.
// NOTE (fmaf): the prologue's db accumulation MUST stay the fused kernel's FMA form --
// with a plain += the compiler chose mul+add here and db drifted +-1-4 ulp on 54% of
// entries (measured); __builtin_fmaf pins it.
// kra P-22: C-form fragment (lane l: row l&15, cols (l>>4) + 4e) -> A-form fragment (row l&15,
// cols 4(l>>4) + e): a 4x4 transpose across the 4 lanes sharing the row, in two 32-bit exchanges
// (swap q bit 1 <-> e bit 1 with lane l^32, then q bit 0 <-> e bit 0 with lane l^16).
__device__ __forceinline__ v4bh g2_c2a(v4bh c, int l) {
    uint32_t d0 = (uint32_t)(unsigned short)c[0] | ((uint32_t)(unsigned short)c[1] << 16);
    uint32_t d1 = (uint32_t)(unsigned short)c[2] | ((uint32_t)(unsigned short)c[3] << 16);
    const bool b1 = (l >> 5) & 1, b0 = (l >> 4) & 1;
    uint32_t r = (uint32_t)__shfl_xor((int)(b1 ? d0 : d1), 32, 64);
    if (b1) d0 = r; else d1 = r;
    const uint32_t snd = b0 ? ((d0 & 0xFFFFu) | (d1 << 16)) : ((d0 >> 16) | (d1 & 0xFFFF0000u));
    r = (uint32_t)__shfl_xor((int)snd, 16, 64);
    if (b0) { d0 = (r & 0xFFFFu) | (d0 & 0xFFFF0000u); d1 = (r >> 16) | (d1 & 0xFFFF0000u); }
    else    { d0 = (d0 & 0xFFFFu) | (r << 16);          d1 = (d1 & 0xFFFFu) | (r & 0xFFFF0000u); }
    return v4bh{(short)(d0 & 0xFFFFu), (short)(d0 >> 16), (short)(d1 & 0xFFFFu), (short)(d1 >> 16)};
}

// kra P-38: g2_c2a with selects instead of divergent branches (identical bit operations)
__device__ __forceinline__ v4bh g2_c2a_sel(v4bh c, int l) {
    uint32_t d0 = (uint32_t)(unsigned short)c[0] | ((uint32_t)(unsigned short)c[1] << 16);
    uint32_t d1 = (uint32_t)(unsigned short)c[2] | ((uint32_t)(unsigned short)c[3] << 16);
    const bool b1 = (l >> 5) & 1, b0 = (l >> 4) & 1;
    uint32_t r = (uint32_t)__shfl_xor((int)(b1 ? d0 : d1), 32, 64);
    d0 = b1 ? r : d0;
    d1 = b1 ? d1 : r;
    const uint32_t snd = b0 ? ((d0 & 0xFFFFu) | (d1 << 16)) : ((d0 >> 16) | (d1 & 0xFFFF0000u));
    r = (uint32_t)__shfl_xor((int)snd, 16, 64);
    const uint32_t e0 = b0 ? ((r & 0xFFFFu) | (d0 & 0xFFFF0000u)) : ((d0 & 0xFFFFu) | (r << 16));
    const uint32_t e1 = b0 ? ((r >> 16) | (d1 & 0xFFFF0000u)) : ((d1 & 0xFFFFu) | (r & 0xFFFF0000u));
    return v4bh{(short)(e0 & 0xFFFFu), (short)(e0 >> 16), (short)(e1 & 0xFFFFu), (short)(e1 >> 16)};
}

struct WyS4AParams {
    const bf16 *kgb, *v_vb, *A, *h, *dv;
    const float* beta;
    const bf16* v_row;
    bf16 *dv2, *dwT;
    float *dA, *db_part;
    int T, H, NT;
    const int* ctok; const int* clen;   // varlen phase 2 (null = dense)
    const bf16* dv2c;                   // kra P-22: C-form dv2 (std order), read when dv is null
    int hf = 0;                         // kra P-96: head-fastest grid (blockIdx.x = head), set by the launcher
};

struct WyS4BParams {
    const bf16 *q_row, *k, *A, *h, *do_, *dh, *v_new;
    const float *g, *beta, *dAqk;
    const bf16* dwT;
    const float *dA, *db_part;
    float *dq, *dk, *dg, *db;
    float scale;
    int T, H, NT;
    const int* ctok; const int* clen;   // varlen phase 2 (null = dense)
    const bf16* do_row;                 // kra P-20: row-major do, read when do_ (aN_do) is null
    const bf16* vnc;                    // kra P-22: C-form v_new (std order), read when v_new is null
    int hf = 0;                         // kra P-96: head-fastest grid (blockIdx.x = head), set by the launcher
    short* dq_bf = nullptr;             // kra P-938q: non-null -> dq / dk go out as the final bf16 (gfx938, no l2norm)
    short* dk_bf = nullptr;
    int vna = 0;                        // kra P-938va: vnc already holds A-form fragments (dav re-emitted them)
};

#if defined(__gfx938__)
#define G2_S4A_LEAN 1   // kra P-938a
#else
#define G2_S4A_LEAN 0
#endif
// VL (varlen phase 2): the row tensors (v_row, A, beta, dv2 here; q_row, k, g, dAqk, dq,
// dk, dg, db in B) are PACKED -- rows r >= clen read as 0 (g: clamped row) and are never
// written.  The dA (dak) handoff between A and B is internal and stays chunk-padded.
template <bool VL, bool AFS = false, int PF = 0>   // kra P-28c: AFS = stage the dA-stage A block in XT; P-83: PF
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))
p3_wy_s4a_kernel(WyS4AParams p_) {
    // kra P-35: field-wise copy, pointers stamped global (global_* ops, not flat)
    const WyS4AParams p{g2_glb(p_.kgb), g2_glb(p_.v_vb), g2_glb(p_.A), g2_glb(p_.h), g2_glb(p_.dv), g2_glb(p_.beta), g2_glb(p_.v_row), g2_glb(p_.dv2), g2_glb(p_.dwT), g2_glb(p_.dA), g2_glb(p_.db_part), p_.T, p_.H, p_.NT, g2_glb(p_.ctok), g2_glb(p_.clen), g2_glb(p_.dv2c), p_.hf};
    __shared__ __align__(16) bf16 S[kD * kPT];      // dvT -> dwT (same buffer reuse as the fused kernel)
#if G2_S4A_LEAN
    // kra P-938a (gfx938): dwT goes straight to HBM from pass W (no S staging), so S is free from pass W on and also
    // hosts the h double buffer + kgB tiles (pass W) and the dA-stage XC / A staging: LDS 30464 -> 17664 B
    bf16* const XT = S;                                 // XC / AFS A staging; h double buffer in pass W
    bf16* const KGB = S + 4096;                         // kgB tiles of steps kt (2 x 1024 shorts), pass W only
#define G2_S4A_KG(slot) (KGB + (slot) * 1024)
#else
    __shared__ __align__(16) bf16 XT[kBT * kPT];    // dA-stage XC staging (dAT staging dropped)
    __shared__ __align__(16) bf16 KGs[2][4 * 64 * 4];   // kra P-28: kgB tile of step kt (double buffer)
#define G2_S4A_KG(slot) (&KGs[slot][0])                 // the original expression (gfx936 codegen unchanged)
#endif
    __shared__ float BETAs[kBT];                        // kra P-28: masked beta of the chunk's rows
    const int tid = threadIdx.x, r = tid >> 6, l = tid & 63, m = l & 15, q = l >> 4, pm = pi16(m);
    // kra P-88: head-fastest grid when launched as dim3(H, NT).  kra P-96: the launcher says so explicitly --
    // inferring it from gridDim fell back to chunk-fastest whenever H == NT (e.g. T 4096 / H 64).
    const bool hf_ = p.hf != 0;
    const int i = hf_ ? (int)blockIdx.y : (int)blockIdx.x, hh = hf_ ? (int)blockIdx.x : (int)blockIdx.y, H = p.H;
    const int tr = 16 * r + m;
    const long tok0 = g2_row_tok<VL>(p.ctok, i);
        const int cl = g2_row_len<VL>(p.clen, i);
    // varlen: ONE uniform branch per CTA -- full chunks run the unmasked body (M=1)
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;
        const bool trok = g2_row_ok<M>(tr, cl);
        const size_t cb = g2_rm_tok(tok0, hh, H, kD), cbA = g2_rm_tok(tok0, hh, H, kBT);
        const size_t cbAp = g2_rm_tok((long)i * kBT, hh, H, kBT);   // dA handoff (chunk-padded)
        const int R = H * kD, RA = H * kBT;
        const bf16* v_ = p.v_row + cb;
        const bf16* A_ = p.A + cbA;
        const float* beta_ = p.beta + (size_t)tok0 * H + hh;
        bf16* dv2_ = p.dv2 + cb;
        const size_t cA = g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
        const size_t cB = g2_nat_blk(i, hh, H, 0, true, 8 * 8 * 256);
        const bf16* hb = p.h + cB;
        const bf16* dvN = p.dv + cA;
        const bf16* vBN = p.v_vb + cA;
        const bf16* kgB = p.kgb + cA;
        auto aN = [&](const bf16* base, int vt) { return ld8(base + ((r * 8 + vt) * 64 + l) * 4); };
        const int ro = tr * R + 4 * q;

        const float beta_t = trok ? beta_[tr * H] : 0.f;
        // kra P-83 PF: the beta loads first, then every front-phase / pass-W operand in consumption order
        v4bh pav[4], pdva[8], pvrow[8], pkg[8];
        f4 phb[8];
        if constexpr (PF) {
            const float bts_ = (tid < kBT && g2_row_ok<M>(tid, cl)) ? beta_[tid * H] : 0.f;
            #pragma unroll
            for (int it_ = 0; it_ < 4; ++it_) {
                const int j = tid + 256 * it_, row = j >> 4, qc = j & 15;
                pav[it_] = g2_row_ok<M>(row, cl)
                    ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0};
            }
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) pdva[vt] = aN(dvN, vt);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) pvrow[vt] = trok ? ld8(v_ + ro + 16 * vt) : v4bh{0, 0, 0, 0};
            const bf16* hsrc0 = hb + 8 * tid;
            const bf16* kgsrc0 = kgB + (((tid >> 6) * 8) * 64 + (tid & 63)) * 4;
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                phb[kt] = *reinterpret_cast<const f4*>(hsrc0 + kt * 2048);
                pkg[kt] = ld8(kgsrc0 + kt * 256);
            }
            if (tid < kBT) BETAs[tid] = bts_;
        } else {
        if (tid < kBT) BETAs[tid] = g2_row_ok<M>(tid, cl) ? beta_[tid * H] : 0.f;   // kra P-28
        }
        v4bh aa[4];                                                         // A^T row tr = column tr of A
        {   // Y4: stage the 64x64 A block into LDS (transiently, S is free before dvT)
            // with coalesced 8B row loads, then gather the transposed column from LDS --
            // replaces 16 stride-RA 2-byte gathers per lane (X2: 20.3% of s4a stalls).
            short* As = reinterpret_cast<short*>(S);
            v4bh av_[4];                                   // kra P-43: all four loads in flight
            if constexpr (PF) {                            // kra P-83: issued up front
                #pragma unroll
                for (int it_ = 0; it_ < 4; ++it_) av_[it_] = pav[it_];
            } else {
            #pragma unroll
            for (int it_ = 0; it_ < 4; ++it_) {
                const int j = tid + 256 * it_, row = j >> 4, qc = j & 15;
                // VL: rows past the chunk's end are NOT in the packed buffer -- never load them
                // (their aa entries are masked below anyway).
                av_[it_] = g2_row_ok<M>(row, cl)
                    ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0};
            }
            }
            #pragma unroll
            for (int it_ = 0; it_ < 4; ++it_) {
                const int j = tid + 256 * it_, row = j >> 4, qc = j & 15;
                *reinterpret_cast<v4bh*>(As + row * 64 + 4 * qc) = av_[it_];
            }
            lds_barrier();                                                  // A block visible
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    aa[st][e] = g2_row_ok<M>(16 * st + 4 * q + e, cl)
                        ? As[(16 * st + 4 * q + e) * 64 + tr] : (short)0;
            lds_barrier();                                                  // aa extracted (S -> dvT)
        }
        v4bh dva[8];
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) dva[vt] = PF ? pdva[vt] : aN(dvN, vt);   // kra P-83
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) reinterpret_cast<short*>(S)[(16 * vt + 4 * q + e) * kPT + tr] = dva[vt][e];
        lds_barrier();                                                      // dvT
        float db = 0.f;
        v4bh vrow[8];                // kra P-17: this lane's row-major v quads, kept for the dA MMAC
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) {
            f4 acc = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int st = 0; st < 4; ++st) acc = mmac(aa[st], ld8(S + (16 * vt + pm) * kPT + 16 * st + 4 * q), acc);
            const v4bh vv = PF ? pvrow[vt] : (trok ? ld8(v_ + ro + 16 * vt) : v4bh{0, 0, 0, 0});   // kra P-83
            vrow[vt] = vv;
            v4bh o;
            #pragma unroll
            for (int e = 0; e < 4; ++e) { o[e] = bfr(acc[e] * beta_t); db = __builtin_fmaf(acc[e], bf2f(vv[e]), db); }
            if (trok) *reinterpret_cast<v4bh*>(dv2_ + ro + 16 * vt) = o;
        }
        f4 dA[4];
        #pragma unroll
        for (int st = 0; st < 4; ++st) dA[st] = f4{0.f, 0.f, 0.f, 0.f};
        if (p.v_vb != nullptr) {
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                #pragma unroll
                for (int st = 0; st < 4; ++st)
                    dA[st] = mmac(dva[vt], ld8(vBN + ((st * 8 + vt) * 64 + l) * 4), dA[st]);
        } else {
            // kra P-17: no vB copy -- the B fragment of (st, vt) is v[16*st + pi16(m)][16*vt+4q..+3],
            // i.e. the row-major quads the CTA just loaded; stage them in S ([64][136] = all of S,
            // free now that every wave finished reading dvT) and read the fragments from LDS.
            constexpr int kVS = 2 * kPT;                            // 136-short rows
            lds_barrier();                                          // dvT reads done
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                *reinterpret_cast<v4bh*>(reinterpret_cast<short*>(S) + tr * kVS + 16 * vt + 4 * q) = vrow[vt];
            lds_barrier();                                          // v tile visible
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                #pragma unroll
                for (int st = 0; st < 4; ++st)
                    dA[st] = mmac(dva[vt], ld8(S + (16 * st + pm) * kVS + 16 * vt + 4 * q), dA[st]);
        }
        lds_barrier();                                                      // dvT/v tile dead -> dwT

        const v4bh (&af)[8] = dva;   // V4 (kra P-3): dvN fragments are already in dva -- no reload
#if G2_S4A_LEAN
        bf16* dwTg = p.dwT + g2_nat_blk(i, hh, H, 0, true, kD * kBT);
#endif
        {   // pass W: -dw -> dwT (S), dA += (-dw) kg^T   [verbatim from the .inc]
            auto body = [&](int kt, const v4bh* bf) {
                f4 a4[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(af[vt], bf[vt], a4[vt & 3]);
                const f4 acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
                const int kc = 16 * kt;
                v4bh dwn;
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    dwn[e] = bfr(-acc[e]);
#if G2_S4A_LEAN
                    reinterpret_cast<short*>(dwTg)[(kc + 4 * q + e) * kBT + tr] = dwn[e];   // [col][tr], pitch kBT
#else
                    reinterpret_cast<short*>(S)[(kc + 4 * q + e) * kPT + tr] = dwn[e];
#endif
                }
                #pragma unroll
                for (int st = 0; st < 4; ++st) {
                    const v4bh kgb = ld8(G2_S4A_KG(kt & 1) + (st * 64 + l) * 4);   // kra P-28
                    dA[st] = mmac(dwn, kgb, dA[st]);
                }
            };
            if constexpr (PF) {   // kra P-83: blocks / tiles already in registers; pass W fully unrolled
                short* hs = reinterpret_cast<short*>(XT);
                *reinterpret_cast<v4bh*>(G2_S4A_KG(0) + 4 * tid) = pkg[0];
                *reinterpret_cast<f4*>(hs + 8 * tid) = phb[0];
                lds_barrier();                                                  // block 0 visible
                #pragma unroll
                for (int kt = 0; kt < 8; ++kt) {
                    v4bh bf[8];
                    const bf16* hk = reinterpret_cast<const bf16*>(hs + (kt & 1) * 2048) + l * 4;
                    #pragma unroll
                    for (int vt = 0; vt < 8; ++vt) bf[vt] = ld8(hk + vt * 256);
                    if (kt < 7) {
                        *reinterpret_cast<f4*>(hs + ((kt + 1) & 1) * 2048 + 8 * tid) = phb[kt + 1];
                        *reinterpret_cast<v4bh*>(G2_S4A_KG((kt + 1) & 1) + 4 * tid) = pkg[kt + 1];
                    }
                    body(kt, bf);
                    lds_barrier();      // block kt+1 visible; reads of block kt done before reuse
                }
            } else {
            // kra P-28: thread tid carries fragment (st = tid>>6, lane = tid&63) of the kgB tile
            const bf16* kgsrc = kgB + (((tid >> 6) * 8) * 64 + (tid & 63)) * 4;
            *reinterpret_cast<v4bh*>(G2_S4A_KG(0) + 4 * tid) = ld8(kgsrc);            // tile 0
            v4bh kgreg = ld8(kgsrc + 1 * 256);                                      // tile 1
            // V5 (kra P-4): the h block of step kt (8 vt fragments x 64 lanes x 8 B = 4 KB) does
            // not depend on the wave, so the 4 waves used to fetch it 4 times with 8 B loads.
            // Now the CTA fetches it ONCE with one 16 B load per thread into XT (free until the
            // dA stage; 2 x 4 KB double buffer) and every wave reads its fragments from LDS in
            // the global layout (lane-contiguous 8 B: bank-conflict free).  Values unchanged.
            short* hs = reinterpret_cast<short*>(XT);
            const bf16* hsrc = hb + 8 * tid;
            f4 hreg = *reinterpret_cast<const f4*>(hsrc);                       // block 0
            *reinterpret_cast<f4*>(hs + 8 * tid) = hreg;
            hreg = *reinterpret_cast<const f4*>(hsrc + 2048);                   // block 1
            lds_barrier();                                                      // block 0 visible
            #pragma unroll 1
            for (int kt = 0; kt < 8; ++kt) {
                v4bh bf[8];
                const bf16* hk = reinterpret_cast<const bf16*>(hs + (kt & 1) * 2048) + l * 4;
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) bf[vt] = ld8(hk + vt * 256);
                if (kt < 7) {
                    // buffer (kt+1)&1 was last read in step kt-1, before the barrier ending kt-1
                    *reinterpret_cast<f4*>(hs + ((kt + 1) & 1) * 2048 + 8 * tid) = hreg;
                    *reinterpret_cast<v4bh*>(G2_S4A_KG((kt + 1) & 1) + 4 * tid) = kgreg;   // kra P-28
                    if (kt < 6) {
                        hreg = *reinterpret_cast<const f4*>(hsrc + (kt + 2) * 2048);
                        kgreg = ld8(kgsrc + (kt + 2) * 256);
                    }
                }
                body(kt, bf);
                lds_barrier();      // block kt+1 visible; reads of block kt done before reuse
            }
            }   // kra P-83: !PF
        }
        // dA stage: da = mask(dA*beta), XC = da A^T -> XT (LDS), dak = mask(-A^T XT) -> global
        // dA.  The dAT LDS staging is dropped: segment B reads the transposed elements from
        // the global dA write directly.
        v4bh da[4];
        #pragma unroll
        for (int st = 0; st < 4; ++st)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int s = 16 * st + 4 * q + e;
                da[st][e] = bfr(tr > s ? dA[st][e] * BETAs[s] : 0.f);   // kra P-28: staged, same mask
            }
        f4 xc[4];
        if constexpr (AFS) {   // kra P-28b/c: A block -> XT once per CTA (XT is free after pass W)
            short* As2 = reinterpret_cast<short*>(XT);
            #pragma unroll   // kra P-34: all loads in flight before the first wait
            for (int it_ = 0; it_ < 4; ++it_) {
                const int j = tid + 256 * it_;
                const int row = j >> 4, qc = j & 15;
                *reinterpret_cast<v4bh*>(As2 + row * kPT + 4 * qc) = (PF == 2) ? pav[it_]   // kra P-83b
                    : (g2_row_ok<M>(row, cl) ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0});
            }
            lds_barrier();                                                  // A block visible
            #pragma unroll
            for (int ut = 0; ut < 4; ++ut) {
                xc[ut] = f4{0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int st = 0; st < 4; ++st)
                    xc[ut] = mmac(da[st], ld8(XT + (16 * ut + pm) * kPT + 16 * st + 4 * q), xc[ut]);
            }
            lds_barrier();                                                  // A reads done -> XC
        } else {
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut) {
            xc[ut] = f4{0.f, 0.f, 0.f, 0.f};
            const bool aok = g2_row_ok<M>(16 * ut + pm, cl);
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                xc[ut] = mmac(da[st], aok ? ld8(A_ + (16 * ut + pm) * RA + 16 * st + 4 * q)
                                          : v4bh{0, 0, 0, 0}, xc[ut]);
        }
        }
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut)
            #pragma unroll
            for (int e = 0; e < 4; ++e) reinterpret_cast<short*>(XT)[(16 * ut + 4 * q + e) * kPT + tr] = bfr(xc[ut][e]);
        lds_barrier();                                                      // XT (and dwT in S) complete
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut) {
            f4 acc = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int st = 0; st < 4; ++st) acc = mmac(aa[st], ld8(XT + (16 * ut + pm) * kPT + 16 * st + 4 * q), acc);
            f4 dak;
            #pragma unroll
            for (int e = 0; e < 4; ++e) { const int u = 16 * ut + 4 * q + e; dak[e] = tr > u ? -acc[e] : 0.f; }
            *reinterpret_cast<f4*>(p.dA + cbAp + tr * RA + 16 * ut + 4 * q) = dak;
        }
        // dwT: bulk copy S -> global [col][tr], pitch kBT.  The XT barrier above orders every
        // lane's pass-W S writes before these cross-lane reads.
#if !G2_S4A_LEAN
        bf16* dwTg = p.dwT + g2_nat_blk(i, hh, H, 0, true, kD * kBT);
        for (int j = tid; j < kD * (kBT / 4); j += 256) {
            const int col = j >> 4, tq = j & 15;
            *reinterpret_cast<v4bh*>(dwTg + col * kBT + 4 * tq) = ld8(S + col * kPT + 4 * tq);
        }
#endif
        p.db_part[g2_nat_blk(i, hh, H, 0, true, 256) + tid] = db;
    });
}

#undef G2_S4A_KG
// segment B: s4_tail<ST=4> verbatim except dwT read from global (pitch kBT), dak/akB
// from the global dA, db from db_part, single-buffered intra tables (the WAR closed by
// a second end-of-iteration barrier instead of the second table buffer).
struct S4BSmem {
    float TA[16 * ipi::TS], TQ[16 * ipi::TS], TK[16 * ipi::TS];
    float gM[4][16];
    float cs1[2][4][16], cs2[2][4][16];
    bf16 stg[2][2][8 * 256];
    bf16 dwTs[16 * 64];          // kra P-29: dwT tile of step kt, column groups XOR-swizzled by row
};

// kra P-938tz (gfx938): s4b's intra tables indexed as 64-float rows with the 4-float group XOR-ed by
// h(r) = 4 (r & 1) ^ ((r >> 1) & 1) ^ 2 ((r >> 2) & 1) (fits inside the padded arrays, so LDS and the P-938t prologue
// footprint are unchanged).  Bank model (§141): the b32 table stores and the b128 operand reads become conflict-free
// (hipprof on 079f58b: 3136 conflict cycles per CTA, ~2700 of them in the 68-float tables).  Layout only.
__device__ __forceinline__ int s4b_tix(int r, int c) {
#if defined(__gfx938__)
    const int h = ((r & 1) << 2) ^ ((r >> 1) & 1) ^ (((r >> 2) & 1) << 1);
    return r * 64 + ((((c >> 2) ^ h) & 15) << 2) + (c & 3);
#else
    return r * ipi::TS + c;
#endif
}
template <int A, int M>
__device__ __forceinline__ void s4b_tail(const WyS4BParams& p, S4BSmem& sm, const v4bh (&aa)[4],
                                         float db, float beta_t, long tok0, int cl) {
    using namespace ipi;
    const int l = threadIdx.x & 63, m = l & 15, q = l >> 4, pm = pi16(m);
    // kra P-88: head-fastest grid when launched as dim3(H, NT).  kra P-96: the launcher says so explicitly --
    // inferring it from gridDim fell back to chunk-fastest whenever H == NT (e.g. T 4096 / H 64).
    const bool hf_ = p.hf != 0;
    const int i = hf_ ? (int)blockIdx.y : (int)blockIdx.x, hh = hf_ ? (int)blockIdx.x : (int)blockIdx.y, H = p.H;
    const int R = H * kD, RA = H * kBT;
    const size_t cb = g2_rm_tok(tok0, hh, H, kD), cbA = g2_rm_tok(tok0, hh, H, kBT);
    const size_t cbAp = g2_rm_tok((long)i * kBT, hh, H, kBT);   // dA handoff (chunk-padded)
    const int row = 16 * A + m;
    const bool rok = g2_row_ok<M>(row, cl);
    const int ro = row * R + 4 * q;
    const int rog = g2_row_g<M>(row, cl) * R + 4 * q;          // g: clamped row
    const int rgn = g2_row_g<M>(kBT - 1, cl) * R + 4 * q;      // chunk-end g row (clen-1)
    const int rlast = M == 2 ? cl - 1 : kBT - 1;                    // the chunk-end dg term's row
    const bf16 *q_ = p.q_row + cb, *k_ = p.k + cb;
    const float* g_ = p.g + cb;
    float *dq_ = p.dq + cb, *dk_ = p.dk + cb, *dg_ = p.dg + cb;
    const float* aq_ = p.dAqk + cbA;
    const float* dAg = p.dA + cbAp;
    const bf16* dwTg = p.dwT + g2_nat_blk(i, hh, H, 0, true, kD * kBT);

    f4 dak[4];
    f4 aqA[A + 1], aqB[4 - A], akB[4 - A];
#if G2_MLS938
    // MLS reads whole 64 B row segments: only for 256 B-aligned tile bases (CTA-uniform); otherwise per-lane loads
    if (M != 2 && ((reinterpret_cast<uintptr_t>(dAg) | reinterpret_cast<uintptr_t>(aq_)) & 255) == 0) {
        // kra P-938t: full chunk.  dA and dAqk (64 x 64 fp32, row pitch RA) -> LDS by MLS once per CTA (the whole
        // S4BSmem is free here), then the rows (dak, aqA) and the column gathers (akB, aqB) are read from LDS.
        // Tile t (wave A loads t = 2A, 2A + 1): rows 32 (t >> 2).., floats 16 (t & 3)..; LDS: dAqk at 0, dA at 16 KB.
        char* base = reinterpret_cast<char*>(&sm);
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int t = 2 * A + u;
            g2_mls_tile(dAg + (size_t)(32 * (t >> 2)) * RA + 16 * (t & 3), 2 * RA, base + 16384 + t * 2048);
            g2_mls_tile(aq_ + (size_t)(32 * (t >> 2)) * RA + 16 * (t & 3), 2 * RA, base + t * 2048);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        lds_barrier();
        auto at = [&](int off, int r, int c) {
            return reinterpret_cast<const float*>(base + off + (((r >> 5) * 4 + (c >> 4)) * 32 + (r & 31)) * 64 + (c & 15) * 4);
        };
        #pragma unroll
        for (int ut = 0; ut < 4; ++ut) dak[ut] = *reinterpret_cast<const f4*>(at(16384, row, 16 * ut + 4 * q));
        #pragma unroll
        for (int b = 0; b <= A; ++b) {
            aqA[b] = *reinterpret_cast<const f4*>(at(0, row, 16 * b + 4 * q));
            if (b == A) {
                #pragma unroll
                for (int x = 0; x < 4; ++x)
                    if (4 * q + x > m) aqA[b][x] = 0.f;
            }
        }
        #pragma unroll
        for (int b = A; b < 4; ++b)
            #pragma unroll
            for (int x = 0; x < 4; ++x) {
                const int j = 16 * b + 4 * q + x;
                const bool keep = b > A || 4 * q + x >= m;
                akB[b - A][x] = keep ? *at(16384, j, row) : 0.f;
                aqB[b - A][x] = keep ? *at(0, j, row) : 0.f;
            }
        lds_barrier();
    } else
#endif
    {
    #pragma unroll
    for (int ut = 0; ut < 4; ++ut) dak[ut] = ld16f(dAg + row * RA + 16 * ut + 4 * q);
    #pragma unroll
    for (int b = 0; b <= A; ++b) {
        aqA[b] = rok ? ld16f(aq_ + row * RA + 16 * b + 4 * q) : f4{0.f, 0.f, 0.f, 0.f};
        if (b == A) {
            #pragma unroll
            for (int x = 0; x < 4; ++x)
                if (4 * q + x > m) aqA[b][x] = 0.f;
        }
    }
    #pragma unroll
    for (int b = A; b < 4; ++b)
        #pragma unroll
        for (int x = 0; x < 4; ++x) {
            const int j = 16 * b + 4 * q + x;
            const bool keep = b > A || 4 * q + x >= m;
            akB[b - A][x] = keep ? dAg[j * RA + row] : 0.f;   // dAT[row][j] == dA[j][row]
            aqB[b - A][x] = (keep && g2_row_ok<M>(j, cl)) ? aq_[j * RA + row] : 0.f;
        }
    }
    const size_t cQ = g2_nat_blk(i, hh, H, 0, true, 8 * 8 * 256);
    const bf16* hb = p.h + cQ;
    const bf16* dhb = p.dh + cQ;
    v4bh ado[8], avn[8];
    {
        if (p.do_ != nullptr) {
            const bf16* doN = p.do_ + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) ado[vt] = ld8(doN + ((A * 8 + vt) * 64 + l) * 4);
        } else {
            // kra P-20: the aN_do fragment (A*8 + vt, l) = do[row][16vt + 4q..+3], row = 16A + m
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                ado[vt] = rok ? ld8(p.do_row + cb + ro + 16 * vt) : v4bh{0, 0, 0, 0};
        }
        if (p.v_new != nullptr) {
            const bf16* vnN = p.v_new + g2_nat_blk(i, hh, H, 0, true, 4 * 8 * 256);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) avn[vt] = ld8(vnN + ((A * 8 + vt) * 64 + l) * 4);
        } else {                // kra P-22: from the C-form Vn via the in-register transpose
            const bf16* vnb = p.vnc + g2_nat_blk(i, hh, H, p.NT, false, 8192) + A * 8 * 256;
            if (p.vna) {        // kra P-938va: A-form fragments from dav, L16 pairs (vt, vt + 1) per 16 B load
                #pragma unroll
                for (int j = 0; j < 4; ++j) ld16x2(vnb + j * 512 + l * 8, avn[2 * j], avn[2 * j + 1]);
            } else {
                const bf16* vnc = vnb + l * 4;
                v4bh raw_[8];                               // kra P-38: all 8 loads in flight first
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) raw_[vt] = ld8(vnc + vt * 256);
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) avn[vt] = g2_c2a_sel(raw_[vt], l);
            }
        }
    }
    const int4* hg4 = reinterpret_cast<const int4*>(hb);
    const int4* dhg4 = reinterpret_cast<const int4*>(dhb);
    int4* stg4 = reinterpret_cast<int4*>(&sm.stg[0][0][0]);
    #pragma unroll
    for (int t0 = 0; t0 < 2; ++t0) {
        stg4[(t0 * 2 + 0) * 256 + threadIdx.x] = hg4[t0 * 256 + threadIdx.x];
        stg4[(t0 * 2 + 1) * 256 + threadIdx.x] = dhg4[t0 * 256 + threadIdx.x];
    }
    lds_barrier();

    float dbacc = 0.f;
    #pragma unroll 1
    for (int kt = 0; kt < 8; ++kt) {
        const int buf = kt & 1, kc = 16 * kt;
        // kra P-29: this kt's dwT tile, one 8 B load per thread (issued first: oldest in vmcnt)
        const int dwr = threadIdx.x >> 4, dwc = threadIdx.x & 15;
        const v4bh dwv = ld8(dwTg + (kc + dwr) * kBT + 4 * dwc);
        int4 pfh, pfd;
        const bf16* sh = &sm.stg[buf][0][0];
        const bf16* sdh = &sm.stg[buf][1][0];
        auto HB = [&](int vt) { return ld8(sh + vt * 256 + l * 4); };
        auto DHB = [&](int vt) { return ld8(sdh + vt * 256 + l * 4); };
        if (kt + 2 < 8) {
            pfh = hg4[(kt + 2) * 256 + threadIdx.x];
            pfd = dhg4[(kt + 2) * 256 + threadIdx.x];
        }
        const f4 gv = ld16f(g_ + rog + kc);
        const v4bh kk = rok ? ld8(k_ + ro + kc) : v4bh{0, 0, 0, 0};
        const v4bh qq = rok ? ld8(q_ + ro + kc) : v4bh{0, 0, 0, 0};
        const f4 gn = ld16f(g_ + rgn + kc);
        f4 dkp, dgK;
        {   // pass K tile kt  [verbatim; cs partials replace the fused kernel's atomics]
            #pragma clang fp contract(off)
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(avn[vt], DHB(vt), a4[vt & 3]);
            const f4 acck = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            float kd[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                dkp[e] = acck[e] * ex2(gn[e] - gv[e]);
                kd[e] = bf2f(kk[e]) * dkp[e];
                dgK[e] = -kd[e];
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float c = msum(kd[e]);
                sm.cs2[buf][A][4 * q + e] = c;            // kra P-40: all 16 lanes hold c
            }
        }
        {
            float hd = 0.f;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const v4bh h0 = HB(2 * A + j), d0 = DHB(2 * A + j);
                #pragma unroll
                for (int e = 0; e < 4; ++e) hd += bf2f(h0[e]) * bf2f(d0[e]);
            }
            hd = xsum16(hd);
            sm.cs1[buf][A][pm] = hd;                      // kra P-40: 4 lanes, same value
        }
        // pass Q tile kt, ahead of the table barrier (ST=4 form)
        f4 dqiE;
        {
            #pragma clang fp contract(off)
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(ado[vt], HB(vt), a4[vt & 3]);
            const f4 accq = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            #pragma unroll
            for (int e = 0; e < 4; ++e) dqiE[e] = accq[e] * ex2(gv[e]) * p.scale;
        }
        f4 ep, en, g8;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            g8[e] = __shfl(gv[e], 16 * q + 8, 64);        // kra P-40: row 16A+8's g (== m==8 lane's gv)
            const float d = gv[e] - g8[e];
            ep[e] = ex2(d);
            en[e] = ex2(-d);
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float kf = bf2f(kk[e]), qf = bf2f(qq[e]);
            const int o = s4b_tix(4 * e + q, row);
            sm.TA[o] = kf * en[e];
            sm.TQ[o] = qf * ep[e];
            sm.TK[o] = kf * beta_t * ep[e];
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) sm.gM[A][4 * e + q] = g8[e];   // kra P-40: every lane, same value
        *reinterpret_cast<v4bh*>(&sm.dwTs[dwr * 64 + 4 * (dwc ^ dwr)]) = dwv;   // kra P-29
        lds_barrier();
        if (kt + 2 < 8) {                          // tile kt+2 -> the buffer tile kt freed
            stg4[(buf * 2 + 0) * 256 + threadIdx.x] = pfh;
            stg4[(buf * 2 + 1) * 256 + threadIdx.x] = pfd;
        }
        // p3 epilogue product: dkgb = A^T (-dw), tile kt -- dwT from GLOBAL (pitch kBT)
        f4 acc = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st)
            acc = mmac(aa[st], ld8(&sm.dwTs[pm * 64 + 4 * ((4 * st + q) ^ pm)]), acc);   // kra P-29

        f4 cq0 = {0.f, 0.f, 0.f, 0.f}, cq1 = cq0, ck0 = cq0, ck1 = cq0, ct0 = cq0, ct1 = cq0;
        {
        const float gm_n = sm.gM[A][m];
        #pragma unroll
        for (int b = 0; b <= A; ++b) {
            f4 t = *reinterpret_cast<const f4*>(&sm.TA[s4b_tix(m, 16 * b + 4 * q)]);
            if (b < A) t *= ex2_diff(gm_n, sm.gM[b][m]);
            cq0 = mm8(lo2(aqA[b]), lo2(t), cq0);
            ck0 = mm8(lo2(dak[b]), lo2(t), ck0);
            cq1 = mm8(hi2(aqA[b]), hi2(t), cq1);
            ck1 = mm8(hi2(dak[b]), hi2(t), ck1);
        }
        #pragma unroll
        for (int b = A; b < 4; ++b) {
            f4 tq = *reinterpret_cast<const f4*>(&sm.TQ[s4b_tix(m, 16 * b + 4 * q)]);
            f4 tk = *reinterpret_cast<const f4*>(&sm.TK[s4b_tix(m, 16 * b + 4 * q)]);
            if (b > A) {
                const float s = ex2_diff(sm.gM[b][m], gm_n);
                tq *= s;
                tk *= s;
            }
            ct0 = mm8(lo2(aqB[b - A]), lo2(tq), ct0);
            ct1 = mm8(lo2(akB[b - A]), lo2(tk), ct1);
            ct0 = mm8(hi2(aqB[b - A]), hi2(tq), ct0);
            ct1 = mm8(hi2(akB[b - A]), hi2(tk), ct1);
        }
        }
        const f4 dqi = dqiE, dki = dkp, dgi = dgK;
        f4 oq, ok, og, dbp;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            #pragma clang fp contract(off)
            const float kf = bf2f(kk[e]), qf = bf2f(qq[e]);
            const float eg = ex2(gv[e]);
            const float kg = kf * eg;
            float dk = dki[e] + acc[e] * (eg * beta_t);
            float dg = dgi[e] + kg * acc[e] * beta_t;
            db += acc[e] * kg;
            {                                             // kra P-40: select, never "+ 0"
                const int c = 4 * q + e;
                const float s1 = ((sm.cs1[buf][0][c] + sm.cs1[buf][1][c]) + sm.cs1[buf][2][c]) + sm.cs1[buf][3][c];
                const float s2 = ((sm.cs2[buf][0][c] + sm.cs2[buf][1][c]) + sm.cs2[buf][2][c]) + sm.cs2[buf][3][c];
                const float dgc = dg + (s1 * ex2(gn[e]) + s2);
                dg = row == rlast ? dgc : dg;
            }
            const float dq2 = (cq0[e] + cq1[e]) * ep[e];
            const float dk2 = (ck0[e] + ck1[e]) * ep[e];
            const float dkt = (ct0[e] + ct1[e]) * en[e];
            dbp[e] = dk2 * kf;
            const float dk2b = dk2 * beta_t;
            const float dqf = dqi[e] + dq2;
            oq[e] = dqf;
            ok[e] = dk + dk2b + dkt;
            og[e] = dg + qf * dqf + (dk2b - dkt) * kf;
        }
        dbacc += (dbp[0] + dbp[1]) + (dbp[2] + dbp[3]);
        if (rok) {
#if defined(__gfx938__)
            // kra P-938q: the tail's dtype cast done here (same bf_rne per element): dq / dk leave as bf16 and the
            // fp32 copies are neither written nor read back by bwd_tail
            if (p.dq_bf != nullptr) {
                v4bh bq, bk;
                #pragma unroll
                for (int e = 0; e < 4; ++e) { bq[e] = bf_rne(oq[e]); bk[e] = bf_rne(ok[e]); }
                *reinterpret_cast<v4bh*>(p.dq_bf + cb + ro + kc) = bq;
                *reinterpret_cast<v4bh*>(p.dk_bf + cb + ro + kc) = bk;
            } else
#endif
            {
            *reinterpret_cast<f4*>(dq_ + ro + kc) = oq;
            *reinterpret_cast<f4*>(dk_ + ro + kc) = ok;
            }
            *reinterpret_cast<f4*>(dg_ + ro + kc) = og;
        }
        lds_barrier();                             // table WAR: kt's reads done before kt+1 writes
    }
    db = xsum16(db) + xsum16(dbacc);
    if (q == 0 && rok) p.db[(size_t)(tok0 + row) * H + hh] = db;
}

template <bool VL>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 3)))
p3_wy_s4b_kernel(WyS4BParams p) {
    __shared__ __align__(16) S4BSmem sm;
    const int tid = threadIdx.x, r = tid >> 6, l = tid & 63, m = l & 15, q = l >> 4;
    // kra P-88: head-fastest grid when launched as dim3(H, NT).  kra P-96: the launcher says so explicitly --
    // inferring it from gridDim fell back to chunk-fastest whenever H == NT (e.g. T 4096 / H 64).
    const bool hf_ = p.hf != 0;
    const int i = hf_ ? (int)blockIdx.y : (int)blockIdx.x, hh = hf_ ? (int)blockIdx.x : (int)blockIdx.y, H = p.H;
    const int tr = 16 * r + m;
    const long tok0 = g2_row_tok<VL>(p.ctok, i);
        const int cl = g2_row_len<VL>(p.clen, i);
    // varlen: ONE uniform branch per CTA -- full chunks run the unmasked body (M=1)
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;
        const size_t cbA = g2_rm_tok(tok0, hh, H, kBT);
        const int RA = H * kBT;
        const bf16* A_ = p.A + cbA;
        const float* beta_ = p.beta + (size_t)tok0 * H + hh;
        v4bh aa[4];
        {   // Y4: stage A into the (still free) stg tile 0 with coalesced 8B loads,
            // gather the transposed column from LDS; stg is re-initialised for the kt
            // loop right after, so the staging is transient (zero LDS growth).
            short* As = reinterpret_cast<short*>(&sm.stg[0][0][0]);
            #pragma unroll   // kra P-34: all loads in flight before the first wait
            for (int it_ = 0; it_ < 4; ++it_) {
                const int j = tid + 256 * it_;
                const int row = j >> 4, qc = j & 15;
                // VL: rows past the chunk's end are NOT in the packed buffer -- never load them
                // (their aa entries are masked below anyway).
                *reinterpret_cast<v4bh*>(As + row * 64 + 4 * qc) = g2_row_ok<M>(row, cl)
                    ? *reinterpret_cast<const v4bh*>(A_ + row * RA + 4 * qc) : v4bh{0, 0, 0, 0};
            }
            lds_barrier();                                                  // A block visible
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    aa[st][e] = g2_row_ok<M>(16 * st + 4 * q + e, cl)
                        ? As[(16 * st + 4 * q + e) * 64 + tr] : (short)0;
            lds_barrier();                                                  // aa out; stg reused
        }
        const float beta_t = g2_row_ok<M>(tr, cl) ? beta_[tr * H] : 0.f;
        const float db0 = p.db_part[g2_nat_blk(i, hh, H, 0, true, 256) + tid];
        switch (__builtin_amdgcn_readfirstlane(r)) {
            case 0: s4b_tail<0, M>(p, sm, aa, db0, beta_t, tok0, cl); break;
            case 1: s4b_tail<1, M>(p, sm, aa, db0, beta_t, tok0, cl); break;
            case 2: s4b_tail<2, M>(p, sm, aa, db0, beta_t, tok0, cl); break;
            default: s4b_tail<3, M>(p, sm, aa, db0, beta_t, tok0, cl); break;
        }
    });
}

// kra P-20: + do_row (row-major do; read by s4b when do_ == nullptr)
// kra P-938q: + dq_bf / dk_bf (bf16 final dq / dk straight from s4b; null = the fp32 hand-off)
extern "C" int g2_wy_s4_split_r2(const void* q_kgb, const void* k, const void* v_vb,
                                 const void* v_new, const void* g, const void* beta,
                                 const void* A, const void* h, const void* do_, const void* dh,
                                 const void* dv, const void* dAqk, void* dq, void* dk,
                                 void* dv2, void* dg, void* db, void* dA, void* dwT,
                                 void* db_part, float scale, int T, int H, int NT,
                                 const void* q_row, const void* v_row, const int* ctok,
                                 const int* clen, const void* do_row, const void* dv2c,
                                 const void* vnc, void* dq_bf, void* dk_bf, int vna, hipStream_t s) {   // kra P-938va: + vna
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row tensors are packed.
    if (T % kBT || NT != T / kBT || ((ctok == nullptr) != (clen == nullptr))) return -1;
    {
        WyS4AParams pa{(const bf16*)q_kgb, (const bf16*)v_vb, (const bf16*)A, (const bf16*)h,
                       (const bf16*)dv, (const float*)beta, (const bf16*)v_row,
                       (bf16*)dv2, (bf16*)dwT, (float*)dA, (float*)db_part, T, H, NT,
                       ctok, clen, (const bf16*)dv2c};
        const dim3 s4grd = g2::g2_env_long("KDA_G2_S4_HF", 1) ? dim3(H, NT) : dim3(NT, H);   // kra P-88 (default on)
        pa.hf = g2::g2_env_long("KDA_G2_S4_HF", 1) != 0;   // kra P-96
        // kra P-28c (H=24 measured mixed on gfx936: 1.036x at 8K, 0.94-0.97x at 32K-64K).  kra P-938s4: on gfx938 the
        // staged form wins at every H (s4a 1.25x at 2048/16, 1.32x at 4096/32, 1.35x at 8192/64)
        int afs_maxh = g2::g2_cur_gfx938() ? (1 << 30) : 12;
        if (const char* e = g2::g2_getenv("KDA_G2_S4A_AFS_MAXH")) afs_maxh = atoi(e);
        const bool afs = H <= afs_maxh;
        const bool pf = g2::g2_env_long("KDA_G2_S4A_PF", 1) != 0;   // kra P-83 (default on)
        if (ctok) {
            if (afs) { if (pf) hipLaunchKernelGGL((p3_wy_s4a_kernel<true, true, 1>), s4grd, dim3(256), 0, s, pa);
                       else    hipLaunchKernelGGL((p3_wy_s4a_kernel<true, true>), s4grd, dim3(256), 0, s, pa); }
            else     { if (pf) hipLaunchKernelGGL((p3_wy_s4a_kernel<true, false, 1>), s4grd, dim3(256), 0, s, pa);
                       else    hipLaunchKernelGGL((p3_wy_s4a_kernel<true, false>), s4grd, dim3(256), 0, s, pa); }
        } else {
            if (afs) { if (pf && g2::g2_env_long("KDA_G2_S4A_AFSR", 1)) hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true, 2>), s4grd, dim3(256), 0, s, pa);   // kra P-83b
                       else if (pf) hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true, 1>), s4grd, dim3(256), 0, s, pa);
                       else    hipLaunchKernelGGL((p3_wy_s4a_kernel<false, true>), s4grd, dim3(256), 0, s, pa); }
            else     { if (pf) hipLaunchKernelGGL((p3_wy_s4a_kernel<false, false, 1>), s4grd, dim3(256), 0, s, pa);
                       else    hipLaunchKernelGGL((p3_wy_s4a_kernel<false, false>), s4grd, dim3(256), 0, s, pa); }
        }
        const int rc = (int)hipGetLastError();
        if (rc) return rc;
    }
    {
        WyS4BParams pb{(const bf16*)q_row, (const bf16*)k, (const bf16*)A, (const bf16*)h,
                       (const bf16*)do_, (const bf16*)dh, (const bf16*)v_new,
                       (const float*)g, (const float*)beta, (const float*)dAqk,
                       (const bf16*)dwT, (const float*)dA, (const float*)db_part,
                       (float*)dq, (float*)dk, (float*)dg, (float*)db, scale, T, H, NT,
                       ctok, clen, (const bf16*)do_row, (const bf16*)vnc};
        const dim3 s4grd = g2::g2_env_long("KDA_G2_S4_HF", 1) ? dim3(H, NT) : dim3(NT, H);   // kra P-88 (default on)
        pb.hf = g2::g2_env_long("KDA_G2_S4_HF", 1) != 0;   // kra P-96
        pb.dq_bf = (short*)dq_bf; pb.dk_bf = (short*)dk_bf;
        pb.vna = vna;
        if (ctok) hipLaunchKernelGGL(p3_wy_s4b_kernel<true>, s4grd, dim3(256), 0, s, pb);
        else hipLaunchKernelGGL(p3_wy_s4b_kernel<false>, s4grd, dim3(256), 0, s, pb);
        return (int)hipGetLastError();
    }
}

extern "C" int g2_wy_s4_split_r(const void* q_kgb, const void* k, const void* v_vb,
                                const void* v_new, const void* g, const void* beta,
                                const void* A, const void* h, const void* do_, const void* dh,
                                const void* dv, const void* dAqk, void* dq, void* dk,
                                void* dv2, void* dg, void* db, void* dA, void* dwT,
                                void* db_part, float scale, int T, int H, int NT,
                                const void* q_row, const void* v_row, const int* ctok,
                                const int* clen, const void* do_row, const void* dv2c,
                                const void* vnc, hipStream_t s) {
    return g2_wy_s4_split_r2(q_kgb, k, v_vb, v_new, g, beta, A, h, do_, dh, dv, dAqk, dq, dk, dv2, dg, db, dA,
                             dwT, db_part, scale, T, H, NT, q_row, v_row, ctok, clen, do_row, dv2c, vnc,
                             nullptr, nullptr, 0, s);
}

extern "C" int g2_wy_s4_split(const void* q_kgb, const void* k, const void* v_vb,
                              const void* v_new, const void* g, const void* beta,
                              const void* A, const void* h, const void* do_, const void* dh,
                              const void* dv, const void* dAqk, void* dq, void* dk,
                              void* dv2, void* dg, void* db, void* dA, void* dwT,
                              void* db_part, float scale, int T, int H, int NT,
                              const void* q_row, const void* v_row, const int* ctok,
                              const int* clen, hipStream_t s) {
    return g2_wy_s4_split_r(q_kgb, k, v_vb, v_new, g, beta, A, h, do_, dh, dv, dAqk, dq, dk, dv2,
                            dg, db, dA, dwT, db_part, scale, T, H, NT, q_row, v_row, ctok, clen,
                            nullptr, nullptr, nullptr, s);
}

// ================================================================ V1: l2n apply
// Backward no longer recomputes l2norm_fwd (sum-of-squares reduction): the
// forward already saved q_rstd/k_rstd, and U3 proved bf16(x*rstd) is bitwise
// equal to l2norm_fwd's output.  One kernel does BOTH q and k.
namespace {
__global__ void __launch_bounds__(256) g2_l2n_apply_kernel(
        const short* __restrict__ q, const short* __restrict__ k,
        const float* __restrict__ q_rstd, const float* __restrict__ k_rstd,
        short* __restrict__ qn, short* __restrict__ kn, long n4) {
    const long t = (long)blockIdx.x * 256 + threadIdx.x;   // v4bh index
    if (t >= n4) return;
    const long tok = t >> 5;                               // 32 quads per token row
    const v4bh qv = *reinterpret_cast<const v4bh*>(q + 4 * t);
    const v4bh kv = *reinterpret_cast<const v4bh*>(k + 4 * t);
    const float qr = q_rstd[tok], kr = k_rstd[tok];
    v4bh qo, ko;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        qo[e] = bf_rne(bf2f(qv[e]) * qr);
        ko[e] = bf_rne(bf2f(kv[e]) * kr);
    }
    *reinterpret_cast<v4bh*>(qn + 4 * t) = qo;
    *reinterpret_cast<v4bh*>(kn + 4 * t) = ko;
}
}  // anonymous namespace
}  // namespace g2

extern "C" int g2_l2n_apply(const void* q, const void* k,
                            const void* q_rstd, const void* k_rstd,
                            void* qn, void* kn, long total_tokens, hipStream_t s) {
    const long n4 = total_tokens * 32;   // v4bh quads (128 bf16 / 4 per token)
    hipLaunchKernelGGL(g2::g2_l2n_apply_kernel, dim3((n4 + 255) / 256), dim3(256),
                       0, s, (const short*)q, (const short*)k,
                       (const float*)q_rstd, (const float*)k_rstd,
                       (short*)qn, (short*)kn, n4);
    return (int)hipGetLastError();
}
