// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// T5: the whole backward tail (k3_g2's B7) as ONE HIP kernel + a fixed-order
// two-stage reduce.  Replaces, call-for-call and line-for-line in math:
//   fla/ops/utils/cumsum.py      chunk_local_cumsum_vector_kernel, REVERSE
//   fla/ops/kda/gate.py          kda_gate_bwd_kernel (USE_LOWER_BOUND) + the
//                                wrapper's round-to-g-dtype BEFORE dbias
//   fla/modules/l2norm.py        l2norm_bwd_kernel
//   fla/ops/common/gate.py       fused_beta_sigmoid_bwd_kernel
//
// Structure: one CTA per (chunk, head), 256 threads, NO big LDS.
//   Phase A (all 4 warps; warp w owns row block w = rows [16w,16w+16), each
//     lane owns TWO K columns l and l+64): reverse-inclusive cumsum of the
//     fp32 dg along the 64 chunk rows (sequential t=63..0 -- FLA's scan order
//     differs in association only), then the gate math per element with the
//     FLA operation order (t1 = lb*dsig; bdg = dyg*t1; bdg *= A), the dg_raw
//     store rounded to g_raw's dtype FIRST, and the per-column ddt / per-CTA
//     dA partials accumulated from the ROUNDED dg (wrapper semantics).
//   Phase B (4 warps, one warp per row): l2norm_bwd for q then k --
//     dx = dy*rstd - (dy.y) y rstd in fp32, fixed-order shfl tree for the row
//     dot, bf16 out.  use_l2=0 -> plain dtype cast.  The beta-sigmoid backward
//     rides along (64 elementwise values).
//   Reduce kernel: dA_log[h] = sum_i dA_part (sequential over chunks);
//     ddt_bias[h,d] = sum_i ddt_part.  Fixed order, NO atomics -- the two runs
//     of the kernel are bitwise identical.
#include "g2_common.h"

#include <type_traits>
namespace g2 {

struct TailParams {
    const float* dq; const float* dk; const float* dg; const float* db;
    const short* qn; const short* kn;
    const float* q_rstd; const float* k_rstd;
    const short* g_raw; const float* A_log; const float* dt_bias;
    const short* beta_raw;
    short* dqo; short* dko; short* dg_raw; short* db_out;
    float* dA_log; float* ddt_bias;          // A_log/dt_bias are fp32 (binding gate)
    float* dA_part; float* ddt_part;         // [NT*H], [NT*H*128]
    float lower_bound; float beta_scale;
    int use_l2; int use_beta;
    int H, NT;
    const int* ctok; const int* clen;        // varlen phase 2 (null = dense)
    // F1: gate_none -> precomputed-g contract (PR #556): no gate activation; dg_raw is
    // the fp32 reverse chunk-cumsum of dg (FLA chunk_local_cumsum(reverse)), no dA/ddt.
    // beta_f32 -> beta_raw / db_out are fp32 (fused_beta_sigmoid_bwd keeps x's dtype).
    int gate_none; int beta_f32;
    int no_cast = 0;                         // kra P-938q: dq / dk already final (s4b wrote bf16): skip the cast
    int order = 0, order_m = 4;              // kra P-938o2: CTA visiting order (g2_cta_order)
};

// VL (varlen phase 2): rows r >= clen load as 0 and are never stored; ALL arithmetic
// (incl. the Kahan updates) still runs on those zeros, so the result is bit-identical
// to the dense kernel on the zero-padded stream.
template <bool VL>
__global__ void __launch_bounds__(256) g2_bwd_tail_kernel(TailParams p) {
    int i, hh;
    g2_cta_order(p.order, p.order_m, p.H, p.NT, i, hh);   // kra P-938o2
    const int tid = threadIdx.x;
    const int wv = tid >> 6, l = tid & 63;              // wave64
    const long t0 = g2_row_tok<VL>(p.ctok, i);          // chunk row 0 (packed token)
    __shared__ float sDA[8 * 64];                      // per-(colhalf,warp) dA partials
    __shared__ float sBlk[8 * 64];                     // phase-A block totals
    __shared__ float sDDt[8 * 64];                     // per-(colhalf,warp) ddt partials
    const int cl = g2_row_len<VL>(p.clen, i);
    // varlen: ONE uniform branch per CTA -- full chunks run the unmasked body (M=1)
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;
    // kra P-938sa (gfx938): chunk-row accesses as a CTA-uniform base (SGPRs) + a 32-bit byte offset, so the compiler
    // emits saddr global ops instead of per-lane 64-bit address math (bwd_tail is VALU bound; ~23% of its instructions
    // were 64-bit address arithmetic).  Offsets stay < 2^32 (one chunk of rows of one tensor).  Same values.
    const size_t cb_ = ((size_t)t0 * p.H + hh) * kK;                  // element offset of (row t0, head hh, col 0)
    const uint32_t rs_ = (uint32_t)p.H * kK;                          // elements per token row
    auto eo = [&](int t, int col) -> uint32_t { return (uint32_t)t * rs_ + (uint32_t)col; };
    auto at = [&](auto* base, uint32_t e) {                            // base + e elements, 32-bit byte offset
        using E = std::remove_pointer_t<decltype(base)>;
        return reinterpret_cast<E*>(reinterpret_cast<std::conditional_t<std::is_const_v<E>, const char*, char*>>(base + cb_)
                                    + e * (uint32_t)sizeof(E));
    };

        // ---- Phase A: dg reverse cumsum + gate backward.  ALL FOUR warps work:
        // warp w owns row block w (rows [16w, 16w+16)); each lane owns TWO K
        // columns (l and l+64).  Pass 1 builds the block-local suffix sums in
        // registers (Kahan) and publishes the block totals; after the barrier
        // each warp adds the carry (suffix of the blocks ABOVE it, fixed order)
        // and does the gate math.
        if (p.gate_none) {                                  // uniform: reverse cumsum only
            const int tb = wv * 16;
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int d = c * 64 + l;
                float loc[16], run = 0.f, runC = 0.f;
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const int t = tb + 15 - u;
                    const float y = (g2_row_ok<M>(t, cl)
                                     ? *at(p.dg, eo(t, d)) : 0.f) - runC;
                    const float tt = run + y;
                    runC = (tt - run) - y;  run = tt;
                    loc[u] = run;
                }
                sBlk[(c * 4 + wv) * 64 + l] = run;
                __syncthreads();
                float carry = 0.f, carryC = 0.f;
                for (int b = wv + 1; b < 4; ++b) {
                    const float y = sBlk[(c * 4 + b) * 64 + l] - carryC, tt = carry + y;
                    carryC = (tt - carry) - y;  carry = tt;
                }
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const int t = tb + 15 - u;
                    if (g2_row_ok<M>(t, cl))
                        *at(reinterpret_cast<float*>(p.dg_raw), eo(t, d)) =
                            loc[u] + carry;
                }
                __syncthreads();                            // sBlk reuse by the next c
            }
        } else {
            const int tb = wv * 16;                         // block's first row
            const float A = expf(p.A_log[hh]);
            // kra P-48: lane l owns the ADJACENT columns d0 = 2l, d0 + 1 (float2 / short2 accesses);
            // partials go to the production (c, l) slot of their column -> production merge order
            const int d0 = 2 * l;
            auto slot = [&](int col, int b_) { return ((col >> 6) * 4 + b_) * 64 + (col & 63); };
            const float2 dtb = *reinterpret_cast<const float2*>(p.dt_bias + (size_t)hh * kK + d0);
#if defined(__gfx938__) && G2_SIGFAST && KDA_B1_SIG
            {   // kra P-938pk2: lane l's two columns (d0, d0 + 1) as ONE packed fp32 pair through the scan, the carry and
                // the gate backward (v_pk_add / v_pk_mul / v_pk_fma_f32: one VALU op for both columns; exp2 / rcp stay per
                // element).  Per column the same operations in the same order as the scalar path below; contraction is
                // off and the fmas the production forms in the dAs chain are written explicitly -> bit-identical.
                // This backend splits a packed fsub into two v_sub_f32 and a packed fneg into two v_xor, so a - b is
                // written as fma(b, -1, a): b * -1 is exact, one rounding of a - b, one v_pk_fma_f32.
                #pragma clang fp contract(off)
                typedef float pf2 __attribute__((ext_vector_type(2)));
                auto ps = [](pf2 a, pf2 b) { return __builtin_elementwise_fma(b, pf2{-1.f, -1.f}, a); };   // a - b
                pf2 loc[16], run = {0.f, 0.f}, runC = {0.f, 0.f};
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const int t = tb + 15 - u;
                    const float2 v = g2_row_ok<M>(t, cl)
                        ? *reinterpret_cast<const float2*>(at(p.dg, eo(t, d0)))
                        : float2{0.f, 0.f};
                    const pf2 y = ps(pf2{v.x, v.y}, runC), tt = run + y;
                    runC = ps(ps(tt, run), y);  run = tt;
                    loc[u] = run;
                }
                sBlk[slot(d0, wv)] = run.x;                 // block totals
                sBlk[slot(d0 + 1, wv)] = run.y;
                __syncthreads();
                pf2 carry = {0.f, 0.f}, carryC = {0.f, 0.f};      // blocks above
                for (int b = wv + 1; b < 4; ++b) {
                    const pf2 y = ps(pf2{sBlk[slot(d0, b)], sBlk[slot(d0 + 1, b)]}, carryC), tt = carry + y;
                    carryC = ps(ps(tt, carry), y);  carry = tt;
                }
                uint32_t gw[16];                            // g_raw of both columns (bf16 pair, column d0 low)
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const int t = tb + 15 - u;
                    gw[u] = *reinterpret_cast<const uint32_t*>(
                        (M == 2 ? p.g_raw + ((size_t)(t0 + g2_row_g<M>(t, cl)) * p.H + hh) * kK + d0   // partial chunk: the clamped
                             : at(p.g_raw, eo(t, d0))));                                      // row can be -1 (cl == 0)
                }
                const pf2 dt2 = {dtb.x, dtb.y};
                // dAs's compensation is carried negated, nC = -dAsC, so its fma needs no fneg.  nC' = y - s rounds to
                // -(s - y) exactly except that s == y gives +0 where -dAsC is -0 (likewise at u == 0); that only flips
                // the sign of an exact-zero y, and dAs + y is then unchanged (dAs starts at +0 and is never -0 under
                // RNE), so dAs, the only output of the chain, is bit-identical.
                pf2 dAs = {0.f, 0.f}, nC = {-0.f, -0.f}, ddt = {0.f, 0.f}, ddtC = {0.f, 0.f};
                uint32_t bo[16];
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const pf2 suffix = loc[u] + carry;
                    const pf2 gg = pf2{__uint_as_float(gw[u] << 16), __uint_as_float(gw[u] & 0xffff0000u)} + dt2;
                    const pf2 e = -(A * gg) * 1.4426950408889634f;                     // g2_sig_b1, both halves
                    const pf2 ex = pf2{__builtin_amdgcn_exp2f(e.x), __builtin_amdgcn_exp2f(e.y)} + 1.f;
                    const pf2 sg = pf2{__builtin_amdgcn_rcpf(ex.x), __builtin_amdgcn_rcpf(ex.y)};
                    pf2 bdg = suffix * (p.lower_bound * (sg * ps(pf2{1.f, 1.f}, sg)));   // dyg*t1
                    bdg *= A;
                    if (u == 0) {   // the production folds dAs = dAsC = 0 here: tt = fma(bdg, gg, 0), dAsC = tt - bdg * gg
                        const pf2 tt = __builtin_elementwise_fma(bdg, gg, pf2{0.f, 0.f});   // contracted (unrounded product)
                        nC = __builtin_elementwise_fma(bdg, gg, ps(pf2{0.f, 0.f}, tt));  dAs = tt;
                    } else {        // dAs += bdg * gg (compensated; the production's fma)
                        const pf2 y = __builtin_elementwise_fma(bdg, gg, nC), tt = dAs + y;
                        nC = ps(y, ps(tt, dAs));  dAs = tt;
                    }
                    const uint32_t b2 = (uint32_t)__builtin_hcu_cvt_pk_bf16_f32(0, bdg.x, 0, bdg.y, false);  // bf_rne x2
                    {   // ddt += bf2f(b16)
                        const pf2 y = ps(pf2{__uint_as_float(b2 << 16), __uint_as_float(b2 & 0xffff0000u)}, ddtC),
                                  tt = ddt + y;
                        ddtC = ps(ps(tt, ddt), y);  ddt = tt;
                    }
                    bo[u] = b2;
                }
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const int t = tb + 15 - u;
                    if (g2_row_ok<M>(t, cl))
                        *reinterpret_cast<uint32_t*>(at(p.dg_raw, eo(t, d0))) = bo[u];
                }
                sDDt[slot(d0, wv)] = ddt.x;                 // merged below in the production order
                sDDt[slot(d0 + 1, wv)] = ddt.y;
                sDA[slot(d0, wv)] = dAs.x;
                sDA[slot(d0 + 1, wv)] = dAs.y;
            }
#else
            float loc0[16], loc1[16], run0 = 0.f, run0C = 0.f, run1 = 0.f, run1C = 0.f;
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                const int t = tb + 15 - u;
                const float2 v = g2_row_ok<M>(t, cl)
                    ? *reinterpret_cast<const float2*>(at(p.dg, eo(t, d0)))
                    : float2{0.f, 0.f};
#if defined(__gfx938__)
                {   // kra P-938pk: both columns' Kahan step as packed fp32 add / sub (no fma here: bit-identical)
                    typedef float pf2 __attribute__((ext_vector_type(2)));
                    const pf2 vv = pf2{v.x, v.y}, rc = pf2{run0C, run1C}, rr = pf2{run0, run1};
                    const pf2 y = vv - rc, tt = rr + y, cn = (tt - rr) - y;
                    run0C = cn.x; run1C = cn.y; run0 = tt.x; run1 = tt.y;
                    loc0[u] = run0; loc1[u] = run1;
                }
#else
                {
                    const float y = v.x - run0C;
                    const float tt = run0 + y;
                    run0C = (tt - run0) - y;  run0 = tt;
                    loc0[u] = run0;
                }
                {
                    const float y = v.y - run1C;
                    const float tt = run1 + y;
                    run1C = (tt - run1) - y;  run1 = tt;
                    loc1[u] = run1;
                }
#endif
            }
            sBlk[slot(d0, wv)] = run0;                      // block totals
            sBlk[slot(d0 + 1, wv)] = run1;
            __syncthreads();
            float carry0 = 0.f, carry0C = 0.f, carry1 = 0.f, carry1C = 0.f;   // blocks above
            for (int b = wv + 1; b < 4; ++b) {
                {
                    const float y = sBlk[slot(d0, b)] - carry0C, tt = carry0 + y;
                    carry0C = (tt - carry0) - y;  carry0 = tt;
                }
                {
                    const float y = sBlk[slot(d0 + 1, b)] - carry1C, tt = carry1 + y;
                    carry1C = (tt - carry1) - y;  carry1 = tt;
                }
            }
            float dAs0 = 0.f, dAs0C = 0.f, ddt0 = 0.f, ddt0C = 0.f;
            float dAs1 = 0.f, dAs1C = 0.f, ddt1 = 0.f, ddt1C = 0.f;
            short b0[16], b1[16];
            short2 gr2[16];                                 // kra P-54a: g_raw of both columns, 4 B per row
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                const int t = tb + 15 - u;
                gr2[u] = *reinterpret_cast<const short2*>(
                    (M == 2 ? p.g_raw + ((size_t)(t0 + g2_row_g<M>(t, cl)) * p.H + hh) * kK + d0   // partial chunk: the clamped
                         : at(p.g_raw, eo(t, d0))));                                      // row can be -1 (cl == 0)
            }
            // per column a separate straight-line block (the opaque branch keeps the SLP vectorizer from
            // pairing the two columns into packed fp32 ops, which would un-fuse the production's fmas)
            auto gate_col = [&](const float (&loc)[16], float carry, int j, float dtbj, short (&bo)[16],
                                float& dAs, float& dAsC, float& ddt, float& ddtC) {
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    const int t = tb + 15 - u;
                    const float suffix = loc[u] + carry;
                    const float gg = bf2f(j ? gr2[u].y : gr2[u].x) + dtbj;   // kra P-54a
#if G2_SIGFAST && KDA_B1_SIG
                    const float sg = g2_sig_b1(A * gg);
#else
                    const float sg = 1.f / (1.f + expf(-(A * gg)));
#endif
                    float bdg = suffix * (p.lower_bound * (sg * (1.f - sg)));   // dyg*t1
                    bdg *= A;
                    {   // dAs += bdg * gg (compensated)
                        const float y = bdg * gg - dAsC, tt = dAs + y;
                        dAsC = (tt - dAs) - y;  dAs = tt;
                    }
                    const short b16 = bf_rne(bdg);          // round to g_raw dtype
                    {   // ddt += bf2f(b16)
                        const float y = bf2f(b16) - ddtC, tt = ddt + y;
                        ddtC = (tt - ddt) - y;  ddt = tt;
                    }
                    bo[u] = b16;
                }
            };
            gate_col(loc0, carry0, 0, dtb.x, b0, dAs0, dAs0C, ddt0, ddt0C);
            if (__builtin_amdgcn_readfirstlane(p.NT) > 0)           // opaque, always true
                gate_col(loc1, carry1, 1, dtb.y, b1, dAs1, dAs1C, ddt1, ddt1C);
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                const int t = tb + 15 - u;
                if (g2_row_ok<M>(t, cl))
                    *reinterpret_cast<short2*>(at(p.dg_raw, eo(t, d0))) = short2{b0[u], b1[u]};
            }
            sDDt[slot(d0, wv)] = ddt0;                      // merged below in the production order
            sDDt[slot(d0 + 1, wv)] = ddt1;
            sDA[slot(d0, wv)] = dAs0;
            sDA[slot(d0 + 1, wv)] = dAs1;
#endif
        }
        __syncthreads();
        if (!p.gate_none && tid < kK) {                     // fixed-order per-column
            const int c = tid >> 6, l2 = tid & 63;           // merge of the 4 warps
            float sd = 0.f;
            #pragma unroll
            for (int b = 0; b < 4; ++b) sd += sDDt[(c * 4 + b) * 64 + l2];
            p.ddt_part[((size_t)(hh * kK + tid)) * p.NT + i] = sd;
        }
#if defined(__gfx938__) && KDA_B2_TREE
        if (!p.gate_none && tid < 64) {                     // kra P-938t2: fixed-order tree on wave 0
            float s = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) s += sDA[j * 64 + tid];
            #pragma unroll
            for (int off = 32; off > 0; off >>= 1) s += __shfl_xor(s, off, 64);   // a + b == b + a: lanes agree
            if (tid == 0) p.dA_part[(size_t)hh * p.NT + i] = s;
        }
#else
        if (!p.gate_none && tid == 0) {                     // fixed-order CTA sum
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < 8 * 64; ++k) s += sDA[k];
            p.dA_part[(size_t)hh * p.NT + i] = s;
        }
#endif

        // ---- Phase B: l2norm backward (one warp per row), then the beta tail
        if (!p.use_l2) {
#if defined(__gfx938__)
            if (p.no_cast) {} else
            // kra P-938c: the cast as 16 B loads (lane: 4 columns of one of 2 rows per instruction), all 16 of the
            // warp's loads (dq, dk x its 16 rows) issued before the first convert; 8 B stores.  Same bf_rne per element.
            {
                const int cq = 4 * (l & 31), rh = l >> 5;
                f4 x[2][8];
                #pragma unroll
                for (int w2 = 0; w2 < 2; ++w2) {
                    const float* dy = w2 ? p.dk : p.dq;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int r = wv * 16 + 2 * j + rh;
                        x[w2][j] = g2_row_ok<M>(r, cl)
                            ? *reinterpret_cast<const f4*>(dy + ((size_t)(t0 + r) * p.H + hh) * kK + cq) : f4{0.f, 0.f, 0.f, 0.f};
                    }
                }
                #pragma unroll
                for (int w2 = 0; w2 < 2; ++w2) {
                    short* out = w2 ? p.dko : p.dqo;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int r = wv * 16 + 2 * j + rh;
                        if (g2_row_ok<M>(r, cl)) {
                            v4bh o;
                            #pragma unroll
                            for (int e = 0; e < 4; ++e) o[e] = bf_rne(x[w2][j][e]);
                            *reinterpret_cast<v4bh*>(out + ((size_t)(t0 + r) * p.H + hh) * kK + cq) = o;
                        }
                    }
                }
            }
            if (false)
#endif
            for (int which = 0; which < 2; ++which) {
                const float* dy = which ? p.dk : p.dq;
                short* out = which ? p.dko : p.dqo;
                for (int r = wv; r < kBT; r += 4) {
                    if (!g2_row_ok<M>(r, cl)) continue;           // wave-uniform
                    const size_t so = ((size_t)(t0 + r) * p.H + hh) * kK;
                    const int c2 = l * 2;
                    const float2 dyv = *reinterpret_cast<const float2*>(dy + so + c2);
                    short2 o;
                    o.x = bf_rne(dyv.x);
                    o.y = bf_rne(dyv.y);
                    *reinterpret_cast<short2*>(out + so + c2) = o;
                }
            }
        } else {
            // kra P-54a: lane j (j < 16) of the warp holds rstd of row 16w + j; rows are broadcast with
            // v_readlane (the per-row uniform-address VMEM loads go).  Rows past the end hold 0, as before.
            float rsl[2];
            {
                const int jr = wv * 16 + (l & 15);
                const bool okr = g2_row_ok<M>(jr, cl);
                rsl[0] = okr ? p.q_rstd[(size_t)(t0 + jr) * p.H + hh] : 0.f;
                rsl[1] = okr ? p.k_rstd[(size_t)(t0 + jr) * p.H + hh] : 0.f;
            }
            // q and k interleaved at the 4-row batch level: 8 row-loads in flight
            for (int rb = 0; rb < 16; rb += 4) {
                const int r0 = wv * 16 + rb;                // warp w owns rows [16w,16w+16)
                if (!g2_row_ok<M>(r0, cl)) continue;       // wave-uniform; rows r0..r0+3
                const long row = t0 + r0;
                const size_t so = (size_t)row * p.H * kK + hh * kK;
                const int c2 = l * 2;
                float2 dyv[2][4]; short2 yv[2][4]; float r4[2][4];
                #pragma unroll
                for (int w2 = 0; w2 < 2; ++w2) {
                    const float* dy = w2 ? p.dk : p.dq;
                    const short* y = w2 ? p.kn : p.qn;
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        if (g2_row_ok<M>(r0 + u, cl)) {
                            dyv[w2][u] = *reinterpret_cast<const float2*>(dy + so + u * (p.H * kK) + c2);
                            yv[w2][u] = *reinterpret_cast<const short2*>(y + so + u * (p.H * kK) + c2);
                            r4[w2][u] = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rsl[w2]), rb + u));
                        } else {
                            dyv[w2][u] = float2{0.f, 0.f}; yv[w2][u] = short2{0, 0}; r4[w2][u] = 0.f;
                        }
                    }
                }
                #pragma unroll
                for (int w2 = 0; w2 < 2; ++w2) {
                    short* out = w2 ? p.dko : p.dqo;
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const float y0 = bf2f(yv[w2][u].x), y1 = bf2f(yv[w2][u].y);
                        float dot = dyv[w2][u].x * y0 + dyv[w2][u].y * y1;
                        #pragma unroll
                        for (int off = 1; off < 64; off <<= 1)
                            dot += __shfl_xor(dot, off, 64);    // fixed-order tree
                        short2 o;
                        o.x = bf_rne_int(dyv[w2][u].x * r4[w2][u] - dot * y0 * r4[w2][u]);
                        o.y = bf_rne_int(dyv[w2][u].y * r4[w2][u] - dot * y1 * r4[w2][u]);
                        if (g2_row_ok<M>(r0 + u, cl))
                            *reinterpret_cast<short2*>(out + so + u * (p.H * kK) + c2) = o;
                    }
                }
            }
        }
        if (p.use_beta && tid < kBT && g2_row_ok<M>(tid, cl)) {   // fused_beta_sigmoid_bwd
            const size_t so = (size_t)(t0 + tid) * p.H + hh;
            const float x = p.beta_f32 ? reinterpret_cast<const float*>(p.beta_raw)[so]
                                       : bf2f(p.beta_raw[so]);
            const float s = 1.f / (1.f + expf(-x));
            float dx = p.db[so] * p.beta_scale * s;         // dy*scale*b_y
            dx *= (1.f - s);
            if (p.beta_f32) reinterpret_cast<float*>(p.db_out)[so] = dx;
            else p.db_out[so] = bf_rne(dx);
        }
    });
}

// Fixed-order second stage: one CTA per head (the single-CTA form capped at
// ~13 GB/s and cost 0.23 ms for 3 MB of partials).  dA: [h][NT]; ddt: [h*K+d][NT].
__global__ void g2_bwd_tail_reduce_kernel(TailParams p) {
    const int hh = blockIdx.x, tid = threadIdx.x;      // grid (H), 256 threads
    const int NT = p.NT;
    if (tid < kK) {                                    // ddt: 128 columns/head
        const size_t base = (size_t)(hh * kK + tid) * NT;
        float s = 0.f, c = 0.f;                        // compensated (NT adds)
        for (int i = 0; i < NT; ++i) {
            const float y = p.ddt_part[base + i] - c, tt = s + y;
            c = (tt - s) - y;  s = tt;
        }
        // FLA's wrapper computes dbias on the bf16-rounded dg tensor with
        // torch.sum, whose OUTPUT dtype is bf16 -- the fp32 ddt_bias the harness
        // sees is that bf16 value widened.  Quantize through bf16 to match.
        p.ddt_bias[(size_t)hh * kK + tid] = bf2f(bf_rne(s));
    } else if (tid < kK + 64) {                        // dA (one warp's worth)
        const int h = hh;                              // dA layout is [h][NT]
        float s = 0.f;
        for (int i = 0; i < NT; ++i) s += p.dA_part[(size_t)h * NT + i];
        if (tid == kK) p.dA_log[h] = s;
    }
}

}  // namespace g2

extern "C" int g2_bwd_tail2(const void* dq, const void* dk, const void* dg,
                           const void* db, const void* qn, const void* kn,
                           const void* q_rstd, const void* k_rstd,
                           const void* g_raw, const void* A_log,
                           const void* dt_bias, const void* beta_raw,
                           void* dqo, void* dko, void* dg_raw, void* dA_log,
                           void* ddt_bias, void* db_out,
                           void* dA_part, void* ddt_part,
                           double lower_bound, double beta_scale,
                           int use_l2, int use_beta,
                           int B, int T, int H, const int* ctok, const int* clen, int NTv,
                           int gate_none, int beta_f32, int no_cast, hipStream_t s) {
    // Dense: T rows, NT = T/64.  Varlen phase 2: T packed rows, NTv chunks (ctok/clen).
    if (B != 1 || (ctok == nullptr && T % 64) || ((ctok == nullptr) != (clen == nullptr)))
        return -1;
    const int NT = ctok ? NTv : T / 64;
    g2::TailParams p{(const float*)dq, (const float*)dk, (const float*)dg,
                     (const float*)db, (const short*)qn, (const short*)kn,
                     (const float*)q_rstd, (const float*)k_rstd,
                     (const short*)g_raw, (const float*)A_log,
                     (const float*)dt_bias, (const short*)beta_raw,
                     (short*)dqo, (short*)dko, (short*)dg_raw, (short*)db_out,
                     (float*)dA_log, (float*)ddt_bias,
                     (float*)dA_part, (float*)ddt_part,
                     (float)lower_bound, (float)beta_scale,
                     use_l2, use_beta, H, NT, ctok, clen, gate_none, beta_f32};
    p.no_cast = (no_cast && !use_l2) ? 1 : 0;
    // kra P-938o2 (gfx938): row-major in and out -> head-fastest at H % 32 == 0
    p.order = (int)g2::g2_env_long("KDA_G2_TAIL_ORDER", g2::g2_cur_gfx938() && H % 32 == 0 ? 1 : 0);
    p.order_m = (int)g2::g2_env_long("KDA_G2_TAIL_ORDER_M", 4);
    if (p.order == 3 && (p.order_m < 1 || H % p.order_m)) p.order = 0;
    const dim3 tgrid(NT, H);
    if (ctok)
        hipLaunchKernelGGL(g2::g2_bwd_tail_kernel<true>, tgrid, dim3(256), 0, s, p);
    else
        hipLaunchKernelGGL(g2::g2_bwd_tail_kernel<false>, tgrid, dim3(256), 0, s, p);
    if (!gate_none)
        hipLaunchKernelGGL(g2::g2_bwd_tail_reduce_kernel, dim3(H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}

extern "C" int g2_bwd_tail(const void* dq, const void* dk, const void* dg,
                           const void* db, const void* qn, const void* kn,
                           const void* q_rstd, const void* k_rstd,
                           const void* g_raw, const void* A_log,
                           const void* dt_bias, const void* beta_raw,
                           void* dqo, void* dko, void* dg_raw, void* dA_log,
                           void* ddt_bias, void* db_out,
                           void* dA_part, void* ddt_part,
                           double lower_bound, double beta_scale,
                           int use_l2, int use_beta,
                           int B, int T, int H, const int* ctok, const int* clen, int NTv,
                           int gate_none, int beta_f32, hipStream_t s) {
    return g2_bwd_tail2(dq, dk, dg, db, qn, kn, q_rstd, k_rstd, g_raw, A_log, dt_bias, beta_raw, dqo, dko, dg_raw,
                        dA_log, ddt_bias, db_out, dA_part, ddt_part, lower_bound, beta_scale, use_l2, use_beta, B, T,
                        H, ctok, clen, NTv, gate_none, beta_f32, 0, s);
}
