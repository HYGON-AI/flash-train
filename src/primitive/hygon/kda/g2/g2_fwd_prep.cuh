// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// R4 (Claude, 2026-10-07): single-pass forward prep (KDA_G2_FI=4).
//
// Replaces head_l2n + head_gate + fwd_intra2a/2b + gn.to_gn (FI=3) with two kernels that never
// materialize qn/kn, the fp32 gate cumsum or the per-chunk Gn permute:
//   prep_a  raw q/k/g/beta -> q/k rstd, Gn, QgA, Kn, R (fp32, as fwd_intra2a), and
//           Kb = bf16(kn * beta * 2^g) (the P3 T2 operand fwd_intra2b used to form from kn + g)
//   prep_b  fwd_intra2b's post with the T2 pass-0 operand read from Kb.
// Bitwise contract: every output equals FI=3 bit for bit.
//   l2norm: per row the 16 "l2n2 lanes" j = 8-column groups are held here by (wave w, q4)
//           with j = 4w + q4 (cbase = 32w + 8q4 is l2n2's 8j); the shfl butterfly 1,2 runs
//           inside the wave (lane ^16, ^32) and steps 4,8 cross waves through LDS as
//           (Y0+Y1)+(Y2+Y3) -- the same tree (fp add is commutative, so every lane of the
//           butterfly holds the same value).  rs and the bf16 rounding are the same expressions.
//   gate:   g2_gate_kernel's body verbatim (contract off, same lane/quarter mapping, Kahan,
//           fixed-order carry), results kept in a 32 KB LDS tile instead of global.
//   P1/RED/post: fwd_intra2a / fis::s_post verbatim; g reads come from the LDS tile.
#include "g2_common.h"
#include <type_traits>

namespace g2 {
namespace r4 {

struct PrepAParams {
    const short *q, *k, *v;
    short *Wn, *Un, *AqkA, *Aqk, *Akk;  // post outputs (single-kernel prep)
    float scale;
    const void* g_raw;                  // GATE=1: bf16 raw gate, GATE=0: fp32 precomputed g
    const float *A_log, *dt_bias, *beta;
    short *QgA, *Kn, *Kb;
    float *R, *q_rstd, *k_rstd, *Gn;
    const int *ctok, *clen;
    float lower_bound, gscale;
    int H, NT;
    int stop;                           // phase timing (G2_PREP_STOP): 0 = full
    int kp = 0;                         // kra P-61e: Kn lane-permuted + Gn natural-k (H <= KPERM_MAX_H)
    short *qn_out = nullptr, *kn_out = nullptr;   // kra P-77b: non-null -> emit qn / kn row-major
    int order = 0, order_m = 4;         // kra P-938o2: CTA visiting order (g2_cta_order)
};

struct PrepBParams {
    const short *Kb, *v; const float *beta, *R;
    short *Wn, *Un, *AqkA, *Aqk, *Akk;
    const int *ctok, *clen;
    float scale;
    int H, NT;
    int kp = 0;   // kra P-62: Un with the pm column order (as Wn) when H <= KPERM_MAX_H
    int stop = 0; // phase timing (G2_PREPB_STOP, dev only): 1 after the R loads, 2 after Aqk / AqkA, 3 after the
                  // solve + Akk, 4 after pass 0 (Wn); 0 = full
    int order = 0, order_m = 4;   // kra P-938o2: CTA visiting order (g2_cta_order)
};


// g2_gate_kernel body (Z6: contraction pinned off), writing the tile instead of global.
// Rows >= cl are not written (never read: g loads use the clamped row).
// Raw gate per lane (lane c: columns 2c, 2c+1; wave w: rows 16w..16w+15): GATE=1 keeps the bf16
// pair as one 32-bit word (the float conversion + dt_bias add happen in gate_scan, the same
// expression bf2f(x) + dt as g2_gate_kernel); GATE=0 keeps the fp32 pair.
template <int GATE> struct GRaw { float2 v[16]; };
template <> struct GRaw<1> { uint32_t v[16]; };

template <int GATE, bool VL>
__device__ __forceinline__ void gate_load(const PrepAParams& p, GRaw<GATE>& gr, int hh, long t0,
                                          int cl) {
    const int c = threadIdx.x & 63, w = threadIdx.x >> 6;
    const int r0 = 16 * w;
#if defined(__gfx938__)
    // kra P-938ga: one wave-uniform chunk base + 32-bit unsigned lane offsets (saddr form, no per-lane 64-bit
    // address pairs); the same addresses as below
    const uint32_t step32 = (uint32_t)p.H * 128u, lo = 2u * (uint32_t)c;
    const size_t cb = ((size_t)t0 * p.H + hh) * 128;
    #pragma unroll
    for (int u = 0; u < 16; ++u) {
        const int r = r0 + u;
        const bool ok = !VL || r < cl;
        const uint32_t o = lo + (uint32_t)r * step32;
        if constexpr (GATE == 1) {
            gr.v[u] = ok ? *reinterpret_cast<const uint32_t*>(reinterpret_cast<const short*>(p.g_raw) + cb + o) : 0u;
        } else {
            gr.v[u] = ok ? *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(p.g_raw) + cb + o)
                         : float2{0.f, 0.f};
        }
    }
#else
    const size_t step = (size_t)p.H * 128;
    const size_t base = ((size_t)t0 * p.H + hh) * 128 + 2 * c;
    #pragma unroll
    for (int u = 0; u < 16; ++u) {
        const int r = r0 + u;
        const bool ok = !VL || r < cl;
        if constexpr (GATE == 1) {
            gr.v[u] = ok ? *reinterpret_cast<const uint32_t*>(
                               reinterpret_cast<const short*>(p.g_raw) + base + (size_t)r * step) : 0u;
        } else {
            gr.v[u] = ok ? *reinterpret_cast<const float2*>(
                               reinterpret_cast<const float*>(p.g_raw) + base + (size_t)r * step)
                         : float2{0.f, 0.f};
        }
    }
#endif
}

#if defined(__gfx938__)
// kra P-938pk2: packed fp32 pairs.  This backend splits a packed fsub / fneg into two scalar ops, so a - b is
// fma(b, -1, a) (b * -1 is exact: one rounding of a - b, one v_pk_fma_f32).  bf16 pairs as one 32-bit word.
__device__ __forceinline__ f2 pk_sub(f2 a, f2 b) { return __builtin_elementwise_fma(b, f2{-1.f, -1.f}, a); }
__device__ __forceinline__ f2 bf2f2(uint32_t w) { return f2{__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)}; }
__device__ __forceinline__ uint32_t bf_rne2(f2 x) { return (uint32_t)__builtin_hcu_cvt_pk_bf16_f32(0, x.x, 0, x.y, false); }
typedef uint32_t u2x __attribute__((ext_vector_type(2)));
#endif
#ifndef G2_PK2                                       // P-938pk2 parts (dev bisect): 1 gate scan / carry, 2 QgA / Kb / Kn, 4 P1a / P1b
#define G2_PK2 7
#endif

// G half tile: 32 rows x 128 fp32 (quarters 2h, 2h+1), columns XOR-swizzled by row.
__device__ __forceinline__ int gsw(int r, int c) { return r * 128 + (c ^ ((r & 7) << 2)); }


// g2_gate_kernel's scan + carry, every lane once (lane c: columns 2c, 2c+1; wave w: quarter w).
// v holds the raw (bf16 + dt_bias / fp32) gate on entry and the in-quarter prefix on exit.
template <int GATE, bool VL>
__device__ __forceinline__ void gate_scan(const PrepAParams& p, const GRaw<GATE>& gr,
                                            float2 (&v)[16], float2* sTot, int hh, int cl) {
    #pragma clang fp contract(off)
    const int c = threadIdx.x & 63, w = threadIdx.x >> 6;
    const int r0 = 16 * w;
    float A = 0.f, dt0 = 0.f, dt1 = 0.f;
    if constexpr (GATE == 1) {
        A = expf(p.A_log[hh]);
        const float2 d2 = *reinterpret_cast<const float2*>(p.dt_bias + hh * 128 + 2 * c);
        dt0 = d2.x; dt1 = d2.y;
    }
#if defined(__gfx938__) && (G2_PK2 & 1)
    if constexpr (GATE == 1 && KDA_B1_SIG) {          // kra P-938pk2: both columns as one packed pair, same ops / order
        const f2 dt2 = {dt0, dt1};
        f2 rn = {0.f, 0.f}, cm = {0.f, 0.f};
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            const f2 e = -(A * (bf2f2(gr.v[u]) + dt2)) * 1.4426950408889634f;     // g2_sig_b1's argument
            const f2 ex = f2{__builtin_amdgcn_exp2f(e.x), __builtin_amdgcn_exp2f(e.y)} + 1.f;
            f2 gt = p.lower_bound * f2{__builtin_amdgcn_rcpf(ex.x), __builtin_amdgcn_rcpf(ex.y)};
            if (VL && r0 + u >= cl) gt = f2{0.f, 0.f};
            const f2 y = pk_sub(gt, cm), t = rn + y;
            cm = pk_sub(pk_sub(t, rn), y); rn = t;
            v[u] = float2{rn.x, rn.y};
        }
        const f2 tot = pk_sub(rn, cm);
        sTot[w * 64 + c] = float2{tot.x, tot.y};
        return;
    }
#endif
    float2 run = float2{0.f, 0.f}, cmp = float2{0.f, 0.f};
#if defined(__gfx938__)
    // P-938sg: one wave-uniform test (all |x| <= 87 and no NaN) selects the range-free expf / divide
    bool fast = false;
    if constexpr (GATE == 1 && !KDA_B1_SIG) {   // kra P-938sd: B1 never reads the range test
        float mx = 0.f, sx = 0.f, sy = 0.f;
        #pragma unroll
        for (int u = 0; u < 16; ++u) {
            const float gx = bf2f((short)(gr.v[u] & 0xffffu)) + dt0, gy = bf2f((short)(gr.v[u] >> 16)) + dt1;
            const float xx = -(A * gx), xy = -(A * gy);
            mx = fmaxf(mx, fmaxf(fabsf(xx), fabsf(xy)));
            sx += xx; sy += xy;                              // NaN anywhere -> NaN
        }
        fast = __all(mx <= 87.f && sx == sx && sy == sy);
    }
#endif
    #pragma unroll
    for (int u = 0; u < 16; ++u) {
        float2 gt;
        if constexpr (GATE == 1) {
            // g2_gate_kernel: x = {bf2f(s2.x) + dt0, bf2f(s2.y) + dt1}; rows past the end load 0
            // there as float2{0,0} WITHOUT the dt add -- they are zeroed below anyway (VL only).
            gt = float2{bf2f((short)(gr.v[u] & 0xffffu)) + dt0, bf2f((short)(gr.v[u] >> 16)) + dt1};
        } else {
            gt = gr.v[u];
        }
        if constexpr (GATE == 1) {
#if defined(__gfx938__)
#if KDA_B1_SIG
            if (true) {
                gt.x = p.lower_bound * g2_sig_b1(A * gt.x);
                gt.y = p.lower_bound * g2_sig_b1(A * gt.y);
            } else
#endif
            if (fast) {
                gt.x = p.lower_bound * rcp1(1.f + expf_core(-(A * gt.x)));
                gt.y = p.lower_bound * rcp1(1.f + expf_core(-(A * gt.y)));
            } else
#endif
            {
            gt.x = p.lower_bound * (1.f / (1.f + expf(-(A * gt.x))));
            gt.y = p.lower_bound * (1.f / (1.f + expf(-(A * gt.y))));
            }
        }
        if (VL && r0 + u >= cl) gt = float2{0.f, 0.f};
        {
            const float yx = gt.x - cmp.x, tx = run.x + yx;
            cmp.x = (tx - run.x) - yx; run.x = tx;
            const float yy = gt.y - cmp.y, ty = run.y + yy;
            cmp.y = (ty - run.y) - yy; run.y = ty;
        }
        v[u] = float2{run.x, run.y};
    }
    sTot[w * 64 + c] = float2{run.x - cmp.x, run.y - cmp.y};
}

// The carry of the lower quarters (fixed order, compensated): after a barrier on sTot.
__device__ __forceinline__ float2 gate_carry(const float2* sTot) {
    #pragma clang fp contract(off)
    const int c = threadIdx.x & 63, w = threadIdx.x >> 6;
#if defined(__gfx938__) && (G2_PK2 & 1)
    {   // kra P-938pk2: packed pair
        f2 cy = {0.f, 0.f}, cq = {0.f, 0.f};
        for (int b = 0; b < w; ++b) {
            const float2 s = sTot[b * 64 + c];
            const f2 y = pk_sub(f2{s.x, s.y}, cq), t = cy + y;
            cq = pk_sub(pk_sub(t, cy), y); cy = t;
        }
        const f2 r = pk_sub(cy, cq);
        return float2{r.x, r.y};
    }
#endif
    float2 carry = float2{0.f, 0.f}, cc = float2{0.f, 0.f};
    for (int b = 0; b < w; ++b) {
        const float yx = sTot[b * 64 + c].x - cc.x, tx = carry.x + yx;
        cc.x = (tx - carry.x) - yx; carry.x = tx;
        const float yy = sTot[b * 64 + c].y - cc.y, ty = carry.y + yy;
        cc.y = (ty - carry.y) - yy; carry.y = ty;
    }
    return float2{carry.x - cc.x, carry.y - cc.y};
}

// V2 (kra P-2): hardware exp2 for arguments PROVEN to lie in [-126, 128).  exp2f lowers to a
// 6-instruction sequence (cmp, cndmask, add, v_exp, cndmask, mul) that only differs from a
// bare v_exp_f32 when the result would be denormal (x < -126); inside the range it adds 0 and
// multiplies by 1.0, so the result is bitwise identical.  Only valid under the safe-gate
// contract (per-token decay >= -5 nat = -7.2135 log2, the same contract the 16-row sub-blocks
// with a mid/start reference rely on): |x| <= 8 rows * 7.2135 = 57.7 (P1a) and
// <= 15 rows * 7.2135 = 108.2 (P1b eO).
__device__ __forceinline__ float exp2_inrange(float x) { return __builtin_amdgcn_exp2f(x); }

// L2N = false (kra P-94): q / k arrive already normalized (Megatron-LM-k3 runs FLA's l2norm before
// the call, use_qk_l2norm_in_kernel=False): qn = q, kn = k bit for bit, so skipping the l2norm block
// leaves every output equal to the X1 chain's (gate -> fwd_intra2_split -> to_gn) on the raw q / k.
template <int GATE, bool VL, bool L2N = true>
__global__ void __launch_bounds__(256) g2_fwd_prep_a_kernel(PrepAParams p) {
    using namespace fi2;
    // One 21 KB arena (3 CTAs/CU), phases alias it (floats):
    //   l2n exchange X      [0, 512)                           before the gate
    //   sTot                [4352, 4864)                       gate scan only
    //   gate half tile G    [0, 4096)                          two rounds (rows 0-31, 32-63)
    //   REF slots 0-7       [4352, 5376)                       written in the rounds; slots 0-3 =
    //                       rows 16rb+8 (P1a), 4-6 = rows 16rb (P1b), 7 = row 63 (gn, Gn, clamp)
    //   T2 / R              [0, 4352) / [0, 5120)              as fwd_intra2a; R blocks 17-19 overlap
    //                       slots 0-5: a barrier precedes the P1b sp=1 RED (last REF read before it)
#if defined(__gfx938__)
    // kra P-938a16 (C4): a 16 KB arena -> 4 CTAs per CU on gfx938 (168 VGPRs still allow 4: resid_probe).  The gate
    // tile goes in 4 rounds of 16 rows (8 KB), the Kn transpose in two token halves (T2 [128][36] bf16), the R partial
    // blocks are flushed to global after P1a and after each P1b half; the REF slots stay at [3072, 4096) throughout.
    constexpr bool P16 = true;
#else
    constexpr bool P16 = false;
#endif
    constexpr int OFF_STOT = P16 ? 2048 : 4352, OFF_XQ = P16 ? 2560 : 4864, OFF_XK = P16 ? 2816 : 5120,
                  OFF_REF = P16 ? 3072 : 4352, GROWS = P16 ? 16 : 32, kT2h = 36;
    __shared__ __align__(16) float lds[P16 ? 4096 : 5376];
    float* const G = lds;
    float* const smem = lds;
    int i, hh;
    g2_cta_order(p.order, p.order_m, p.H, p.NT, i, hh);   // kra P-938o2
    const int wv = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int l = threadIdx.x & 63, m = l & 15, q4 = l >> 4;
    const long tok0 = g2_row_tok<VL>(p.ctok, i);
    const int cl = g2_row_len<VL>(p.clen, i);
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;
    #define PA_RED(B, A) { f4* d_ = reinterpret_cast<f4*>(smem + (B) * 256 + l * 4); \
                          *d_ = (s == 0) ? (A) : (*d_ + (A)); }
    const int cbase = 32 * wv + 8 * q4;
    const size_t cta0 = g2_rm_tok(tok0, hh, p.H, kK);
    const short* __restrict__ qc = p.q + cta0;
    const short* __restrict__ kc = p.k + cta0;
    const int rs = p.H * kK;
    // kra P-938sa: CTA-uniform bases + 32-bit byte offsets (saddr global ops instead of per-lane 64-bit address math)
    auto pa = [](auto* base, uint32_t e) {
        using E = std::remove_pointer_t<decltype(base)>;
        return reinterpret_cast<E*>(reinterpret_cast<std::conditional_t<std::is_const_v<E>, const char*, char*>>(base)
                                    + e * (uint32_t)sizeof(E));
    };

    // ---- raw gate loads first (latency overlaps the l2norm below)
    GRaw<GATE> grw;
    gate_load<GATE, VL>(p, grw, hh, tok0, cl);
    float2 graw[16];                                  // in-quarter prefix (gate_scan output)
    // ---- l2norm (q and k): raw loads in the fwd_intra2a mapping, l2n2's reduction tree
    v4bh qv[4][2], kv[4][2];
    #pragma unroll
    for (int rb = 0; rb < 4; ++rb) {
        const int o = (16 * rb + m) * rs + cbase;
        const bool rok = g2_row_ok<M>(16 * rb + m, cl);
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            qv[rb][h] = rok ? *reinterpret_cast<const v4bh*>(pa(qc, (uint32_t)(o + 4 * h))) : v4bh{0, 0, 0, 0};
            kv[rb][h] = rok ? *reinterpret_cast<const v4bh*>(pa(kc, (uint32_t)(o + 4 * h))) : v4bh{0, 0, 0, 0};
        }
    }
    {
        float* Xq = lds + OFF_XQ;                     // [64 rows][4 waves] (REF slots 4-7: free
        float* Xk = lds + OFF_XK;                     // until the round-0 barrier)
        if constexpr (L2N) {
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            float sq = 0.f, sk = 0.f;
            #pragma unroll
            for (int u = 0; u < 8; ++u) {             // l2n2: ss += f[u] * f[u], u = 0..7
                const float fq = bf2f(qv[rb][u >> 2][u & 3]);
                const float fk = bf2f(kv[rb][u >> 2][u & 3]);
                sq += fq * fq;
                sk += fk * fk;
            }
            sq += __shfl_xor(sq, 16, 64);             // butterfly step 1 (j ^ 1 = q4 ^ 1)
            sk += __shfl_xor(sk, 16, 64);
            sq += __shfl_xor(sq, 32, 64);             // step 2 (j ^ 2 = q4 ^ 2)
            sk += __shfl_xor(sk, 32, 64);
            if (q4 == 0) {
                Xq[(16 * rb + m) * 4 + wv] = sq;
                Xk[(16 * rb + m) * 4 + wv] = sk;
            }
        }
        }
        gate_scan<GATE, VL>(p, grw, graw, reinterpret_cast<float2*>(lds + OFF_STOT), hh, cl);
        __syncthreads();                              // X and sTot visible (one barrier for both)
        if constexpr (L2N) {
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            const int row = 16 * rb + m;
            const f4 yq = *reinterpret_cast<const f4*>(Xq + row * 4);
            const f4 yk = *reinterpret_cast<const f4*>(Xk + row * 4);
            const float ssq = (yq[0] + yq[1]) + (yq[2] + yq[3]);    // steps 4 and 8
            const float ssk = (yk[0] + yk[1]) + (yk[2] + yk[3]);
            const float rq = 1.f / sqrtf(ssq + 1e-6f);
            const float rk = 1.f / sqrtf(ssk + 1e-6f);
            if (wv == 0 && q4 == 0 && g2_row_ok<M>(row, cl)) {
                const size_t ri = g2_rm_tok(tok0 + row, hh, p.H, 1);
                p.q_rstd[ri] = rq;
                p.k_rstd[ri] = rk;
            }
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                qv[rb][u >> 2][u & 3] = bf_rne(bf2f(qv[rb][u >> 2][u & 3]) * rq);
                kv[rb][u >> 2][u & 3] = bf_rne(bf2f(kv[rb][u >> 2][u & 3]) * rk);
            }
            if (p.qn_out != nullptr && g2_row_ok<M>(row, cl)) {   // kra P-77b: qn / kn for the backward
                const size_t o2 = cta0 + (size_t)row * rs + cbase;
                *reinterpret_cast<v4bh*>(p.qn_out + o2) = qv[rb][0];
                *reinterpret_cast<v4bh*>(p.qn_out + o2 + 4) = qv[rb][1];
                *reinterpret_cast<v4bh*>(p.kn_out + o2) = kv[rb][0];
                *reinterpret_cast<v4bh*>(p.kn_out + o2 + 4) = kv[rb][1];
            }
        }
        }
    }

    if (p.stop == 1) return;
    // ---- gate: the scan ran above (one per lane, the gate kernel's arithmetic); the tile is
    // exchanged in two rounds of 32 rows (16 KB): waves 2h, 2h+1 write quarters 2h, 2h+1 (the
    // scale is applied here, as the gate kernel's store does) and every lane takes its rows /
    // reference slot.  sTot / X (in the REF area) are dead once every lane passed round 0's barrier.
    const float2 carry = gate_carry(reinterpret_cast<const float2*>(lds + OFF_STOT));
    f4 gv[4][2];
    float* const REF = lds + OFF_REF;
    const int rj = threadIdx.x >> 5, rc4 = 4 * (threadIdx.x & 31);
    const int rrow = g2_row_g<M>(rj < 4 ? 16 * rj + 8 : (rj < 7 ? 16 * (rj - 3) : 63), cl);
    #pragma unroll
    for (int hr = 0; hr < 64 / GROWS; ++hr) {
        if ((P16 ? wv : (wv >> 1)) == hr) {
            #pragma clang fp contract(off)
            const int c = threadIdx.x & 63;
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                const int r = 16 * wv + u;
                if (!VL || r < cl)
                    *reinterpret_cast<float2*>(G + gsw(r - GROWS * hr, 2 * c)) =
                        float2{(graw[u].x + carry.x) * p.gscale, (graw[u].y + carry.y) * p.gscale};
            }
        }
        __syncthreads();
        #pragma unroll
        for (int rb = (GROWS / 16) * hr; rb < (GROWS / 16) * (hr + 1); ++rb) {
            const int row = 16 * rb + m;
            if (g2_row_ok<M>(row, cl)) {
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                    gv[rb][h] = *reinterpret_cast<const f4*>(G + gsw(row - GROWS * hr, cbase + 4 * h));
            }
        }
        if ((rrow >> (P16 ? 4 : 5)) == hr) {          // (a clamped row of -1 matches no round, as before)
            const f4 refv = *reinterpret_cast<const f4*>(G + gsw(rrow - GROWS * hr, rc4));
            *reinterpret_cast<f4*>(REF + rj * 128 + rc4) = refv;
            if (rj == 7) {                            // Gn (B,H,NT,8,4,4): [a][c][d] = g[16a+4d+c]
                float* gnp = p.Gn + g2_nat_blk(i, hh, p.H, p.NT, false, 128);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int col = rc4 + e;
                    gnp[p.kp ? col : (col >> 4) * 16 + (col & 3) * 4 + ((col & 15) >> 2)] = exp2f(refv[e]);   // kra P-52, P-61e
                }
            }
        }
        __syncthreads();                              // round read: next round / T2 may overwrite
    }
    if constexpr (M == 2) {                           // rows past the end: the clamped row (slot 7)
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb)
            if (!g2_row_ok<M>(16 * rb + m, cl)) {
                gv[rb][0] = *reinterpret_cast<const f4*>(REF + 7 * 128 + cbase);
                gv[rb][1] = *reinterpret_cast<const f4*>(REF + 7 * 128 + cbase + 4);
            }
    }
    if (p.stop == 3) return;
    auto Q = [&](int rb, int s) { return bf2f(qv[rb][s >> 2][s & 3]); };
    auto K = [&](int rb, int s) { return bf2f(kv[rb][s >> 2][s & 3]); };
    auto G_ = [&](int rb, int s) { return gv[rb][s >> 2][s & 3]; };
    auto ldref = [&](int j, float* x) {               // slot j: rows 16rb+8 (0-3) | 16rb (4-6) | 63
        const f4 a = *reinterpret_cast<const f4*>(REF + j * 128 + cbase);
        const f4 b = *reinterpret_cast<const f4*>(REF + j * 128 + cbase + 4);
        #pragma unroll
        for (int e = 0; e < 4; ++e) { x[e] = a[e]; x[4 + e] = b[e]; }
    };
    const size_t nblk = g2_nat_blk(i, hh, p.H, p.NT, false, 8192);
    {   // ---- QgA + Kn (fwd_intra2a verbatim)
        short* T2 = reinterpret_cast<short*>(smem);
        float gn[8];
        ldref(7, gn);
        auto t2i = [&](int dim, int tok, int half) { return P16 ? dim * kT2h + tok - 32 * half : dim * kT2 + tok; };
        #pragma unroll
        for (int half = 0; half < (P16 ? 2 : 1); ++half) {
        #pragma unroll
        for (int rr = 0; rr < (P16 ? 2 : 4); ++rr) {
            const int rb = P16 ? 2 * half + rr : rr;
            v4bh qg[2], kb[2];
            const int row = 16 * rb + m;
            const bool rok = g2_row_ok<M>(row, cl);
            const float bta = rok ? *pa(p.beta + g2_rm_tok(tok0, hh, p.H, 1), (uint32_t)row * (uint32_t)p.H) : 0.f;
#if defined(__gfx938__) && (G2_PK2 & 2)
            // kra P-938pk2: column pairs (s, s + 1) as packed fp32, both bf16 roundings in one cvt_pk; same ops.  Only
            // without the in-kernel l2norm: there it lifts VGPR 153 -> 229 (138 -> 167 here, still 3 CTAs per CU).
            if constexpr (!L2N) {
                u2x qw[2], kw[2];
                #pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    const int s0 = 2 * kt, s1 = s0 + 1, h = kt >> 1, e = kt & 1;
                    const f2 eg = f2{__builtin_amdgcn_exp2f(G_(rb, s0)), __builtin_amdgcn_exp2f(G_(rb, s1))};
                    const f2 qf = bf2f2(__builtin_bit_cast(u2x, qv[rb][h])[e]);
                    const f2 kf = bf2f2(__builtin_bit_cast(u2x, kv[rb][h])[e]);
                    qw[h][e] = bf_rne2(qf * eg);
                    kw[h][e] = bf_rne2(kf * bta * eg);
                    const f2 d = pk_sub(f2{gn[s0], gn[s1]}, f2{G_(rb, s0), G_(rb, s1)});
                    const uint32_t tw = bf_rne2(kf * f2{__builtin_amdgcn_exp2f(d.x), __builtin_amdgcn_exp2f(d.y)});
                    T2[t2i(cbase + s0, 16 * rb + m, half)] = (short)tw;
                    T2[t2i(cbase + s1, 16 * rb + m, half)] = (short)(tw >> 16);
                }
                #pragma unroll
                for (int h = 0; h < 2; ++h) { qg[h] = __builtin_bit_cast(v4bh, qw[h]); kb[h] = __builtin_bit_cast(v4bh, kw[h]); }
            } else
#endif
            {
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                const float eg = __builtin_amdgcn_exp2f(G_(rb, s));    // shared by QgA and Kb (same argument)
                qg[s >> 2][s & 3] = bf_rne(Q(rb, s) * eg);
                // Kb = bf16(kn * beta * 2^g): fwd_intra2b's T2 pass-0 value, row-major like kn
                kb[s >> 2][s & 3] = bf_rne(K(rb, s) * bta * eg);
                T2[t2i(cbase + s, 16 * rb + m, half)] = bf_rne(K(rb, s) * __builtin_amdgcn_exp2f(gn[s] - G_(rb, s)));
            }
            }
            if (rok) {
                *reinterpret_cast<v4bh*>(pa(p.Kb + cta0, (uint32_t)(row * rs + cbase))) = kb[0];
                *reinterpret_cast<v4bh*>(pa(p.Kb + cta0, (uint32_t)(row * rs + cbase + 4))) = kb[1];
            }
            const size_t qb0 = nblk + (size_t)(2048 * rb + 256 * (cbase >> 4) + 4 * m
                               + 128 * ((cbase >> 3) & 1));
            *reinterpret_cast<v4bh*>(pa(p.QgA + nblk, (uint32_t)nat16(qb0 - nblk))) = qg[0];        // nblk % 512 == 0
            *reinterpret_cast<v4bh*>(pa(p.QgA + nblk, (uint32_t)nat16(qb0 - nblk + 64))) = qg[1];
        }
        lds_barrier();
        #pragma unroll
        for (int it = 0; it < 4; ++it) {                    // kra L16: 16B pair emit
            const int P = threadIdx.x + 256 * it;           // pair P = (group P>>6, lane P&63)
            if (P16 && ((P >> 6) & 1) != half) continue;    // P-938a16: tokens 32 half .. (wave-uniform: wv & 1)
            v4bh h[2];
            #pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int U = 128 * (P >> 6) + 64 * s + (P & 63);   // quad of sub-block 2g+s
                const int dim = 16 * (U >> 8) + (U & 15), tok = 16 * ((U >> 6) & 3) + 4 * ((U >> 4) & 3);
                h[s] = *reinterpret_cast<const v4bh*>(T2 + t2i(dim, tok, half));
            }
            st16x2(pa(p.Kn + nblk, 8u * (uint32_t)(p.kp ? ((P & ~15) | pi16(P & 15)) : P)), h[0], h[1]);   // kra P-61e
        }
        lds_barrier();
        }   // half
    }
    if (p.stop == 4) return;
    f4* const Rg4 = reinterpret_cast<f4*>(p.R + g2_nat_blk(i, hh, p.H, p.NT, false, 5120));
    {   // ---- P1a (verbatim; reference = REF slot rb)
        f4 acc[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) acc[b] = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            float gm[8];
            ldref(rb, gm);
            #pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                const int s0 = 2 * kt, s1 = 2 * kt + 1;
#if defined(__gfx938__)   // kra P-938ng: gm - G == -(G - gm) in IEEE (non-NaN); the negation folds into v_exp
#if G2_PK2 & 4
                const f2 d01 = pk_sub(f2{G_(rb, s0), G_(rb, s1)}, f2{gm[s0], gm[s1]});   // kra P-938pk2
                const float d0 = d01.x, d1 = d01.y;
#else
                const float d0 = G_(rb, s0) - gm[s0], d1 = G_(rb, s1) - gm[s1];
#endif
                const f2 eA = f2{exp2_inrange(d0), exp2_inrange(d1)};
                const f2 eB = f2{exp2_inrange(-d0), exp2_inrange(-d1)};
#else
                const f2 eA = f2{exp2_inrange(G_(rb, s0) - gm[s0]), exp2_inrange(G_(rb, s1) - gm[s1])};
                const f2 eB = f2{exp2_inrange(gm[s0] - G_(rb, s0)), exp2_inrange(gm[s1] - G_(rb, s1))};
#endif
                const f2 Bd = f2{K(rb, s0), K(rb, s1)} * eB;
                acc[rb] = mmac32(f2{Q(rb, s0), Q(rb, s1)} * eA, Bd, acc[rb]);
                acc[4 + rb] = mmac32(f2{K(rb, s0), K(rb, s1)} * eA, Bd, acc[4 + rb]);
            }
        }
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            switch ((wv + s) & 3) {
                case 0: PA_RED(0, acc[0]) PA_RED(1, acc[1]) break;
                case 1: PA_RED(2, acc[2]) PA_RED(3, acc[3]) break;
                case 2: PA_RED(4, acc[4]) PA_RED(5, acc[5]) break;
                default: PA_RED(6, acc[6]) PA_RED(7, acc[7]) break;
            }
            lds_barrier();
        }
        if constexpr (P16) {                          // P-938a16: R blocks 0-7 out, the arena reused by P1b
            const f4* s4 = reinterpret_cast<const f4*>(smem);
            #pragma unroll
            for (int j = threadIdx.x; j < 512; j += 256) Rg4[j] = s4[j];
            lds_barrier();
        }
    }
    if (p.stop == 5) return;
    {   // ---- P1b (verbatim; reference = REF slot 3 + rb)
        #pragma unroll
        for (int sp = 0; sp < 2; ++sp) {
            f4 aq[3], ak[3];
            #pragma unroll
            for (int b = 0; b < 3; ++b) { aq[b] = f4{0.f, 0.f, 0.f, 0.f}; ak[b] = aq[b]; }
            #pragma unroll
            for (int rb = 1; rb < 4; ++rb) {
                if ((sp == 0) != (rb < 3)) continue;
                float gr[8];
                ldref(3 + rb, gr);
                #pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    const int s0 = 2 * kt, s1 = 2 * kt + 1;
#if defined(__gfx938__) && (G2_PK2 & 4)   // kra P-938pk2: the exp2 arguments as packed pairs
                    const f2 dO = pk_sub(f2{G_(rb, s0), G_(rb, s1)}, f2{gr[s0], gr[s1]});
                    const f2 eO = f2{exp2_inrange(dO.x), exp2_inrange(dO.y)};
#else
                    const f2 eO = f2{exp2_inrange(G_(rb, s0) - gr[s0]), exp2_inrange(G_(rb, s1) - gr[s1])};
#endif
                    const f2 Aq = f2{Q(rb, s0), Q(rb, s1)} * eO;
                    const f2 Ak = f2{K(rb, s0), K(rb, s1)} * eO;
                    #pragma unroll
                    for (int cb = 0; cb < rb; ++cb) {
#if defined(__gfx938__) && (G2_PK2 & 4)
                        const f2 dBo = pk_sub(f2{gr[s0], gr[s1]}, f2{G_(cb, s0), G_(cb, s1)});
                        const f2 eBo = f2{__builtin_amdgcn_exp2f(dBo.x), __builtin_amdgcn_exp2f(dBo.y)};
#else
                        const f2 eBo = f2{__builtin_amdgcn_exp2f(gr[s0] - G_(cb, s0)), __builtin_amdgcn_exp2f(gr[s1] - G_(cb, s1))};
#endif
                        const f2 Bo = f2{K(cb, s0), K(cb, s1)} * eBo;
                        const int x = pidx(rb, cb) - 3 * sp;
                        aq[x] = mmac32(Aq, Bo, aq[x]);
                        ak[x] = mmac32(Ak, Bo, ak[x]);
                    }
                }
            }
            const int o = 3 * sp;
            if (!P16 && sp == 1) lds_barrier();       // sp=1 RED (blocks 17-19) overwrites REF
            // P-938a16: blocks 8+o..10+o / 14+o..16+o in arena slots 0-2 / 3-5
            #define PBK(B) (P16 ? ((B) < 14 ? (B) - 8 - o : (B) - 11 - o) : (B))
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                switch ((wv + s) & 3) {
                    case 0: PA_RED(PBK(8 + o), aq[0]) PA_RED(PBK(14 + o), ak[0]) break;
                    case 1: PA_RED(PBK(9 + o), aq[1]) PA_RED(PBK(15 + o), ak[1]) break;
                    case 2: PA_RED(PBK(10 + o), aq[2]) break;
                    default: PA_RED(PBK(16 + o), ak[2]) break;
                }
                lds_barrier();
            }
            #undef PBK
            if constexpr (P16) {                      // P-938a16: the half's 6 blocks out
                const f4* s4 = reinterpret_cast<const f4*>(smem);
                for (int j = threadIdx.x; j < 384; j += 256) {
                    const int slot = j >> 6;
                    Rg4[(slot < 3 ? 8 + o + slot : 11 + o + slot) * 64 + (j & 63)] = s4[j];
                }
                if (sp == 0) lds_barrier();           // flush reads before the next half's RED writes
            }
        }
    }
    if (p.stop == 6) return;
    #undef PA_RED
    if constexpr (!P16) {
        const f4* s4 = reinterpret_cast<const f4*>(smem);
        #pragma unroll
        for (int j = threadIdx.x; j < 1280; j += 256) Rg4[j] = s4[j];
    }
    });   // g2_vl_dispatch
}

#ifndef KDA_P938B2
#define KDA_P938B2 1
#endif
// kra P-938w: prep_b's T2 index.  gfx938: unpadded [128][64] (the union fits 16 KB -> 4 CTAs per CU) with the
// token's bits 2..5 XORed by the dim's low 4 bits instead of the +4 padding (4-token vectors stay contiguous, the
// 16-dim operand reads stay conflict-free).  Layout only.
__device__ __forceinline__ int prepb_t2i(int dim, int tok) {
#if defined(__gfx938__)
    return dim * 64 + (tok ^ ((dim & 15) << 2));
#else
    return dim * fi2::kT2 + tok;
#endif
}

// fis::s_post verbatim except the T2 pass-0 operand: Kb (already bf16(kn * beta * 2^g)).
template <int I, int M>
__device__ __forceinline__ void b_post(const PrepBParams& p, float* smem, const float* Rg,
                                       int i, int hh, int l, long t0, int cl) {
    using namespace fi2;
    const int m = l & 15, q4 = l >> 4, pm = pi16(m);
    const long tok0 = t0;
    const size_t nblk = g2_nat_blk(i, hh, p.H, p.NT, false, 8192);
    const size_t nblk64 = g2_nat_blk(i, hh, p.H, p.NT, false, 4096);
    constexpr int NA = I > 0 ? I : 1;
    auto R4 = [&](int b) {
        const float* r = Rg + b * 256 + 4 * m + q4;
        return f4{r[0], r[64], r[128], r[192]};
    };

#if G2_MLS938
    // kra P-938b (gfx938): R blocks I..I+1 and 4+I..5+I (2 KB contiguous each) by MLS into a wave-private 4 KB LDS
    // slot (smem is free until the first barrier; 4 slots = 16 KB) -- 2 tile loads instead of 8 x 4 B gathers per lane
    float* rl = smem + I * 1024;
    g2_mls_tile(Rg + I * 256, 32, rl);
    g2_mls_tile(Rg + (4 + I) * 256, 32, rl + 512);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    auto R4l = [&](int slot) { const float* r = rl + slot * 256 + 4 * m + q4; return f4{r[0], r[64], r[128], r[192]}; };
    f4 Aqkd = R4l(0), Ld = R4l(2);
    f4 Aqko[NA], Akko[NA];
#if KDA_P938B2
    // P-938b2: the wave's off-diagonal runs (I blocks from 8 + pidx(I, 0), I blocks from 14 + pidx(I, 0)) in two more
    // rounds through the same slot, 2 KB tiles aligned to even blocks (all inside the CTA's 20 blocks)
    auto run = [&](f4* dst, int b0) {
        const int bb = b0 & ~1;
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");    // previous LDS reads of the slot have returned
        #pragma unroll
        for (int t = 0; t < 2; ++t) if (bb + 2 * t <= b0 + I - 1) g2_mls_tile(Rg + (bb + 2 * t) * 256, 32, rl + 512 * t);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        #pragma unroll
        for (int cb = 0; cb < I; ++cb) dst[cb] = R4l(b0 + cb - bb);
    };
    if constexpr (I > 0) { run(Aqko, 8 + pidx(I, 0)); run(Akko, 14 + pidx(I, 0)); }
#else
    #pragma unroll
    for (int cb = 0; cb < I; ++cb) {
        Aqko[cb] = R4(8 + pidx(I, cb));
        Akko[cb] = R4(14 + pidx(I, cb));
    }
#endif
#else
    f4 Aqkd = R4(I), Ld = R4(4 + I);
    f4 Aqko[NA], Akko[NA];
    #pragma unroll
    for (int cb = 0; cb < I; ++cb) {
        Aqko[cb] = R4(8 + pidx(I, cb));
        Akko[cb] = R4(14 + pidx(I, cb));
    }
#endif
    lds_barrier();
#if defined(__gfx938__)
    if (p.stop == 1) return;   // G2_PREPB_STOP (dev, gfx938 build only)
#endif

    const int t = 16 * I + m;
    const bool tok_ = g2_row_ok<M>(t, cl);
    const float bm = tok_ ? p.beta[g2_rm_tok(tok0 + t, hh, p.H, 1)] : 0.f;
    {
        short* rowp = p.Aqk + g2_rm_tok(tok0 + t, hh, p.H, kBT);
        #pragma unroll
        for (int cb = 0; cb < 4; ++cb) {
            f4 x = f4{0.f, 0.f, 0.f, 0.f};
            if (cb < I) {
                x = Aqko[cb < NA ? cb : 0] * p.scale;
            } else if (cb == I) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) x[e] = (4 * q4 + e <= m) ? Aqkd[e] * p.scale : 0.f;
            }
            const v4bh b = pack4(x);
            if (tok_) *reinterpret_cast<v4bh*>(rowp + 16 * cb + 4 * q4) = b;
            *reinterpret_cast<v4bh*>(p.AqkA + nat16(nblk64 + (size_t)(cb + 4 * I) * 256 + 4 * l)) = b;
        }
    }
#if defined(__gfx938__)
    if (p.stop == 2) return;
#endif
    f4 Ai[4];
    {
        f4 raw;
        #pragma unroll
        for (int e = 0; e < 4; ++e) raw[e] = (4 * q4 + e < m) ? -Ld[e] * bm : 0.f;
        f4 row = raw;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            f4 rj;
            #pragma unroll
            for (int e = 0; e < 4; ++e) rj[e] = __shfl(row[e], 16 * q4 + j, 64);
            const float aj = __shfl(raw[j & 3], 16 * (j >> 2) + m, 64);
            #pragma unroll
            for (int e = 0; e < 4; ++e) row[e] = (m > j) ? fmaf(aj, rj[e], row[e]) : row[e];
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) if (4 * q4 + e == m) row[e] += 1.f;
        Ai[I] = row;
    }
    #pragma unroll
    for (int cb = 0; cb < I; ++cb) Akko[cb] = Akko[cb] * bm;

    float* D = smem;
    float* P = smem + 4 * kTile;
    float* S = smem + (7 + I) * kTile;
    auto put = [&](float* tile, f4 x) {
        *reinterpret_cast<f4*>(tile + m * kTS + 4 * q4) = x;
    };
    auto stage = [&](f4 x) -> const float* {
        wave_lds_order();
        put(S, x);
        wave_lds_order();
        return S;
    };
    put(D + I * kTile, Ai[I]);
    lds_barrier();
    f4 P20 = {0.f, 0.f, 0.f, 0.f}, P30 = P20, P31 = P20;
    if constexpr (I == 1) {
        Ai[0] = -mmB(Ai[1], stage(mmB(Akko[0], D, q4, pm)), q4, pm);
        put(P, Ai[0]);
    } else if constexpr (I == 2) {
        Ai[1] = -mmB(Ai[2], stage(mmB(Akko[1], D + kTile, q4, pm)), q4, pm);
        put(P + kTile, Ai[1]);
        P20 = mmB(Akko[0], D, q4, pm);
    } else if constexpr (I == 3) {
        Ai[2] = -mmB(Ai[3], stage(mmB(Akko[2], D + 2 * kTile, q4, pm)), q4, pm);
        P30 = mmB(Akko[0], D, q4, pm);
        P31 = mmB(Akko[1], D + kTile, q4, pm);
    }
    lds_barrier();
    if constexpr (I == 2) {
        const f4 s = P20 + mmB(Akko[1], P, q4, pm);
        Ai[0] = -mmB(Ai[2], stage(s), q4, pm);
        put(P + 2 * kTile, Ai[0]);
    } else if constexpr (I == 3) {
        const f4 s = P31 + mmB(Akko[2], P + kTile, q4, pm);
        Ai[1] = -mmB(Ai[3], stage(s), q4, pm);
        P30 = P30 + mmB(Akko[1], P, q4, pm);
    }
    lds_barrier();
    if constexpr (I == 3) {
        const f4 s = P30 + mmB(Akko[2], P + 2 * kTile, q4, pm);
        Ai[0] = -mmB(Ai[3], stage(s), q4, pm);
    }
    {
        short* rowp = p.Akk + g2_rm_tok(tok0 + t, hh, p.H, kBT);
        if (tok_) {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                *reinterpret_cast<v4bh*>(rowp + 16 * j + 4 * q4) =
                    pack4(j <= I ? Ai[j] : f4{0.f, 0.f, 0.f, 0.f});
        }
    }

#if defined(__gfx938__)
    if (p.stop == 3) return;
#endif
    short* T2 = reinterpret_cast<short*>(smem);
    lds_barrier();
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
#if defined(__gfx938__)
        if (pass == 1 && p.stop == 4) return;
#endif
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int tid = 64 * I + l;
            const int row = (tid >> 2) & 63, cq = 8 * (tid & 3) + 32 * it;
            const bool rok = g2_row_ok<M>(row, cl);
            const size_t o = g2_rm_tok(tok0 + row, hh, p.H, kK) + cq;
            const v4bh z4 = v4bh{0, 0, 0, 0};
            if (pass == 0) {
                const v4bh k0 = rok ? *reinterpret_cast<const v4bh*>(p.Kb + o) : z4;
                const v4bh k1 = rok ? *reinterpret_cast<const v4bh*>(p.Kb + o + 4) : z4;
                #pragma unroll
                for (int d = 0; d < 8; ++d)
                    T2[prepb_t2i(cq + d, row)] = d < 4 ? k0[d & 3] : k1[d & 3];
            } else {
                const float b = rok ? p.beta[g2_rm_tok(tok0 + row, hh, p.H, 1)] : 0.f;
                const v4bh v0 = rok ? *reinterpret_cast<const v4bh*>(p.v + o) : z4;
                const v4bh v1 = rok ? *reinterpret_cast<const v4bh*>(p.v + o + 4) : z4;
                #pragma unroll
                for (int d = 0; d < 8; ++d)
                    T2[prepb_t2i(cq + d, row)] = bf_rne(bf2f(d < 4 ? v0[d & 3] : v1[d & 3]) * b);
            }
        }
        lds_barrier();
        f4 acc[8];
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) acc[vt] = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int j = 0; j <= I; ++j) {
            const v4bh a = pack4(Ai[j]);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {
                const int n = (pass == 0 || p.kp) ? pm : m;   // kra P-62
                acc[vt] = mmac(a, *reinterpret_cast<const v4bh*>(T2 + prepb_t2i(16 * vt + n, 16 * j + 4 * q4)), acc[vt]);
            }
        }
        short* out = pass == 0 ? p.Wn : p.Un;
        #pragma unroll
        for (int j = 0; j < 4; ++j)                         // kra L16: vt pair = one 16B store
            st16x2(out + nblk + (size_t)(2 * j + 8 * I) * 256 + 8 * l, pack4(acc[2 * j]), pack4(acc[2 * j + 1]));
        lds_barrier();
    }
}

template <bool VL>
__global__ void __launch_bounds__(256) g2_fwd_prep_b_kernel(PrepBParams p) {
#if defined(__gfx938__)
    __shared__ __align__(16) float smem[4096];       // kra P-938w: 16 KB (4 CTAs per CU), T2 swizzled
#else
    __shared__ __align__(16) float smem[4352];
#endif
    int i, hh;
    g2_cta_order(p.order, p.order_m, p.H, p.NT, i, hh);   // kra P-938o2
    const int wv = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int l = threadIdx.x & 63;
    const long tok0 = g2_row_tok<VL>(p.ctok, i);
    const int cl = g2_row_len<VL>(p.clen, i);
    const float* Rg = p.R + g2_nat_blk(i, hh, p.H, p.NT, false, 5120);
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;
    switch (wv) {
        case 0: b_post<0, M>(p, smem, Rg, i, hh, l, tok0, cl); break;
        case 1: b_post<1, M>(p, smem, Rg, i, hh, l, tok0, cl); break;
        case 2: b_post<2, M>(p, smem, Rg, i, hh, l, tok0, cl); break;
        default: b_post<3, M>(p, smem, Rg, i, hh, l, tok0, cl); break;
    }
    });
}

}  // namespace r4
}  // namespace g2

// q/k/v [1, T, H, 128] bf16 (packed rows with ctok/clen), g_raw bf16 (gate=1) or fp32 (gate=0),
// beta fp32 [1, T, H]; T = 64*NT dense.  Outputs as fwd_intra2_split + rstd + Gn; R / Kb scratch.
// kra P-94: l2n = 0 skips the in-kernel l2norm (q / k already normalized; q_rstd / k_rstd /
// qn_out / kn_out unused) -- the forward of use_qk_l2norm_in_kernel=False without the X1 chain.
extern "C" int g2_fwd_prep_l2n(const void* q, const void* k, const void* v, const void* g_raw,
                               const void* A_log, const void* dt_bias, const void* beta,
                               void* Wn, void* Kn, void* Un, void* QgA, void* AqkA, void* Aqk,
                               void* Akk, void* q_rstd, void* k_rstd, void* Gn, void* R, void* Kb,
                               double lower_bound, double gscale, int gate, double scale, int H,
                               int NT, const int* ctok, const int* clen, void* qn_out, void* kn_out,
                               int l2n, hipStream_t s) {
    if ((ctok == nullptr) != (clen == nullptr) || NT <= 0) return -1;
    if (l2n && (q_rstd == nullptr || k_rstd == nullptr)) return -1;
    g2::r4::PrepAParams pa{(const short*)q, (const short*)k, (const short*)v, (short*)Wn,
                           (short*)Un, (short*)AqkA, (short*)Aqk, (short*)Akk, (float)scale,
                           g_raw, (const float*)A_log, (const float*)dt_bias, (const float*)beta,
                           (short*)QgA, (short*)Kn, (short*)Kb, (float*)R, (float*)q_rstd,
                           (float*)k_rstd, (float*)Gn, ctok, clen, (float)lower_bound,
                           (float)gscale, H, NT, 0};
    if (const char* e = g2::g2_getenv("G2_PREP_STOP")) pa.stop = atoi(e);
    pa.kp = g2::g2_kp_layout(H);   // kra P-61e
    pa.qn_out = (short*)qn_out; pa.kn_out = (short*)kn_out;    // kra P-77b
    // kra P-938o2 (gfx938, measured over T = 2K / 8K / 32K x H = 8-64): diagonal order at H % 64 == 0 (the chunk row
    // stride H * 256 B and the head-major natives both advance by powers of two in lock-step otherwise)
    const bool o938 = g2::g2_cur_gfx938();
    pa.order = (int)g2::g2_env_long("KDA_G2_PREPA_ORDER", o938 && H % 64 == 0 ? 2 : 0);
    pa.order_m = (int)g2::g2_env_long("KDA_G2_PREPA_ORDER_M", 4);
    if (pa.order == 3 && (pa.order_m < 1 || H % pa.order_m)) pa.order = 0;
    const dim3 grid(NT, H), blk(256);
#define G2_PREP_A(GT, VLT, L2) hipLaunchKernelGGL((g2::r4::g2_fwd_prep_a_kernel<GT, VLT, L2>), grid, blk, 0, s, pa)
    if (l2n) {
        if (gate) { if (ctok) G2_PREP_A(1, true, true); else G2_PREP_A(1, false, true); }
        else      { if (ctok) G2_PREP_A(0, true, true); else G2_PREP_A(0, false, true); }
    } else {
        pa.qn_out = pa.kn_out = nullptr;
        if (gate) { if (ctok) G2_PREP_A(1, true, false); else G2_PREP_A(1, false, false); }
        else      { if (ctok) G2_PREP_A(0, true, false); else G2_PREP_A(0, false, false); }
    }
#undef G2_PREP_A
    int rc = (int)hipGetLastError();
    if (rc) return rc;
    g2::r4::PrepBParams pb{(const short*)Kb, (const short*)v, (const float*)beta, (const float*)R,
                           (short*)Wn, (short*)Un, (short*)AqkA, (short*)Aqk, (short*)Akk,
                           ctok, clen, (float)scale, H, NT};
    pb.kp = pa.kp;   // kra P-62
    if (const char* e = g2::g2_getenv("G2_PREPB_STOP")) pb.stop = atoi(e);
    pb.order = (int)g2::g2_env_long("KDA_G2_PREPB_ORDER", o938 && H % 64 == 0 ? 2 : 0);   // kra P-938o2
    pb.order_m = (int)g2::g2_env_long("KDA_G2_PREPB_ORDER_M", 4);
    if (pb.order == 3 && (pb.order_m < 1 || H % pb.order_m)) pb.order = 0;
    if (ctok) hipLaunchKernelGGL(g2::r4::g2_fwd_prep_b_kernel<true>, grid, blk, 0, s, pb);
    else hipLaunchKernelGGL(g2::r4::g2_fwd_prep_b_kernel<false>, grid, blk, 0, s, pb);
    return (int)hipGetLastError();
}

extern "C" int g2_fwd_prep(const void* q, const void* k, const void* v, const void* g_raw,
                           const void* A_log, const void* dt_bias, const void* beta,
                           void* Wn, void* Kn, void* Un, void* QgA, void* AqkA, void* Aqk,
                           void* Akk, void* q_rstd, void* k_rstd, void* Gn, void* R, void* Kb,
                           double lower_bound, double gscale, int gate, double scale, int H,
                           int NT, const int* ctok, const int* clen, void* qn_out, void* kn_out,
                           hipStream_t s) {   // kra P-77b: qn_out / kn_out
    return g2_fwd_prep_l2n(q, k, v, g_raw, A_log, dt_bias, beta, Wn, Kn, Un, QgA, AqkA, Aqk, Akk, q_rstd,
                           k_rstd, Gn, R, Kb, lower_bound, gscale, gate, scale, H, NT, ctok, clen, qn_out,
                           kn_out, 1, s);
}
