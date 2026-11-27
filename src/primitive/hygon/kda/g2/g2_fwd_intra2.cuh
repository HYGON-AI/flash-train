// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// S3-16 (Claude): HIP fwd_intra v2 -- same math/outputs as g2_fwd_intra.cuh (v1),
// restructured for throughput.  v1 measured 9.45 ms @8K/48 (FLA chain 3.74): every
// MMAC step re-read g from global (6 scalar loads + 4 exp2 per k-tile), the four
// waves were unbalanced (wave3 8 pair blocks, wave0 2), LDS 54 KB => 1 CTA/CU,
// and all emits were 2-byte scattered scalars.
//
// v2 structure (one CTA = one (chunk, head), 4 waves):
//  P1  K-split products: wave w owns K columns [32w, 32w+32).  Every lane loads
//      its own q/k/g elements once, forms the gate-scaled operands in registers
//      (FLA safe-gate references: diag = sub-block middle row, off = row block's
//      first row) and accumulates partial sums for ALL 20 16x16 blocks
//      (4 Aqk diag, 4 L diag, 6 Aqk off, 6 Akk off) with fp32 MMAC.
//      The B operand's key rows are pi16-permuted, so every C fragment holds 4
//      CONTIGUOUS columns ("C-pi" layout: lane (m,q4) element e = col 4*q4+e).
//  RED deterministic cross-wave reduction through LDS (rotating groups, fixed
//      per-block order), then wave I owns row block I.
//  P2  per wave: Aqk emit (row-major + AqkA, 8-byte stores straight from C-pi),
//      16x16 diag forward substitution in registers (__shfl, no LDS),
//      layered merge (3 levels, B operands through LDS tiles), Akk emit.
//  P3  w/u: T2 = bf16(k*beta*2^g) / bf16(v*beta) staged [dim][token]; the A
//      operand is the wave's Ai row straight from registers (C-pi == bf16 A
//      fragment); Wn (A form) via pi-permuted B, Un (C form) via plain B.
//      QgA / Kn are emitted from the staging pass.
// LDS 20 KB (R / merge tiles / T2 alias).  Numerics: fp32 MMAC for A and the
// inverse, bf16 MMAC for w/u on bf16(Ai) -- same operand values as v1/FLA; only
// the fp32 summation association differs (S2 adjudication, not bitwise).
#include "g2_common.h"

namespace g2 {

#ifndef G2_F32_MMAC32_DEFINED
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

namespace fi2 {

constexpr int kTS = 20;             // merge tile row stride (floats, 16B-aligned rows)
constexpr int kTile = 16 * kTS;
constexpr int kT2 = 68;             // T2 [dim][token] row stride (bf16)

// Wave-scope LDS ordering.  LDS ops of one wave execute in order; this only has
// to stop the compiler from moving LDS accesses across it and drain lgkmcnt.
// (NO s_barrier: it is called from wave-divergent code.)
__device__ __forceinline__ void wave_lds_order() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}
__device__ __forceinline__ v4bh pack4(f4 x) {
    v4bh r;
    r[0] = bf_rne(x[0]); r[1] = bf_rne(x[1]); r[2] = bf_rne(x[2]); r[3] = bf_rne(x[3]);
    return r;
}
__device__ __forceinline__ constexpr int pidx(int rb, int cb) { return rb * (rb - 1) / 2 + cb; }
__device__ __forceinline__ float lo_bf(uint32_t w) { return bf2f((short)(w & 0xffffu)); }
__device__ __forceinline__ float hi_bf(uint32_t w) { return bf2f((short)(w >> 16)); }

// C-pi (A) x LDS tile (B, row-major stride kTS) -> C-pi.  k-slot bijection:
// tile kt, lane q4, element i  <->  physical index 4*q4 + 2*kt + i (A col / B row).
__device__ __forceinline__ f4 mmB(f4 a, const float* Bt, int q4, int pm) {
    f4 c = f4{0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const f2 a2 = f2{a[2 * kt], a[2 * kt + 1]};
        const f2 b2 = f2{Bt[(4 * q4 + 2 * kt) * kTS + pm], Bt[(4 * q4 + 2 * kt + 1) * kTS + pm]};
        c = mmac32(a2, b2, c);
    }
    return c;
}

// VL (varlen phase 2): t0/cl = chunk's first packed token / valid rows.  q/k/v/beta rows
// r >= cl read as 0, g rows read from the clamped row (address only), Aqk/Akk rows r >= cl
// are not stored.  VL=false: t0 = 64i, cl = 64 -> the dense code.
template <int I, int M>
__device__ __forceinline__ void post(const FwdIParams& p, float* smem, int i, int hh, int l,
                                     long t0, int cl) {
    const int m = l & 15, q4 = l >> 4, pm = pi16(m), tid = 64 * I + l;
    const long tok0 = t0;
    const size_t nblk = g2_nat_blk(i, hh, p.H, p.NT, false, 8192);
    const size_t nblk64 = g2_nat_blk(i, hh, p.H, p.NT, false, 4096);
    constexpr int NA = I > 0 ? I : 1;
    auto R4 = [&](int b) {                          // C layout -> C-pi (transposed read)
        const float* r = smem + b * 256 + 4 * m + q4;
        return f4{r[0], r[64], r[128], r[192]};
    };

    f4 Aqkd = R4(I), Ld = R4(4 + I);
    f4 Aqko[NA], Akko[NA];
    #pragma unroll
    for (int cb = 0; cb < I; ++cb) {
        Aqko[cb] = R4(8 + pidx(I, cb));
        Akko[cb] = R4(14 + pidx(I, cb));
    }
    lds_barrier();                                   // B1: R consumed (tiles alias it)

    const int t = 16 * I + m;
    const bool tok_ = g2_row_ok<M>(t, cl);
    const float bm = tok_ ? p.beta[g2_rm_tok(tok0 + t, hh, p.H, 1)] : 0.f;
    {   // ---- Aqk: scale, causal mask, emit row-major + AqkA (upper blocks exact 0)
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
    f4 Ai[4];
    {   // ---- diag inverse (I + L)^-1: raw = -L (strict lower, beta row scale),
        // row_m += raw[m][j] * row_j for j < m (row_j final), then + I.
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

    float* D = smem;                                 // D[j] = diag Ai_jj, j=0..3
    float* P = smem + 4 * kTile;                     // P0 = Ai10, P1 = Ai21, P2 = Ai20
    float* S = smem + (7 + I) * kTile;               // wave scratch
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
    lds_barrier();                                   // B2: diag inverses visible
    f4 P20 = {0.f, 0.f, 0.f, 0.f}, P30 = P20, P31 = P20;
    if constexpr (I == 1) {
        Ai[0] = -mmB(Ai[1], stage(mmB(Akko[0], D, q4, pm)), q4, pm);          // Ai10
        put(P, Ai[0]);
    } else if constexpr (I == 2) {
        Ai[1] = -mmB(Ai[2], stage(mmB(Akko[1], D + kTile, q4, pm)), q4, pm);  // Ai21
        put(P + kTile, Ai[1]);
        P20 = mmB(Akko[0], D, q4, pm);                                        // A20 Ai00
    } else if constexpr (I == 3) {
        Ai[2] = -mmB(Ai[3], stage(mmB(Akko[2], D + 2 * kTile, q4, pm)), q4, pm);  // Ai32
        P30 = mmB(Akko[0], D, q4, pm);                                        // A30 Ai00
        P31 = mmB(Akko[1], D + kTile, q4, pm);                                // A31 Ai11
    }
    lds_barrier();                                   // B3: Ai10, Ai21 visible
    if constexpr (I == 2) {
        const f4 s = P20 + mmB(Akko[1], P, q4, pm);                           // + A21 Ai10
        Ai[0] = -mmB(Ai[2], stage(s), q4, pm);                                // Ai20
        put(P + 2 * kTile, Ai[0]);
    } else if constexpr (I == 3) {
        const f4 s = P31 + mmB(Akko[2], P + kTile, q4, pm);                   // + A32 Ai21
        Ai[1] = -mmB(Ai[3], stage(s), q4, pm);                                // Ai31
        P30 = P30 + mmB(Akko[1], P, q4, pm);                                  // + A31 Ai10
    }
    lds_barrier();                                   // B4: Ai20 visible
    if constexpr (I == 3) {
        const f4 s = P30 + mmB(Akko[2], P + 2 * kTile, q4, pm);               // + A32 Ai20
        Ai[0] = -mmB(Ai[3], stage(s), q4, pm);                                // Ai30
    }
    {   // ---- Akk (= Ai) emit, row-major, upper blocks exact 0
        short* rowp = p.Akk + g2_rm_tok(tok0 + t, hh, p.H, kBT);
        if (tok_) {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                *reinterpret_cast<v4bh*>(rowp + 16 * j + 4 * q4) =
                    pack4(j <= I ? Ai[j] : f4{0.f, 0.f, 0.f, 0.f});
        }
    }
    if (p.stop == 2) return;

    short* T2 = reinterpret_cast<short*>(smem);
    lds_barrier();                                   // B5: merge tiles consumed
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        #pragma unroll
        for (int it = 0; it < 4; ++it) {             // 4 lanes read 64 contiguous bytes of a row
            const int row = (tid >> 2) & 63, cq = 8 * (tid & 3) + 32 * it;
            const bool rok = g2_row_ok<M>(row, cl);
            const size_t o = g2_rm_tok(tok0 + row, hh, p.H, kK) + cq;
            const size_t og = g2_rm_tok(tok0 + g2_row_g<M>(row, cl), hh, p.H, kK) + cq;
            const float b = rok ? p.beta[g2_rm_tok(tok0 + row, hh, p.H, 1)] : 0.f;
            const v4bh z4 = v4bh{0, 0, 0, 0};
            if (pass == 0) {
                const v4bh k0 = rok ? *reinterpret_cast<const v4bh*>(p.kn + o) : z4;
                const v4bh k1 = rok ? *reinterpret_cast<const v4bh*>(p.kn + o + 4) : z4;
                const f4 g0 = *reinterpret_cast<const f4*>(p.g + og);
                const f4 g1 = *reinterpret_cast<const f4*>(p.g + og + 4);
                #pragma unroll
                for (int d = 0; d < 8; ++d)
                    T2[(cq + d) * kT2 + row] = bf_rne(bf2f(d < 4 ? k0[d & 3] : k1[d & 3]) * b
                                                      * exp2f(d < 4 ? g0[d & 3] : g1[d & 3]));
            } else {
                const v4bh v0 = rok ? *reinterpret_cast<const v4bh*>(p.v + o) : z4;
                const v4bh v1 = rok ? *reinterpret_cast<const v4bh*>(p.v + o + 4) : z4;
                #pragma unroll
                for (int d = 0; d < 8; ++d)
                    T2[(cq + d) * kT2 + row] = bf_rne(bf2f(d < 4 ? v0[d & 3] : v1[d & 3]) * b);
            }
        }
        lds_barrier();                               // B6 / B8: T2 staged
        f4 acc[8];
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) acc[vt] = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int j = 0; j <= I; ++j) {
            const v4bh a = pack4(Ai[j]);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {
                const int n = pass == 0 ? pm : m;    // w: pi (A form out), u: plain (C form)
                acc[vt] = mmac(a, *reinterpret_cast<const v4bh*>(T2 + (16 * vt + n) * kT2
                                                               + 16 * j + 4 * q4), acc[vt]);
            }
        }
        short* out = pass == 0 ? p.Wn : p.Un;
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt)
            *reinterpret_cast<v4bh*>(out + nat16(nblk + (size_t)(vt + 8 * I) * 256 + 4 * l)) = pack4(acc[vt]);
        if (pass == 0 && p.stop == 3) return;
        lds_barrier();                               // B7 / B9: T2 consumed
    }
}

}  // namespace fi2

template <bool VL>
__global__ void __launch_bounds__(256, 2) g2_fwd_intra2_kernel(FwdIParams p) {
    using namespace fi2;
    __shared__ __align__(16) float smem[5120];       // 20 KB: R | merge tiles | T2
    const int i = blockIdx.x, hh = blockIdx.y;
    const int wv = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int l = threadIdx.x & 63, m = l & 15, q4 = l >> 4;
    const long tok0 = g2_row_tok<VL>(p.ctok, i);
        const int cl = g2_row_len<VL>(p.clen, i);
    // varlen: ONE uniform branch per CTA -- full chunks run the unmasked body (M=1)
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;

        // ---- P1: K-split partial products for all 20 blocks.
        // Lane (m, q4) of wave w owns rows 16rb+m (rb = 0..3) x the 8 CONTIGUOUS
        // columns c = 32w + 8*q4 + s; fp32 MMAC tile kt uses s = 2*kt + i for BOTH
        // operands, and the B operand's key rows are the lane's own rows, so q/k/g are
        // loaded ONCE per CTA into registers and reused by every pass (v2a re-read g
        // four times from DRAM).  C fragments come out in the standard layout; owners
        // read the reduction buffer transposed into C-pi (see post::R4).
        // RED: rotating deterministic reduction.  Step s: wave w adds its partials for
        // group (w+s)&3; group G therefore always sums waves G, G-1, G-2, G-3.
        #define FI2_RED(B, A) { f4* d_ = reinterpret_cast<f4*>(smem + (B) * 256 + l * 4); \
                                *d_ = (s == 0) ? (A) : (*d_ + (A)); }
        const int cbase = 32 * wv + 8 * q4;
        // Chunk-local 32-bit offsets from the g2_rm_tok chunk base (row stride H*K):
        // keeps the per-lane address math in one VGPR instead of 64-bit pairs.
        const size_t cta0 = g2_rm_tok(tok0, hh, p.H, kK);
        const short* __restrict__ qc = p.qn + cta0;
        const short* __restrict__ kc = p.kn + cta0;
        const float* __restrict__ gc = p.g + cta0;
        const int rs = p.H * kK;
        v4bh qv[4][2], kv[4][2];
        f4 gv[4][2];
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            const int o = (16 * rb + m) * rs + cbase;
            const int og = g2_row_g<M>(16 * rb + m, cl) * rs + cbase;
            const bool rok = g2_row_ok<M>(16 * rb + m, cl);
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                qv[rb][h] = rok ? *reinterpret_cast<const v4bh*>(qc + o + 4 * h) : v4bh{0, 0, 0, 0};
                kv[rb][h] = rok ? *reinterpret_cast<const v4bh*>(kc + o + 4 * h) : v4bh{0, 0, 0, 0};
                gv[rb][h] = *reinterpret_cast<const f4*>(gc + og + 4 * h);
            }
        }
        auto Q = [&](int rb, int s) { return bf2f(qv[rb][s >> 2][s & 3]); };
        auto K = [&](int rb, int s) { return bf2f(kv[rb][s >> 2][s & 3]); };
        auto G = [&](int rb, int s) { return gv[rb][s >> 2][s & 3]; };
        auto ldref = [&](int row, float* x) {           // 8 g values of a reference row
            const int o = g2_row_g<M>(row, cl) * rs + cbase;
            const f4 a = *reinterpret_cast<const f4*>(gc + o);
            const f4 b = *reinterpret_cast<const f4*>(gc + o + 4);
            #pragma unroll
            for (int e = 0; e < 4; ++e) { x[e] = a[e]; x[4 + e] = b[e]; }
        };
        const size_t nblk = g2_nat_blk(i, hh, p.H, p.NT, false, 8192);
        {   // ---- QgA (A form, straight from registers) + Kn (B form, via T2 transpose)
            short* T2 = reinterpret_cast<short*>(smem);
            float gn[8];
            ldref(63, gn);
            #pragma unroll
            for (int rb = 0; rb < 4; ++rb) {
                v4bh qg[2];
                #pragma unroll
                for (int s = 0; s < 8; ++s) {
                    qg[s >> 2][s & 3] = bf_rne(Q(rb, s) * exp2f(G(rb, s)));
                    T2[(cbase + s) * kT2 + 16 * rb + m] = bf_rne(K(rb, s) * exp2f(gn[s] - G(rb, s)));
                }
                const size_t qb0 = nblk + (size_t)(2048 * rb + 256 * (cbase >> 4) + 4 * m
                                   + 128 * ((cbase >> 3) & 1));
                *reinterpret_cast<v4bh*>(p.QgA + nat16(qb0)) = qg[0];
                *reinterpret_cast<v4bh*>(p.QgA + nat16(qb0 + 64)) = qg[1];
            }
            lds_barrier();
            #pragma unroll
            for (int it = 0; it < 8; ++it) {
                const int U = threadIdx.x + 256 * it;   // Kn offset 4U: [d0][d1][q][n][e]
                const int dim = 16 * (U >> 8) + (U & 15), tok = 16 * ((U >> 6) & 3) + 4 * ((U >> 4) & 3);
                *reinterpret_cast<v4bh*>(p.Kn + nat16(nblk + 4 * (size_t)U)) =
                    *reinterpret_cast<const v4bh*>(T2 + dim * kT2 + tok);
            }
            lds_barrier();                               // T2 consumed (R aliases it)
        }
        if (p.stop == 6) return;
        {   // ---- P1a: diag blocks, reference = the sub-block's middle row
            f4 acc[8];
            #pragma unroll
            for (int b = 0; b < 8; ++b) acc[b] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int rb = 0; rb < 4; ++rb) {
                float gm[8];
                ldref(16 * rb + 8, gm);
                #pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    const int s0 = 2 * kt, s1 = 2 * kt + 1;
                    const f2 eA = f2{exp2f(G(rb, s0) - gm[s0]), exp2f(G(rb, s1) - gm[s1])};
                    const f2 eB = f2{exp2f(gm[s0] - G(rb, s0)), exp2f(gm[s1] - G(rb, s1))};
                    const f2 Bd = f2{K(rb, s0), K(rb, s1)} * eB;
                    acc[rb] = mmac32(f2{Q(rb, s0), Q(rb, s1)} * eA, Bd, acc[rb]);
                    acc[4 + rb] = mmac32(f2{K(rb, s0), K(rb, s1)} * eA, Bd, acc[4 + rb]);
                }
            }
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                switch ((wv + s) & 3) {
                    case 0: FI2_RED(0, acc[0]) FI2_RED(1, acc[1]) break;
                    case 1: FI2_RED(2, acc[2]) FI2_RED(3, acc[3]) break;
                    case 2: FI2_RED(4, acc[4]) FI2_RED(5, acc[5]) break;
                    default: FI2_RED(6, acc[6]) FI2_RED(7, acc[7]) break;
                }
                lds_barrier();
            }
        }
        if (p.stop == 5) return;
        // ---- P1b: off blocks (rb > cb), reference = the row block's first row.
        // Two sub-passes (rb in {1,2}: pairs 0..2; rb = 3: pairs 3..5), 6 accumulators each.
        #pragma unroll
        for (int sp = 0; sp < 2; ++sp) {
            f4 aq[3], ak[3];
            #pragma unroll
            for (int b = 0; b < 3; ++b) { aq[b] = f4{0.f, 0.f, 0.f, 0.f}; ak[b] = aq[b]; }
            #pragma unroll
            for (int rb = 1; rb < 4; ++rb) {
                if ((sp == 0) != (rb < 3)) continue;
                float gr[8];
                ldref(16 * rb, gr);
                #pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    const int s0 = 2 * kt, s1 = 2 * kt + 1;
                    const f2 eO = f2{exp2f(G(rb, s0) - gr[s0]), exp2f(G(rb, s1) - gr[s1])};
                    const f2 Aq = f2{Q(rb, s0), Q(rb, s1)} * eO;
                    const f2 Ak = f2{K(rb, s0), K(rb, s1)} * eO;
                    #pragma unroll
                    for (int cb = 0; cb < rb; ++cb) {
                        const f2 eBo = f2{exp2f(gr[s0] - G(cb, s0)), exp2f(gr[s1] - G(cb, s1))};
                        const f2 Bo = f2{K(cb, s0), K(cb, s1)} * eBo;
                        const int x = pidx(rb, cb) - 3 * sp;
                        aq[x] = mmac32(Aq, Bo, aq[x]);
                        ak[x] = mmac32(Ak, Bo, ak[x]);
                    }
                }
            }
            const int o = 3 * sp;                        // slots: Aqk off 8+p, Akk off 14+p
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                switch ((wv + s) & 3) {
                    case 0: FI2_RED(8 + o, aq[0]) FI2_RED(14 + o, ak[0]) break;
                    case 1: FI2_RED(9 + o, aq[1]) FI2_RED(15 + o, ak[1]) break;
                    case 2: FI2_RED(10 + o, aq[2]) break;
                    default: FI2_RED(16 + o, ak[2]) break;
                }
                lds_barrier();
            }
        }
        #undef FI2_RED
        if (p.stop == 1) return;
        switch (wv) {
            case 0: post<0, M>(p, smem, i, hh, l, tok0, cl); break;
            case 1: post<1, M>(p, smem, i, hh, l, tok0, cl); break;
            case 2: post<2, M>(p, smem, i, hh, l, tok0, cl); break;
            default: post<3, M>(p, smem, i, hh, l, tok0, cl); break;
        }
    });
}


// ================================================================ X1: split variant (FI=3)
// KDA_G2_FI=3: the same computation as g2_fwd_intra2_kernel split at the R boundary.
// Segment a: P1 register loads + QgA/Kn emits + P1a/P1b products + RED -> R dumped to
// a global fp32 buffer (lossless).  Segment b: post only, R read back from global.
// Rationale: the fused kernel holds the P1 q/k/g tiles resident across post, which
// costs 30 VGPR spills; each segment compiles spill-free (148 / 72 VGPR) and both run
// 3 CTAs/CU.  Expressions are verbatim copies; outputs are bit-identical (x1_bench,
// 6 shapes).  NOTE: plain __launch_bounds__(256) on both -- a (256,3) min-blocks hint
// over-constrains the allocator (94 spills measured); the natural allocation already
// reaches 3 CTAs/CU.
namespace fis {

struct FI2AParams {
    const short *qn, *kn; const float* g;
    short *QgA, *Kn; float* R;
    const int *ctok, *clen;              // VL: packed-row map (nullptr -> dense)
    int H, NT;
};

struct FI2BParams {
    const short *kn, *v; const float *g, *beta, *R;
    short *Wn, *Un, *AqkA, *Aqk, *Akk;
    const int *ctok, *clen;              // VL: packed-row map (nullptr -> dense)
    float scale;
    int H, NT;
};

template <bool VL>
__global__ void __launch_bounds__(256) g2_fwd_intra2a_kernel(FI2AParams p) {
    using namespace fi2;
    __shared__ __align__(16) float smem[5120];       // R (T2 aliases before RED writes)
    const int i = blockIdx.x, hh = blockIdx.y;
    const int wv = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int l = threadIdx.x & 63, m = l & 15, q4 = l >> 4;
    const long tok0 = g2_row_tok<VL>(p.ctok, i);
        const int cl = g2_row_len<VL>(p.clen, i);
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;
    #define FI2A_RED(B, A) { f4* d_ = reinterpret_cast<f4*>(smem + (B) * 256 + l * 4); \
                            *d_ = (s == 0) ? (A) : (*d_ + (A)); }
    const int cbase = 32 * wv + 8 * q4;
    const size_t cta0 = g2_rm_tok(tok0, hh, p.H, kK);
    const short* __restrict__ qc = p.qn + cta0;
    const short* __restrict__ kc = p.kn + cta0;
    const float* __restrict__ gc = p.g + cta0;
    const int rs = p.H * kK;
    v4bh qv[4][2], kv[4][2];
    f4 gv[4][2];
    #pragma unroll
    for (int rb = 0; rb < 4; ++rb) {
        const int o = (16 * rb + m) * rs + cbase;
        const int og = g2_row_g<M>(16 * rb + m, cl) * rs + cbase;
        const bool rok = g2_row_ok<M>(16 * rb + m, cl);
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            qv[rb][h] = rok ? *reinterpret_cast<const v4bh*>(qc + o + 4 * h) : v4bh{0, 0, 0, 0};
            kv[rb][h] = rok ? *reinterpret_cast<const v4bh*>(kc + o + 4 * h) : v4bh{0, 0, 0, 0};
            gv[rb][h] = *reinterpret_cast<const f4*>(gc + og + 4 * h);
        }
    }
    auto Q = [&](int rb, int s) { return bf2f(qv[rb][s >> 2][s & 3]); };
    auto K = [&](int rb, int s) { return bf2f(kv[rb][s >> 2][s & 3]); };
    auto G = [&](int rb, int s) { return gv[rb][s >> 2][s & 3]; };
    auto ldref = [&](int row, float* x) {
        const int o = g2_row_g<M>(row, cl) * rs + cbase;
        const f4 a = *reinterpret_cast<const f4*>(gc + o);
        const f4 b = *reinterpret_cast<const f4*>(gc + o + 4);
        #pragma unroll
        for (int e = 0; e < 4; ++e) { x[e] = a[e]; x[4 + e] = b[e]; }
    };
    const size_t nblk = g2_nat_blk(i, hh, p.H, p.NT, false, 8192);
    {   // ---- QgA (A form, from registers) + Kn (B form, via T2 transpose)
        short* T2 = reinterpret_cast<short*>(smem);
        float gn[8];
        ldref(63, gn);
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            v4bh qg[2];
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                qg[s >> 2][s & 3] = bf_rne(Q(rb, s) * exp2f(G(rb, s)));
                T2[(cbase + s) * kT2 + 16 * rb + m] = bf_rne(K(rb, s) * exp2f(gn[s] - G(rb, s)));
            }
            const size_t qb0 = nblk + (size_t)(2048 * rb + 256 * (cbase >> 4) + 4 * m
                               + 128 * ((cbase >> 3) & 1));
            *reinterpret_cast<v4bh*>(p.QgA + nat16(qb0)) = qg[0];
            *reinterpret_cast<v4bh*>(p.QgA + nat16(qb0 + 64)) = qg[1];
        }
        lds_barrier();
        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int U = threadIdx.x + 256 * it;
            const int dim = 16 * (U >> 8) + (U & 15), tok = 16 * ((U >> 6) & 3) + 4 * ((U >> 4) & 3);
            *reinterpret_cast<v4bh*>(p.Kn + nat16(nblk + 4 * (size_t)U)) =
                *reinterpret_cast<const v4bh*>(T2 + dim * kT2 + tok);
        }
        lds_barrier();                               // T2 consumed (R aliases it)
    }
    {   // ---- P1a: diag blocks, reference = the sub-block's middle row
        f4 acc[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) acc[b] = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            float gm[8];
            ldref(16 * rb + 8, gm);
            #pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                const int s0 = 2 * kt, s1 = 2 * kt + 1;
                const f2 eA = f2{exp2f(G(rb, s0) - gm[s0]), exp2f(G(rb, s1) - gm[s1])};
                const f2 eB = f2{exp2f(gm[s0] - G(rb, s0)), exp2f(gm[s1] - G(rb, s1))};
                const f2 Bd = f2{K(rb, s0), K(rb, s1)} * eB;
                acc[rb] = mmac32(f2{Q(rb, s0), Q(rb, s1)} * eA, Bd, acc[rb]);
                acc[4 + rb] = mmac32(f2{K(rb, s0), K(rb, s1)} * eA, Bd, acc[4 + rb]);
            }
        }
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            switch ((wv + s) & 3) {
                case 0: FI2A_RED(0, acc[0]) FI2A_RED(1, acc[1]) break;
                case 1: FI2A_RED(2, acc[2]) FI2A_RED(3, acc[3]) break;
                case 2: FI2A_RED(4, acc[4]) FI2A_RED(5, acc[5]) break;
                default: FI2A_RED(6, acc[6]) FI2A_RED(7, acc[7]) break;
            }
            lds_barrier();
        }
    }
    {   // ---- P1b: off blocks (rb > cb), reference = the row block's first row
        #pragma unroll
        for (int sp = 0; sp < 2; ++sp) {
            f4 aq[3], ak[3];
            #pragma unroll
            for (int b = 0; b < 3; ++b) { aq[b] = f4{0.f, 0.f, 0.f, 0.f}; ak[b] = aq[b]; }
            #pragma unroll
            for (int rb = 1; rb < 4; ++rb) {
                if ((sp == 0) != (rb < 3)) continue;
                float gr[8];
                ldref(16 * rb, gr);
                #pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    const int s0 = 2 * kt, s1 = 2 * kt + 1;
                    const f2 eO = f2{exp2f(G(rb, s0) - gr[s0]), exp2f(G(rb, s1) - gr[s1])};
                    const f2 Aq = f2{Q(rb, s0), Q(rb, s1)} * eO;
                    const f2 Ak = f2{K(rb, s0), K(rb, s1)} * eO;
                    #pragma unroll
                    for (int cb = 0; cb < rb; ++cb) {
                        const f2 eBo = f2{exp2f(gr[s0] - G(cb, s0)), exp2f(gr[s1] - G(cb, s1))};
                        const f2 Bo = f2{K(cb, s0), K(cb, s1)} * eBo;
                        const int x = pidx(rb, cb) - 3 * sp;
                        aq[x] = mmac32(Aq, Bo, aq[x]);
                        ak[x] = mmac32(Ak, Bo, ak[x]);
                    }
                }
            }
            const int o = 3 * sp;
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                switch ((wv + s) & 3) {
                    case 0: FI2A_RED(8 + o, aq[0]) FI2A_RED(14 + o, ak[0]) break;
                    case 1: FI2A_RED(9 + o, aq[1]) FI2A_RED(15 + o, ak[1]) break;
                    case 2: FI2A_RED(10 + o, aq[2]) break;
                    default: FI2A_RED(16 + o, ak[2]) break;
                }
                lds_barrier();
            }
        }
    }
    #undef FI2A_RED
    // R dump: 20 blocks x 256 fp32 -> global (lossless; segment b reads it back)
    // (R is an internal chunk-padded tensor: no VL row mapping)
    {
        const f4* s4 = reinterpret_cast<const f4*>(smem);
        f4* d4 = reinterpret_cast<f4*>(p.R + g2_nat_blk(i, hh, p.H, p.NT, false, 5120));
        #pragma unroll
        for (int j = threadIdx.x; j < 1280; j += 256) d4[j] = s4[j];
    }
    });   // g2_vl_dispatch
}

// segment b: post<I,M> verbatim except R4 reads the global R buffer.  VL semantics
// mirror fi2::post: row-major saves gated by tok_, g rows clamped, invalid rows zero.
template <int I, int M>
__device__ __forceinline__ void s_post(const FI2BParams& p, float* smem, const float* Rg,
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

    f4 Aqkd = R4(I), Ld = R4(4 + I);
    f4 Aqko[NA], Akko[NA];
    #pragma unroll
    for (int cb = 0; cb < I; ++cb) {
        Aqko[cb] = R4(8 + pidx(I, cb));
        Akko[cb] = R4(14 + pidx(I, cb));
    }
    lds_barrier();                                   // B1 (kept: harmless without LDS R)

    const int t = 16 * I + m;
    const bool tok_ = g2_row_ok<M>(t, cl);
    const float bm = tok_ ? p.beta[g2_rm_tok(tok0 + t, hh, p.H, 1)] : 0.f;
    {   // ---- Aqk emit
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
    f4 Ai[4];
    {   // ---- diag inverse (I + L)^-1
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
    lds_barrier();                                   // B2
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
    lds_barrier();                                   // B3
    if constexpr (I == 2) {
        const f4 s = P20 + mmB(Akko[1], P, q4, pm);
        Ai[0] = -mmB(Ai[2], stage(s), q4, pm);
        put(P + 2 * kTile, Ai[0]);
    } else if constexpr (I == 3) {
        const f4 s = P31 + mmB(Akko[2], P + kTile, q4, pm);
        Ai[1] = -mmB(Ai[3], stage(s), q4, pm);
        P30 = P30 + mmB(Akko[1], P, q4, pm);
    }
    lds_barrier();                                   // B4
    if constexpr (I == 3) {
        const f4 s = P30 + mmB(Akko[2], P + 2 * kTile, q4, pm);
        Ai[0] = -mmB(Ai[3], stage(s), q4, pm);
    }
    {   // ---- Akk emit
        short* rowp = p.Akk + g2_rm_tok(tok0 + t, hh, p.H, kBT);
        if (tok_) {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                *reinterpret_cast<v4bh*>(rowp + 16 * j + 4 * q4) =
                    pack4(j <= I ? Ai[j] : f4{0.f, 0.f, 0.f, 0.f});
        }
    }

    short* T2 = reinterpret_cast<short*>(smem);
    lds_barrier();                                   // B5: merge tiles consumed
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int tid = 64 * I + l;
            const int row = (tid >> 2) & 63, cq = 8 * (tid & 3) + 32 * it;
            const bool rok = g2_row_ok<M>(row, cl);
            const size_t o = g2_rm_tok(tok0 + row, hh, p.H, kK) + cq;
            const size_t og = g2_rm_tok(tok0 + g2_row_g<M>(row, cl), hh, p.H, kK) + cq;
            const float b = rok ? p.beta[g2_rm_tok(tok0 + row, hh, p.H, 1)] : 0.f;
            const v4bh z4 = v4bh{0, 0, 0, 0};
            if (pass == 0) {
                const v4bh k0 = rok ? *reinterpret_cast<const v4bh*>(p.kn + o) : z4;
                const v4bh k1 = rok ? *reinterpret_cast<const v4bh*>(p.kn + o + 4) : z4;
                const f4 g0 = *reinterpret_cast<const f4*>(p.g + og);
                const f4 g1 = *reinterpret_cast<const f4*>(p.g + og + 4);
                #pragma unroll
                for (int d = 0; d < 8; ++d)
                    T2[(cq + d) * kT2 + row] = bf_rne(bf2f(d < 4 ? k0[d & 3] : k1[d & 3]) * b
                                                      * exp2f(d < 4 ? g0[d & 3] : g1[d & 3]));
            } else {
                const v4bh v0 = rok ? *reinterpret_cast<const v4bh*>(p.v + o) : z4;
                const v4bh v1 = rok ? *reinterpret_cast<const v4bh*>(p.v + o + 4) : z4;
                #pragma unroll
                for (int d = 0; d < 8; ++d)
                    T2[(cq + d) * kT2 + row] = bf_rne(bf2f(d < 4 ? v0[d & 3] : v1[d & 3]) * b);
            }
        }
        lds_barrier();                               // B6 / B8
        f4 acc[8];
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) acc[vt] = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int j = 0; j <= I; ++j) {
            const v4bh a = pack4(Ai[j]);
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {
                const int n = pass == 0 ? pm : m;
                acc[vt] = mmac(a, *reinterpret_cast<const v4bh*>(T2 + (16 * vt + n) * kT2
                                                               + 16 * j + 4 * q4), acc[vt]);
            }
        }
        short* out = pass == 0 ? p.Wn : p.Un;
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt)
            *reinterpret_cast<v4bh*>(out + nat16(nblk + (size_t)(vt + 8 * I) * 256 + 4 * l)) = pack4(acc[vt]);
        lds_barrier();                               // B7 / B9
    }
}

template <bool VL>
__global__ void __launch_bounds__(256) g2_fwd_intra2b_kernel(FI2BParams p) {
    // merge tiles + T2 union: max(8*kTile floats = 10 KB, 128*kT2 shorts = 17.4 KB)
    __shared__ __align__(16) float smem[4352];
    const int i = blockIdx.x, hh = blockIdx.y;
    const int wv = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int l = threadIdx.x & 63;
    const long tok0 = g2_row_tok<VL>(p.ctok, i);
        const int cl = g2_row_len<VL>(p.clen, i);
    const float* Rg = p.R + g2_nat_blk(i, hh, p.H, p.NT, false, 5120);
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;
    switch (wv) {
        case 0: fis::s_post<0, M>(p, smem, Rg, i, hh, l, tok0, cl); break;
        case 1: fis::s_post<1, M>(p, smem, Rg, i, hh, l, tok0, cl); break;
        case 2: fis::s_post<2, M>(p, smem, Rg, i, hh, l, tok0, cl); break;
        default: fis::s_post<3, M>(p, smem, Rg, i, hh, l, tok0, cl); break;
    }
    });   // g2_vl_dispatch
}

}  // namespace fis

}  // namespace g2

extern "C" int g2_fwd_intra2(const void* qn, const void* kn, const void* v, const void* g,
                             const void* beta, void* Wn, void* Kn, void* Un, void* QgA,
                             void* AqkA, void* Aqk, void* Akk, double scale,
                             int B, int T, int H, const int* ctok, const int* clen,
                             hipStream_t s) {
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row tensors are packed.
    if (B != 1 || T % 64 || ((ctok == nullptr) != (clen == nullptr))) return -1;
    int stop = 0;
    if (const char* e = g2::g2_getenv("G2_FI_STOP")) stop = atoi(e);
    g2::FwdIParams p{(const short*)qn, (const short*)kn, (const short*)v, (const float*)g,
                     (const float*)beta, (short*)Wn, (short*)Kn, (short*)Un, (short*)QgA,
                     (short*)AqkA, (short*)Aqk, (short*)Akk, (float)scale, H, T / 64,
                     ctok, clen, stop};
    if (ctok)
        hipLaunchKernelGGL(g2::g2_fwd_intra2_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    else
        hipLaunchKernelGGL(g2::g2_fwd_intra2_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}

template <bool VL>
static int g2_fwd_intra2_split_vl(const void* qn, const void* kn, const void* v,
                                  const void* g, const void* beta, void* Wn, void* Kn,
                                  void* Un, void* QgA, void* AqkA, void* Aqk, void* Akk,
                                  void* R, double scale, int B, int T, int H, int NT,
                                  const int* ctok, const int* clen, hipStream_t s);

extern "C" int g2_fwd_intra2_split(const void* qn, const void* kn, const void* v,
                                   const void* g, const void* beta, void* Wn, void* Kn,
                                   void* Un, void* QgA, void* AqkA, void* Aqk, void* Akk,
                                   void* R, double scale, int B, int T, int H,
                                   const int* ctok, const int* clen, hipStream_t s) {
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row tensors are packed.
    if (B != 1 || T % 64 || ((ctok == nullptr) != (clen == nullptr))) return -1;
    const int NT = T / 64;
    if (ctok)
        return g2_fwd_intra2_split_vl<true>(qn, kn, v, g, beta, Wn, Kn, Un, QgA, AqkA, Aqk,
                                            Akk, R, scale, B, T, H, NT, ctok, clen, s);
    return g2_fwd_intra2_split_vl<false>(qn, kn, v, g, beta, Wn, Kn, Un, QgA, AqkA, Aqk,
                                         Akk, R, scale, B, T, H, NT, ctok, clen, s);
}

template <bool VL>
static int g2_fwd_intra2_split_vl(const void* qn, const void* kn, const void* v,
                                  const void* g, const void* beta, void* Wn, void* Kn,
                                  void* Un, void* QgA, void* AqkA, void* Aqk, void* Akk,
                                  void* R, double scale, int B, int T, int H, int NT,
                                  const int* ctok, const int* clen, hipStream_t s) {
    {
        g2::fis::FI2AParams pa{(const short*)qn, (const short*)kn, (const float*)g,
                           (short*)QgA, (short*)Kn, (float*)R, ctok, clen, H, NT};
        hipLaunchKernelGGL(g2::fis::g2_fwd_intra2a_kernel<VL>, dim3(NT, H), dim3(256), 0, s, pa);
        const int rc = (int)hipGetLastError();
        if (rc) return rc;
    }
    {
        g2::fis::FI2BParams pb{(const short*)kn, (const short*)v, (const float*)g,
                           (const float*)beta, (const float*)R,
                           (short*)Wn, (short*)Un, (short*)AqkA, (short*)Aqk, (short*)Akk,
                           ctok, clen, (float)scale, H, NT};
        hipLaunchKernelGGL(g2::fis::g2_fwd_intra2b_kernel<VL>, dim3(NT, H), dim3(256), 0, s, pb);
        return (int)hipGetLastError();
    }
}
