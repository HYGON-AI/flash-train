// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Handwritten gfx936 G2 kernels.
// clang-format off
// S3-9/S3-11: recompute_w_u as one HIP kernel emitting ALL P1/P2 native operands.
//
// Math (FLA wy_fast.recompute_w_u_fwd, dense, safe-gate tier):
//   kb = bf16((k * beta) * 2^g)        w = A @ kb      (A = saved Akk, the UT inverse)
//   vb = bf16(v * beta)                u = A @ vb
//   kg = bf16(k * 2^(gn - g))  (gn = chunk-last g row)  -> A form (KgA) + B form (Kn)
//   qg = bf16(q * 2^g)                                       -> B form (Qn)
// The MMAC tier matches FLA (bf16 operands, fp32 accumulate); outputs verified
// bitwise against the FLA + torch-conversion reference chain.
//
// S3-11 structure: ONE 17.4KB LDS slab reused by the MMAC phases (kbT / vbT) plus
// a 16.9KB row-major tile for the gated emits -- ~34KB total but only ONE of the
// two slabs is live per phase... (both are declared, 34.6KB, 1 CTA/CU was the S3-9
// trap) -- instead the tile reuses the SAME storage via a union, keeping peak LDS
// at 17.4KB so 2+ CTAs fit per CU.
//
// T1 (GLM): the phase-1/2 fill loops additionally emit P3's two B5 operands
// straight from the k/g and v loads they already perform:
//   kgB = bf16(k * 2^g)   absolute gate -- NOT the 2^(gn-g) of phases 3/4, no beta
//   vB  = v               raw bf16, no beta
// both in the vB layout = A form with the row low nibble pi16-permuted and the
// P3 block order (i*H + hh).  The source quad is contiguous on BOTH sides, so the
// permutation only moves the quad base -- one extra 8B store per quad, no LDS
// round-trip, and the two layout2 convert2 calls (which re-read k/g/v, ~400 MB
// at 8K/48) disappear from B5.
//
// T2 instrumentation: G2_WU_STOP=N (env, read at launch) makes every CTA return
// right after stage N -- 1 sBeta/sGl/sA+fill1(kbT+kgB+KgA/Kn), 2 w-MMAC,
// 3 Wn/Wb emit, 4 fill2, 5 u-MMAC, 6 Un emit, 7 Qn direct pass; 0 = full.
// Stage costs come from consecutive differences (tests/g2/probe_wu_stages.py).
// Uniform branch, no data effect at stop=0.
//
// T2 round 2: (a) C-pi -- kbT staged with pi16-permuted rows so the w-MMAC C
// fragment holds lane-contiguous native Wn columns: the Wn emit is ONE aligned
// 8B quad per lane (was 4 stride-64 scalars) and Wb lands at stride 4 (was 16);
// vbT/u-MMAC keep identity staging (Un's C form IS the fragment layout).
// (b) A cached in LDS (both MMAC phases re-read it, 50 MB each).  (c) Qn
// emitted by a light standalone pass instead of the slab-tile round-trip.
//
// Emitting from the MMAC C fragments is free of layout translation except the
// known swaps: lane l holds (row = l&15, col = (l>>4)+4e), so the A form needs
// stride-64 scalars, the B form stride-16, the C form l*4 vecs.
#include "g2_common.h"

namespace g2 {

struct WuParams {
    const short* k; const short* v; const short* q; const float* beta; const short* A;
    const float* g;
    short* Wn; short* Wb; short* Un; short* KgA; short* Kn; short* Qn;
    short* kgB; short* vB;
    int H, NT;
    const int* ctok; const int* clen;   // varlen phase 2 (null = dense)
    int stop;      // T2 staged timing (G2_WU_STOP): 0 = full kernel; 1..9 return
                   // right after the named stage (uniform branch, no data effect)
    int kp = 0;    // kra P-61e: Kn lane-permuted (H <= KPERM_MAX_H)
    const short* HbN = nullptr;   // kra P-82: saved bN(h) -> v_new = u - W h formed here (VnOut)
    short* VnOut = nullptr;
    int order = 0;   // kra P-95: CTA visiting order (see g2_wu_kernel)
    int order_m = 4; // order 3: heads interleaved per chunk (divides H)
};

// vB-layout quad base for source quad (row j, column vq..vq+3): A form with the
// row low nibble pi16-permuted and the P3 block order (g2_layout2 L2_KGB/L2_VB).
__device__ __forceinline__ size_t wu_vb_quad(int j, int vq, size_t nblk3) {
    return nblk3 + (size_t)(4 * pi16(j & 15) + 64 * ((vq & 15) >> 2)
                            + 256 * (vq >> 4) + 2048 * (j >> 4));
}

constexpr int kPadJ2 = 68;
constexpr int kPadA = 68;   // sA row stride (shorts): 136 B rows spread the MMAC A-fragment reads over all banks

// V1 (kra P-1): bank-conflict-free LDS layouts.  Pure address remaps -- every value is
// computed exactly as before, so all outputs stay bitwise identical.
//  slab: the 8 B column granule (j >> 2) of row r is XOR-ed with (r >> 3) & 15, which
//        spreads the column-wise b16 fill stores of phases 1/2 (8-way -> 2-way) and keeps
//        the MMAC B-fragment b64 reads conflict-free.
//  tile: the 8 B granule of element e is XOR-ed with (e >> 10) & 7 (stays inside the
//        8192-element block), spreading the B-form scalar stores (8-way -> 2-way) while
//        the linear 8 B emit reads stay conflict-free.
__device__ __forceinline__ int wu_slab_col(int r, int j) {
    return ((((j >> 2) ^ (r >> 3)) & 15) << 2) | (j & 3);
}
__device__ __forceinline__ int wu_tile_idx(int e) {
    return (((e >> 2) ^ ((e >> 10) & 7)) << 2) | (e & 3);
}

// VL (varlen phase 2): k/v/q/beta/A rows r >= clen read as 0 (the zero padding), g rows
// are read from the CLAMPED row (address only, the value math is the dense code), incl.
// the chunk-end row sGl = g[clen-1].  No row-major outputs.
// kra L16: emit this CTA's 8192-element native block from the LDS tile as linear 16 B pairs
// (pair t = sub-blocks 2(t>>6), 2(t>>6)+1 at lane t&63 -> nat16 position 8t).
__device__ __forceinline__ void wu_emit16(short* dst, const short* tile, int tid) {
    for (int t = tid; t < 1024; t += 256) {
        const int o0 = 512 * (t >> 6) + 4 * (t & 63);
        st16x2(dst + 8 * (size_t)t, *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0)),
               *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0 + 256)));
    }
}

// kra P-61e: wu_emit16 with the STORE lane pi16-permuted inside every 16-lane group (tile reads unchanged)
__device__ __forceinline__ void wu_emit16s(short* dst, const short* tile, int tid) {
    for (int t = tid; t < 1024; t += 256) {
        const int o0 = 512 * (t >> 6) + 4 * (t & 63);
        st16x2(dst + 8 * (size_t)((t & ~15) | pi16(t & 15)), *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0)),
               *reinterpret_cast<const v4bh*>(tile + wu_tile_idx(o0 + 256)));
    }
}

template <bool VL, bool KS = false, bool QS = false>   // kra P-84b: KS; P-84c: QS
__global__ void __launch_bounds__(256) g2_wu_kernel(WuParams p_) {
    // kra P-35: field-wise copy, pointers stamped global (global_* ops, not flat)
    const WuParams p{g2_glb(p_.k), g2_glb(p_.v), g2_glb(p_.q), g2_glb(p_.beta), g2_glb(p_.A), g2_glb(p_.g), g2_glb(p_.Wn), g2_glb(p_.Wb), g2_glb(p_.Un), g2_glb(p_.KgA), g2_glb(p_.Kn), g2_glb(p_.Qn), g2_glb(p_.kgB), g2_glb(p_.vB), p_.H, p_.NT, g2_glb(p_.ctok), g2_glb(p_.clen), p_.stop, p_.kp, g2_glb(p_.HbN), g2_glb(p_.VnOut), p_.order, p_.order_m};   // kra P-82, P-95
    // kra P-95: CTA visiting order (bitwise-neutral: every CTA is independent; launcher picks it).  0 = chunk-fastest
    // (blockIdx.x = chunk); 1 = head-fastest; 2 = diagonal (i = (L/H + hh) % NT): co-resident CTAs
    // then differ by one chunk AND one head, so neither the row streams (H*16 KiB per chunk) nor the
    // natives (NT*16 KiB per head) advance in lock-step by a power of two when H / NT are.
    // 3 = M-head interleave (M = order_m | H): CTAs L .. L+M-1 take M adjacent heads at one chunk.
    const int L_ = blockIdx.y * gridDim.x + blockIdx.x;
    int hh, i;
    if (p_.order == 0) { hh = (int)blockIdx.y; i = (int)blockIdx.x; }
    else if (p_.order == 3) {
        const int M = p_.order_m, r_ = L_ / M;
        hh = (r_ / p_.NT) * M + L_ % M; i = r_ % p_.NT;
    } else { hh = L_ % p_.H; i = p_.order == 1 ? L_ / p_.H : (L_ / p_.H + hh) % p_.NT; }
    const int tid = threadIdx.x;
    const int wv = tid >> 6, l = tid & 63;
    __shared__ __align__(16) short slab[128 * kPadJ2];
    __shared__ float sBeta[kBT];
    __shared__ __align__(16) float sGl[kK];   // chunk-last g row: kg = k * 2^(gl-g)
    __shared__ __align__(16) short sA[kBT * kPadA];
    const int cl = g2_row_len<VL>(p.clen, i);
    // varlen: ONE uniform branch per CTA -- full chunks run the unmasked body (M=1)
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;
        const size_t rstride = (size_t)p.H * kK;           // [B,T,H,128] row stride
        const size_t sbase = ((size_t)g2_row_tok<VL>(p.ctok, i) * p.H + hh) * kK;
        const size_t nblk = (size_t)(hh * p.NT + i) * 8192;
        const size_t nblk3 = g2_nat_blk(i, hh, p.H, p.NT, true, 8192);   // P3's (i*H+hh)
        // 17.4KB union: transposed MMAC staging (kbT/vbT) OR the row-major gated tile
        // A (UT inverse, 64x64 bf16): read by BOTH MMAC phases -- cache it in LDS
        // (8KB; occupancy is VGPR-bound at 2 CTAs/CU, so the LDS is free).

        if (tid < kBT)
            sBeta[tid] = g2_row_ok<M>(tid, cl)
                ? p.beta[(size_t)(g2_row_tok<VL>(p.ctok, i) + tid) * p.H + hh] : 0.f;
        if (tid < 32)                  // 128 floats = 32 f4 quads
            reinterpret_cast<f4*>(sGl)[tid] =
                ld16f(p.g + sbase + g2_row_g<M>(63, cl) * rstride + 4 * tid);
        #pragma unroll   // kra P-34: all loads in flight before the first wait
        for (int it_ = 0; it_ < 4; ++it_) {
            const int t = tid + 256 * it_;          // sA: 64x64 bf16 row-major
            const int r = t >> 4, c4 = (t & 15) << 2;
            *reinterpret_cast<v4bh*>(sA + r * kPadA + c4) = g2_row_ok<M>(r, cl)
                ? *reinterpret_cast<const v4bh*>(
                      p.A + g2_rm_tok(g2_row_tok<VL>(p.ctok, i) + r, hh, p.H, kBT) + c4)
                : v4bh{0, 0, 0, 0};
        }
        __syncthreads();   // sBeta/sGl/sA must land before the phase-1 fill reads them

        // ---- MMAC phase: A @ (B-side staged transposed in slab) -> emit from C frags.
        // A comes from sA (cached).  EMIT CONVENTION: the w phase stages kbT with
        // pi16-permuted rows (see fill1), so its C fragment holds lane-contiguous
        // native columns: Wn (A form) is ONE aligned 8B quad store per lane and Wb
        // (B form) lands at native stride 4.  The u phase stages vbT with identity
        // rows and only uses the C-form emit (the C form IS the fragment layout).
        v4bh wA[8];   // kra P-82: the w phase's A fragments (== fwd_h_bn's wf for this wave / lane)
        auto mmac_phase = [&](short* outA, short* outB, short* outC, int stop_acc) {
            __syncthreads();
            f4 acc[8];
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) acc[vt] = f4{0.f, 0.f, 0.f, 0.f};
            for (int js = 0; js < 4; ++js) {
                const v4bh af = *reinterpret_cast<const v4bh*>(
                    sA + (size_t)(16 * wv + (l & 15)) * kPadA + 16 * js + (l >> 4) * 4);
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) {
                    const int br = 16 * vt + (l & 15);
                    const size_t lo = (size_t)br * kPadJ2 + wu_slab_col(br, 16 * js + (l >> 4) * 4);
                    acc[vt] = mmac(af, *reinterpret_cast<const v4bh*>(slab + lo), acc[vt]);
                }
            }
            if (p.stop == stop_acc) return;
            v4bh we8[8];
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt)
                #pragma unroll
                for (int e = 0; e < 4; ++e) we8[vt][e] = bf_rne(acc[vt][e]);
            if (p.VnOut != nullptr && stop_acc == 2) {   // kra P-82: keep W for v_new
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) wA[vt] = we8[vt];
            }
            if (p.VnOut != nullptr && stop_acc == 5) {   // kra P-82: v_new = u - W h (fwd_h_bn's GEMM1 + rounding)
                short* hb = slab;                        // 2 x [32 v][kPadK] hB tiles
                const int m = l & 15, q4 = l >> 4;
                #pragma unroll   // static we8 indices (R55: a runtime vs puts we8 in scratch)
                for (int pr = 0; pr < 2; ++pr) {
                    __syncthreads();                     // slab free: MMAC reads / previous pair's reads done
                    #pragma unroll
                    for (int it = 0; it < 8; ++it) {     // 2 slabs x 8 kb x 2 vt x 64 lanes v4bh / 256 threads
                        const int jj = tid + 256 * it;
                        const int ln = jj & 63, vt = (jj >> 6) & 1, kb = (jj >> 7) & 7, sp = jj >> 10;
                        const int vs = 2 * pr + sp;
                        const v4bh val = *reinterpret_cast<const v4bh*>(
                            p.HbN + ((((size_t)i * p.H + hh) * 8 + kb) * 8 + (2 * vs + vt)) * 256 + ln * 4);
                        const int qq = ln >> 4, mm = ln & 15;
                        #pragma unroll
                        for (int e = 0; e < 4; ++e)
                            hb[sp * 32 * kPadK + (16 * vt + 4 * qq + e) * kPadK + 16 * kb + pi16(mm)] = val[e];
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int sp = 0; sp < 2; ++sp) {
                        const int vs = 2 * pr + sp;
                        const short* hs = hb + sp * 32 * kPadK;
                        f4 acc0 = f4{0.f, 0.f, 0.f, 0.f}, acc1 = f4{0.f, 0.f, 0.f, 0.f};
                        #pragma unroll
                        for (int kt = 0; kt < 8; ++kt) {
                            const v4bh hbf0 = *reinterpret_cast<const v4bh*>(hs + (0 * 16 + m) * kPadK + 16 * kt + q4 * 4);
                            const v4bh hbf1 = *reinterpret_cast<const v4bh*>(hs + (1 * 16 + m) * kPadK + 16 * kt + q4 * 4);
                            acc0 = mmac(wA[kt], hbf0, acc0);
                            acc1 = mmac(wA[kt], hbf1, acc1);
                        }
                        v4bh x0, x1;
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float u0 = __uint_as_float(((uint32_t)(unsigned short)we8[2 * vs][e]) << 16);
                            const float u1 = __uint_as_float(((uint32_t)(unsigned short)we8[2 * vs + 1][e]) << 16);
                            x0[e] = bf_rne(u0 - acc0[e]);
                            x1[e] = bf_rne(u1 - acc1[e]);
                        }
                        *reinterpret_cast<v4bh*>(p.VnOut + nblk + (size_t)(8 * wv + 2 * vs + 0) * 256 + l * 4) = x0;
                        *reinterpret_cast<v4bh*>(p.VnOut + nblk + (size_t)(8 * wv + 2 * vs + 1) * 256 + l * 4) = x1;
                    }
                }
                __syncthreads();                         // slab reuse by the next phase
            }
            // kra L16: C form (outC) and pi-staged A form (outA: 4(l&15) + 64(l>>4) == 4l) put
            // lane l's quad of sub-block vt + 8wv at slot l, so the vt pair (2j, 2j+1) is ONE
            // full 16 B store at nat16 position base(2j + 8wv) + 8l.
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const size_t pb = nblk + (size_t)(2 * j + 8 * wv) * 256 + 8 * l;
                if (outC) st16x2(outC + pb, we8[2 * j], we8[2 * j + 1]);
                if (outA) st16x2(outA + pb, we8[2 * j], we8[2 * j + 1]);
            }
            if (outB) { // B form, pi-staged: t = 16wv+l&15, v = 16vt+4(l>>4)+e
                // kra L16: scalars -> LDS tile (the slab: every wave's MMAC reads are done
                // after the barrier) -> linear 16 B pair emit; the barrier after the emit
                // protects the tile from the next phase's slab fill.
                short* tile = slab;
                __syncthreads();
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) {
                    const int ib = ((l & 15) & 3) + 16 * (l >> 4) + 64 * ((l & 15) >> 2)
                                   + 256 * wv + 1024 * vt;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) tile[wu_tile_idx(ib + 4 * e)] = we8[vt][e];
                }
                __syncthreads();
                wu_emit16(outB + nblk, tile, tid);
                __syncthreads();
            }
        };

        // ---- phase 1: kbT = bf16((k*beta)*2^g)^T ; w = A @ kb -> Wn (A) + Wb (B)
        // T1: the same k/g loads also emit kgB = bf16(k * 2^g) (absolute gate, no beta)
        // into the vB layout -- one extra 8B store per quad, no extra traffic reads.
        // T2: the SAME pass also computes kg = bf16(k * 2^(gl-g)) (gl = chunk-last g
        // row, sGl) and emits KgA (A form, aligned 8B quads) + Kn (B form, stride-4
        // scalars) directly -- the old kg phases re-read k+g (290 MB @8K/48) purely
        // redundantly.  Value expressions are character-identical to the retired
        // stage_gated(last_ref)/emit_AB path (bitwise gate holds).
        v4bh kgr[8];                                     // kra P-19: kg quads carried to phase 3
        f4 egr[8];                                       // kra P-19: exp2f(g) carried to phase 3
        if constexpr (KS) {   // kra P-84b: every k / g quad in flight before the first use (the kgB / KgA
            #pragma unroll    // stores of an iteration would otherwise pin the next iteration's loads)
            for (int it = 0; it < 8; ++it) {
                const int t = tid + 256 * it;
                const int j = t >> 5, vq = (t & 31) << 2;
                const size_t so = sbase + (size_t)j * rstride + vq;
                kgr[it] = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so) : v4bh{0, 0, 0, 0};
                egr[it] = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);
            }
        }
        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int t = tid + 256 * it;
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh kv = KS ? kgr[it] : (g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.k + so)
                                                 : v4bh{0, 0, 0, 0});
            const f4 gf = KS ? egr[it] : ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);
            const f4 gl = *reinterpret_cast<const f4*>(sGl + vq);
            const float b = sBeta[j];
            // T7: one exp2f(gf) per element serves both kb and kgB (identical
            // value; the multiply associations below are unchanged).
            f4 eg;
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv) eg[dv] = __builtin_amdgcn_exp2f(gf[dv]);
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv) {
                // kbT staged with pi16-permuted rows (C-pi): the w-MMAC C
                // fragment then holds lane-contiguous native Wn columns.
                const int vv = vq + dv;
                const int sr = 16 * (vv >> 4) + pi16(vv & 15);
                slab[(size_t)sr * kPadJ2 + wu_slab_col(sr, j)] =
                    bf_rne(bf2f(kv[dv]) * b * eg[dv]);
            }
            short gv[4], gkv[4];
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv) {
                gv[dv] = bf_rne(bf2f(kv[dv]) * eg[dv]);
                gkv[dv] = bf_rne(bf2f(kv[dv]) * __builtin_amdgcn_exp2f(gl[dv] - gf[dv]));
            }
            kgr[it] = *reinterpret_cast<const v4bh*>(gkv);
            egr[it] = eg;
            *reinterpret_cast<v4bh*>(p.kgB + wu_vb_quad(j, vq, nblk3)) =
                *reinterpret_cast<const v4bh*>(gv);
            // KgA: A form, plain row nibble, P1/P2 block order
            *reinterpret_cast<v4bh*>(p.KgA + nat16(nblk + (size_t)(4 * (j & 15)
                + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4)))) =
                *reinterpret_cast<const v4bh*>(gkv);
            // Kn (B form): emitted in phase 3 through the LDS tile (kra L16)
        }
        if (p.stop == 1) return;
        mmac_phase(p.Wn, p.Wb, nullptr, 2);
        if (p.stop == 2 || p.stop == 3) return;

        // ---- phase 2: vbT = bf16(v*beta)^T ; u = A @ vb -> Un (C)
        // T1: vB = raw v (no beta) into the same vB layout, from the same load.
        if (p.vB == nullptr) {   // kra P-34b: production route, no dead store in the loop
        #pragma unroll   // kra P-34: all loads in flight before the first wait
        for (int it_ = 0; it_ < 8; ++it_) {
            const int t = tid + 256 * it_;
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh vv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.v + so)
                                                 : v4bh{0, 0, 0, 0};
            const float b = sBeta[j];
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv)
                slab[(size_t)(vq + dv) * kPadJ2 + wu_slab_col(vq + dv, j)] = bf_rne(bf2f(vv[dv]) * b);
        }
        } else {
        #pragma unroll   // kra P-34: all loads in flight before the first wait
        for (int it_ = 0; it_ < 8; ++it_) {
            const int t = tid + 256 * it_;
            const int j = t >> 5, vq = (t & 31) << 2;
            const size_t so = sbase + (size_t)j * rstride + vq;
            const v4bh vv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.v + so)
                                                 : v4bh{0, 0, 0, 0};
            const float b = sBeta[j];
            #pragma unroll
            for (int dv = 0; dv < 4; ++dv)
                slab[(size_t)(vq + dv) * kPadJ2 + wu_slab_col(vq + dv, j)] = bf_rne(bf2f(vv[dv]) * b);
            if (p.vB != nullptr)   // kra P-17: production route reads v row-major in s4a instead
                *reinterpret_cast<v4bh*>(p.vB + wu_vb_quad(j, vq, nblk3)) = vv;
        }
        }
        if (p.stop == 4) return;
        mmac_phase(nullptr, nullptr, p.Un, 5);
        if (p.stop == 5 || p.stop == 6) return;

        // ---- phase 3 (T7): qg = bf16(q * 2^g) to Qn (B form) through a B-form
        // tile in the now-FREE slab: the stride-4 global scatter (the measured
        // 465 GB/s tax, Kn/Qn class) moves to the LDS side as scalar stores and
        // the global emit becomes linear 8B quads.  tile = this CTA's whole
        // 8192-element native block, aliasing the slab union (16 KB <= 17.4 KB).
        {
            short* tile = slab;                             // 64x128 in native order
            auto qg_of = [&](int t, short* qgv) {           // T7 value, unchanged
                const int j = t >> 5, vq = (t & 31) << 2;
                const size_t so = sbase + (size_t)j * rstride + vq;
                const v4bh qv = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.q + so)
                                                     : v4bh{0, 0, 0, 0};
                const f4 gf = ld16f(p.g + sbase + (size_t)g2_row_g<M>(j, cl) * rstride + vq);
                #pragma unroll
                for (int dv = 0; dv < 4; ++dv) qgv[dv] = bf_rne(bf2f(qv[dv]) * __builtin_amdgcn_exp2f(gf[dv]));
                return gf;
            };
            auto ibn_of = [](int t) {
                const int j = t >> 5, vq = (t & 31) << 2;
                return (j & 3) + 4 * (vq & 15) + 64 * ((j >> 2) & 3) + 256 * (j >> 4) + 1024 * (vq >> 4);
            };
            if (p.Kn != nullptr) {                          // null: keep-state backward
                // kra L16: one pass over k / q / g -- kg (phase 1's exact expression, same
                // inputs) -> tile -> linear 16 B pair emit to Kn; qg waits in registers.
                short qst[8][4];
                __syncthreads();                            // every wave's u-MMAC slab reads done
                #pragma unroll
                for (int it = 0; it < 8; ++it) {
                    const int tt = tid + 256 * it;
                    const int j = tt >> 5, vq = (tt & 31) << 2;
                    // kra P-19: qg_of's value with the carried exp2f(g); kg = the carried quad
                    const v4bh qv = g2_row_ok<M>(j, cl)
                        ? *reinterpret_cast<const v4bh*>(p.q + sbase + (size_t)j * rstride + vq)
                        : v4bh{0, 0, 0, 0};
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv) qst[it][dv] = bf_rne(bf2f(qv[dv]) * egr[it][dv]);
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv)
                        tile[wu_tile_idx(ibn_of(tt) + 4 * dv)] = kgr[it][dv];
                }
                __syncthreads();
                if (p.kp) wu_emit16s(p.Kn + nblk, tile, tid);   // kra P-61e
                else      wu_emit16(p.Kn + nblk, tile, tid);
                __syncthreads();
                #pragma unroll
                for (int it = 0; it < 8; ++it)
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv) tile[wu_tile_idx(ibn_of(tid + 256 * it) + 4 * dv)] = qst[it][dv];
            } else if constexpr (QS) {   // kra P-84c: carried exp2f(g) (the Kn branch's expression), q quads first
                v4bh qvs[8];
                #pragma unroll
                for (int it = 0; it < 8; ++it) {
                    const int tt = tid + 256 * it;
                    const int j = tt >> 5, vq = (tt & 31) << 2;
                    qvs[it] = g2_row_ok<M>(j, cl)
                        ? *reinterpret_cast<const v4bh*>(p.q + sbase + (size_t)j * rstride + vq) : v4bh{0, 0, 0, 0};
                }
                __syncthreads();                            // every wave's u-MMAC slab reads done
                #pragma unroll
                for (int it = 0; it < 8; ++it)
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv)
                        tile[wu_tile_idx(ibn_of(tid + 256 * it) + 4 * dv)] = bf_rne(bf2f(qvs[it][dv]) * egr[it][dv]);
            } else {
                for (int t = tid; t < 2048; t += 256) {
                    short qgv[4];
                    qg_of(t, qgv);
                    #pragma unroll
                    for (int dv = 0; dv < 4; ++dv) tile[wu_tile_idx(ibn_of(t) + 4 * dv)] = qgv[dv];
                }
            }
            __syncthreads();
            wu_emit16(p.Qn + nblk, tile, tid);              // kra L16: linear 16B pair emit
        }
        if (p.stop == 7) return;
    });
}

}  // namespace g2

extern "C" int g2_wu(const void* k, const void* v, const void* q, const void* beta,
                     const void* A, const void* g,
                     void* Wn, void* Wb, void* Un, void* KgA, void* Kn, void* Qn,
                     void* kgB, void* vB,
                     int B, int T, int H, const int* ctok, const int* clen,
                     const void* hbn, void* vn_out, hipStream_t s) {   // kra P-82
    // T = chunk-padded rows (NT = T/64); with ctok/clen the row inputs are packed.
    if (B != 1 || T % 64 || ((ctok == nullptr) != (clen == nullptr))) return -1;
    int stop = 0;
    if (const char* e = g2::g2_getenv("G2_WU_STOP")) stop = atoi(e);
    g2::WuParams p{(const short*)k, (const short*)v, (const short*)q, (const float*)beta,
                   (const short*)A, (const float*)g,
                   (short*)Wn, (short*)Wb, (short*)Un, (short*)KgA, (short*)Kn, (short*)Qn,
                   (short*)kgB, (short*)vB,
                   H, T / 64, ctok, clen, stop};
    p.kp = H <= g2::kPermMaxH;   // kra P-61e
    p.HbN = (const short*)hbn; p.VnOut = (short*)vn_out;      // kra P-82
    // kra P-95: the diagonal order whenever the chunk row stride H * 16 KiB is a multiple of 512 KiB
    // (H % 32 == 0): chunk-fastest CTAs then step through k / v / q / g / kgB in lock-step on the same
    // HBM channels (measured on gfx936: wu 1.1-1.5x at H = 64 / 32, 0.85-1.0x at H = 16 / 8).
    // kra P-97: 4 adjacent heads per chunk for the remaining multiples of 8 (wu 1.06-1.09x at H = 16 / 48,
    // 1.01-1.03x at H = 8 / 24; 0.95-0.99x at H = 12 / 20 / 36 / 60, so not there).
    p.order = H % 32 == 0 ? 2 : (H % 8 == 0 ? 3 : 0);
    p.order_m = 4;
    if (p.order == 3 && (p.order_m < 1 || H % p.order_m)) p.order = 0;
    constexpr bool ks = true;   // kra P-84b
    constexpr bool qs = true;   // kra P-84c
    if (ctok) {
        if (ks && qs) hipLaunchKernelGGL((g2::g2_wu_kernel<true, true, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else if (ks)  hipLaunchKernelGGL((g2::g2_wu_kernel<true, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else          hipLaunchKernelGGL(g2::g2_wu_kernel<true>, dim3(T / 64, H), dim3(256), 0, s, p);
    } else {
        if (ks && qs) hipLaunchKernelGGL((g2::g2_wu_kernel<false, true, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else if (ks)  hipLaunchKernelGGL((g2::g2_wu_kernel<false, true>), dim3(T / 64, H), dim3(256), 0, s, p);
        else          hipLaunchKernelGGL(g2::g2_wu_kernel<false>, dim3(T / 64, H), dim3(256), 0, s, p);
    }
    return (int)hipGetLastError();
}
