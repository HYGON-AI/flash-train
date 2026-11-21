// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// S3-11: HIP chunk_kda_bwd_dAv -- reads v_new in its NATIVE C-form layout
// (P1's Vn output, killing from_vn) and emits dv straight into the C-form
// native dVn (killing the dVn conversion).
//
// Math (fla chunk_bwd.chunk_kda_bwd_kernel_dAv, dense):
//   A    = Aqk row-major bf16 [B,T,H,64], causal-masked col <= row (FLA leaves
//          intermittent garbage above the diagonal, so the mask is required)
//   dA   = (do @ v_new^T) * scale, fp32 row-major, kept where row >= col else 0
//   dv   = A^T @ do, emitted here directly in C-form native
// Two MMACs per (chunk, head).  v_new arrives as C-form native: the quad at
// sb*256 + c*4 holds v_new[t = 16*d0 + (c&15)][v = 16*d1 + (c>>4) + 4e] with
// sb = d1 + 8*d0, so contiguous 8B reads scatter into the [v][t] LDS staging.
#include "g2_common.h"

namespace g2 {

struct DavParams {
    const short* do_; const short* Vn; const short* A;
    float* dA; short* dVn;
    short* dOT; short* aNdo;      // T3: P2's dOT (B form) + P3's aN_do (A form, p3 order)
    float scale; int H, NT;
    const int* ctok; const int* clen;        // varlen phase 2 (null = dense)
    int order = 0, order_m = 4;              // kra P-938o2: CTA visiting order (g2_cta_order)
    short* VnA = nullptr;                    // kra P-938va: non-null -> v_new re-emitted in A form (std order) here
};

constexpr int kPadJ3 = 68;

// VL (varlen phase 2): do / Aqk rows r >= clen read as 0 (= the zero padding; they only
// feed MMAC operands), dA rows r >= clen are not stored.  Natives keep NT chunks.
template <bool VL>
__global__ void __launch_bounds__(256) g2_dav_kernel(DavParams p_) {
    // kra P-35: field-wise copy, pointers stamped global (global_* ops, not flat)
    const DavParams p{g2_glb(p_.do_), g2_glb(p_.Vn), g2_glb(p_.A), g2_glb(p_.dA), g2_glb(p_.dVn), g2_glb(p_.dOT), g2_glb(p_.aNdo), p_.scale, p_.H, p_.NT, g2_glb(p_.ctok), g2_glb(p_.clen), p_.order, p_.order_m, g2_glb(p_.VnA)};
    int i, hh;
    g2_cta_order(p.order, p.order_m, p.H, p.NT, i, hh);   // kra P-938o2
    const int tid = threadIdx.x;
    const int wv = tid >> 6, l = tid & 63;
    const long t0 = g2_row_tok<VL>(p.ctok, i);
#if defined(__gfx938__)
    // kra P-938v: unpadded 16 KB slab (4 CTAs per CU instead of 3) with an XOR swizzle of column bits 2..5 by the
    // row's low 4 bits in place of the +4 padding: vectors stay contiguous, the 16-row operand reads stay
    // conflict-free.  Layout only -- bit-identical.
    __shared__ __align__(16) short slab[128 * 64];
    auto i1 = [](int r, int c) { return (size_t)r * kK + (c ^ ((r & 15) << 2)); };       // pass 1 [j][v]
    // kra P-938bc2: pass 2's swizzle also takes v bits 4..5 into the granule's low bits: the do scatter (lanes with v = 4k + d,
    // one t) then spreads over 16 granules (8-way -> 2-way bank conflicts, 1792 -> 256 extra cycles per CTA); the dOT and
    // MMAC b64 reads stay conflict-free (rows 16 vt + m and the 8 even / odd rows of a dOT group still map to distinct granules).
    auto i2 = [](int v, int j) { return (size_t)v * 64 + (j ^ (((v & 15) ^ ((v >> 4) & 3)) << 2)); };   // pass 2 [v][t]
    auto iA = [](int t, int j) { return (size_t)t * 64 + (j ^ ((t & 15) << 2)); };       // dA staging fp32
#else
    __shared__ __align__(16) short slab[128 * kPadJ3];       // 8704 shorts:
    auto i1 = [](int r, int c) { return (size_t)r * (kK + 4) + c; };
    auto i2 = [](int v, int j) { return (size_t)v * kPadJ3 + j; };
    auto iA = [](int t, int j) { return (size_t)t * kPadJ3 + j; };
#endif
    const int cl = g2_row_len<VL>(p.clen, i);
    // varlen: ONE uniform branch per CTA -- full chunks run the unmasked body (M=1)
    g2_vl_dispatch<VL>(cl, [&](auto mode_) {
    constexpr int M = decltype(mode_)::value;
        const size_t rstride = (size_t)p.H * kV;                  // do_ rows ([B,T,H,128])
        const size_t vbase = ((size_t)t0 * p.H + hh) * kV;
        const size_t nblk = (size_t)(hh * p.NT + i) * 8192;       // Vn / dVn native
        const size_t nblk3 = g2_nat_blk(i, hh, p.H, p.NT, true, 8192);   // aN_do's (i*H+hh)
#if defined(__gfx938__)
        // kra P-938d: every global operand of both passes issued up front (pass 1's do rows, pass 2's do tile and A^T
        // gathers): their latency hid nowhere before (SQTT: vmcnt waits 32% of dav).  Same loads, same values.
        v4bh dfa[8], dva[8], afa[4];
        {
            const int rdo = 16 * wv + (l & 15);
            const bool dok = g2_row_ok<M>(rdo, cl);
            #pragma unroll
            for (int vs = 0; vs < 8; ++vs)
                dfa[vs] = dok ? *reinterpret_cast<const v4bh*>(p.do_ + g2_rm_tok(t0 + rdo, hh, p.H, kV) + 16 * vs + (l >> 4) * 4)
                              : v4bh{0, 0, 0, 0};
            #pragma unroll
            for (int it_ = 0; it_ < 8; ++it_) {
                const int t = tid + 256 * it_;
                const int j = t >> 5, vq = (t & 31) << 2;
                dva[it_] = g2_row_ok<M>(j, cl) ? *reinterpret_cast<const v4bh*>(p.do_ + vbase + (size_t)j * rstride + vq)
                                               : v4bh{0, 0, 0, 0};
            }
            const int tp = 16 * wv + (l & 15);
            #pragma unroll
            for (int js = 0; js < 4; ++js)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int t = 16 * js + 4 * (l >> 4) + e;
                    afa[js][e] = (t >= tp && g2_row_ok<M>(t, cl)) ?
                        *reinterpret_cast<const short*>(p.A + g2_rm_tok(t0 + t, hh, p.H, kBT) + tp) : 0;
                }
        }
#endif
        // pass 1 stages the B operand as [j][v] (n = j, k = v -- 132-stride, v spans
        // 128); pass 2 stages [v][t] (n = v, k = t -- 68-stride).  Both fit 17.4KB.
                                                                  // pass1 [j][132] = 8448,
                                                                  // pass2 [v][68] = 8704

        // ---- pass 1: dA = do @ vT, reduction over v (8 tiles), 4 j-tile outputs
        // (dA == nullptr skips the pass -- bisection hook, bindings never pass null)
        if (p.dA != nullptr)
        #pragma unroll   // kra P-34: all loads in flight before the first wait
        for (int it_ = 0; it_ < 8; ++it_) {
            const int t = tid + 256 * it_;
            const int sb = t >> 6, c = t & 63;                    // 32 sub-blocks x 64 quads
            const v4bh vq = *reinterpret_cast<const v4bh*>(p.Vn + nblk + (size_t)sb * 256 + c * 4);
            const int d0 = sb >> 3, d1 = sb & 7, m = c & 15, q = c >> 4;
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                slab[i1(16 * d0 + m, 16 * d1 + q + 4 * e)] = vq[e];
        }
        __syncthreads();
        // kra P-938va: the staged [j][v] tile IS v_new's A form: fragment (row tile a, column tile vt) of lane (q, m) =
        // v_new[16a + m][16vt + 4q .. +3] = one 8 B slab read at the pass-1 operand position.  Re-emitted over this
        // CTA's own C-form block (std order, lane-ordered 16 B stores; every lane's Vn loads were consumed before the
        // barrier above), so s4b reads A-form fragments and skips its in-register C -> A transpose.  Same values.
        // L16 pairs (as Wn / Kn): sub-blocks 2g, 2g + 1 of lane c are one 16 B store at 512 g + 8 c.
        if (p.VnA != nullptr && p.dA != nullptr)
        #pragma unroll
        for (int it_ = 0; it_ < 4; ++it_) {
            const int t = tid + 256 * it_;
            const int g = t >> 6, c = t & 63, a = g >> 2, vt = 2 * (g & 3);
            const int r = 16 * a + (c & 15), cc = 4 * (c >> 4);
            st16x2(p.VnA + nblk + (size_t)g * 512 + c * 8, *reinterpret_cast<const v4bh*>(slab + i1(r, 16 * vt + cc)),
                   *reinterpret_cast<const v4bh*>(slab + i1(r, 16 * vt + 16 + cc)));
        }
        if (p.dA != nullptr)
        {
            f4 acc[4];
            #pragma unroll
            for (int jt = 0; jt < 4; ++jt) acc[jt] = f4{0.f, 0.f, 0.f, 0.f};
            const int rdo = 16 * wv + (l & 15);
            const bool dok = g2_row_ok<M>(rdo, cl);
#if defined(__gfx938__)
            #pragma unroll
#endif
            for (int vs = 0; vs < 8; ++vs) {                      // reduction over v
#if defined(__gfx938__)
                const v4bh df = dfa[vs];
#else
                const v4bh df = dok ? *reinterpret_cast<const v4bh*>(
                    p.do_ + g2_rm_tok(t0 + rdo, hh, p.H, kV)
                    + 16 * vs + (l >> 4) * 4) : v4bh{0, 0, 0, 0};
#endif
                #pragma unroll
                for (int jt = 0; jt < 4; ++jt) {
                    const size_t lo = i1(16 * jt + (l & 15), 16 * vs + (l >> 4) * 4);
                    acc[jt] = mmac(df, *reinterpret_cast<const v4bh*>(slab + lo), acc[jt]);
                }
            }
            // kra P-14: the same values, staged through the (now free) slab so they leave
            // as 16 B row stores instead of 16 scalar 4 B stores per lane
            __syncthreads();                                      // all waves done with Vn
            float* dAs = reinterpret_cast<float*>(slab);          // [64][kPadJ3] fp32 = slab
            #pragma unroll
            for (int jt = 0; jt < 4; ++jt) {
                const int t = 16 * wv + (l & 15), j0 = 16 * jt + (l >> 4);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {           // C-frag col = (l>>4) + 4e
                    const float val = (t >= j0 + 4 * e) ? acc[jt][e] * p.scale : 0.f;
                    dAs[iA(t, j0 + 4 * e)] = val;
                }
            }
            __syncthreads();
            #pragma unroll
            for (int it = 0; it < 4; ++it) {                      // 64 rows x 16 quads
                const int c = tid + 256 * it, t = c >> 4, j = (c & 15) * 4;
                if (!g2_row_ok<M>(t, cl)) continue;
                *reinterpret_cast<f4*>(p.dA + g2_rm_tok(t0 + t, hh, p.H, kBT) + j) =
                    *reinterpret_cast<const f4*>(dAs + iA(t, j));
            }
        }

        // ---- pass 2: dv = A^T @ do, reduction over t (4 tiles), 8 v-tile outputs.
        // doT[v][t] staged transposed; A^T fragments are gathered straight from the
        // row-major Aqk (4 scalar loads per lane, L1-hot) with the causal zeroing --
        // FLA's upper-triangle garbage must not leak in.
        __syncthreads();
        if (p.dVn != nullptr)
        if (p.aNdo == nullptr) {   // kra P-34b: production route, no dead store in the loop
        #pragma unroll   // kra P-34: all loads in flight before the first wait
        for (int it_ = 0; it_ < 8; ++it_) {
            const int t = tid + 256 * it_;
            const int j = t >> 5, vq = (t & 31) << 2;
#if defined(__gfx938__)
            const v4bh dv_ = dva[it_];
#else
            const v4bh dv_ = g2_row_ok<M>(j, cl)
                ? *reinterpret_cast<const v4bh*>(p.do_ + vbase + (size_t)j * rstride + vq)
                : v4bh{0, 0, 0, 0};
#endif
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                slab[i2(vq + d, j)] = dv_[d];
            // T3: the SAME do loads also emit P2's dOT (B form, stride-4 quads, std
            // block order) and P3's aN_do (A form, plain row nibble, P3 block order)
            // -- the two B3.5 convert2 calls (which re-read do, ~200 MB @8K/48)
            // disappear.  Index math identical to g2_wu's verified Kn/KgA emits.
        }
        } else {
        #pragma unroll   // kra P-34: all loads in flight before the first wait
        for (int it_ = 0; it_ < 8; ++it_) {
            const int t = tid + 256 * it_;
            const int j = t >> 5, vq = (t & 31) << 2;
#if defined(__gfx938__)
            const v4bh dv_ = dva[it_];
#else
            const v4bh dv_ = g2_row_ok<M>(j, cl)
                ? *reinterpret_cast<const v4bh*>(p.do_ + vbase + (size_t)j * rstride + vq)
                : v4bh{0, 0, 0, 0};
#endif
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                slab[i2(vq + d, j)] = dv_[d];
            // T3: the SAME do loads also emit P2's dOT (B form, stride-4 quads, std
            // block order) and P3's aN_do (A form, plain row nibble, P3 block order)
            // -- the two B3.5 convert2 calls (which re-read do, ~200 MB @8K/48)
            // disappear.  Index math identical to g2_wu's verified Kn/KgA emits.
            if (p.aNdo != nullptr)   // kra P-20: production route -- s4b reads do row-major
                *reinterpret_cast<v4bh*>(p.aNdo + nblk3 + (size_t)(4 * (j & 15)
                    + 64 * ((vq & 15) >> 2) + 256 * (vq >> 4) + 2048 * (j >> 4))) = dv_;
        }
        }
        __syncthreads();
        // kra P-14: dOT (B form) from the transposed staging, 16 B per store.  dOT index of
        // element (j, v) = (j&3) + 4(v&15) + 64((j>>2)&3) + 256(j>>4) + 1024(v>>4): the 8
        // shorts of chunk c are rows j0..j0+3 at columns v0 and v0+1 -- two 8 B slab reads.
        if (p.dVn != nullptr)
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int c = tid + 256 * it, x = 8 * c;
            const int j0 = 4 * ((x >> 6) & 3) + 16 * ((x >> 8) & 3);
            const int v0 = ((x >> 2) & 15) + 16 * (x >> 10);
            const v4bh lo = *reinterpret_cast<const v4bh*>(slab + i2(v0, j0));
            const v4bh hi = *reinterpret_cast<const v4bh*>(slab + i2(v0 + 1, j0));
            st16x2(p.dOT + nblk + x, lo, hi);
        }
        if (p.dVn != nullptr)
        {
            f4 acc[8];
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) acc[vt] = f4{0.f, 0.f, 0.f, 0.f};
            const int tp = 16 * wv + (l & 15);                    // this lane's t' row
#if defined(__gfx938__)
            #pragma unroll
#endif
            for (int js = 0; js < 4; ++js) {                      // reduction over t
                v4bh af;
#if defined(__gfx938__)
                af = afa[js];
#else
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int t = 16 * js + 4 * (l >> 4) + e;
                    af[e] = (t >= tp && g2_row_ok<M>(t, cl)) ?           // causal keep
                        *reinterpret_cast<const short*>(
                            p.A + g2_rm_tok(t0 + t, hh, p.H, kBT) + tp) : 0;
                }
#endif
                #pragma unroll
                for (int vt = 0; vt < 8; ++vt) {
                    const size_t lo = i2(16 * vt + (l & 15), 16 * js + (l >> 4) * 4);
                    acc[vt] = mmac(af, *reinterpret_cast<const v4bh*>(slab + lo), acc[vt]);
                }
            }
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) {                       // C-form: vec store
                v4bh x;
                #pragma unroll
                for (int e = 0; e < 4; ++e) x[e] = bf_rne(acc[vt][e]);
                *reinterpret_cast<v4bh*>(p.dVn + nblk + (size_t)(vt + 8 * wv) * 256 + l * 4) = x;
            }
        }
    });
}

}  // namespace g2

extern "C" int g2_dav(const void* do_, const void* Vn, const void* A,
                      void* dA, void* dVn, void* dOT, void* aNdo, double scale,
                      int B, int T, int H, const int* ctok, const int* clen,
                      void* VnA, hipStream_t s) {   // kra P-938va: + VnA
    // T = chunk-padded rows (NT = T/64); with ctok/clen do/A/dA rows are packed.
    if (B != 1 || T % 64 || ((ctok == nullptr) != (clen == nullptr))) return -1;
    g2::DavParams p{(const short*)do_, (const short*)Vn, (const short*)A,
                    (float*)dA, (short*)dVn, (short*)dOT, (short*)aNdo,
                    (float)scale, H, T / 64, ctok, clen};
    p.VnA = (short*)VnA;
    // kra P-938v: residency probe knob (dev only): dynamic LDS on top of the static slab caps the CTAs per CU
    const int lds_extra = (int)g2::g2_env_long("KDA_G2_DAV_LDS_EXTRA", 0);
    // kra P-938o2 (gfx938): diagonal order at H % 64 == 0 (row-major do / A against the p3-order natives)
    p.order = (int)g2::g2_env_long("KDA_G2_DAV_ORDER", g2::g2_cur_gfx938() && H % 64 == 0 ? 2 : 0);
    p.order_m = (int)g2::g2_env_long("KDA_G2_DAV_ORDER_M", 4);
    if (p.order == 3 && (p.order_m < 1 || H % p.order_m)) p.order = 0;
    if (ctok)
        hipLaunchKernelGGL(g2::g2_dav_kernel<true>, dim3(T / 64, H), dim3(256), lds_extra, s, p);
    else
        hipLaunchKernelGGL(g2::g2_dav_kernel<false>, dim3(T / 64, H), dim3(256), lds_extra, s, p);
    return (int)hipGetLastError();
}
