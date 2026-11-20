// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// G2 P1 v8: fused output + optional bN(h) emission (REVIEW S3-6 step 2 item 1).
//
// Extends g2_fwd_h_v7_kernel<8> (same recurrence, same numerics for h/v_new):
//   * o fused: each wave owns the [16x16] o block (t-tile r1, v-tile v1a) of its
//     CTA's 32-column slice; per chunk step
//         o_block = (qg @ h_i) * scale + Aqk @ v_new_i
//     with qg / Aqk supplied A-fragment-native (same layout family as Wn; Aqk is the
//     64-wide variant).  h is read from the shared hB exactly where GEMM1 reads it
//     (B operand, col=v d=k) and v_new from vB (col=v d=t) -- zero extra staging.
//     The forward pass no longer writes h and no longer calls FLA's o kernel.
//   * bN emit (backward recompute path): bN slot (kt, vt, l, e) holds
//     h[16*kt + pi(m)][16*vt + 4q + e] (DESIGN section 4; verified in
//     scripts/verify_ref.py).  After the barrier where hB holds h_i complete, wave
//     wv emits the kt=wv blocks for its CTA's two global v-tiles straight from hB,
//     replacing from_native_h + to_bN (the non-coalesced 8-D permutes that dominated
//     the S3-5 wall-clock breakdown).
//
// Register note: qg/Aqk fragment loads use dedicated arrays; the recurrence's wf[]
// (W operands) stay resident as in v7.
#include "g2_common.h"

namespace g2 {

struct FwdOParams {
    const bf16 *Wn, *Kn, *Un; const float* Gn;
    const bf16* QgA;                                   // bf16(q * 2^g), A-native (128-wide)
    const bf16* AqkA;                                  // FLA Aqk (scale in), A-native (64-wide)
    float* O;                                          // [B,T,H,128] fp32, null = skip
    bf16* Ob;                                          // T9a: [B,T,H,128] bf16, null = skip
    bf16* Vn;                                          // null = skip (forward: unused)
    bf16* HbN;                                         // bN(h), null = skip
    bf16* ANv;                                         // aN(v_new), null = skip
    float scale; int H, NT;
    // Varlen/batch (DESIGN_VARLEN_BATCH.md): chunk-aligned segments.  seg[0..nseg] are
    // chunk offsets; null = one segment [0, NT).  grid.y = nseg * H.
    const int* seg; int nseg;
    // Varlen phase 2: packed o rows (chunk i -> tokens ctok[i] .. +clen[i]); null = dense.
    const int* ctok; const int* clen;
    const float* GnRaw;                                // kra P-52: raw G_last (GE=false instance)
    // Recurrent-state ports, fp32 [nseg, H, 128, 128] (FLA's [N, H, K, V]; ST instances only):
    // segment sg starts from H0[sg] (null = zero state) and writes its final state to Ht[sg]
    // (null = skip).
    const float* H0 = nullptr;
    float* Ht = nullptr;
    // CP summary pass (v8o4, ST only): HtAux set -> grid.x doubles; the second half of the V-slab
    // CTAs carries a K x K state from H0Aux with a zero value term (the transition matrix) into HtAux.
    const float* H0Aux = nullptr;
    float* HtAux = nullptr;
};

// The state slice a wave holds: the lane (m, q4) of the wave owning key block kb keeps
// st[vt][e] = S[16 kb + kc][kBV vs + 16 vt + vr], the GEMM2 C fragment as write_h stores it:
// kc = c_col(l, e) (KP: 4 q4 + e, the lane-permuted Kn), vr = m (BP: pi16(m)).
template <bool KP = false, bool BP = false>
__device__ __forceinline__ size_t g2_state_off(int sg, int H, int bh, int kb, int vs, int vt, int l, int e) {
    const int m = l & 15, q4 = l >> 4;
    const int kc = KP ? 4 * q4 + e : c_col(l, e), vr = BP ? pi16(m) : m;
    return (((size_t)sg * H + bh) * kK + 16 * kb + kc) * kV + kBV * vs + 16 * vt + vr;
}

template <bool KP = false, bool BP = false>
__device__ __forceinline__ void g2_state_load(const float* s, int sg, int H, int bh, int kb, int vs, int l, f4* st) {
    #pragma unroll
    for (int vt = 0; vt < 2; ++vt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) st[vt][e] = s ? s[g2_state_off<KP, BP>(sg, H, bh, kb, vs, vt, l, e)] : 0.f;
}

template <bool KP = false, bool BP = false>
__device__ __forceinline__ void g2_state_store(float* s, int sg, int H, int bh, int kb, int vs, int l, const f4* st) {
    #pragma unroll
    for (int vt = 0; vt < 2; ++vt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) s[g2_state_off<KP, BP>(sg, H, bh, kb, vs, vt, l, e)] = st[vt][e];
}

// T9a: wave-scope LDS ordering -- wave-private oStage regions need the compiler
// fence + lgkmcnt drain only, NOT a CTA barrier (S3-16 lesson: a barrier-carrying
// wave sync deadlocks when called from wave-divergent code; here all lanes of a
// wave participate, but s_barrier would still serialize the 8 waves needlessly).
__device__ __forceinline__ void o_wave_order() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

// ST: compile the H0/Ht ports in (false: zero start state, no final state -- the stateless
// instance stays identical to the pre-port kernel; the runtime-null form cost v8o4 ~2%).
template <bool VL, bool ST = false>
__global__ void __launch_bounds__(512) g2_fwd_h_v8o_kernel(FwdOParams p) {
    __shared__ __align__(16) bf16 hB[kBV * kPadK];
    __shared__ __align__(16) bf16 vB[kBV * kPadT];
    __shared__ __align__(16) bf16 oStage[8][256];   // T9a: wave-private 16x16 bf16
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int vs = blockIdx.x;
    const int sg = blockIdx.y / p.H, bh = blockIdx.y - sg * p.H;
    const int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;
    f4 ht[2];
    if constexpr (ST) g2_state_load(p.H0, sg, p.H, bh, wv, vs, l, ht);
    else { ht[0] = f4{0.f,0.f,0.f,0.f}; ht[1] = ht[0]; }
    if (c0 >= c1) {                                    // empty segment (whole CTA exits): S_final = h0
        if constexpr (ST) if (p.Ht != nullptr) g2_state_store(p.Ht, sg, p.H, bh, wv, vs, l, ht);
        return;
    }
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv & 3, v1a = wv >> 2;
    const int m = l & 15, q4 = l >> 4;
    const size_t rowT = (size_t)p.H * kK;               // token stride of [T,H,128]
    const size_t tok0 = chunk * kBT * rowT;             // chunk row 0, whole [H,128] block

    v4bh wf[8], kf[4], uf, qgf[8], aqf[4];
    f4 gl;

    auto loadW = [&](int i) {
        const bf16* src = p.Wn + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, wf[2 * j], wf[2 * j + 1]);
    };
    auto loadQg = [&](int i) {
        const bf16* src = p.QgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, qgf[2 * j], qgf[2 * j + 1]);
    };
    auto loadAq = [&](int i) {
        const bf16* src = p.AqkA + (((chunk + i) * 4 + r1) * 4) * 256 + l * 8;   // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 2; ++j) ld16x2(src + j * 512, aqf[2 * j], aqf[2 * j + 1]);
    };
    auto loadU = [&](int i) {
        uf = ld8(p.Un + nat16((((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a) * 256 + l * 4));
    };
    auto loadG = [&](int i) {
        gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + wv) * 16 + q4 * 4);
    };
    auto loadK = [&](int i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j)                      // kra L16: paired
            ld16x2(p.Kn + (((chunk + i) * 8 + wv) * 4) * 256 + j * 512 + l * 8, kf[2 * j], kf[2 * j + 1]);
    };
    v4bh hx[2];
    auto write_h = [&]() {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne_int(ht[vt][e]);
                reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * wv + c_col(l, e)] = hx[vt][e];
            }
    };

    loadW(c0); loadU(c0); loadG(c0); loadK(c0);
    write_h();
    for (int i = c0; i < c1; ++i) {                    // h = h0 (default 0) at the segment start
        // varlen: issue the chunk's row-map scalar loads up front (latency hidden by the
        // GEMMs; issued at the o store they sat on the per-step critical path)
        const long otok = g2_row_tok<VL>(p.ctok, i);
        const int olen = g2_row_len<VL>(p.clen, i);
        const int nxt = i + 1 < c1 ? i + 1 : i;
        lds_barrier();                                   // hB (h_i) ready
        // ---- optional bN(h_i) emit: wave wv writes blocks kt=wv, vt in {2vs, 2vs+1}
        if (p.HbN != nullptr) {
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                // slot (l, e) = h[16*wv + pi(m)][16*(2vs+vt) + 4*q4 + e]
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(hB)[(16 * vt + 4 * q4 + e) * kPadK + 16 * wv + pi16(m)];
                // bN block order is P3's (B,NT,H): block = i*H + bh (B=1)
                const size_t off = ((((size_t)i * p.H + bh) * 8 + wv) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *reinterpret_cast<v4bh*>(p.HbN + off) = val;
            }
        }
        // ---- GEMM1: v_new = u - W @ h_i   (identical to v7)
        f4 acc = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            acc = mmac(wf[kt], ld8(hB + (16 * v1a + m) * kPadK + 16 * kt + q4 * 4), acc);
        loadW(nxt);
        {
            v4bh x;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float uv = __uint_as_float(((uint32_t)(unsigned short)uf[e]) << 16);
                x[e] = bf_rne_int(uv - acc[e]);
                reinterpret_cast<short*>(vB)[(16 * v1a + c_col(l, e)) * kPadT + 16 * r1 + m] = x[e];
            }
            if (p.Vn != nullptr)                         // forward: Vn unused (null)
                *reinterpret_cast<v4bh*>(p.Vn + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a) * 256 + l * 4) = x;
            if (p.ANv != nullptr)   // aN(v_new): A form, P3 block order; the C-frag
                #pragma unroll      // col (l>>4)+4e swaps into 64*e + (l>>4)
                for (int e = 0; e < 4; ++e)
                    reinterpret_cast<short*>(p.ANv)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1a) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x[e];
        }
        loadU(nxt);
        {
            f4 d;
            #pragma unroll
            for (int e = 0; e < 4; ++e) d[e] = g2_vcopy(gl[e]);   // kra P-52: Gn = exp2f(G)
            ht[0] *= d; ht[1] *= d;
        }
        loadG(nxt);
        // ---- fused output, part 1: oh = qg @ h_i  BEFORE the second barrier.  hB is
        // written only by write_h at loop end, so the second barrier itself protects
        // these reads exactly the way it protects GEMM1's; after the barrier a fast
        // wave's write_h would race a slow wave's hB reads (h_1 leaking into o_i -- the
        // single-chunk test catches it: h_0 = 0 yet o came out 0.017).
        f4 oh = f4{0.f,0.f,0.f,0.f};
        if (p.O != nullptr || p.Ob != nullptr) {
            loadQg(i); loadAq(i);
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt)
                oh = mmac(qgf[kt], ld8(hB + (16 * v1a + m) * kPadK + 16 * kt + q4 * 4), oh);
        }
        lds_barrier();                                   // vB (v_new_i) ready
        // ---- fused output, part 2: oa = Aqk @ v_new_i (vB reads only: the next vB
        // writer is the next iteration's scatter, which sits behind the next barrier)
        if (p.O != nullptr || p.Ob != nullptr) {
            f4 oa = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int st = 0; st < 4; ++st)
                oa = mmac(aqf[st], ld8(vB + (16 * v1a + m) * kPadT + 16 * st + q4 * 4), oa);
            // o = (qg @ h) * scale + Aqk @ v_new (FLA chunk_gla_fwd_kernel_o:
            // b_o *= scale BEFORE += dot(A, v); Aqk carries scale from fwd_intra).
            // T9a: the bf16 path consumes the SAME fp32 value (fusion status of
            // this mul+add must not flip -- R-bitwise gate arbitrates).
            f4 o;
            #pragma unroll
            for (int e = 0; e < 4; ++e) o[e] = oh[e] * p.scale + oa[e];
            // C fragment: row = 16*r1 + m (token), col = 16*v1a + q4 + 4e (within slice)
            const int orow = 16 * r1 + m;                 // row within the chunk
            const bool ook = g2_row_ok<VL ? 2 : 0>(orow, olen);   // masked mode
            const size_t tok = (size_t)(otok + orow);
            const size_t obase = tok * rowT + (size_t)bh * kK + 32 * vs + 16 * v1a;
            if (p.O != nullptr) {
                if (ook) {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        p.O[obase + q4 + 4 * e] = o[e];
                }
            } else {
                // T9a: bf16 o via the wave-private LDS transpose.  Lane (m, q4)
                // scatters its 4 strided C-fragment columns into oStage[wv]; after
                // the wave-scope ordering, lane (m', g') reads the 4 CONSECUTIVE
                // columns as one 8B quad and stores one 8B vector to global
                // (4 scalar 2B stores are forbidden by the task).
                bf16* st = oStage[wv];
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    reinterpret_cast<short*>(st)[m * 16 + q4 + 4 * e] = bf_rne_int(o[e]);
                o_wave_order();
                const int m2 = l & 15, g2_ = l >> 4;
                const v4bh ov = *reinterpret_cast<const v4bh*>(
                        reinterpret_cast<const short*>(st) + m2 * 16 + 4 * g2_);
                if (ook) *reinterpret_cast<v4bh*>(p.Ob + obase + 4 * g2_) = ov;
            }
        }
        // ---- GEMM2: h update (identical to v7)
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 upd = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + m) * kPadT + 16 * rt + q4 * 4);
                upd = mmac(va, kf[rt], upd);
            }
            ht[vt] += upd;
        }
        loadK(nxt);
        write_h();
    }
    if constexpr (ST) if (p.Ht != nullptr) g2_state_store(p.Ht, sg, p.H, bh, wv, vs, l, ht);
}


// ================================================================ R1a: 4-wave v8o
// KDA_G2_R1A=1 (default): 256 threads / 4 waves.  Wave w owns GEMM1 tiles (r1=w, both
// v1a halves) and GEMM2 K-blocks {w, w+4}, so W/QgA/AqkA are loaded ONCE per CTA per
// step (the 8-wave form loads them twice -- v1a=0/1 waves with equal r1 read the same
// rows).  Every output tile's MMAC operand order is unchanged; only the owning wave and
// the load multiplicity differ -> all outputs bit-identical (bench-verified).  LDS also
// halves (oStage 4x256 instead of 8x256); runtime occupancy 3 -> 4 CTAs/CU.
__device__ __forceinline__ void r1a_o_wave_order() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

template <bool VL, int MODE = 0, bool GE = true, bool SG = false, bool PF = false, bool KP = false, bool VP = false, bool BP = false, bool ST = false>   // ST: as g2_fwd_h_v8o_kernel   // kra P-58: SG = SGPR wave index   // kra P-52: GE = Gn holds exp2f(G)
__global__ void __launch_bounds__(256) g2_fwd_h_v8o4_kernel(FwdOParams p) {
    // kra P-9: compile-time outputs.  With runtime-optional VMEM ops in the loop the compiler's
    // waitcnt pass must count the path that skips them, so a vmcnt for an OLD load also forced
    // younger, actually-issued loads to land (v8o4 fwd: one GEMM1 wait = 15.7% of the step).
    // MODE 0 generic | 1 fwd (Ob) | 2 fwd keep-state (Ob+HbN+Vn+ANv) | 3 bwd recompute (HbN+Vn+ANv)
    // | 4 bwd recompute without the aN copy (HbN+Vn; s4b rebuilds aN from Vn) [kra P-22]
    // | 6 state only (no output: the CP summary recurrence)
    const bool hasO  = MODE == 0 ? p.O != nullptr : false;
    const bool hasOb = MODE == 0 ? p.Ob != nullptr : (MODE != 3 && MODE != 4 && MODE != 6);
    const bool hasBN = MODE == 0 ? p.HbN != nullptr : (MODE != 1 && MODE != 6);
    const bool hasVn = MODE == 0 ? p.Vn != nullptr : (MODE != 1 && MODE != 5 && MODE != 6);   // kra P-82: MODE 5 = Ob + HbN
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : (MODE != 1 && MODE != 4 && MODE != 5 && MODE != 6);
    __shared__ __align__(16) bf16 hB[kBV * kPadK];
    __shared__ __align__(16) bf16 vB[kBV * kPadT];
    __shared__ __align__(16) bf16 oStage[4][256];
    const int tid = threadIdx.x, wv = SG ? __builtin_amdgcn_readfirstlane(tid >> 6) : (tid >> 6), l = tid & 63;   // kra P-58   // wv in 0..3
    const bool ax = ST && p.HtAux != nullptr && blockIdx.x >= kK / kBV;   // transition-matrix slab
    const int vs = ax ? blockIdx.x - kK / kBV : blockIdx.x;
    const float* H0 = ax ? p.H0Aux : p.H0;
    float* Ht = ax ? p.HtAux : p.Ht;
    const int sg = blockIdx.y / p.H, bh = blockIdx.y - sg * p.H;
    const int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;
    const int kb0 = wv, kb1 = wv + 4;        // GEMM2 K-blocks
    f4 ht0[2], ht1[2];
    if constexpr (ST) {
        g2_state_load<KP, BP>(H0, sg, p.H, bh, kb0, vs, l, ht0);
        g2_state_load<KP, BP>(H0, sg, p.H, bh, kb1, vs, l, ht1);
    } else {
        ht0[0] = f4{0.f,0.f,0.f,0.f}; ht0[1] = ht0[0];
        ht1[0] = ht0[0];              ht1[1] = ht0[0];
    }
    if (c0 >= c1) {                          // empty segment: S_final = h0
        if (ST && Ht != nullptr) {
            g2_state_store<KP, BP>(Ht, sg, p.H, bh, kb0, vs, l, ht0);
            g2_state_store<KP, BP>(Ht, sg, p.H, bh, kb1, vs, l, ht1);
        }
        return;
    }
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv;                       // GEMM1/o T-tile
    const int m = l & 15, q4 = l >> 4;
    const size_t rowT = (size_t)p.H * kK;
    const size_t tok0 = chunk * kBT * rowT;

    v4bh wf[8], kf0[4], kf1[4], uf0, uf1, qgf[8], aqf[4];
    f4 gl0, gl1;

    auto loadW = [&](int i) {
        const bf16* src = p.Wn + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, wf[2 * j], wf[2 * j + 1]);
    };
    auto loadQg = [&](int i) {
        const bf16* src = p.QgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, qgf[2 * j], qgf[2 * j + 1]);
    };
    auto loadAq = [&](int i) {
        const bf16* src = p.AqkA + (((chunk + i) * 4 + r1) * 4) * 256 + l * 8;   // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 2; ++j) ld16x2(src + j * 512, aqf[2 * j], aqf[2 * j + 1]);
    };
    auto loadU = [&](int i, int v1a_, v4bh& uf) {
        // kra L16: uf0/uf1 are one 16 B pair -- loaded on the v1a_ == 0 call (always paired)
        if (ax) {                     // zero value term: the state carried is the transition matrix
            if (v1a_ == 0) { uf0 = v4bh{0, 0, 0, 0}; uf1 = uf0; }
        } else if (v1a_ == 0) ld16x2(p.Un + (((chunk + i) * 4 + r1) * 8 + 2 * vs) * 256 + l * 8, uf0, uf1);
        (void)uf;
    };
    auto loadG = [&](int i, int kb, f4& gl) {
        gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + kb) * 16 + q4 * 4);
    };
    auto loadK = [&](int i, int kb, v4bh* kf) {
        #pragma unroll
        for (int j = 0; j < 2; ++j)                      // kra L16: paired
            ld16x2(p.Kn + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, kf[2 * j], kf[2 * j + 1]);
    };
    v4bh hx[2];
    // kra P-63: selector A fragment S[m][k] = [k == pi16(m)] (bf16 1.0 / 0), lane (q4, m) holds k = 4 q4 + j
    v4bh bsel;
    #pragma unroll
    for (int j = 0; j < 4; ++j) bsel[j] = (4 * q4 + j == pi16(m)) ? (short)0x3f80 : (short)0;
    const int hrow = BP ? pi16(m) : m;   // kra P-63: GEMM2 / hB row of this lane
    v4bh bnv[2][2];   // kra P-63c: bN(h) values [kb0/kb1][vt], stored after the next barrier A
    auto write_h = [&](int kb, f4* ht, int ib = -1) {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne_int(ht[vt][e]);
                if constexpr (!KP)
                    reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = hx[vt][e];
            }
            if constexpr (KP)   // kra P-61e: Kn lane-permuted -> lane holds k = 4 q4 + e (consecutive)
                *reinterpret_cast<v4bh*>(hB + (16 * vt + hrow) * kPadK + 16 * kb + 4 * q4) = hx[vt];
            if constexpr (BP) {   // kra P-63: bN(h) slot straight from the registers
                if (hasBN && ib >= 0) {
                    const f4 nz = f4{-0.f, -0.f, -0.f, -0.f};
                    const f4 c = mmac(bsel, hx[vt], nz);
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)   // kra P-63d: exact bf16 values -> high half
                        bnv[kb == kb0 ? 0 : 1][vt][e] = (short)(__float_as_uint(c[e]) >> 16);
                }
            }
        }
    };
    auto bN_emit = [&](int i2, int kb) {
        if (hasBN) {
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(hB)[(16 * vt + 4 * q4 + e) * kPadK + 16 * kb + pi16(m)];
                const size_t off = ((((size_t)i2 * p.H + bh) * 8 + kb) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *reinterpret_cast<v4bh*>(p.HbN + off) = val;
            }
        }
    };

    loadW(c0); loadU(c0, 0, uf0); loadU(c0, 1, uf1);
    loadG(c0, kb0, gl0); loadG(c0, kb1, gl1);
    loadK(c0, kb0, kf0); loadK(c0, kb1, kf1);
    write_h(kb0, ht0, c0); write_h(kb1, ht1, c0);   // kra P-63: BP emits bN(h_c0) here
    if constexpr (PF) { loadQg(c0); loadAq(c0); }   // kra P-60: QgA/AqkA one step ahead
    for (int i = c0; i < c1; ++i) {
        const long otok = g2_row_tok<VL>(p.ctok, i);
        const int olen = g2_row_len<VL>(p.clen, i);
        const int nxt = i + 1 < c1 ? i + 1 : i;
        lds_barrier();                                   // hB (h_i) ready
        if (!PF && (hasO || hasOb)) {         // R3: issue this step's
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
            loadQg(i); loadAq(i);                        // QgA/AqkA right after bar1
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        }
        if constexpr (!BP) {   // kra P-63: BP emitted bN(h_i) at the end of the previous step
            bN_emit(i, kb0);
            bN_emit(i, kb1);
        } else if (hasBN) {    // kra P-63c: production store position, values from the registers
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk)
                #pragma unroll
                for (int vt = 0; vt < 2; ++vt) {
                    const size_t off = ((((size_t)i * p.H + bh) * 8 + (kk ? kb1 : kb0)) * 8 + (2 * vs + vt)) * 256 + l * 4;
                    *reinterpret_cast<v4bh*>(p.HbN + off) = bnv[kk][vt];
                }
        }
        // ---- GEMM1 both V-halves with ONE W load (the R1a dedup)
        f4 acc0 = f4{0.f,0.f,0.f,0.f}, acc1 = f4{0.f,0.f,0.f,0.f};
        v4bh hbf0[8], hbf1[8];   // kra P-53: h fragments kept for the o pass (hB is unchanged until write_h)
        if constexpr (KP) {   // kra P-65: all 16 hB fragments first, then the MMAC chain
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                hbf0[kt] = ld8(hB + (0 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);
                hbf1[kt] = ld8(hB + (1 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);
            }
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                acc0 = mmac(wf[kt], hbf0[kt], acc0);
                acc1 = mmac(wf[kt], hbf1[kt], acc1);
            }
        } else {
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            hbf0[kt] = ld8(hB + (0 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);   // kra P-62
            hbf1[kt] = ld8(hB + (1 * 16 + (VP ? pi16(m) : m)) * kPadK + 16 * kt + q4 * 4);
            acc0 = mmac(wf[kt], hbf0[kt], acc0);
            acc1 = mmac(wf[kt], hbf1[kt], acc1);
        }
        }
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        loadW(nxt);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        {
            v4bh x0, x1;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float u0 = __uint_as_float(((uint32_t)(unsigned short)uf0[e]) << 16);
                const float u1 = __uint_as_float(((uint32_t)(unsigned short)uf1[e]) << 16);
                x0[e] = bf_rne_int(u0 - acc0[e]);
                x1[e] = bf_rne_int(u1 - acc1[e]);
                const int vc = VP ? 4 * q4 + e : c_col(l, e);   // kra P-62: lane's v columns
                reinterpret_cast<short*>(vB)[(16 * 0 + vc) * kPadT + 16 * r1 + m] = x0[e];
                reinterpret_cast<short*>(vB)[(16 * 1 + vc) * kPadT + 16 * r1 + m] = x1[e];
            }
            if constexpr (!VP) {
            if (hasVn) {                       // forward: Vn unused (null)
                *reinterpret_cast<v4bh*>(p.Vn + (((chunk + i) * 4 + r1) * 8 + 2 * vs + 0) * 256 + l * 4) = x0;
                *reinterpret_cast<v4bh*>(p.Vn + (((chunk + i) * 4 + r1) * 8 + 2 * vs + 1) * 256 + l * 4) = x1;
            }
            if (hasAN)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    reinterpret_cast<short*>(p.ANv)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + 0) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x0[e];
                    reinterpret_cast<short*>(p.ANv)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + 1) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x1[e];
                }
            }
        }
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        loadU(nxt, 0, uf0); loadU(nxt, 1, uf1);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        {
            f4 d0, d1;
            if constexpr (GE) {   // kra P-52: decays precomputed
                #pragma unroll
                for (int e = 0; e < 4; ++e) { d0[e] = g2_vcopy(gl0[e]); d1[e] = g2_vcopy(gl1[e]); }
            } else {               // production instance (raw G_last)
                #pragma unroll
                for (int e = 0; e < 4; ++e) { d0[e] = exp2f(gl0[e]); d1[e] = exp2f(gl1[e]); }
            }
            ht0[0] *= d0; ht0[1] *= d0;
            ht1[0] *= d1; ht1[1] *= d1;
        }
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        loadG(nxt, kb0, gl0); loadG(nxt, kb1, gl1);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        // ---- oh for both tiles (QgA/AqkA loaded ONCE)
        f4 oh0 = f4{0.f,0.f,0.f,0.f}, oh1 = f4{0.f,0.f,0.f,0.f};
        if (hasO || hasOb) {
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                oh0 = mmac(qgf[kt], hbf0[kt], oh0);   // kra P-53: GEMM1's fragments
                oh1 = mmac(qgf[kt], hbf1[kt], oh1);
            }
        }
        if constexpr (PF) { if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0); loadQg(nxt); if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0); }   // kra P-60: qgf's last use was the oh pass
        lds_barrier();                                   // vB (v_new_i) ready
        if constexpr (VP) if (hasVn || hasAN) {   // kra P-62: old-layout Vn/ANv fragments from vB
            v4bh x0, x1;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                x0[e] = reinterpret_cast<const short*>(vB)[(16 * 0 + c_col(l, e)) * kPadT + 16 * r1 + m];
                x1[e] = reinterpret_cast<const short*>(vB)[(16 * 1 + c_col(l, e)) * kPadT + 16 * r1 + m];
            }
            if (hasVn) {                       // forward: Vn unused (null)
                *reinterpret_cast<v4bh*>(p.Vn + (((chunk + i) * 4 + r1) * 8 + 2 * vs + 0) * 256 + l * 4) = x0;
                *reinterpret_cast<v4bh*>(p.Vn + (((chunk + i) * 4 + r1) * 8 + 2 * vs + 1) * 256 + l * 4) = x1;
            }
            if (hasAN)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    reinterpret_cast<short*>(p.ANv)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + 0) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x0[e];
                    reinterpret_cast<short*>(p.ANv)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + 1) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x1[e];
                }
        }
        if (hasO || hasOb) {
            #pragma unroll
            for (int v1a = 0; v1a < 2; ++v1a) {
                const f4& oh = v1a ? oh1 : oh0;
                f4 oa = f4{0.f,0.f,0.f,0.f};
                #pragma unroll
                for (int st = 0; st < 4; ++st)
                    oa = mmac(aqf[st], ld8(vB + (16 * v1a + (VP ? pi16(m) : m)) * kPadT + 16 * st + q4 * 4), oa);   // kra P-62
                f4 o;
                #pragma unroll
                for (int e = 0; e < 4; ++e) o[e] = oh[e] * p.scale + oa[e];
                const int orow = 16 * r1 + m;
                const bool ook = g2_row_ok<VL ? 2 : 0>(orow, olen);
                const size_t tok = (size_t)(otok + orow);
                const size_t obase = tok * rowT + (size_t)bh * kK + 32 * vs + 16 * v1a;
                if constexpr (VP) {   // kra P-62: lane holds v = 4 q4 + e -> direct stores, no oStage
                    if (ook) {
                        if (hasO) {
                            *reinterpret_cast<f4*>(p.O + obase + 4 * q4) = o;
                        } else {
                            v4bh ov;
                            #pragma unroll
                            for (int e = 0; e < 4; ++e) ov[e] = bf_rne_int(o[e]);
                            *reinterpret_cast<v4bh*>(p.Ob + obase + 4 * q4) = ov;
                        }
                    }
                } else if (hasO) {
                    if (ook) {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e)
                            p.O[obase + q4 + 4 * e] = o[e];
                    }
                } else {
                    bf16* st_ = oStage[wv];
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        reinterpret_cast<short*>(st_)[m * 16 + q4 + 4 * e] = bf_rne_int(o[e]);
                    r1a_o_wave_order();
                    const int m2 = l & 15, g2_ = l >> 4;
                    const v4bh ov = *reinterpret_cast<const v4bh*>(
                            reinterpret_cast<const short*>(st_) + m2 * 16 + 4 * g2_);
                    if (ook) *reinterpret_cast<v4bh*>(p.Ob + obase + 4 * g2_) = ov;
                }
            }
        }
        if constexpr (PF) { if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0); loadAq(nxt); if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0); }   // kra P-60: aqf's last use was the o pass
        // ---- GEMM2 for both K-blocks
        if constexpr (KP) {   // kra P-65: the 8 vB fragments (shared by both K-blocks) first, then the MMACs
            v4bh vaf[2][4];
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt)
                #pragma unroll
                for (int rt = 0; rt < 4; ++rt)
                    vaf[vt][rt] = ld8(vB + (16 * vt + hrow) * kPadT + 16 * rt + q4 * 4);
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                f4 upd = f4{0.f,0.f,0.f,0.f};
                #pragma unroll
                for (int rt = 0; rt < 4; ++rt) upd = mmac(vaf[vt][rt], kf0[rt], upd);
                ht0[vt] += upd;
            }
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                f4 upd = f4{0.f,0.f,0.f,0.f};
                #pragma unroll
                for (int rt = 0; rt < 4; ++rt) upd = mmac(vaf[vt][rt], kf1[rt], upd);
                ht1[vt] += upd;
            }
        } else {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 upd = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + hrow) * kPadT + 16 * rt + q4 * 4);   // kra P-63
                upd = mmac(va, kf0[rt], upd);
            }
            ht0[vt] += upd;
        }
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 upd = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + hrow) * kPadT + 16 * rt + q4 * 4);   // kra P-63
                upd = mmac(va, kf1[rt], upd);
            }
            ht1[vt] += upd;
        }
        }
        if constexpr (!PF) {
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
            loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);   // kra P-11: keep load spacing
        }
        write_h(kb0, ht0, i + 1 < c1 ? i + 1 : -1);   // kra P-63: h_{i+1} -> bN(h_{i+1})
        write_h(kb1, ht1, i + 1 < c1 ? i + 1 : -1);
        if constexpr (PF) {   // kra P-64B: K prefetch after write_h
            __builtin_amdgcn_sched_barrier(0);
            loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);
        }
    }
    if constexpr (ST) if (Ht != nullptr) {
        g2_state_store<KP, BP>(Ht, sg, p.H, bh, kb0, vs, l, ht0);
        g2_state_store<KP, BP>(Ht, sg, p.H, bh, kb1, vs, l, ht1);
    }
}

// g2_fwd_h_v8o4_kernel's instance for a template-argument set, without (ST = false) or with the
// state ports: the launcher picks the same instance either way and only appends ST.
template <bool VL, int MODE = 0, bool GE = true, bool SG = false, bool PF = false, bool KP = false, bool VP = false, bool BP = false>
struct V8o4K {
    template <bool ST> static constexpr auto get() { return g2_fwd_h_v8o4_kernel<VL, MODE, GE, SG, PF, KP, VP, BP, ST>; }
};


// kra P-49: g2_fwd_h_v8o4_kernel with NVH 16-column V-tiles per CTA (NVH = 4: 64 columns, 2 CTAs/head)
// ST: the H0 / Ht state ports, as in g2_fwd_h_v8o4_kernel (false: the pre-port kernel).  A wave's lane (m, q4) of
// key block kb holds ht[vt][e] = S[16 kb + c_col(l, e)][16 NVH vs + 16 vt + m], as write_h stores it.
template <int NVH>
__device__ __forceinline__ size_t g2w_state_off(int sg, int H, int bh, int kb, int vs, int vt, int l, int e) {
    return (((size_t)sg * H + bh) * kK + 16 * kb + c_col(l, e)) * kV + 16 * NVH * vs + 16 * vt + (l & 15);
}

template <bool VL, int MODE = 0, int NVH = 4, bool ST = false>
__global__ void __launch_bounds__(256) g2_fwd_h_v8o4w_kernel(FwdOParams p) {
    const bool hasO  = MODE == 0 ? p.O != nullptr : false;
    const bool hasOb = MODE == 0 ? p.Ob != nullptr : (MODE != 3 && MODE != 4 && MODE != 6);
    const bool hasBN = MODE == 0 ? p.HbN != nullptr : (MODE != 1 && MODE != 6);   // MODE 6: state only, as v8o4
    const bool hasVn = MODE == 0 ? p.Vn != nullptr : (MODE != 1 && MODE != 5 && MODE != 6);   // kra P-82: MODE 5 = Ob + HbN
    const bool hasAN = MODE == 0 ? p.ANv != nullptr : (MODE != 1 && MODE != 4 && MODE != 5 && MODE != 6);
    constexpr int BVW = 16 * NVH;
    __shared__ __align__(16) bf16 hB[BVW * kPadK];
    __shared__ __align__(16) bf16 vB[BVW * kPadT];
    __shared__ __align__(16) bf16 oStage[4][256];
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int vs = blockIdx.x;
    const int sg = blockIdx.y / p.H, bh = blockIdx.y - sg * p.H;
    const int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;
    if constexpr (!ST) { if (c0 >= c1) return; }
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv;
    const int kb0 = wv, kb1 = wv + 4;
    const int m = l & 15, q4 = l >> 4;
    const size_t rowT = (size_t)p.H * kK;

    f4 ht0[NVH], ht1[NVH];
    #pragma unroll
    for (int v = 0; v < NVH; ++v) { ht0[v] = f4{0.f,0.f,0.f,0.f}; ht1[v] = ht0[v]; }
    if constexpr (ST) {
        if (p.H0 != nullptr) {
            #pragma unroll
            for (int vt = 0; vt < NVH; ++vt)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    ht0[vt][e] = p.H0[g2w_state_off<NVH>(sg, p.H, bh, kb0, vs, vt, l, e)];
                    ht1[vt][e] = p.H0[g2w_state_off<NVH>(sg, p.H, bh, kb1, vs, vt, l, e)];
                }
        }
        if (c0 >= c1) {                          // empty segment: S_final = h0
            if (p.Ht != nullptr) {
                #pragma unroll
                for (int vt = 0; vt < NVH; ++vt)
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        p.Ht[g2w_state_off<NVH>(sg, p.H, bh, kb0, vs, vt, l, e)] = ht0[vt][e];
                        p.Ht[g2w_state_off<NVH>(sg, p.H, bh, kb1, vs, vt, l, e)] = ht1[vt][e];
                    }
            }
            return;
        }
    }
    v4bh wf[8], kf0[4], kf1[4], uf[NVH], qgf[8], aqf[4];
    f4 gl0, gl1;

    auto loadW = [&](int i) {
        const bf16* src = p.Wn + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, wf[2 * j], wf[2 * j + 1]);
    };
    auto loadQg = [&](int i) {
        const bf16* src = p.QgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, qgf[2 * j], qgf[2 * j + 1]);
    };
    auto loadAq = [&](int i) {
        const bf16* src = p.AqkA + (((chunk + i) * 4 + r1) * 4) * 256 + l * 8;
        #pragma unroll
        for (int j = 0; j < 2; ++j) ld16x2(src + j * 512, aqf[2 * j], aqf[2 * j + 1]);
    };
    auto loadU = [&](int i) {                         // paired V-tiles (NVH*vs + 2pr, +1)
        if constexpr (NVH == 1) {   // kra P-938z: one 16-column V-tile, the even / odd half of its pair's 16 B lane quad
            uf[0] = ld8(p.Un + (((chunk + i) * 4 + r1) * 8 + (vs & ~1)) * 256 + l * 8 + (vs & 1) * 4);
        } else {
        #pragma unroll
        for (int pr = 0; pr < NVH / 2; ++pr)
            ld16x2(p.Un + (((chunk + i) * 4 + r1) * 8 + NVH * vs + 2 * pr) * 256 + l * 8, uf[2 * pr], uf[2 * pr + 1]);
        }
    };
    auto loadG = [&](int i, int kb, f4& gl) {
        gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + kb) * 16 + q4 * 4);
    };
    auto loadK = [&](int i, int kb, v4bh* kf) {
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            ld16x2(p.Kn + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, kf[2 * j], kf[2 * j + 1]);
    };
    auto write_h = [&](int kb, f4* ht) {
        #pragma unroll
        for (int vt = 0; vt < NVH; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                reinterpret_cast<short*>(hB)[(16 * vt + m) * kPadK + 16 * kb + c_col(l, e)] = bf_rne_int(ht[vt][e]);
    };
    auto bN_emit = [&](int i2, int kb) {
        if (hasBN) {
            #pragma unroll
            for (int vt = 0; vt < NVH; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(hB)[(16 * vt + 4 * q4 + e) * kPadK + 16 * kb + pi16(m)];
                const size_t off = ((((size_t)i2 * p.H + bh) * 8 + kb) * 8 + (NVH * vs + vt)) * 256 + l * 4;
                *reinterpret_cast<v4bh*>(p.HbN + off) = val;
            }
        }
    };

    loadW(c0); loadU(c0);
    loadG(c0, kb0, gl0); loadG(c0, kb1, gl1);
    loadK(c0, kb0, kf0); loadK(c0, kb1, kf1);
    write_h(kb0, ht0); write_h(kb1, ht1);
    for (int i = c0; i < c1; ++i) {
        const long otok = g2_row_tok<VL>(p.ctok, i);
        const int olen = g2_row_len<VL>(p.clen, i);
        const int nxt = i + 1 < c1 ? i + 1 : i;
        lds_barrier();                                   // hB (h_i) ready
        if (hasO || hasOb) {
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
            loadQg(i); loadAq(i);
            if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        }
        bN_emit(i, kb0);
        bN_emit(i, kb1);
        f4 acc[NVH];
        #pragma unroll
        for (int v = 0; v < NVH; ++v) acc[v] = f4{0.f,0.f,0.f,0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            #pragma unroll
            for (int v = 0; v < NVH; ++v)
                acc[v] = mmac(wf[kt], ld8(hB + (16 * v + m) * kPadK + 16 * kt + q4 * 4), acc[v]);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        loadW(nxt);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (int v = 0; v < NVH; ++v) {
            v4bh x;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float u = __uint_as_float(((uint32_t)(unsigned short)uf[v][e]) << 16);
                x[e] = bf_rne_int(u - acc[v][e]);
                reinterpret_cast<short*>(vB)[(16 * v + c_col(l, e)) * kPadT + 16 * r1 + m] = x[e];
            }
            if (hasVn)
                *reinterpret_cast<v4bh*>(p.Vn + (((chunk + i) * 4 + r1) * 8 + NVH * vs + v) * 256 + l * 4) = x;
            if (hasAN)
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    reinterpret_cast<short*>(p.ANv)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + NVH * vs + v) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x[e];
        }
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        loadU(nxt);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        {
            f4 d0, d1;
            #pragma unroll
            for (int e = 0; e < 4; ++e) { d0[e] = g2_vcopy(gl0[e]); d1[e] = g2_vcopy(gl1[e]); }   // kra P-52: Gn = exp2f(G)
            #pragma unroll
            for (int v = 0; v < NVH; ++v) { ht0[v] *= d0; ht1[v] *= d1; }
        }
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        loadG(nxt, kb0, gl0); loadG(nxt, kb1, gl1);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        f4 oh[NVH];
        #pragma unroll
        for (int v = 0; v < NVH; ++v) oh[v] = f4{0.f,0.f,0.f,0.f};
        if (hasO || hasOb) {
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt)
                #pragma unroll
                for (int v = 0; v < NVH; ++v)
                    oh[v] = mmac(qgf[kt], ld8(hB + (16 * v + m) * kPadK + 16 * kt + q4 * 4), oh[v]);
        }
        lds_barrier();                                   // vB (v_new_i) ready
        if (hasO || hasOb) {
            #pragma unroll
            for (int v1a = 0; v1a < NVH; ++v1a) {
                f4 oa = f4{0.f,0.f,0.f,0.f};
                #pragma unroll
                for (int st = 0; st < 4; ++st)
                    oa = mmac(aqf[st], ld8(vB + (16 * v1a + m) * kPadT + 16 * st + q4 * 4), oa);
                f4 o;
                #pragma unroll
                for (int e = 0; e < 4; ++e) o[e] = oh[v1a][e] * p.scale + oa[e];
                const int orow = 16 * r1 + m;
                const bool ook = g2_row_ok<VL ? 2 : 0>(orow, olen);
                const size_t tok = (size_t)(otok + orow);
                const size_t obase = tok * rowT + (size_t)bh * kK + BVW * vs + 16 * v1a;
                if (hasO) {
                    if (ook) {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e)
                            p.O[obase + q4 + 4 * e] = o[e];
                    }
                } else {
                    // kra P-938os: the wave's 16 x 16 staging tile with its 8 B column granule XOR-ed by (row >> 2) & 3: the
                    // 32 B row pitch put rows m, m + 4, m + 8, m + 12 on one bank group (C-fragment b16 stores 4-way);
                    // now stores and the 8 B row reads are conflict-free (hipprof, 8192/64: 704 -> 128 conflict cycles per CTA-step)
                    bf16* st_ = oStage[wv];
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        reinterpret_cast<short*>(st_)[m * 16 + 4 * (e ^ ((m >> 2) & 3)) + q4] = bf_rne_int(o[e]);
                    r1a_o_wave_order();
                    const int m2 = l & 15, g2_ = l >> 4;
                    const v4bh ov = *reinterpret_cast<const v4bh*>(
                            reinterpret_cast<const short*>(st_) + m2 * 16 + 4 * (g2_ ^ ((m2 >> 2) & 3)));
                    if (ook) *reinterpret_cast<v4bh*>(p.Ob + obase + 4 * g2_) = ov;
                }
            }
        }
        #pragma unroll
        for (int vt = 0; vt < NVH; ++vt) {
            f4 upd = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + m) * kPadT + 16 * rt + q4 * 4);
                upd = mmac(va, kf0[rt], upd);
            }
            ht0[vt] += upd;
        }
        #pragma unroll
        for (int vt = 0; vt < NVH; ++vt) {
            f4 upd = f4{0.f,0.f,0.f,0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                const v4bh va = ld8(vB + (16 * vt + m) * kPadT + 16 * rt + q4 * 4);
                upd = mmac(va, kf1[rt], upd);
            }
            ht1[vt] += upd;
        }
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        loadK(nxt, kb0, kf0); loadK(nxt, kb1, kf1);
        if constexpr (MODE == 1) __builtin_amdgcn_sched_barrier(0);
        write_h(kb0, ht0);
        write_h(kb1, ht1);
    }
    if constexpr (ST) if (p.Ht != nullptr) {
        #pragma unroll
        for (int vt = 0; vt < NVH; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                p.Ht[g2w_state_off<NVH>(sg, p.H, bh, kb0, vs, vt, l, e)] = ht0[vt][e];
                p.Ht[g2w_state_off<NVH>(sg, p.H, bh, kb1, vs, vt, l, e)] = ht1[vt][e];
            }
    }
}

// kra P-52: Gn -> exp2f(Gn) in place, the consumers' former per-step exp2f evaluated once
__global__ void __launch_bounds__(256) g2_exp2_kernel(float* __restrict__ x, long n) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] = exp2f(x[i]);
}
// kra P-61e: Gn_raw (pi16-permuted k: [16a + j] = k 16a + pi16(j)) -> Gn = exp2f(raw) in the permuted order (dhu),
// Gn_nat = exp2f(raw) in natural k order (recompute), raw -> natural k order in place (deep-saturation raw recompute
// instance).  One element per lane: a 16-group never straddles a wave and every lane's store depends on its own
// load, so the wave's loads of the group return before any of its in-place stores.  Same exp2f -> bit-identical.
__global__ void __launch_bounds__(256) g2_gn_kp_kernel(float* __restrict__ raw, float* __restrict__ gp,
                                                       float* __restrict__ gnat, long n) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        const float x = raw[i];
        const float v = exp2f(x);
        const long j = (i & ~15L) | pi16((int)(i & 15));
        gp[i] = v;
        gnat[j] = v;
        raw[j] = x;
    }
}
extern "C" int g2_gn_kp(float* raw, float* gp, float* gnat, long n, hipStream_t s) {
    if (n <= 0) return 0;
    if (n % 16) return -1;
    hipLaunchKernelGGL(g2_gn_kp_kernel, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, s, raw, gp, gnat, n);
    return (int)hipGetLastError();
}
// kra P-61e: in-place kp re-layout of a non-kp producer's Kn (16 B lane pairs: lane (q, m) <- lane (q, pi16(m))
// inside every 512-element group) and Gn (permuted -> natural k).  One 64-lane wave per 512-element Kn group.
__global__ void __launch_bounds__(256) g2_kp_relayout_kernel(short* __restrict__ kn, long nkg,
                                                             float* __restrict__ gn, long ngg,
                                                             short* __restrict__ un) {
    const long w = ((long)blockIdx.x * 256 + threadIdx.x) >> 6;
    const int l = threadIdx.x & 63;
    if (un != nullptr && w < nkg) {   // kra P-62: Un -> pm column order (16 B pairs, nat16 inside 512)
        short v[8];
        #pragma unroll
        for (int sb = 0; sb < 2; ++sb)
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                v[4 * sb + e] = un[512 * w + 8 * (16 * e + (l & 15)) + 4 * sb + (l >> 4)];
        v4bh a, b;
        #pragma unroll
        for (int e = 0; e < 4; ++e) { a[e] = v[e]; b[e] = v[4 + e]; }
        st16x2(un + 512 * w + 8 * l, a, b);
    }
    if (w < nkg) {
        v4bh a, b;
        // one wave owns the group: its load instruction returns before its store issues (data dependence)
        ld16x2(reinterpret_cast<const bf16*>(kn) + 512 * w + 8 * ((l & 48) | pi16(l & 15)), a, b);
        st16x2(kn + 512 * w + 8 * l, a, b);
    }
    const long gi = (long)blockIdx.x * 256 + threadIdx.x;   // Gn: one element per lane (as g2_gn_kp)
    if (gi < ngg) {
        const float x = gn[gi];
        gn[(gi & ~15L) | pi16((int)(gi & 15))] = x;
    }
}
extern "C" int g2_kp_relayout(void* kn, long kn_elems, float* gn, long gn_elems, void* un, hipStream_t s) {
    if (kn_elems % 512 || gn_elems % 16) return -1;
    const long nkg = kn_elems / 512, ngg = gn_elems;
    const long thr = nkg * 64 > ngg ? nkg * 64 : ngg;
    if (thr == 0) return 0;
    hipLaunchKernelGGL(g2_kp_relayout_kernel, dim3((unsigned)((thr + 255) / 256)), dim3(256), 0, s,
                       (short*)kn, nkg, gn, ngg, (short*)un);
    return (int)hipGetLastError();
}
extern "C" int g2_exp2_inplace(float* x, long n, hipStream_t s) {
    if (n <= 0) return 0;
    hipLaunchKernelGGL(g2_exp2_kernel, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, s, x, n);
    return (int)hipGetLastError();
}

extern "C" int g2_fwd_h_v8o2(const void* Wn, const void* Kn, const void* Un, const void* Gn,
                             const void* QgA, const void* AqkA, void* O, void* Ob, void* Vn,
                             void* HbN, void* ANv,
                             float scale, int B, int T, int H, const int* seg, int nseg,
                             const int* ctok, const int* clen, const void* GnRaw, const void* H0, void* Ht,
                             const void* H0Aux, void* HtAux, hipStream_t s) {
    // T is the chunk-padded row count (NT = T / 64); with ctok/clen the o rows are packed.
    if (T % kBT || B != 1) return -1;                  // batch = segments (seg/nseg)
    if (seg == nullptr) nseg = 1;
    if ((ctok == nullptr) != (clen == nullptr)) return -2;
    FwdOParams p{(const bf16*)Wn, (const bf16*)Kn, (const bf16*)Un, (const float*)Gn,
                 (const bf16*)QgA, (const bf16*)AqkA, (float*)O, (bf16*)Ob, (bf16*)Vn,
                 (bf16*)HbN, (bf16*)ANv,
                 scale, H, T / kBT, seg, nseg, ctok, clen, (const float*)GnRaw, (const float*)H0, (float*)Ht};
    int r1a = 1;   // R1a: 4-wave dedup form (bitwise equal; 0 restores the 8-wave form)
    if (const char* e = g2::g2_getenv("KDA_G2_R1A")) r1a = atoi(e);
    p.H0Aux = (const float*)H0Aux; p.HtAux = (float*)HtAux;
    if (HtAux != nullptr && (Ht == nullptr || O != nullptr || Ob != nullptr || Vn || HbN || ANv)) return -3;
    const bool kp = g2_kp_layout(H);   // kra P-61e: Kn lane-permuted / Gn natural-k producers
    if (kp) r1a = 1;
    const dim3 grid(kK / kBV * (HtAux ? 2 : 1), (unsigned)nseg * H);
    // kra P-9: the production output sets get compile-time instances, others the generic one
    // Mode 1 (forward, o only) is NOT selected: measured 4% slower (kra P-9) -- without the
    // conservative waits the compiler clusters the loads and VMEM issue stalls return, while the
    // generic instance's waits happen to pace them.  Modes 2/3 measured ~5% faster.
    int mode = 0;
    if (O == nullptr && Ob != nullptr && !Vn && !HbN && !ANv) mode = 1;   // kra P-11: paced mode 1
    else if (O == nullptr && Ob != nullptr && Vn && HbN && ANv) mode = 2;
    else if (O == nullptr && Ob == nullptr && Vn && HbN && ANv) mode = 3;
    else if (O == nullptr && Ob == nullptr && Vn && HbN && !ANv) mode = 4;   // kra P-22
    else if (O == nullptr && Ob != nullptr && !Vn && HbN && !ANv) mode = 5;   // kra P-82: forward saves h
    // kra P-24: at deep saturation cap v8o4 at 2 CTAs/CU (24 KB LDS per CTA) so the K-side
    // operands shared by a head's 4 V-slab CTAs stay in L2.
    const bool throttle = 4L * nseg * H >= g2_env_long("KDA_G2_V8O4_THROTTLE_MIN_CTAS", 224);
    auto launch4 = [&](auto kern) {
        const size_t dyn = throttle ? g2_lds_topup(kern, 24 * 1024) : 0;
        hipLaunchKernelGGL(kern, grid, dim3(256), dyn, s, p);
    };
    // kra P-52: narrow MODE 4 at deep saturation keeps the production instance on the raw Gn
    const bool raw4 = GnRaw != nullptr && 4L * nseg * H >= g2_env_long("KDA_G2_BN_RAW_MIN_CTAS", 192);
    auto launch4raw = [&](auto kern) {
        FwdOParams q = p; q.Gn = (const float*)GnRaw;
        const size_t dyn = throttle ? g2_lds_topup(kern, 24 * 1024) : 0;
        hipLaunchKernelGGL(kern, grid, dim3(256), dyn, s, q);
    };
    // kra P-58: SGPR wave index only in the chain-bound regime (at most one CTA per CU)
    const bool sg = 4L * nseg * H <= g2_env_long("KDA_G2_SGPR_MAX_CTAS", 80);
    const bool st = H0 != nullptr || Ht != nullptr;   // state ports: the same instance with ST
    // kra P-938cp1: the CP summary pass (state only, no output) on its own compile-time instance (MODE 6) instead of
    // the generic one, whose runtime-optional outputs make the waitcnt pass count the skip path (P-9).  gfx938, 16K:
    // S_loc 0.42 -> 0.35 ms (H = 8), 0.39 -> 0.33 (12), 0.58 -> 0.48 (24), 0.75 -> 0.66 (48); [S_loc | M] 0.55 ->
    // 0.46 (12), 0.74 -> 0.64 (24), 1.35 -> 1.17 (48); bit-identical.  SG (P-58) on the kp layouts: H = 12 0.37 ->
    // 0.33, without kp (H = 8) 0.35 -> 0.36.  KDA_G2_CP_FSUM: 0 generic | 1 measured | 2 never SG | 3 always SG.
    const bool sum = st && O == nullptr && Ob == nullptr && !Vn && !HbN && !ANv;
    const long fsum = g2_env_long("KDA_G2_CP_FSUM", g2_cur_gfx938() ? 1 : 0);
    if (sum && fsum) mode = 6;
    auto launch4s = [&](auto k) {
        using K = decltype(k);
        if (st) launch4(K::template get<true>()); else launch4(K::template get<false>());
    };
    auto launch4raws = [&](auto k) {
        using K = decltype(k);
        if (st) launch4raw(K::template get<true>()); else launch4raw(K::template get<false>());
    };
    auto v8o4 = [&](auto vl) {
        constexpr bool VL = decltype(vl)::value;
        switch (mode) {   // kra P-61e: KP instances whenever the producers emit the kp layouts
            case 5: if (sg && kp) launch4s(V8o4K<VL, 5, true, true, true, true, true, true>{});   // kra P-82
                    else if (kp) launch4s(V8o4K<VL, 5, true, false, false, true, true, true>{});
                    else    launch4s(V8o4K<VL, 5>{});
                    break;
            case 1: if (sg) { if (kp) launch4s(V8o4K<VL, 1, true, true, true, true, true>{});   // kra P-60
                              else    launch4s(V8o4K<VL, 1, true, true, true>{}); }
                    else if (kp) launch4s(V8o4K<VL, 1, true, false, false, true, true>{});
                    else    launch4s(V8o4K<VL, 1>{});
                    break;
            case 2: if (kp) launch4s(V8o4K<VL, 2, true, false, false, true, true, true>{});
                    else    launch4s(V8o4K<VL, 2>{});
                    break;
            case 3: if (kp) launch4s(V8o4K<VL, 3, true, false, false, true, false, true>{});
                    else    launch4s(V8o4K<VL, 3>{});
                    break;
            case 4: if (raw4) { if (kp) launch4raws(V8o4K<VL, 4, false, false, false, true, false, true>{});
                                else    launch4raws(V8o4K<VL, 4, false>{}); }
                    else if (sg) { if (kp) launch4s(V8o4K<VL, 4, true, true, false, true, false, true>{});
                                   else    launch4s(V8o4K<VL, 4, true, true>{}); }
                    else if (kp) launch4s(V8o4K<VL, 4, true, false, false, true, false, true>{});
                    else      launch4s(V8o4K<VL, 4>{});
                    break;
            case 6: {   // kra P-938cp1
                const bool sg6 = fsum == 3 || (fsum == 1 && kp);
                if (kp) { if (sg6) launch4s(V8o4K<VL, 6, true, true, false, true, true>{});
                          else     launch4s(V8o4K<VL, 6, true, false, false, true, true>{}); }
                else    { if (sg6) launch4s(V8o4K<VL, 6, true, true>{});
                          else     launch4s(V8o4K<VL, 6>{}); }
                break;
            }
            default: if (kp) launch4s(V8o4K<VL, 0, true, false, false, true, true>{});
                     else    launch4s(V8o4K<VL, 0>{});
                     break;
        }
    };
    // kra P-49: wide-head launches use 64-column V-slabs (2 CTAs per head): the K-side tiles of a step
    // are issued by 2 CTAs instead of 4 (bit-identical per column)
    // gated to long sequences: at NT < 512 the 96-CTA wide grid's CU imbalance cost more than the shared
    // loads it saves (16K/48 mode 4 0.91x, 8K/96 0.68x); 32K/48 1.02-1.12x, 64K/48 1.12-1.40x
    if (HtAux != nullptr) r1a = 1;   // the transition-matrix slabs exist in v8o4 only
    // kra P-938z: gfx938 at H <= 8 (<= 32 CTAs of 32 columns on 72 CUs): 16-column slabs, 8 CTAs per head.  The step
    // latency scales with the slab width (64-col 2.0-2.3x slower than 32-col), and 16-col is 1.12-1.18x faster than the
    // kp 32-col default at H = 8 (bit-identical).  At H >= 16 it loses (0.64-0.87x), so not there.
    // kra P-938cp3: on gfx938 the calls with state ports (CP) follow the same slab policy -- 16K: the o recurrence
    // from h0 0.60 -> 0.43 ms at H = 8 (narrow), 1.81 -> 1.53 at H = 64 and 2.64 -> 2.49 at H = 96 (wide); the summary
    // pass 0.35 -> 0.30 at H = 8 (narrow) but 0.93 -> 1.03 at H = 64 and 1.38 -> 1.69 at H = 96, so it never goes wide.
    // The transition-matrix slabs exist in v8o4 only.  KDA_G2_CP_WST=0: state ports always on v8o4.
    const bool wst = HtAux == nullptr && (!st || (g2_cur_gfx938() && g2_env_long("KDA_G2_CP_WST", 1) != 0));
    const bool narrow938 = r1a && !kp && wst && H <= 8 && g2_cur_gfx938();
    // kra P-938wd: gfx938 at H >= 64 (>= 128 wide CTAs): wide at every NT -- the hand-off-free ceiling showed fwd_h's
    // K-side reads at 30-55% of its time there; wide 1.04x (2K) .. 1.16x (16K).  At H = 32 (64 CTAs) it loses 0.74-0.79x.
    const long wmin_nt = g2_env_long("KDA_G2_V8O4W_MIN_NT", g2_cur_gfx938() && H >= 64 ? 0 : 512);
    if (narrow938 || (r1a && !kp && wst && !sum && H >= g2_env_long("KDA_G2_V8O4W_MIN_H", 40) && T / kBT >= wmin_nt)) {
        const int nvh = (int)g2_env_long("KDA_G2_V8O4W_NVH", narrow938 ? 1 : 4);   // kra P-938z: 1 = 16-column slabs
        const dim3 gridw(kK / (16 * nvh), (unsigned)nseg * H);
        auto v8o4w = [&](auto vl) {
            constexpr bool VL = decltype(vl)::value;
            if (nvh == 1) {
                switch (mode) {
                    case 1: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 1, 1, true> : g2_fwd_h_v8o4w_kernel<VL, 1, 1>), gridw, dim3(256), 0, s, p); break;
                    case 2: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 2, 1, true> : g2_fwd_h_v8o4w_kernel<VL, 2, 1>), gridw, dim3(256), 0, s, p); break;
                    case 3: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 3, 1, true> : g2_fwd_h_v8o4w_kernel<VL, 3, 1>), gridw, dim3(256), 0, s, p); break;
                    case 4: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 4, 1, true> : g2_fwd_h_v8o4w_kernel<VL, 4, 1>), gridw, dim3(256), 0, s, p); break;
                    case 5: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 5, 1, true> : g2_fwd_h_v8o4w_kernel<VL, 5, 1>), gridw, dim3(256), 0, s, p); break;
                    case 6: hipLaunchKernelGGL((g2_fwd_h_v8o4w_kernel<VL, 6, 1, true>), gridw, dim3(256), 0, s, p); break;   // kra P-938cp1
                    default: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 0, 1, true> : g2_fwd_h_v8o4w_kernel<VL, 0, 1>), gridw, dim3(256), 0, s, p); break;
                }
                return;
            }
            switch (mode) {
                case 1: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 1, 4, true> : g2_fwd_h_v8o4w_kernel<VL, 1>), gridw, dim3(256), 0, s, p); break;
                case 2: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 2, 4, true> : g2_fwd_h_v8o4w_kernel<VL, 2>), gridw, dim3(256), 0, s, p); break;
                case 3: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 3, 4, true> : g2_fwd_h_v8o4w_kernel<VL, 3>), gridw, dim3(256), 0, s, p); break;
                case 4: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 4, 4, true> : g2_fwd_h_v8o4w_kernel<VL, 4>), gridw, dim3(256), 0, s, p); break;
                case 5: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 5, 4, true> : g2_fwd_h_v8o4w_kernel<VL, 5>), gridw, dim3(256), 0, s, p); break;   // kra P-82
                default: hipLaunchKernelGGL((st ? g2_fwd_h_v8o4w_kernel<VL, 0, 4, true> : g2_fwd_h_v8o4w_kernel<VL, 0>), gridw, dim3(256), 0, s, p); break;
            }
        };
        if (ctok != nullptr) v8o4w(std::true_type{}); else v8o4w(std::false_type{});
        return (int)hipGetLastError();
    }
    if (ctok != nullptr) {
        if (r1a) v8o4(std::true_type{});
        else     hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<true>, grid, dim3(512), 0, s, p);
    } else {
        if (r1a) v8o4(std::false_type{});
        else     hipLaunchKernelGGL(g2_fwd_h_v8o_kernel<false>, grid, dim3(512), 0, s, p);
    }
    return (int)hipGetLastError();
}

extern "C" int g2_fwd_h_v8o(const void* Wn, const void* Kn, const void* Un, const void* Gn,
                            const void* QgA, const void* AqkA, void* O, void* Ob, void* Vn,
                            void* HbN, void* ANv,
                            float scale, int B, int T, int H, const int* seg, int nseg,
                            const int* ctok, const int* clen, const void* GnRaw, const void* H0, void* Ht,
                            hipStream_t s) {
    return g2_fwd_h_v8o2(Wn, Kn, Un, Gn, QgA, AqkA, O, Ob, Vn, HbN, ANv, scale, B, T, H, seg, nseg, ctok,
                         clen, GnRaw, H0, Ht, nullptr, nullptr, s);
}

}  // namespace g2
