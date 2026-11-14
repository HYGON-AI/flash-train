// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// clang-format off
// G2 engine P1 (forward state recurrence) + P2 (reverse state recurrence), gfx936.
//
// Layout contract (native MMAC fragments):
//   Wn  [BH][NT][r4][kt8][lane][4]   A(m=t,d=k)     Kn [BH][NT][kt8][rt4][lane][4]  B(d=t,n=k)
//   Un  [BH][NT][r4][VT8][lane][4]   C(m=t,n=v)     Gn [BH][NT][kt8][q4][e4] fp32  (k=16kt+q+4e)
//   Hn  [BH][NT][VT8][KT8][lane][4]  C(m=v,n=k)     Vn like Un
// Block order is (B, H, NT): head-major, chunk-minor.  P3 in g2_wy.cuh uses (NT, H).
#include "g2_common.h"

namespace g2 {

struct FwdNParams {
    const bf16* Wn; const bf16* Kn; const bf16* Un; const float* Gn;
    bf16* Hn; bf16* Vn; int H, NT;
};

// ================================================================ P2: reverse engine (FLA dhu role)
// per chunk i = NT-1 .. 0, per (seq, head, V-slice of 32), 8 waves:
//   dh_out[i] = dh                                                  (bf16)
//   dv2       = Kg dh + dv                                          (stored bf16)
//   dh        = diag(2^{gk_last}) dh + scale * Qg^T dO - W^T dv2
// Fragment-native operands (our own producers choose them):
//   KgA [BH][NT][r4][kt8][lane][4]  A(m=t,d=k)    Qg,Wb [BH][NT][kt8][rt4][lane][4]  B(d=t,n=k)
//   dOT [BH][NT][VT8][rt4][lane][4] A(m=v,d=t)    dv, dv2 [BH][NT][r4][VT8][lane][4]  C(m=t,n=v)
//   Gn  [BH][NT][kt8][q4][e4]                     dh_out [BH][NT][VT8][KT8][lane][4]  C(m=v,n=k)
struct BwdNParams {
    const bf16* KgA; const bf16* Qg; const bf16* Wb; const bf16* dOT; const bf16* dv; const float* Gn;
    bf16* dh; bf16* dv2; float scale; int H, NT;
    const bf16* HbN;                                  // REVIEW S3-6: bN(dh) direct emit (null = off)
    bf16* ANv2;                                       // S3-10: aN(dv2) direct emit (null = off)
    // Varlen/batch: chunk-aligned segments, as in FwdOParams
    const int* seg; int nseg;
};
template <int MODE = 0, bool LIN = false>
__global__ void __launch_bounds__(512) g2_dhu_v8b_kernel(BwdNParams p) {
    // kra P-10: MODE 1 = production output set (HbN + ANv2; dv2 and dh null) as compile-time
    // constants -- runtime-optional VMEM ops make the waitcnt pass count the skip path.
    const bool hasBN = MODE == 0 ? p.HbN != nullptr : true;
    const bool hasDH = MODE == 0 ? p.dh != nullptr : false;
    const bool hasDV2 = MODE == 0 ? p.dv2 != nullptr : MODE == 2;    // kra P-22: MODE 2 = HbN + dv2
    const bool hasAN2 = MODE == 0 ? p.ANv2 != nullptr : MODE == 1;
    __shared__ __align__(16) bf16 dhB[kBV * kPadK];      // dh as [v][k]   (GEMM-A B operand)
    __shared__ __align__(16) bf16 vB[kBV * kPadT];       // dv2 as [v][t]  (GEMM-B A operand)
    __shared__ __align__(16) v4bh dOs[2][8 * 64];        // dO^T A-frags [vt*4+rt][lane], double buffered
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int vs = blockIdx.x;
    const int sg = blockIdx.y / p.H, bh = blockIdx.y - sg * p.H;
    const int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;
    if (c0 >= c1) return;                                // empty segment
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv & 3, v1 = wv >> 2;                 // GEMM-A tile: t-tile r1, v-tile v1
    f4 st[2];                                            // DH^T C-frags [vt]: v=16vt+c_row, k=16wv+c_col
    st[0] = f4{0.f, 0.f, 0.f, 0.f}; st[1] = st[0];
    v4bh ka[8], qb[4], wb[4], dof, dvf; f4 gl;       // dof: this wave's 1/8 of dO^T
    auto loadKA = [&](int i) {
        const bf16* src = p.KgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, ka[2 * j], ka[2 * j + 1]);
    };
    auto loadDV = [&](int i) { dvf = ld8(p.dv + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1) * 256 + l * 4); };
    auto loadG = [&](int i) { gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + wv) * 16 + (l >> 4) * 4); };
    auto loadQW = [&](int i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {                    // kra L16: paired
            ld16x2(p.Qg + (((chunk + i) * 8 + wv) * 4) * 256 + j * 512 + l * 8, qb[2 * j], qb[2 * j + 1]);
            ld16x2(p.Wb + (((chunk + i) * 8 + wv) * 4) * 256 + j * 512 + l * 8, wb[2 * j], wb[2 * j + 1]);
        }
    };
    auto loadDO = [&](int i) {
        dof = ld8(p.dOT + (((chunk + i) * 8 + 2 * vs + (wv >> 2)) * 4 + (wv & 3)) * 256 + l * 4);
    };
    v4bh hx[2];
    auto write_dh = [&]() {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne(st[vt][e]);
                reinterpret_cast<short*>(dhB)[(16 * vt + c_row(l)) * kPadK + 16 * wv + c_col(l, e)] = hx[vt][e];
            }
    };
    const int last = c1 - 1;                             // dh = 0 at the segment end
    loadKA(last); loadDV(last); loadG(last); loadQW(last); loadDO(last);
    write_dh();
    {
        const int i = last;
        lds_barrier();
        // bN(dh) emit: dhB holds this wave's k-tile after write_dh; slot (kt=wv, vt,
        // l, e) = dh[16*wv + pi(m)][16*(2*vs+vt) + 4*(l>>4) + e]
        if (hasBN) {
            const int q4_ = l >> 4, m_ = l & 15;
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * wv + pi16(m_)];
                const size_t off = ((((size_t)i * p.H + bh) * 8 + wv) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;
            }
        }
    }
    for (int s = 0; s < c1 - c0; ++s) {
        const int i = last - s, nxt = i > c0 ? i - 1 : c0;
        if (hasDH) {
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt)
                *reinterpret_cast<v4bh*>(p.dh + (((chunk + i) * 8 + 2 * vs + vt) * 8 + wv) * 256 + l * 4) = hx[vt];
        }
        dOs[s & 1][wv * 64 + l] = dof;                   // publish chunk i's dO^T slice
        loadDO(nxt);                                     // a full step ahead of its publish
        lds_barrier();                                   // dhB (entry state of chunk i) + dOs ready
        // bN(dh) emit: slot (kt=wv, vt, l, e) = dh[16*wv + pi(m)][16*(2*vs+vt) + 4*(l>>4) + e]
        if (hasBN) {
            const int q4_ = l >> 4, m_ = l & 15;
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * wv + pi16(m_)];
                const size_t off = ((((size_t)i * p.H + bh) * 8 + wv) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;
            }
        }
        f4 acc = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            acc = mmac(ka[kt], ld8(dhB + (16 * v1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc);
        loadKA(nxt);
        {
            v4bh x;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float d = acc[e] + __uint_as_float(((uint32_t)(unsigned short)dvf[e]) << 16);
                x[e] = bf_rne(d);
                reinterpret_cast<short*>(vB)[(16 * v1 + c_col(l, e)) * kPadT + 16 * r1 + c_row(l)] = x[e];
            }
            if (hasDV2)  // T9b: the global dV2n write is optional; the
                *reinterpret_cast<v4bh*>(p.dv2 + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1) * 256 + l * 4) = x;
            if (hasAN2 && !LIN)  // aN(dv2): same C-frag swap as the engines' aN emits
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    reinterpret_cast<short*>(p.ANv2)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x[e];
        }
        loadDV(nxt);
        {
            f4 d;
            #pragma unroll
            for (int e = 0; e < 4; ++e) d[e] = g2_vcopy(gl[e]);   // kra P-52: Gn = exp2f(G)
            st[0] *= d; st[1] *= d;
        }
        loadG(nxt);
        lds_barrier();                                   // vB ready, dhB reads done
        if (hasAN2 && LIN) {   // kra P-22c/d: aN(dv2) as linear 8 B stores gathered from the staged vB tile
            #pragma unroll
            for (int v1g = v1; v1g < v1 + 1; ++v1g) {
                v4bh an;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    an[e] = reinterpret_cast<const short*>(vB)[(16 * v1g + 4 * (l >> 4) + e) * kPadT + 16 * r1 + (l & 15)];
                *reinterpret_cast<v4bh*>(p.ANv2 + ((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1g) * 256 + l * 4) = an;
            }
        }
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 qd = f4{0.f, 0.f, 0.f, 0.f}, wd = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                qd = mmac(dOs[s & 1][(vt * 4 + rt) * 64 + l], qb[rt], qd);
                wd = mmac(ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4), wb[rt], wd);
            }
            st[vt] += qd * p.scale - wd;
        }
        loadQW(nxt);
        write_dh();
    }
}

// ================================================================ R1a-dhu: 4-wave form
// 256 threads / 4 waves.  Wave w owns GEMM tiles (r1=w, both v1 halves) and state
// K-blocks {w, w+4}: KgA is fetched ONCE per CTA per step (the 8-wave form loads it
// twice -- waves (r1,0)/(r1,1) read the same rows).  Every output element keeps its
// MMAC operand order and accumulation order -> bit-identical (bench: 1.027x H=12,
// 1.026x H=24, 1.093x H=48, occ 3->3, 185 VGPR 0 spill).  At deep saturation
// (H=96, 384 CTAs) latency hiding is WORSE (0.961x) -> host auto-selects by CTA
// count.
template <int MODE = 0, bool LIN = false>
__global__ void __launch_bounds__(256) g2_dhu_v8b4_kernel(BwdNParams p) {
    // kra P-10: MODE 1 = production output set (HbN + ANv2; dv2 and dh null) as compile-time
    // constants -- runtime-optional VMEM ops make the waitcnt pass count the skip path.
    const bool hasBN = MODE == 0 ? p.HbN != nullptr : true;
    const bool hasDH = MODE == 0 ? p.dh != nullptr : false;
    const bool hasDV2 = MODE == 0 ? p.dv2 != nullptr : MODE == 2;    // kra P-22: MODE 2 = HbN + dv2
    const bool hasAN2 = MODE == 0 ? p.ANv2 != nullptr : MODE == 1;
    __shared__ __align__(16) bf16 dhB[kBV * kPadK];
    __shared__ __align__(16) bf16 vB[kBV * kPadT];
    __shared__ __align__(16) v4bh dOs[2][8 * 64];
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;   // wv in 0..3
    const int vs = blockIdx.x;
    const int sg = blockIdx.y / p.H, bh = blockIdx.y - sg * p.H;
    const int c0 = p.seg ? p.seg[sg] : 0, c1 = p.seg ? p.seg[sg + 1] : p.NT;
    if (c0 >= c1) return;
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv, kb0 = wv, kb1 = wv + 4;
    f4 st0[2], st1[2];
    st0[0] = f4{0.f, 0.f, 0.f, 0.f}; st0[1] = st0[0];
    st1[0] = f4{0.f, 0.f, 0.f, 0.f}; st1[1] = st1[0];
    v4bh ka[8], qb0[4], wb0[4], qb1[4], wb1[4], dof0, dof1, dvf0, dvf1;
    f4 gl0, gl1;
    auto loadKA = [&](int i) {
        const bf16* src = p.KgA + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, ka[2 * j], ka[2 * j + 1]);
    };
    auto loadDV = [&](int i, int v1, v4bh& dvf) {
        dvf = ld8(p.dv + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1) * 256 + l * 4);
    };
    auto loadG = [&](int i, int kb, f4& gl) {
        gl = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + kb) * 16 + (l >> 4) * 4);
    };
    auto loadQW = [&](int i, int kb, v4bh* qb, v4bh* wb) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {                    // kra L16: paired
            ld16x2(p.Qg + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, qb[2 * j], qb[2 * j + 1]);
            ld16x2(p.Wb + (((chunk + i) * 8 + kb) * 4) * 256 + j * 512 + l * 8, wb[2 * j], wb[2 * j + 1]);
        }
    };
    auto loadDO = [&](int i, int v1, v4bh& dof) {
        dof = ld8(p.dOT + (((chunk + i) * 8 + 2 * vs + v1) * 4 + r1) * 256 + l * 4);
    };
    v4bh hx0[2], hx1[2];
    auto write_dh = [&](int kb, f4* st, v4bh* hx) {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne(st[vt][e]);
                reinterpret_cast<short*>(dhB)[(16 * vt + c_row(l)) * kPadK + 16 * kb + c_col(l, e)] = hx[vt][e];
            }
    };
    auto bN_emit = [&](int i, int kb) {
        if (hasBN) {
            const int q4_ = l >> 4, m_ = l & 15;
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                v4bh val;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    val[e] = reinterpret_cast<const short*>(dhB)[(16 * vt + 4 * q4_ + e) * kPadK + 16 * kb + pi16(m_)];
                const size_t off = ((((size_t)i * p.H + bh) * 8 + kb) * 8 + (2 * vs + vt)) * 256 + l * 4;
                *const_cast<v4bh*>(reinterpret_cast<const v4bh*>(p.HbN + off)) = val;
            }
        }
    };
    const int last = c1 - 1;                             // dh = 0 at the segment end
    loadKA(last); loadDV(last, 0, dvf0); loadDV(last, 1, dvf1);
    loadG(last, kb0, gl0); loadG(last, kb1, gl1);
    loadQW(last, kb0, qb0, wb0); loadQW(last, kb1, qb1, wb1);
    loadDO(last, 0, dof0); loadDO(last, 1, dof1);
    write_dh(kb0, st0, hx0); write_dh(kb1, st1, hx1);
    {
        const int i = last;
        lds_barrier();
        bN_emit(i, kb0);
        bN_emit(i, kb1);
    }
    for (int s = 0; s < c1 - c0; ++s) {
        const int i = last - s, nxt = i > c0 ? i - 1 : c0;
        if (hasDH) {
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                *reinterpret_cast<v4bh*>(p.dh + (((chunk + i) * 8 + 2 * vs + vt) * 8 + kb0) * 256 + l * 4) = hx0[vt];
                *reinterpret_cast<v4bh*>(p.dh + (((chunk + i) * 8 + 2 * vs + vt) * 8 + kb1) * 256 + l * 4) = hx1[vt];
            }
        }
        dOs[s & 1][wv * 64 + l] = dof0;                  // slot w   = (v1=0, r1=w)
        dOs[s & 1][(wv + 4) * 64 + l] = dof1;            // slot w+4 = (v1=1, r1=w)
        loadDO(nxt, 0, dof0); loadDO(nxt, 1, dof1);
        lds_barrier();                                   // dhB (entry state of chunk i) + dOs ready
        bN_emit(i, kb0);
        bN_emit(i, kb1);
        f4 acc0 = f4{0.f, 0.f, 0.f, 0.f}, acc1 = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            acc0 = mmac(ka[kt], ld8(dhB + (16 * 0 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc0);
            acc1 = mmac(ka[kt], ld8(dhB + (16 * 1 + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc1);
        }
        loadKA(nxt);
        #pragma unroll
        for (int v1 = 0; v1 < 2; ++v1) {
            const f4 acc = v1 ? acc1 : acc0;
            const v4bh dvf = v1 ? dvf1 : dvf0;
            v4bh x;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float d = acc[e] + __uint_as_float(((uint32_t)(unsigned short)dvf[e]) << 16);
                x[e] = bf_rne(d);
                reinterpret_cast<short*>(vB)[(16 * v1 + c_col(l, e)) * kPadT + 16 * r1 + c_row(l)] = x[e];
            }
            if (hasDV2)
                *reinterpret_cast<v4bh*>(p.dv2 + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1) * 256 + l * 4) = x;
            if (hasAN2 && !LIN)
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    reinterpret_cast<short*>(p.ANv2)[((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1) * 256
                           + (l >> 4) + 4 * (l & 15) + 64 * e] = x[e];
        }
        loadDV(nxt, 0, dvf0); loadDV(nxt, 1, dvf1);
        {
            f4 d0, d1;
            #pragma unroll
            for (int e = 0; e < 4; ++e) { d0[e] = g2_vcopy(gl0[e]); d1[e] = g2_vcopy(gl1[e]); }   // kra P-52: Gn = exp2f(G)
            st0[0] *= d0; st0[1] *= d0;
            st1[0] *= d1; st1[1] *= d1;
        }
        loadG(nxt, kb0, gl0); loadG(nxt, kb1, gl1);
        lds_barrier();                                   // vB ready, dhB reads done
        if (hasAN2 && LIN) {   // kra P-22c/d: aN(dv2) as linear 8 B stores gathered from the staged vB tile
            #pragma unroll
            for (int v1g = 0; v1g < 2; ++v1g) {
                v4bh an;
                #pragma unroll
                for (int e = 0; e < 4; ++e)
                    an[e] = reinterpret_cast<const short*>(vB)[(16 * v1g + 4 * (l >> 4) + e) * kPadT + 16 * r1 + (l & 15)];
                *reinterpret_cast<v4bh*>(p.ANv2 + ((size_t)i * p.H + bh) * 8192 + (8 * r1 + 2 * vs + v1g) * 256 + l * 4) = an;
            }
        }
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 qd = f4{0.f, 0.f, 0.f, 0.f}, wd = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                qd = mmac(dOs[s & 1][(vt * 4 + rt) * 64 + l], qb0[rt], qd);
                wd = mmac(ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4), wb0[rt], wd);
            }
            st0[vt] += qd * p.scale - wd;
        }
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt) {
            f4 qd = f4{0.f, 0.f, 0.f, 0.f}, wd = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int rt = 0; rt < 4; ++rt) {
                qd = mmac(dOs[s & 1][(vt * 4 + rt) * 64 + l], qb1[rt], qd);
                wd = mmac(ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4), wb1[rt], wd);
            }
            st1[vt] += qd * p.scale - wd;
        }
        loadQW(nxt, kb0, qb0, wb0); loadQW(nxt, kb1, qb1, wb1);
        write_dh(kb0, st0, hx0);
        write_dh(kb1, st1, hx1);
    }
}

extern "C" int g2_dhu_v8b(const void* KgA, const void* Qg, const void* Wb, const void* dOT, const void* dv,
                          const void* Gn, void* dv2, void* HbN, void* ANv2, float scale, int B, int T, int H,
                          const int* seg, int nseg, hipStream_t s) {
    if (T % kBT || B != 1) return -1;                    // batch = segments (seg/nseg)
    if (seg == nullptr) nseg = 1;
    BwdNParams p{(const bf16*)KgA, (const bf16*)Qg, (const bf16*)Wb, (const bf16*)dOT, (const bf16*)dv,
                 (const float*)Gn, nullptr, (bf16*)dv2, scale, H, T / kBT, (const bf16*)HbN,
                 (bf16*)ANv2, seg, nseg};
    // auto window from MEASURED production A/B (dhu-only, kernel time): 48 CTAs
    // (H=12) -9.6%, 96 CTAs (H=24) +2.6% bench / +0.1% e2e, 192 CTAs (H=48) +4.6%,
    // 384 CTAs (H=96) -3.2%.  Low CTA counts need the 8-wave form's latency hiding;
    // deep saturation too; the dedup only pays in between.
    const long ctas = 4L * nseg * H;
    // kra P-85: re-measured on P-82b -- the 8-wave form wins from 128 CTAs on (H=48: 1.05-1.15x)
    const int dhu4 = (ctas >= 96 && ctas <= kDhu4MaxCtas) ? 1 : 0;
    // kra P-24: at deep saturation cap dhu at 1 CTA/CU (33 KB LDS per CTA): the operands shared
    // by a head's 4 V-slab CTAs then stay in L2.
    const bool dthrottle = ctas >= kDhuThrottleMinCtas;
    auto dyn = [&](auto kern) -> size_t { return dthrottle ? g2_lds_topup(kern, 33 * 1024) : 0; };
    // kra P-22d: linear aN(dv2) emit only in the latency-bound regime (few CTAs); at deep saturation
    // the scattered stores pace the loads (R21).
    const bool lin = ctas <= 96;
    // kra P-10: the production output set gets the compile-time instance
    const bool prod = HbN != nullptr && ANv2 != nullptr && dv2 == nullptr;
    const bool prod2 = HbN != nullptr && ANv2 == nullptr && dv2 != nullptr;   // kra P-22
    if (dhu4) {
        // v8b4: the compile-time instance measured 8% SLOWER (kra P-10) -> generic only
        if (lin) hipLaunchKernelGGL((g2_dhu_v8b4_kernel<0, true>), dim3(kK / kBV, nseg * H), dim3(256), dyn((g2_dhu_v8b4_kernel<0, true>)), s, p);
        else     hipLaunchKernelGGL((g2_dhu_v8b4_kernel<0, false>), dim3(kK / kBV, nseg * H), dim3(256), dyn((g2_dhu_v8b4_kernel<0, false>)), s, p);
    } else {
        if (prod && lin) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true>)), s, p);
        else if (prod) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, false>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, false>)), s, p);
        else if (prod2) hipLaunchKernelGGL(g2_dhu_v8b_kernel<2>, dim3(kK / kBV, nseg * H), dim3(512), dyn(g2_dhu_v8b_kernel<2>), s, p);
        else      hipLaunchKernelGGL(g2_dhu_v8b_kernel<0>, dim3(kK / kBV, nseg * H), dim3(512), dyn(g2_dhu_v8b_kernel<0>), s, p);
    }
    return (int)hipGetLastError();
}

}  // namespace g2
