// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// G2 engine P1 (forward state recurrence) + P2 (reverse state recurrence), gfx936.
//
// Verbatim extraction of the two verified prototypes:
//   deepseek/handoff_20260925_G2_integration/src/prototypes/g2_engine.hip
//     P1 = g2_fwd_h_v7_kernel<8> / g2_fwd_h_v7w8
//     P2 = g2_dhu_w8b_kernel    / g2_dhu_w8b
// sha256(ca9f6fa3...) -- kernel bodies are copied without semantic change; only the
// surrounding declarations moved into g2_common.h.
//
// Layout contract (native fragments, see DESIGN.md section 4):
//   Wn  [BH][NT][r4][kt8][lane][4]   A(m=t,d=k)     Kn [BH][NT][kt8][rt4][lane][4]  B(d=t,n=k)
//   Un  [BH][NT][r4][VT8][lane][4]   C(m=t,n=v)     Gn [BH][NT][kt8][q4][e4] fp32  (k=16kt+q+4e)
//   Hn  [BH][NT][VT8][KT8][lane][4]  C(m=v,n=k)     Vn like Un
// Block order is (B, H, NT): head-major, chunk-minor.  P3 in g2_wy.cuh uses (NT, H).
#include "g2_common.h"

#include <type_traits>
namespace g2 {

struct FwdNParams {
    const bf16* Wn; const bf16* Kn; const bf16* Un; const float* Gn;
    bf16* Hn; bf16* Vn; int H, NT;
};

// ---------------------------------------------------------------- P1
template <int NW>
__global__ void __launch_bounds__(64 * NW) g2_fwd_h_v7_kernel(FwdNParams p) {
    constexpr int KPW = 8 / NW;                // state k-tiles per wave: 2 (NW=4) or 1 (NW=8)
    __shared__ __align__(16) bf16 hB[kBV * kPadK];
    __shared__ __align__(16) bf16 vB[kBV * kPadT];
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int vs = blockIdx.x, bh = blockIdx.y;
    const size_t chunk = (size_t)bh * p.NT;
    // GEMM1 ownership: t-tile r1, v-tiles [v1a, v1a + NV1)
    constexpr int NV1 = NW == 4 ? 2 : 1;
    const int r1 = NW == 4 ? wv : (wv & 3);
    const int v1a = NW == 4 ? 0 : (wv >> 2);
    f4 ht[2][KPW];
    #pragma unroll
    for (int a = 0; a < 2; ++a) for (int b = 0; b < KPW; ++b) ht[a][b] = f4{0.f, 0.f, 0.f, 0.f};
    v4bh wf[8], kf[KPW][4], uf[NV1]; f4 gl[KPW];
    auto loadW = [&](int i) {
        const bf16* src = p.Wn + (((chunk + i) * 4 + r1) * 8) * 256 + l * 8;      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, wf[2 * j], wf[2 * j + 1]);
    };
    auto loadU = [&](int i) {
        #pragma unroll
        for (int j = 0; j < NV1; ++j)
            uf[j] = ld8(p.Un + nat16((((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a + j) * 256 + l * 4));
    };
    auto loadG = [&](int i) {
        #pragma unroll
        for (int kl = 0; kl < KPW; ++kl)
            gl[kl] = *reinterpret_cast<const f4*>(p.Gn + ((chunk + i) * 8 + KPW * wv + kl) * 16 + (l >> 4) * 4);
    };
    auto loadK = [&](int i) {
        #pragma unroll
        for (int kl = 0; kl < KPW; ++kl)
            #pragma unroll
            for (int j = 0; j < 2; ++j)                  // kra L16: paired
                ld16x2(p.Kn + (((chunk + i) * 8 + KPW * wv + kl) * 4) * 256 + j * 512 + l * 8,
                       kf[kl][2 * j], kf[kl][2 * j + 1]);
    };
    v4bh hx[2][KPW];
    auto write_h = [&]() {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int kl = 0; kl < KPW; ++kl)
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    hx[vt][kl][e] = bf_rne_int(ht[vt][kl][e]);
                    reinterpret_cast<short*>(hB)[(16 * vt + c_row(l)) * kPadK + 16 * (KPW * wv + kl) + c_col(l, e)] = hx[vt][kl][e];
                }
    };
    loadW(0); loadU(0); loadG(0); loadK(0);
    write_h();
    for (int i = 0; i < p.NT; ++i) {
        const int nxt = i + 1 < p.NT ? i + 1 : i;
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int kl = 0; kl < KPW; ++kl)
                *reinterpret_cast<v4bh*>(p.Hn + (((chunk + i) * 8 + 2 * vs + vt) * 8 + KPW * wv + kl) * 256 + l * 4) = hx[vt][kl];
        lds_barrier();
        f4 acc[NV1];
        #pragma unroll
        for (int j = 0; j < NV1; ++j) acc[j] = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt)
            #pragma unroll
            for (int j = 0; j < NV1; ++j)
                acc[j] = mmac(wf[kt], ld8(hB + (16 * (v1a + j) + (l & 15)) * kPadK + 16 * kt + (l >> 4) * 4), acc[j]);
        loadW(nxt);
        #pragma unroll
        for (int j = 0; j < NV1; ++j) {
            v4bh x;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float uv = __uint_as_float(((uint32_t)(unsigned short)uf[j][e]) << 16);
                x[e] = bf_rne_int(uv - acc[j][e]);
                reinterpret_cast<short*>(vB)[(16 * (v1a + j) + c_col(l, e)) * kPadT + 16 * r1 + c_row(l)] = x[e];
            }
            *reinterpret_cast<v4bh*>(p.Vn + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1a + j) * 256 + l * 4) = x;
        }
        loadU(nxt);
        #pragma unroll
        for (int kl = 0; kl < KPW; ++kl) {
            f4 d;
            #pragma unroll
            for (int e = 0; e < 4; ++e) d[e] = g2_vcopy(gl[kl][e]);   // kra P-52: Gn = exp2f(G)
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) ht[vt][kl] *= d;
        }
        loadG(nxt);
        lds_barrier();
        #pragma unroll
        for (int rt = 0; rt < 4; ++rt)
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt) {
                const v4bh va = ld8(vB + (16 * vt + (l & 15)) * kPadT + 16 * rt + (l >> 4) * 4);
                #pragma unroll
                for (int kl = 0; kl < KPW; ++kl) ht[vt][kl] = mmac(va, kf[kl][rt], ht[vt][kl]);
            }
        loadK(nxt);
        write_h();
    }
}
extern "C" int g2_fwd_h_v7(const void* Wn, const void* Kn, const void* Un, const void* Gn,
                           void* Hn, void* Vn, int B, int T, int H, hipStream_t s) {
    if (T % kBT) return -1;
    FwdNParams p{(const bf16*)Wn, (const bf16*)Kn, (const bf16*)Un, (const float*)Gn,
                 (bf16*)Hn, (bf16*)Vn, H, T / kBT};
    hipLaunchKernelGGL(g2_fwd_h_v7_kernel<4>, dim3(kK / kBV, B * H), dim3(256), 0, s, p);
    return (int)hipGetLastError();
}
extern "C" int g2_fwd_h_v7w8(const void* Wn, const void* Kn, const void* Un, const void* Gn,
                             void* Hn, void* Vn, int B, int T, int H, hipStream_t s) {
    if (T % kBT) return -1;
    FwdNParams p{(const bf16*)Wn, (const bf16*)Kn, (const bf16*)Un, (const float*)Gn,
                 (bf16*)Hn, (bf16*)Vn, H, T / kBT};
    hipLaunchKernelGGL(g2_fwd_h_v7_kernel<8>, dim3(kK / kBV, B * H), dim3(512), 0, s, p);
    return (int)hipGetLastError();
}

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
    // Varlen/batch (DESIGN_VARLEN_BATCH.md): chunk-aligned segments, as in FwdOParams
    // (dhu_v8b only; the w8b variant ignores them and keeps a single segment).
    const int* seg; int nseg;
    // Recurrent-state gradient ports, fp32 [nseg, H, 128, 128] (FLA's [N, H, K, V]; ST instances
    // only): segment sg's reverse recurrence starts from DHt[sg] (null = zero) and writes the
    // gradient of its start state to DH0[sg] (null = skip).
    const float* DHt = nullptr; float* DH0 = nullptr;
};
extern "C" __global__ void __launch_bounds__(512) g2_dhu_w8b_kernel(BwdNParams p) {
    __shared__ __align__(16) bf16 dhB[kBV * kPadK];      // dh as [v][k]   (GEMM-A B operand)
    __shared__ __align__(16) bf16 vB[kBV * kPadT];       // dv2 as [v][t]  (GEMM-B A operand)
    __shared__ __align__(16) v4bh dOs[2][8 * 64];        // dO^T A-frags [vt*4+rt][lane], double buffered
    const int tid = threadIdx.x, wv = tid >> 6, l = tid & 63;
    const int vs = blockIdx.x, bh = blockIdx.y;
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
                hx[vt][e] = bf_rne_int(st[vt][e]);
                reinterpret_cast<short*>(dhB)[(16 * vt + c_row(l)) * kPadK + 16 * wv + c_col(l, e)] = hx[vt][e];
            }
    };
    const int last = p.NT - 1;
    loadKA(last); loadDV(last); loadG(last); loadQW(last); loadDO(last);
    write_dh();
    for (int s = 0; s < p.NT; ++s) {
        const int i = last - s, nxt = i > 0 ? i - 1 : 0;
        if (p.dh != nullptr) {
            #pragma unroll
            for (int vt = 0; vt < 2; ++vt)
                *reinterpret_cast<v4bh*>(p.dh + (((chunk + i) * 8 + 2 * vs + vt) * 8 + wv) * 256 + l * 4) = hx[vt];
        }
        dOs[s & 1][wv * 64 + l] = dof;                   // publish chunk i's dO^T slice
        loadDO(nxt);                                     // a full step ahead of its publish
        lds_barrier();                                   // dhB (entry state of chunk i) + dOs ready
        // bN(dh) emit: slot (kt=wv, vt, l, e) = dh[16*wv + pi(m)][16*(2*vs+vt) + 4*(l>>4) + e]
        if (p.HbN != nullptr) {
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
                x[e] = bf_rne_int(d);
                reinterpret_cast<short*>(vB)[(16 * v1 + c_col(l, e)) * kPadT + 16 * r1 + c_row(l)] = x[e];
            }
            *reinterpret_cast<v4bh*>(p.dv2 + (((chunk + i) * 4 + r1) * 8 + 2 * vs + v1) * 256 + l * 4) = x;
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

// ST: compile the DHt/DH0 ports in (false: the stateless instance is the pre-port kernel).
template <int MODE = 0, bool LIN = false, bool ST = false>
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
    if (c0 >= c1) {                                      // empty segment: dh0 = dht
        if constexpr (ST)
            if (p.DH0) {
                f4 t[2];
                g2_state_load(p.DHt, sg, p.H, bh, wv, vs, l, t);
                g2_state_store(p.DH0, sg, p.H, bh, wv, vs, l, t);
            }
        return;
    }
    const size_t chunk = (size_t)bh * p.NT;
    // kra P-938sa: this head's chunk blocks through CTA-uniform bases + 32-bit byte offsets (saddr global ops instead
    // of per-step per-lane 64-bit address math on the recurrence chain).  Offsets < NT x 16 KB.
    auto dga = [](auto* base, int e) {
        using E = std::remove_pointer_t<decltype(base)>;
        return reinterpret_cast<E*>(reinterpret_cast<std::conditional_t<std::is_const_v<E>, const char*, char*>>(base)
                                    + (uint32_t)e * (uint32_t)sizeof(E));
    };
    const bf16 *KgAc = p.KgA + chunk * 8192, *dvc = p.dv + chunk * 8192, *Qgc = p.Qg + chunk * 8192,
               *Wbc = p.Wb + chunk * 8192, *dOTc = p.dOT + chunk * 8192;
    const float* Gnc = p.Gn + chunk * 128;
    bf16* dv2c = p.dv2 + chunk * 8192;
    const int r1 = wv & 3, v1 = wv >> 2;                 // GEMM-A tile: t-tile r1, v-tile v1
    f4 st[2];                                            // DH^T C-frags [vt]: v=16vt+c_row, k=16wv+c_col
    if constexpr (ST)
        g2_state_load(p.DHt, sg, p.H, bh, wv, vs, l, st);
    else {
        st[0] = f4{0.f, 0.f, 0.f, 0.f};
        st[1] = st[0];
    }
    v4bh ka[8], qb[4], wb[4], dof, dvf; f4 gl;       // dof: this wave's 1/8 of dO^T
    auto loadKA = [&](int i) {
        const bf16* src = dga(KgAc, ((i * 4 + r1) * 8) * 256 + l * 8);      // kra L16: paired
        #pragma unroll
        for (int j = 0; j < 4; ++j) ld16x2(src + j * 512, ka[2 * j], ka[2 * j + 1]);
    };
    auto loadDV = [&](int i) { dvf = ld8(dga(dvc, ((i * 4 + r1) * 8 + 2 * vs + v1) * 256 + l * 4)); };
    auto loadG = [&](int i) { gl = *reinterpret_cast<const f4*>(dga(Gnc, (i * 8 + wv) * 16 + (l >> 4) * 4)); };
    auto loadQW = [&](int i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {                    // kra L16: paired
            ld16x2(dga(Qgc, ((i * 8 + wv) * 4) * 256 + j * 512 + l * 8), qb[2 * j], qb[2 * j + 1]);
            ld16x2(dga(Wbc, ((i * 8 + wv) * 4) * 256 + j * 512 + l * 8), wb[2 * j], wb[2 * j + 1]);
        }
    };
    auto loadDO = [&](int i) {
        dof = ld8(dga(dOTc, ((i * 8 + 2 * vs + (wv >> 2)) * 4 + (wv & 3)) * 256 + l * 4));
    };
    v4bh hx[2];
    auto write_dh = [&]() {
        #pragma unroll
        for (int vt = 0; vt < 2; ++vt)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                hx[vt][e] = bf_rne_int(st[vt][e]);
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
                x[e] = bf_rne_int(d);
                reinterpret_cast<short*>(vB)[(16 * v1 + c_col(l, e)) * kPadT + 16 * r1 + c_row(l)] = x[e];
            }
            if (hasDV2)  // T9b: the global dV2n write is optional; the
                *reinterpret_cast<v4bh*>(dga(dv2c, ((i * 4 + r1) * 8 + 2 * vs + v1) * 256 + l * 4)) = x;
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
    if constexpr (ST)
        if (p.DH0) g2_state_store(p.DH0, sg, p.H, bh, wv, vs, l, st);
}

// ================================================================ R1a-dhu: 4-wave form
// 256 threads / 4 waves.  Wave w owns GEMM tiles (r1=w, both v1 halves) and state
// K-blocks {w, w+4}: KgA is fetched ONCE per CTA per step (the 8-wave form loads it
// twice -- waves (r1,0)/(r1,1) read the same rows).  Every output element keeps its
// MMAC operand order and accumulation order -> bit-identical (bench: 1.027x H=12,
// 1.026x H=24, 1.093x H=48, occ 3->3, 185 VGPR 0 spill).  At deep saturation
// (H=96, 384 CTAs) latency hiding is WORSE (0.961x) -> host auto-selects by CTA
// count; KDA_G2_R1A_DHU=0/1 forces the 8-wave/4-wave form.
template <int MODE = 0, bool LIN = false, bool ST = false>
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
    if (c0 >= c1) {                                      // empty segment: dh0 = dht
        if constexpr (ST)
            if (p.DH0) {
                f4 t[2];
                g2_state_load(p.DHt, sg, p.H, bh, wv, vs, l, t);
                g2_state_store(p.DH0, sg, p.H, bh, wv, vs, l, t);
                g2_state_load(p.DHt, sg, p.H, bh, wv + 4, vs, l, t);
                g2_state_store(p.DH0, sg, p.H, bh, wv + 4, vs, l, t);
            }
        return;
    }
    const size_t chunk = (size_t)bh * p.NT;
    const int r1 = wv, kb0 = wv, kb1 = wv + 4;
    f4 st0[2], st1[2];
    if constexpr (ST) {
        g2_state_load(p.DHt, sg, p.H, bh, kb0, vs, l, st0);
        g2_state_load(p.DHt, sg, p.H, bh, kb1, vs, l, st1);
    } else {
        st0[0] = f4{0.f, 0.f, 0.f, 0.f}; st0[1] = st0[0];
        st1[0] = f4{0.f, 0.f, 0.f, 0.f}; st1[1] = st1[0];
    }
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
                hx[vt][e] = bf_rne_int(st[vt][e]);
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
    const int last = c1 - 1;                             // dh = dht (0 without ST) at the segment end
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
                x[e] = bf_rne_int(d);
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
    if constexpr (ST)
        if (p.DH0) {
            g2_state_store(p.DH0, sg, p.H, bh, kb0, vs, l, st0);
            g2_state_store(p.DH0, sg, p.H, bh, kb1, vs, l, st1);
        }
}

extern "C" int g2_dhu_w8b(const void* KgA, const void* Qg, const void* Wb, const void* dOT, const void* dv,
                          const void* Gn, void* dh, void* dv2, float scale, int B, int T, int H, hipStream_t s) {
    if (T % kBT) return -1;
    BwdNParams p{(const bf16*)KgA, (const bf16*)Qg, (const bf16*)Wb, (const bf16*)dOT, (const bf16*)dv,
                 (const float*)Gn, (bf16*)dh, (bf16*)dv2, scale, H, T / kBT};
    hipLaunchKernelGGL(g2_dhu_w8b_kernel, dim3(kK / kBV, B * H), dim3(512), 0, s, p);
    return (int)hipGetLastError();
}

extern "C" int g2_dhu_v8b(const void* KgA, const void* Qg, const void* Wb, const void* dOT, const void* dv,
                          const void* Gn, void* dv2, void* HbN, void* ANv2, float scale, int B, int T, int H,
                          const int* seg, int nseg, const void* DHt, void* DH0, hipStream_t s) {
    if (T % kBT || B != 1) return -1;                    // batch = segments (seg/nseg)
    if (seg == nullptr) nseg = 1;
    BwdNParams p{(const bf16*)KgA, (const bf16*)Qg, (const bf16*)Wb, (const bf16*)dOT, (const bf16*)dv,
                 (const float*)Gn, nullptr, (bf16*)dv2, scale, H, T / kBT, (const bf16*)HbN,
                 (bf16*)ANv2, seg, nseg};
    p.DHt = (const float*)DHt;
    p.DH0 = (float*)DH0;
    // auto window from MEASURED production A/B (dhu-only, kernel time): 48 CTAs
    // (H=12) -9.6%, 96 CTAs (H=24) +2.6% bench / +0.1% e2e, 192 CTAs (H=48) +4.6%,
    // 384 CTAs (H=96) -3.2%.  Low CTA counts need the 8-wave form's latency hiding;
    // deep saturation too; the dedup only pays in between.
    const long ctas = 4L * nseg * H;
    // kra P-85: re-measured on P-82b -- the 8-wave form wins from 128 CTAs on (H=48: 1.05-1.15x)
    int dhu4 = (ctas >= 96 && ctas <= g2_env_long("KDA_G2_DHU4_MAX_CTAS", 127)) ? 1 : 0;
    // kra P-938cp2: gfx938's CP summary pass (dh0 only, no hand-off written) stays on the 8-wave form -- 16K, H = 24
    // (96 CTAs): 0.85 -> 0.78 ms; the 4-wave form also loses at 48 CTAs (0.49 -> 0.60) and 192 (1.45 -> 1.77).
    if (HbN == nullptr && ANv2 == nullptr && dv2 == nullptr && g2_cur_gfx938()) dhu4 = 0;
    if (const char* e = g2::g2_getenv("KDA_G2_R1A_DHU")) dhu4 = atoi(e);
    // kra P-24: at deep saturation cap dhu at 1 CTA/CU (33 KB LDS per CTA): the operands shared
    // by a head's 4 V-slab CTAs then stay in L2.
    // kra P-938dt: gfx938 re-sweep (H = 64, 256 CTAs): off 1.13x at NT = 16, 1.01x at 32, 0.97x at 64, 0.83-0.93x at
    // 128-512 -- short chains have too few steps for the L2 sharing to pay; keep the throttle from 64 chunks on.
    const bool dthrottle = ctas >= g2_env_long("KDA_G2_DHU_THROTTLE_MIN_CTAS", 192) &&
                           (!g2_cur_gfx938() || T / kBT >= g2_env_long("KDA_G2_DHU_THROTTLE_MIN_NT", 64));
    auto dyn = [&](auto kern) -> size_t { return dthrottle ? g2_lds_topup(kern, 33 * 1024) : 0; };
    // kra P-22d: linear aN(dv2) emit only in the latency-bound regime (few CTAs); at deep saturation
    // the scattered stores pace the loads (R21).  KDA_G2_DHU_LIN=0/1 forces it.
    // kra P-938l: on gfx938 the linear emit wins at every grid (dhu 1.25x at 128 CTAs, 1.33x at 256, neutral at 32-64):
    // the scattered 2 B stores' issue stalls (SQTT: ~17% of dhu at 8192/64) cost more than the vB re-read
    bool lin = ctas <= 96 || g2_cur_gfx938();
    if (const char* e = g2::g2_getenv("KDA_G2_DHU_LIN")) lin = atoi(e) != 0;
    // kra P-10: the production output set gets the compile-time instance
    const bool prod = HbN != nullptr && ANv2 != nullptr && dv2 == nullptr;
    const bool prod2 = HbN != nullptr && ANv2 == nullptr && dv2 != nullptr;   // kra P-22
    if (DHt != nullptr || DH0 != nullptr) {             // state ports: own instances, the stateless ones stay as is
        if (dhu4) {
            if (lin) hipLaunchKernelGGL((g2_dhu_v8b4_kernel<0, true, true>), dim3(kK / kBV, nseg * H), dim3(256), dyn((g2_dhu_v8b4_kernel<0, true, true>)), s, p);
            else     hipLaunchKernelGGL((g2_dhu_v8b4_kernel<0, false, true>), dim3(kK / kBV, nseg * H), dim3(256), dyn((g2_dhu_v8b4_kernel<0, false, true>)), s, p);
        } else if (prod && lin) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, true, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, true, true>)), s, p);
        else if (prod) hipLaunchKernelGGL((g2_dhu_v8b_kernel<1, false, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<1, false, true>)), s, p);
        else      hipLaunchKernelGGL((g2_dhu_v8b_kernel<0, false, true>), dim3(kK / kBV, nseg * H), dim3(512), dyn((g2_dhu_v8b_kernel<0, false, true>)), s, p);
        return (int)hipGetLastError();
    }
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
