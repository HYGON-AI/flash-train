// B2 dhu: serial reverse recursion of the state gradient for the GDN-2 backward
// (FLA: chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 with gk = per-channel cumulative log2 gate,
//  q = qg, k = kg, w = w_wy, as called from fla/ops/gdn2/chunk_bwd.py).
//
// Per sequence/head and per value tile of width BV = 64, walks the chunks from the last to the first.
// With dh the state gradient at the END of chunk n (dht, or 0, for the last chunk) and gclast the
// chunk-end row of the chunk-local cumulative log2 gate (per key channel kk):
//     dh_store[n] = bf16(dh)
//     dv2[t]      = dv_local[t] + kg_t . dh                                   (t in chunk n)
//     dh[:, kk]   = 2^(gclast[kk]) dh[:, kk] + scale * sum_t qg_t^T do_t - sum_t w_t^T dv2_t
// qg = q 2^(gc_t) and kg = k 2^(gclast - gc_t) already carry the per-token gate, so unlike the scalar
// GDN kernel nothing is rescaled per token; the only gate left is the per-channel decay of the state
// columns, applied to the fp32 registers (the D layout hands a lane 4 consecutive kk per tile -> one
// 16 B gclast load per tile).  After chunk 0 the remaining dh is dh0.
//
// Structure = flash_train_gdn_bwd_dhu.hip: every global access is 16 B, fp32 state in registers
// (dh^T[v][kk], mmac D layout), bf16 copies staged in LDS, two block barriers per chunk, next chunk's
// loads issued right after the first barrier.
//  * kg . dh (contraction over K): A = kg fragments straight from global (wave owns 16 rows t), B = dh^T
//    rows from LDS; B rows permuted so a lane ends with 8 consecutive v of row t -> dv loaded / dv2
//    stored as 16 B pieces directly from/to registers.
//  * the two update GEMMs contract over t: qg, w are transposed through a wave-private LDS region X
//    (the wave's own KW columns), do^T through sDT (shared); qg^T do runs between the barriers, only
//    w^T dv2 (A operand -dv2^T through sVT) follows the second barrier.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../common/flash_train_gdn2_bwd.h"
#include "../common/gdn2_chunk_mmac.h"

namespace {

using gdn2_mmac::bf2f;
using gdn2_mmac::f2bf;
using gdn2_mmac::u16x4;
using gdn2_mmac::v4f;
using gdn2_mmac::v4s;
typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));

constexpr int kC = 64;   // chunk length (FLA BT)
constexpr int kBV = 64;  // value columns per block
constexpr int kNW = 4;   // waves per block
constexpr int kThreads = 64 * kNW;

// Block barrier waiting for LDS only (global memory is never read back in the loop).
__device__ __forceinline__ void lds_barrier() {
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_waitcnt(0xC07F);
    asm volatile("s_barrier" ::: "memory");  // not the builtin: it makes LLVM drain vmcnt(0) before every barrier on gfx936
    asm volatile("" ::: "memory");
}

template <int K, bool DH_VK>
__global__ __launch_bounds__(kThreads) void gdn2_bwd_dhu_kernel(
    const uint16_t* __restrict__ qg, const uint16_t* __restrict__ kg, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ dout, const uint16_t* __restrict__ dv, const float* __restrict__ gl,
    const float* __restrict__ dht, uint16_t* __restrict__ dh, uint16_t* __restrict__ dv2,
    float* __restrict__ dh0, int S, int HV, int V, float scale) {
    constexpr int KT16 = K / 16;
    constexpr int KW = K / kNW;   // state columns (kk) owned by a wave
    constexpr int TKW = KW / 16;  // 16-column tiles per wave
    constexpr int TPW = 4 * TKW;  // dh^T tiles per wave: 4 v-tiles x TKW
    constexpr int NK8 = KW / 8;   // consecutive t rows per lane of the qg / w load (= lanes per row)
    constexpr int XSTR = kC + 4;  // X[kk][t]
    constexpr int HSTR = K + 8;   // sDh[v][kk]  (rows stay 16B aligned)
    constexpr int DSTR = kC + 4;  // sDT[v][t]
    constexpr int VTSTR = kC + 4; // sVT[v][t]
    constexpr int NH8 = K * kBV / 8 / kThreads;  // u16x8 of dh per thread
    constexpr int KSTR = K + 8;   // kg tile row stride (16 rows per wave, shares the wave's X region)
    constexpr int XREG = (KW * XSTR > 16 * KSTR) ? KW * XSTR : 16 * KSTR;  // X region per wave (elements)
    static_assert(KW % 16 == 0 && NK8 * 8 == KW && NH8 >= 1, "wave split");

    __shared__ __align__(16) unsigned short sDh[kBV * HSTR];
    __shared__ __align__(16) unsigned short sX[kNW * XREG];  // wave-private: kg tile, then qg^T, then w^T
    __shared__ __align__(16) unsigned short sDT[kBV * DSTR];
    __shared__ __align__(16) unsigned short sVT[kBV * VTSTR];

    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int lo = lane & 15, hq = lane >> 4, hi4 = hq * 4;
    const int v0 = blockIdx.x * kBV;
    const int bhv = blockIdx.y;
    const int b = bhv / HV, hv = bhv % HV;
    const int NT = S / kC;
    const int pl = 4 * (lo & 3) + (lo >> 2);  // mmac row permutation: D column hq+4r <-> index 4hq+r
    const int tw = wave * 16 + lo;            // GEMM1 row (t) of this lane

    // fp32 state: tile i -> (tv = i / TKW, tk = wave*TKW + i % TKW); lane holds
    // dh^T[tv*16+lo][tk*16+4hq+r], r = 0..3.
    v4f hreg[TPW];
#pragma unroll
    for (int i = 0; i < TPW; ++i) {
        const int tv = i / TKW, tk = wave * TKW + i % TKW;
        if (dht == nullptr) {
            hreg[i] = v4f{0.f, 0.f, 0.f, 0.f};
        } else if (DH_VK) {  // dht[bhv][v][kk]
            hreg[i] = *reinterpret_cast<const v4f*>(
                dht + (static_cast<std::size_t>(bhv) * V + v0 + tv * 16 + lo) * K + tk * 16 + 4 * hq);
        } else {             // dht[bhv][kk][v]
#pragma unroll
            for (int r = 0; r < 4; ++r)
                hreg[i][r] = dht[(static_cast<std::size_t>(bhv) * K + tk * 16 + 4 * hq + r) * V + v0 + tv * 16 + lo];
        }
    }

    // ---- addressing: wave-uniform bases + per-lane 32-bit offsets ----
    const std::size_t rowHVK = static_cast<std::size_t>(HV) * K;
    const std::size_t rowHVV = static_cast<std::size_t>(HV) * V;
    const uint16_t* kB = kg + static_cast<std::size_t>(b) * S * rowHVK + static_cast<std::size_t>(hv) * K;
    const uint16_t* qB = qg + static_cast<std::size_t>(b) * S * rowHVK + static_cast<std::size_t>(hv) * K;
    const uint16_t* wB = w + static_cast<std::size_t>(b) * S * rowHVK + static_cast<std::size_t>(hv) * K;
    const uint16_t* doB = dout + static_cast<std::size_t>(b) * S * rowHVV + static_cast<std::size_t>(hv) * V + v0;
    const uint16_t* dvB = dv + static_cast<std::size_t>(b) * S * rowHVV + static_cast<std::size_t>(hv) * V + v0;
    uint16_t* dv2B = dv2 + static_cast<std::size_t>(b) * S * rowHVV + static_cast<std::size_t>(hv) * V + v0;
    // gclast[b][n][hv][kk]: this lane's 4 consecutive kk of each of the wave's TKW tiles
    const float* glB = gl + (static_cast<std::size_t>(b) * NT * HV + hv) * K + wave * KW + 4 * hq;

    // kg: coalesced rows (8 lanes x 16 B = 128 B per row, 8 rows per load); instruction j covers rows 8(j / (K/64)) + lane/8 of
    // the wave's 16 rows and columns 64 (j % (K/64)) + 8 (lane % 8) .. +7.  The tile is turned into A fragments through LDS.
    const unsigned offK = static_cast<unsigned>((wave * 16 + (lane >> 3)) * rowHVK + 8 * (lane & 7));
    // qg / w: NK8 consecutive lanes take consecutive 16 B of one row (the wave's KW columns); a lane
    // owns NK8 consecutive t rows, so the transposed LDS write packs NK8 t per access.
    const int rt0 = (lane / NK8) * NK8;
    const int xc = (lane % NK8) * 8;  // column of the piece inside the wave's KW columns
    const unsigned offQ = static_cast<unsigned>(rt0 * rowHVK + wave * KW + xc);
    // do: 8 lanes per row (128 B), a lane owns 2 consecutive t rows.
    const int dt0 = 2 * (tid / 8), dv8 = (tid % 8) * 8;
    const unsigned offD = static_cast<unsigned>(dt0 * rowHVV + dv8);
    // dv / dv2: a lane owns row tw, 8 consecutive v at 32P + 8hq (P = 0, 1).
    const unsigned offV = static_cast<unsigned>(tw * rowHVV + 8 * hq);

    u16x8 pa[KT16 / 2];  // kg tile pieces of this lane (see offK)
    u16x8 pq[NK8], pw[NK8], pd[2], pv[2];
    v4f pgl[TKW];
    auto prefetch_a = [&](int n) {  // consumed before the first barrier or right after it
        const int t0 = n * kC;
        const uint16_t* kc = kB + static_cast<std::size_t>(t0) * rowHVK;
#pragma unroll
        for (int j = 0; j < KT16 / 2; ++j)
            pa[j] = *reinterpret_cast<const u16x8*>(kc + offK + static_cast<std::size_t>(j / (K / 64)) * 8 * rowHVK + (j % (K / 64)) * 64);
        const uint16_t* qc = qB + static_cast<std::size_t>(t0) * rowHVK;
#pragma unroll
        for (int r = 0; r < NK8; ++r) pq[r] = *reinterpret_cast<const u16x8*>(qc + offQ + static_cast<unsigned>(r * rowHVK));
        const uint16_t* dc = doB + static_cast<std::size_t>(t0) * rowHVV;
#pragma unroll
        for (int r = 0; r < 2; ++r) pd[r] = *reinterpret_cast<const u16x8*>(dc + offD + static_cast<unsigned>(r * rowHVV));
        const float* gc = glB + static_cast<std::size_t>(n) * rowHVK;
#pragma unroll
        for (int j = 0; j < TKW; ++j) pgl[j] = *reinterpret_cast<const v4f*>(gc + 16 * j);
    };
    auto prefetch_w = [&](int n) {
        const uint16_t* wc = wB + static_cast<std::size_t>(n * kC) * rowHVK;
#pragma unroll
        for (int r = 0; r < NK8; ++r) pw[r] = *reinterpret_cast<const u16x8*>(wc + offQ + static_cast<unsigned>(r * rowHVK));
    };
    auto prefetch_v = [&](int n) {
        const uint16_t* vc = dvB + static_cast<std::size_t>(n * kC) * rowHVV;
#pragma unroll
        for (int P = 0; P < 2; ++P) pv[P] = *reinterpret_cast<const u16x8*>(vc + offV + 32 * P);
    };
    prefetch_a(NT - 1);
    prefetch_w(NT - 1);
    prefetch_v(NT - 1);

    unsigned short* xw = sX + wave * XREG;  // this wave's transposed-operand region
    // Transposed write of NK8 t rows x 8 columns held in registers into X[col][t].
    auto write_x = [&](const u16x8* p) {
#pragma unroll
        for (int qq = 0; qq < 8; ++qq) {
            unsigned short* xp = xw + (xc + qq) * XSTR + rt0;
            if constexpr (NK8 == 4)
                *reinterpret_cast<u16x4*>(xp) = u16x4{p[0][qq], p[1][qq], p[2][qq], p[3][qq]};
            else
                *reinterpret_cast<unsigned*>(xp) = static_cast<unsigned>(p[0][qq]) | (static_cast<unsigned>(p[1][qq]) << 16);
        }
    };

    for (int n = NT - 1; n >= 0; --n) {
        const int t0 = n * kC;

        // ---- kg tile -> wave-private X (free here: the previous chunk's last GEMM has read w^T, qg^T is written after barrier 1) ----
#pragma unroll
        for (int j = 0; j < KT16 / 2; ++j)
            *reinterpret_cast<u16x8*>(xw + (8 * (j / (K / 64)) + (lane >> 3)) * KSTR + 64 * (j % (K / 64)) + 8 * (lane & 7)) = pa[j];

        // ---- stage this chunk: dh bf16, do^T -> LDS ----
#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            const int tv = i / TKW, tk = wave * TKW + i % TKW;
            *reinterpret_cast<u16x4*>(sDh + (tv * 16 + lo) * HSTR + tk * 16 + 4 * hq) =
                u16x4{f2bf(hreg[i][0]), f2bf(hreg[i][1]), f2bf(hreg[i][2]), f2bf(hreg[i][3])};
        }
#pragma unroll
        for (int qq = 0; qq < 8; ++qq)
            *reinterpret_cast<unsigned*>(sDT + (dv8 + qq) * DSTR + dt0) =
                static_cast<unsigned>(pd[0][qq]) | (static_cast<unsigned>(pd[1][qq]) << 16);
        v4f cg[TKW];
#pragma unroll
        for (int j = 0; j < TKW; ++j) cg[j] = pgl[j];
        lds_barrier();

        // ---- kg A fragments kg[tw][32j + 8hq .. +7] (reduction order permuted consistently with the sDh reads) out of the tile;
        //      issued after the barrier so the write->read LDS latency hides behind it, and before qg^T overwrites X ----
        u16x8 ca[KT16 / 2];
        asm volatile("" ::: "memory");
#pragma unroll
        for (int j = 0; j < KT16 / 2; ++j) ca[j] = *reinterpret_cast<const u16x8*>(xw + lo * KSTR + 32 * j + 8 * hq);
        asm volatile("" ::: "memory");

        // ---- qg^T into X ----
        write_x(pq);
        if (n > 0) prefetch_a(n - 1);

        // ---- dh rows out of sDh: ready since the barrier ----
        {
            if constexpr (DH_VK) {
                uint16_t* hc = dh + ((static_cast<std::size_t>(b) * NT + n) * HV + hv) * V * K + static_cast<std::size_t>(v0) * K;
#pragma unroll
                for (int r = 0; r < NH8; ++r) {
                    const int e = tid + r * kThreads, vr = e / (K / 8), c = (e % (K / 8)) * 8;
                    *reinterpret_cast<u16x8*>(hc + vr * K + c) = *reinterpret_cast<const u16x8*>(sDh + vr * HSTR + c);
                }
            } else {
                uint16_t* hc = dh + ((static_cast<std::size_t>(b) * NT + n) * HV + hv) * K * V + v0;
                constexpr int UPR = kBV / 8, HKSTEP = kThreads / UPR;
                const int hkk = tid / UPR, uv8 = (tid % UPR) * 8;
#pragma unroll
                for (int r = 0; r < NH8; ++r) {
                    const int kk = hkk + r * HKSTEP;
                    u16x8 x;
#pragma unroll
                    for (int qq = 0; qq < 8; ++qq) x[qq] = sDh[(uv8 + qq) * HSTR + kk];
                    *reinterpret_cast<u16x8*>(hc + static_cast<unsigned>(kk * V) + uv8) = x;
                }
            }
        }

        // ---- dh = diag(2^gclast) dh + scale qg^T do   (independent of dh(n): off the chain) ----
#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            const int tv = i / TKW, tkl = i % TKW;
            const unsigned short* ap = sDT + (tv * 16 + lo) * DSTR + hi4;
            const unsigned short* bp = xw + (tkl * 16 + pl) * XSTR + hi4;
            v4f acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int k0 = 0; k0 < kC; k0 += 16)
                acc = __builtin_amdgcn_mmac_f32_16x16x16bf16(*reinterpret_cast<const v4s*>(ap + k0),
                                                             *reinterpret_cast<const v4s*>(bp + k0), acc);
#pragma unroll
            for (int r = 0; r < 4; ++r) hreg[i][r] = hreg[i][r] * __builtin_amdgcn_exp2f(cg[tkl][r]) + acc[r] * scale;
        }

        // ---- w^T into X (qg^T is dead) ----
        write_x(pw);
        if (n > 0) prefetch_w(n - 1);

        // ---- GEMM1: dv2[t][v] = dv + sum_kk kg[t][kk] dh^T[v][kk] ----
        {
            uint16_t* oc = dv2B + static_cast<std::size_t>(t0) * rowHVV + offV;
#pragma unroll
            for (int P = 0; P < 2; ++P) {
                // acc0[i] <-> v = 32P + 8hq + i, acc1[i] <-> v = 32P + 8hq + 4 + i: B lane lo = hq + 4i
                // loads row 32P + 8hq + i (resp. +4) of sDh.
                const unsigned short* b0 = sDh + (32 * P + 8 * (lo & 3) + (lo >> 2)) * HSTR + 8 * hq;
                const unsigned short* b1 = b0 + 4 * HSTR;
                v4f acc0 = {0.f, 0.f, 0.f, 0.f}, acc1 = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int jj = 0; jj < KT16 / 2; ++jj) {
                    const u16x8 bb0 = *reinterpret_cast<const u16x8*>(b0 + 32 * jj);
                    const u16x8 bb1 = *reinterpret_cast<const u16x8*>(b1 + 32 * jj);
                    const v4s alo = __builtin_bit_cast(v4s, __builtin_shufflevector(ca[jj], ca[jj], 0, 1, 2, 3));
                    const v4s ahi = __builtin_bit_cast(v4s, __builtin_shufflevector(ca[jj], ca[jj], 4, 5, 6, 7));
                    acc0 = __builtin_amdgcn_mmac_f32_16x16x16bf16(alo, __builtin_bit_cast(v4s, __builtin_shufflevector(bb0, bb0, 0, 1, 2, 3)), acc0);
                    acc1 = __builtin_amdgcn_mmac_f32_16x16x16bf16(alo, __builtin_bit_cast(v4s, __builtin_shufflevector(bb1, bb1, 0, 1, 2, 3)), acc1);
                    acc0 = __builtin_amdgcn_mmac_f32_16x16x16bf16(ahi, __builtin_bit_cast(v4s, __builtin_shufflevector(bb0, bb0, 4, 5, 6, 7)), acc0);
                    acc1 = __builtin_amdgcn_mmac_f32_16x16x16bf16(ahi, __builtin_bit_cast(v4s, __builtin_shufflevector(bb1, bb1, 4, 5, 6, 7)), acc1);
                }
                u16x8 o;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float x = bf2f(pv[P][e]) + (e < 4 ? acc0[e] : acc1[e - 4]);
                    const unsigned short ob = f2bf(x);
                    o[e] = ob;
                    sVT[(32 * P + 8 * hq + e) * VTSTR + tw] = static_cast<unsigned short>(ob ^ 0x8000u);  // -dv2
                }
                *reinterpret_cast<u16x8*>(oc + 32 * P) = o;
            }
        }
        if (n > 0) prefetch_v(n - 1);

        lds_barrier();

        // ---- dh -= w^T dv2 ----
#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            const int tv = i / TKW, tkl = i % TKW;
            const unsigned short* ap = sVT + (tv * 16 + lo) * VTSTR + hi4;
            const unsigned short* bp = xw + (tkl * 16 + pl) * XSTR + hi4;
            v4f acc = hreg[i];
#pragma unroll
            for (int k0 = 0; k0 < kC; k0 += 16)
                acc = __builtin_amdgcn_mmac_f32_16x16x16bf16(*reinterpret_cast<const v4s*>(ap + k0),
                                                             *reinterpret_cast<const v4s*>(bp + k0), acc);
            hreg[i] = acc;
        }
    }

    if (dh0 != nullptr) {
#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            const int tv = i / TKW, tk = wave * TKW + i % TKW;
            if constexpr (DH_VK) {  // dh0[bhv][v][kk]
                *reinterpret_cast<v4f*>(dh0 + (static_cast<std::size_t>(bhv) * V + v0 + tv * 16 + lo) * K + tk * 16 + 4 * hq) = hreg[i];
            } else {      // dh0[bhv][kk][v]
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    dh0[(static_cast<std::size_t>(bhv) * K + tk * 16 + 4 * hq + r) * V + v0 + tv * 16 + lo] = hreg[i][r];
            }
        }
    }
}

template <int K, bool DH_VK>
int launch(const FlashTrainGDN2BwdDhuArgs& a, hipStream_t st) {
    dim3 grid(a.V / kBV, a.B * a.HV);
    hipLaunchKernelGGL((gdn2_bwd_dhu_kernel<K, DH_VK>), grid, dim3(kThreads), 0, st, a.qg, a.kg, a.w, a.do_, a.dv_local,
                       a.gclast, a.dht, a.dh, a.dv2, a.dh0, a.S, a.HV, a.V, a.scale);
    return hipGetLastError() == hipSuccess ? 0 : 5;
}

}  // namespace

extern "C" int flash_train_gdn2_bwd_dhu_bf16(const FlashTrainGDN2BwdDhuArgs* a, void* stream) {
    if (a == nullptr || a->qg == nullptr || a->kg == nullptr || a->w == nullptr || a->do_ == nullptr ||
        a->dv_local == nullptr || a->gclast == nullptr || a->dh == nullptr || a->dv2 == nullptr)
        return 1;
    if (a->B <= 0 || a->S <= 0 || a->HV <= 0 || a->S % kC != 0 || a->V % kBV != 0) return 2;
    hipStream_t st = static_cast<hipStream_t>(stream);
    if (a->K == 128) return a->dh_vk ? launch<128, true>(*a, st) : launch<128, false>(*a, st);
    if (a->K == 64) return a->dh_vk ? launch<64, true>(*a, st) : launch<64, false>(*a, st);
    return 2;
}
