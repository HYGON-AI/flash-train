// B4 of the GDN-2 chunked backward: intra-chunk back-propagation + epilogue (FLA chunk_gdn2_bwd_intra +
// chunk_local_cumsum(reverse) + GQA group sum + dtype casts).
//
// Forward intra defines (per v-head, chunk-local rows t,s, channel kk, gc = chunk cumsum of g/ln2):
//     Aqk[t][s] = scale sum_kk q[t,kk] k[s,kk] E[t,s,kk]      (s <= t)
//     Akk[t][s] =       sum_kk (k b)[t,kk] k[s,kk] E[t,s,kk]  (s <  t),   E = 2^(gc_t - gc_s)
// With dAqk (B1, scale included) and dAkk (B3) the gradients through them are
//     dq2[t]  = sum_{s<=t} dAqk[t][s] k[s] E          D2[t] = sum_{s<t} dAkk[t][s] k[s] E
//     dkt[s]  = sum_{t>=s} dAqk[t][s] q[t] E + sum_{t>s} dAkk[t][s] (k b)[t] E
//     dq = dq_in + dq2             dk = dk_in + D2 b + dkt        db = db_in + D2 k
//     dgp = dg_in + q dq2 + (D2 b - dkt) k                        dg[t] = sum_{t' >= t, same chunk} dgp[t']
// dg is the gradient w.r.t. the natural-log g: d/dgc of 2^x carries ln2, gc = log2e * cumsum(g), ln2*log2e = 1,
// so no unit factor appears (FLA does the same: reverse cumsum, no scale).
//
// GQA (HV != H): the kernel is one block per (chunk, v-head); dq,dk are then written in place as fp32 over
// dq_in/dk_in ([B,S,HV,K]) and a second tiny kernel sums the HV/H heads of each key head in a fixed order into
// the bf16 dq,dk [B,S,H,K] (deterministic, no atomics).  With HV == H the first kernel writes the bf16 outputs directly.
//
// Per-channel decay as GEMM (forward intra's decay split, transposed).  The reference row r is the first row of
// the 16-row sub-block of the contracted index, so every applied exponent is <= 0 or bounded by 15 steps of decay
// (the >= 1 factor is clamped at 2^kExpClamp).  tf32 operands (fp32 dA, fp32 products).
//   wave w owns the 16 output rows of sub-block w for BOTH products:
//   dq2/D2 : tiles (w, tn<=w), contraction over s:  mmac(dA[t][s], k[s] 2^(gc_r(tn)-gc_s)), epilogue 2^(gc_t-gc_r(tn))
//   dkt    : tiles (tm>=w, w), contraction over t:  mmac(dA^T[s][t], q[t]/kb[t] 2^(gc_t-gc_r(w))), epilogue 2^(gc_r(w)-gc_s)
// Channels are processed 16 at a time; the mmac B-column n holds channel 4(n&3)+(n>>2), so a lane's D fragment is
// 4 consecutive channels (16 B fp32 global I/O, 4 lanes cover 64 contiguous bytes of a row).
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../common/flash_train_gdn2_bwd.h"
#include "../common/gdn2_chunk_mmac.h"

namespace {

using gdn2_mmac::bf2f;
using gdn2_mmac::f2bf;
using gdn2_mmac::u16x4;
using gdn2_mmac::v2f;
using gdn2_mmac::v4f;

constexpr int kC = 64;
constexpr int kThreads = 256;
constexpr int kStr = 20;  // LDS row stride (floats) of the [64][16] slab tiles
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kExpClamp = 120.0f;

// Block barrier waiting for LDS only: the __syncthreads builtin makes LLVM drain vmcnt (the prefetch) on gfx936.
__device__ __forceinline__ void lds_barrier() {
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_waitcnt(0xC07F);
    asm volatile("s_barrier" ::: "memory");
    asm volatile("" ::: "memory");
}

__device__ __forceinline__ float ex2(float x) { return __builtin_amdgcn_exp2f(x); }

template <int CTRL>
__device__ __forceinline__ float dpp_f(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), CTRL, 0xF, 0xF, false));
}
// Inclusive prefix sum over the 16 lanes of a DPP row.
__device__ __forceinline__ float row_scan(float x) {
    x += dpp_f<0x111>(x);
    x += dpp_f<0x112>(x);
    x += dpp_f<0x114>(x);
    x += dpp_f<0x118>(x);
    return x;
}
// Inclusive suffix sum (sum over lanes >= this lane) over the 16 lanes of a DPP row.
__device__ __forceinline__ float row_scan_rev(float x) {
    x += dpp_f<0x101>(x);
    x += dpp_f<0x102>(x);
    x += dpp_f<0x104>(x);
    x += dpp_f<0x108>(x);
    return x;
}

struct Slab {
    v4f g, b, dqi, dki, dbi, dgi;
    u16x4 q, k;
};

template <int K, bool F32OUT>
__global__ __launch_bounds__(kThreads) void gdn2_bwd_intra_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k, const float* __restrict__ g,
    const float* __restrict__ bg, const float* __restrict__ dAqk, const float* __restrict__ dAkk, float* dq_in,
    float* dk_in, const float* __restrict__ db_in, const float* dg_in, uint16_t* __restrict__ dq,
    uint16_t* __restrict__ dk, uint16_t* __restrict__ db, float* dg, int S, int H, int HV, int lane_order,
    float* dq_f32, float* dk_f32) {
    constexpr int NS = K / 16;
    __shared__ __align__(16) float sGc[kC * kStr];
    __shared__ __align__(16) float sQ[kC * kStr];
    __shared__ __align__(16) float sK[kC * kStr];
    __shared__ __align__(16) float sKB[kC * kStr];
    __shared__ __align__(16) float sGr[4 * 16];
    __shared__ __align__(16) float sTotF[4 * 16];
    __shared__ __align__(16) float sTotR[4 * 16];

    const int tid = threadIdx.x, lane = tid & 63;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int lo = lane & 15, hq = lane >> 4;
    const int cl = 4 * (lo & 3) + (lo >> 2);  // channel of the B-operand column held by this lane
    int id = blockIdx.x;
    const int hv = id % HV;  // v-head fastest: neighbouring blocks share q/k rows of a key head in L2
    id /= HV;
    const int hk = hv / (HV / H);
    const int NT = S / kC;
    const int n = id % NT, bb = id / NT;
    const std::size_t row0 = static_cast<std::size_t>(bb) * S + static_cast<std::size_t>(n) * kC;
    const int r = 16 * wave + lo;
    const std::size_t grow = row0 + r;

    // ---- dA fragments of this head (kept in registers for all channel slabs) ----
    // dq side: A[m = t = r][k = s = 16 tn + 4 hq + j]; dkt side: A[m = s = 16 wave + lo][k = t = 16 tm + 4 hq + j]
    v4f aq[4], ak[4], tq[4], tk[4];
#pragma unroll
    for (int t4 = 0; t4 < 4; ++t4) {
        if (t4 <= wave) {
            const std::size_t o = (grow * HV + hv) * kC + 16 * t4 + 4 * hq;
            aq[t4] = *reinterpret_cast<const v4f*>(dAqk + o);
            ak[t4] = *reinterpret_cast<const v4f*>(dAkk + o);
        } else {
            aq[t4] = v4f{0.f, 0.f, 0.f, 0.f};
            ak[t4] = aq[t4];
        }
        if (t4 >= wave) {
            v4f x, y;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::size_t o = ((row0 + 16 * t4 + 4 * hq + j) * HV + hv) * kC + 16 * wave + lo;
                x[j] = dAqk[o];
                y[j] = dAkk[o];
            }
            tq[t4] = x;
            tk[t4] = y;
        } else {
            tq[t4] = v4f{0.f, 0.f, 0.f, 0.f};
            tk[t4] = tq[t4];
        }
    }
    // diagonal tiles: dAqk valid for col <= row, dAkk for col < row
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int cc = 4 * hq + j;
#pragma unroll
        for (int t4 = 0; t4 < 4; ++t4) {
            if (t4 == wave) {
                if (cc > lo) aq[t4][j] = 0.f;
                if (cc >= lo) ak[t4][j] = 0.f;
                if (cc < lo) tq[t4][j] = 0.f;
                if (cc <= lo) tk[t4][j] = 0.f;
            }
        }
    }

    // lane-order partials (B3's store order, see the header): the 16 channels of slab ks are 2 lane groups x 2 v4f
    // of one (s, p) block; the four lanes hq of this slab read 4 x 256 B contiguous segments.
    const std::size_t ptile = (static_cast<std::size_t>(bb) * NT + n) * HV + hv;
    auto lane_idx = [&](int ks) {
        const int ch0 = ks * 16 + 4 * hq, s64 = ch0 >> 6, rem = ch0 & 63, pp = rem >> 5, rem2 = rem & 31;
        const int hq3 = rem2 >> 3, h2 = (rem2 & 7) >> 2;
        return ((((((ptile * (K / 64) + s64) * 2 + pp) * 2 + h2) * 4 + wave) * 64) + hq3 * 16 + lo) * 4;
    };
    auto load_slab = [&](int ks) {
        Slab s;
        const std::size_t oh = (grow * HV + hv) * K + ks * 16 + 4 * hq;
        const std::size_t okq = (grow * H + hk) * K + ks * 16 + 4 * hq;
        const std::size_t op = lane_order ? lane_idx(ks) : oh;
        s.g = *reinterpret_cast<const v4f*>(g + oh);
        s.b = *reinterpret_cast<const v4f*>(bg + oh);
        s.dqi = *reinterpret_cast<const v4f*>(dq_in + op);
        s.dki = *reinterpret_cast<const v4f*>(dk_in + op);
        s.dbi = *reinterpret_cast<const v4f*>(db_in + op);
        s.dgi = *reinterpret_cast<const v4f*>(dg_in + op);
        s.q = *reinterpret_cast<const u16x4*>(q + okq);
        s.k = *reinterpret_cast<const u16x4*>(k + okq);
        return s;
    };

    Slab cur = load_slab(0);
#pragma unroll 1
    for (int ks = 0; ks < NS; ++ks) {
        Slab nxt;
        if (ks + 1 < NS) nxt = load_slab(ks + 1);

        // ---- gc: cumsum of g * log2e over the 64 rows of the chunk ----
        v4f gc, qf, kf, kb;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            gc[i] = row_scan(cur.g[i] * kLog2e);
            qf[i] = bf2f(cur.q[i]);
            kf[i] = bf2f(cur.k[i]);
            kb[i] = kf[i] * cur.b[i];
        }
        if (lo == 15) *reinterpret_cast<v4f*>(sTotF + wave * 16 + 4 * hq) = gc;
        lds_barrier();
#pragma unroll
        for (int w2 = 0; w2 < 4; ++w2)
            if (w2 < wave) gc += *reinterpret_cast<const v4f*>(sTotF + w2 * 16 + 4 * hq);
        *reinterpret_cast<v4f*>(sGc + r * kStr + 4 * hq) = gc;
        *reinterpret_cast<v4f*>(sQ + r * kStr + 4 * hq) = qf;
        *reinterpret_cast<v4f*>(sK + r * kStr + 4 * hq) = kf;
        *reinterpret_cast<v4f*>(sKB + r * kStr + 4 * hq) = kb;
        if (lo == 0) *reinterpret_cast<v4f*>(sGr + wave * 16 + 4 * hq) = gc;
        lds_barrier();

        // ---- dq2, D2: output rows of sub-block `wave`, contraction over s in sub-blocks tn <= wave ----
        v4f dq2 = {0.f, 0.f, 0.f, 0.f}, d2 = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int tn = 0; tn < 4; ++tn) {
            if (tn <= wave) {
                const float gr = sGr[tn * 16 + cl];
                float kd[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int s = 16 * tn + 4 * hq + j;
                    kd[j] = sK[s * kStr + cl] * ex2(fminf(gr - sGc[s * kStr + cl], kExpClamp));
                }
                v4f accq = {0.f, 0.f, 0.f, 0.f}, acck = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const v2f bk = {kd[2 * c], kd[2 * c + 1]};
                    accq = __builtin_amdgcn_mmac_f32_16x16x8tf32(v2f{aq[tn][2 * c], aq[tn][2 * c + 1]}, bk, accq);
                    acck = __builtin_amdgcn_mmac_f32_16x16x8tf32(v2f{ak[tn][2 * c], ak[tn][2 * c + 1]}, bk, acck);
                }
                const v4f grv = *reinterpret_cast<const v4f*>(sGr + tn * 16 + 4 * hq);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float f = ex2(gc[i] - grv[i]);
                    dq2[i] += accq[i] * f;
                    d2[i] += acck[i] * f;
                }
            }
        }

        // ---- dkt: output rows s of sub-block `wave`, contraction over t in sub-blocks tm >= wave ----
        v4f dkt = {0.f, 0.f, 0.f, 0.f};
        {
            const float gw = sGr[wave * 16 + cl];
#pragma unroll
            for (int tm = 0; tm < 4; ++tm) {
                if (tm >= wave) {
                    float qd[4], kbd[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int t = 16 * tm + 4 * hq + j;
                        const float e = ex2(sGc[t * kStr + cl] - gw);
                        qd[j] = sQ[t * kStr + cl] * e;
                        kbd[j] = sKB[t * kStr + cl] * e;
                    }
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        dkt = __builtin_amdgcn_mmac_f32_16x16x8tf32(v2f{tq[tm][2 * c], tq[tm][2 * c + 1]},
                                                                      v2f{qd[2 * c], qd[2 * c + 1]}, dkt);
                        dkt = __builtin_amdgcn_mmac_f32_16x16x8tf32(v2f{tk[tm][2 * c], tk[tm][2 * c + 1]},
                                                                      v2f{kbd[2 * c], kbd[2 * c + 1]}, dkt);
                    }
                }
            }
            const v4f grv = *reinterpret_cast<const v4f*>(sGr + wave * 16 + 4 * hq);
#pragma unroll
            for (int i = 0; i < 4; ++i) dkt[i] *= ex2(fminf(grv[i] - gc[i], kExpClamp));
        }

        // ---- epilogue ----
        v4f dq_o, dk_o, db_o, dgp;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float d2b = d2[i] * cur.b[i];
            dq_o[i] = dq2[i] + cur.dqi[i];
            dk_o[i] = d2b + dkt[i] + cur.dki[i];
            db_o[i] = d2[i] * kf[i] + cur.dbi[i];
            dgp[i] = qf[i] * dq2[i] + (d2b - dkt[i]) * kf[i] + cur.dgi[i];
            dgp[i] = row_scan_rev(dgp[i]);
        }
        if (lo == 0) *reinterpret_cast<v4f*>(sTotR + wave * 16 + 4 * hq) = dgp;
        lds_barrier();
#pragma unroll
        for (int w2 = 0; w2 < 4; ++w2)
            if (w2 > wave) dgp += *reinterpret_cast<const v4f*>(sTotR + w2 * 16 + 4 * hq);

        const std::size_t oh = (grow * HV + hv) * K + ks * 16 + 4 * hq;
        *reinterpret_cast<v4f*>(dg + oh) = dgp;
        *reinterpret_cast<u16x4*>(db + oh) = u16x4{f2bf(db_o[0]), f2bf(db_o[1]), f2bf(db_o[2]), f2bf(db_o[3])};
        if constexpr (F32OUT) {
            float* pq = dq_f32 ? dq_f32 : dq_in;
            float* pk = dk_f32 ? dk_f32 : dk_in;
            *reinterpret_cast<v4f*>(pq + oh) = dq_o;
            *reinterpret_cast<v4f*>(pk + oh) = dk_o;
        } else {
            const std::size_t okq = (grow * H + hk) * K + ks * 16 + 4 * hq;
            *reinterpret_cast<u16x4*>(dq + okq) = u16x4{f2bf(dq_o[0]), f2bf(dq_o[1]), f2bf(dq_o[2]), f2bf(dq_o[3])};
            *reinterpret_cast<u16x4*>(dk + okq) = u16x4{f2bf(dk_o[0]), f2bf(dk_o[1]), f2bf(dk_o[2]), f2bf(dk_o[3])};
        }
        cur = nxt;
    }
}

// dq[b,s,hk,:] = bf16(sum_{g<G} dq_f32[b,s,hk*G+g,:]), same for dk.  One thread = 8 channels (16 B bf16 store).
template <int K>
__global__ __launch_bounds__(256) void gdn2_bwd_group_sum_kernel(const float* __restrict__ dqf, const float* __restrict__ dkf,
                                                                  uint16_t* __restrict__ dq, uint16_t* __restrict__ dk,
                                                                  std::size_t rows_h, int G, int H) {
    typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));
    constexpr int PER = K / 8;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows_h * PER) return;
    const std::size_t row = idx / PER;  // (b*S + s)*H + hk
    const int c = static_cast<int>(idx % PER) * 8;
    const std::size_t bs = row / H;
    const int hk = static_cast<int>(row % H);
    const std::size_t base = ((bs * H + hk) * G) * K + c;
    v4f aq0 = {0.f, 0.f, 0.f, 0.f}, aq1 = aq0, ak0 = aq0, ak1 = aq0;
    for (int gg = 0; gg < G; ++gg) {
        aq0 += *reinterpret_cast<const v4f*>(dqf + base + static_cast<std::size_t>(gg) * K);
        aq1 += *reinterpret_cast<const v4f*>(dqf + base + static_cast<std::size_t>(gg) * K + 4);
        ak0 += *reinterpret_cast<const v4f*>(dkf + base + static_cast<std::size_t>(gg) * K);
        ak1 += *reinterpret_cast<const v4f*>(dkf + base + static_cast<std::size_t>(gg) * K + 4);
    }
    u16x8 oq, ok;
    for (int i = 0; i < 4; ++i) {
        oq[i] = f2bf(aq0[i]);
        oq[4 + i] = f2bf(aq1[i]);
        ok[i] = f2bf(ak0[i]);
        ok[4 + i] = f2bf(ak1[i]);
    }
    *reinterpret_cast<u16x8*>(dq + row * K + c) = oq;
    *reinterpret_cast<u16x8*>(dk + row * K + c) = ok;
}

}  // namespace

extern "C" int flash_train_gdn2_bwd_intra_bf16(const FlashTrainGDN2BwdIntraArgs* a, void* stream) {
    if (!a || !a->q || !a->k || !a->g || !a->b || !a->dAqk || !a->dAkk || !a->dq_in || !a->dk_in || !a->db_in ||
        !a->dg_in || !a->dq || !a->dk || !a->db || !a->dg)
        return 1;
    if (a->B <= 0 || a->S <= 0 || a->S % kC || a->H <= 0 || a->HV <= 0 || a->HV % a->H || (a->K != 64 && a->K != 128))
        return 2;
    if (a->partials_lane_order && a->HV != a->H && (!a->dq_f32 || !a->dk_f32)) return 1;
    if (a->partials_lane_order && a->dg == a->dg_in) return 2;  // lane-order partials cannot be updated in place
    hipStream_t st = static_cast<hipStream_t>(stream);
    const dim3 grid(static_cast<unsigned>(static_cast<std::size_t>(a->S / kC) * a->B * a->HV));
    const bool gqa = a->HV != a->H;
#define L(KK, FF)                                                                                                \
    if (a->K == KK && gqa == FF) {                                                                               \
        hipLaunchKernelGGL((gdn2_bwd_intra_kernel<KK, FF>), grid, dim3(kThreads), 0, st, a->q, a->k, a->g, a->b, \
                           a->dAqk, a->dAkk, a->dq_in, a->dk_in, a->db_in, a->dg_in, a->dq, a->dk, a->db,         \
                           a->dg, a->S, a->H, a->HV, a->partials_lane_order, a->dq_f32, a->dk_f32);                                                            \
        if (hipGetLastError() != hipSuccess) return 5;                                                           \
        if (FF) {                                                                                                \
            const std::size_t rows_h = static_cast<std::size_t>(a->B) * a->S * a->H;                            \
            const std::size_t items = rows_h * (KK / 8);                                                         \
            hipLaunchKernelGGL((gdn2_bwd_group_sum_kernel<KK>), dim3(static_cast<unsigned>((items + 255) / 256)), \
                               dim3(256), 0, st, a->dq_f32 ? a->dq_f32 : a->dq_in, a->dk_f32 ? a->dk_f32 : a->dk_in, a->dq, a->dk, rows_h, a->HV / a->H, a->H); \
        }                                                                                                        \
        return hipGetLastError() == hipSuccess ? 0 : 5;                                                          \
    }
    L(128, false) L(128, true) L(64, false) L(64, true)
#undef L
    return 2;
}
