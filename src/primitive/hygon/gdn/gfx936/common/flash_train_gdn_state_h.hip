// Inter-chunk state recursion for the large-shape GDN forward route
// (stage 3 of 4: kkt_solve -> w_u -> state_h -> chunk_o).
//
// Per sequence/head and per value tile of width BV, walks the chunks serially:
//     h_store[n]   = bf16(h)                                   (state at chunk start)
//     v_new        = u - w @ h                                 (C x V)
//     h            = exp2(g_last) * h + k^T @ (exp2(g_last - g) * v_new)
// with g the chunk-local cumulative log2 gate, exactly as FLA's
// chunk_gated_delta_rule_fwd_kernel_h_blockdim64, so FLA's w/u/g can be fed in
// and h/v_new compared directly.
//
// Only full tiles are supported (S % 64 == 0, V % BV == 0); the router keeps
// other shapes on the fused kernel.
//
// What decides the speed of this kernel is the number of global memory
// instructions per chunk, not the GEMMs: every global access below is a 16-byte
// vector access.  u is staged in LDS, v_new is formed in place there and
// written back as rows, and h goes out of its LDS copy as rows -- instead of
// the 2-byte scattered accesses the mmac D layout would otherwise force.
// LDS is shared between waves only through two block barriers per chunk (sU is
// double buffered so no barrier is needed at the end of a chunk); global
// memory is never read back, so the barriers wait for LDS only.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "flash_train_gdn_state_h.h"
#include "gdn_chunk_mmac.h"

namespace {

using gdn_mmac::bf2f;
using gdn_mmac::f2bf;
using gdn_mmac::u16x4;
using gdn_mmac::v4f;
using gdn_mmac::v4s;
typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));

constexpr int kC = 64;  // chunk length (FLA BT)

// Block barrier waiting for LDS only (see header comment).  gfx9 s_waitcnt
// encoding: vmcnt=63, expcnt=7, lgkmcnt=0.  Measured equal to a plain
// __syncthreads() here once global I/O is vectorised; kept because it states
// what the barrier is for.
__device__ __forceinline__ void lds_barrier() {
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_waitcnt(0xC07F);
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
}

template <int K, int BV, int NW>
__global__ __launch_bounds__(64 * NW) void gdn_state_h_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ u, const float* __restrict__ g,
    const float* __restrict__ h0, uint16_t* __restrict__ h,
    uint16_t* __restrict__ v_new, float* __restrict__ ht,
    int S, int H, int HV, int V, int h_vk) {
    constexpr int THREADS = 64 * NW;
    constexpr int KT16 = K / 16, VT16 = BV / 16;
    constexpr int WG = NW / 4;            // waves sharing one GEMM1 t-tile
    constexpr int VPW = VT16 / WG;        // GEMM1 v-tiles per wave
    constexpr int TPW = VT16 * KT16 / NW; // h^T tiles per wave
    constexpr int KTSTR = kC + 4;         // sKT[kk][t]
    constexpr int HSTR = K + 8;           // sHb[v][kk]
    constexpr int VTSTR = kC + 4;         // sVT[v][t]
    constexpr int USTR = BV + 8;          // sU[t][v]  (rows stay 16B aligned)
    constexpr int NK8 = kC * K / 8 / THREADS;   // u16x8 of k per thread
    constexpr int NU8 = kC * BV / 8 / THREADS;  // u16x8 of u (and v_new) per thread
    constexpr int NH8 = K * BV / 8 / THREADS;   // u16x8 of h per thread
    static_assert(NW % 4 == 0 && VT16 % WG == 0 && (VT16 * KT16) % NW == 0, "wave split");
    static_assert(NK8 == 1 || NK8 == 2 || NK8 == 4, "k packing");
    static_assert(NK8 >= 1 && NU8 >= 1 && NH8 >= 1, "vector staging must cover the tile");
    static_assert((kC * K / 8) % THREADS == 0 && (kC * BV / 8) % THREADS == 0 &&
                  (K * BV / 8) % THREADS == 0, "vector staging must divide evenly");

    __shared__ __align__(16) unsigned short sKT[K * KTSTR];
    __shared__ __align__(16) unsigned short sHb[BV * HSTR];
    __shared__ __align__(16) unsigned short sVT[BV * VTSTR];
    __shared__ __align__(16) unsigned short sU[2 * kC * USTR];  // double buffered: the v_new store of chunk n overlaps the staging of n+1
    __shared__ float sG[kC];

    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int lo = lane & 15, hq = lane >> 4, hi4 = hq * 4;
    const int v0 = blockIdx.x * BV;
    const int bhv = blockIdx.y;
    const int b = bhv / HV, hv = bhv % HV, hk = hv / (HV / H);
    const int NT = S / kC;
    const int pl = 4 * (lo & 3) + (lo >> 2);  // row permutation: D column hq+4r <-> index 4hq+r
    const int tw = (wave % 4) * 16 + lo;  // GEMM1 row (t) of this lane
    const int vg = wave / 4;              // GEMM1 v-tiles vg, vg + WG, ...

    // fp32 state: tile i = wave + i*NW -> (tv, tk); lane holds
    // h^T[tv*16+lo][tk*16+hq+4r].
    v4f hreg[TPW];
#pragma unroll
    for (int i = 0; i < TPW; ++i) {
        const int t = wave + i * NW, tv = t / KT16, tk = t % KT16;
        if (h0 == nullptr) {
            hreg[i] = v4f{0.f, 0.f, 0.f, 0.f};
        } else if (h_vk) {  // h0[bhv][v][kk]
            hreg[i] = *reinterpret_cast<const v4f*>(
                h0 + (static_cast<std::size_t>(bhv) * V + v0 + tv * 16 + lo) * K + tk * 16 + 4 * hq);
        } else {            // h0[bhv][kk][v]
#pragma unroll
            for (int r = 0; r < 4; ++r)
                hreg[i][r] = h0[(static_cast<std::size_t>(bhv) * K + tk * 16 + 4 * hq + r) * V + v0 + tv * 16 + lo];
        }
    }

    // ---- addressing: wave-uniform bases + per-lane 32-bit offsets ----
    const std::size_t rowHVK = static_cast<std::size_t>(HV) * K;
    const std::size_t rowHVV = static_cast<std::size_t>(HV) * V;
    const std::size_t rowHK = static_cast<std::size_t>(H) * K;
    const uint16_t* wB = w + static_cast<std::size_t>(b) * S * rowHVK + static_cast<std::size_t>(hv) * K;
    const uint16_t* uB = u + static_cast<std::size_t>(b) * S * rowHVV + static_cast<std::size_t>(hv) * V + v0;
    uint16_t* vnB = v_new + static_cast<std::size_t>(b) * S * rowHVV + static_cast<std::size_t>(hv) * V + v0;
    const uint16_t* kB = k + static_cast<std::size_t>(b) * S * rowHK + static_cast<std::size_t>(hk) * K;
    const float* gB = g + static_cast<std::size_t>(b) * S * HV + hv;
    uint16_t* hB = h + (static_cast<std::size_t>(b) * NT * HV + hv) * K * V + v0;
    const unsigned offW = static_cast<unsigned>(tw * rowHVK + 8 * hq);
    // k: each thread owns NK8 consecutive t rows x 8 kk, so the transposed LDS
    // write packs NK8 t values per kk into one 2/4/8-byte access.
    // LJ consecutive lanes take consecutive 16 B of one row (LJ*16 B per
    // cache-line segment); a lane owns NK8 consecutive t rows, so the transposed
    // LDS write stays one packed access per kk.
    constexpr int KLN = K / 8;                        // 16 B pieces per k row
    constexpr int LJ = 4 < KLN ? 4 : KLN;  // measured best of 4/8/16 (llm32k 2.35/2.41/2.45 ms)
    constexpr int WL = 64 / LJ;                       // row groups per wave
    constexpr int CGN = KLN / LJ;                     // column groups
    static_assert(NW == CGN * (kC / (WL * NK8)), "k load mapping");
    const int kslab = wave / CGN, kcg = wave % CGN;
    const int kt0 = kslab * (WL * NK8) + (lane / LJ) * NK8;
    const int kk8 = (kcg * LJ + lane % LJ) * 8;
    const unsigned offK = static_cast<unsigned>(kt0 * rowHK + kk8);
    // u / v_new: rows of BV, 8 v per access.
    constexpr int UPR = BV / 8;  // accesses per row
    const int ut = tid / UPR, uv8 = (tid % UPR) * 8;
    constexpr int UTSTEP = THREADS / UPR;
    // h: rows of BV for each kk, 8 v per access.
    const int hkk = tid / UPR;
    constexpr int HKSTEP = THREADS / UPR;

    // ---- prefetch registers for one chunk (all 16-byte or 8-byte accesses) ----
    u16x8 pa[KT16 / 2];  // GEMM1 A fragments: w[tw][32*j + 8*hq .. +7] = two mmac k-steps (reduction order is permuted consistently with sHb reads)
    u16x8 pk[NK8];
    u16x8 pu[NU8];
    float pg = 0.0f;
    auto prefetch = [&](int n) {
        const int t0 = n * kC;
        const uint16_t* wc = wB + t0 * rowHVK;
#pragma unroll
        for (int j = 0; j < KT16 / 2; ++j) pa[j] = *reinterpret_cast<const u16x8*>(wc + offW + 32 * j);
        const uint16_t* kc = kB + t0 * rowHK;
#pragma unroll
        for (int r = 0; r < NK8; ++r) pk[r] = *reinterpret_cast<const u16x8*>(kc + offK + static_cast<unsigned>(r * rowHK));
        const uint16_t* uc = uB + t0 * rowHVV;
#pragma unroll
        for (int r = 0; r < NU8; ++r)
            pu[r] = *reinterpret_cast<const u16x8*>(uc + static_cast<unsigned>((ut + r * UTSTEP) * rowHVV) + uv8);
        if (tid < kC) pg = gB[(t0 + tid) * HV];
    };
    prefetch(0);

    for (int n = 0; n < NT; ++n) {
        const int t0 = n * kC;
        unsigned short* sUc = sU + (n & 1) * (kC * USTR);

        // ---- stage this chunk: h^T bf16, k^T, u, g -> LDS ----
#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            const int t = wave + i * NW, tv = t / KT16, tk = t % KT16;
            *reinterpret_cast<u16x4*>(sHb + (tv * 16 + lo) * HSTR + tk * 16 + 4 * hq) =
                u16x4{f2bf(hreg[i][0]), f2bf(hreg[i][1]), f2bf(hreg[i][2]), f2bf(hreg[i][3])};
        }
#pragma unroll
        for (int r = 0; r < NU8; ++r)
            *reinterpret_cast<u16x8*>(sUc + (ut + r * UTSTEP) * USTR + uv8) = pu[r];
        if (tid < kC) sG[tid] = pg;
        u16x8 ca[KT16 / 2];
#pragma unroll
        for (int j = 0; j < KT16 / 2; ++j) ca[j] = pa[j];
        lds_barrier();
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            unsigned short* kp = sKT + (kk8 + q) * KTSTR + kt0;
            if constexpr (NK8 == 4) {
                *reinterpret_cast<u16x4*>(kp) = u16x4{pk[0][q], pk[1][q], pk[2][q], pk[3][q]};
            } else if constexpr (NK8 == 2) {
                *reinterpret_cast<unsigned*>(kp) = static_cast<unsigned>(pk[0][q]) | (static_cast<unsigned>(pk[1][q]) << 16);
            } else {
#pragma unroll
                for (int r = 0; r < NK8; ++r) kp[r] = pk[r][q];
            }
        }
        if (n + 1 < NT) prefetch(n + 1);
        const float g_last = sG[kC - 1];

        // ---- GEMM1: v_new[t][v] = u - sum_kk w[t][kk] h^T[v][kk] ----
        {
            const float dec = exp2f(g_last - sG[tw]);
#pragma unroll
            for (int j = 0; j < VPW; ++j) {
                const int tn2 = vg + j * WG;
                const unsigned short* bp = sHb + (tn2 * 16 + pl) * HSTR + 8 * hq;
                v4f acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int jj = 0; jj < KT16 / 2; ++jj) {
                    const u16x8 bb = *reinterpret_cast<const u16x8*>(bp + 32 * jj);
                    acc = __builtin_amdgcn_mmac_f32_16x16x16bf16(
                        __builtin_bit_cast(v4s, __builtin_shufflevector(ca[jj], ca[jj], 0, 1, 2, 3)),
                        __builtin_bit_cast(v4s, __builtin_shufflevector(bb, bb, 0, 1, 2, 3)), acc);
                    acc = __builtin_amdgcn_mmac_f32_16x16x16bf16(
                        __builtin_bit_cast(v4s, __builtin_shufflevector(ca[jj], ca[jj], 4, 5, 6, 7)),
                        __builtin_bit_cast(v4s, __builtin_shufflevector(bb, bb, 4, 5, 6, 7)), acc);
                }
                unsigned short* up = sUc + tw * USTR + tn2 * 16 + 4 * hq;
                const u16x4 u4 = *reinterpret_cast<const u16x4*>(up);
                u16x4 o4;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float x = bf2f(u4[r]) - acc[r];
                    o4[r] = f2bf(x);  // in place: this lane owns (tw, 4hq..4hq+3)
                    sVT[(tn2 * 16 + 4 * hq + r) * VTSTR + tw] = f2bf(x * dec);
                }
                *reinterpret_cast<u16x4*>(up) = o4;
            }
        }
        // ---- h (chunk-start state) rows out of sHb: ready since the barrier ----
        {
            if (h_vk) {
                // h[b][n][hv][v][kk]: rows of K along kk, straight from sHb rows.
                uint16_t* hc = h + ((static_cast<std::size_t>(b) * NT + n) * HV + hv) * V * K +
                               static_cast<std::size_t>(v0) * K;
#pragma unroll
                for (int r = 0; r < NH8; ++r) {
                    const int e = tid + r * THREADS, vr = e / (K / 8), c = (e % (K / 8)) * 8;
                    *reinterpret_cast<u16x8*>(hc + vr * K + c) =
                        *reinterpret_cast<const u16x8*>(sHb + vr * HSTR + c);
                }
            } else {
                uint16_t* hc = hB + static_cast<std::size_t>(n) * HV * K * V;
#pragma unroll
                for (int r = 0; r < NH8; ++r) {
                    const int kk = hkk + r * HKSTEP;
                    u16x8 x;
#pragma unroll
                    for (int q = 0; q < 8; ++q) x[q] = sHb[(uv8 + q) * HSTR + kk];
                    *reinterpret_cast<u16x8*>(hc + static_cast<unsigned>(kk * V) + uv8) = x;
                }
            }
        }

        lds_barrier();

        // ---- v_new rows out of sU ----
        {
            uint16_t* vc = vnB + t0 * rowHVV;
#pragma unroll
            for (int r = 0; r < NU8; ++r) {
                const int t = ut + r * UTSTEP;
                *reinterpret_cast<u16x8*>(vc + static_cast<unsigned>(t * rowHVV) + uv8) =
                    *reinterpret_cast<const u16x8*>(sUc + t * USTR + uv8);
            }
        }

        // ---- GEMM2: h^T[v][kk] = exp2(g_last) h^T + sum_t vT[v][t] kT[kk][t] ----
        {
            const float decay = exp2f(g_last);
#pragma unroll
            for (int i = 0; i < TPW; ++i) {
                const int t = wave + i * NW, tv = t / KT16, tk = t % KT16;
                const unsigned short* ap = sVT + (tv * 16 + lo) * VTSTR + hi4;
                const unsigned short* bp = sKT + (tk * 16 + pl) * KTSTR + hi4;
                v4f acc = hreg[i] * decay;
#pragma unroll
                for (int k0 = 0; k0 < kC; k0 += 16)
                    acc = __builtin_amdgcn_mmac_f32_16x16x16bf16(
                        *reinterpret_cast<const v4s*>(ap + k0),
                        *reinterpret_cast<const v4s*>(bp + k0), acc);
                hreg[i] = acc;
            }
        }
    }

    if (ht != nullptr) {
#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            const int t = wave + i * NW, tv = t / KT16, tk = t % KT16;
            if (h_vk) {  // ht[bhv][v][kk]
                *reinterpret_cast<v4f*>(
                    ht + (static_cast<std::size_t>(bhv) * V + v0 + tv * 16 + lo) * K + tk * 16 + 4 * hq) = hreg[i];
            } else {     // ht[bhv][kk][v]
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    ht[(static_cast<std::size_t>(bhv) * K + tk * 16 + 4 * hq + r) * V + v0 + tv * 16 + lo] = hreg[i][r];
            }
        }
    }
}

template <int K, int BV, int NW>
int launch(const FlashTrainGDNStateHArgs& a, hipStream_t st) {
    if (a.S % kC != 0 || a.V % BV != 0) return 2;
    dim3 grid(a.V / BV, a.B * a.HV);
    hipLaunchKernelGGL((gdn_state_h_kernel<K, BV, NW>), grid, dim3(64 * NW), 0, st,
                       a.k, a.w, a.u, a.g, a.h0, a.h, a.v_new, a.ht, a.S, a.H, a.HV, a.V, a.h_vk);
    return hipGetLastError() == hipSuccess ? 0 : 5;
}

}  // namespace

// bv selects the tiling: 64 = BV 64 with 4 waves, 164 = BV 64 with 8 waves,
// 0 = measured best (4 waves for both K; llm32k: K=128 2.71 ms, K=64 1.70 ms; 8 waves 3.11 / 2.01 ms).
extern "C" int flash_train_gdn_state_h_bf16(const FlashTrainGDNStateHArgs* a, void* stream) {
    if (a == nullptr || a->k == nullptr || a->w == nullptr || a->u == nullptr ||
        a->g == nullptr || a->h == nullptr || a->v_new == nullptr)
        return 1;
    if (a->B <= 0 || a->S <= 0 || a->H <= 0 || a->HV % a->H != 0) return 2;
    hipStream_t st = static_cast<hipStream_t>(stream);
    const int bv = a->bv != 0 ? a->bv : 64;
    if (a->K == 128 && bv == 64) return launch<128, 64, 4>(*a, st);
    if (a->K == 128 && bv == 164) return launch<128, 64, 8>(*a, st);
    if (a->K == 64 && bv == 64) return launch<64, 64, 4>(*a, st);
    if (a->K == 64 && bv == 164) return launch<64, 64, 8>(*a, st);
    return 2;
}
