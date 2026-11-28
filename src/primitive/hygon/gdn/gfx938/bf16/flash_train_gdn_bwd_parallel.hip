// Chunk-parallel stages of the large-shape GDN backward (chunk = 64, FLA layouts).
//   B1 dv_local: dv_l[s] = sum_{t>=s} scale 2^(gc_t-gc_s) (q_t.k_s) do_t   (within a chunk)
//
// One block per (chunk, b*hv), hv fastest in the 1-D grid, 4 waves.  Wave w owns the rows
// s = 16w..16w+15.  P^T[s][t] = k_s . q_t is computed with bf16 mmac (k from global in
// registers, q tile shared through LDS); the D layout hands a lane the columns
// t = 16ts + hq + 4i, so gate + causal mask are applied in registers and the result is
// used directly as the A operand of P^T.do.  The contraction therefore runs over that
// permuted t order and do^T is staged in LDS with the same permutation.
// All global traffic is 16-byte vectors; every load is issued before the first wait.
#include "../common/gfx938_device_symbols.h"
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../common/flash_train_gdn_bwd.h"
#include "../common/gdn_chunk_mmac.h"

namespace {

using gdn_mmac::f2bf;
using gdn_mmac::u16x4;
using gdn_mmac::v4f;
using gdn_mmac::v4s;
typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));

constexpr int kC = 64;
constexpr int kThreads = 256;

// Column XOR for sDT, as a function of m = row >> 3.  The transposed stage write (b16) has a lane write every 8th
// row (rows c + r, c a multiple of 8), so all lanes of a group land in bank 0/16 and no row padding can fix it
// (the row stride 4 * VS bytes is always even in banks).  Instead the column (the t index) is XORed with a function
// of the 8-row group.  Masks found by exhaustive search against the gfx936 bank model; the XOR moves whole 4-element
// (8 B) groups, so the b64 B-operand reads stay contiguous.
__device__ __forceinline__ int dt_swz(int m) {
    return (((m >> 1) & 1) | (((m >> 2) & 1) << 1) | (((m >> 3) & 1) << 2) | ((((m >> 1) ^ (m >> 2)) & 1) << 3)) << 2;
}

__device__ __forceinline__ v4f mmac(v4s a, v4s b, v4f c) {
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
}

// VT is the row width of do / dv_local; a block covers V columns and the grid splits VT / V ways, so a
// V = 256 problem runs as two V = 128 blocks per (chunk, hv) (a single V = 256 block needs 35 KB LDS and
// 95 VGPRs: 1 block/CU instead of 3).  VT == V is the unsplit kernel.
template <int K, int V, int VT = V>
__global__ __launch_bounds__(kThreads) void gdn_bwd_dv_local_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ dout, const float* __restrict__ gc,
    uint16_t* __restrict__ dv, int S, int H, int HV, float scale) {
    constexpr int KS = K + 4;   // sQ row stride (bf16)
    constexpr int VS = kC + 4;  // sDT row stride (bf16), [v][permuted t]
    constexpr int OS = V + 8;   // output staging stride (16B rows)
    constexpr int QSZ = kC * KS, DSZ = V * VS, OSZ = kC * OS;
    constexpr int BUF = (QSZ > DSZ ? (QSZ > OSZ ? QSZ : OSZ) : (DSZ > OSZ ? DSZ : OSZ));
    // One region, three lifetimes: q tile -> do^T -> output staging.
    __shared__ __align__(16) unsigned short sBuf[BUF];
    __shared__ float sGc[kC];
    unsigned short* const sQ = sBuf;
    unsigned short* const sDT = sBuf;
    unsigned short* const sO = sBuf;

    constexpr int KT16 = K / 16, VT16 = V / 16;
    constexpr int QPER = K / 8, NQ = kC * QPER / kThreads;  // q tile: 16B items
    constexpr int DPER = V / 8, ND = kC * DPER / kThreads;  // do tile: 16B items
    constexpr int NKC = K / 32;                              // 16B k loads per lane
    static_assert(kC * QPER % kThreads == 0 && kC * DPER % kThreads == 0, "tile mapping");
    static_assert(KT16 == 2 * NKC, "k chunks cover K");

    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int lo = lane & 15, hq = lane >> 4;
    const int NT = S / kC;
    int id = blockIdx.x;
    const int hv = id % HV;
    id /= HV;
    constexpr int NSPLIT = VT / V;
    static_assert(NSPLIT * V == VT, "VT must be a multiple of V");
    const int vh = id % NSPLIT;
    id /= NSPLIT;
    const int n = id % NT, b = id / NT;
    const int hk = hv / (HV / H);
    const std::size_t row0 = static_cast<std::size_t>(b) * S + n * kC;
    const std::size_t qkRow = static_cast<std::size_t>(H) * K;
    const std::size_t dRow = static_cast<std::size_t>(HV) * VT;
    const int tm = wave, tr = tm * 16 + lo;  // this lane's k row (= s)

    // ---- every global load is issued before the first wait ----
    u16x8 xq[NQ], xd[ND], xk[NKC];
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        const int e = tid + i * kThreads;
        xq[i] = *reinterpret_cast<const u16x8*>(q + (row0 + e / QPER) * qkRow +
                                                static_cast<std::size_t>(hk) * K + (e % QPER) * 8);
    }
#pragma unroll
    for (int i = 0; i < ND; ++i) {
        const int e = tid + i * kThreads;
        xd[i] = *reinterpret_cast<const u16x8*>(dout + (row0 + e / DPER) * dRow +
                                                static_cast<std::size_t>(hv) * VT + vh * V + (e % DPER) * 8);
    }
    // k rows as 16B loads: lane hq takes 16B chunk hq + 4c of the row, so the 4 lanes of a
    // row read 64 contiguous bytes.  Each chunk feeds two mmac steps (elements 0..3, 4..7);
    // the q fragments below use the identical kk set, so the contraction order is consistent.
#pragma unroll
    for (int c = 0; c < NKC; ++c)
        xk[c] = *reinterpret_cast<const u16x8*>(k + (row0 + tr) * qkRow +
                                                static_cast<std::size_t>(hk) * K + (hq + 4 * c) * 8);
    const float gcv = tid < kC ? gc[(row0 + tid) * HV + hv] : 0.f;

    // ---- stage q tile (LDS rows K+4: 8B stores, conflict-free fragment reads) ----
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        const int e = tid + i * kThreads, t = e / QPER, c = (e % QPER) * 8;
        *reinterpret_cast<u16x4*>(sQ + t * KS + c) = u16x4{xq[i][0], xq[i][1], xq[i][2], xq[i][3]};
        *reinterpret_cast<u16x4*>(sQ + t * KS + c + 4) = u16x4{xq[i][4], xq[i][5], xq[i][6], xq[i][7]};
    }
    if (tid < kC) sGc[tid] = gcv;
    __syncthreads();

    // ---- P^T[s][t] = k_s . q_t for the t tiles ts >= tm ----
    v4f pacc[4];
#pragma unroll
    for (int ts = 0; ts < 4; ++ts) {
        pacc[ts] = v4f{0.f, 0.f, 0.f, 0.f};
        if (ts >= tm) {
            const unsigned short* qp = sQ + (ts * 16 + lo) * KS + hq * 8;
#pragma unroll
            for (int c = 0; c < NKC; ++c) {
                const v4s a0 = v4s{static_cast<short>(xk[c][0]), static_cast<short>(xk[c][1]),
                                   static_cast<short>(xk[c][2]), static_cast<short>(xk[c][3])};
                const v4s a1 = v4s{static_cast<short>(xk[c][4]), static_cast<short>(xk[c][5]),
                                   static_cast<short>(xk[c][6]), static_cast<short>(xk[c][7])};
                pacc[ts] = mmac(a0, *reinterpret_cast<const v4s*>(qp + 32 * c), pacc[ts]);
                pacc[ts] = mmac(a1, *reinterpret_cast<const v4s*>(qp + 32 * c + 4), pacc[ts]);
            }
        }
    }
    __syncthreads();  // sQ is overwritten by do^T

    // ---- stage do^T[v][perm(t)]; position within a 16-group: (t&3)<<2 | (t>>2)&3 ----
#pragma unroll
    for (int i = 0; i < ND; ++i) {
        const int e = tid + i * kThreads, t = e / DPER, c = (e % DPER) * 8;
        const int pt = (t & ~15) | ((t & 3) << 2) | ((t >> 2) & 3);
        const int ptw = pt ^ dt_swz(c >> 3);
#pragma unroll
        for (int r = 0; r < 8; ++r) sDT[(c + r) * VS + ptw] = xd[i][r];
    }
    __syncthreads();

    // ---- gate + causal mask in registers: P^T becomes the A operand ----
    const float gs = sGc[tr];
    v4s af[4];
#pragma unroll
    for (int ts = 0; ts < 4; ++ts) {
        af[ts] = v4s{0, 0, 0, 0};
        if (ts >= tm) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int t = ts * 16 + hq + 4 * i;
                const float x = (t >= tr) ? pacc[ts][i] * exp2f(sGc[t] - gs) * scale : 0.0f;
                af[ts][i] = static_cast<short>(f2bf(x));
            }
        }
    }

    // ---- dv[s][v] = sum_t P^T[s][t] do[t][v]; B-operand column of lane lo is the actual v
    // column 4*(lo&3)+(lo>>2) of its tile, so a lane ends up with 4 consecutive v. ----
    const int fl = 4 * (lo & 3) + (lo >> 2);
    v4f acc[VT16];
#pragma unroll
    for (int tv = 0; tv < VT16; ++tv) {
        acc[tv] = v4f{0.f, 0.f, 0.f, 0.f};
        const unsigned short* bp = sDT + (tv * 16 + fl) * VS;
        const int bz = dt_swz(2 * tv + ((lo >> 1) & 1));
#pragma unroll
        for (int ts = 0; ts < 4; ++ts)
            if (ts >= tm) acc[tv] = mmac(af[ts], *reinterpret_cast<const v4s*>(bp + ((hq * 4 + ts * 16) ^ bz)), acc[tv]);
    }
    __syncthreads();  // everyone is done with do^T

    // ---- output: stage rows in LDS (wave-local rows), then 16B row stores ----
#pragma unroll
    for (int tv = 0; tv < VT16; ++tv)
        *reinterpret_cast<u16x4*>(sO + tr * OS + tv * 16 + hq * 4) =
            u16x4{f2bf(acc[tv][0]), f2bf(acc[tv][1]), f2bf(acc[tv][2]), f2bf(acc[tv][3])};
    __builtin_amdgcn_wave_barrier();
    uint16_t* const oB = dv + (row0 * HV + hv) * VT + vh * V;
#pragma unroll
    for (int e = lane; e < 16 * DPER; e += 64) {
        const int rr = tm * 16 + e / DPER, c = (e % DPER) * 8;
        const u16x4 x0 = *reinterpret_cast<const u16x4*>(sO + rr * OS + c);
        const u16x4 x1 = *reinterpret_cast<const u16x4*>(sO + rr * OS + c + 4);
        *reinterpret_cast<u16x8*>(oB + static_cast<std::size_t>(rr) * dRow + c) =
            u16x8{x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]};
    }
}

}  // namespace

extern "C" int flash_train_gdn_bwd_dv_local_bf16(const FlashTrainGDNBwdDvLocalArgs* a, void* stream) {
    if (!a || !a->q || !a->k || !a->do_ || !a->gc || !a->dv_local) return 1;
    if (a->S % kC || a->HV % a->H || a->B <= 0 || a->S <= 0) return 2;
    hipStream_t st = static_cast<hipStream_t>(stream);
    const dim3 grid(static_cast<unsigned>((a->S / kC) * a->B * a->HV));
#define L(KK, VV)                                                                                \
    if (a->K == KK && a->V == VV) {                                                              \
        hipLaunchKernelGGL((gdn_bwd_dv_local_kernel<KK, VV>), grid, dim3(kThreads), 0, st,       \
                           a->q, a->k, a->do_, a->gc, a->dv_local, a->S, a->H, a->HV, a->scale); \
        return hipGetLastError() == hipSuccess ? 0 : 5;                                          \
    }
    L(128, 128) L(64, 128) L(128, 64) L(64, 64)
#undef L
    if (a->V == 256) {  // two V = 128 blocks per (chunk, hv)
        const dim3 grid2(grid.x * 2);
        if (a->K == 128) {
            hipLaunchKernelGGL((gdn_bwd_dv_local_kernel<128, 128, 256>), grid2, dim3(kThreads), 0, st, a->q, a->k,
                               a->do_, a->gc, a->dv_local, a->S, a->H, a->HV, a->scale);
        } else if (a->K == 64) {
            hipLaunchKernelGGL((gdn_bwd_dv_local_kernel<64, 128, 256>), grid2, dim3(kThreads), 0, st, a->q, a->k,
                               a->do_, a->gc, a->dv_local, a->S, a->H, a->HV, a->scale);
        } else {
            return 2;
        }
        return hipGetLastError() == hipSuccess ? 0 : 5;
    }
    return 2;
}
