// Chunk-parallel stages of the GDN-2 backward (chunk = 64, FLA layouts).
//   B1 dAv: dAqk = scale tril(do v_new^T) (fp32),  dv_local = tril(Aqk)^T do (bf16)   [FLA chunk_kda_bwd_dAv]
//
// One block per (chunk, b*hv), hv fastest in the 1-D grid, 4 waves.  Wave w owns the token rows
// 16w..16w+15 of both products.
//   dAqk rows: A = do rows, B = v_new rows, both 16 B loads straight into registers (lane hq takes
//     chunk hq+4c of its row, so 4 lanes read 64 contiguous bytes; A and B use the same contraction
//     order).  The B rows of the n-th mmac column are permuted (4(n&3)+(n>>2)) so the D layout hands a
//     lane 4 consecutive columns -> one 16 B fp32 store.  Tiles right of the diagonal are never computed.
//   dv_local rows: contraction over t, so do^T and Aqk^T are staged in LDS (2 B scatter), then both mmac
//     operands are 8 B fragment reads; the output goes through LDS rows to 16 B stores.
// All global loads are issued before the first wait.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../common/flash_train_gdn2_bwd.h"
#include "../common/gdn2_chunk_mmac.h"

namespace {

using gdn2_mmac::f2bf;
using gdn2_mmac::u16x4;
using gdn2_mmac::v4f;
using gdn2_mmac::v4s;
typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));

constexpr int kC = 64;
constexpr int kThreads = 256;

__device__ __forceinline__ v4f mmac(v4s a, v4s b, v4f c) {
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
}

__device__ __forceinline__ v4s frag4(const u16x8& x, int h) {
    return v4s{static_cast<short>(x[4 * h]), static_cast<short>(x[4 * h + 1]),
               static_cast<short>(x[4 * h + 2]), static_cast<short>(x[4 * h + 3])};
}

template <int V>
__global__ __launch_bounds__(kThreads) void gdn2_bwd_dav_kernel(
    const uint16_t* __restrict__ vn, const uint16_t* __restrict__ dout, const uint16_t* __restrict__ aqk,
    float* __restrict__ dA, uint16_t* __restrict__ dv, int S, int HV, float scale) {
    constexpr int DS = kC + 4;   // sDT row stride (bf16), [v][t]
    constexpr int AT = kC + 4;   // sAT row stride (bf16), [s][t]
    constexpr int OS = V + 8;    // output staging stride (16B rows)
    constexpr int DSZ = V * DS, OSZ = kC * OS;
    constexpr int BUF = DSZ > OSZ ? DSZ : OSZ;
    constexpr int NKC = V / 32;  // 16B do / v loads per lane and tile
    constexpr int VT16 = V / 16;
    constexpr int DPER = V / 8;
    constexpr int NA = kC * 8 / kThreads;  // 16B Aqk items per thread
    // do^T, later the output staging rows
    __shared__ __align__(16) unsigned short sBuf[BUF];
    __shared__ __align__(16) unsigned short sAT[kC * AT];
    unsigned short* const sDT = sBuf;
    unsigned short* const sO = sBuf;

    const int tid = threadIdx.x, lane = tid & 63, tm = tid >> 6;
    const int lo = lane & 15, hq = lane >> 4;
    const int NT = S / kC;
    int id = blockIdx.x;
    const int hv = id % HV;
    id /= HV;
    const int n = id % NT, b = id / NT;
    const std::size_t row0 = static_cast<std::size_t>(b) * S + n * kC;
    const std::size_t dRow = static_cast<std::size_t>(HV) * V;
    const std::size_t aRow = static_cast<std::size_t>(HV) * kC;
    const int tr = tm * 16 + lo;                  // this lane's token row (A operand row, output row)
    const int pl = 4 * (lo & 3) + (lo >> 2);      // permuted column of the B operand

    // ---- every global load is issued before the first wait ----
    u16x8 xd[NKC], xa[NA], xv[4][NKC];
#pragma unroll
    for (int c = 0; c < NKC; ++c)
        xd[c] = *reinterpret_cast<const u16x8*>(dout + (row0 + tr) * dRow + static_cast<std::size_t>(hv) * V + (hq + 4 * c) * 8);
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int e = tid + i * kThreads;
        xa[i] = *reinterpret_cast<const u16x8*>(aqk + (row0 + (e >> 3)) * aRow + static_cast<std::size_t>(hv) * kC + (e & 7) * 8);
    }
#pragma unroll
    for (int tn = 0; tn < 4; ++tn)
        if (tn <= tm)
#pragma unroll
            for (int c = 0; c < NKC; ++c)
                xv[tn][c] = *reinterpret_cast<const u16x8*>(vn + (row0 + tn * 16 + pl) * dRow + static_cast<std::size_t>(hv) * V + (hq + 4 * c) * 8);

    // ---- stage do^T[v][t] and the masked Aqk^T[s][t] (2 B scatter) ----
#pragma unroll
    for (int c = 0; c < NKC; ++c)
#pragma unroll
        for (int r = 0; r < 8; ++r) sDT[((hq + 4 * c) * 8 + r) * DS + tr] = xd[c][r];
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int e = tid + i * kThreads, t = e >> 3, s0 = (e & 7) * 8;
#pragma unroll
        for (int r = 0; r < 8; ++r) sAT[(s0 + r) * AT + t] = (t >= s0 + r) ? xa[i][r] : static_cast<unsigned short>(0);
    }

    // ---- dAqk rows: dA[tr][16 tn + 4 hq + qd] = scale * do_tr . v_new_col, lower triangle only ----
    v4f dacc[4];
#pragma unroll
    for (int tn = 0; tn < 4; ++tn) {
        dacc[tn] = v4f{0.f, 0.f, 0.f, 0.f};
        if (tn <= tm) {
#pragma unroll
            for (int c = 0; c < NKC; ++c) {
                dacc[tn] = mmac(frag4(xd[c], 0), frag4(xv[tn][c], 0), dacc[tn]);
                dacc[tn] = mmac(frag4(xd[c], 1), frag4(xv[tn][c], 1), dacc[tn]);
            }
        }
    }
    float* const dAp = dA + (row0 + tr) * aRow + static_cast<std::size_t>(hv) * kC;
#pragma unroll
    for (int tn = 0; tn < 4; ++tn) {
        v4f o;
#pragma unroll
        for (int qd = 0; qd < 4; ++qd) o[qd] = (tn * 16 + 4 * hq + qd <= tr) ? dacc[tn][qd] * scale : 0.0f;
        *reinterpret_cast<v4f*>(dAp + tn * 16 + 4 * hq) = o;
    }
    __syncthreads();

    // ---- dv_local[s][v] = sum_{t >= s} Aqk^T[s][t] do[t][v]; B column n of tile tv is v = 16 tv + 4(n&3) + (n>>2) ----
    v4s af[4];
#pragma unroll
    for (int ts = 0; ts < 4; ++ts)
        if (ts >= tm) af[ts] = *reinterpret_cast<const v4s*>(sAT + tr * AT + ts * 16 + hq * 4);
    v4f acc[VT16];
#pragma unroll
    for (int tv = 0; tv < VT16; ++tv) {
        acc[tv] = v4f{0.f, 0.f, 0.f, 0.f};
        const unsigned short* bp = sDT + (tv * 16 + pl) * DS + hq * 4;
#pragma unroll
        for (int ts = 0; ts < 4; ++ts)
            if (ts >= tm) acc[tv] = mmac(af[ts], *reinterpret_cast<const v4s*>(bp + ts * 16), acc[tv]);
    }
    __syncthreads();  // everyone is done with do^T

    // ---- output: stage rows in LDS (wave-local rows), then 16B row stores ----
#pragma unroll
    for (int tv = 0; tv < VT16; ++tv)
        *reinterpret_cast<u16x4*>(sO + tr * OS + tv * 16 + hq * 4) =
            u16x4{f2bf(acc[tv][0]), f2bf(acc[tv][1]), f2bf(acc[tv][2]), f2bf(acc[tv][3])};
    __builtin_amdgcn_wave_barrier();
    uint16_t* const oB = dv + (row0 * HV + hv) * V;
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

extern "C" int flash_train_gdn2_bwd_dav_bf16(const FlashTrainGDN2BwdDAvArgs* a, void* stream) {
    if (!a || !a->v_new || !a->do_ || !a->aqk || !a->dAqk || !a->dv_local) return 1;
    if (a->S % kC || a->B <= 0 || a->S <= 0 || a->HV <= 0 || (a->V != 64 && a->V != 128)) return 2;
    hipStream_t st = static_cast<hipStream_t>(stream);
    const dim3 grid(static_cast<unsigned>((a->S / kC) * a->B * a->HV));
#define L(VV)                                                                                         \
    if (a->V == VV) {                                                                                 \
        hipLaunchKernelGGL((gdn2_bwd_dav_kernel<VV>), grid, dim3(kThreads), 0, st, a->v_new, a->do_, \
                           a->aqk, a->dAqk, a->dv_local, a->S, a->HV, a->scale);                      \
        return hipGetLastError() == hipSuccess ? 0 : 5;                                               \
    }
    L(128) L(64)
#undef L
    return 2;
}
