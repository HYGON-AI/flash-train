// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <type_traits>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#    include <du_mma.h>
#endif

#include "flash_train/env.h"
#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/proj_rms_compute_h_fwd/fp16_proj_rms_compute_h_fwd.hpp"

static inline bool proj_rms_mma_supported() {
    int dev = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx928") != nullptr || strstr(prop.gcnArchName, "gfx936") != nullptr ||
           strstr(prop.gcnArchName, "gfx938") != nullptr;
}

// Adapted from the fused_dsa/fused_mHC kernel implementations.

#define PRMS_THREADS 256
#define PRMS_MTILE 64  // M rows per block (4 warps x 16 rows)
#define PRMS_NPAD 32   // N padded to a 16-multiple, capped at 32

template<typename T>
struct PrmsCfg {
    static constexpr int kfrag  = (sizeof(T) == 2) ? 16 : 8;
    static constexpr int kchunk = (sizeof(T) == 2) ? 128 : 64;  // fwd K chunk
    static constexpr int kld    = kchunk + 8;
    static constexpr int veclen = (sizeof(T) == 2) ? 8 : 4;  // vector copy width
};

template<typename T>
__device__ __forceinline__ float prms_to_float(T v);
template<>
__device__ __forceinline__ float prms_to_float(float v) {
    return v;
}
template<>
__device__ __forceinline__ float prms_to_float(__half v) {
    return __half2float(v);
}
template<>
__device__ __forceinline__ float prms_to_float(__hip_bfloat16 v) {
    return __bfloat162float(v);
}

template<typename T>
__device__ __forceinline__ T prms_from_float(float v);
template<>
__device__ __forceinline__ float prms_from_float(float v) {
    return v;
}
template<>
__device__ __forceinline__ __half prms_from_float<__half>(float v) {
    return __float2half(v);
}
template<>
__device__ __forceinline__ __hip_bfloat16 prms_from_float<__hip_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template<typename T>
__device__ __forceinline__ T prms_zero();
template<>
__device__ __forceinline__ float prms_zero<float>() {
    return 0.0f;
}
template<>
__device__ __forceinline__ __half prms_zero<__half>() {
    return __float2half(0.0f);
}
template<>
__device__ __forceinline__ __hip_bfloat16 prms_zero<__hip_bfloat16>() {
    return __float2bfloat16(0.0f);
}

template<typename T>
__device__ __forceinline__ static void prms_copy_vec(T* dst, const T* src) {
    if constexpr (sizeof(T) == 4) {
        *(float4*)dst = *(const float4*)src;
    } else {
        ((float2*)dst)[0] = ((const float2*)src)[0];
        ((float2*)dst)[1] = ((const float2*)src)[1];
    }
}

template<typename Dst, typename Src>
__device__ __forceinline__ static Dst prms_convert(Src v) {
    if constexpr (std::is_same<Dst, Src>::value) {
        return v;
    } else {
        return prms_from_float<Dst>(prms_to_float(v));
    }
}

template<typename Dst, typename Src, int VEC>
__device__ __forceinline__ static void prms_copy_convert_vec(Dst* dst, const Src* src) {
    if constexpr (std::is_same<Dst, Src>::value) {
        prms_copy_vec<Dst>(dst, src);
    } else {
#pragma unroll
        for (int i = 0; i < VEC; i++) dst[i] = prms_convert<Dst>(src[i]);
    }
}

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;

template<typename T, typename W, typename KernelStorage, int CFREE>
static __global__ void __launch_bounds__(PRMS_THREADS)
    proj_rms_ch_fwd_mma_v2_kernel(const T* __restrict__ x, const W* __restrict__ w, T* __restrict__ proj,
                                  T* __restrict__ r, const float* __restrict__ alpha_pre,
                                  const float* __restrict__ alpha_post, const float* __restrict__ alpha_res,
                                  const float* __restrict__ bias, T* __restrict__ h_pre, T* __restrict__ h_post,
                                  T* __restrict__ h_res, int M, int N, int K, float eps, int n_ch,
                                  float compute_h_eps) {
    constexpr int KFRAG  = PrmsCfg<KernelStorage>::kfrag;
    constexpr int KCHUNK = PrmsCfg<KernelStorage>::kchunk;
    constexpr int VECLEN = PrmsCfg<KernelStorage>::veclen;
    // Conflict-free K stride for the 8B MMA fragment loads:
    //   fp32 : bank(r) = (KLD*r + col) mod 32  -> KLD%32 == 10 gives 16
    //          distinct banks for r = 0..15.
    //   fp16/bf16: bank(r) = (KLD*r + col)/2 mod 32 -> KLD%32 == 2 gives
    //          (r + col/2) mod 32 -> distinct for r = 0..15.
    // CFREE == 1: conflict-free K stride (KLD%32 == 10 for fp32 / == 2 for
    // fp16/bf16), rows are 8B-aligned -> 8B LDS stores.
    // CFREE == 0: classic KLD = KCHUNK + 8 (16B-aligned rows, 16B stores,
    // 2-4 way MMA-load bank conflicts).
    constexpr int KLD = (CFREE != 0) ? ((sizeof(KernelStorage) == 4) ? (KCHUNK + 10) : (KCHUNK + 2)) : (KCHUNK + 8);
    constexpr int NT  = PRMS_NPAD / 16;  // 2 (max n-tiles)

    const int m0   = blockIdx.x * PRMS_MTILE;
    const int lane = threadIdx.x & 63;
    const int warp = threadIdx.x >> 6;

    extern __shared__ char smem_raw[];
    KernelStorage* x_lds = (KernelStorage*)smem_raw;          // [64][KLD]
    KernelStorage* w_lds = x_lds + (size_t)PRMS_MTILE * KLD;  // [NT*16][KLD]

    DUFragment<accumulator, 16, 16, KFRAG, float> acc[NT];
#    pragma unroll
    for (int nt = 0; nt < NT; nt++) du_fill_fragment(acc[nt], 0.0f);
    float row_sq = 0.0f;

    // Zero both LDS tiles once (stale bits must never enter an MMA).
    for (int i = threadIdx.x; i < (PRMS_MTILE + NT * 16) * KLD; i += PRMS_THREADS)
        ((KernelStorage*)smem_raw)[i] = prms_zero<KernelStorage>();
    __syncthreads();

    constexpr int NXV = PRMS_MTILE * (KCHUNK / VECLEN) / PRMS_THREADS;
    constexpr int NWV = (NT * 16 * (KCHUNK / VECLEN) + PRMS_THREADS - 1) / PRMS_THREADS;
    KernelStorage rx[NXV][VECLEN];
    KernelStorage rw[NWV][VECLEN];
    const int nchunk = (K + KCHUNK - 1) / KCHUNK;

    auto stage_ld = [&](int k0, int vper) {
#    pragma unroll
        for (int i = 0; i < NXV; i++) {
            int u  = threadIdx.x + i * PRMS_THREADS;
            int rr = u / vper;
            int vv = u - rr * vper;
            if (rr < PRMS_MTILE) {
                int g_row = m0 + rr;
                if (g_row < M)
                    prms_copy_convert_vec<KernelStorage, T, VECLEN>(rx[i], x + (size_t)g_row * K + k0 + vv * VECLEN);
            }
        }
#    pragma unroll
        for (int i = 0; i < NWV; i++) {
            int u = threadIdx.x + i * PRMS_THREADS;
            if (u < NT * 16 * vper) {
                int rr  = u / vper;
                int vv  = u - rr * vper;
                int g_n = rr;  // 0..NT*16-1
                if (g_n < N)
                    prms_copy_convert_vec<KernelStorage, W, VECLEN>(rw[i], w + (size_t)g_n * K + k0 + vv * VECLEN);
            }
        }
    };
    // 8B stores: KLD no longer keeps rows 16B-aligned, so split the 16B
    // register vector into two 8B LDS stores (fully conflict-free).
    auto store_lds = [&](KernelStorage* dst, const KernelStorage* v) {
        if (CFREE != 0) {
            if (sizeof(KernelStorage) == 4) {
                float2 a            = *(const float2*)v;
                float2 b            = *((const float2*)v + 1);
                *(float2*)dst       = a;
                *((float2*)dst + 1) = b;
            } else {
                *(float2*)dst       = *(const float2*)v;
                *((float2*)dst + 1) = *((const float2*)v + 1);
            }
        } else {
            prms_copy_vec<KernelStorage>(dst, v);
        }
    };
    auto stage_st = [&](int k0, int vper) {
#    pragma unroll
        for (int i = 0; i < NXV; i++) {
            int u  = threadIdx.x + i * PRMS_THREADS;
            int rr = u / vper;
            int vv = u - rr * vper;
            if (rr < PRMS_MTILE) {
                int g_row = m0 + rr;
                if (g_row < M) store_lds(x_lds + (size_t)rr * KLD + vv * VECLEN, rx[i]);
            }
        }
#    pragma unroll
        for (int i = 0; i < NWV; i++) {
            int u = threadIdx.x + i * PRMS_THREADS;
            if (u < NT * 16 * vper) {
                int rr = u / vper;
                int vv = u - rr * vper;
                if (rr < N) store_lds(w_lds + (size_t)rr * KLD + vv * VECLEN, rw[i]);
            }
        }
    };

    stage_ld(0, (K < KCHUNK ? K : KCHUNK) / VECLEN);
    stage_st(0, (K < KCHUNK ? K : KCHUNK) / VECLEN);
    __syncthreads();

// Two barriers per chunk: the next chunk's global loads are issued into
// registers before the current MMAs (latency hidden by the MMAs), then
// the trailing barrier waits until EVERY warp has finished the current
// chunk's MMAC LDS reads before the single-buffered LDS tiles are
// overwritten with the next chunk.  (The previous one-barrier version
// staged before the trailing barrier, so the ds_writes could overlap
// other warps' in-flight MMAC LDS reads -- the source of the
// intermittent fp32 proj_rms_compute_h corruption.)
#    pragma unroll 1
    for (int c = 0; c < nchunk; c++) {
        const int k0 = c * KCHUNK;
        int ck       = K - c * KCHUNK;
        if (ck > KCHUNK) ck = KCHUNK;

        if (c + 1 < nchunk) {
            int ck1 = K - (c + 1) * KCHUNK;
            if (ck1 > KCHUNK) ck1 = KCHUNK;
            stage_ld(k0 + KCHUNK, ck1 / VECLEN);
        }
        __syncthreads();
        {
#    pragma unroll
            for (int kk = 0; kk < KCHUNK / KFRAG; kk++) {
                if (kk < ck / KFRAG) {
                    DUFragment<matrix_a, 16, 16, KFRAG, KernelStorage, row_major> a;
                    du_load_matrix_sync(a, x_lds + (size_t)(warp * 16) * KLD + kk * KFRAG, KLD);
                    // RMS is not an MMAC operation: preserve the original
                    // activation precision even when mixed x/w tiles are
                    // promoted to FP32 for the matrix multiply.
                    constexpr int FE = KFRAG / 4;
                    const int xrow   = m0 + warp * 16 + (lane & 15);
                    if (xrow < M) {
#    pragma unroll
                        for (int e = 0; e < FE; ++e) {
                            const float xv =
                                prms_to_float(x[(size_t)xrow * K + k0 + kk * KFRAG + (lane >> 4) * FE + e]);
                            row_sq += xv * xv;
                        }
                    }
#    pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        DUFragment<matrix_b, 16, 16, KFRAG, KernelStorage, col_major> b;
                        du_load_matrix_sync(b, w_lds + (size_t)(nt * 16) * KLD + kk * KFRAG, KLD);
                        du_mma_sync(acc[nt], a, b, acc[nt]);
                    }
                }
            }
        }
        __syncthreads();
        if (c + 1 < nchunk) {
            int ck1 = K - (c + 1) * KCHUNK;
            if (ck1 > KCHUNK) ck1 = KCHUNK;
            stage_st(k0 + KCHUNK, ck1 / VECLEN);
        }
    }

    row_sq += __shfl_xor(row_sq, 16);
    row_sq += __shfl_xor(row_sq, 32);

    const int row     = m0 + warp * 16 + (lane & 15);
    const bool row_ok = (row < M);
    const float rv    = sqrtf(row_sq) / sqrtf((float)K);
    const float inv_r = 1.0f / (rv + eps);
    if (row_ok) {
        if (lane < 16) r[row] = prms_from_float<T>(rv);
        const float ap  = alpha_pre[0];
        const float apo = alpha_post[0];
        const float ar  = alpha_res[0];
#    pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const int col = nt * 16 + (lane >> 4);
#    pragma unroll
            for (int q = 0; q < 4; q++) {
                const int cl = col + 4 * q;
                if (cl < N) {
                    const float pv             = prms_to_float(acc[nt].x[q]);
                    proj[(size_t)row * N + cl] = prms_from_float<T>(pv);
                    const float alpha          = (cl < n_ch) ? ap : ((cl < 2 * n_ch) ? apo : ar);
                    const float h              = pv * alpha * inv_r + bias[cl];
                    if (cl < n_ch)
                        h_pre[(size_t)row * n_ch + cl] = prms_from_float<T>(1.0f / (1.0f + expf(-h)) + compute_h_eps);
                    else if (cl < 2 * n_ch)
                        h_post[(size_t)row * n_ch + (cl - n_ch)] = prms_from_float<T>(2.0f / (1.0f + expf(-h)));
                    else
                        h_res[(size_t)row * (N - 2 * n_ch) + (cl - 2 * n_ch)] = prms_from_float<T>(h);
                }
            }
        }
    }
}

template<typename T, typename W, typename KernelStorage, int CH_MODE>
static __global__ void __launch_bounds__(PRMS_THREADS)
    proj_rms_fwd_mma_kernel_impl(const T* __restrict__ x, const W* __restrict__ w, T* __restrict__ proj,
                                 T* __restrict__ r, const float* __restrict__ alpha_pre,
                                 const float* __restrict__ alpha_post, const float* __restrict__ alpha_res,
                                 const float* __restrict__ bias, T* __restrict__ h_pre, T* __restrict__ h_post,
                                 T* __restrict__ h_res, float* __restrict__ proj_f32, float* __restrict__ row_sq_f32,
                                 int* __restrict__ done, int M, int N, int K, float eps, int n_tiles, int ksplit,
                                 int n_ch, float compute_h_eps) {
    // CH_MODE == 0: proj_rms (r = 1/(norm/sqrt(K)+eps), no h outputs).
    // CH_MODE == 2: proj_rms + compute_h fused epilogue (r = norm/sqrt(K),
    //               h_pre/h_post/h_res written in the same launch).
    __shared__ int smem_last;
    constexpr int KFRAG  = PrmsCfg<KernelStorage>::kfrag;
    constexpr int KCHUNK = PrmsCfg<KernelStorage>::kchunk;
    constexpr int KLD    = PrmsCfg<KernelStorage>::kld;
    constexpr int VECLEN = PrmsCfg<KernelStorage>::veclen;

    const int m0    = blockIdx.x * PRMS_MTILE;
    const int ntile = blockIdx.y;  // 0..n_tiles-1
    const int lane  = threadIdx.x & 63;
    const int warp  = threadIdx.x >> 6;

    extern __shared__ char smem_raw[];
    KernelStorage* x_lds = (KernelStorage*)smem_raw;          // [64][KLD]
    KernelStorage* w_lds = x_lds + (size_t)PRMS_MTILE * KLD;  // [16][KLD] (this n-tile)

    DUFragment<accumulator, 16, 16, KFRAG, float> acc;
    du_fill_fragment(acc, 0.0f);
    float row_sq = 0.0f;  // per-thread partial row norm

    // Zero both LDS tiles once at kernel start: w rows beyond N, x rows
    // beyond M (partial M tile) and x/w columns beyond a partial K chunk
    // are never re-staged.  Zeroing keeps stale INF/NaN bit patterns left
    // in LDS by a previous kernel out of every MMA (0 * stale = NaN).
    for (int i = threadIdx.x; i < (PRMS_MTILE + 16) * KLD; i += PRMS_THREADS)
        ((KernelStorage*)smem_raw)[i] = prms_zero<KernelStorage>();
    if (threadIdx.x == 0) smem_last = 0;
    __syncthreads();

    // K is streamed in chunks; the tail chunk may be partial (K % KCHUNK
    // != 0).  The stage loops are bounded by the runtime per-chunk vper so
    // only live columns are touched; the LDS tail stays zero from above.
    // Split-K (ksplit > 1): each z-slice owns a private K range, so small-M
    // grids get more blocks instead of serializing the whole K loop.
    const int k_len   = (ksplit > 1) ? (K / ksplit) : K;
    const int k_lo    = (ksplit > 1) ? (blockIdx.z * k_len) : 0;
    const int nchunk  = (k_len + KCHUNK - 1) / KCHUNK;
    constexpr int NXV = PRMS_MTILE * (KCHUNK / VECLEN) / PRMS_THREADS;               // x vecs per thread
    constexpr int NWV = (16 * (KCHUNK / VECLEN) + PRMS_THREADS - 1) / PRMS_THREADS;  // w vecs per thread
    // Software pipeline: global loads are issued one chunk ahead into
    // registers so the MMAs of the current chunk hide the load latency.
    KernelStorage rx[NXV][VECLEN];
    KernelStorage rw[NWV][VECLEN];

    // Issue global loads for chunk k0 into registers (no LDS writes).
    auto stage_ld = [&](int k0, int vper) {
#    pragma unroll
        for (int i = 0; i < NXV; i++) {
            int u  = threadIdx.x + i * PRMS_THREADS;
            int rr = u / vper;
            int vv = u - rr * vper;
            if (rr < PRMS_MTILE) {
                int g_row = m0 + rr;
                if (g_row < M)
                    prms_copy_convert_vec<KernelStorage, T, VECLEN>(rx[i], x + (size_t)g_row * K + k0 + vv * VECLEN);
            }
        }
#    pragma unroll
        for (int i = 0; i < NWV; i++) {
            int u = threadIdx.x + i * PRMS_THREADS;
            if (u < 16 * vper) {
                int rr  = u / vper;
                int vv  = u - rr * vper;
                int g_n = ntile * 16 + rr;
                if (g_n < N)
                    prms_copy_convert_vec<KernelStorage, W, VECLEN>(rw[i], w + (size_t)g_n * K + k0 + vv * VECLEN);
            }
        }
    };
    // Write the previously loaded registers of chunk k0 into LDS.
    auto stage_st = [&](int k0, int vper) {
#    pragma unroll
        for (int i = 0; i < NXV; i++) {
            int u  = threadIdx.x + i * PRMS_THREADS;
            int rr = u / vper;
            int vv = u - rr * vper;
            if (rr < PRMS_MTILE) {
                int g_row = m0 + rr;
                if (g_row < M) prms_copy_vec<KernelStorage>(x_lds + (size_t)rr * KLD + vv * VECLEN, rx[i]);
            }
        }
#    pragma unroll
        for (int i = 0; i < NWV; i++) {
            int u = threadIdx.x + i * PRMS_THREADS;
            if (u < 16 * vper) {
                int rr  = u / vper;
                int vv  = u - rr * vper;
                int g_n = ntile * 16 + rr;
                if (g_n < N) prms_copy_vec<KernelStorage>(w_lds + (size_t)rr * KLD + vv * VECLEN, rw[i]);
            }
        }
    };

    stage_ld(k_lo, (k_len < KCHUNK ? k_len : KCHUNK) / VECLEN);
#    pragma unroll 1
    for (int c = 0; c < nchunk; c++) {
        const int k0 = k_lo + c * KCHUNK;
        int ck       = k_len - c * KCHUNK;
        if (ck > KCHUNK) ck = KCHUNK;
        stage_st(k0, ck / VECLEN);  // waits for loads(c) issued last iter
        if (c + 1 < nchunk) {
            int ck1 = k_len - (c + 1) * KCHUNK;
            if (ck1 > KCHUNK) ck1 = KCHUNK;
            stage_ld(k0 + KCHUNK, ck1 / VECLEN);  // issue next loads; MMA(c) hides them
        }
        __syncthreads();

// MMAC over this K chunk; A = x (rows x K), B = w^T (K x cols).
#    pragma unroll
        for (int kk = 0; kk < KCHUNK / KFRAG; kk++) {
            if (kk < ck / KFRAG) {
                DUFragment<matrix_a, 16, 16, KFRAG, KernelStorage, row_major> a;
                du_load_matrix_sync(a, x_lds + (size_t)(warp * 16) * KLD + kk * KFRAG, KLD);
                // Every block accumulates the full row norm (not just
                // ntile == 0) so the fused epilogue can run in this launch
                // for every n-tile.
                constexpr int FE = KFRAG / 4;
                const int xrow   = m0 + warp * 16 + (lane & 15);
                if (xrow < M) {
#    pragma unroll
                    for (int e = 0; e < FE; ++e) {
                        const float xv  = prms_to_float(x[(size_t)xrow * K + k0 + kk * KFRAG + (lane >> 4) * FE + e]);
                        row_sq         += xv * xv;
                    }
                }
                DUFragment<matrix_b, 16, 16, KFRAG, KernelStorage, col_major> b;
                du_load_matrix_sync(b, w_lds + kk * KFRAG, KLD);
                du_mma_sync(acc, a, b, acc);
            }
        }
        __syncthreads();  // free LDS for chunk c+1
    }

    // Row norm: the 4 quarter-warp lanes sharing a row each hold a disjoint
    // k slice of x^2; shuffle-reduce to get the full |x|^2 per row.  Every
    // block computes the complete row norm so the fused epilogue below can
    // run in the same launch for every n-tile.
    row_sq += __shfl_xor(row_sq, 16);
    row_sq += __shfl_xor(row_sq, 32);

    const int row     = m0 + warp * 16 + (lane & 15);
    const int col     = ntile * 16 + (lane >> 4);
    const bool row_ok = (row < M);

    if (ksplit > 1) {
        // Split-K: each z-slice stores its partials into a private fp32 slot
        // (single writer per element, plain stores — no atomics), then the
        // last block of this M-tile (across all n-tiles and z-slices)
        // finalizes every tile in-place by reducing the z-slots in FIXED
        // order, so the fp32 result is bitwise deterministic between runs
        // (the old atomicAdd accumulation order depended on block
        // scheduling).  Release on the done counter + agent-scope atomic
        // loads make the other blocks' slot stores visible to the finalizer.
        const int z = blockIdx.z;
        if (row_ok) {
            // The row norm is idempotent across n-tiles: only ntile == 0
            // blocks store it so each z-slot holds exactly one |x|^2 per row
            // (proj partials below are n-tile-specific).
            if (ntile == 0 && lane < 16) row_sq_f32[(size_t)z * M + row] = row_sq;
            if (col < N) proj_f32[((size_t)z * M + row) * N + col] = acc.x[0];
            if (col + 4 < N) proj_f32[((size_t)z * M + row) * N + col + 4] = acc.x[1];
            if (col + 8 < N) proj_f32[((size_t)z * M + row) * N + col + 8] = acc.x[2];
            if (col + 12 < N) proj_f32[((size_t)z * M + row) * N + col + 12] = acc.x[3];
        }
        // Each writer fences its partials; all waves must finish before thread 0 publishes completion.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            int* dcell = done + (size_t)(m0 / PRMS_MTILE);
            if (__hip_atomic_fetch_add(dcell, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == n_tiles * ksplit - 1)
                smem_last = 1;
        }
        __syncthreads();
        if (smem_last) {
            __threadfence();
            const int row0  = m0;
            const float ap  = (CH_MODE == 2) ? alpha_pre[0] : 0.0f;
            const float apo = (CH_MODE == 2) ? alpha_post[0] : 0.0f;
            const float ar  = (CH_MODE == 2) ? alpha_res[0] : 0.0f;
            const int ncols = n_tiles * 16;
            for (int u = threadIdx.x; u < PRMS_MTILE * ncols; u += PRMS_THREADS) {
                const int rw = row0 + (u / ncols);
                const int cl = u - (u / ncols) * ncols;
                if (rw < M && cl < N) {
                    float pv = 0.0f;
#    pragma unroll
                    for (int zz = 0; zz < ksplit; zz++)
                        pv += __hip_atomic_load(&proj_f32[((size_t)zz * M + rw) * N + cl], __ATOMIC_RELAXED,
                                                __HIP_MEMORY_SCOPE_AGENT);
                    proj[(size_t)rw * N + cl] = prms_from_float<T>(pv);
                    if (CH_MODE == 2) {
                        float sq = 0.0f;
#    pragma unroll
                        for (int zz = 0; zz < ksplit; zz++)
                            sq += __hip_atomic_load(&row_sq_f32[(size_t)zz * M + rw], __ATOMIC_RELAXED,
                                                    __HIP_MEMORY_SCOPE_AGENT);
                        const float rv    = sqrtf(sq) / sqrtf((float)K);
                        const float inv_r = 1.0f / (rv + eps);
                        const float alpha = (cl < n_ch) ? ap : ((cl < 2 * n_ch) ? apo : ar);
                        const float h     = pv * alpha * inv_r + bias[cl];
                        if (cl < n_ch)
                            h_pre[(size_t)rw * n_ch + cl] =
                                prms_from_float<T>(1.0f / (1.0f + expf(-h)) + compute_h_eps);
                        else if (cl < 2 * n_ch)
                            h_post[(size_t)rw * n_ch + (cl - n_ch)] = prms_from_float<T>(2.0f / (1.0f + expf(-h)));
                        else
                            h_res[(size_t)rw * (N - 2 * n_ch) + (cl - 2 * n_ch)] = prms_from_float<T>(h);
                    }
                }
            }
            for (int u = threadIdx.x; u < PRMS_MTILE; u += PRMS_THREADS) {
                const int rw = row0 + u;
                if (rw < M) {
                    float sq = 0.0f;
#    pragma unroll
                    for (int zz = 0; zz < ksplit; zz++)
                        sq += __hip_atomic_load(&row_sq_f32[(size_t)zz * M + rw], __ATOMIC_RELAXED,
                                                __HIP_MEMORY_SCOPE_AGENT);
                    if (CH_MODE == 0)
                        r[rw] = prms_from_float<T>(1.0f / (sqrtf(sq) / sqrtf((float)K) + eps));
                    else
                        r[rw] = prms_from_float<T>(sqrtf(sq) / sqrtf((float)K));
                }
            }
        }
    } else if (CH_MODE == 0) {
        // ksplit == 1, proj_rms: complete epilogue in-kernel (no second
        // launch).  r is idempotent across n-tiles (same value per row).
        if (row_ok) {
            if (lane < 16) {
                const float norm = sqrtf(row_sq);
                r[row]           = prms_from_float<T>(1.0f / (norm / sqrtf((float)K) + eps));
            }
            if (col < N) proj[(size_t)row * N + col] = prms_from_float<T>(acc.x[0]);
            if (col + 4 < N) proj[(size_t)row * N + col + 4] = prms_from_float<T>(acc.x[1]);
            if (col + 8 < N) proj[(size_t)row * N + col + 8] = prms_from_float<T>(acc.x[2]);
            if (col + 12 < N) proj[(size_t)row * N + col + 12] = prms_from_float<T>(acc.x[3]);
        }
    } else {
        // ksplit == 1, compute_h: fused epilogue from the accumulator.
        const float rv    = sqrtf(row_sq) / sqrtf((float)K);
        const float inv_r = 1.0f / (rv + eps);
        if (row_ok) {
            if (lane < 16) r[row] = prms_from_float<T>(rv);
            const float ap  = alpha_pre[0];
            const float apo = alpha_post[0];
            const float ar  = alpha_res[0];
#    pragma unroll
            for (int q = 0; q < 4; q++) {
                const int cl = col + 4 * q;
                if (cl < N) {
                    const float pv             = prms_to_float(acc.x[q]);
                    proj[(size_t)row * N + cl] = prms_from_float<T>(pv);
                    const float alpha          = (cl < n_ch) ? ap : ((cl < 2 * n_ch) ? apo : ar);
                    const float h              = pv * alpha * inv_r + bias[cl];
                    if (cl < n_ch)
                        h_pre[(size_t)row * n_ch + cl] = prms_from_float<T>(1.0f / (1.0f + expf(-h)) + compute_h_eps);
                    else if (cl < 2 * n_ch)
                        h_post[(size_t)row * n_ch + (cl - n_ch)] = prms_from_float<T>(2.0f / (1.0f + expf(-h)));
                    else
                        h_res[(size_t)row * (N - 2 * n_ch) + (cl - 2 * n_ch)] = prms_from_float<T>(h);
                }
            }
        }
    }
}

#else  // no MMAC intrinsics on this arch — kernels are never launched at runtime.

template<typename T, typename W, typename KernelStorage, int CH_MODE>
static __global__ void __launch_bounds__(PRMS_THREADS)
    proj_rms_fwd_mma_kernel_impl(const T*, const W*, T*, T*, const float*, const float*, const float*, const float*, T*,
                                 T*, T*, float*, float*, int*, int, int, int, float, int, int, int, float) {}

template<typename T, typename W, typename KernelStorage, int CFREE>
static __global__ void __launch_bounds__(PRMS_THREADS)
    proj_rms_ch_fwd_mma_v2_kernel(const T*, const W*, T*, T*, const float*, const float*, const float*, const float*,
                                  T*, T*, T*, int, int, int, float, int, float) {}

#endif

template<typename T>
static bool proj_rms_mma_path_ok(int N, int K) {
    return proj_rms_mma_supported() && N > 0 && N <= PRMS_NPAD && K >= (int)PrmsCfg<T>::kfrag &&
           K % (int)PrmsCfg<T>::kfrag == 0;
}

template<typename T>
static int proj_rms_fwd_pick_ksplit(int M, int N, int K, int n_tiles) {
    const int n_mblocks = (M + PRMS_MTILE - 1) / PRMS_MTILE;
    if (n_mblocks * n_tiles * 4 >= 64 || K < 4096) return 1;
    constexpr int KFRAG = PrmsCfg<T>::kfrag;  // 16 (bf16/fp16), 8 (fp32)
    for (int s = 8; s > 1; s >>= 1) {
        if (K % s == 0 && (K / s) % KFRAG == 0 && K / s >= KFRAG) return s;
    }
    return 1;
}

template<typename KernelStorage>
static int proj_rms_ch_fwd_pick_ksplit(int M, int N, int K, int n_tiles) {
    static const int ks_override = [] {
        const char* e = getenv(MHX_KS);
        return e ? atoi(e) : 0;
    }();
    int ksplit = proj_rms_fwd_pick_ksplit<KernelStorage>(M, N, K, n_tiles);
    // The fused two-N-tile path launches only ceil(M/64) CTAs.  At
    // M=512..1024 this leaves most CUs idle; split the long reduction.
    // Keep large-M shapes on the measured faster register epilogue path.
    if (M <= 1024 && n_tiles <= 2 && K >= 4096) {
        // Mixed bf16-input/fp32-weight computation uses the FP32 fragment
        // path too. Its small grids need additional CTAs just like bf16.
        // Aim for 128 CTAs on the two-N-tile path; large M keeps v2.
        const int preferred_split = sizeof(KernelStorage) == 4 && M > 512 ? 4 : 8;
        for (int split = preferred_split; split > 1; split >>= 1) {
            if (K % split == 0 && (K / split) % PrmsCfg<KernelStorage>::kfrag == 0) {
                ksplit = split;
                break;
            }
        }
    }
    if (ks_override > 0) {
        constexpr int KFRAG = PrmsCfg<KernelStorage>::kfrag;
        if (K % ks_override != 0 || (K / ks_override) % KFRAG != 0) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "MHX_KS must divide K into complete MMAC fragments");
        }
        ksplit = ks_override;
    }
    return ksplit;
}

static bool prms_bind_scratch(hipStream_t stream, int M, int N, int n_mblocks, int ksplit, void* workspace,
                              float** proj_f32, float** row_sq_f32, int** done) {
    if (!workspace) return false;
    *proj_f32   = static_cast<float*>(workspace);
    *row_sq_f32 = *proj_f32 + (size_t)ksplit * M * N;
    *done       = reinterpret_cast<int*>(*row_sq_f32 + (size_t)ksplit * M);
    return hipMemsetAsync(*done, 0, (size_t)n_mblocks * sizeof(int), stream) == hipSuccess;
}

template<typename T, typename W>
using PrmsCompute = typename std::conditional<std::is_same<T, W>::value, T, float>::type;

namespace ftrain {
namespace {
template<typename T, typename W, typename KernelStorage>
static bool launch_proj_rms_ch_fwd_mma_t(hipStream_t stream, const T* x, const W* w, T* proj, T* r_out,
                                         const float* alpha_pre, const float* alpha_post, const float* alpha_res,
                                         const float* bias, T* h_pre, T* h_post, T* h_res, int M, int N, int n, int K,
                                         float eps, float compute_h_eps, void* workspace);
size_t proj_rms_fwd_workspace_bytes(int M, int N, int K, bool compute_h);
bool launch_proj_rms_ch_fwd_mma(hipStream_t stream, const void* x, const void* w, void* proj, void* r_out,
                                const float* alpha_pre, const float* alpha_post, const float* alpha_res,
                                const float* bias, void* h_pre, void* h_post, void* h_res, int M, int N, int n, int K,
                                float eps, float compute_h_eps, void* workspace);

template<typename T, typename W, typename KernelStorage>
static bool launch_proj_rms_ch_fwd_mma_t(hipStream_t stream, const T* x, const W* w, T* proj, T* r_out,
                                         const float* alpha_pre, const float* alpha_post, const float* alpha_res,
                                         const float* bias, T* h_pre, T* h_post, T* h_res, int M, int N, int n, int K,
                                         float eps, float compute_h_eps, void* workspace) {
    if (!proj_rms_mma_path_ok<KernelStorage>(N, K)) return false;
    constexpr int KLD     = PrmsCfg<KernelStorage>::kld;
    size_t shm            = (size_t)(PRMS_MTILE + 16) * KLD * sizeof(KernelStorage);
    const int n_tiles     = (N + 15) / 16;
    const int n_mblocks   = (M + PRMS_MTILE - 1) / PRMS_MTILE;
    static const int nov2 = [] {
        const char* e = getenv(MHX_NOV2);
        return e ? atoi(e) : 0;
    }();
    const int ksplit  = proj_rms_ch_fwd_pick_ksplit<KernelStorage>(M, N, K, n_tiles);
    dim3 grid         = (ksplit > 1) ? dim3(n_mblocks, n_tiles, ksplit) : dim3(n_mblocks, n_tiles);
    float* proj_f32   = nullptr;
    float* row_sq_f32 = nullptr;
    int* done         = nullptr;
    if (ksplit > 1 && !prms_bind_scratch(stream, M, N, n_mblocks, ksplit, workspace, &proj_f32, &row_sq_f32, &done))
        return false;
    if (ksplit == 1 && n_tiles <= 2 && !nov2) {
        // v2 fast path: all n-tiles fused into one block; CFREE selects the
        // conflict-free LDS padding (8B stores) vs the classic 16B layout.
        static const int cfree = [] {
            const char* e = getenv(MHX_CFREE);
            // fp32: conflict-free KLD (8B stores) measured 2x faster; for
            // fp16/bf16 the 16B store path (classic KLD) is faster.
            return e ? atoi(e) : ((sizeof(KernelStorage) == 4) ? 1 : 0);
        }();
        constexpr int KLD2 =
            (sizeof(KernelStorage) == 4) ? (PrmsCfg<KernelStorage>::kchunk + 10) : (PrmsCfg<KernelStorage>::kchunk + 2);
        size_t shm2 = (size_t)(PRMS_MTILE + 32) * KLD2 * sizeof(KernelStorage);
        if (cfree != 0) {
            hipLaunchKernelGGL((proj_rms_ch_fwd_mma_v2_kernel<T, W, KernelStorage, 1>), dim3(n_mblocks), PRMS_THREADS,
                               shm2, stream, x, w, proj, r_out, alpha_pre, alpha_post, alpha_res, bias, h_pre, h_post,
                               h_res, M, N, K, eps, n, compute_h_eps);
            {
                const hipError_t status = hipGetLastError();
                if (status != hipSuccess) {
                    throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                            "proj_rms_compute_h_fwd kernel launch failed: %s",
                                            hipGetErrorString(status));
                }
            }
        } else {
            constexpr int KLD0 = PrmsCfg<KernelStorage>::kchunk + 8;
            size_t shm0        = (size_t)(PRMS_MTILE + 32) * KLD0 * sizeof(KernelStorage);
            hipLaunchKernelGGL((proj_rms_ch_fwd_mma_v2_kernel<T, W, KernelStorage, 0>), dim3(n_mblocks), PRMS_THREADS,
                               shm0, stream, x, w, proj, r_out, alpha_pre, alpha_post, alpha_res, bias, h_pre, h_post,
                               h_res, M, N, K, eps, n, compute_h_eps);
            {
                const hipError_t status = hipGetLastError();
                if (status != hipSuccess) {
                    throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                            "proj_rms_compute_h_fwd kernel launch failed: %s",
                                            hipGetErrorString(status));
                }
            }
        }
    } else {
        hipLaunchKernelGGL((proj_rms_fwd_mma_kernel_impl<T, W, KernelStorage, 2>), grid, PRMS_THREADS, shm, stream, x,
                           w, proj, r_out, alpha_pre, alpha_post, alpha_res, bias, h_pre, h_post, h_res, proj_f32,
                           row_sq_f32, done, M, N, K, eps, n_tiles, ksplit, n, compute_h_eps);
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "proj_rms_compute_h_fwd kernel launch failed: %s",
                                        hipGetErrorString(status));
            }
        }
    }
    return true;
}

static size_t proj_rms_fwd_workspace_bytes(int M, int N, int K, bool compute_h) {
    size_t bytes = 0;
    {
        using KernelStorage = PrmsCompute<__half, __half>;
        const int n_tiles   = (N + 15) / 16;
        const int split     = compute_h ? proj_rms_ch_fwd_pick_ksplit<KernelStorage>(M, N, K, n_tiles)
                                        : proj_rms_fwd_pick_ksplit<KernelStorage>(M, N, K, n_tiles);
        if (split > 1) {
            bytes =
                (size_t)split * M * (N + 1) * sizeof(float) + (size_t)((M + PRMS_MTILE - 1) / PRMS_MTILE) * sizeof(int);
        }
    }
    return bytes;
}

static bool launch_proj_rms_ch_fwd_mma(hipStream_t stream, const void* x, const void* w, void* proj, void* r_out,
                                       const float* alpha_pre, const float* alpha_post, const float* alpha_res,
                                       const float* bias, void* h_pre, void* h_post, void* h_res, int M, int N, int n,
                                       int K, float eps, float compute_h_eps, void* workspace) {
    {
        using T             = __half;
        using W             = __half;
        using KernelStorage = PrmsCompute<T, W>;
        return launch_proj_rms_ch_fwd_mma_t<T, W, KernelStorage>(
            stream, static_cast<const T*>(x), static_cast<const W*>(w), static_cast<T*>(proj), static_cast<T*>(r_out),
            alpha_pre, alpha_post, alpha_res, bias, static_cast<T*>(h_pre), static_cast<T*>(h_post),
            static_cast<T*>(h_res), M, N, n, K, eps, compute_h_eps, workspace);
    }
}
}  // namespace

bool launchFp16ProjRmsComputeHFwd(FTrainStream stream, const void* x, const void* weight, void* proj, void* r,
                                  const float* alpha_pre, const float* alpha_post, const float* alpha_res,
                                  const float* bias, void* h_pre, void* h_post, void* h_res, int rows, int outputs,
                                  int num_streams, int reduction, float eps, float compute_h_eps, void* workspace) {
    return launch_proj_rms_ch_fwd_mma(stream, x, weight, proj, r, alpha_pre, alpha_post, alpha_res, bias, h_pre, h_post,
                                      h_res, rows, outputs, num_streams, reduction, eps, compute_h_eps, workspace);
}

std::uint64_t fp16ProjRmsComputeHFwdWorkspaceBytes(int rows, int outputs, int reduction, bool compute_h) {
    return proj_rms_fwd_workspace_bytes(rows, outputs, reduction, compute_h);
}

}  // namespace ftrain
