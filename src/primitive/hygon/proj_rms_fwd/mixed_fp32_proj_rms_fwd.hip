// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <math.h>
#include <type_traits>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx936__) || defined(__gfx938__)
#    include <du_mma.h>
#endif

#include "flash_train/env.h"
#include "flash_train/error.hpp"
#include <stdint.h>

namespace hip_dsa_numeric {
// Small counter/accumulator resets are cheaper as a stream-ordered kernel
// than a graph memcpy/memset node. Large or byte-aligned regions keep HIP's memset.
static __global__ void zero_workspace_words(uint32_t* memory, uint64_t words) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < words) memory[i] = 0;
}
static inline hipError_t zero_workspace(void* memory, uint64_t bytes, hipStream_t stream) {
    if (bytes == 0) return hipSuccess;
    if (bytes <= 65536 && bytes % 4 == 0 && (reinterpret_cast<uintptr_t>(memory) & 3) == 0) {
        hipLaunchKernelGGL(zero_workspace_words, dim3((bytes + 1023) / 1024), dim3(256), 0, stream,
                           static_cast<uint32_t*>(memory), bytes / 4);
        return hipGetLastError();
    }
    return hipMemsetAsync(memory, 0, bytes, stream);
}


}  // namespace hip_dsa_numeric
#include "flash_train/primitive/hygon/proj_rms_fwd/mixed_fp32_proj_rms_fwd.hpp"


// Adapted from the fused_dsa/fused_mHC kernel implementations.

// Adapted from fused_mHC/csrc/fused_mhc_proj_rms_mma.cu.

#define PRMS_THREADS 256
#define PRMS_MTILE 64  // M rows per block (4 warps x 16 rows)
#define PRMS_NPAD 32   // N padded to a 16-multiple, capped at 32

// Per-dtype config: fp16/bf16 use 16x16x16 MMAC, fp32 uses 16x16x8.
template<typename T>
struct PrmsCfg {
    static constexpr int kfrag  = (sizeof(T) == 2) ? 16 : 8;
    static constexpr int kchunk = (sizeof(T) == 2) ? 128 : 64;  // fwd K chunk
    static constexpr int kld    = kchunk + 8;
    static constexpr int veclen = (sizeof(T) == 2) ? 8 : 4;  // vector copy width
};

// ---------------------------------------------------------------------------
// Device helpers (dtype conversion / vectorized staging)
// ---------------------------------------------------------------------------
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

// Coalesced 16-byte copy of VECLEN elements (8x fp16/bf16 or 4x fp32).
template<typename T>
__device__ __forceinline__ void prms_copy_vec(T* dst, const T* src) {
    if constexpr (sizeof(T) == 4) {
        *(float4*)dst = *(const float4*)src;
    } else {
        ((float2*)dst)[0] = ((const float2*)src)[0];
        ((float2*)dst)[1] = ((const float2*)src)[1];
    }
}

template<typename Dst, typename Src>
__device__ __forceinline__ Dst prms_convert(Src v) {
    if constexpr (std::is_same<Dst, Src>::value) {
        return v;
    } else {
        return prms_from_float<Dst>(prms_to_float(v));
    }
}

template<typename Compute, typename Src>
__device__ __forceinline__ float prms_to_compute_float(Src v) {
    return prms_to_float(prms_convert<Compute>(v));
}

template<typename Dst, typename Src, int VEC>
__device__ __forceinline__ void prms_copy_convert_vec(Dst* dst, const Src* src) {
    if constexpr (std::is_same<Dst, Src>::value) {
        prms_copy_vec<Dst>(dst, src);
    } else {
#pragma unroll
        for (int i = 0; i < VEC; i++) dst[i] = prms_convert<Dst>(src[i]);
    }
}

template<typename T, int VEC>
__device__ __forceinline__ void prms_load_vec(float* out, const T* src) {
    if constexpr (sizeof(T) == 4 && VEC == 4) {
        float4 v = *(const float4*)src;
        out[0]   = v.x;
        out[1]   = v.y;
        out[2]   = v.z;
        out[3]   = v.w;
    } else if constexpr (sizeof(T) == 2 && VEC == 8) {
        float4 v   = *(const float4*)src;
        const T* p = (const T*)&v;
#pragma unroll
        for (int i = 0; i < 8; i++) out[i] = prms_to_float(p[i]);
    } else {
#pragma unroll
        for (int i = 0; i < VEC; i++) out[i] = prms_to_float(src[i]);
    }
}

template<typename T, int VEC>
__device__ __forceinline__ void prms_store_vec(T* dst, const float* src) {
    if constexpr (sizeof(T) == 4 && VEC == 4) {
        *(float4*)dst = make_float4(src[0], src[1], src[2], src[3]);
    } else if constexpr (sizeof(T) == 2 && VEC == 8) {
        float4 r;
        T* p = (T*)&r;
#pragma unroll
        for (int i = 0; i < 8; i++) p[i] = prms_from_float<T>(src[i]);
        *(float4*)dst = r;
    } else {
#pragma unroll
        for (int i = 0; i < VEC; i++) dst[i] = prms_from_float<T>(src[i]);
    }
}

// ===========================================================================
// Backward portable fallback (MODE 1, any N/K, 16x16 block semantics).
// The fast path is proj_rms_bwd_fused_mma_kernel below (MMAC grad_w + FMA
// grad_x in one launch).  This kernel keeps MODE == 1 for unsupported
// shapes only.
// ===========================================================================


// ===========================================================================
// Fused Proj_RMS + compute_h  (mHC HyperConnection::compute_mappings)
//
//   r      = |x|_2 / sqrt(K)            (raw norm: no eps, NOT reciprocal)
//   h      = proj * alpha / (r + eps) + bias
//   h_pre  = sigmoid(h[:, :n]) + compute_h_eps
//   h_post = sigmoid(h[:, n:2n]) * 2
//   h_res  = h[:, 2n:]
// alpha = concat(alpha_pre.expand(n), alpha_post.expand(n),
//                alpha_res.expand(N - 2n)).
//
// Mirrors Megatron's ``_torch_proj_rms_compute_h`` / ``CutileProjRmsComputeH``
// exactly, including the ``r`` convention (which differs from ``proj_rms``).
//
// Forward fuses the mapping epilogue into the projection pass. Backward
// prepares FP32 score gradients and reductions, then computes grad_x/grad_w
// on the current stream without cross-CTA spinning.
// ===========================================================================

// dh for compute_h column c of row ``row`` — the chain of the activation:
//   h_pre  = sigmoid(h) + compute_h_eps  -> dh = g_h_pre * sig*(1-sig), sig = h_pre - compute_h_eps
//   h_post = 2*sigmoid(h)                -> dh = g_h_post * 2*sig*(1-sig), sig = h_post/2
//   h_res  = h                           -> dh = g_h_res
template<typename T>
__device__ __forceinline__ float prms_ch_dh(int c, int row, int n, int nres, const T* __restrict__ grad_h_pre,
                                            const T* __restrict__ grad_h_post, const T* __restrict__ grad_h_res,
                                            const T* __restrict__ h_pre, const T* __restrict__ h_post,
                                            float compute_h_eps) {
    if (c < n) {
        const float sig = prms_to_float(h_pre[(size_t)row * n + c]) - compute_h_eps;
        return prms_to_float(grad_h_pre[(size_t)row * n + c]) * sig * (1.0f - sig);
    } else if (c < 2 * n) {
        const int cn    = c - n;
        const float sig = prms_to_float(h_post[(size_t)row * n + cn]) * 0.5f;
        return prms_to_float(grad_h_post[(size_t)row * n + cn]) * 2.0f * sig * (1.0f - sig);
    } else {
        return prms_to_float(grad_h_res[(size_t)row * nres + (c - 2 * n)]);
    }
}

#if (defined(__gfx936__) || defined(__gfx938__))

#    include <du_mma.h>
using namespace du::dumma;

// ===========================================================================
// Forward v2 (compute_h): one block per 64-row M tile with ALL n-tiles fused
// into the same block (N <= 32 -> at most 2 tiles of 16).  Grid is 1D
// (M/64,).  Compared with proj_rms_fwd_mma_kernel_impl:
//   * the A (x) tile is loaded once per kk and shared by every n-tile, and
//     the per-row norm (frag_sq) is accumulated once per block instead of
//     once per n-tile -> ~2x less LDS traffic + norm work for N in (16,32];
//   * the LDS K-stride is padded to a conflict-free value for the 8B MMA
//     fragment loads (KLD % 32 == 10 for fp32 / == 2 for fp16/bf16), so the
//     matrix_a/matrix_b loads have no bank conflicts;
//   * LDS tiles are written with 8B vector stores (KLD no longer forces a
//     16B row stride), keeping the staging fully vectorized.
// The epilogue (r = norm/sqrt(K), sigmoid h_pre/h_post, h_res) is fused in
// the same single launch.  Only used when ksplit == 1 (no split-K).
// ===========================================================================


// ===========================================================================
// Forward: SINGLE fused kernel — proj = x @ w^T and r = 1/(|x|_2/sqrt(K)+eps).
// Grid: (ceil(M/64), n_tiles), block: 256 threads; warp w owns M rows
// [w*16, w*16+16); block y owns n-tile [nt*16, nt*16+16).
// LDS: x tile [64][KLD] + weight tile [16][KLD]; K streamed in chunks.
// Loop optimization: global staging is software-pipelined (loads for the
// next K chunk are issued into registers before the current chunk's MMAs,
// hiding LDS/global latency even with few blocks), the kk MMAC loop is fully
// unrolled, and the n-tile grid split gives 2x more blocks for N > 16 while
// halving the weight LDS tile.
// ===========================================================================
template<typename T, typename W, typename C, int CH_MODE, typename P = T, typename O = T>
__global__ void __launch_bounds__(PRMS_THREADS)
    proj_rms_fwd_mma_kernel_impl(const T* __restrict__ x, const W* __restrict__ w, P* __restrict__ proj,
                                 O* __restrict__ r, const float* __restrict__ alpha_pre,
                                 const float* __restrict__ alpha_post, const float* __restrict__ alpha_res,
                                 const float* __restrict__ bias, O* __restrict__ h_pre, O* __restrict__ h_post,
                                 O* __restrict__ h_res, float* __restrict__ proj_f32, float* __restrict__ row_sq_f32,
                                 int* __restrict__ done, int M, int N, int K, float eps, int n_tiles, int ksplit,
                                 int n_ch, float compute_h_eps, bool eps_inside_sqrt) {
    // Epsilon is placed inside sqrt or added outside it, never both.
    // CH_MODE == 0: proj_rms stores the reciprocal normalization factor.
    // CH_MODE == 2: compute_h stores RMS and applies that factor to the maps.
    __shared__ int smem_last;
    constexpr int KFRAG  = PrmsCfg<C>::kfrag;
    constexpr int KCHUNK = PrmsCfg<C>::kchunk;
    constexpr int KLD    = PrmsCfg<C>::kld;
    constexpr int VECLEN = PrmsCfg<C>::veclen;

    const int m0    = blockIdx.x * PRMS_MTILE;
    const int ntile = blockIdx.y;  // 0..n_tiles-1
    const int lane  = threadIdx.x & 63;
    const int warp  = threadIdx.x >> 6;

    extern __shared__ char smem_raw[];
    C* x_lds = (C*)smem_raw;                      // [64][KLD]
    C* w_lds = x_lds + (size_t)PRMS_MTILE * KLD;  // [16][KLD] (this n-tile)

    DUFragment<accumulator, 16, 16, KFRAG, float> acc;
    du_fill_fragment(acc, 0.0f);
    float row_sq = 0.0f;  // per-thread partial row norm

    // Zero both LDS tiles once at kernel start: w rows beyond N, x rows
    // beyond M (partial M tile) and x/w columns beyond a partial K chunk
    // are never re-staged.  Zeroing keeps stale INF/NaN bit patterns left
    // in LDS by a previous kernel out of every MMA (0 * stale = NaN).
    for (int i = threadIdx.x; i < (PRMS_MTILE + 16) * KLD; i += PRMS_THREADS) ((C*)smem_raw)[i] = prms_zero<C>();
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
    C rx[NXV][VECLEN];
    C rw[NWV][VECLEN];

    // Issue global loads for chunk k0 into registers (no LDS writes).
    auto stage_ld = [&](int k0, int vper) {
#    pragma unroll
        for (int i = 0; i < NXV; i++) {
            int u  = threadIdx.x + i * PRMS_THREADS;
            int rr = u / vper;
            int vv = u - rr * vper;
            if (rr < PRMS_MTILE) {
                int g_row = m0 + rr;
                if (g_row < M) prms_copy_convert_vec<C, T, VECLEN>(rx[i], x + (size_t)g_row * K + k0 + vv * VECLEN);
            }
        }
#    pragma unroll
        for (int i = 0; i < NWV; i++) {
            int u = threadIdx.x + i * PRMS_THREADS;
            if (u < 16 * vper) {
                int rr  = u / vper;
                int vv  = u - rr * vper;
                int g_n = ntile * 16 + rr;
                if (g_n < N) prms_copy_convert_vec<C, W, VECLEN>(rw[i], w + (size_t)g_n * K + k0 + vv * VECLEN);
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
                if (g_row < M) prms_copy_vec<C>(x_lds + (size_t)rr * KLD + vv * VECLEN, rx[i]);
            }
        }
#    pragma unroll
        for (int i = 0; i < NWV; i++) {
            int u = threadIdx.x + i * PRMS_THREADS;
            if (u < 16 * vper) {
                int rr  = u / vper;
                int vv  = u - rr * vper;
                int g_n = ntile * 16 + rr;
                if (g_n < N) prms_copy_vec<C>(w_lds + (size_t)rr * KLD + vv * VECLEN, rw[i]);
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
                DUFragment<matrix_a, 16, 16, KFRAG, C, row_major> a;
                du_load_matrix_sync(a, x_lds + (size_t)(warp * 16) * KLD + kk * KFRAG, KLD);
                // SDK lane mapping: row=lane&15, col=(lane>>4)*(KFRAG/4).
                // Reusing A removes the redundant global norm loads.
                constexpr int FE = KFRAG / 4;
#    pragma unroll
                for (int e = 0; e < FE; ++e) {
                    const float xv  = prms_to_float(a.x[e]);
                    row_sq         += xv * xv;
                }
                DUFragment<matrix_b, 16, 16, KFRAG, C, col_major> b;
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
        __threadfence();
        // Every writer must finish its fence before the counter publishes this block.
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
                    proj[(size_t)rw * N + cl] = prms_from_float<P>(pv);
                    if (CH_MODE == 2) {
                        float sq = 0.0f;
#    pragma unroll
                        for (int zz = 0; zz < ksplit; zz++)
                            sq += __hip_atomic_load(&row_sq_f32[(size_t)zz * M + rw], __ATOMIC_RELAXED,
                                                    __HIP_MEMORY_SCOPE_AGENT);
                        const float rv    = sqrtf(sq / (float)K + (eps_inside_sqrt ? eps : 0.0f));
                        const float inv_r = 1.0f / (rv + (eps_inside_sqrt ? 0.0f : eps));
                        const float alpha = (cl < n_ch) ? ap : ((cl < 2 * n_ch) ? apo : ar);
                        const float h     = pv * alpha * inv_r + bias[cl];
                        if (cl < n_ch)
                            h_pre[(size_t)rw * n_ch + cl] =
                                prms_from_float<O>(1.0f / (1.0f + expf(-h)) + compute_h_eps);
                        else if (cl < 2 * n_ch)
                            h_post[(size_t)rw * n_ch + (cl - n_ch)] = prms_from_float<O>(2.0f / (1.0f + expf(-h)));
                        else
                            h_res[(size_t)rw * (N - 2 * n_ch) + (cl - 2 * n_ch)] = prms_from_float<O>(h);
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
                    const float rv = sqrtf(sq / (float)K + (eps_inside_sqrt ? eps : 0.0f));
                    if (CH_MODE == 0)
                        r[rw] = prms_from_float<O>(1.0f / (rv + (eps_inside_sqrt ? 0.0f : eps)));
                    else
                        r[rw] = prms_from_float<O>(rv);
                }
            }
        }
    } else if (CH_MODE == 0) {
        // ksplit == 1, proj_rms: complete epilogue in-kernel (no second
        // launch).  r is idempotent across n-tiles (same value per row).
        if (row_ok) {
            if (lane < 16) {
                const float rv = sqrtf(row_sq / (float)K + (eps_inside_sqrt ? eps : 0.0f));
                r[row]         = prms_from_float<O>(1.0f / (rv + (eps_inside_sqrt ? 0.0f : eps)));
            }
            if (col < N) proj[(size_t)row * N + col] = prms_from_float<P>(acc.x[0]);
            if (col + 4 < N) proj[(size_t)row * N + col + 4] = prms_from_float<P>(acc.x[1]);
            if (col + 8 < N) proj[(size_t)row * N + col + 8] = prms_from_float<P>(acc.x[2]);
            if (col + 12 < N) proj[(size_t)row * N + col + 12] = prms_from_float<P>(acc.x[3]);
        }
    } else {
        // ksplit == 1, compute_h: fused epilogue from the accumulator.
        const float rv    = sqrtf(row_sq / (float)K + (eps_inside_sqrt ? eps : 0.0f));
        const float inv_r = 1.0f / (rv + (eps_inside_sqrt ? 0.0f : eps));
        if (row_ok) {
            if (lane < 16) r[row] = prms_from_float<O>(rv);
            const float ap  = alpha_pre[0];
            const float apo = alpha_post[0];
            const float ar  = alpha_res[0];
#    pragma unroll
            for (int q = 0; q < 4; q++) {
                const int cl = col + 4 * q;
                if (cl < N) {
                    const float pv             = prms_to_float(acc.x[q]);
                    proj[(size_t)row * N + cl] = prms_from_float<P>(pv);
                    const float alpha          = (cl < n_ch) ? ap : ((cl < 2 * n_ch) ? apo : ar);
                    const float h              = pv * alpha * inv_r + bias[cl];
                    if (cl < n_ch)
                        h_pre[(size_t)row * n_ch + cl] = prms_from_float<O>(1.0f / (1.0f + expf(-h)) + compute_h_eps);
                    else if (cl < 2 * n_ch)
                        h_post[(size_t)row * n_ch + (cl - n_ch)] = prms_from_float<O>(2.0f / (1.0f + expf(-h)));
                    else
                        h_res[(size_t)row * (N - 2 * n_ch) + (cl - 2 * n_ch)] = prms_from_float<O>(h);
                }
            }
        }
    }
}

// ===========================================================================
// Backward FAST path — SINGLE fused kernel: grad_w + grad_x in ONE launch.
// Grid (1D): [0, nbw) = grad_w MMAC blocks, [nbw, nbw+nbx) = grad_x blocks.
//
// grad_w[n,k] = sum_m grad_proj[m,n] * x[m,k]:
//   one block owns 16 n-rows x 64 k-cols (4 warps, one 16k slice each);
//   M is streamed as the MMA contraction dim in 64/32-row steps.  Four
//   independent MMA accumulators (one per m sub-tile) keep the MFMA chains
//   short and are summed once at the end — the full-M reduction lives inside
//   the accumulators, so there are no partials, no atomics, no second kernel.
//   gp^T is staged once per step (shared by all warps); x^T is staged per
//   warp with 16B vectorized, fully-coalesced global reads (transposed in
//   LDS).  The LDS tiles are double-buffered so the next step's global
//   latency overlaps the current step's MMAs (1 barrier per step).
//
// grad_x[m,k] = sum_n grad_proj[m,n] * w[n,k] + grad_r*dr/dx:
//   blocks cover MB_X=32 rows x KS_X cols; the w[n][KS_X] tile and the
//   grad_proj tile [32][N] are staged in LDS.  Loop optimization: each
//   thread registers 8 rows at once (acc[8][VEC]), so the w tile is read
//   from LDS once per n and reused across all 8 rows (8x less LDS traffic
//   than a 1-row-per-thread layout), and the RMS epilogue is fused per row.
// ===========================================================================


// compute_h backward has separate producer and consumer launches on the same
// stream. A launch boundary publishes scratch without cross-CTA spinning.


// FP32 grad_x tile. Four waves share one weight tile and cover 64 input rows;
// each wave reuses its A fragment across four output-column fragments. The
// staging storage is reused for coalesced output after all MMAC reads finish.


#else  // no MMAC intrinsics on this arch — kernels are never launched at runtime.


template<typename T, typename W, typename C, int CH_MODE, typename P = T, typename O = T>
__global__ void __launch_bounds__(PRMS_THREADS)
    proj_rms_fwd_mma_kernel_impl(const T*, const W*, P*, O*, const float*, const float*, const float*, const float*, O*,
                                 O*, O*, float*, float*, int*, int, int, int, float, int, int, int, float, bool) {}


#endif


// One CTA owns a row: FP32 norm reduction, then one thread per projection
// column. Only tail/wide shapes use this path; aligned production uses MMAC.


// ===========================================================================


static bool prms_bind_scratch(hipStream_t stream, int M, int N, int n_mblocks, int ksplit, void* workspace,
                              float** proj_f32, float** row_sq_f32, int** done) {
    *proj_f32   = static_cast<float*>(workspace);
    *row_sq_f32 = *proj_f32 + (size_t)ksplit * M * N;
    *done       = reinterpret_cast<int*>(*row_sq_f32 + (size_t)ksplit * M);
    return hip_dsa_numeric::zero_workspace(*done, (size_t)n_mblocks * sizeof(int), stream) == hipSuccess;
}

namespace ftrain {
namespace {
template<class Implementation, typename T, typename W, typename KernelStorage>
static bool launch_proj_rms_fwd_mma_t(hipStream_t stream, const T* x, const W* w,
                                      T* proj, T* r, int M, int N, int K, float eps, void* workspace);

template<class Implementation, typename T, typename W, typename KernelStorage>
static bool launch_proj_rms_fwd_mma_t(hipStream_t stream, const T* x, const W* w,
                                      T* proj, T* r, int M, int N, int K, float eps, void* workspace) {
    constexpr int KLD   = PrmsCfg<KernelStorage>::kld;
    size_t shm          = (size_t)(PRMS_MTILE + 16) * KLD * sizeof(KernelStorage);
    const int n_tiles   = (N + 15) / 16;
    const int n_mblocks = (M + PRMS_MTILE - 1) / PRMS_MTILE;
    const int ksplit    = Implementation::kSplit;
    dim3 grid           = (ksplit > 1) ? dim3(n_mblocks, n_tiles, ksplit) : dim3(n_mblocks, n_tiles);
    float* proj_f32     = nullptr;
    float* row_sq_f32   = nullptr;
    int* done           = nullptr;
    if (ksplit > 1 && !prms_bind_scratch(stream, M, N, n_mblocks, ksplit, workspace, &proj_f32, &row_sq_f32, &done))
        return false;
    hipLaunchKernelGGL((proj_rms_fwd_mma_kernel_impl<T, W, KernelStorage, 0>), grid, PRMS_THREADS, shm, stream, x, w,
                       proj, r, (const float*)nullptr, (const float*)nullptr, (const float*)nullptr,
                       (const float*)nullptr, (T*)nullptr, (T*)nullptr, (T*)nullptr, proj_f32, row_sq_f32, done, M, N,
                       K, eps, n_tiles, ksplit, 0, 0.0f, false);
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "proj_rms_fwd kernel launch failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}

template<class Implementation, typename T, typename W>
static bool launch_proj_rms_fwd_mma(hipStream_t stream, const void* x, const void* w, void* proj, void* r, int M, int N,
                                    int K, float eps, void* workspace) {
    {
        using KernelStorage = float;
        return launch_proj_rms_fwd_mma_t<Implementation, T, W, KernelStorage>(stream, static_cast<const T*>(x),
                                                              static_cast<const W*>(w), static_cast<T*>(proj),
                                                              static_cast<T*>(r), M, N, K, eps, workspace);
    }
}
}  // namespace

}  // namespace ftrain

namespace ftrain {

template<class Implementation, FTrainNumericType X_TYPE, FTrainNumericType W_TYPE>
bool launchMixedFp32ProjRmsFwd(FTrainStream stream, const void* x, const void* weight, void* proj, void* r, int rows,
                               int outputs, int reduction, float eps, void* workspace, FTrainNumericType x_type,
                               FTrainNumericType weight_type) {
    if constexpr (X_TYPE == FTRAIN_NUMERIC_TYPE_BF16 && W_TYPE == FTRAIN_NUMERIC_TYPE_FP16) {
        return launch_proj_rms_fwd_mma<Implementation, __hip_bfloat16, __half>(stream, x, weight, proj, r, rows, outputs,
                                                               reduction, eps, workspace);
    }
    if constexpr (X_TYPE == FTRAIN_NUMERIC_TYPE_BF16 && W_TYPE == FTRAIN_NUMERIC_TYPE_FP32) {
        return launch_proj_rms_fwd_mma<Implementation, __hip_bfloat16, float>(stream, x, weight, proj, r, rows, outputs,
                                                              reduction, eps, workspace);
    }
    if constexpr (X_TYPE == FTRAIN_NUMERIC_TYPE_FP16 && W_TYPE == FTRAIN_NUMERIC_TYPE_BF16) {
        return launch_proj_rms_fwd_mma<Implementation, __half, __hip_bfloat16>(stream, x, weight, proj, r, rows, outputs,
                                                               reduction, eps, workspace);
    }
    if constexpr (X_TYPE == FTRAIN_NUMERIC_TYPE_FP16 && W_TYPE == FTRAIN_NUMERIC_TYPE_FP32) {
        return launch_proj_rms_fwd_mma<Implementation, __half, float>(stream, x, weight, proj, r, rows, outputs, reduction,
                                                      eps, workspace);
    }
    if constexpr (X_TYPE == FTRAIN_NUMERIC_TYPE_FP32 && W_TYPE == FTRAIN_NUMERIC_TYPE_BF16) {
        return launch_proj_rms_fwd_mma<Implementation, float, __hip_bfloat16>(stream, x, weight, proj, r, rows, outputs,
                                                              reduction, eps, workspace);
    }
    if constexpr (X_TYPE == FTRAIN_NUMERIC_TYPE_FP32 && W_TYPE == FTRAIN_NUMERIC_TYPE_FP16) {
        return launch_proj_rms_fwd_mma<Implementation, float, __half>(stream, x, weight, proj, r, rows, outputs, reduction,
                                                      eps, workspace);
    }
    return false;
}

// Explicit fixed launch instances used by the C++ configuration code.
#define FTRAIN_INSTANTIATE_LAUNCH(Implementation, X_TYPE, W_TYPE)                                           \
    template bool launchMixedFp32ProjRmsFwd<Implementation, X_TYPE, W_TYPE>(                                \
        FTrainStream stream, const void* x, const void* weight, void* proj, void* r, int rows, int outputs, \
        int reduction, float eps, void* workspace, FTrainNumericType x_type, FTrainNumericType weight_type);
#define FTRAIN_INSTANTIATE_LAUNCH_SET_0(Implementation, W_TYPE)                  \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, FTRAIN_NUMERIC_TYPE_FP32, W_TYPE); \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, FTRAIN_NUMERIC_TYPE_BF16, W_TYPE);
#define FTRAIN_INSTANTIATE_LAUNCH_SET_1(Implementation, W_TYPE)                  \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, FTRAIN_NUMERIC_TYPE_FP32, W_TYPE); \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, FTRAIN_NUMERIC_TYPE_FP16, W_TYPE);
#define FTRAIN_INSTANTIATE_LAUNCH_SET_2(Implementation, W_TYPE)                  \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, FTRAIN_NUMERIC_TYPE_FP16, W_TYPE); \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, FTRAIN_NUMERIC_TYPE_BF16, W_TYPE);
FTRAIN_INSTANTIATE_LAUNCH_SET_2(MixedFp32ProjRmsFwd, FTRAIN_NUMERIC_TYPE_FP32);
FTRAIN_INSTANTIATE_LAUNCH_SET_0(MixedFp32ProjRmsFwd, FTRAIN_NUMERIC_TYPE_FP16);
FTRAIN_INSTANTIATE_LAUNCH_SET_1(MixedFp32ProjRmsFwd, FTRAIN_NUMERIC_TYPE_BF16);
FTRAIN_INSTANTIATE_LAUNCH_SET_2(MixedFp32ProjRmsFwdSplit2, FTRAIN_NUMERIC_TYPE_FP32);
FTRAIN_INSTANTIATE_LAUNCH_SET_0(MixedFp32ProjRmsFwdSplit2, FTRAIN_NUMERIC_TYPE_FP16);
FTRAIN_INSTANTIATE_LAUNCH_SET_1(MixedFp32ProjRmsFwdSplit2, FTRAIN_NUMERIC_TYPE_BF16);
FTRAIN_INSTANTIATE_LAUNCH_SET_2(MixedFp32ProjRmsFwdSplit4, FTRAIN_NUMERIC_TYPE_FP32);
FTRAIN_INSTANTIATE_LAUNCH_SET_0(MixedFp32ProjRmsFwdSplit4, FTRAIN_NUMERIC_TYPE_FP16);
FTRAIN_INSTANTIATE_LAUNCH_SET_1(MixedFp32ProjRmsFwdSplit4, FTRAIN_NUMERIC_TYPE_BF16);
FTRAIN_INSTANTIATE_LAUNCH_SET_2(MixedFp32ProjRmsFwdSplit8, FTRAIN_NUMERIC_TYPE_FP32);
FTRAIN_INSTANTIATE_LAUNCH_SET_0(MixedFp32ProjRmsFwdSplit8, FTRAIN_NUMERIC_TYPE_FP16);
FTRAIN_INSTANTIATE_LAUNCH_SET_1(MixedFp32ProjRmsFwdSplit8, FTRAIN_NUMERIC_TYPE_BF16);
#undef FTRAIN_INSTANTIATE_LAUNCH_SET_2
#undef FTRAIN_INSTANTIATE_LAUNCH_SET_1
#undef FTRAIN_INSTANTIATE_LAUNCH_SET_0
#undef FTRAIN_INSTANTIATE_LAUNCH

}  // namespace ftrain
