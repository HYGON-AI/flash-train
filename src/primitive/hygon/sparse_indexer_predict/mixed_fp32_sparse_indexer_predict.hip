// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <math.h>

#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#    include <du_mma.h>
#endif

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_indexer_predict/mixed_fp32_sparse_indexer_predict.hpp"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

static inline int dsa_shared_memory_limit() {
    int device = 0, limit = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&limit, hipDeviceAttributeMaxSharedMemoryPerBlock, device) != hipSuccess)
        return 0;
    return limit;
}

static inline bool dsa_shared_memory_fits(size_t bytes) {
    return bytes <= static_cast<size_t>(dsa_shared_memory_limit());
}

template<typename T>
__device__ __forceinline__ static float dsa_to_float(T value) {
    return static_cast<float>(value);
}

template<>
__device__ __forceinline__ float dsa_to_float<float>(float value) {
    return value;
}

template<>
__device__ __forceinline__ float dsa_to_float<__half>(__half value) {
    return __half2float(value);
}

template<typename T>
__device__ __forceinline__ static T dsa_from_float(float value) {
    return static_cast<T>(value);
}

template<>
__device__ __forceinline__ float dsa_from_float<float>(float value) {
    return value;
}

template<>
__device__ __forceinline__ __half dsa_from_float<__half>(float value) {
    return __float2half(value);
}

template<typename Compute>
struct DsaMmacTraits {
    static constexpr int k               = sizeof(Compute) == 4 ? 8 : 16;
    static constexpr int vector_elements = 16 / sizeof(Compute);
    static constexpr int dense_dchunk    = sizeof(Compute) == 4 ? 64 : 128;
    static constexpr int sparse_chunk    = sizeof(Compute) == 4 ? 16 : 32;
};

template<typename Compute>
__device__ __forceinline__ static Compute dsa_load_as(const void* data, size_t index) {
    return reinterpret_cast<const Compute*>(data)[index];
}

template<typename Compute>
__device__ __forceinline__ static float dsa_load_compute_float(const void* data, size_t index) {
    return dsa_to_float(dsa_load_as<Compute>(data, index));
}

template<typename Compute, int Count>
__device__ __forceinline__ static void dsa_copy_as(Compute* dst, const void* src, size_t index) {
    constexpr int bytes = Count * sizeof(Compute);
    if constexpr (bytes == 16) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(reinterpret_cast<const Compute*>(src) + index);
    } else if constexpr (bytes == 8) {
        *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(reinterpret_cast<const Compute*>(src) + index);
    } else {
#pragma unroll
        for (int i = 0; i < Count; ++i) { dst[i] = reinterpret_cast<const Compute*>(src)[index + i]; }
    }
}

#define SPARSE_NWARPS 4
#define SPARSE_CHUNK 16
#define SPARSE_CSLICE 256

static inline bool sparse_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx928") != nullptr || strstr(prop.gcnArchName, "gfx936") != nullptr ||
           strstr(prop.gcnArchName, "gfx938") != nullptr;
}

static inline int sparse_mmac_max_lds() {
    const int max_lds = dsa_shared_memory_limit();
    return max_lds;
}

static inline bool sparse_mmac_shm_fits(int Hq, int D, int topk) {
    const int max_lds = sparse_mmac_max_lds();
    const int Hq_pad  = (Hq + 15) & ~15;
    size_t shm_lds    = (size_t)Hq_pad * (D + 8) * 2 + SPARSE_CHUNK * (D + 8) * 2 + (size_t)SPARSE_CSLICE * 8;
    if (Hq <= 64 && shm_lds + 2048 <= (size_t)max_lds) return true;
    const int dk = D / 16;
    size_t shm_g = (size_t)SPARSE_CHUNK * (D + 8) * 2 + (size_t)SPARSE_CSLICE * 8;
    return Hq > 0 && Hq <= 128 && shm_g + 2048 <= (size_t)max_lds &&
           (dk == 4 || dk == 8 || dk == 16 || dk == 28 || dk == 32 || dk == 36);
}

namespace {

namespace {
__global__ static void bf16_to_fp32Kernel(const hip_bfloat16* source, float* destination, std::uint64_t elements) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) { destination[index] = float(static_cast<float>(source[index])); }
}
}  // namespace

static void launchBf16ToFp32(const void* source, void* destination, std::uint64_t elements, FTrainStream stream) {
    hipLaunchKernelGGL(bf16_to_fp32Kernel, dim3((elements + 255) / 256), dim3(256), 0, stream,
                       static_cast<const hip_bfloat16*>(source), static_cast<float*>(destination), elements);
}

namespace {
__global__ static void fp16_to_fp32Kernel(const __half* source, float* destination, std::uint64_t elements) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) { destination[index] = float(static_cast<float>(source[index])); }
}
}  // namespace

static void launchFp16ToFp32(const void* source, void* destination, std::uint64_t elements, FTrainStream stream) {
    hipLaunchKernelGGL(fp16_to_fp32Kernel, dim3((elements + 255) / 256), dim3(256), 0, stream,
                       static_cast<const __half*>(source), static_cast<float*>(destination), elements);
}

namespace {
__global__ static void fp32_to_bf16Kernel(const float* source, hip_bfloat16* destination, std::uint64_t elements) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) { destination[index] = hip_bfloat16(static_cast<float>(source[index])); }
}
}  // namespace

static void launchFp32ToBf16(const void* source, void* destination, std::uint64_t elements, FTrainStream stream) {
    hipLaunchKernelGGL(fp32_to_bf16Kernel, dim3((elements + 255) / 256), dim3(256), 0, stream,
                       static_cast<const float*>(source), static_cast<hip_bfloat16*>(destination), elements);
}

namespace {
__global__ static void fp32_to_fp16Kernel(const float* source, __half* destination, std::uint64_t elements) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) { destination[index] = __half(static_cast<float>(source[index])); }
}
}  // namespace

static void launchFp32ToFp16(const void* source, void* destination, std::uint64_t elements, FTrainStream stream) {
    hipLaunchKernelGGL(fp32_to_fp16Kernel, dim3((elements + 255) / 256), dim3(256), 0, stream,
                       static_cast<const float*>(source), static_cast<__half*>(destination), elements);
}

}  // namespace

namespace {
using KernelStorage = float;
}  // namespace

// Adapted from fused_dsa/hip_dsa/kernels/sparse_mmac.hip.

#define SPARSE_NWARPS 4
#define SPARSE_CHUNK 16

#define SPARSE_CSLICE 256

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;

template<int MODE>
static __global__ void __launch_bounds__(256)
    mixed_fp32_sparse_indexer_predict_kernel(const void* __restrict__ q, const void* __restrict__ k,
                                             const void* __restrict__ w, const float* __restrict__ lse,
                                             const int* __restrict__ idxs, float* __restrict__ raw,
                                             int* __restrict__ row_count, int n_slices, int B, int Sq, int Sk, int Hq,
                                             int D, int topk, float scale, bool topk_indices_global) {
    constexpr int KFRAG  = DsaMmacTraits<KernelStorage>::k;
    constexpr int DCHUNK = DsaMmacTraits<KernelStorage>::dense_dchunk;
    constexpr int VEC    = DsaMmacTraits<KernelStorage>::vector_elements;
    constexpr int PAD    = sizeof(KernelStorage) == 4 ? 10 : 2;
    constexpr int LD     = DCHUNK + PAD;

    const int row     = blockIdx.x;
    const int slice   = blockIdx.y;
    const int b       = row / Sq;
    const int lane    = threadIdx.x & 63;
    const int warp    = threadIdx.x >> 6;
    const int local_h = warp * 16 + (lane & 15);
    const int t0      = slice * SPARSE_CSLICE;
    const int ck_all  = min(SPARSE_CSLICE, topk - t0);

    extern __shared__ char smem_raw[];
    KernelStorage* q_lds = reinterpret_cast<KernelStorage*>(smem_raw);  // [64][LD]
    KernelStorage* k_lds = q_lds + (size_t)64 * LD;                     // [32][LD]
    int* idx_lds         = reinterpret_cast<int*>(k_lds + (size_t)SPARSE_CHUNK * LD);
    float* score         = reinterpret_cast<float*>(idx_lds + SPARSE_CSLICE);

    for (int t = threadIdx.x; t < ck_all; t += 256) {
        idx_lds[t] = idxs[(size_t)row * topk + t0 + t];
        score[t]   = 0.0f;
    }
    __syncthreads();

    const int n_hpass = (Hq + 63) / 64;
    for (int hp = 0; hp < n_hpass; ++hp) {
        const int head = hp * 64 + local_h;
        float head_a;
        if constexpr (MODE == 1) {
            head_a = head < Hq ? lse[(size_t)row * Hq + head] : INFINITY;
        } else {
            head_a = head < Hq ? dsa_load_compute_float<KernelStorage>(w, (size_t)row * Hq + head) : 0.0f;
        }

        for (int chunk = 0; chunk < ck_all; chunk += SPARSE_CHUNK) {
            int ck = ck_all - chunk;
            if (ck > SPARSE_CHUNK) ck = SPARSE_CHUNK;
            const int ntiles = (ck + 15) >> 4;
            DUFragment<accumulator, 16, 16, KFRAG, float> acc[2];
#    pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                if (nt < ntiles) du_fill_fragment(acc[nt], 0.0f);

            for (int d0 = 0; d0 < D; d0 += DCHUNK) {
                int dc = D - d0;
                if (dc > DCHUNK) dc = DCHUNK;
                for (int i = threadIdx.x; i < (64 + SPARSE_CHUNK) * LD; i += 256) {
                    if (i < 64 * LD)
                        q_lds[i] = dsa_from_float<KernelStorage>(0.0f);
                    else
                        k_lds[i - 64 * LD] = dsa_from_float<KernelStorage>(0.0f);
                }
                __syncthreads();

                const int qv = CEIL_DIV(dc, VEC);
                for (int u = threadIdx.x; u < 64 * qv; u += 256) {
                    const int lh     = u / qv;
                    const int dv     = (u % qv) * VEC;
                    const int gh     = hp * 64 + lh;
                    const size_t src = ((size_t)row * Hq + gh) * D + d0 + dv;
                    if (gh < Hq && dv + VEC <= dc) {
                        dsa_copy_as<KernelStorage, VEC>(q_lds + (size_t)lh * LD + dv, q, src);
                    } else if (gh < Hq && dv < dc) {
                        for (int e = 0; e < VEC && dv + e < dc; ++e)
                            q_lds[(size_t)lh * LD + dv + e] = dsa_load_as<KernelStorage>(q, src + e);
                    }
                }

                const int kv = CEIL_DIV(dc, VEC);
                for (int u = threadIdx.x; u < ck * kv; u += 256) {
                    const int kr = u / kv;
                    const int dv = (u % kv) * VEC;
                    int idx      = idx_lds[chunk + kr];
                    if (idx < 0 || idx >= (topk_indices_global ? B * Sk : Sk)) idx = 0;
                    const size_t src = ((size_t)(topk_indices_global ? 0 : b) * Sk + idx) * D + d0 + dv;
                    if (dv + VEC <= dc) {
                        dsa_copy_as<KernelStorage, VEC>(k_lds + (size_t)kr * LD + dv, k, src);
                    } else if (dv < dc) {
                        for (int e = 0; e < VEC && dv + e < dc; ++e)
                            k_lds[(size_t)kr * LD + dv + e] = dsa_load_as<KernelStorage>(k, src + e);
                    }
                }
                __syncthreads();

                for (int kk = 0; kk < dc; kk += KFRAG) {
                    DUFragment<matrix_a, 16, 16, KFRAG, KernelStorage, row_major> afr;
                    du_load_matrix_sync(afr, q_lds + (size_t)(warp * 16) * LD + kk, LD);
#    pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        if (nt < ntiles) {
                            DUFragment<matrix_b, 16, 16, KFRAG, KernelStorage, col_major> bfr;
                            du_load_matrix_sync(bfr, k_lds + (size_t)(nt * 16) * LD + kk, LD);
                            du_mma_sync(acc[nt], afr, bfr, acc[nt]);
                        }
                    }
                }
                __syncthreads();
            }

#    pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                if (nt < ntiles) {
                    float v0, v1, v2, v3;
                    if constexpr (MODE == 1) {
                        v0 = expf(acc[nt].x[0] * scale - head_a);
                        v1 = expf(acc[nt].x[1] * scale - head_a);
                        v2 = expf(acc[nt].x[2] * scale - head_a);
                        v3 = expf(acc[nt].x[3] * scale - head_a);
                    } else {
                        const float d0 = acc[nt].x[0] * scale;
                        const float d1 = acc[nt].x[1] * scale;
                        const float d2 = acc[nt].x[2] * scale;
                        const float d3 = acc[nt].x[3] * scale;
                        v0             = (d0 > 0.0f ? d0 : 0.0f) * head_a;
                        v1             = (d1 > 0.0f ? d1 : 0.0f) * head_a;
                        v2             = (d2 > 0.0f ? d2 : 0.0f) * head_a;
                        v3             = (d3 > 0.0f ? d3 : 0.0f) * head_a;
                    }
#    pragma unroll
                    for (int off = 8; off > 0; off >>= 1) {
                        v0 += __shfl_xor(v0, off);
                        v1 += __shfl_xor(v1, off);
                        v2 += __shfl_xor(v2, off);
                        v3 += __shfl_xor(v3, off);
                    }
                    if ((lane & 15) == 0) {
                        const int g   = lane >> 4;
                        const int col = chunk + nt * 16 + g;
                        if (col < chunk + ck) atomicAdd(score + col, v0);
                        if (col + 4 < chunk + ck) atomicAdd(score + col + 4, v1);
                        if (col + 8 < chunk + ck) atomicAdd(score + col + 8, v2);
                        if (col + 12 < chunk + ck) atomicAdd(score + col + 12, v3);
                    }
                }
            __syncthreads();
        }
    }

    for (int t = threadIdx.x; t < ck_all; t += 256) {
        const int idx = idx_lds[t];
        float v       = score[t];
        if (idx < 0 || idx >= (topk_indices_global ? B * Sk : Sk)) v = MODE == 0 ? -INFINITY : 0.0f;
        raw[(size_t)row * topk + t0 + t] = v;
    }

    __threadfence();
    __syncthreads();
    __shared__ int last_block;
    if (threadIdx.x == 0) last_block = atomicAdd(row_count + row, 1) == n_slices - 1;
    __syncthreads();
    if (last_block) {
        float* rrow = raw + (size_t)row * topk;
        __shared__ float norm_red[256];
        if constexpr (MODE == 1) {
            float sum = 0.0f;
            for (int t = threadIdx.x; t < topk; t += 256) sum += rrow[t];
            norm_red[threadIdx.x] = sum;
            __syncthreads();
            for (int st = 128; st > 0; st >>= 1) {
                if (threadIdx.x < st) norm_red[threadIdx.x] += norm_red[threadIdx.x + st];
                __syncthreads();
            }
            {
                const float denom = isnan(norm_red[0]) ? norm_red[0] : fmaxf(norm_red[0], 1e-12f);
                const float inv   = 1.0f / denom;
                for (int t = threadIdx.x; t < topk; t += 256) rrow[t] *= inv;
            }
        } else {
            float mx = -INFINITY;
            for (int t = threadIdx.x; t < topk; t += 256) mx = fmaxf(mx, rrow[t]);
            norm_red[threadIdx.x] = mx;
            __syncthreads();
            for (int st = 128; st > 0; st >>= 1) {
                if (threadIdx.x < st) norm_red[threadIdx.x] = fmaxf(norm_red[threadIdx.x], norm_red[threadIdx.x + st]);
                __syncthreads();
            }
            mx        = norm_red[0];
            // Every wave must read the maximum before red is reused for the sum.
            __syncthreads();
            float sum = 0.0f;
            for (int t = threadIdx.x; t < topk; t += 256) {
                const float e  = (mx == -INFINITY ? 0.0f : expf(rrow[t] - mx));
                rrow[t]        = e;
                sum           += e;
            }
            norm_red[threadIdx.x] = sum;
            __syncthreads();
            for (int st = 128; st > 0; st >>= 1) {
                if (threadIdx.x < st) norm_red[threadIdx.x] += norm_red[threadIdx.x + st];
                __syncthreads();
            }
            const float inv = 1.0f / (norm_red[0] + 1e-30f);
            for (int t = threadIdx.x; t < topk; t += 256) rrow[t] *= inv;
        }
    }
}

#else  // gfx906 / gfx926: no MMAC intrinsics — never launched at runtime.

template<int MODE>
static __global__ void __launch_bounds__(256)
    mixed_fp32_sparse_indexer_predict_kernel(const void*, const void*, const void*, const float*, const int*, float*,
                                             int*, int, int, int, int, int, int, int, float, bool) {}

#endif  // supported gfx archs

static inline size_t sparse_mmac_dtype_shm() {
    constexpr int dchunk = DsaMmacTraits<KernelStorage>::dense_dchunk;
    constexpr int pad    = sizeof(KernelStorage) == 4 ? 10 : 2;
    constexpr int ld     = dchunk + pad;
    return (size_t)(64 + SPARSE_CHUNK) * ld * sizeof(KernelStorage) +
           (size_t)SPARSE_CSLICE * (sizeof(int) + sizeof(float));
}

namespace ftrain {
namespace {
template<int MODE>
static inline bool launch_sparse_gather_mmac_dtype(hipStream_t stream, const void* q, const void* k, const void* w,
                                                   const float* lse, const int* topk_idxs, float* raw, int* row_count,
                                                   int B, int Sq, int Sk, int Hq, int D, int topk, float scale,
                                                   bool topk_indices_global);
template<int MODE>
static bool launch_sparse_gather_mmac(hipStream_t stream, const void* q, const void* k, const void* w, const float* lse,
                                      const int* topk_idxs, float* raw, int* row_count, int B, int Sq, int Sk, int Hq,
                                      int D, int topk, float scale, bool topk_indices_global);
static bool launch_sparse_indexer_predict_mmac(hipStream_t stream, const void* q, const void* k, const void* w,
                                               const int* topk_idxs, float* raw, int* row_count, int B, int Sq, int Sk,
                                               int Hq, int D, int topk, float sm_scale, bool topk_indices_global);

static bool launch_sparse_indexer_predict(hipStream_t stream, const void* q, const void* k, const void* w,
                                          const int* topk_idxs, float* predict, int* row_count, int B, int Sq, int Sk,
                                          int Hq, int D, int topk, float sm_scale, bool topk_indices_global) {
    // Fused MMAC gather-GEMM path (tensor core) only; no non-fused fallback.
    // Hq may exceed 64 and D may be 512 (DSv4-flash/pro): the dispatcher picks
    // the LDS-Q kernel or the global-A kernel variant as the shape requires.
    if (!(sparse_mmac_supported() && Hq >= 1 && (D % 16) == 0 && topk >= 1 && sparse_mmac_shm_fits(Hq, D, topk))) {
        return false;
    }
    if (!launch_sparse_indexer_predict_mmac(stream, q, k, w, topk_idxs, predict, row_count, B, Sq, Sk, Hq, D, topk,
                                            sm_scale, topk_indices_global)) {
        return false;
    }
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                    "mixed_fp32_sparse_indexer_predict HIP call failed: %s", hipGetErrorString(status));
        }
    }
    return true;
}

template<int MODE>
static inline bool launch_sparse_gather_mmac_dtype(hipStream_t stream, const void* q, const void* k, const void* w,
                                                   const float* lse, const int* topk_idxs, float* raw, int* row_count,
                                                   int B, int Sq, int Sk, int Hq, int D, int topk, float scale,
                                                   bool topk_indices_global) {
    const int n_rows   = B * Sq;
    const int n_slices = CEIL_DIV(topk, SPARSE_CSLICE);
    if (!dsa_shared_memory_fits(sparse_mmac_dtype_shm() + 2048)) return false;
    hipLaunchKernelGGL((mixed_fp32_sparse_indexer_predict_kernel<MODE>), dim3(n_rows, n_slices), 256,
                       sparse_mmac_dtype_shm(), stream, q, k, w, lse, topk_idxs, raw, row_count, n_slices, B, Sq, Sk,
                       Hq, D, topk, scale, topk_indices_global);
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                    "mixed_fp32_sparse_indexer_predict HIP call failed: %s", hipGetErrorString(status));
        }
    }
    return true;
}

template<int MODE>
static bool launch_sparse_gather_mmac(hipStream_t stream, const void* q, const void* k, const void* w, const float* lse,
                                      const int* topk_idxs, float* raw, int* row_count, int B, int Sq, int Sk, int Hq,
                                      int D, int topk, float scale, bool topk_indices_global) {
    int n_rows   = B * Sq;
    int n_slices = (topk + SPARSE_CSLICE - 1) / SPARSE_CSLICE;
    {
        {
            return launch_sparse_gather_mmac_dtype<MODE>(stream, q, k, w, lse, topk_idxs, raw, row_count, B, Sq, Sk, Hq,
                                                         D, topk, scale, topk_indices_global);
        }
    }
}

static bool launch_sparse_indexer_predict_mmac(hipStream_t stream, const void* q, const void* k, const void* w,
                                               const int* topk_idxs, float* raw, int* row_count, int B, int Sq, int Sk,
                                               int Hq, int D, int topk, float sm_scale, bool topk_indices_global) {
    // The binding owns a fresh, zeroed counter for this in-flight call.
    if (!row_count || B <= 0 || Sq <= 0 || Sk <= 0 || Hq <= 0 || Hq > 128) return false;
    return launch_sparse_gather_mmac<0>(stream, q, k, w, nullptr, topk_idxs, raw, row_count, B, Sq, Sk, Hq, D, topk,
                                        sm_scale, topk_indices_global);
}
}  // namespace

bool mixedFp32SparseIndexerPredictSparseMmacSupported() { return sparse_mmac_supported(); }

bool mixedFp32SparseIndexerPredictSparseMmacShmFits(int query_heads, int head_dim, int selection_width) {
    return sparse_mmac_shm_fits(query_heads, head_dim, selection_width);
}

bool launchMixedFp32SparseIndexerPredict(FTrainStream stream, const void* q, const void* k, const void* w,
                                         const int* topk_indices, float* predict, int* row_count, int batch,
                                         int query_length, int key_length, int query_heads, int head_dim,
                                         int selection_width, float sm_scale, bool topk_indices_global) {
    return launch_sparse_indexer_predict(stream, q, k, w, topk_indices, predict, row_count, batch, query_length,
                                         key_length, query_heads, head_dim, selection_width, sm_scale,
                                         topk_indices_global);
}

void mixedFp32SparseIndexerPredictZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    const hipError_t status = hipMemsetAsync(memory, 0, static_cast<std::size_t>(bytes), stream);
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Workspace zero-fill failed: %s", hipGetErrorString(status));
    }
}

void castMixedFp32SparseIndexerPredict(const void* source, void* destination, FTrainNumericType source_type,
                                       FTrainNumericType destination_type, std::uint64_t elements,
                                       FTrainStream stream) {
    if (elements == 0) { return; }
    if (source_type == FTRAIN_NUMERIC_TYPE_BF16 && destination_type == FTRAIN_NUMERIC_TYPE_FP32) {
        launchBf16ToFp32(source, destination, elements, stream);
    } else if (source_type == FTRAIN_NUMERIC_TYPE_FP16 && destination_type == FTRAIN_NUMERIC_TYPE_FP32) {
        launchFp16ToFp32(source, destination, elements, stream);
    } else if (source_type == FTRAIN_NUMERIC_TYPE_FP32 && destination_type == FTRAIN_NUMERIC_TYPE_BF16) {
        launchFp32ToBf16(source, destination, elements, stream);
    } else if (source_type == FTRAIN_NUMERIC_TYPE_FP32 && destination_type == FTRAIN_NUMERIC_TYPE_FP16) {
        launchFp32ToFp16(source, destination, elements, stream);
    } else {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "unsupported mixed-storage conversion");
    }
    const auto status = hipGetLastError();
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "storage conversion launch failed: %s",
                        hipGetErrorString(status));
    }
}

}  // namespace ftrain
