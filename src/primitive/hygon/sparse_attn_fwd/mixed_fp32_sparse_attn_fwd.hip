// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <math.h>

#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

// DTK du_mma.h defines a non-inline host function; keep vendor fragments local.
namespace {
#include <du_mma.h>
}

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_attn_fwd/mixed_fp32_sparse_attn_fwd.hpp"

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

template<>
__device__ __forceinline__ __hip_bfloat16 dsa_from_float<__hip_bfloat16>(float value) {
    return __float2bfloat16(value);
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

template<typename Native>
__device__ __forceinline__ static void dsa_store_from_float(void* data, size_t index, float value) {
    reinterpret_cast<Native*>(data)[index] = dsa_from_float<Native>(value);
}

#define SAF_THREADS 256
#define SAF_HBLK 64
#define SAF_MAX_D 576

static inline bool sparse_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx928") != nullptr || strstr(prop.gcnArchName, "gfx936") != nullptr ||
           strstr(prop.gcnArchName, "gfx938") != nullptr;
}

static inline bool sparse_attn_fused_supported() { return sparse_mmac_supported(); }

namespace hip_dsa_mla64 {

#if defined(__gfx936__) || defined(__gfx938__)
using namespace du::dumma;

#endif
}  // namespace hip_dsa_mla64

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

// Adapted from fused_dsa/hip_dsa/kernels/sparse_attn_fwd.hip.

typedef uint16_t bf16;

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
using namespace du::dumma;
typedef __hip_bfloat16 hbf16;

#endif  // supported gfx archs

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

template<typename C, int NK>
static __global__ void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_fwd_dtype_kernel(const void* __restrict__ q, const void* __restrict__ kv,
                                 const float* __restrict__ attn_sink, const int* __restrict__ idxs,
                                 const int* __restrict__ topk_length, void* __restrict__ out, float* __restrict__ lse,
                                 float* __restrict__ max_logits, float* __restrict__ lse_indexer, int indexer_topk,
                                 int T, int H, int D_rt, int D_v, int T_kv, int topk, float softmax_scale) {
    constexpr int KFRAG      = DsaMmacTraits<C>::k;
    constexpr int FRAG_ELEMS = sizeof(C) == 4 ? 2 : 4;
    constexpr int D          = NK * KFRAG;
    constexpr int NOUT       = D / 16;
    constexpr int VEC        = DsaMmacTraits<C>::vector_elements;
    constexpr int PAD        = sizeof(C) == 4 ? 10 : 8;
    constexpr int KLD        = D + PAD;
    (void)T;
    (void)D_rt;

    const int t        = blockIdx.x;
    const int hb       = blockIdx.y * SAF_HBLK;
    const int lane     = threadIdx.x & 63;
    const int warp     = threadIdx.x >> 6;
    const int r        = lane & 15;
    const int g        = lane >> 4;
    const int h        = hb + warp * 16 + r;
    const bool lane_ok = h < H;

    extern __shared__ char smem_raw[];
    C* kv_lds    = reinterpret_cast<C*>(smem_raw);  // [16][KLD]
    int* idx_lds = reinterpret_cast<int*>(kv_lds + (size_t)16 * KLD);
    C* p_lds     = reinterpret_cast<C*>(idx_lds + topk);  // [4][16][16]

    for (int i = threadIdx.x; i < topk; i += SAF_THREADS) idx_lds[i] = idxs[(size_t)t * topk + i];
    __syncthreads();

    int n_valid = topk;
    if (topk_length) {
        const int n = topk_length[t];
        if (n < n_valid) n_valid = n > 0 ? n : 0;
    }

    C qreg[NK][4];
    const int qh = lane_ok ? h : (H - 1);
#    pragma unroll
    for (int kk = 0; kk < NK; ++kk) {
#    pragma unroll
        for (int e = 0; e < FRAG_ELEMS; ++e) {
            qreg[kk][e] = lane_ok ? dsa_load_as<C>(q, ((size_t)t * H + qh) * D + kk * KFRAG + g * FRAG_ELEMS + e)
                                  : dsa_from_float<C>(0.0f);
        }
    }

    DUFragment<accumulator, 16, 16, KFRAG, float> outacc[NOUT];
#    pragma unroll
    for (int dc = 0; dc < NOUT; ++dc) du_fill_fragment(outacc[dc], 0.0f);

    float sink_val = -INFINITY;
    float m        = -INFINITY;
    float l        = 0.0f;
    float m_kv     = -INFINITY;
    float m_idx    = -INFINITY;
    float l_idx    = 0.0f;
    if (lane_ok) { sink_val = attn_sink[h]; }

    const int ntiles = CEIL_DIV(topk, 16);
    for (int kt = 0; kt < ntiles; ++kt) {
        int ck = topk - kt * 16;
        if (ck > 16) ck = 16;
        for (int i = threadIdx.x; i < 16 * KLD; i += SAF_THREADS) kv_lds[i] = dsa_from_float<C>(0.0f);
        __syncthreads();
        const int vecs = CEIL_DIV(D, VEC);
        for (int u = threadIdx.x; u < ck * vecs; u += SAF_THREADS) {
            const int kr  = u / vecs;
            const int dv  = (u % vecs) * VEC;
            const int key = kt * 16 + kr;
            int idx       = idx_lds[key];
            if (key >= n_valid || idx < 0 || idx >= T_kv) continue;
            const size_t src = (size_t)idx * D + dv;
            if (dv + VEC <= D) {
                dsa_copy_as<C, VEC>(kv_lds + (size_t)kr * KLD + dv, kv, src);
            } else {
                for (int e = 0; e < VEC && dv + e < D; ++e)
                    kv_lds[(size_t)kr * KLD + dv + e] = dsa_load_as<C>(kv, src + e);
            }
        }
        __syncthreads();

        DUFragment<accumulator, 16, 16, KFRAG, float> sacc;
        du_fill_fragment(sacc, 0.0f);
#    pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
            DUFragment<matrix_a, 16, 16, KFRAG, C, row_major> afr;
#    pragma unroll
            for (int e = 0; e < FRAG_ELEMS; ++e) afr.x[e] = qreg[kk][e];
            DUFragment<matrix_b, 16, 16, KFRAG, C, col_major> bfr;
            du_load_matrix_sync(bfr, kv_lds + kk * KFRAG, KLD);
            du_mma_sync(sacc, afr, bfr, sacc);
        }

        const int c0 = kt * 16 + g;
        const int c1 = c0 + 4, c2 = c0 + 8, c3 = c0 + 12;
        const bool ok0 = c0 < n_valid && idx_lds[c0] >= 0 && idx_lds[c0] < T_kv;
        const bool ok1 = c1 < n_valid && idx_lds[c1] >= 0 && idx_lds[c1] < T_kv;
        const bool ok2 = c2 < n_valid && idx_lds[c2] >= 0 && idx_lds[c2] < T_kv;
        const bool ok3 = c3 < n_valid && idx_lds[c3] >= 0 && idx_lds[c3] < T_kv;
        const float s0 = lane_ok ? sacc.x[0] * softmax_scale : -INFINITY;
        const float s1 = lane_ok ? sacc.x[1] * softmax_scale : -INFINITY;
        const float s2 = lane_ok ? sacc.x[2] * softmax_scale : -INFINITY;
        const float s3 = lane_ok ? sacc.x[3] * softmax_scale : -INFINITY;
        float tmax     = -INFINITY;
        if (ok0) tmax = fmaxf(tmax, s0);
        if (ok1) tmax = fmaxf(tmax, s1);
        if (ok2) tmax = fmaxf(tmax, s2);
        if (ok3) tmax = fmaxf(tmax, s3);
        tmax = fmaxf(tmax, __shfl_xor(tmax, 16));
        tmax = fmaxf(tmax, __shfl_xor(tmax, 32));
        m_kv = fmaxf(m_kv, tmax);
        if (lse_indexer != nullptr) {
            float imax = -INFINITY;
            if (ok0 && c0 < indexer_topk) imax = fmaxf(imax, s0);
            if (ok1 && c1 < indexer_topk) imax = fmaxf(imax, s1);
            if (ok2 && c2 < indexer_topk) imax = fmaxf(imax, s2);
            if (ok3 && c3 < indexer_topk) imax = fmaxf(imax, s3);
            imax = fmaxf(imax, __shfl_xor(imax, 16));
            imax = fmaxf(imax, __shfl_xor(imax, 32));
            if (imax > m_idx) {
                l_idx *= expf(m_idx - imax);
                m_idx  = imax;
            }
            float ies = 0.0f;
            if (ok0 && c0 < indexer_topk) ies += expf(s0 - m_idx);
            if (ok1 && c1 < indexer_topk) ies += expf(s1 - m_idx);
            if (ok2 && c2 < indexer_topk) ies += expf(s2 - m_idx);
            if (ok3 && c3 < indexer_topk) ies += expf(s3 - m_idx);
            ies   += __shfl_xor(ies, 16);
            ies   += __shfl_xor(ies, 32);
            l_idx += ies;
        }
        if (tmax > m) {
            const float corr  = expf(m - tmax);
            l                *= corr;
#    pragma unroll
            for (int dc = 0; dc < NOUT; ++dc) {
                if (dc * 16 >= D_v) continue;
                outacc[dc].x[0] *= corr;
                outacc[dc].x[1] *= corr;
                outacc[dc].x[2] *= corr;
                outacc[dc].x[3] *= corr;
            }
            m = tmax;
        }
        const float e0  = ok0 && lane_ok ? expf(s0 - m) : 0.0f;
        const float e1  = ok1 && lane_ok ? expf(s1 - m) : 0.0f;
        const float e2  = ok2 && lane_ok ? expf(s2 - m) : 0.0f;
        const float e3  = ok3 && lane_ok ? expf(s3 - m) : 0.0f;
        float es        = e0 + e1 + e2 + e3;
        es             += __shfl_xor(es, 16);
        es             += __shfl_xor(es, 32);
        l              += es;

        C* pw                       = p_lds + (size_t)warp * 256;
        pw[(size_t)r * 16 + g]      = dsa_from_float<C>(e0);
        pw[(size_t)r * 16 + g + 4]  = dsa_from_float<C>(e1);
        pw[(size_t)r * 16 + g + 8]  = dsa_from_float<C>(e2);
        pw[(size_t)r * 16 + g + 12] = dsa_from_float<C>(e3);

#    pragma unroll
        for (int pk = 0; pk < 16; pk += KFRAG) {
            DUFragment<matrix_a, 16, 16, KFRAG, C, row_major> pfr;
            du_load_matrix_sync(pfr, pw + pk, 16);
#    pragma unroll
            for (int dc = 0; dc < NOUT; ++dc) {
                if (dc * 16 >= D_v) continue;
                DUFragment<matrix_b, 16, 16, KFRAG, C, row_major> vfr;
                du_load_matrix_sync(vfr, kv_lds + (size_t)pk * KLD + dc * 16, KLD);
                du_mma_sync(outacc[dc], pfr, vfr, outacc[dc]);
            }
        }
        __syncthreads();
    }

    // Keep (m, l) KV-only.  The public lse contract excludes the sink, so
    // folding sink into l and subtracting it again here is numerically
    // invalid once the sink dominates: in fp32, 1 + tiny rounds to 1 and
    // the subtraction destroys the complete KV mass.  Merge the sink only
    // while normalizing the value numerator, as in the optimized bf16 path.
    if (lane_ok) {
        float out_scale = 0.0f;
        if (sink_val != sink_val || l != l) {
            out_scale = NAN;
        } else if (isinf(sink_val) && sink_val > 0.0f) {
            out_scale = 0.0f;
        } else if (l > 0.0f) {
            const float merged_m  = fmaxf(m, sink_val);
            const float kv_corr   = expf(m - merged_m);
            const float sink_mass = (sink_val > -INFINITY) ? expf(sink_val - merged_m) : 0.0f;
            out_scale             = kv_corr / (l * kv_corr + sink_mass + 1e-30f);
        }
#    pragma unroll
        for (int dc = 0; dc < NOUT; ++dc) {
            if (dc * 16 >= D_v) continue;
            // MMAC accumulator lane mapping: x[0..3] are columns
            // g, g+4, g+8, g+12 (not four consecutive columns).
            const int col = dc * 16 + g;
            if (col < D_v) dsa_store_from_float<C>(out, ((size_t)t * H + h) * D_v + col, outacc[dc].x[0] * out_scale);
            if (col + 4 < D_v)
                dsa_store_from_float<C>(out, ((size_t)t * H + h) * D_v + col + 4, outacc[dc].x[1] * out_scale);
            if (col + 8 < D_v)
                dsa_store_from_float<C>(out, ((size_t)t * H + h) * D_v + col + 8, outacc[dc].x[2] * out_scale);
            if (col + 12 < D_v)
                dsa_store_from_float<C>(out, ((size_t)t * H + h) * D_v + col + 12, outacc[dc].x[3] * out_scale);
        }
        if (g == 0) {
            lse[(size_t)t * H + h] = l > 0.0f ? m + logf(l) : (l != l ? NAN : -INFINITY);
            if (max_logits != nullptr) max_logits[(size_t)t * H + h] = m_kv;
            if (lse_indexer != nullptr) {
                const float mi                 = fmaxf(m_idx, sink_val);
                const float li                 = expf(m_idx - mi) * l_idx + expf(sink_val - mi);
                lse_indexer[(size_t)t * H + h] = li > 0.0f ? mi + logf(li + 1e-30f) : -INFINITY;
            }
        }
    }
}

#else

template<typename C, int ND>
static __global__ void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_fwd_dtype_kernel(const void*, const void*, const float*, const int*, const int*, void*, float*, float*,
                                 float*, int, int, int, int, int, int, int, float) {}

#endif

namespace hip_dsa_mla64 {

#if defined(__gfx936__) || defined(__gfx938__)

#else

#endif
}  // namespace hip_dsa_mla64

typedef uint16_t bf16;

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;
typedef __hip_bfloat16 hbf16;

#else  // gfx906 / gfx926 / others: no MMAC intrinsics - never launched.

#endif  // supported gfx archs

namespace ftrain {
namespace {
template<typename C>
static bool launch_sparse_attn_fwd_dtype(hipStream_t stream, const void* q, const void* kv, const float* attn_sink,
                                         const int* topk_idxs, const int* topk_length, void* out, float* lse, int T,
                                         int H, int D, int D_v, int T_kv, int topk, float softmax_scale,
                                         float* max_logits = nullptr, float* lse_indexer = nullptr,
                                         int indexer_topk = 0);
bool launch_sparse_attn_fwd_fused(hipStream_t stream, const void* q, const void* kv, const float* attn_sink,
                                  const int* topk_idxs, void* out, float* lse, const int* topk_length, int T, int H,
                                  int D, int D_v, int T_kv, int topk, float softmax_scale);

template<typename C>
static bool launch_sparse_attn_fwd_dtype(hipStream_t stream, const void* q, const void* kv, const float* attn_sink,
                                         const int* topk_idxs, const int* topk_length, void* out, float* lse, int T,
                                         int H, int D, int D_v, int T_kv, int topk, float softmax_scale,
                                         float* max_logits, float* lse_indexer, int indexer_topk) {
    constexpr int kfrag = DsaMmacTraits<C>::k;
    constexpr int pad   = sizeof(C) == 4 ? 10 : 8;
    const int kld       = D + pad;
    const size_t shm    = (size_t)16 * kld * sizeof(C) + (size_t)topk * sizeof(int) + (size_t)4 * 256 * sizeof(C);
    if (!dsa_shared_memory_fits(shm)) return false;
    dim3 grid(T, CEIL_DIV(H, SAF_HBLK));
#define SAF_DTYPE_CASE(NK)                                                                                          \
    case NK:                                                                                                        \
        hipLaunchKernelGGL((sparse_attn_fwd_dtype_kernel<C, NK>), grid, SAF_THREADS, shm, stream, q, kv, attn_sink, \
                           topk_idxs, topk_length, out, lse, max_logits, lse_indexer, indexer_topk, T, H, D, D_v,   \
                           T_kv, topk, softmax_scale);                                                              \
        break
    if constexpr (kfrag == 8) {
        switch (D / kfrag) {
            SAF_DTYPE_CASE(64);  // D=512
            SAF_DTYPE_CASE(72);  // D=576
            default: return false;
        }
    } else {
        switch (D / kfrag) {
            SAF_DTYPE_CASE(32);  // D=512
            SAF_DTYPE_CASE(36);  // D=576
            default: return false;
        }
    }
#undef SAF_DTYPE_CASE
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "mixed_fp32_sparse_attn_fwd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}

static bool launch_sparse_attn_fwd_fused(hipStream_t stream, const void* q, const void* kv, const float* attn_sink,
                                         const int* topk_idxs, void* out, float* lse, const int* topk_length, int T,
                                         int H, int D, int D_v, int T_kv, int topk, float softmax_scale) {
    if (!sparse_attn_fused_supported()) return false;
    if (D % 16 != 0 || D > SAF_MAX_D || (D_v != 512 || D < 512)) return false;
    if (T <= 0 || H <= 0 || topk <= 0) return false;

    {
        return launch_sparse_attn_fwd_dtype<float>(stream, q, kv, attn_sink, topk_idxs, topk_length, out, lse, T, H, D,
                                                   D_v, T_kv, topk, softmax_scale);
    }
}
}  // namespace

bool mixedFp32SparseAttnFwdLaunchFits(int width, int columns) {
    if (width != 512 && width != 576) return false;
    return dsa_shared_memory_fits((16ULL * (width + 10) + 1024) * 4 + static_cast<std::size_t>(columns) * 4);
}

bool mixedFp32SparseAttnFwdSparseAttnFusedSupported() { return sparse_attn_fused_supported(); }

bool launchMixedFp32SparseAttnFwd(FTrainStream stream, const void* q, const void* kv, const float* attn_sink,
                                  const int* topk_indices, void* out, float* lse, const int* topk_length, int tokens,
                                  int heads, int head_dim, int value_dim, int kv_tokens, int selection_width,
                                  float softmax_scale) {
    return launch_sparse_attn_fwd_fused(stream, q, kv, attn_sink, topk_indices, out, lse, topk_length, tokens, heads,
                                        head_dim, value_dim, kv_tokens, selection_width, softmax_scale);
}

void castMixedFp32SparseAttnFwd(const void* source, void* destination, FTrainNumericType source_type,
                                FTrainNumericType destination_type, std::uint64_t elements, FTrainStream stream) {
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
