// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <math.h>
#include <stdexcept>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#    include <du_mma.h>
#endif

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/dense_attn_score/fp16_dense_attn_score.hpp"

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

#define DENSE_CHUNK 32
#define DENSE_THREADS 256
#define DENSE_DCHUNK 128
#define DENSE_MAX_TILES 36  // D <= 576 (DSv4 absorbed MLA)

static inline bool dense_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx928") != nullptr || strstr(prop.gcnArchName, "gfx936") != nullptr ||
           strstr(prop.gcnArchName, "gfx938") != nullptr;
}

static size_t dense_mmac_shm(int Hq, int D, int Sk, bool bwd) {
    int Hq_pad  = (Hq + 15) & ~15;
    int dd      = bwd ? D : (D < DENSE_DCHUNK ? D : DENSE_DCHUNK);
    size_t sz   = (size_t)DENSE_CHUNK * (dd + 8) * 2;  // k_lds
    sz         += (size_t)Sk * 4;                      // score
    if (bwd) {
        // K, score-gradient, reduction and BF16 dS tiles; Q stays in registers.
        sz += DENSE_THREADS * 4;                       // red
        sz += (size_t)Hq_pad * (DENSE_CHUNK + 2) * 2;  // a_lds (ALDS = 34)
    } else {
        sz += (size_t)Hq_pad * (dd + 8) * 2;  // q_lds (forward kernels)
        sz += DENSE_THREADS * 4;              // red (denom)
    }
    return sz;
}

static inline bool dense_mmac_shm_fits(int Hq, int D, int Sk, bool bwd) {
    const int max_lds = dsa_shared_memory_limit();
    if (!bwd) return dense_mmac_shm(Hq, D, Sk, false) <= (size_t)max_lds;
    int Hq_pad = (Hq + 15) & ~15;
    return dense_mmac_shm(Hq, D, Sk, true) <= (size_t)max_lds;
}

template<typename C>
static inline size_t dense_mmac_dtype_shm(int score_stride) {
    constexpr int dchunk = DsaMmacTraits<C>::dense_dchunk;
    constexpr int pad    = sizeof(C) == 4 ? 10 : 2;
    constexpr int ld     = dchunk + pad;
    return (size_t)(64 + DENSE_CHUNK) * ld * sizeof(C) + (size_t)score_stride * sizeof(float) +
           (size_t)DENSE_THREADS * sizeof(float);
}

static inline bool dense_mmac_storage_fits(int heads, int width, int keys, bool packed) {
    if (heads <= 0 || heads > 128 || width <= 0 || width > 576 || width % 16 || keys <= 0) return false;
    if (!dense_mmac_shm_fits(heads, width, keys, false)) return false;

    { return dsa_shared_memory_fits(dense_mmac_dtype_shm<__half>(keys)); }
}

struct ThdRowMeta {
    int batch_id;    // segment (batch) index owning this row
    int pos_in_seq;  // row position inside its segment
    int kv_offset;   // first packed KV row of the segment
    int seqlen_kv;   // number of packed KV rows of the segment
};

__device__ __forceinline__ static int thd_batch_of_row(const int* __restrict__ cu_seqlens_q, int n_seqs, int row) {
    int lo = 0, hi = n_seqs;  // invariant: cu[lo] <= row < cu[hi]
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (cu_seqlens_q[mid] <= row)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

__device__ __forceinline__ static ThdRowMeta thd_row_meta(const int* __restrict__ cu_seqlens_q,
                                                          const int* __restrict__ cu_seqlens_kv, int n_seqs, int row,
                                                          int total_k) {
    ThdRowMeta m;
    m.batch_id   = thd_batch_of_row(cu_seqlens_q, n_seqs, row);
    const int qs = cu_seqlens_q[m.batch_id], qe = cu_seqlens_q[m.batch_id + 1];
    const int ks = cu_seqlens_kv[m.batch_id], ke = cu_seqlens_kv[m.batch_id + 1];
    const bool valid = qs >= 0 && qs <= row && row < qe && ks >= 0 && ks <= ke && ke <= total_k;
    m.pos_in_seq     = valid ? row - qs : 0;
    m.kv_offset      = valid ? ks : 0;
    m.seqlen_kv      = valid ? ke - ks : 0;
    return m;
}

__device__ __forceinline__ static int thd_col_limit(const ThdRowMeta& m, int ratio) {
    int lim = (int)(((int64_t)m.pos_in_seq + 1) / ratio);
    if (lim > m.seqlen_kv) lim = m.seqlen_kv;
    return lim;
}

__device__ __forceinline__ static int thd_col_limit(const ThdRowMeta& m, int ratio,
                                                    const int* __restrict__ q_causal_offsets) {
    const int offset = q_causal_offsets == nullptr ? 0 : q_causal_offsets[m.batch_id];
    int64_t lim      = ((int64_t)offset + m.pos_in_seq + 1) / ratio;
    if (lim < 0) lim = 0;
    if (lim > m.seqlen_kv) lim = m.seqlen_kv;
    return lim;
}

namespace {
using KernelStorage = __half;
}  // namespace

// Adapted from fused_dsa/hip_dsa/kernels/dense_mmac.hip.

#define DENSE_CHUNK 32
#define DENSE_THREADS 256

#define DENSE_DCHUNK 128
#define DENSE_MAX_TILES 36  // D <= 576 (DSv4 absorbed MLA)

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;

template<int MODE>
__device__ __forceinline__ static float dm_epi(float dot, float head_a, float scale) {
    if (MODE == 2) return expf(dot * scale - head_a);
    float d = dot * scale;
    // Match torch.relu's NaN propagation (relu(NaN) == NaN) so degenerate
    // NaN inputs behave identically to the eager reference instead of
    // collapsing NaN to 0.
    return (d > 0.0f ? d : (d != d ? d : 0.0f)) * head_a;
}

template<int MODE, bool THD>
static __global__ void __launch_bounds__(DENSE_THREADS)
    fp16_dense_attn_score_kernel(const void* __restrict__ q, const void* __restrict__ k, const void* __restrict__ w,
                                 const float* __restrict__ lse, float* __restrict__ out, float* __restrict__ denom,
                                 const int* __restrict__ cu_q, const int* __restrict__ cu_kv,
                                 const int* __restrict__ q_causal_offsets, int rows, int Sq, int Sk, int Hq, int D,
                                 int n_seqs, int ratio, float scale, int score_stride) {
    constexpr int KFRAG  = DsaMmacTraits<KernelStorage>::k;
    constexpr int DCHUNK = DsaMmacTraits<KernelStorage>::dense_dchunk;
    constexpr int VEC    = DsaMmacTraits<KernelStorage>::vector_elements;
    constexpr int PAD    = sizeof(KernelStorage) == 4 ? 10 : 2;
    constexpr int LD     = DCHUNK + PAD;

    const int row     = blockIdx.x;
    const int lane    = threadIdx.x & 63;
    const int warp    = threadIdx.x >> 6;
    const int local_h = warp * 16 + (lane & 15);

    int kv_offset;
    int col_limit;
    if constexpr (THD) {
        // In this shared BSHD/THD kernel, Sq carries actual packed K storage
        // length for THD; Sk remains the padded output score stride.
        const ThdRowMeta meta = thd_row_meta(cu_q, cu_kv, n_seqs, row, Sq);
        kv_offset             = meta.kv_offset;
        col_limit             = thd_col_limit(meta, ratio, q_causal_offsets);
        if (col_limit > Sk) col_limit = Sk;
    } else {
        const int b  = row / Sq;
        const int sq = row - b * Sq;
        kv_offset    = b * Sk;
        col_limit    = (sq + 1) / ratio;
        if (col_limit > Sk) col_limit = Sk;
    }

    extern __shared__ char smem_raw[];
    KernelStorage* q_lds = reinterpret_cast<KernelStorage*>(smem_raw);  // [64][LD]
    KernelStorage* k_lds = q_lds + (size_t)64 * LD;                     // [32][LD]
    float* score         = reinterpret_cast<float*>(k_lds + (size_t)DENSE_CHUNK * LD);
    float* red           = score + score_stride;

    for (int i = threadIdx.x; i < score_stride; i += DENSE_THREADS) score[i] = 0.0f;
    __syncthreads();

    const int n_hpass = (Hq + 63) / 64;
    for (int hp = 0; hp < n_hpass; ++hp) {
        const int head = hp * 64 + local_h;
        float head_a   = 0.0f;
        if constexpr (MODE == 2) {
            head_a = head < Hq ? lse[(size_t)row * Hq + head] : INFINITY;
        } else {
            head_a = head < Hq ? dsa_load_compute_float<KernelStorage>(w, (size_t)row * Hq + head) : 0.0f;
        }

        for (int chunk = 0; chunk < col_limit; chunk += DENSE_CHUNK) {
            int ck = col_limit - chunk;
            if (ck > DENSE_CHUNK) ck = DENSE_CHUNK;
            const int ntiles = (ck + 15) >> 4;
            DUFragment<accumulator, 16, 16, KFRAG, float> acc[2];
#    pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                if (nt < ntiles) du_fill_fragment(acc[nt], 0.0f);

            for (int d0 = 0; d0 < D; d0 += DCHUNK) {
                int dc = D - d0;
                if (dc > DCHUNK) dc = DCHUNK;

                // Zero padding and inactive head/key rows before staging.
                for (int i = threadIdx.x; i < (64 + DENSE_CHUNK) * LD; i += DENSE_THREADS) {
                    if (i < 64 * LD)
                        q_lds[i] = dsa_from_float<KernelStorage>(0.0f);
                    else
                        k_lds[i - 64 * LD] = dsa_from_float<KernelStorage>(0.0f);
                }
                __syncthreads();

                const int q_vecs = 64 * CEIL_DIV(dc, VEC);
                for (int u = threadIdx.x; u < q_vecs; u += DENSE_THREADS) {
                    const int lh = u / CEIL_DIV(dc, VEC);
                    const int dv = (u % CEIL_DIV(dc, VEC)) * VEC;
                    const int gh = hp * 64 + lh;
                    if (gh < Hq && dv + VEC <= dc) {
                        dsa_copy_as<KernelStorage, VEC>(q_lds + (size_t)lh * LD + dv, q,
                                                        ((size_t)row * Hq + gh) * D + d0 + dv);
                    } else if (gh < Hq && dv < dc) {
                        for (int e = 0; e < VEC && dv + e < dc; ++e)
                            q_lds[(size_t)lh * LD + dv + e] =
                                dsa_load_as<KernelStorage>(q, ((size_t)row * Hq + gh) * D + d0 + dv + e);
                    }
                }

                const int k_vecs_per_row = CEIL_DIV(dc, VEC);
                for (int u = threadIdx.x; u < ck * k_vecs_per_row; u += DENSE_THREADS) {
                    const int kr     = u / k_vecs_per_row;
                    const int dv     = (u % k_vecs_per_row) * VEC;
                    const size_t src = ((size_t)kv_offset + chunk + kr) * D + d0 + dv;
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
                    if (local_h - (lane & 15) < 64) du_load_matrix_sync(afr, q_lds + (size_t)(warp * 16) * LD + kk, LD);
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
            for (int nt = 0; nt < 2; ++nt) {
                if (nt < ntiles) {
                    float v0 = dm_epi<MODE>(acc[nt].x[0], head_a, scale);
                    float v1 = dm_epi<MODE>(acc[nt].x[1], head_a, scale);
                    float v2 = dm_epi<MODE>(acc[nt].x[2], head_a, scale);
                    float v3 = dm_epi<MODE>(acc[nt].x[3], head_a, scale);
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
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < score_stride; i += DENSE_THREADS)
        out[(size_t)row * score_stride + i] = i < col_limit ? score[i] : -INFINITY;
    if constexpr (MODE == 0) return;

    if constexpr (MODE == 1) {
        float mx = -INFINITY;
        for (int i = threadIdx.x; i < col_limit; i += DENSE_THREADS) mx = fmaxf(mx, score[i]);
        red[threadIdx.x] = mx;
        __syncthreads();
        for (int st = DENSE_THREADS / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
            __syncthreads();
        }
        mx = red[0];
        // Current launchers do not select MODE=1; keep its scratch reuse safe as well.
        __syncthreads();
        float sum = 0.0f;
        for (int i = threadIdx.x; i < col_limit; i += DENSE_THREADS) sum += expf(score[i] - mx);
        red[threadIdx.x] = sum;
        __syncthreads();
        for (int st = DENSE_THREADS / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
            __syncthreads();
        }
        if (threadIdx.x == 0) denom[row] = col_limit == 0 ? -INFINITY : mx + logf(red[0]);
    } else {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < col_limit; i += DENSE_THREADS) sum += score[i];
        red[threadIdx.x] = sum;
        __syncthreads();
        for (int st = DENSE_THREADS / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
            __syncthreads();
        }
        if (threadIdx.x == 0) denom[row] = red[0];
    }
}

#else  // gfx906 / gfx926 / others: no MMAC intrinsics — never launched at runtime.

template<int MODE, bool THD>
static __global__ void __launch_bounds__(DENSE_THREADS)
    fp16_dense_attn_score_kernel(const void*, const void*, const void*, const float*, float*, float*, const int*,
                                 const int*, const int*, int, int, int, int, int, int, int, float, int) {}

#endif  // supported gfx archs

namespace ftrain {
namespace {
template<int MODE, bool THD>
static inline void launch_dense_score_mmac_dtype(hipStream_t stream, const void* q, const void* k, const void* w,
                                                 const float* lse, float* out, float* denom, const int* cu_q,
                                                 const int* cu_kv, const int* q_causal_offsets, int rows, int Sq,
                                                 int Sk, int Hq, int D, int n_seqs, int ratio, float scale,
                                                 int score_stride = 0);

template<int MODE>
static inline void launch_dense_score_mmac_dispatch(hipStream_t stream, const void* q, const void* k, const void* w,
                                                    const float* lse, float* out, float* denom, int B, int Sq, int Sk,
                                                    int Hq, int D, int ratio, float scale, int score_stride = 0);
static void launch_dense_attn_score_mmac(hipStream_t stream, const void* q, const void* k, const float* lse, float* out,
                                         float* denom, int B, int Sq, int Sk, int Hq, int D, int ratio, float sm_scale,
                                         int score_stride = 0);

template<int MODE, bool THD>
static inline void launch_dense_score_mmac_dtype(hipStream_t stream, const void* q, const void* k, const void* w,
                                                 const float* lse, float* out, float* denom, const int* cu_q,
                                                 const int* cu_kv, const int* q_causal_offsets, int rows, int Sq,
                                                 int Sk, int Hq, int D, int n_seqs, int ratio, float scale,
                                                 int score_stride) {
    if (!dsa_shared_memory_fits(dense_mmac_dtype_shm<KernelStorage>(score_stride)))
        throw std::runtime_error("dense score: shared memory exceeds the device limit for this dtype");
    hipLaunchKernelGGL((fp16_dense_attn_score_kernel<MODE, THD>), rows, DENSE_THREADS,
                       dense_mmac_dtype_shm<KernelStorage>(score_stride), stream, q, k, w, lse, out, denom, cu_q, cu_kv,
                       q_causal_offsets, rows, Sq, Sk, Hq, D, n_seqs, ratio, scale, score_stride);
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "fp16_dense_attn_score HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
}

template<int MODE>
static inline void launch_dense_score_mmac_dispatch(hipStream_t stream, const void* q, const void* k, const void* w,
                                                    const float* lse, float* out, float* denom, int B, int Sq, int Sk,
                                                    int Hq, int D, int ratio, float scale, int score_stride) {
    if (score_stride <= 0) score_stride = Sk;
    launch_dense_score_mmac_dtype<MODE, false>(stream, q, k, w, lse, out, denom, nullptr, nullptr, nullptr, B * Sq, Sq,
                                               Sk, Hq, D, 0, ratio, scale, score_stride);
}

static void launch_dense_attn_score_mmac(hipStream_t stream, const void* q, const void* k, const float* lse, float* out,
                                         float* denom, int B, int Sq, int Sk, int Hq, int D, int ratio, float sm_scale,
                                         int score_stride) {
    launch_dense_score_mmac_dispatch<2>(stream, q, k, nullptr, lse, out, denom, B, Sq, Sk, Hq, D, ratio, sm_scale,
                                        score_stride);
}

static bool launch_dense_attn_score(hipStream_t stream, const void* q, const void* k, const float* lse, float* out,
                                    float* denom, int B, int Sq, int Sk, int Hq, int Hkv, int D, int ratio,
                                    float softmax_scale) {
    // Fused single-kernel path (MMAC tensor core) only; no non-fused
    // fallback and no hipblas GEMM anywhere in this operator.
    if (!(Hkv == 1 && D % 16 == 0 && D <= 576 && Hq <= 128 && dense_mmac_supported() &&
          dense_mmac_shm_fits(Hq, D, Sk, false))) {
        return false;
    }
    launch_dense_attn_score_mmac(stream, q, k, lse, out, denom, B, Sq, Sk, Hq, D, ratio, softmax_scale);
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "fp16_dense_attn_score HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}
}  // namespace

bool fp16DenseAttnScoreLaunchFits(int heads, int width, int keys) {
    return dense_mmac_storage_fits(heads, width, keys, false);
}

bool fp16DenseAttnScoreDenseMmacSupported() { return dense_mmac_supported(); }

bool launchFp16DenseAttnScore(FTrainStream stream, const void* q, const void* k, const float* lse, float* out,
                              float* denom, int batch, int query_length, int key_length, int query_heads, int kv_heads,
                              int head_dim, int ratio, float softmax_scale) {
    return launch_dense_attn_score(stream, q, k, lse, out, denom, batch, query_length, key_length, query_heads,
                                   kv_heads, head_dim, ratio, softmax_scale);
}

}  // namespace ftrain
