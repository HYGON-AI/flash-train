// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// This implementation uses HIP warp synchronization intrinsics.
#define HIP_ENABLE_WARP_SYNC_BUILTINS 1

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <math.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#    include <du_mma.h>
#endif

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_dense_thd/fp16_indexer_bwd_dense_thd.hpp"

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

template<typename Native>
__device__ __forceinline__ static void dsa_store_from_float(void* data, size_t index, float value) {
    reinterpret_cast<Native*>(data)[index] = dsa_from_float<Native>(value);
}

__device__ __forceinline__ static void dsa_dense_score_grad(const float* target, float target_norm, float* logits,
                                                            float logsumexp, float* score, float* reduction,
                                                            int columns, int valid, float scale) {
    const float eps = expf(-100.0f);
    float local     = 0.0f;
    if (threadIdx.x < 128) {
        for (int i = threadIdx.x; i < valid; i += 128) {
            const float t          = target[i] / (target_norm + 1e-10f);
            const float effective  = t >= eps ? t : eps;
            const float gate       = logits[i] - logsumexp >= -100.0f ? 1.0f : 0.0f;
            local                 += -effective * gate * scale;
        }
        for (int offset = 16; offset; offset >>= 1) local += __shfl_down(local, offset, 32);
        if ((threadIdx.x & 31) == 0) reduction[threadIdx.x >> 5] = local;
    }
    __syncthreads();
    const float sum = reduction[0] + reduction[1] + reduction[2] + reduction[3];
    for (int i = threadIdx.x; i < columns; i += blockDim.x) {
        float value = 0.0f;
        if (i < valid) {
            const float log_p     = logits[i] - logsumexp;
            const float t         = target[i] / (target_norm + 1e-10f);
            const float effective = t >= eps ? t : eps;
            const float gate      = log_p >= -100.0f ? 1.0f : 0.0f;
            value                 = -effective * gate * scale - expf(log_p) * sum;
        }
        score[i]  = value;
        logits[i] = value;
    }
    __syncthreads();
}

static inline bool dense_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx928") != nullptr || strstr(prop.gcnArchName, "gfx936") != nullptr ||
           strstr(prop.gcnArchName, "gfx938") != nullptr;
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

// Adapted from fused_dsa/hip_dsa/kernels/indexer_bwd.hip.

#define IDXB_CHUNK 16  // 16-key chunk: half the K arena -> more CTAs/CU
#define IDXB_THREADS 256

#if defined(__HIP_PLATFORM_AMD__) && __has_builtin(__builtin_amdgcn_flat_atomic_fadd_f32)

#else

#endif

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;

#    if defined(__gfx936__) || defined(__gfx938__)
using idxb_floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
#    endif

__device__ __forceinline__ static void idxb_atomic_add_mmac_row(float* row16, int g, float x0, float x1, float x2,
                                                                float x3) {
#    if defined(__gfx936__) || defined(__gfx938__)
    const float p0 = __shfl_xor(x0, 16, 64);
    const float p1 = __shfl_xor(x1, 16, 64);
    const float p2 = __shfl_xor(x2, 16, 64);
    const float p3 = __shfl_xor(x3, 16, 64);
    const float y0 = (g & 1) ? p1 : x0;
    const float y1 = (g & 1) ? x1 : p0;
    const float y2 = (g & 1) ? p3 : x2;
    const float y3 = (g & 1) ? x3 : p2;
    const float q0 = __shfl_xor(y0, 32, 64);
    const float q1 = __shfl_xor(y1, 32, 64);
    const float q2 = __shfl_xor(y2, 32, 64);
    const float q3 = __shfl_xor(y3, 32, 64);
    idxb_floatx4 v;
    v[0] = (g & 2) ? q2 : y0;
    v[1] = (g & 2) ? q3 : y1;
    v[2] = (g & 2) ? y2 : q0;
    v[3] = (g & 2) ? y3 : q1;
#        if defined(__gfx938__)
    __builtin_hcu_global_atomic_fadd_f32_x4(row16 + 4 * g, v);
#        else
    __builtin_amdgcn_global_atomic_fadd_v4f32((idxb_floatx4*)(row16 + 4 * g), v);
#        endif
#    else
    float* p = row16 + g;
    atomicAdd(&p[0], x0);
    atomicAdd(&p[4], x1);
    atomicAdd(&p[8], x2);
    atomicAdd(&p[12], x3);
#    endif
}

template<bool SCORE_GLOBAL = false>
__device__ __forceinline__ static void indexer_sparse_score_grad(float* trow, const float* predict, float* score,
                                                                 float* red, int topk, float c) {
    // Match cuDNN SM90's 128-thread score_grad: index_score already
    // contains the forward probabilities. No second score/softmax pass.
    const float eps = expf(-100.f);
    if (threadIdx.x < 128) {
        float local_sum = 0.f;
        for (int pos = threadIdx.x; pos < topk; pos += 128) {
            const float target_eff  = fmaxf(trow[pos], eps);
            const float pp          = predict[pos];
            const float gate        = pp >= eps ? 1.f : 0.f;
            local_sum              += -target_eff * gate * c;
        }
        red[threadIdx.x] = local_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.f;
#    pragma unroll
        for (int i = 0; i < 128; i++) sum += red[i];
        red[0] = sum;
    }
    __syncthreads();
    const float sum_grad = red[0];
    if constexpr (SCORE_GLOBAL) {
        float values[8];
#    pragma unroll
        for (int j = 0; j < 8; j++) {
            const int pos = threadIdx.x + j * IDXB_THREADS;
            if (pos < topk) {
                const float target_eff = fmaxf(trow[pos], eps);
                const float pp         = predict[pos];
                const float gate       = pp >= eps ? 1.f : 0.f;
                const float gi         = -target_eff * gate * c;
                values[j]              = gi - pp * sum_grad;
            }
        }
        __syncthreads();
#    pragma unroll
        for (int j = 0; j < 8; j++) {
            const int pos = threadIdx.x + j * IDXB_THREADS;
            if (pos < topk) trow[pos] = values[j];
        }
        __syncthreads();
    } else {
        for (int pos = threadIdx.x; pos < topk; pos += IDXB_THREADS) {
            const float target_eff = fmaxf(trow[pos], eps);
            const float pp         = predict[pos];
            const float gate       = pp >= eps ? 1.f : 0.f;
            const float gi         = -target_eff * gate * c;
            score[pos]             = gi - pp * sum_grad;
        }
        __syncthreads();
        for (int pos = threadIdx.x; pos < topk; pos += IDXB_THREADS) trow[pos] = score[pos];
        __syncthreads();
    }
}

template<typename C, int NK>
static __global__ void __launch_bounds__(IDXB_THREADS, 1)
    indexer_bwd_sparse_dtype_kernel(const void* __restrict__ q, const void* __restrict__ w, const void* __restrict__ k,
                                    float* __restrict__ target, const float* __restrict__ target_norm,
                                    const float* __restrict__ predict, float* __restrict__ grad_out,
                                    const int* __restrict__ idxs, void* __restrict__ dq, void* __restrict__ dw,
                                    float* __restrict__ dk_float, void* __restrict__ dk, int* __restrict__ done_count,
                                    int B, int Sq, int Sk, int Hq, int D_rt, int topk, float sm_scale, float loss_coeff,
                                    float grad_loss, bool topk_indices_global, const int* __restrict__ cu_q,
                                    const int* __restrict__ cu_kv, const int* __restrict__ q_causal_offsets, int n_seqs,
                                    int rows, int ratio, int dk_rows, bool dense) {
    constexpr int KFRAG      = DsaMmacTraits<C>::k;
    constexpr int FRAG_ELEMS = sizeof(C) == 4 ? 2 : 4;
    constexpr int D          = NK * KFRAG;
    constexpr int NOUT       = D / 16;
    constexpr int VEC        = DsaMmacTraits<C>::vector_elements;
    constexpr int PAD        = sizeof(C) == 4 ? 10 : 8;
    constexpr int KLD        = D + PAD;
    (void)D_rt;

    const int row = blockIdx.x;
    if (row >= rows) return;
    const int b        = row / Sq;
    const int lane     = threadIdx.x & 63;
    const int warp     = threadIdx.x >> 6;
    const int r        = lane & 15;
    const int g        = lane >> 4;
    const int local_h  = warp * 16 + r;
    const bool lane_ok = local_h < Hq;

    int kv_offset = b * Sk;
    int col_limit = topk;
    if (dense) {
        if (cu_q != nullptr) {
            const ThdRowMeta m = thd_row_meta(cu_q, cu_kv, n_seqs, row, dk_rows);
            kv_offset          = m.kv_offset;
            col_limit          = thd_col_limit(m, ratio, q_causal_offsets);
        } else {
            col_limit = ((row % Sq) + 1) / ratio;
            if (col_limit > Sk) col_limit = Sk;
        }
        if (col_limit > topk) col_limit = topk;
    }

    extern __shared__ char smem_raw[];
    C* k_lds     = reinterpret_cast<C*>(smem_raw);  // [32][KLD]
    int* idx_lds = reinterpret_cast<int*>(k_lds + (size_t)IDXB_CHUNK * KLD);
    float* score = reinterpret_cast<float*>(idx_lds + topk);
    float* red   = score + topk;
    C* a_lds     = reinterpret_cast<C*>(red + IDXB_THREADS);  // [4][16][32]
    C* at_lds    = a_lds + 4 * 16 * IDXB_CHUNK;               // [32][64]
    C* q_lds     = at_lds + IDXB_CHUNK * 64;                  // [4][16][16]

    for (int i = threadIdx.x; i < topk; i += IDXB_THREADS) {
        idx_lds[i] = dense ? (i < col_limit ? i : -1) : idxs[(size_t)row * topk + i];
        score[i]   = 0.0f;
    }
    __syncthreads();

    const float w_h   = lane_ok ? dsa_load_compute_float<C>(w, (size_t)row * Hq + local_h) : 0.0f;
    const int nchunks = CEIL_DIV(topk, IDXB_CHUNK);

    if (!dense) {
        indexer_sparse_score_grad(target + (size_t)row * topk, predict + (size_t)row * topk, score, red, topk,
                                  (loss_coeff / (float)rows) * grad_loss);
    } else {
        // In dense mode predict points to the supplied row LSE; grad_out
        // holds the supplied raw logits and is overwritten with grad_signal.
        dsa_dense_score_grad(target + (size_t)row * topk, target_norm[row], grad_out + (size_t)row * topk, predict[row],
                             score, red, topk, col_limit, (loss_coeff / (float)rows) * grad_loss);
    }

    DUFragment<accumulator, 16, 16, KFRAG, float> dqacc[NOUT];
#    pragma unroll
    for (int dc = 0; dc < NOUT; ++dc) du_fill_fragment(dqacc[dc], 0.0f);
    float dwacc = 0.0f;

    for (int ch = 0; ch < nchunks; ++ch) {
        int ck = topk - ch * IDXB_CHUNK;
        if (ck > IDXB_CHUNK) ck = IDXB_CHUNK;
        for (int i = threadIdx.x; i < IDXB_CHUNK * KLD; i += IDXB_THREADS) k_lds[i] = dsa_from_float<C>(0.0f);
        __syncthreads();
        const int vecs = CEIL_DIV(D, VEC);
        for (int u = threadIdx.x; u < ck * vecs; u += IDXB_THREADS) {
            const int kr = u / vecs, dv = (u % vecs) * VEC;
            int idx = idx_lds[ch * IDXB_CHUNK + kr];
            if (idx < 0 || idx >= (topk_indices_global && !dense ? B * Sk : Sk)) continue;
            const size_t krow = dense ? (size_t)kv_offset + idx : (size_t)(topk_indices_global ? 0 : b) * Sk + idx;
            const size_t src  = krow * D + dv;
            if (dv + VEC <= D)
                dsa_copy_as<C, VEC>(k_lds + (size_t)kr * KLD + dv, k, src);
            else
                for (int e = 0; e < VEC && dv + e < D; ++e)
                    k_lds[(size_t)kr * KLD + dv + e] = dsa_load_as<C>(k, src + e);
        }
        __syncthreads();

        const int ntiles = CEIL_DIV(ck, 16);
        DUFragment<accumulator, 16, 16, KFRAG, float> dot[2];
#    pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            if (nt < ntiles) du_fill_fragment(dot[nt], 0.0f);
#    pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
            DUFragment<matrix_a, 16, 16, KFRAG, C, row_major> qf;
#    pragma unroll
            for (int e = 0; e < FRAG_ELEMS; ++e) {
                const size_t at =
                    ((size_t)row * Hq + (lane_ok ? local_h : Hq - 1)) * D + kk * KFRAG + g * FRAG_ELEMS + e;
                qf.x[e] = lane_ok ? dsa_load_as<C>(q, at) : dsa_from_float<C>(0.0f);
            }
#    pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                if (nt < ntiles) {
                    DUFragment<matrix_b, 16, 16, KFRAG, C, col_major> kf;
                    du_load_matrix_sync(kf, k_lds + (size_t)(nt * 16) * KLD + kk * KFRAG, KLD);
                    du_mma_sync(dot[nt], qf, kf, dot[nt]);
                }
        }

        C* aw = a_lds + (size_t)warp * 16 * IDXB_CHUNK;
#    pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            if (nt < ntiles) {
                // The last tile may contain fewer than 16 logical scores.
                const float ds0                          = nt * 16 + g < ck ? score[ch * IDXB_CHUNK + nt * 16 + g] : 0.0f;
                const float ds1                          = nt * 16 + g + 4 < ck ? score[ch * IDXB_CHUNK + nt * 16 + g + 4] : 0.0f;
                const float ds2                          = nt * 16 + g + 8 < ck ? score[ch * IDXB_CHUNK + nt * 16 + g + 8] : 0.0f;
                const float ds3                          = nt * 16 + g + 12 < ck ? score[ch * IDXB_CHUNK + nt * 16 + g + 12] : 0.0f;
                const float rv0                          = fmaxf(dot[nt].x[0] * sm_scale, 0.0f);
                const float rv1                          = fmaxf(dot[nt].x[1] * sm_scale, 0.0f);
                const float rv2                          = fmaxf(dot[nt].x[2] * sm_scale, 0.0f);
                const float rv3                          = fmaxf(dot[nt].x[3] * sm_scale, 0.0f);
                const float a0                           = dot[nt].x[0] * sm_scale > 0.0f ? ds0 * sm_scale * w_h : 0.0f;
                const float a1                           = dot[nt].x[1] * sm_scale > 0.0f ? ds1 * sm_scale * w_h : 0.0f;
                const float a2                           = dot[nt].x[2] * sm_scale > 0.0f ? ds2 * sm_scale * w_h : 0.0f;
                const float a3                           = dot[nt].x[3] * sm_scale > 0.0f ? ds3 * sm_scale * w_h : 0.0f;
                aw[(size_t)r * IDXB_CHUNK + nt * 16 + g] = dsa_from_float<C>(a0);
                aw[(size_t)r * IDXB_CHUNK + nt * 16 + g + 4]       = dsa_from_float<C>(a1);
                aw[(size_t)r * IDXB_CHUNK + nt * 16 + g + 8]       = dsa_from_float<C>(a2);
                aw[(size_t)r * IDXB_CHUNK + nt * 16 + g + 12]      = dsa_from_float<C>(a3);
                at_lds[(size_t)(nt * 16 + g) * 64 + local_h]       = dsa_from_float<C>(a0);
                at_lds[(size_t)(nt * 16 + g + 4) * 64 + local_h]   = dsa_from_float<C>(a1);
                at_lds[(size_t)(nt * 16 + g + 8) * 64 + local_h]   = dsa_from_float<C>(a2);
                at_lds[(size_t)(nt * 16 + g + 12) * 64 + local_h]  = dsa_from_float<C>(a3);
                dwacc                                             += ds0 * rv0 + ds1 * rv1 + ds2 * rv2 + ds3 * rv3;
            }
        __syncthreads();

#    pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            if (nt < ntiles) {
#    pragma unroll
                for (int pk = 0; pk < 16; pk += KFRAG) {
                    DUFragment<matrix_a, 16, 16, KFRAG, C, row_major> af;
                    du_load_matrix_sync(af, aw + nt * 16 + pk, IDXB_CHUNK);
#    pragma unroll
                    for (int dc = 0; dc < NOUT; ++dc) {
                        DUFragment<matrix_b, 16, 16, KFRAG, C, row_major> kf;
                        du_load_matrix_sync(kf, k_lds + (size_t)(nt * 16 + pk) * KLD + dc * 16, KLD);
                        du_mma_sync(dqacc[dc], af, kf, dqacc[dc]);
                    }
                }
            }

        const int rounds = CEIL_DIV(NOUT, 4);
        for (int rd = 0; rd < rounds; ++rd) {
            const int dc = rd * 4 + warp;
            DUFragment<accumulator, 16, 16, KFRAG, float> dkacc[2];
            du_fill_fragment(dkacc[0], 0.0f);
            du_fill_fragment(dkacc[1], 0.0f);
            for (int hg = 0; hg < 64; hg += KFRAG) {
                C* qt = q_lds + (size_t)warp * 256;
                for (int u = lane; u < KFRAG * 16; u += 64) {
                    const int hr = u / 16, col = u % 16;
                    const int gh = hg + hr;
                    if (dc < NOUT && gh < Hq)
                        qt[(size_t)hr * 16 + col] = dsa_load_as<C>(q, ((size_t)row * Hq + gh) * D + dc * 16 + col);
                    else
                        qt[(size_t)hr * 16 + col] = dsa_from_float<C>(0.0f);
                }
                __syncwarp();
                if (dc < NOUT) {
                    DUFragment<matrix_b, 16, 16, KFRAG, C, row_major> qf;
                    du_load_matrix_sync(qf, qt, 16);
#    pragma unroll
                    for (int nt = 0; nt < 2; ++nt)
                        if (nt < ntiles) {
                            DUFragment<matrix_a, 16, 16, KFRAG, C, row_major> atf;
                            du_load_matrix_sync(atf, at_lds + (size_t)(nt * 16) * 64 + hg, 64);
                            du_mma_sync(dkacc[nt], atf, qf, dkacc[nt]);
                        }
                }
                __syncwarp();
            }
            if (dc < NOUT) {
#    pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    if (nt < ntiles && r < ck - nt * 16) {
                        const int idx = idx_lds[ch * IDXB_CHUNK + nt * 16 + r];
                        if (idx >= 0 && idx < (topk_indices_global && !dense ? B * Sk : Sk)) {
                            const size_t krow =
                                dense ? (size_t)kv_offset + idx : (size_t)(topk_indices_global ? 0 : b) * Sk + idx;
                            float* dst = dk_float + krow * D + dc * 16;
                            idxb_atomic_add_mmac_row(dst, g, dkacc[nt].x[0], dkacc[nt].x[1], dkacc[nt].x[2],
                                                     dkacc[nt].x[3]);
                        }
                    }
            }
        }
        __syncthreads();
    }

    if (lane_ok) {
#    pragma unroll
        for (int dc = 0; dc < NOUT; ++dc) {
            const int col     = dc * 16 + g;
            const size_t base = ((size_t)row * Hq + local_h) * D + col;
            dsa_store_from_float<C>(dq, base, dqacc[dc].x[0]);
            dsa_store_from_float<C>(dq, base + 4, dqacc[dc].x[1]);
            dsa_store_from_float<C>(dq, base + 8, dqacc[dc].x[2]);
            dsa_store_from_float<C>(dq, base + 12, dqacc[dc].x[3]);
        }
        dwacc += __shfl_xor(dwacc, 16);
        dwacc += __shfl_xor(dwacc, 32);
        if (g == 0) dsa_store_from_float<C>(dw, (size_t)row * Hq + local_h, dwacc);
    }

    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (threadIdx.x == 0) last = atomicAdd(done_count, 1) == gridDim.x - 1;
    __syncthreads();
    if (last) {
        const size_t n = (size_t)dk_rows * D;
        for (size_t i = threadIdx.x; i < n; i += IDXB_THREADS) dsa_store_from_float<C>(dk, i, dk_float[i]);
    }
}

#else  // no MMAC intrinsics on this arch

template<typename C, int NK>
static __global__ void __launch_bounds__(IDXB_THREADS, 1)
    indexer_bwd_sparse_dtype_kernel(const void*, const void*, const void*, float*, const float*, const float*, float*,
                                    const int*, void*, void*, float*, void*, int*, int, int, int, int, int, int, float,
                                    float, float, bool, const int*, const int*, const int*, int, int, int, int, bool) {}

#endif

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

#else

#endif

namespace ftrain {
namespace {

static bool launch_indexer_bwd_dense_thd(hipStream_t stream, const void* index_q, const void* weights,
                                         const void* index_k, const float* attn_score, const float* attn_l1norm,
                                         float* index_score, const float* index_lse, void* dq, void* dw, void* dk,
                                         float* dk_float, int* done_count, const int* cu_seqlens_q,
                                         const int* cu_seqlens_kv, const int* q_causal_offsets, int total_q,
                                         int total_k, int max_sk, int H, int D, int n_seqs, int ratio, float sm_scale,
                                         float loss_coeff, float grad_loss) {
    if (D <= 0 || D % 16 != 0 || D > 128 || H < 1 || H > 64 || !dense_mmac_supported()) return false;

    {
        const int kfrag  = 16;
        const int pad    = kfrag == 8 ? 10 : 8;
        const size_t shm = (size_t)(IDXB_CHUNK * (D + pad) + 5 * 1024) * (kfrag == 8 ? 4 : 2) +
                           (size_t)max_sk * (sizeof(int) + sizeof(float)) + (size_t)IDXB_THREADS * sizeof(float);
        if (!dsa_shared_memory_fits(shm)) return false;
#define IDXB_DENSE_THD_LAUNCH(TYPE, NK)                                                                             \
    hipLaunchKernelGGL((indexer_bwd_sparse_dtype_kernel<TYPE, NK>), total_q, IDXB_THREADS, shm, stream, index_q,    \
                       weights, index_k, (float*)attn_score, attn_l1norm, index_lse, index_score, nullptr, dq, dw,  \
                       dk_float, dk, done_count, total_q, 1, max_sk, H, D, max_sk, sm_scale, loss_coeff, grad_loss, \
                       false, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, n_seqs, total_q, ratio, total_k, true)
#define IDXB_DENSE_THD_SWITCH(TYPE)                      \
    switch (D / kfrag) {                                 \
        case 1: IDXB_DENSE_THD_LAUNCH(TYPE, 1); break;   \
        case 2: IDXB_DENSE_THD_LAUNCH(TYPE, 2); break;   \
        case 3: IDXB_DENSE_THD_LAUNCH(TYPE, 3); break;   \
        case 4: IDXB_DENSE_THD_LAUNCH(TYPE, 4); break;   \
        case 5: IDXB_DENSE_THD_LAUNCH(TYPE, 5); break;   \
        case 6: IDXB_DENSE_THD_LAUNCH(TYPE, 6); break;   \
        case 7: IDXB_DENSE_THD_LAUNCH(TYPE, 7); break;   \
        case 8: IDXB_DENSE_THD_LAUNCH(TYPE, 8); break;   \
        case 10: IDXB_DENSE_THD_LAUNCH(TYPE, 10); break; \
        case 12: IDXB_DENSE_THD_LAUNCH(TYPE, 12); break; \
        case 14: IDXB_DENSE_THD_LAUNCH(TYPE, 14); break; \
        case 16: IDXB_DENSE_THD_LAUNCH(TYPE, 16); break; \
        default: return false;                           \
    }
        {
            { IDXB_DENSE_THD_SWITCH(__half); }
        }
#undef IDXB_DENSE_THD_SWITCH
#undef IDXB_DENSE_THD_LAUNCH
        // hipLaunchKernelGGL is a void launch macro; check the selected case here
        // before returning success. Execution stays asynchronous on this stream.
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "fp16_indexer_bwd_dense_thd HIP call failed: %s",
                                        hipGetErrorString(status));
            }
        }
        return true;
    }
}
}  // namespace

bool fp16IndexerBwdDenseThdLaunchFits(int heads, int width, int columns, int queries) {
    (void)heads;
    (void)queries;
    const std::size_t bytes = (16ULL * (width + 8) + 5 * 1024) * 2 + static_cast<std::size_t>(columns) * 8 + 256 * 4;
    return dsa_shared_memory_fits(bytes);
}

bool launchFp16IndexerBwdDenseThd(FTrainStream stream, const void* index_q, const void* weights, const void* index_k,
                                  const float* attn_score, const float* attn_l1norm, float* index_score,
                                  const float* index_lse, void* dq, void* dw, void* dk, float* dk_float,
                                  int* done_count, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                  const int* q_causal_offsets, int total_q, int total_k, int max_seqlen_kv, int heads,
                                  int head_dim, int num_sequences, int ratio, float sm_scale, float loss_coeff,
                                  float grad_loss) {
    return launch_indexer_bwd_dense_thd(stream, index_q, weights, index_k, attn_score, attn_l1norm, index_score,
                                        index_lse, dq, dw, dk, dk_float, done_count, cu_seqlens_q, cu_seqlens_k,
                                        q_causal_offsets, total_q, total_k, max_seqlen_kv, heads, head_dim,
                                        num_sequences, ratio, sm_scale, loss_coeff, grad_loss);
}

bool fp16IndexerBwdDenseThdDenseMmacSupported() { return dense_mmac_supported(); }

void fp16IndexerBwdDenseThdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    const hipError_t status = hipMemsetAsync(memory, 0, static_cast<std::size_t>(bytes), stream);
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Workspace zero-fill failed: %s", hipGetErrorString(status));
    }
}

}  // namespace ftrain
