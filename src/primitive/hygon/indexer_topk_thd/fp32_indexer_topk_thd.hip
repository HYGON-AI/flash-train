// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <math.h>

#include <hip/hip_runtime.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/indexer_topk_thd/fp32_indexer_topk_thd.hpp"

constexpr int DSA_TOPK_MAX_COLS = 16384;
constexpr int DSA_TOPK_MAX_K    = 4096;

// Adapted from fused_dsa/hip_dsa/kernels/indexer_topk.hip.

template<int NTHREADS, int VPT>
static __global__ void indexer_topk_bitonic_thd_kernel(const float* __restrict__ input_values,
                                                       const int* __restrict__ cu_seqlens_q,
                                                       const int* __restrict__ cu_seqlens_kv,
                                                       const int* __restrict__ q_causal_offsets,
                                                       int* __restrict__ out_indices, float* __restrict__ out_values,
                                                       int n_rows, int n_cols, int top_k, int n_seqs, int ratio) {
    constexpr int N2 = NTHREADS * VPT;  // padded sort length (power of two)
    const int row    = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;

    int effective_len = 0;
    if (row < cu_seqlens_q[n_seqs]) {
        int lo = 0, hi = n_seqs;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) >> 1;
            if (cu_seqlens_q[mid] <= row)
                lo = mid;
            else
                hi = mid;
        }
        const int b      = lo;
        const int pos    = row - cu_seqlens_q[b] + (q_causal_offsets == nullptr ? 0 : q_causal_offsets[b]);
        const int kv_len = cu_seqlens_kv[b + 1] - cu_seqlens_kv[b];
        effective_len    = (pos + 1) / ratio;
        if (effective_len > kv_len) effective_len = kv_len;
        if (effective_len > n_cols) effective_len = n_cols;
    }
    const int k_eff = (top_k < effective_len) ? top_k : effective_len;

    float val[VPT];
    int idx[VPT];
    const float* row_values = input_values + (size_t)row * n_cols;
#pragma unroll
    for (int v = 0; v < VPT; v++) {
        int col = v * NTHREADS + tid;
        if (col < effective_len) {
            val[v] = row_values[col];
            idx[v] = col;
        } else {
            val[v] = -INFINITY;
            idx[v] = -1;
        }
    }

    extern __shared__ char smem_raw[];
    float* s_val = (float*)smem_raw;
    int* s_idx   = (int*)(smem_raw + N2 * sizeof(float));

    bool in_lds = false;
    for (int k = 2; k <= N2; k <<= 1) {
        for (int j = k >> 1; j >= 1; j >>= 1) {
            if (j >= NTHREADS) {
                if (in_lds) {
#pragma unroll
                    for (int v = 0; v < VPT; v++) {
                        val[v] = s_val[v * NTHREADS + tid];
                        idx[v] = s_idx[v * NTHREADS + tid];
                    }
                    __syncthreads();
                    in_lds = false;
                }
                const int vshift = j / NTHREADS;
#pragma unroll
                for (int v = 0; v < VPT; v++) {
                    int vp = v ^ vshift;
                    if (vp > v) {
                        const bool desc = ((v * NTHREADS + tid) & k) == 0;
                        const bool swap = desc ? (val[v] < val[vp]) : (val[v] > val[vp]);
                        if (swap) {
                            float tv = val[v];
                            val[v]   = val[vp];
                            val[vp]  = tv;
                            int ti   = idx[v];
                            idx[v]   = idx[vp];
                            idx[vp]  = ti;
                        }
                    }
                }
            } else {
                if (!in_lds) {
#pragma unroll
                    for (int v = 0; v < VPT; v++) {
                        s_val[v * NTHREADS + tid] = val[v];
                        s_idx[v * NTHREADS + tid] = idx[v];
                    }
                    __syncthreads();
                    in_lds = true;
                }
#pragma unroll
                for (int v = 0; v < VPT; v++) {
                    int i = v * NTHREADS + tid;
                    int l = i ^ j;
                    if (l > i) {
                        const bool desc = (i & k) == 0;
                        const bool swap = desc ? (s_val[i] < s_val[l]) : (s_val[i] > s_val[l]);
                        if (swap) {
                            float tv = s_val[i];
                            s_val[i] = s_val[l];
                            s_val[l] = tv;
                            int ti   = s_idx[i];
                            s_idx[i] = s_idx[l];
                            s_idx[l] = ti;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    if (in_lds) {
#pragma unroll
        for (int v = 0; v < VPT; v++) {
            val[v] = s_val[v * NTHREADS + tid];
            idx[v] = s_idx[v * NTHREADS + tid];
        }
        __syncthreads();
    }

// ---- Write top_k results (descending) --------------------------------
#pragma unroll
    for (int v = 0; v < VPT; v++) {
        int p = v * NTHREADS + tid;
        if (p < top_k) {
            if (p < k_eff && isfinite(val[v])) {
                out_indices[(size_t)row * top_k + p] = idx[v];
                if (out_values) out_values[(size_t)row * top_k + p] = val[v];
            } else {
                out_indices[(size_t)row * top_k + p] = -1;
                if (out_values) out_values[(size_t)row * top_k + p] = -INFINITY;
            }
        }
    }
    for (int p = N2 + tid; p < top_k; p += NTHREADS) {
        out_indices[(size_t)row * top_k + p] = -1;
        if (out_values) out_values[(size_t)row * top_k + p] = -INFINITY;
    }
}

__device__ static inline int topk_wave_slot(bool take, int* count) {
    const int lane                = threadIdx.x & 63;
    const unsigned long long mask = __ballot(take);
    int base                      = 0;
    if (mask) {
        const int leader = __ffsll(mask) - 1;
        if (lane == leader) base = atomicAdd(count, __popcll(mask));
        base = __shfl(base, leader, 64);
    }
    return base + __popcll(mask & ((1ull << lane) - 1));
}

template<int VPT, bool ORDER_VALUES, bool THD>
static __global__ __launch_bounds__(256) void indexer_topk_wide_radix_kernel(
    const float* __restrict__ input_values, const int* __restrict__ seq_lens, const int* __restrict__ cu_kv,
    const int* __restrict__ q_offsets, int* __restrict__ out_indices, float* __restrict__ out_values, int n_cols,
    int top_k, int n_seqs, int next_n, int sort_n) {
    const int tx = threadIdx.x, row = blockIdx.x;
    int effective_len = 0;
    if constexpr (THD) {
        if (row < seq_lens[n_seqs]) {
            int lo = 0, hi = n_seqs;
            while (lo + 1 < hi) {
                const int mid = (lo + hi) / 2;
                if (seq_lens[mid] <= row)
                    lo = mid;
                else
                    hi = mid;
            }
            const int pos = row - seq_lens[lo] + (q_offsets ? q_offsets[lo] : 0);
            effective_len = min((pos + 1) / next_n, cu_kv[lo + 1] - cu_kv[lo]);
        }
    } else {
        const int batch = row / next_n;
        if (batch < n_seqs) effective_len = seq_lens[batch] - next_n + row % next_n + 1;
    }
    effective_len        = max(0, min(n_cols, effective_len));
    const int k_eff      = min(top_k, effective_len);
    const size_t out_row = (size_t)row * top_k;
    for (int p = tx; p < top_k; p += 256) {
        out_indices[out_row + p] = -1;
        if constexpr (ORDER_VALUES) out_values[out_row + p] = -INFINITY;
    }
    if (k_eff == 0) return;
    if constexpr (!ORDER_VALUES) {
        if (k_eff == effective_len) {
            for (int p = tx; p < k_eff; p += 256) {
                if (!THD || isfinite(input_values[(size_t)row * n_cols + p])) out_indices[out_row + p] = p;
            }
            return;
        }
    }
    float values[VPT];
    unsigned keys[VPT];
#pragma unroll
    for (int v = 0; v < VPT; ++v) {
        int col       = v * 256 + tx;
        values[v]     = col < effective_len ? input_values[(size_t)row * n_cols + col] : -INFINITY;
        unsigned bits = __float_as_uint(values[v]);
        keys[v]       = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    }
    __shared__ unsigned hist[4][256];
    __shared__ unsigned prefix;
    __shared__ int remaining, count;
    extern __shared__ char candidates[];
    float* cand_val = reinterpret_cast<float*>(candidates);
    int* cand_idx   = reinterpret_cast<int*>(cand_val + sort_n);
    if (tx == 0) {
        prefix    = 0;
        remaining = k_eff;
        count     = 0;
    }
    if constexpr (ORDER_VALUES) {
        for (int p = tx; p < sort_n; p += 256) {
            cand_val[p] = -INFINITY;
            cand_idx[p] = -1;
        }
    }
    __syncthreads();
    unsigned prefix_mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int w = 0; w < 4; ++w) hist[w][tx] = 0;
        __syncthreads();
        const unsigned wanted = prefix;
#pragma unroll
        for (int v = 0; v < VPT; ++v) {
            if (v * 256 + tx < effective_len && (keys[v] & prefix_mask) == wanted)
                atomicAdd(&hist[tx / 64][(keys[v] >> shift) & 255], 1u);
        }
        __syncthreads();
        unsigned sum = hist[0][tx] + hist[1][tx] + hist[2][tx] + hist[3][tx];
        __syncthreads();
        hist[0][tx] = sum;
        __syncthreads();
        // Inclusive suffix sum: one lane locates the kth radix bucket.
        for (int delta = 1; delta < 256; delta <<= 1) {
            unsigned add = tx + delta < 256 ? hist[0][tx + delta] : 0;
            __syncthreads();
            sum         += add;
            hist[0][tx]  = sum;
            __syncthreads();
        }
        const unsigned above = tx < 255 ? hist[0][tx + 1] : 0;
        const int rank       = remaining;
        __syncthreads();
        if (above < rank && sum >= rank) {
            prefix    = wanted | ((unsigned)tx << shift);
            remaining = rank - above;
        }
        prefix_mask |= 255u << shift;
        __syncthreads();
    }
    const unsigned threshold = prefix;
    // Strict winners first, followed by enough equal keys to fill K.
    for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
        for (int v = 0; v < VPT; ++v) {
            const int col   = v * 256 + tx;
            const bool take = col < effective_len && (pass == 0 ? keys[v] > threshold : keys[v] == threshold);
            const int p     = topk_wave_slot(take, &count);
            if (take && p < k_eff) {
                if constexpr (ORDER_VALUES) {
                    cand_val[p] = values[v];
                    cand_idx[p] = col;
                } else if (!THD || isfinite(values[v])) {
                    out_indices[out_row + p] = col;
                }
            }
        }
        __syncthreads();
    }
    if constexpr (ORDER_VALUES) {
        for (int k = 2; k <= sort_n; k <<= 1) {
            for (int j = k >> 1; j; j >>= 1) {
                for (int p = tx; p < sort_n; p += 256) {
                    const int other = p ^ j;
                    if (other > p) {
                        const bool swap = (p & k) == 0 ? cand_val[p] < cand_val[other] : cand_val[p] > cand_val[other];
                        if (swap) {
                            float val       = cand_val[p];
                            cand_val[p]     = cand_val[other];
                            cand_val[other] = val;
                            int idx         = cand_idx[p];
                            cand_idx[p]     = cand_idx[other];
                            cand_idx[other] = idx;
                        }
                    }
                }
                __syncthreads();
            }
        }
        for (int p = tx; p < k_eff; p += 256) {
            if (isfinite(cand_val[p])) {
                out_indices[out_row + p] = cand_idx[p];
                out_values[out_row + p]  = cand_val[p];
            }
        }
    }
}

namespace ftrain {
namespace {
template<bool ORDER_VALUES, bool THD>
static void launch_topk_wide(hipStream_t stream, const float* input, const int* seq, const int* cu_kv,
                             const int* offsets, int* indices, float* values, int rows, int cols, int k, int n_seqs,
                             int next_n);
bool launch_indexer_topk_thd(hipStream_t stream, const float* input_values, const int* cu_seqlens_q,
                             const int* cu_seqlens_kv, const int* q_causal_offsets, int* out_indices, float* out_values,
                             int n_rows, int n_cols, int top_k, int n_seqs, int ratio, bool order_values);

template<bool ORDER_VALUES, bool THD>
static void launch_topk_wide(hipStream_t stream, const float* input, const int* seq, const int* cu_kv,
                             const int* offsets, int* indices, float* values, int rows, int cols, int k, int n_seqs,
                             int next_n) {
    int sort_n = 1;
    while (sort_n < min(k, cols)) sort_n <<= 1;
    const size_t smem = ORDER_VALUES ? sort_n * 8 : 0;
#define WIDE_CASE(V)                                                                                                 \
    hipLaunchKernelGGL((indexer_topk_wide_radix_kernel<V, ORDER_VALUES, THD>), dim3(rows), 256, smem, stream, input, \
                       seq, cu_kv, offsets, indices, values, cols, k, n_seqs, next_n, sort_n)
    if (cols <= 2048) {
        WIDE_CASE(8);
    } else if (cols <= 4096) {
        WIDE_CASE(16);
    } else if (cols <= 8192) {
        WIDE_CASE(32);
    } else {
        WIDE_CASE(64);
    }
#undef WIDE_CASE
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "fp32_indexer_topk_thd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
}

static bool launch_indexer_topk_thd(hipStream_t stream, const float* input_values, const int* cu_seqlens_q,
                                    const int* cu_seqlens_kv, const int* q_causal_offsets, int* out_indices,
                                    float* out_values, int n_rows, int n_cols, int top_k, int n_seqs, int ratio,
                                    bool order_values) {
    if (n_cols > DSA_TOPK_MAX_COLS || top_k > DSA_TOPK_MAX_K || n_cols < 1 || top_k < 1 || n_seqs < 1 || ratio < 1)
        return false;
    if (n_cols > 1024) {
        if (order_values)
            launch_topk_wide<true, true>(stream, input_values, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets,
                                         out_indices, out_values, n_rows, n_cols, top_k, n_seqs, ratio);
        else
            launch_topk_wide<false, true>(stream, input_values, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets,
                                          out_indices, out_values, n_rows, n_cols, top_k, n_seqs, ratio);
        return true;
    }
    int n2 = 4;
    while (n2 < n_cols) n2 <<= 1;
    if (n2 < 4 || n2 > 1024 || n_seqs < 1) return false;
    size_t smem = (size_t)n2 * 2 * sizeof(float);
    switch (n2) {
        case 1024:
            hipLaunchKernelGGL((indexer_topk_bitonic_thd_kernel<256, 4>), dim3(n_rows), 256, smem, stream, input_values,
                               cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, n_rows, n_cols,
                               top_k, n_seqs, ratio);
            break;
        case 512:
            hipLaunchKernelGGL((indexer_topk_bitonic_thd_kernel<256, 2>), dim3(n_rows), 256, smem, stream, input_values,
                               cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, n_rows, n_cols,
                               top_k, n_seqs, ratio);
            break;
        case 256:
            hipLaunchKernelGGL((indexer_topk_bitonic_thd_kernel<256, 1>), dim3(n_rows), 256, smem, stream, input_values,
                               cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, n_rows, n_cols,
                               top_k, n_seqs, ratio);
            break;
        case 128:
            hipLaunchKernelGGL((indexer_topk_bitonic_thd_kernel<128, 1>), dim3(n_rows), 128, smem, stream, input_values,
                               cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, n_rows, n_cols,
                               top_k, n_seqs, ratio);
            break;
        case 64:
            hipLaunchKernelGGL((indexer_topk_bitonic_thd_kernel<64, 1>), dim3(n_rows), 64, smem, stream, input_values,
                               cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, n_rows, n_cols,
                               top_k, n_seqs, ratio);
            break;
        case 32:
            hipLaunchKernelGGL((indexer_topk_bitonic_thd_kernel<32, 1>), dim3(n_rows), 32, smem, stream, input_values,
                               cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, n_rows, n_cols,
                               top_k, n_seqs, ratio);
            break;
        case 16:
            hipLaunchKernelGGL((indexer_topk_bitonic_thd_kernel<16, 1>), dim3(n_rows), 16, smem, stream, input_values,
                               cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, n_rows, n_cols,
                               top_k, n_seqs, ratio);
            break;
        case 8:
            hipLaunchKernelGGL((indexer_topk_bitonic_thd_kernel<8, 1>), dim3(n_rows), 8, smem, stream, input_values,
                               cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, n_rows, n_cols,
                               top_k, n_seqs, ratio);
            break;
        case 4:
            hipLaunchKernelGGL((indexer_topk_bitonic_thd_kernel<4, 1>), dim3(n_rows), 4, smem, stream, input_values,
                               cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, n_rows, n_cols,
                               top_k, n_seqs, ratio);
            break;
        default: return false;
    }
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "fp32_indexer_topk_thd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}
}  // namespace

bool launchFp32IndexerTopKThd(FTrainStream stream, const float* input_values, const int* cu_seqlens_q,
                              const int* cu_seqlens_kv, const int* q_causal_offsets, int* out_indices,
                              float* out_values, int num_rows, int num_columns, int top_k, int num_sequences, int ratio,
                              bool order_values) {
    return launch_indexer_topk_thd(stream, input_values, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices,
                                   out_values, num_rows, num_columns, top_k, num_sequences, ratio, order_values);
}

}  // namespace ftrain
