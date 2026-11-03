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

#include <hipcub/hipcub.hpp>

__device__ inline int topk_wave_slot(bool take, int* count) {
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

// Wide rows: four radix histogram passes find the exact kth float key.
// Values stay in registers; LDS usage depends on the selected K, not N*K.
// The optional bitonic network sorts only the selected candidates.
template<int VPT, bool ORDER_VALUES, bool THD, int EMIT_GROUP = 0>
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
#if (defined(__gfx936__) || defined(__gfx938__))
        // One length per independent row is the common BSHD contract. Read
        // current device metadata on every replay, without generic division.
        if (!ORDER_VALUES && next_n == 1) {
            if (row < n_seqs) effective_len = seq_lens[row];
        } else
#endif
        {
            const int batch = row / next_n;
            if (batch < n_seqs) effective_len = seq_lens[batch] - next_n + row % next_n + 1;
        }
    }
    effective_len        = max(0, min(n_cols, effective_len));
    const int k_eff      = min(top_k, effective_len);
    const size_t out_row = (size_t)row * top_k;
#if (defined(__gfx936__) || defined(__gfx938__))
    // BSHD indices-only always fills every selected slot, including NaN keys.
    // Initialize only the padding tail. THD nonfinite filtering can leave holes
    // and therefore retains full initialization on both architectures.
    const int init_begin = (!ORDER_VALUES && !THD) ? k_eff : 0;
#else
    const int init_begin = 0;
#endif
    for (int p = tx + init_begin; p < top_k; p += 256) {
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
#if (defined(__gfx936__) || defined(__gfx938__))
    __shared__ int bucket_complete;
#endif
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
        unsigned sum, above;
        if constexpr (ORDER_VALUES) {
            sum = hist[0][tx] + hist[1][tx] + hist[2][tx] + hist[3][tx];
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
            above = tx < 255 ? hist[0][tx + 1] : 0;
        } else {
            const unsigned own = hist[0][tx] + hist[1][tx] + hist[2][tx] + hist[3][tx];
            // Exact integer suffix scan: six wave shuffles and four wave totals.
            // Retire all histogram readers before reusing its first four slots.
            __syncthreads();
            sum            = own;
            const int lane = tx & 63, wave = tx >> 6;
#pragma unroll
            for (int delta = 1; delta < 64; delta <<= 1) {
                const unsigned other = __shfl_down(sum, delta, 64);
                if (lane + delta < 64) sum += other;
            }
            if (lane == 0) hist[0][wave] = sum;
            __syncthreads();
#pragma unroll
            for (int w = 0; w < 4; w++)
                if (w > wave) sum += hist[0][w];
            above = sum - own;
        }
        const int rank = remaining;
        __syncthreads();
        if (above < rank && sum >= rank) {
            prefix    = wanted | ((unsigned)tx << shift);
            remaining = rank - above;
#if (defined(__gfx936__) || defined(__gfx938__))
            if constexpr (!ORDER_VALUES && !THD) bucket_complete = (sum == rank);
#endif
        }
        prefix_mask |= 255u << shift;
        __syncthreads();
#if (defined(__gfx936__) || defined(__gfx938__))
        // All keys in this bucket exactly fill the remaining slots, so its
        // lower radix bytes cannot change the selected set. Keep the existing
        // winners/equality and optional sorting stages unchanged.
        if constexpr (!ORDER_VALUES && !THD)
            if (bucket_complete) break;
#endif
    }
    const unsigned threshold = prefix;
    // Strict winners precede equality exactly as before. Reserve several
    // coalesced wave groups at once; BSHD indices-only has no order promise.
    for (int pass = 0; pass < 2; ++pass) {
        if constexpr (EMIT_GROUP > 0 && !ORDER_VALUES && !THD) {
            const int lane                 = tx & 63;
            const unsigned long long lower = (1ull << lane) - 1;
#pragma unroll
            for (int vb = 0; vb < VPT; vb += EMIT_GROUP) {
                unsigned long long masks[EMIT_GROUP];
                unsigned total = 0;
#pragma unroll
                for (int j = 0; j < EMIT_GROUP; j++) {
                    const int v = vb + j, col = v * 256 + tx;
                    const bool take  = col < effective_len && (pass == 0 ? keys[v] > threshold : keys[v] == threshold);
                    masks[j]         = __ballot(take);
                    total           += (unsigned)__popcll(masks[j]);
                }
                int base = 0;
                if (lane == 0 && total) base = atomicAdd(&count, (int)total);
                base = __shfl(base, 0, 64);
#pragma unroll
                for (int j = 0; j < EMIT_GROUP; j++) {
                    const unsigned long long mask = masks[j];
                    const int pos                 = base + __popcll(mask & lower);
                    if (((mask >> lane) & 1ull) && pos < k_eff) out_indices[out_row + pos] = (vb + j) * 256 + tx;
                    base += __popcll(mask);
                }
            }
        } else {
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
        }
        __syncthreads();
        // BSHD indices-only writes every reserved strict-winner slot.
        // After this CTA barrier, count>=K proves equality cannot add output.
        if constexpr (EMIT_GROUP > 0 && !ORDER_VALUES && !THD) {
            if (pass == 0 && count >= k_eff) break;
        }
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

// Source-backed DTK BlockRadixSort. Use the original unsigned float-key
// transform for exact ordering, including signed zero and NaN bit patterns.
// The comparison network is not a total order for NaN; replacing it also
// prevents finite candidates being lost beyond a short row's k_eff boundary.
#if defined(__HIP_DEVICE_COMPILE__)
template<int ITEMS>
__device__ __forceinline__ void topk_ordered_uint_row(const float* input, int* output_indices, float* output_values,
                                                      int row, int n_cols, int effective_len, int top_k, int k_eff,
                                                      char* storage) {
    using Sort = hipcub::BlockRadixSort<unsigned int, 256, ITEMS, int>;
    unsigned keys[ITEMS];
    int ids[ITEMS];
#    pragma unroll
    for (int u = 0; u < ITEMS; u++) {
        const int col       = threadIdx.x * ITEMS + u;
        const bool valid    = col < effective_len;
        const unsigned bits = valid ? __float_as_uint(input[(size_t)row * n_cols + col]) : 0;
        keys[u]             = valid ? (bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u)) : 0u;
        ids[u]              = valid ? col : -1;
    }
    Sort(*reinterpret_cast<typename Sort::TempStorage*>(storage)).SortDescendingBlockedToStriped(keys, ids);
#    pragma unroll
    for (int u = 0; u < ITEMS; u++) {
        const int pos = u * 256 + threadIdx.x;
        if (pos < top_k) {
            const unsigned bits = keys[u] ^ ((keys[u] & 0x80000000u) ? 0x80000000u : 0xffffffffu);
            const bool valid    = pos < k_eff && ids[u] >= 0 && (bits & 0x7f800000u) != 0x7f800000u;
            output_indices[(size_t)row * top_k + pos] = valid ? ids[u] : -1;
            output_values[(size_t)row * top_k + pos]  = valid ? __uint_as_float(bits) : -INFINITY;
        }
    }
    for (int pos = threadIdx.x + 256 * ITEMS; pos < top_k; pos += 256) {
        output_indices[(size_t)row * top_k + pos] = -1;
        output_values[(size_t)row * top_k + pos]  = -INFINITY;
    }
}
#endif

namespace ftrain {
namespace {
template<bool ORDER_VALUES, bool THD, int CAPACITY>
static void launch_topk_wide(hipStream_t stream, const float* input, const int* seq, const int* cu_kv,
                             const int* offsets, int* indices, float* values, int rows, int cols, int k, int n_seqs,
                             int next_n);

template<bool ORDER_VALUES, bool THD, int CAPACITY>
static void launch_topk_wide(hipStream_t stream, const float* input, const int* seq, const int* cu_kv,
                             const int* offsets, int* indices, float* values, int rows, int cols, int k, int n_seqs,
                             int next_n) {
    int sort_n = 1;
    while (sort_n < min(k, cols)) sort_n <<= 1;
    const size_t smem = ORDER_VALUES ? sort_n * 8 : 0;
#define WIDE_CASE(V)                                                                                                 \
    hipLaunchKernelGGL((indexer_topk_wide_radix_kernel<V, ORDER_VALUES, THD>), dim3(rows), 256, smem, stream, input, \
                       seq, cu_kv, offsets, indices, values, cols, k, n_seqs, next_n, sort_n)
    WIDE_CASE(CAPACITY);
#undef WIDE_CASE
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "fp32_indexer_topk_thd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
}

template<class Implementation, int SORT_SIZE, int CAPACITY, bool ORDER>
static bool launch_indexer_topk_thd(hipStream_t stream, const float* input_values, const int* cu_seqlens_q,
                                    const int* cu_seqlens_kv, const int* q_causal_offsets, int* out_indices,
                                    float* out_values, int n_rows, int n_cols, int top_k, int n_seqs, int ratio,
                                    bool order_values) {
    if constexpr ((Implementation::kWideRadix)) {
        if constexpr (ORDER)
            launch_topk_wide<true, true, CAPACITY>(stream, input_values, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets,
                                                   out_indices, out_values, n_rows, n_cols, top_k, n_seqs, ratio);
        else
            launch_topk_wide<false, true, CAPACITY>(stream, input_values, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets,
                                                    out_indices, out_values, n_rows, n_cols, top_k, n_seqs, ratio);
        return true;
    }
    constexpr int n2 = SORT_SIZE;
    size_t smem = (size_t)n2 * 2 * sizeof(float);
    constexpr int threads = SORT_SIZE < 256 ? SORT_SIZE : 256;
    constexpr int items   = SORT_SIZE / threads;
    hipLaunchKernelGGL((indexer_topk_bitonic_thd_kernel<threads, items>), dim3(n_rows), threads, smem, stream,
                       input_values, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, n_rows,
                       n_cols, top_k, n_seqs, ratio);
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

}  // namespace ftrain

namespace ftrain {

template<class Implementation, int SORT_SIZE, int CAPACITY, bool ORDER>
bool launchFp32IndexerTopKThd(FTrainStream stream, const float* input_values, const int* cu_seqlens_q,
                              const int* cu_seqlens_kv, const int* q_causal_offsets, int* out_indices,
                              float* out_values, int num_rows, int num_columns, int top_k, int num_sequences, int ratio,
                              bool order_values) {
    return launch_indexer_topk_thd<Implementation, SORT_SIZE, CAPACITY, ORDER>(
        stream, input_values, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, num_rows,
        num_columns, top_k, num_sequences, ratio, order_values);
}

// Explicit fixed launch instances used by the C++ configuration code.
#define FTRAIN_INSTANTIATE_LAUNCH(Implementation, SORT_SIZE, CAPACITY, ORDER)                                       \
    template bool launchFp32IndexerTopKThd<Implementation, SORT_SIZE, CAPACITY, ORDER>(                             \
        FTrainStream stream, const float* input_values, const int* cu_seqlens_q, const int* cu_seqlens_kv,          \
        const int* q_causal_offsets, int* out_indices, float* out_values, int num_rows, int num_columns, int top_k, \
        int num_sequences, int ratio, bool order_values);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThd, 1024, 8, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThd, 1024, 8, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize128, 128, 8, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize128, 128, 8, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize16, 16, 8, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize16, 16, 8, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize256, 256, 8, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize256, 256, 8, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize32, 32, 8, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize32, 32, 8, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize4, 4, 8, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize4, 4, 8, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize512, 512, 8, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize512, 512, 8, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize64, 64, 8, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize64, 64, 8, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize8, 8, 8, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdSize8, 8, 8, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdWideRadix, 1024, 8, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdWideRadix, 1024, 8, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdWideRadixCapacity16, 1024, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdWideRadixCapacity16, 1024, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdWideRadixCapacity32, 1024, 32, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdWideRadixCapacity32, 1024, 32, true);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdWideRadixCapacity64, 1024, 64, false);
FTRAIN_INSTANTIATE_LAUNCH(Fp32IndexerTopKThdWideRadixCapacity64, 1024, 64, true);
#undef FTRAIN_INSTANTIATE_LAUNCH

}  // namespace ftrain
