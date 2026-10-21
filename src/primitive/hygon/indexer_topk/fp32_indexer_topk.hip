// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <math.h>

#include <hip/hip_runtime.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/indexer_topk/fp32_indexer_topk.hpp"

// Adapted from fused_dsa/hip_dsa/kernels/indexer_topk.hip.

template<int NTHREADS, int VPT>
static __global__ void indexer_topk_bitonic_kernel(const float* __restrict__ input_values,
                                                   const int* __restrict__ seq_lens, int* __restrict__ out_indices,
                                                   float* __restrict__ out_values, int n_rows, int n_cols, int top_k,
                                                   int batch_size, int next_n) {
    constexpr int N2 = NTHREADS * VPT;  // padded sort length (power of two)
    const int row    = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = threadIdx.x;

    int effective_len = n_cols;
    if (seq_lens != NULL) {
        int batch_idx = (next_n > 0) ? row / next_n : 0;
        int task_id   = (next_n > 0) ? row % next_n : 0;
        if (batch_idx >= batch_size) effective_len = 0;
        if (batch_idx < batch_size) {
            effective_len = seq_lens[batch_idx] - next_n + task_id + 1;
            if (effective_len > n_cols) effective_len = n_cols;
            if (effective_len < 0) effective_len = 0;
        }
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
    // ---- Bitonic network: k = 2,4,...,N2 ; j = k/2 ... 1 ------------------
    for (int k = 2; k <= N2; k <<= 1) {
        for (int j = k >> 1; j >= 1; j >>= 1) {
            if (j >= NTHREADS) {
                // Register compare-exchange (same thread, partner v ^ shift).
                if (in_lds) {
// Read the LDS state back into registers first.
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
                // LDS compare-exchange (partner (v, tid ^ j) in another thread).
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

    for (int p = N2 + tid; p < top_k; p += NTHREADS) {
        out_indices[(size_t)row * top_k + p] = -1;
        if (out_values) out_values[(size_t)row * top_k + p] = -INFINITY;
    }

// ---- Write top_k results (descending) --------------------------------
#pragma unroll
    for (int v = 0; v < VPT; v++) {
        int p = v * NTHREADS + tid;
        if (p < top_k) {
            if (p < k_eff) {
                out_indices[(size_t)row * top_k + p] = idx[v];
                if (out_values) out_values[(size_t)row * top_k + p] = val[v];
            } else {
                out_indices[(size_t)row * top_k + p] = -1;
                if (out_values) out_values[(size_t)row * top_k + p] = -INFINITY;
            }
        }
    }
}

__device__ static inline uint32_t itk_twiddle(float v) {
    uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template<int NTHREADS, int VPT, bool ORDER_VALUES>
static __global__ void __launch_bounds__(1024, 2)
    indexer_topk_radix_kernel(const float* __restrict__ input_values, const int* __restrict__ seq_lens,
                              int* __restrict__ out_indices, float* __restrict__ out_values, int n_rows, int n_cols,
                              int top_k, int batch_size, int next_n) {
    constexpr int N2 = NTHREADS * VPT;  // 1024 padded row length
    __shared__ int hist[257];           // [0..255] suffix scan, [256] spare
    __shared__ int s_counter;           // selected-so-far output slot
    __shared__ int s_num[2];            // staged counts (double buffered)
    __shared__ int s_threshold;
    __shared__ int s_remaining;

    // Dynamic LDS: s_idx0 | s_idx1 | cand_val | cand_idx (ORDER only).
    extern __shared__ char smem_raw[];
    int* s_idx0     = (int*)smem_raw;
    int* s_idx1     = (int*)(smem_raw + 4096);
    float* cand_val = (float*)(smem_raw + 8192);
    int* cand_idx   = (int*)(smem_raw + 12288);

    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tx   = threadIdx.x;
    const int lane = tx & 63;

    int effective_len = n_cols;
    if (seq_lens != NULL) {
        int batch_idx = (next_n > 0) ? row / next_n : 0;
        int task_id   = (next_n > 0) ? row % next_n : 0;
        if (batch_idx >= batch_size) effective_len = 0;
        if (batch_idx < batch_size) {
            effective_len = seq_lens[batch_idx] - next_n + task_id + 1;
            if (effective_len > n_cols) effective_len = n_cols;
            if (effective_len < 0) effective_len = 0;
        }
    }
    const int k_eff = (top_k < effective_len) ? top_k : effective_len;

    if (k_eff == 0) {
        for (int p = tx; p < top_k; p += NTHREADS) {
            out_indices[(size_t)row * top_k + p] = -1;
            if (out_values) out_values[(size_t)row * top_k + p] = -INFINITY;
        }
        return;
    }

    const float* row_values = input_values + (size_t)row * n_cols;

    // ---- One element per thread (VPT == 1) -------------------------------
    const bool live    = (tx < effective_len);
    const int col0     = live ? tx : -1;
    const float fv0    = live ? row_values[tx] : -INFINITY;
    const uint32_t tw0 = live ? itk_twiddle(fv0) : 0u;

    // ---- Selection sink ----------------------------------------------------
    auto sink = [&](int pos, int col, float fv) {
        if (ORDER_VALUES) {
            cand_val[pos] = fv;
            cand_idx[pos] = col;
        } else {
            out_indices[(size_t)row * top_k + pos] = col;
            if (out_values) out_values[(size_t)row * top_k + pos] = fv;
        }
    };

    auto fill_pad = [&]() {
        for (int p = k_eff + tx; p < top_k; p += NTHREADS) {
            out_indices[(size_t)row * top_k + p] = -1;
            if (out_values) out_values[(size_t)row * top_k + p] = -INFINITY;
        }
    };

    if (k_eff == effective_len) {
        // Every live element is selected.
        if (!ORDER_VALUES) {
            if (live) {
                out_indices[(size_t)row * top_k + tx] = col0;
                if (out_values) out_values[(size_t)row * top_k + tx] = fv0;
            }
            fill_pad();
            return;
        }
        cand_val[tx] = fv0;
        cand_idx[tx] = col0;
        __syncthreads();
    } else {
        // ---- Round 0: histogram of the top key byte -----------------------
        if (tx < 257) hist[tx] = 0;
        if (tx == 0) {
            s_counter   = 0;
            s_num[0]    = 0;
            s_num[1]    = 0;
            s_threshold = -1;
            s_remaining = 0;
        }
        __syncthreads();
        if (live) {
            const int byte = (int)(tw0 >> 24);
            atomicAdd(&hist[byte], 1);
        }
        __syncthreads();

// Suffix scan: hist[b] -> count(byte >= b).
#pragma unroll
        for (int step = 1; step < 256; step <<= 1) {
            int v = 0, v2 = 0;
            if (tx < 256) {
                v = hist[tx];
                if (tx + step < 256) v2 = hist[tx + step];
            }
            __syncthreads();
            if (tx < 256) hist[tx] = v + v2;
            __syncthreads();
        }

        // Threshold bin b: hist[b] >= k_eff > hist[b+1].
        if (tx < 256) {
            const int above = (tx < 255) ? hist[tx + 1] : 0;
            if (hist[tx] >= k_eff && above < k_eff) {
                s_threshold = tx;
                s_remaining = k_eff - above;
            }
        }
        __syncthreads();

        int threshold = s_threshold;
        int remaining = s_remaining;
        bool done     = (remaining == 0);

        // ---- Byte rounds 0..3 (offsets 24,16,8,0) -------------------------
        int r = 0;
        while (!done && r < 4) {
            // Reset the next-round histogram before the fused staging
            // atomics; also reset the staging counter of this round.
            if (tx < 257) hist[tx] = 0;
            if (r > 0 && tx == 0) s_num[r & 1] = 0;
            __syncthreads();

            const int byte_off = 24 - r * 8;
            bool is_above = false, is_bound = false;
            int col_r    = -1;
            float fvr    = -INFINITY;
            uint32_t twr = 0u;

            if (r == 0) {
                col_r    = col0;
                fvr      = fv0;
                twr      = tw0;
                is_above = live && ((int)(twr >> 24) > threshold);
                is_bound = live && ((int)(twr >> 24) == threshold);
            } else {
                const int src_r   = (r - 1) & 1;
                const int num_src = s_num[src_r];
                if (tx < num_src) {
                    col_r            = (src_r == 0) ? s_idx0[tx] : s_idx1[tx];
                    fvr              = row_values[col_r];
                    twr              = itk_twiddle(fvr);
                    const int byte_r = (int)((twr >> byte_off) & 0xFFu);
                    is_above         = (byte_r > threshold);
                    is_bound         = (byte_r == threshold);
                }
            }

            // "Above" elements: final slot via warp-ballot prefix.
            const unsigned long long ball_a = __ballot(is_above ? 1 : 0);
            int base_a                      = 0;
            if (lane == 0) base_a = atomicAdd(&s_counter, (int)__popcll(ball_a));
            base_a = __shfl(base_a, 0);
            if (is_above) {
                const int intra = (int)__popcll(ball_a & ((1ull << lane) - 1ull));
                sink(base_a + intra, col_r, fvr);
            }

            // Threshold-bin survivors: stage (index only) + fused
            // next-byte histogram.  The last round needs no staging.
            if (r < 3) {
                const unsigned long long ball_b = __ballot(is_bound ? 1 : 0);
                int base_b                      = 0;
                if (lane == 0) base_b = atomicAdd(&s_num[r & 1], (int)__popcll(ball_b));
                base_b = __shfl(base_b, 0);
                if (is_bound) {
                    const int intra = (int)__popcll(ball_b & ((1ull << lane) - 1ull));
                    const int p     = base_b + intra;
                    if ((r & 1) == 0)
                        s_idx0[p] = col_r;
                    else
                        s_idx1[p] = col_r;
                    const int byte_n = (int)((twr >> (byte_off - 8)) & 0xFFu);
                    atomicAdd(&hist[byte_n], 1);
                }
            }
            __syncthreads();

            if (r < 3) {
                const int n_staged = s_num[r & 1];
                if (n_staged == remaining) {
                    // All staged survivors are selected.
                    const int base = k_eff - remaining;
                    for (int i = tx; i < n_staged; i += NTHREADS) {
                        const int c = ((r & 1) == 0) ? s_idx0[i] : s_idx1[i];
                        sink(base + i, c, row_values[c]);
                    }
                    done = true;
                } else {
// Suffix scan of the fused next-byte histogram.
#pragma unroll
                    for (int step = 1; step < 256; step <<= 1) {
                        int v = 0, v2 = 0;
                        if (tx < 256) {
                            v = hist[tx];
                            if (tx + step < 256) v2 = hist[tx + step];
                        }
                        __syncthreads();
                        if (tx < 256) hist[tx] = v + v2;
                        __syncthreads();
                    }
                    if (tx < 256) {
                        const int above = (tx < 255) ? hist[tx + 1] : 0;
                        if (hist[tx] >= remaining && above < remaining) {
                            s_threshold = tx;
                            s_remaining = remaining - above;
                        }
                    }
                    __syncthreads();
                    threshold = s_threshold;
                    remaining = s_remaining;
                    if (remaining == 0) done = true;
                }
            } else {
                // r == 3: lowest byte; all threshold-bin survivors share
                // one 32-bit key (identical value), so take exactly
                // `remaining` of them in any order.
                if (is_bound) {
                    const int p = atomicAdd(&s_counter, 1);
                    if (p < k_eff) sink(p, col_r, fvr);
                }
                done = true;
            }
            r++;
        }

        if (!ORDER_VALUES) {
            fill_pad();
            return;
        }
        // ORDER_VALUES: selected (value,index) pairs sit in cand arrays.
        __syncthreads();
    }

    // ---- Final ordering (ORDER_VALUES only): bitonic sort the k_eff
    // candidates by value, descending, carrying the index.
    {
        int N2c = 1;
        while (N2c < k_eff) N2c <<= 1;
        for (int i = k_eff + tx; i < N2c; i += NTHREADS) {
            cand_val[i] = -INFINITY;
            cand_idx[i] = -1;
        }
        __syncthreads();
        // N2c <= N2 == NTHREADS: one element per thread (slot i == tx).
        for (int k = 2; k <= N2c; k <<= 1) {
            for (int j = k >> 1; j >= 1; j >>= 1) {
                const int i = tx;
                const int l = i ^ j;
                if (i < N2c && l > i) {
                    const bool desc = ((i & k) == 0);
                    const bool swap = desc ? (cand_val[i] < cand_val[l]) : (cand_val[i] > cand_val[l]);
                    if (swap) {
                        const float tv = cand_val[i];
                        cand_val[i]    = cand_val[l];
                        cand_val[l]    = tv;
                        const int ti   = cand_idx[i];
                        cand_idx[i]    = cand_idx[l];
                        cand_idx[l]    = ti;
                    }
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }

// ---- Write the top_k row ----------------------------------------------
#pragma unroll
    for (int v = 0; v < VPT; v++) {
        const int p = v * NTHREADS + tx;
        if (p < top_k) {
            if (p < k_eff && isfinite(cand_val[p])) {
                out_indices[(size_t)row * top_k + p] = cand_idx[p];
                if (out_values) out_values[(size_t)row * top_k + p] = cand_val[p];
            } else {
                out_indices[(size_t)row * top_k + p] = -1;
                if (out_values) out_values[(size_t)row * top_k + p] = -INFINITY;
            }
        }
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
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "fp32_indexer_topk HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
}

static void launch_indexer_topk(hipStream_t stream, const float* input_values, const int* seq_lens, int* out_indices,
                                float* out_values, int n_rows, int n_cols, int top_k, int batch_size, int next_n,
                                bool order_values) {
    if (n_cols > 1024 || (n_cols > 512 && top_k > 1024)) {
        if (order_values)
            launch_topk_wide<true, false>(stream, input_values, seq_lens, nullptr, nullptr, out_indices, out_values,
                                          n_rows, n_cols, top_k, batch_size, next_n);
        else
            launch_topk_wide<false, false>(stream, input_values, seq_lens, nullptr, nullptr, out_indices, out_values,
                                           n_rows, n_cols, top_k, batch_size, next_n);
        return;
    }
    // ---- Bitonic fast path: padded sort length <= 1024 --------------------
    {
        int n2 = 4;
        while (n2 < n_cols) n2 <<= 1;
        if (n2 >= 4 && n2 <= 1024) {
            size_t smem = (size_t)n2 * 2 * sizeof(float);
            if (n2 == 1024) {
                // Radix-select beats the full 1024-element bitonic sort for
                // top_k < 1024: O(8 passes) selection + already-sorted
                // candidates instead of O(log^2 n) compare-exchange rounds.
                const size_t smem = order_values ? (16 * 1024) : (8 * 1024);
                if (order_values) {
                    hipLaunchKernelGGL((indexer_topk_radix_kernel<1024, 1, true>), dim3(n_rows), 1024, smem, stream,
                                       input_values, seq_lens, out_indices, out_values, n_rows, n_cols, top_k,
                                       batch_size, next_n);
                } else {
                    hipLaunchKernelGGL((indexer_topk_radix_kernel<1024, 1, false>), dim3(n_rows), 1024, smem, stream,
                                       input_values, seq_lens, out_indices, out_values, n_rows, n_cols, top_k,
                                       batch_size, next_n);
                }
            } else if (n2 == 512) {
                hipLaunchKernelGGL((indexer_topk_bitonic_kernel<256, 2>), dim3(n_rows), 256, smem, stream, input_values,
                                   seq_lens, out_indices, out_values, n_rows, n_cols, top_k, batch_size, next_n);
            } else if (n2 == 256) {
                hipLaunchKernelGGL((indexer_topk_bitonic_kernel<256, 1>), dim3(n_rows), 256, smem, stream, input_values,
                                   seq_lens, out_indices, out_values, n_rows, n_cols, top_k, batch_size, next_n);
            } else if (n2 == 128) {
                hipLaunchKernelGGL((indexer_topk_bitonic_kernel<128, 1>), dim3(n_rows), 128, smem, stream, input_values,
                                   seq_lens, out_indices, out_values, n_rows, n_cols, top_k, batch_size, next_n);
            } else if (n2 == 64) {
                hipLaunchKernelGGL((indexer_topk_bitonic_kernel<64, 1>), dim3(n_rows), 64, smem, stream, input_values,
                                   seq_lens, out_indices, out_values, n_rows, n_cols, top_k, batch_size, next_n);
            } else if (n2 == 32) {
                hipLaunchKernelGGL((indexer_topk_bitonic_kernel<32, 1>), dim3(n_rows), 32, smem, stream, input_values,
                                   seq_lens, out_indices, out_values, n_rows, n_cols, top_k, batch_size, next_n);
            } else if (n2 == 16) {
                hipLaunchKernelGGL((indexer_topk_bitonic_kernel<16, 1>), dim3(n_rows), 16, smem, stream, input_values,
                                   seq_lens, out_indices, out_values, n_rows, n_cols, top_k, batch_size, next_n);
            } else if (n2 == 8) {
                hipLaunchKernelGGL((indexer_topk_bitonic_kernel<8, 1>), dim3(n_rows), 8, smem, stream, input_values,
                                   seq_lens, out_indices, out_values, n_rows, n_cols, top_k, batch_size, next_n);
            } else {
                hipLaunchKernelGGL((indexer_topk_bitonic_kernel<4, 1>), dim3(n_rows), 4, smem, stream, input_values,
                                   seq_lens, out_indices, out_values, n_rows, n_cols, top_k, batch_size, next_n);
            }
            {
                const hipError_t status = hipGetLastError();
                if (status != hipSuccess) {
                    throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "fp32_indexer_topk HIP call failed: %s",
                                            hipGetErrorString(status));
                }
            }
            return;
        }
    }
}
}  // namespace

bool launchFp32IndexerTopKK(FTrainStream stream, const float* input_values, const int* seq_lens, int* out_indices,
                            float* out_values, int num_rows, int num_columns, int top_k, int batch_size, int next_n,
                            bool order_values) {
    launch_indexer_topk(stream, input_values, seq_lens, out_indices, out_values, num_rows, num_columns, top_k,
                        batch_size, next_n, order_values);
    return true;
}

}  // namespace ftrain
