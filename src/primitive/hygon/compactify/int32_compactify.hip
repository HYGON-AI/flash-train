// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/compactify/int32_compactify.hpp"

// Adapted from fused_dsa/hip_dsa/kernels/compactify.hip.

static __global__ void compactify_kernel(const int* __restrict__ indices,  // (n_rows, topk)
                                         int* __restrict__ output,         // (n_rows, topk)
                                         int* __restrict__ topk_length,    // (n_rows,)
                                         int n_rows, int topk) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    int tid       = threadIdx.x;
    int n_threads = blockDim.x;

    // Compute next power of 2 for scan (must be power-of-2 for Blelloch)
    int n_pow2 = 1;
    while (n_pow2 < n_threads) n_pow2 <<= 1;

    // Shared memory: counts for scan (size = next_pow2, max 256)
    __shared__ int scan_buf[256];

    // ---- Phase 1: Count valid entries per thread ---------------------------
    int my_count = 0;
    for (int k = tid; k < topk; k += n_threads) {
        if (indices[row * topk + k] >= 0) { my_count++; }
    }
    scan_buf[tid] = my_count;

    // Zero-pad the scan buffer for positions >= n_threads (for power-of-2 scan)
    for (int i = n_threads + tid; i < n_pow2; i += n_threads) { scan_buf[i] = 0; }
    __syncthreads();

    // ---- Phase 2: Blelloch exclusive scan on n_pow2 elements ---------------
    // Upsweep
    for (int stride = 1; stride < n_pow2; stride <<= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < n_pow2) { scan_buf[idx] += scan_buf[idx - stride]; }
        __syncthreads();
    }

    // Save total and clear root for exclusive scan
    if (tid == 0) {
        int total            = scan_buf[n_pow2 - 1];
        scan_buf[n_pow2 - 1] = 0;
        topk_length[row]     = total;
    }
    __syncthreads();

    // Downsweep
    for (int stride = n_pow2 / 2; stride > 0; stride >>= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < n_pow2) {
            int tmp                 = scan_buf[idx - stride];
            scan_buf[idx - stride]  = scan_buf[idx];
            scan_buf[idx]          += tmp;
        }
        __syncthreads();
    }

    // Now: scan_buf[tid] = exclusive prefix sum for tid < n_threads

    // ---- Phase 3: Scatter valid entries ------------------------------------
    int write_pos = scan_buf[tid];  // my starting output position
    for (int k = tid; k < topk; k += n_threads) {
        int idx = indices[row * topk + k];
        if (idx >= 0) {
            output[row * topk + write_pos] = idx;
            write_pos++;
        }
    }

    __syncthreads();

    // ---- Phase 4: Fill trailing slots with -1 ------------------------------
    // Read total_valid from shared memory (written by thread 0 in Phase 2)
    __shared__ int total_valid_shared;
    if (tid == 0) { total_valid_shared = topk_length[row]; }
    __syncthreads();

    int total_valid = total_valid_shared;
    for (int k = total_valid + tid; k < topk; k += n_threads) { output[row * topk + k] = -1; }
}

namespace ftrain {
namespace {
void launch_compactify(hipStream_t stream, const int* indices, int* output, int* topk_length, int n_rows, int topk);

static void launch_compactify(hipStream_t stream, const int* indices, int* output, int* topk_length, int n_rows,
                              int topk) {
    int threads = min(topk, 256);
    hipLaunchKernelGGL(compactify_kernel, n_rows, threads, 0, stream, indices, output, topk_length, n_rows, topk);
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "int32_compactify HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
}
}  // namespace

void launchInt32Compactify(FTrainStream stream, const int* indices, int* output, int* topk_length, int num_rows,
                           int selection_width) {
    launch_compactify(stream, indices, output, topk_length, num_rows, selection_width);
}

}  // namespace ftrain
