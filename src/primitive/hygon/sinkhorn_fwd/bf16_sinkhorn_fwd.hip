// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sinkhorn_fwd/bf16_sinkhorn_fwd.hpp"

template<typename T>
__device__ __forceinline__ float to_float(T v);
template<>
__device__ __forceinline__ float to_float(float v) {
    return v;
}

template<>
__device__ __forceinline__ float to_float(hip_bfloat16 v) {
    return float(v);
}

template<typename T>
__device__ __forceinline__ T from_float(float v);
template<>
__device__ __forceinline__ float from_float(float v) {
    return v;
}

template<>
__device__ __forceinline__ hip_bfloat16 from_float<hip_bfloat16>(float v) {
    return hip_bfloat16(v);
}

// Adapted from fused_mHC/csrc/sinkhorn.cu.

// A matrix occupies HC*HC lanes; each wave handles 4 HC=4 matrices.
// All reductions stay within that group, with FP32 state and arithmetic.
template<int HC>
__device__ __forceinline__ float sk_row_sum(float x, int e) {
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < HC; j++) s += __shfl(x, (e / HC) * HC + j, HC * HC);
    return s;
}
template<int HC>
__device__ __forceinline__ float sk_col_sum(float x, int e) {
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < HC; j++) s += __shfl(x, j * HC + e % HC, HC * HC);
    return s;
}
template<typename T, int HC>
__global__ void sinkhorn_wave_fwd(const T* input, T* output, float* init, int N, int iters, float eps) {
    constexpr int NE  = HC * HC;
    const int64_t pos = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int e       = threadIdx.x % NE;
    const int64_t b   = pos / NE;
    float v = b < N ? to_float(input[pos]) : 0.f, mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < HC; j++) mx = fmaxf(mx, __shfl(v, (e / HC) * HC + j, NE));
    v = expf(v - mx);
    v = v / sk_row_sum<HC>(v, e) + eps;
    if (b < N) init[pos] = v;
    for (int t = 0; t < iters; t++) {
        v = v / (sk_row_sum<HC>(v, e) + eps);
        v = v / (sk_col_sum<HC>(v, e) + eps);
    }
    if (b < N) output[pos] = from_float<T>(v);
}

template<typename T>
static __global__ void sinkhorn_fwd_kernel(const T* __restrict__ input, T* __restrict__ output,
                                           float* __restrict__ M_init_out, int N_batch, int HC, int num_iters,
                                           float eps) {
    int bid = blockIdx.x;
    if (bid >= N_batch) return;

    int tid    = threadIdx.x;
    int ne     = HC * HC;
    int stride = ((ne + blockDim.x - 1) / blockDim.x) * blockDim.x;  // padded ne

    extern __shared__ float smem[];
    float* M   = smem;           // [stride]
    float* aux = smem + stride;  // [HC*3]: row_max, row_sum, col_sum

    float* row_max_buf = aux;
    float* row_sum_buf = aux + HC;
    float* col_sum_buf = aux + 2 * HC;

    // Load input into shared memory
    for (int i = tid; i < ne; i += blockDim.x) M[i] = to_float(input[(size_t)bid * ne + i]);
    __syncthreads();

    // --- softmax along rows ---
    if (tid < HC) {
        float mx = -INFINITY;
        for (int c = 0; c < HC; c++) {
            float v = M[tid * HC + c];
            if (v > mx) mx = v;
        }
        row_max_buf[tid] = mx;
    }
    __syncthreads();
    for (int i = tid; i < ne; i += blockDim.x) {
        int r = i / HC;
        M[i]  = expf(M[i] - row_max_buf[r]);
    }
    __syncthreads();
    if (tid < HC) {
        float s = 0.0f;
        for (int c = 0; c < HC; c++) s += M[tid * HC + c];
        row_sum_buf[tid] = s;
    }
    __syncthreads();
    // M = softmax_output + eps  (store as M_init)
    for (int i = tid; i < ne; i += blockDim.x) {
        int r = i / HC;
        M[i]  = M[i] / row_sum_buf[r] + eps;
    }
    __syncthreads();
    // Save M_init
    for (int i = tid; i < ne; i += blockDim.x) M_init_out[(size_t)bid * ne + i] = M[i];

    // --- Iterative row/col normalization ---
    for (int it = 0; it < num_iters; it++) {
        // Row normalization
        if (tid < HC) {
            float s = 0.0f;
            for (int c = 0; c < HC; c++) s += M[tid * HC + c];
            row_sum_buf[tid] = s;
        }
        __syncthreads();
        for (int i = tid; i < ne; i += blockDim.x) {
            int r = i / HC;
            M[i]  = M[i] / (row_sum_buf[r] + eps);
        }
        __syncthreads();

        // Column normalization
        if (tid < HC) {
            float s = 0.0f;
            for (int r = 0; r < HC; r++) s += M[r * HC + tid];
            col_sum_buf[tid] = s;
        }
        __syncthreads();
        for (int i = tid; i < ne; i += blockDim.x) {
            int c = i % HC;
            M[i]  = M[i] / (col_sum_buf[c] + eps);
        }
        __syncthreads();
    }

    // Store output
    for (int i = tid; i < ne; i += blockDim.x) output[(size_t)bid * ne + i] = from_float<T>(M[i]);
}

namespace ftrain {
namespace {

static bool checkLaunch(const char* what) {
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "%s launch failed: %s", what, hipGetErrorString(status));
    }
    return true;
}

static int sinkhornSharedFwd(int size) {
    const int elements = size * size;
    const int stride   = ((elements + 255) / 256) * 256;
    return (stride + size * 3) * static_cast<int>(sizeof(float));
}

template<typename T>
static bool launchSinkhornFwdTyped(hipStream_t stream, const void* logits, void* output, float* m_init, int batch,
                                   int size, int num_iterations, float eps) {
    // Preserve the original wave dispatch; short HC=4 batches use the CTA fallback.
    if (size == 8 || (size == 4 && (batch >= 1024 || num_iterations >= 10))) {
        const dim3 grid((static_cast<uint64_t>(batch) * size * size + 255) / 256);
        if (size == 4) {
            hipLaunchKernelGGL((sinkhorn_wave_fwd<T, 4>), grid, dim3(256), 0, stream, static_cast<const T*>(logits),
                               static_cast<T*>(output), m_init, batch, num_iterations, eps);
        } else {
            hipLaunchKernelGGL((sinkhorn_wave_fwd<T, 8>), grid, dim3(256), 0, stream, static_cast<const T*>(logits),
                               static_cast<T*>(output), m_init, batch, num_iterations, eps);
        }
        return checkLaunch("SinkhornFwd");
    }
    int block_dim = std::min(256, size * size);
    if (block_dim < 1) { block_dim = 1; }
    hipLaunchKernelGGL((sinkhorn_fwd_kernel<T>), dim3(batch), dim3(block_dim), sinkhornSharedFwd(size), stream,
                       reinterpret_cast<const T*>(logits), reinterpret_cast<T*>(output), m_init, batch, size,
                       num_iterations, eps);
    return checkLaunch("SinkhornFwd");
}
}  // namespace

bool launchBf16SinkhornFwd(FTrainStream stream, const void* logits, void* output, float* m_init, int batch, int size,
                           int num_iterations, float eps) {
    return launchSinkhornFwdTyped<hip_bfloat16>(stream, logits, output, m_init, batch, size, num_iterations, eps);
}

}  // namespace ftrain
