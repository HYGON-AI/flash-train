// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sinkhorn_bwd/bf16_sinkhorn_bwd.hpp"

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
    // XOR partners stay within one matrix row; logarithmic wave reduction.
#pragma unroll
    for (int delta = 1; delta < HC; delta *= 2) x += __shfl_xor(x, delta, HC * HC);
    return x;
}
template<int HC>
__device__ __forceinline__ float sk_col_sum(float x, int e) {
    // Column partners differ only in the row bits of the matrix lane index.
#pragma unroll
    for (int delta = 1; delta < HC; delta *= 2) x += __shfl_xor(x, delta * HC, HC * HC);
    return x;
}
template<typename T, int HC, int IT>
__global__ void sinkhorn_wave_bwd(const T* go, const float* init, T* gi, int N, float eps) {
    constexpr int NE  = HC * HC;
    const int64_t pos = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int e       = threadIdx.x % NE;
    const int64_t b   = pos / NE;
    float v = b < N ? init[pos] : 1.f / HC, g = b < N ? to_float(go[pos]) : 0.f;
    const float soft = v - eps;
    float row[IT], col[IT], rs[IT], cs[IT];
#pragma unroll
    for (int t = 0; t < IT; t++) {
        rs[t]  = 1.f;
        if (t > 0) {
            rs[t] = sk_row_sum<HC>(v, e) + eps;
            v     = v / rs[t];
        }
        row[t] = v;
        cs[t]  = sk_col_sum<HC>(v, e) + eps;
        v      = v / cs[t];
        col[t] = v;
    }
#pragma unroll
    for (int j = 0; j < IT; j++) {
        int t = IT - 1 - j;
        g     = g / cs[t];
        g     = g - sk_col_sum<HC>(g * col[t], e);
        if (t > 0) {
            g = g / rs[t];
            g = g - sk_row_sum<HC>(g * row[t], e);
        }
    }
    g = soft * (g - sk_row_sum<HC>(g * soft, e));
    if (b < N) gi[pos] = from_float<T>(g);
}

template<typename T>
static __global__ void sinkhorn_bwd_kernel(const T* __restrict__ grad_output, const float* __restrict__ M_init,
                                           T* __restrict__ grad_input, int N_batch, int HC, int num_iters, float eps) {
    int bid = blockIdx.x;
    if (bid >= N_batch) return;

    int tid = threadIdx.x;
    int ne  = HC * HC;

    extern __shared__ float smem[];
    // Layout in smem:
    //   M            [ne]        — current M (working)
    //   grad         [ne]        — current gradient
    //   temp         [ne]        — temporary
    //   row_buf      [HC]        — row sum buffer
    //   col_buf      [HC]        — col sum buffer
    //   saved_states [...]       — iteration states (if they fit)
    //
    // saved_states layout (per iter t, 0..num_iters-1):
    //   row_sum[t]  : HC floats
    //   col_sum[t]  : HC floats
    //   M_before_row[t] : ne floats (M before row norm at iter t)
    //   M_before_col[t] : ne floats (M before col norm at iter t)
    //
    // Total saved: num_iters * (2*HC + 2*ne) floats
    // For HC=8, num_iters=20: 20*(16+128) = 2880 floats = 11.5 KB

    int saved_offset = ne * 3 + HC * 2;  // after M, grad, temp, row_buf, col_buf
    float* M_work    = smem;
    float* grad_work = smem + ne;
    float* temp      = smem + ne * 2;
    float* row_buf   = smem + ne * 3;
    float* col_buf   = smem + ne * 3 + HC;
    float* states    = smem + saved_offset;
    // states layout: [num_iters, 2*HC + 2*ne]

    // Load M = M_init and grad = grad_output
    for (int i = tid; i < ne; i += blockDim.x) {
        M_work[i]    = M_init[(size_t)bid * ne + i];
        grad_work[i] = to_float(grad_output[(size_t)bid * ne + i]);
    }
    __syncthreads();

    // ----- Forward recompute, saving states -----
    for (int t = 0; t < num_iters; t++) {
        // --- Row normalization ---
        // Save M_before_row[t]
        for (int i = tid; i < ne; i += blockDim.x) states[(t * (2 * HC + 2 * ne)) + 0 + i] = M_work[i];
        __syncthreads();

        if (tid < HC) {
            float s = 0.0f;
            for (int c = 0; c < HC; c++) s += M_work[tid * HC + c];
            states[(t * (2 * HC + 2 * ne)) + 2 * ne + tid] = s;  // row_sum[t]
        }
        __syncthreads();

        for (int i = tid; i < ne; i += blockDim.x) {
            int r     = i / HC;
            if (t > 0) M_work[i] = M_work[i] / (states[(t * (2 * HC + 2 * ne)) + 2 * ne + r] + eps);
        }
        __syncthreads();

        // --- Column normalization ---
        // Save M_before_col[t]
        for (int i = tid; i < ne; i += blockDim.x) states[(t * (2 * HC + 2 * ne)) + ne + i] = M_work[i];
        __syncthreads();

        if (tid < HC) {
            float s = 0.0f;
            for (int r = 0; r < HC; r++) s += M_work[r * HC + tid];
            states[(t * (2 * HC + 2 * ne)) + 2 * ne + HC + tid] = s;  // col_sum[t]
        }
        __syncthreads();

        for (int i = tid; i < ne; i += blockDim.x) {
            int c     = i % HC;
            M_work[i] = M_work[i] / (states[(t * (2 * HC + 2 * ne)) + 2 * ne + HC + c] + eps);
        }
        __syncthreads();
    }

    // ----- Backward pass (reverse order) -----
    for (int t_rev = 0; t_rev < num_iters; t_rev++) {
        int t = num_iters - 1 - t_rev;

        // Backward through column normalization
        // Load col_sum[t] and M_before_col[t]
        for (int i = tid; i < ne; i += blockDim.x)
            temp[i] = states[(t * (2 * HC + 2 * ne)) + ne + i];  // M_before_col[t]
        if (tid < HC) col_buf[tid] = states[(t * (2 * HC + 2 * ne)) + 2 * ne + HC + tid];
        __syncthreads();

        // grad /= (col_sum + eps)
        for (int i = tid; i < ne; i += blockDim.x) {
            int c        = i % HC;
            grad_work[i] = grad_work[i] / (col_buf[c] + eps);
        }
        __syncthreads();

        // col_corr[c] = sum_r(grad[r,c] * M_after_col[r,c])
        // M_after_col = M_before_col / (col_sum + eps).  The VJP of
        // y = x/S_c (S_c = col sum) is gx = g/S_c - sum_r(g*x)/S_c^2, i.e. a
        // per-COLUMN correction summed over rows; matches the Triton/cuTile
        // reference kernels (tl.sum(grad * M, axis=0)).
        if (tid < HC) {
            float corr = 0.0f;
            for (int r = 0; r < HC; r++) corr += grad_work[r * HC + tid] * (temp[r * HC + tid] / (col_buf[tid] + eps));
            col_buf[tid] = corr;
        }
        __syncthreads();

        // grad -= col_corr[None, :]
        for (int i = tid; i < ne; i += blockDim.x) {
            int c        = i % HC;
            grad_work[i] = grad_work[i] - col_buf[c];
        }
        __syncthreads();

        if (t > 0) {
            // Backward through row normalization
            // Load row_sum[t] and M_after_row[t] (= M_before_col[t], saved above)
            for (int i = tid; i < ne; i += blockDim.x)
                temp[i] = states[(t * (2 * HC + 2 * ne)) + ne + i];  // M_after_row[t]
            if (tid < HC) row_buf[tid] = states[(t * (2 * HC + 2 * ne)) + 2 * ne + tid];
            __syncthreads();

            // The initial softmax is differentiated after this loop; only
            // iterations t>0 contain an additional row normalization.
            for (int i = tid; i < ne; i += blockDim.x) {
                int r        = i / HC;
                grad_work[i] = grad_work[i] / (row_buf[r] + eps);
            }
            __syncthreads();

            // row_corr[r] = sum_c(grad[r,c] * M_after_row[r,c])
            // The VJP of y = x/S_r (S_r = row sum) is gx = g/S_r - sum_c(g*x)/S_r^2,
            // i.e. a per-ROW correction summed over columns; matches the
            // Triton/cuTile reference kernels (tl.sum(grad * M, axis=1)).
            if (tid < HC) {
                float corr = 0.0f;
                for (int c = 0; c < HC; c++) corr += grad_work[tid * HC + c] * temp[tid * HC + c];
                row_buf[tid] = corr;
            }
            __syncthreads();

            // grad -= row_corr[:, None]
            for (int i = tid; i < ne; i += blockDim.x) {
                int r        = i / HC;
                grad_work[i] = grad_work[i] - row_buf[r];
            }
            __syncthreads();
        }
    }

    // Megatron's first row normalization is the softmax itself. FT retains
    // softmax(logits)+eps as its FP32 saved state; reverse that softmax once,
    // after all column norms and the row norms from iterations t>0.
    // Compute softmax output = M_init - eps
    for (int i = tid; i < ne; i += blockDim.x) temp[i] = M_init[(size_t)bid * ne + i] - eps;  // softmax_output
    __syncthreads();

    // row_sum = sum(grad_work * softmax_output, dim=-1)
    if (tid < HC) {
        float s = 0.0f;
        for (int c = 0; c < HC; c++) s += grad_work[tid * HC + c] * temp[tid * HC + c];
        row_buf[tid] = s;
    }
    __syncthreads();

    // grad_logits = softmax_output * (grad_work - row_sum)
    for (int i = tid; i < ne; i += blockDim.x) {
        int r        = i / HC;
        float sm_out = temp[i];
        grad_work[i] = sm_out * (grad_work[i] - row_buf[r]);
    }
    __syncthreads();

    // Store grad_input
    for (int i = tid; i < ne; i += blockDim.x) grad_input[(size_t)bid * ne + i] = from_float<T>(grad_work[i]);
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

static int sinkhornSharedBwd(int size, int num_iterations) {
    const int elements = size * size;
    return (elements * 3 + size * 2 + num_iterations * (2 * size + 2 * elements)) * static_cast<int>(sizeof(float));
}

template<class Implementation, typename T, int SIZE, int ITERATIONS>
static bool launchSinkhornBwdTyped(hipStream_t stream, const void* grad_output, const float* m_init, void* grad_logits,
                                   int batch, int size, int num_iterations, float eps) {
    if constexpr (Implementation::kWave) {
        const unsigned threads = batch < 256 ? 64 : 256;
        const dim3 grid((static_cast<uint64_t>(batch) * size * size + threads - 1) / threads);
#define SK_BWD(H, I)                                                                 \
    hipLaunchKernelGGL((sinkhorn_wave_bwd<T, H, I>), grid, dim3(threads), 0, stream, \
                       static_cast<const T*>(grad_output), m_init, static_cast<T*>(grad_logits), batch, eps)
        SK_BWD(SIZE, ITERATIONS);
#undef SK_BWD
        return checkLaunch("SinkhornBwd");
    }
    int block_dim = std::min(256, size * size);
    if (block_dim < 1) { block_dim = 1; }
    const int shared = sinkhornSharedBwd(size, num_iterations);
    hipLaunchKernelGGL((sinkhorn_bwd_kernel<T>), dim3(batch), dim3(block_dim), shared, stream,
                       reinterpret_cast<const T*>(grad_output), m_init, reinterpret_cast<T*>(grad_logits), batch, size,
                       num_iterations, eps);
    return checkLaunch("SinkhornBwd");
}
}  // namespace

}  // namespace ftrain

namespace ftrain {

template<class Implementation, int SIZE, int ITERATIONS>
bool launchBf16SinkhornBwd(FTrainStream stream, const void* grad_output, const float* m_init, void* grad_logits,
                           int batch, int size, int num_iterations, float eps) {
    return launchSinkhornBwdTyped<Implementation, hip_bfloat16, SIZE, ITERATIONS>(
        stream, grad_output, m_init, grad_logits, batch, size, num_iterations, eps);
}

// Explicit fixed launch instances used by the C++ configuration code.
#define FTRAIN_INSTANTIATE_LAUNCH(Implementation, SIZE, ITERATIONS)                                                \
    template bool launchBf16SinkhornBwd<Implementation, SIZE, ITERATIONS>(                                         \
        FTrainStream stream, const void* grad_output, const float* m_init, void* grad_logits, int batch, int size, \
        int num_iterations, float eps);
#define FTRAIN_INSTANTIATE_LAUNCH_SET_1(Implementation, SIZE) \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, SIZE, 10);      \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, SIZE, 15);      \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, SIZE, 20);      \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, SIZE, 3);       \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, SIZE, 5);       \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, SIZE, 7);
FTRAIN_INSTANTIATE_LAUNCH(Bf16SinkhornBwd, 0, 0);
FTRAIN_INSTANTIATE_LAUNCH_SET_1(Bf16SinkhornBwdWave, 4);
FTRAIN_INSTANTIATE_LAUNCH_SET_1(Bf16SinkhornBwdWave, 8);
#undef FTRAIN_INSTANTIATE_LAUNCH_SET_1
#undef FTRAIN_INSTANTIATE_LAUNCH

}  // namespace ftrain
