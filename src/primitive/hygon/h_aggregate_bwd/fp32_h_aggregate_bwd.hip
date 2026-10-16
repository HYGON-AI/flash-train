// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/h_aggregate_bwd/fp32_h_aggregate_bwd.hpp"

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

// Adapted from fused_mHC/csrc/h_aggregate.hip.

template<typename T>
static __global__ void h_aggregate_bwd_kernel(const T* __restrict__ grad_out, const T* __restrict__ x,
                                              const float* __restrict__ h_pre, T* __restrict__ grad_x,
                                              float* __restrict__ grad_h, int SB, int N, int C) {
    int sb = blockIdx.x;
    if (sb >= SB) return;
    int tid    = threadIdx.x;
    int TILE_C = blockDim.x;

    extern __shared__ float buf[];
    float* h_shared = buf;  // N floats
    // Use TILE_C * N floats for reduction workspace
    float* reduce_buf = buf + N;  // TILE_C * N

    // Pre-load h_pre
    for (int n = tid; n < N; n += TILE_C) h_shared[n] = to_float(h_pre[(size_t)sb * N + n]);
    __syncthreads();

    // Each thread accumulates partial grad_h for each n across C
    // Store partials in reduce_buf
    for (int n = 0; n < N; n++) reduce_buf[tid * N + n] = 0.0f;
    __syncthreads();

    for (int c_start = 0; c_start < C; c_start += TILE_C) {
        int c = c_start + tid;
        if (c >= C) continue;

        float go = to_float(grad_out[(size_t)sb * C + c]);
        for (int n = 0; n < N; n++) {
            float xv                                = to_float(x[(size_t)sb * N * C + n * C + c]);
            grad_x[(size_t)sb * N * C + n * C + c]  = from_float<T>(go * h_shared[n]);
            reduce_buf[tid * N + n]                += go * xv;
        }
    }
    __syncthreads();

    // Parallel reduction for each n
    for (int n = 0; n < N; n++) {
        for (int stride = TILE_C / 2; stride > 0; stride /= 2) {
            if (tid < stride) { reduce_buf[tid * N + n] += reduce_buf[(tid + stride) * N + n]; }
            __syncthreads();
        }
        if (tid == 0) grad_h[(size_t)sb * N + n] = reduce_buf[n];
    }
}

template<typename T, int VEC>
__device__ __forceinline__ void mhc_load16(float* out, const T* src) {
    static_assert(sizeof(T) * VEC == 16, "16-byte vector width required");
    if constexpr (sizeof(T) == 4) {
        float4 v = *(const float4*)src;
        out[0]   = v.x;
        out[1]   = v.y;
        out[2]   = v.z;
        out[3]   = v.w;
    } else {
        float4 v   = *(const float4*)src;
        const T* p = (const T*)&v;
#pragma unroll
        for (int i = 0; i < 8; i++) out[i] = to_float(p[i]);
    }
}

template<typename T, int VEC>
__device__ __forceinline__ void mhc_store16(T* dst, const float* src) {
    static_assert(sizeof(T) * VEC == 16, "16-byte vector width required");
    if constexpr (sizeof(T) == 4) {
        *(float4*)dst = make_float4(src[0], src[1], src[2], src[3]);
    } else {
        float4 r;
        T* p = (T*)&r;
#pragma unroll
        for (int i = 0; i < 8; i++) p[i] = from_float<T>(src[i]);
        *(float4*)dst = r;
    }
}

template<typename T, int VEC>
__global__ void __launch_bounds__(256)
    h_aggregate_bwd_v3_kernel(const T* __restrict__ grad_out, const T* __restrict__ x, const float* __restrict__ h_pre,
                              T* __restrict__ grad_x, float* __restrict__ grad_h, int SB, int N, int C) {
    int sb = blockIdx.x;
    if (sb >= SB) return;
    int tid = threadIdx.x;

    extern __shared__ float buf[];
    float* h_shared   = buf;
    float* reduce_buf = buf + N;  // [blockDim.x * N]

    for (int n = tid; n < N; n += blockDim.x) h_shared[n] = to_float(h_pre[(size_t)sb * N + n]);
    for (int i = tid; i < blockDim.x * N; i += blockDim.x) reduce_buf[i] = 0.0f;
    __syncthreads();

    const int stride = blockDim.x * VEC;
    const T* x_base  = x + (size_t)sb * N * C;
    const T* go_row  = grad_out + (size_t)sb * C;
    T* gx_base       = grad_x + (size_t)sb * N * C;
    // Vectorize the last partial CTA iteration; only a partial vector
    // needs scalar handling (launchers guarantee C % VEC == 0).
    const int tail0 = (C / VEC) * VEC;

    for (int c0 = tid * VEC; c0 < tail0; c0 += stride) {
        float gov[VEC];
        mhc_load16<T, VEC>(gov, go_row + c0);
#pragma unroll
        for (int n = 0; n < N; n++) {
            float xv[VEC];
            mhc_load16<T, VEC>(xv, x_base + (size_t)n * C + c0);
            const float h = h_shared[n];
            float gv[VEC];
            float part = 0.0f;
#pragma unroll
            for (int v = 0; v < VEC; v++) {
                gv[v]  = gov[v] * h;
                part  += gov[v] * xv[v];
            }
            mhc_store16<T, VEC>(gx_base + (size_t)n * C + c0, gv);
            reduce_buf[tid * N + n] += part;
        }
    }
    for (int c = tail0 + tid; c < C; c += blockDim.x) {
        const float go = to_float(go_row[c]);
#pragma unroll
        for (int n = 0; n < N; n++) {
            const float xv              = to_float(x_base[(size_t)n * C + c]);
            gx_base[(size_t)n * C + c]  = from_float<T>(go * h_shared[n]);
            reduce_buf[tid * N + n]    += go * xv;
        }
    }
    __syncthreads();

// Tree reduction over threads for each n.
#pragma unroll
    for (int n = 0; n < N; n++) {
        for (int stride2 = blockDim.x / 2; stride2 > 0; stride2 >>= 1) {
            if (tid < stride2) reduce_buf[tid * N + n] += reduce_buf[(tid + stride2) * N + n];
            __syncthreads();
        }
        if (tid == 0) grad_h[(size_t)sb * N + n] = reduce_buf[n];
    }
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

template<typename T>
static bool launchHAggregateBwdTyped(hipStream_t stream, const void* grad_aggregated, const void* x, const float* h_pre,
                                     void* grad_x, float* grad_h_pre, int batch, int num_streams, int channels) {
    if (channels % 8 == 0 && (256 * num_streams + num_streams) * sizeof(float) <= 65536) {
        constexpr int VEC = 16 / sizeof(T);
        const int shared  = (256 * num_streams + num_streams) * sizeof(float);
        hipLaunchKernelGGL((h_aggregate_bwd_v3_kernel<T, VEC>), dim3(batch), dim3(256), shared, stream,
                           static_cast<const T*>(grad_aggregated), static_cast<const T*>(x), h_pre,
                           static_cast<T*>(grad_x), grad_h_pre, batch, num_streams, channels);
        return checkLaunch("HAggregateBwd");
    }

    constexpr int kTileChannels = 64;
    const int shared            = (kTileChannels * num_streams + num_streams) * static_cast<int>(sizeof(float));
    hipLaunchKernelGGL((h_aggregate_bwd_kernel<T>), dim3(batch), dim3(kTileChannels), shared, stream,
                       reinterpret_cast<const T*>(grad_aggregated), reinterpret_cast<const T*>(x), h_pre,
                       reinterpret_cast<T*>(grad_x), grad_h_pre, batch, num_streams, channels);
    return checkLaunch("HAggregateBwd");
}
}  // namespace

bool launchFp32HAggregateBwd(FTrainStream stream, const void* grad_aggregated, const void* x, const float* h_pre,
                             void* grad_x, float* grad_h_pre, int batch_stride, int num_streams, int channels) {
    return launchHAggregateBwdTyped<float>(stream, grad_aggregated, x, h_pre, grad_x, grad_h_pre, batch_stride,
                                           num_streams, channels);
}

}  // namespace ftrain
