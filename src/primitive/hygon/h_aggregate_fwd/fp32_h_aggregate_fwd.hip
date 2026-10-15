// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/h_aggregate_fwd/fp32_h_aggregate_fwd.hpp"

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
static __global__ void h_aggregate_fwd_v2_kernel(const T* __restrict__ x, const float* __restrict__ h_pre,
                                                 T* __restrict__ out, int SB, int N, int C) {
    int sb = blockIdx.x;
    if (sb >= SB) return;
    int tid    = threadIdx.x;
    int TILE_C = blockDim.x;
    int c_tile = blockIdx.y;
    int c      = c_tile * TILE_C + tid;

    extern __shared__ float h_shared[];
    for (int n = tid; n < N; n += TILE_C) h_shared[n] = to_float(h_pre[(size_t)sb * N + n]);
    __syncthreads();
    if (c >= C) return;

    float acc = 0.0f;
    for (int n = 0; n < N; n++) acc += h_shared[n] * to_float(x[(size_t)sb * N * C + n * C + c]);

    out[(size_t)sb * C + c] = from_float<T>(acc);
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
    h_aggregate_fwd_v3_kernel(const T* __restrict__ x, const float* __restrict__ h_pre, T* __restrict__ out, int SB,
                              int N, int C) {
    int sb = blockIdx.x;
    if (sb >= SB) return;
    int tid = threadIdx.x;

    extern __shared__ float h_shared[];
    for (int n = tid; n < N; n += blockDim.x) h_shared[n] = to_float(h_pre[(size_t)sb * N + n]);
    __syncthreads();

    const int stride = blockDim.x * VEC;
    const T* x_base  = x + (size_t)sb * N * C;
    T* out_row       = out + (size_t)sb * C;
    // Vectorize the last partial CTA iteration; only a partial vector
    // needs scalar handling (launchers guarantee C % VEC == 0).
    const int tail0 = (C / VEC) * VEC;

    for (int c0 = tid * VEC; c0 < tail0; c0 += stride) {
        float acc[VEC];
#pragma unroll
        for (int v = 0; v < VEC; v++) acc[v] = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++) {
            float xv[VEC];
            mhc_load16<T, VEC>(xv, x_base + (size_t)n * C + c0);
            const float h = h_shared[n];
#pragma unroll
            for (int v = 0; v < VEC; v++) acc[v] += h * xv[v];
        }
        mhc_store16<T, VEC>(out_row + c0, acc);
    }
    // Scalar tail (C not a multiple of VEC).
    for (int c = tail0 + tid; c < C; c += blockDim.x) {
        float acc = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++) acc += h_shared[n] * to_float(x_base[(size_t)n * C + c]);
        out_row[c] = from_float<T>(acc);
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
static bool launchHAggregateFwdTyped(hipStream_t stream, const void* x, const float* h_pre, void* aggregated, int batch,
                                     int num_streams, int channels) {
    if (channels % 8 == 0) {
        constexpr int VEC = 16 / sizeof(T);
        hipLaunchKernelGGL((h_aggregate_fwd_v3_kernel<T, VEC>), dim3(batch), dim3(256), num_streams * sizeof(float),
                           stream, static_cast<const T*>(x), h_pre, static_cast<T*>(aggregated), batch, num_streams,
                           channels);
        return checkLaunch("HAggregateFwd");
    }

    constexpr int kTileChannels = 256;
    const dim3 grid(batch, (channels + kTileChannels - 1) / kTileChannels);
    hipLaunchKernelGGL((h_aggregate_fwd_v2_kernel<T>), grid, dim3(kTileChannels),
                       num_streams * static_cast<int>(sizeof(float)), stream, reinterpret_cast<const T*>(x), h_pre,
                       reinterpret_cast<T*>(aggregated), batch, num_streams, channels);
    return checkLaunch("HAggregateFwd");
}
}  // namespace

bool launchFp32HAggregateFwd(FTrainStream stream, const void* x, const float* h_pre, void* aggregated, int batch_stride,
                             int num_streams, int channels) {
    return launchHAggregateFwdTyped<float>(stream, x, h_pre, aggregated, batch_stride, num_streams, channels);
}

}  // namespace ftrain
