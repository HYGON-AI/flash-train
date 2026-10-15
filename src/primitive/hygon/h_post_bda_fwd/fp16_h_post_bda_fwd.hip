// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/h_post_bda_fwd/fp16_h_post_bda_fwd.hpp"

template<typename T>
__device__ __forceinline__ float to_float(T v);
template<>
__device__ __forceinline__ float to_float(float v) {
    return v;
}
template<>
__device__ __forceinline__ float to_float(__half v) {
    return __half2float(v);
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
__device__ __forceinline__ __half from_float<__half>(float v) {
    return __float2half(v);
}
template<>
__device__ __forceinline__ hip_bfloat16 from_float<hip_bfloat16>(float v) {
    return hip_bfloat16(v);
}

// Adapted from fused_mHC/csrc/h_post_bda.hip.

template<typename T, int N>
static __global__ void __launch_bounds__(256)
    h_post_bda_fwd_kernel(const T* __restrict__ h_res, const T* __restrict__ orig, const T* __restrict__ h_post,
                          const T* __restrict__ x, const float* __restrict__ bias, T* __restrict__ out, int SB, int C,
                          int has_bias) {
    int sb     = blockIdx.x;
    int tid    = threadIdx.x;
    int TILE_C = blockDim.x;
    int c      = blockIdx.y * TILE_C + tid;
    bool valid = (c < C);

    extern __shared__ float smem[];
    float* hr    = smem;          // N*N
    float* hp    = smem + N * N;  // N
    const int NN = N * N;
    for (int i = tid; i < NN; i += TILE_C) hr[i] = to_float(h_res[(size_t)sb * NN + i]);
    for (int i = tid; i < N; i += TILE_C) hp[i] = to_float(h_post[(size_t)sb * N + i]);
    __syncthreads();

    float xb = (valid ? to_float(x[(size_t)sb * C + c]) : 0.0f) + (valid && has_bias ? to_float(bias[c]) : 0.0f);

    float orig_n[N];
#pragma unroll
    for (int j = 0; j < N; j++) orig_n[j] = valid ? to_float(orig[(int64_t)sb * N * C + j * C + c]) : 0.0f;

#pragma unroll
    for (int n = 0; n < N; n++) {
        float acc = hp[n] * xb;
#pragma unroll
        for (int j = 0; j < N; j++) acc += hr[j * N + n] * orig_n[j];  // h_res^T[n,j] = h_res[j,n]
        if (valid) out[(int64_t)sb * N * C + n * C + c] = from_float<T>(acc);
    }
}

template<typename T>
static __global__ void __launch_bounds__(256)
    h_post_bda_fwd_generic_kernel(const T* __restrict__ h_res, const T* __restrict__ orig, const T* __restrict__ h_post,
                                  const T* __restrict__ x, const float* __restrict__ bias, T* __restrict__ out, int SB,
                                  int N, int C, int has_bias) {
    int sb     = blockIdx.x;
    int tid    = threadIdx.x;
    int TILE_C = blockDim.x;
    int c      = blockIdx.y * TILE_C + tid;
    bool valid = (c < C);

    extern __shared__ float smem[];
    float* hr = smem;          // N*N
    float* hp = smem + N * N;  // N
    for (int i = tid; i < N * N; i += TILE_C) hr[i] = to_float(h_res[(size_t)sb * N * N + i]);
    for (int i = tid; i < N; i += TILE_C) hp[i] = to_float(h_post[(size_t)sb * N + i]);
    __syncthreads();

    float xb = (valid ? to_float(x[(size_t)sb * C + c]) : 0.0f) + (valid && has_bias ? to_float(bias[c]) : 0.0f);

    constexpr int MAX_N = 32;
    float orig_n[MAX_N];
    for (int j = 0; j < N && j < MAX_N; j++) orig_n[j] = valid ? to_float(orig[(int64_t)sb * N * C + j * C + c]) : 0.0f;

    for (int n = 0; n < N; n++) {
        float acc = hp[n] * xb;
        for (int j = 0; j < N && j < MAX_N; j++) acc += hr[j * N + n] * orig_n[j];
        if (valid) out[(int64_t)sb * N * C + n * C + c] = from_float<T>(acc);
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

template<typename T, int N, int VEC>
__global__ void __launch_bounds__(256)
    h_post_bda_fwd_v2_kernel(const T* __restrict__ h_res, const T* __restrict__ orig, const T* __restrict__ h_post,
                             const T* __restrict__ x, const float* __restrict__ bias, T* __restrict__ out, int SB,
                             int C, int has_bias) {
    int sb = blockIdx.x;
    if (sb >= SB) return;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* hr    = smem;          // N*N
    float* hp    = smem + N * N;  // N
    const int NN = N * N;
    for (int i = tid; i < NN; i += blockDim.x) hr[i] = to_float(h_res[(size_t)sb * NN + i]);
    for (int i = tid; i < N; i += blockDim.x) hp[i] = to_float(h_post[(size_t)sb * N + i]);
    __syncthreads();

    const int stride      = blockDim.x * VEC;
    const T* orig_base    = orig + (size_t)sb * N * C;
    const T* x_row        = x + (size_t)sb * C;
    const float* bias_row = (has_bias ? bias : nullptr);
    T* out_base           = out + (size_t)sb * N * C;
    // Vectorize the last partial CTA iteration; only a partial vector
    // needs scalar handling (launchers guarantee C % VEC == 0).
    const int tail0 = (C / VEC) * VEC;

    for (int c0 = tid * VEC; c0 < tail0; c0 += stride) {
        float xb[VEC];
        mhc_load16<T, VEC>(xb, x_row + c0);
        if (has_bias) {
#pragma unroll
            for (int v = 0; v < VEC; v++) xb[v] += bias_row[c0 + v];
        }
        float orig_n[N][VEC];
#pragma unroll
        for (int j = 0; j < N; j++) mhc_load16<T, VEC>(orig_n[j], orig_base + (size_t)j * C + c0);
#pragma unroll
        for (int n = 0; n < N; n++) {
            float acc[VEC];
#pragma unroll
            for (int v = 0; v < VEC; v++) acc[v] = hp[n] * xb[v];
#pragma unroll
            for (int j = 0; j < N; j++) {
                const float h = hr[j * N + n];
#pragma unroll
                for (int v = 0; v < VEC; v++) acc[v] += h * orig_n[j][v];
            }
            mhc_store16<T, VEC>(out_base + (size_t)n * C + c0, acc);
        }
    }
    // Scalar tail.
    for (int c = tail0 + tid; c < C; c += blockDim.x) {
        const float xbv = to_float(x_row[c]) + (has_bias ? to_float(bias_row[c]) : 0.0f);
        float orv[N];
#pragma unroll
        for (int j = 0; j < N; j++) orv[j] = to_float(orig_base[(size_t)j * C + c]);
#pragma unroll
        for (int n = 0; n < N; n++) {
            float acc = hp[n] * xbv;
#pragma unroll
            for (int j = 0; j < N; j++) acc += hr[j * N + n] * orv[j];
            out_base[(size_t)n * C + c] = from_float<T>(acc);
        }
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
static bool launchHPostBdaFwdTyped(hipStream_t stream, const void* h_res, const void* residual, const void* h_post,
                                   const void* x, const float* bias, void* output, int batch, int num_streams,
                                   int channels, bool has_bias) {
    if (channels % 8 == 0 && (num_streams == 2 || num_streams == 4 || num_streams == 8 || num_streams == 16)) {
        constexpr int VEC = 16 / sizeof(T);
        const int shared  = (num_streams * num_streams + num_streams) * sizeof(float);
#define HPB_FWD(N)                                                                                                 \
    case N:                                                                                                        \
        hipLaunchKernelGGL((h_post_bda_fwd_v2_kernel<T, N, VEC>), dim3(batch), dim3(256), shared, stream,          \
                           static_cast<const T*>(h_res), static_cast<const T*>(residual),                          \
                           static_cast<const T*>(h_post), static_cast<const T*>(x), bias, static_cast<T*>(output), \
                           batch, channels, int(has_bias));                                                        \
        break
        switch (num_streams) {
            HPB_FWD(2);
            HPB_FWD(4);
            HPB_FWD(8);
            HPB_FWD(16);
        }
#undef HPB_FWD
        return checkLaunch("HPostBdaFwd");
    }
    constexpr int kTileChannels = 256;
    const int shared            = (num_streams * num_streams + num_streams) * static_cast<int>(sizeof(float));
    const dim3 grid(batch, (channels + kTileChannels - 1) / kTileChannels);
    const dim3 block(kTileChannels);
    const int has_bias_flag = has_bias ? 1 : 0;
    switch (num_streams) {
        case 2:
            hipLaunchKernelGGL((h_post_bda_fwd_kernel<T, 2>), grid, block, shared, stream,
                               reinterpret_cast<const T*>(h_res), reinterpret_cast<const T*>(residual),
                               reinterpret_cast<const T*>(h_post), reinterpret_cast<const T*>(x), bias,
                               reinterpret_cast<T*>(output), batch, channels, has_bias_flag);
            break;
        case 4:
            hipLaunchKernelGGL((h_post_bda_fwd_kernel<T, 4>), grid, block, shared, stream,
                               reinterpret_cast<const T*>(h_res), reinterpret_cast<const T*>(residual),
                               reinterpret_cast<const T*>(h_post), reinterpret_cast<const T*>(x), bias,
                               reinterpret_cast<T*>(output), batch, channels, has_bias_flag);
            break;
        case 8:
            hipLaunchKernelGGL((h_post_bda_fwd_kernel<T, 8>), grid, block, shared, stream,
                               reinterpret_cast<const T*>(h_res), reinterpret_cast<const T*>(residual),
                               reinterpret_cast<const T*>(h_post), reinterpret_cast<const T*>(x), bias,
                               reinterpret_cast<T*>(output), batch, channels, has_bias_flag);
            break;
        case 16:
            hipLaunchKernelGGL((h_post_bda_fwd_kernel<T, 16>), grid, block, shared, stream,
                               reinterpret_cast<const T*>(h_res), reinterpret_cast<const T*>(residual),
                               reinterpret_cast<const T*>(h_post), reinterpret_cast<const T*>(x), bias,
                               reinterpret_cast<T*>(output), batch, channels, has_bias_flag);
            break;
        default:
            hipLaunchKernelGGL((h_post_bda_fwd_generic_kernel<T>), grid, block, shared, stream,
                               reinterpret_cast<const T*>(h_res), reinterpret_cast<const T*>(residual),
                               reinterpret_cast<const T*>(h_post), reinterpret_cast<const T*>(x), bias,
                               reinterpret_cast<T*>(output), batch, num_streams, channels, has_bias_flag);
            break;
    }
    return checkLaunch("HPostBdaFwd");
}
}  // namespace

bool launchFp16HPostBdaFwd(FTrainStream stream, const void* h_res, const void* residual, const void* h_post,
                           const void* x, const float* bias, void* output, int batch_stride, int num_streams,
                           int channels, bool has_bias) {
    return launchHPostBdaFwdTyped<__half>(stream, h_res, residual, h_post, x, bias, output, batch_stride, num_streams,
                                          channels, has_bias);
}

}  // namespace ftrain
