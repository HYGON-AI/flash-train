// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/h_post_bda_bwd/fp32_h_post_bda_bwd.hpp"

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

// Adapted from fused_mHC/csrc/h_post_bda.hip.

template<typename T>
__device__ __forceinline__ static void hpb_bwd_gxorig_generic_body(int sb, int ct, int tid, int nt, float* smem,
                                                                   const T* __restrict__ go,
                                                                   const T* __restrict__ h_res,
                                                                   const T* __restrict__ h_post, T* __restrict__ g_orig,
                                                                   T* __restrict__ g_x, int N, int C) {
    int c      = ct * nt + tid;
    bool valid = (c < C);

    float* hr = smem;          // N*N
    float* hp = smem + N * N;  // N
    for (int i = tid; i < N * N; i += nt) hr[i] = to_float(h_res[(size_t)sb * N * N + i]);
    for (int i = tid; i < N; i += nt) hp[i] = to_float(h_post[(size_t)sb * N + i]);
    __syncthreads();

    constexpr int MAX_N = 32;
    float go_n[MAX_N];
    for (int n = 0; n < N && n < MAX_N; n++) go_n[n] = valid ? to_float(go[(int64_t)sb * N * C + n * C + c]) : 0.0f;

    float gx = 0.0f;
    for (int n = 0; n < N && n < MAX_N; n++) gx += hp[n] * go_n[n];
    if (valid) g_x[(size_t)sb * C + c] = from_float<T>(gx);

    for (int i = 0; i < N; i++) {
        float gi = 0.0f;
        for (int j = 0; j < N && j < MAX_N; j++) gi += hr[i * N + j] * go_n[j];
        if (valid) g_orig[(int64_t)sb * N * C + i * C + c] = from_float<T>(gi);
    }
}

template<typename T>
__device__ __forceinline__ static void hpb_bwd_ghphr_fallback_body(int sb, int tid, int tile, float* buf,
                                                                   const T* __restrict__ go, const T* __restrict__ orig,
                                                                   const T* __restrict__ x,
                                                                   const float* __restrict__ bias, T* __restrict__ g_hr,
                                                                   T* __restrict__ g_hp, int N, int C, int has_bias) {
    // Shared memory layout:
    //   ghp_buf[tid*N + n]    — partial g_hp per thread
    //   ghr_buf[tid*N*N + ij] — partial g_hr per thread
    float* ghp_buf = buf;             // tile * N
    float* ghr_buf = buf + tile * N;  // tile * N * N

    for (int i = tid; i < tile * (N + N * N); i += tile) buf[i] = 0.0f;
    __syncthreads();

    for (int c_start = 0; c_start < C; c_start += tile) {
        int c = c_start + tid;
        if (c >= C) continue;

        float xb = to_float(x[(size_t)sb * C + c]) + (has_bias ? to_float(bias[c]) : 0.0f);

        for (int n = 0; n < N; n++) {
            float gon             = to_float(go[(size_t)sb * N * C + n * C + c]);
            ghp_buf[tid * N + n] += gon * xb;

            for (int i = 0; i < N; i++) {
                float orig_i                      = to_float(orig[(size_t)sb * N * C + i * C + c]);
                ghr_buf[tid * N * N + i * N + n] += orig_i * gon;
            }
        }
    }
    __syncthreads();

    // Reduce g_hp
    for (int n = 0; n < N; n++) {
        for (int stride = tile / 2; stride > 0; stride /= 2) {
            if (tid < stride) ghp_buf[tid * N + n] += ghp_buf[(tid + stride) * N + n];
            __syncthreads();
        }
        if (tid == 0) g_hp[(size_t)sb * N + n] = from_float<T>(ghp_buf[n]);
    }

    // Reduce g_hr
    for (int ij = 0; ij < N * N; ij++) {
        for (int stride = tile / 2; stride > 0; stride /= 2) {
            if (tid < stride) ghr_buf[tid * N * N + ij] += ghr_buf[(tid + stride) * N * N + ij];
            __syncthreads();
        }
        if (tid == 0) g_hr[(size_t)sb * N * N + ij] = from_float<T>(ghr_buf[ij]);
    }
}

template<typename T>
static __global__ void __launch_bounds__(256)
    h_post_bda_bwd_fused_generic_kernel(const T* __restrict__ go, const T* __restrict__ orig, const T* __restrict__ x,
                                        const float* __restrict__ bias, const T* __restrict__ h_res,
                                        const T* __restrict__ h_post, T* __restrict__ g_orig, T* __restrict__ g_x,
                                        T* __restrict__ g_hr, T* __restrict__ g_hp, int SB, int N, int C, int has_bias,
                                        int CTILES, int TILE_B) {
    int bx  = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];
    int nA = SB * CTILES;
    if (bx < nA) {
        int sb = bx / CTILES;
        int ct = bx % CTILES;
        hpb_bwd_gxorig_generic_body<T>(sb, ct, tid, blockDim.x, smem, go, h_res, h_post, g_orig, g_x, N, C);
    } else {
        int sb = bx - nA;
        if (tid < TILE_B)
            hpb_bwd_ghphr_fallback_body<T>(sb, tid, TILE_B, smem, go, orig, x, bias, g_hr, g_hp, N, C, has_bias);
    }
}

template<typename T>
static __global__ void __launch_bounds__(256)
    h_post_bda_bias_grad_kernel(const T* __restrict__ go, const T* __restrict__ hp, float* __restrict__ g_bias, int SB,
                                int N, int C) {
    if (C >= 128 && SB >= 256) {
        // Adjacent lanes read adjacent columns.  Four independent sums
        // expose load parallelism through the long batch reduction.
        const int col = blockIdx.x * 16 + (threadIdx.x & 15);
        const int row = threadIdx.x >> 4;
        float sums[4] = {0.f, 0.f, 0.f, 0.f};
        for (int r = row; r < SB; r += 64) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (col < C && r + j * 16 < SB) {
                    const size_t base = (size_t)(r + j * 16) * N;
                    for (int n = 0; n < N; ++n) sums[j] += to_float(go[(base + n) * C + col]) * to_float(hp[base + n]);
                }
            }
        }
        __shared__ float partial[256];
        partial[threadIdx.x] = (sums[0] + sums[1]) + (sums[2] + sums[3]);
        __syncthreads();
        if (threadIdx.x < 16 && col < C) {
            float v = 0.f;
#pragma unroll
            for (int j = 0; j < 16; j++) v += partial[threadIdx.x + j * 16];
            g_bias[col] = v;
        }
        return;
    }

    // One block reduces the SB dimension for one bias column in fp32.
    const int c = blockIdx.x;
    if (c >= C) return;
    float sum = 0.0f;
    // Reduce the unrounded go * h_post products, not the low-precision g_x output.
    for (int sb = threadIdx.x; sb < SB; sb += blockDim.x)
        for (int n = 0; n < N; ++n) {
            const size_t row  = (size_t)sb * N + n;
            sum              += to_float(go[row * C + c]) * to_float(hp[row]);
        }
    __shared__ float red[256];
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) g_bias[c] = red[0];
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
__device__ __forceinline__ void hpb_bwd_gxorig_v2_body(int sb, int tid, int nt, float* smem, const T* __restrict__ go,
                                                       const T* __restrict__ h_res, const T* __restrict__ h_post,
                                                       T* __restrict__ g_orig, T* __restrict__ g_x, int C) {
    float* hr    = smem;          // N*N
    float* hp    = smem + N * N;  // N
    const int NN = N * N;
    for (int i = tid; i < NN; i += nt) hr[i] = to_float(h_res[(size_t)sb * NN + i]);
    for (int i = tid; i < N; i += nt) hp[i] = to_float(h_post[(size_t)sb * N + i]);
    __syncthreads();

    const int stride = nt * VEC;
    const T* go_base = go + (size_t)sb * N * C;
    T* g_orig_base   = g_orig + (size_t)sb * N * C;
    T* g_x_row       = g_x + (size_t)sb * C;
    // Vectorize the last partial CTA iteration; only a partial vector
    // needs scalar handling (launchers guarantee C % VEC == 0).
    const int tail0 = (C / VEC) * VEC;

    for (int c0 = tid * VEC; c0 < tail0; c0 += stride) {
        float go_n[N][VEC];
#pragma unroll
        for (int n = 0; n < N; n++) mhc_load16<T, VEC>(go_n[n], go_base + (size_t)n * C + c0);

        float gx[VEC];
#pragma unroll
        for (int v = 0; v < VEC; v++) gx[v] = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++) {
            const float h = hp[n];
#pragma unroll
            for (int v = 0; v < VEC; v++) gx[v] += h * go_n[n][v];
        }
        mhc_store16<T, VEC>(g_x_row + c0, gx);

#pragma unroll
        for (int i = 0; i < N; i++) {
            float gi[VEC];
#pragma unroll
            for (int v = 0; v < VEC; v++) gi[v] = 0.0f;
#pragma unroll
            for (int j = 0; j < N; j++) {
                const float h = hr[i * N + j];
#pragma unroll
                for (int v = 0; v < VEC; v++) gi[v] += h * go_n[j][v];
            }
            mhc_store16<T, VEC>(g_orig_base + (size_t)i * C + c0, gi);
        }
    }
    // Scalar tail.
    for (int c = tail0 + tid; c < C; c += nt) {
        float gon[N];
#pragma unroll
        for (int n = 0; n < N; n++) gon[n] = to_float(go_base[(size_t)n * C + c]);
        float gx = 0.0f;
#pragma unroll
        for (int n = 0; n < N; n++) gx += hp[n] * gon[n];
        g_x_row[c] = from_float<T>(gx);
#pragma unroll
        for (int i = 0; i < N; i++) {
            float gi = 0.0f;
#pragma unroll
            for (int j = 0; j < N; j++) gi += hr[i * N + j] * gon[j];
            g_orig_base[(size_t)i * C + c] = from_float<T>(gi);
        }
    }
}

template<typename T, int N>
__device__ __forceinline__ void hpb_bwd_gxorig_tile_body(int sb, int ct, int tid, int nt, float* smem,
                                                         const T* __restrict__ go, const T* __restrict__ h_res,
                                                         const T* __restrict__ h_post, T* __restrict__ g_orig,
                                                         T* __restrict__ g_x, int C) {
    int c      = ct * nt + tid;
    bool valid = (c < C);

    float* hr    = smem;          // N*N
    float* hp    = smem + N * N;  // N
    const int NN = N * N;
    for (int i = tid; i < NN; i += nt) hr[i] = to_float(h_res[(size_t)sb * NN + i]);
    for (int i = tid; i < N; i += nt) hp[i] = to_float(h_post[(size_t)sb * N + i]);
    __syncthreads();

    float go_n[N];
#pragma unroll
    for (int n = 0; n < N; n++) go_n[n] = valid ? to_float(go[(int64_t)sb * N * C + n * C + c]) : 0.0f;

    // g_x
    float gx = 0.0f;
#pragma unroll
    for (int n = 0; n < N; n++) gx += hp[n] * go_n[n];
    if (valid) g_x[(size_t)sb * C + c] = from_float<T>(gx);

// g_orig
#pragma unroll
    for (int i = 0; i < N; i++) {
        float gi = 0.0f;
#pragma unroll
        for (int j = 0; j < N; j++) gi += hr[i * N + j] * go_n[j];
        if (valid) g_orig[(int64_t)sb * N * C + i * C + c] = from_float<T>(gi);
    }
}

template<typename T, int N, int JCHUNK>
__device__ __forceinline__ void hpb_bwd_ghphr_v6_body(int sb, int jc, int lane, const T* __restrict__ go,
                                                      const T* __restrict__ orig, const T* __restrict__ x,
                                                      const float* __restrict__ bias, T* __restrict__ g_hr,
                                                      T* __restrict__ g_hp, int C, int has_bias) {
    constexpr int CH = 64;
    int j0           = jc * JCHUNK;
    int ncc          = (C + CH - 1) / CH;

    const T* go_sb   = go + (int64_t)sb * N * C;
    const T* orig_sb = orig + (int64_t)sb * N * C;
    const T* x_sb    = x + (int64_t)sb * C;

    float ghr[N * JCHUNK];
    float ghp[JCHUNK];
#pragma unroll
    for (int m = 0; m < N * JCHUNK; m++) ghr[m] = 0.0f;
#pragma unroll
    for (int k = 0; k < JCHUNK; k++) ghp[k] = 0.0f;

    for (int cc = 0; cc < ncc; cc++) {
        int c      = cc * CH + lane;
        bool valid = (c < C);
        float xb   = valid ? to_float(x_sb[c]) + (has_bias ? to_float(bias[c]) : 0.0f) : 0.0f;

        float goj[JCHUNK];
#pragma unroll
        for (int k = 0; k < JCHUNK; k++) goj[k] = valid ? to_float(go_sb[(j0 + k) * C + c]) : 0.0f;

#pragma unroll
        for (int k = 0; k < JCHUNK; k++) ghp[k] += goj[k] * xb;

#pragma unroll
        for (int i = 0; i < N; i++) {
            float oi = valid ? to_float(orig_sb[i * C + c]) : 0.0f;
#pragma unroll
            for (int k = 0; k < JCHUNK; k++) ghr[i * JCHUNK + k] += oi * goj[k];
        }
    }

#pragma unroll
    for (int k = 0; k < JCHUNK; k++) {
        float v = ghp[k];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off, 64);
        if (lane == 0) g_hp[(int64_t)sb * N + (j0 + k)] = from_float<T>(v);
    }
#pragma unroll
    for (int m = 0; m < N * JCHUNK; m++) {
        float v = ghr[m];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off, 64);
        if (lane == 0) {
            int i                                        = m / JCHUNK;
            int k                                        = m % JCHUNK;
            g_hr[(int64_t)sb * N * N + i * N + (j0 + k)] = from_float<T>(v);
        }
    }
}

template<typename T, int N, int VEC, int JCHUNK>
__global__ void __launch_bounds__(256)
    h_post_bda_bwd_fused_v2_kernel(const T* __restrict__ go, const T* __restrict__ orig, const T* __restrict__ x,
                                   const float* __restrict__ bias, const T* __restrict__ h_res,
                                   const T* __restrict__ h_post, T* __restrict__ g_orig, T* __restrict__ g_x,
                                   T* __restrict__ g_hr, T* __restrict__ g_hp, int SB, int C, int has_bias,
                                   int JCHUNKS) {
    int bx  = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];
    if (bx < SB) {
        hpb_bwd_gxorig_v2_body<T, N, VEC>(bx, tid, blockDim.x, smem, go, h_res, h_post, g_orig, g_x, C);
    } else {
        const int waves = SB >= 256 ? 4 : 1;
        int off         = (bx - SB) * waves + (tid >> 6);
        int sb          = off / JCHUNKS;
        int jc          = off % JCHUNKS;
        if ((tid >> 6) < waves && sb < SB)
            hpb_bwd_ghphr_v6_body<T, N, JCHUNK>(sb, jc, tid & 63, go, orig, x, bias, g_hr, g_hp, C, has_bias);
    }
}

template<typename T, int N, int JCHUNK>
__global__ void __launch_bounds__(256)
    h_post_bda_bwd_fused_tile_kernel(const T* __restrict__ go, const T* __restrict__ orig, const T* __restrict__ x,
                                     const float* __restrict__ bias, const T* __restrict__ h_res,
                                     const T* __restrict__ h_post, T* __restrict__ g_orig, T* __restrict__ g_x,
                                     T* __restrict__ g_hr, T* __restrict__ g_hp, int SB, int C, int has_bias,
                                     int JCHUNKS, int CTILES) {
    int bx  = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];
    int nA = SB * CTILES;
    if (bx < nA) {
        int sb = bx / CTILES;
        int ct = bx % CTILES;
        hpb_bwd_gxorig_tile_body<T, N>(sb, ct, tid, blockDim.x, smem, go, h_res, h_post, g_orig, g_x, C);
    } else {
        const int waves = SB >= 256 ? 4 : 1;
        int off         = (bx - nA) * waves + (tid >> 6);
        int sb          = off / JCHUNKS;
        int jc          = off % JCHUNKS;
        if ((tid >> 6) < waves && sb < SB)
            hpb_bwd_ghphr_v6_body<T, N, JCHUNK>(sb, jc, tid & 63, go, orig, x, bias, g_hr, g_hp, C, has_bias);
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

#define HPB_BWD_FUSED_V2_CASE(NN, JJ)                                                                                  \
    if (N == (NN) && jchunk == (JJ)) {                                                                                 \
        hipLaunchKernelGGL((h_post_bda_bwd_fused_v2_kernel<T, (NN), VEC, (JJ)>), grid, 256, smem, stream, go, orig, x, \
                           bias, hr, hp, g_orig, g_x, g_hr, g_hp, SB, C, has_bias, JCHUNKS);                           \
        return;                                                                                                        \
    }

#define HPB_BWD_FUSED_TILE_CASE(NN, JJ)                                                                             \
    if (N == (NN) && jchunk == (JJ)) {                                                                              \
        hipLaunchKernelGGL((h_post_bda_bwd_fused_tile_kernel<T, (NN), (JJ)>), grid, 256, smem, stream, go, orig, x, \
                           bias, hr, hp, g_orig, g_x, g_hr, g_hp, SB, C, has_bias, JCHUNKS, CTILES);                \
        return;                                                                                                     \
    }

template<typename T>
static void launch_hpb_bwd_fused(hipStream_t stream, const T* go, const T* orig, const T* x, const float* bias,
                                 const T* hr, const T* hp, T* g_orig, T* g_x, T* g_hr, T* g_hp, int SB, int N, int C,
                                 int has_bias) {
    constexpr int VEC = (sizeof(T) == 4) ? 4 : 8;
    const bool n_fast = (N == 2 || N == 4 || N == 8 || N == 16);
    if (n_fast) {
        int max_jcs = (N == 16) ? 16 : N;
        int want    = (int)((128 + SB - 1) / SB);
        if (want < 1) want = 1;
        if (want > max_jcs) want = max_jcs;
        if (N == 16 && want == 1) want = 2;
        while (N % want != 0) want--;
        const int JCHUNKS = want;
        const int jchunk  = N / JCHUNKS;
        const int smem    = (N * N + N) * (int)sizeof(float);
        if (C % 8 == 0) {
            const int waves = SB >= 256 ? 4 : 1;
            dim3 grid(SB + (SB * JCHUNKS + waves - 1) / waves);
            HPB_BWD_FUSED_V2_CASE(2, 1)
            HPB_BWD_FUSED_V2_CASE(2, 2) HPB_BWD_FUSED_V2_CASE(4, 1) HPB_BWD_FUSED_V2_CASE(4, 2)
                HPB_BWD_FUSED_V2_CASE(4, 4) HPB_BWD_FUSED_V2_CASE(8, 1) HPB_BWD_FUSED_V2_CASE(8, 2)
                    HPB_BWD_FUSED_V2_CASE(8, 4) HPB_BWD_FUSED_V2_CASE(8, 8) HPB_BWD_FUSED_V2_CASE(16, 1)
                        HPB_BWD_FUSED_V2_CASE(16, 2) HPB_BWD_FUSED_V2_CASE(16, 4) HPB_BWD_FUSED_V2_CASE(16, 8) return;
        }
        const int CTILES = (C + 255) / 256;
        const int waves  = SB >= 256 ? 4 : 1;
        dim3 grid(SB * CTILES + (SB * JCHUNKS + waves - 1) / waves);
        HPB_BWD_FUSED_TILE_CASE(2, 1)
        HPB_BWD_FUSED_TILE_CASE(2, 2) HPB_BWD_FUSED_TILE_CASE(4, 1) HPB_BWD_FUSED_TILE_CASE(4, 2)
            HPB_BWD_FUSED_TILE_CASE(4, 4) HPB_BWD_FUSED_TILE_CASE(8, 1) HPB_BWD_FUSED_TILE_CASE(8, 2)
                HPB_BWD_FUSED_TILE_CASE(8, 4) HPB_BWD_FUSED_TILE_CASE(8, 8) HPB_BWD_FUSED_TILE_CASE(16, 1)
                    HPB_BWD_FUSED_TILE_CASE(16, 2) HPB_BWD_FUSED_TILE_CASE(16, 4) HPB_BWD_FUSED_TILE_CASE(16, 8) return;
    }
    // Portable single launch for N not in {2,4,8,16}.
    const int CTILES = (C + 255) / 256;
    int TILE_B       = 64;
    while ((int64_t)TILE_B * N * (N + 1) * 4 > 65536) TILE_B /= 2;
    const int s1   = (N * N + N) * (int)sizeof(float);
    const int s2   = TILE_B * (N + N * N) * (int)sizeof(float);
    const int smem = s1 > s2 ? s1 : s2;
    dim3 grid(SB * CTILES + SB);
    hipLaunchKernelGGL((h_post_bda_bwd_fused_generic_kernel<T>), grid, 256, smem, stream, go, orig, x, bias, hr, hp,
                       g_orig, g_x, g_hr, g_hp, SB, N, C, has_bias, CTILES, TILE_B);
}

#undef HPB_BWD_FUSED_V2_CASE
#undef HPB_BWD_FUSED_TILE_CASE

template<typename T>
static bool launchHPostBdaBwdTyped(hipStream_t stream, const void* grad_output, const void* residual, const void* x,
                                   const float* bias, const void* h_res, const void* h_post, void* grad_residual,
                                   void* grad_x, void* grad_h_res, void* grad_h_post, float* grad_bias, int batch,
                                   int num_streams, int channels, bool has_bias) {
    launch_hpb_bwd_fused<T>(stream, static_cast<const T*>(grad_output), static_cast<const T*>(residual),
                            static_cast<const T*>(x), bias, static_cast<const T*>(h_res), static_cast<const T*>(h_post),
                            static_cast<T*>(grad_residual), static_cast<T*>(grad_x), static_cast<T*>(grad_h_res),
                            static_cast<T*>(grad_h_post), batch, num_streams, channels, int(has_bias));
    if (!checkLaunch("HPostBdaBwd")) { return false; }

    if (!has_bias || grad_bias == nullptr) { return true; }
    const int bias_blocks = (channels >= 128 && batch >= 256) ? (channels + 15) / 16 : channels;
    hipLaunchKernelGGL((h_post_bda_bias_grad_kernel<T>), dim3(bias_blocks), dim3(256), 0, stream,
                       reinterpret_cast<const T*>(grad_output), reinterpret_cast<const T*>(h_post), grad_bias, batch,
                       num_streams, channels);
    return checkLaunch("HPostBdaBiasGrad");
}
}  // namespace

bool launchFp32HPostBdaBwd(FTrainStream stream, const void* grad_output, const void* residual, const void* x,
                           const float* bias, const void* h_res, const void* h_post, void* grad_residual, void* grad_x,
                           void* grad_h_res, void* grad_h_post, float* grad_bias, int batch_stride, int num_streams,
                           int channels, bool has_bias) {
    return launchHPostBdaBwdTyped<float>(stream, grad_output, residual, x, bias, h_res, h_post, grad_residual, grad_x,
                                         grad_h_res, grad_h_post, grad_bias, batch_stride, num_streams, channels,
                                         has_bias);
}

}  // namespace ftrain
