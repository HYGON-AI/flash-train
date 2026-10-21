// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdlib>
#include <math.h>
#include <type_traits>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#    include <du_mma.h>
#endif

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/proj_rms_bwd/mixed_fp32_proj_rms_bwd.hpp"

// Adapted from the fused_dsa/fused_mHC kernel implementations.

#define PRMS_THREADS 256

template<typename T>
struct PrmsCfg {
    static constexpr int kfrag  = (sizeof(T) == 2) ? 16 : 8;
    static constexpr int kchunk = (sizeof(T) == 2) ? 128 : 64;  // fwd K chunk
    static constexpr int kld    = kchunk + 8;
    static constexpr int veclen = (sizeof(T) == 2) ? 8 : 4;  // vector copy width
};

template<typename T>
__device__ __forceinline__ float prms_to_float(T v);
template<>
__device__ __forceinline__ float prms_to_float(float v) {
    return v;
}
template<>
__device__ __forceinline__ float prms_to_float(__half v) {
    return __half2float(v);
}
template<>
__device__ __forceinline__ float prms_to_float(__hip_bfloat16 v) {
    return __bfloat162float(v);
}

template<typename T>
__device__ __forceinline__ T prms_from_float(float v);
template<>
__device__ __forceinline__ float prms_from_float(float v) {
    return v;
}
template<>
__device__ __forceinline__ __half prms_from_float<__half>(float v) {
    return __float2half(v);
}
template<>
__device__ __forceinline__ __hip_bfloat16 prms_from_float<__hip_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template<typename T>
__device__ __forceinline__ T prms_zero();
template<>
__device__ __forceinline__ float prms_zero<float>() {
    return 0.0f;
}
template<>
__device__ __forceinline__ __half prms_zero<__half>() {
    return __float2half(0.0f);
}
template<>
__device__ __forceinline__ __hip_bfloat16 prms_zero<__hip_bfloat16>() {
    return __float2bfloat16(0.0f);
}

template<typename T>
__device__ __forceinline__ static void prms_copy_vec(T* dst, const T* src) {
    if constexpr (sizeof(T) == 4) {
        *(float4*)dst = *(const float4*)src;
    } else {
        ((float2*)dst)[0] = ((const float2*)src)[0];
        ((float2*)dst)[1] = ((const float2*)src)[1];
    }
}

template<typename Dst, typename Src>
__device__ __forceinline__ static Dst prms_convert(Src v) {
    if constexpr (std::is_same<Dst, Src>::value) {
        return v;
    } else {
        return prms_from_float<Dst>(prms_to_float(v));
    }
}

template<typename Compute, typename Src>
__device__ __forceinline__ static float prms_to_compute_float(Src v) {
    return prms_to_float(prms_convert<Compute>(v));
}

template<typename Dst, typename Src, int VEC>
__device__ __forceinline__ static void prms_copy_convert_vec(Dst* dst, const Src* src) {
    if constexpr (std::is_same<Dst, Src>::value) {
        prms_copy_vec<Dst>(dst, src);
    } else {
#pragma unroll
        for (int i = 0; i < VEC; i++) dst[i] = prms_convert<Dst>(src[i]);
    }
}

template<typename T, int VEC>
__device__ __forceinline__ static void prms_load_vec(float* out, const T* src) {
    if constexpr (sizeof(T) == 4 && VEC == 4) {
        float4 v = *(const float4*)src;
        out[0]   = v.x;
        out[1]   = v.y;
        out[2]   = v.z;
        out[3]   = v.w;
    } else if constexpr (sizeof(T) == 2 && VEC == 8) {
        float4 v   = *(const float4*)src;
        const T* p = (const T*)&v;
#pragma unroll
        for (int i = 0; i < 8; i++) out[i] = prms_to_float(p[i]);
    } else {
#pragma unroll
        for (int i = 0; i < VEC; i++) out[i] = prms_to_float(src[i]);
    }
}

template<typename T, int VEC>
__device__ __forceinline__ static void prms_store_vec(T* dst, const float* src) {
    if constexpr (sizeof(T) == 4 && VEC == 4) {
        *(float4*)dst = make_float4(src[0], src[1], src[2], src[3]);
    } else if constexpr (sizeof(T) == 2 && VEC == 8) {
        float4 r;
        T* p = (T*)&r;
#pragma unroll
        for (int i = 0; i < 8; i++) p[i] = prms_from_float<T>(src[i]);
        *(float4*)dst = r;
    } else {
#pragma unroll
        for (int i = 0; i < VEC; i++) dst[i] = prms_from_float<T>(src[i]);
    }
}

template<typename T, typename W, typename KernelStorage, int MODE>
static __global__ void __launch_bounds__(256)
    proj_rms_bwd_fused_kernel(const T* __restrict__ grad_proj, const T* __restrict__ grad_r, const T* __restrict__ x,
                              const W* __restrict__ weight, const T* __restrict__ r_fwd, T* __restrict__ grad_x,
                              W* __restrict__ grad_w, int M, int N, int K, float eps) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    // ---- MODE 1: portable fallback (16x16 block semantics) ----
    const int TX = 16, TY = 16;
    const int nkx = (K + TX - 1) / TX;
    const int nbw = nkx * ((N + TY - 1) / TY);
    const int txx = tid % TX;  // k within block
    const int tyy = tid / TX;  // row/n within block

    if (bid < nbw) {
        const int nt = bid / nkx;
        const int kt = bid - nt * nkx;
        const int n  = nt * TY + tyy;
        const int k  = kt * TX + txx;
        if (n >= N || k >= K) return;
        float acc       = 0.0f;
        const T* gp_col = grad_proj + n;
        const T* x_col  = x + k;
        for (int m = 0; m < M; m++)
            acc += prms_to_compute_float<KernelStorage>(gp_col[(size_t)m * N]) *
                   prms_to_compute_float<KernelStorage>(x_col[(size_t)m * K]);
        grad_w[(size_t)n * K + k] = prms_from_float<W>(acc);
        return;
    }

    const int gx  = bid - nbw;
    const int kt  = gx % nkx;
    const int row = (gx / nkx) * TY + tyy;
    const int k   = kt * TX + txx;
    if (row >= M || k >= K) return;

    const float xv   = prms_to_float(x[(size_t)row * K + k]);
    const float rf   = fmaxf(prms_to_float(r_fwd[row]), 1e-10f);
    const float gr   = prms_to_float(grad_r[row]);
    const float norm = sqrtf((float)K) * (1.0f / rf - eps);

    float gp_acc    = 0.0f;
    const T* gp_row = grad_proj + (size_t)row * N;
    const W* w_col  = weight + k;
    for (int n = 0; n < N; n++)
        gp_acc += prms_to_compute_float<KernelStorage>(gp_row[n]) *
                  prms_to_compute_float<KernelStorage>(w_col[(size_t)n * K]);

    float dr_acc = 0.0f;
    if (norm > 0.0f) dr_acc = gr * (-rf * rf) * xv / (norm * sqrtf((float)K));
    grad_x[(size_t)row * K + k] = prms_from_float<T>(gp_acc + dr_acc);
}

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;

template<typename T, typename W, typename KernelStorage>
static __global__ void __launch_bounds__(PRMS_THREADS)
    proj_rms_bwd_fused_mma_kernel(const T* __restrict__ grad_proj, const T* __restrict__ grad_r,
                                  const T* __restrict__ x, const W* __restrict__ weight, const T* __restrict__ r_fwd,
                                  T* __restrict__ grad_x, W* __restrict__ grad_w, int M, int N, int K, float eps) {
    constexpr int VEC    = (sizeof(KernelStorage) == 4) ? 4 : 8;
    constexpr int KS_X   = 64 * VEC;   // grad_x k-width per block
    constexpr int MB_X   = 32;         // grad_x rows per block
    constexpr int GW_KT  = 16;         // grad_w k-cols per warp
    constexpr int GW_K   = 4 * GW_KT;  // grad_w k-cols per block (64)
    constexpr int MK     = PrmsCfg<KernelStorage>::kfrag;
    constexpr int M_STEP = 4 * MK;  // grad_w M rows per step
    // 16B-aligned padded LDS row stride that also drops the MMA
    // fragment-load bank conflicts from 16-way (stride 64/32 words-aligned)
    // to 2-way: bf16/fp16 rows stride 72 (36 words, bank = 4*row + col/2)
    // and fp32 rows stride 40 (bank = 8*row + col).  Only the first M_STEP
    // columns of each staging row are consumed by the MMAs; the tail is
    // padding.
    constexpr int GW_MS = (sizeof(KernelStorage) == 4) ? 40 : 72;
    constexpr int GW_NT = 16;  // grad_w n-rows per block
    const int bid       = blockIdx.x;
    const int tid       = threadIdx.x;

    extern __shared__ char smem_raw[];

    const int nkx     = K / KS_X;
    const int nbx     = nkx * ((M + MB_X - 1) / MB_X);
    const int nbt     = (N + GW_KT - 1) / GW_KT;
    const int kkblock = K / GW_K;
    const int nbw     = nbt * kkblock;

    if (bid < nbw) {
        const int nt   = bid / kkblock;
        const int ktb  = bid - nt * kkblock;
        const int n0   = nt * GW_KT;
        const int warp = tid >> 6;
        const int lane = tid & 63;
        const int k0   = ktb * GW_K + warp * GW_KT;

        constexpr int A_SZ   = GW_NT * GW_MS;             // a tile per buffer
        constexpr int B_SZ   = GW_KT * GW_MS;             // b tile per warp per buffer
        KernelStorage* a_lds = (KernelStorage*)smem_raw;  // [2][16n][GW_MS]
        KernelStorage* b_lds = a_lds + (size_t)2 * A_SZ;  // [2][4][16k][GW_MS]

        DUFragment<accumulator, 16, 16, MK, float> acc[4];
#    pragma unroll
        for (int q = 0; q < 4; q++) du_fill_fragment(acc[q], 0.0f);

        const int nsteps = (M + M_STEP - 1) / M_STEP;

        // Stage gp^T (shared) + x^T (per warp) for step s into LDS buffer buf.
        auto stage = [&](int s, int buf) {
            const int m0 = s * M_STEP;
            // gp^T [16 n][M_STEP m] -> a_lds[buf][n*GW_MS + m], 16B n loads.
            constexpr int AN = 16 / (int)sizeof(KernelStorage);
            for (int u = tid; u < M_STEP * (GW_NT / AN); u += PRMS_THREADS) {
                const int mm = u / (GW_NT / AN);
                const int nn = (u % (GW_NT / AN)) * AN;
                KernelStorage tmp[AN];
                const int g_m = m0 + mm;
                if (g_m < M) {
                    if (n0 + nn + AN <= N) {
                        prms_copy_convert_vec<KernelStorage, T, AN>(tmp, grad_proj + (size_t)g_m * N + n0 + nn);
                    } else {
#    pragma unroll
                        for (int e = 0; e < AN; e++)
                            tmp[e] = (n0 + nn + e < N)
                                         ? prms_convert<KernelStorage>(grad_proj[(size_t)g_m * N + n0 + nn + e])
                                         : prms_zero<KernelStorage>();
                    }
                } else {
#    pragma unroll
                    for (int e = 0; e < AN; e++) tmp[e] = prms_zero<KernelStorage>();
                }
#    pragma unroll
                for (int e = 0; e < AN; e++) a_lds[(size_t)buf * A_SZ + (size_t)(nn + e) * GW_MS + mm] = tmp[e];
            }
            // x^T per warp [16 k][M_STEP m] -> b_lds[buf][warp][k*GW_MS + m].
            // 16B vector loads along k (GV elems), transposed LDS stores.
            constexpr int GV  = 16 / (int)sizeof(KernelStorage);
            KernelStorage* bw = b_lds + (size_t)(buf * 4 + warp) * B_SZ;
#    pragma unroll
            for (int j = 0; j < M_STEP * (int)sizeof(KernelStorage) / 64; j++) {
                int mm, kb;
                if constexpr (sizeof(KernelStorage) == 4) {
                    mm = lane >> 1;  // 32m x 16k
                    kb = (lane & 1) * 8 + j * 4;
                } else {
                    mm = (lane >> 1) + 32 * j;  // 64m x 16k
                    kb = (lane & 1) * 8;
                }
                const int g_m = m0 + mm;
                if (g_m + GV <= M) {
                    KernelStorage tmp[GV];
                    prms_copy_convert_vec<KernelStorage, T, GV>(tmp, x + (size_t)g_m * K + k0 + kb);
#    pragma unroll
                    for (int e = 0; e < GV; e++) bw[(size_t)(kb + e) * GW_MS + mm] = tmp[e];
                } else {
#    pragma unroll
                    for (int e = 0; e < GV; e++) {
                        bw[(size_t)(kb + e) * GW_MS + mm] =
                            (g_m < M) ? prms_convert<KernelStorage>(x[(size_t)g_m * K + k0 + kb + e])
                                      : prms_zero<KernelStorage>();
                    }
                }
            }
        };

        stage(0, 0);
        __syncthreads();

#    pragma unroll 1
        for (int s = 0; s < nsteps; s++) {
            if (s + 1 < nsteps) stage(s + 1, (s + 1) & 1);  // prefetch next step
            const KernelStorage* ab = a_lds + (size_t)(s & 1) * A_SZ;
            const KernelStorage* bb = b_lds + (size_t)((s & 1) * 4 + warp) * B_SZ;
#    pragma unroll
            for (int mc = 0; mc < 4; mc++) {
                DUFragment<matrix_a, 16, 16, MK, KernelStorage, row_major> a;
                du_load_matrix_sync(a, ab + mc * MK, GW_MS);
                DUFragment<matrix_b, 16, 16, MK, KernelStorage, col_major> b;
                du_load_matrix_sync(b, bb + mc * MK, GW_MS);
                du_mma_sync(acc[mc], a, b, acc[mc]);
            }
            __syncthreads();
        }

        // Sum the 4 per-m-phase accumulators and write the final tile once.
        float sum[4];
#    pragma unroll
        for (int q = 0; q < 4; q++) sum[q] = acc[0].x[q] + acc[1].x[q] + acc[2].x[q] + acc[3].x[q];
        const int n = n0 + (lane & 15);
        int col     = k0 + (lane >> 4);
        if (n < N) {
            if (col < K) grad_w[(size_t)n * K + col] = prms_from_float<W>(sum[0]);
            if (col + 4 < K) grad_w[(size_t)n * K + col + 4] = prms_from_float<W>(sum[1]);
            if (col + 8 < K) grad_w[(size_t)n * K + col + 8] = prms_from_float<W>(sum[2]);
            if (col + 12 < K) grad_w[(size_t)n * K + col + 12] = prms_from_float<W>(sum[3]);
        }
        return;
    }

    // ---- grad_x: MB_X rows x KS_X cols, LDS-staged w + grad_proj ----
    const int gx   = bid - nbw;
    const int kx   = gx % nkx;
    const int rblk = gx / nkx;
    const int r4   = tid >> 6;
    const int kg   = tid & 63;
    const int k0   = kx * KS_X + kg * VEC;

    KernelStorage* w_lds = (KernelStorage*)smem_raw;            // [N][KS_X]
    float* gp_lds        = (float*)(w_lds + (size_t)N * KS_X);  // [MB_X][N]
    {
        constexpr int VPR = KS_X / VEC;
        for (int u = tid; u < N * VPR; u += PRMS_THREADS) {
            int nn = u / VPR;
            int vv = u - nn * VPR;
            prms_copy_convert_vec<KernelStorage, W, VEC>(w_lds + (size_t)nn * KS_X + vv * VEC,
                                                         weight + (size_t)nn * K + kx * KS_X + vv * VEC);
        }
        for (int u = tid; u < MB_X * N; u += PRMS_THREADS) {
            int rr    = u / N;
            int nn    = u - rr * N;
            int row   = rblk * MB_X + rr;
            gp_lds[u] = (row < M) ? prms_to_compute_float<KernelStorage>(grad_proj[(size_t)row * N + nn]) : 0.0f;
        }
    }
    __syncthreads();

    constexpr int ROWS = 8;  // rows per thread
    float acc[ROWS][VEC];
#    pragma unroll
    for (int r = 0; r < ROWS; r++)
#    pragma unroll
        for (int v = 0; v < VEC; v++) acc[r][v] = 0.0f;

    const float* gp_rows = gp_lds + (size_t)r4 * ROWS * N;
#    pragma unroll 4
    for (int n = 0; n < N; n++) {
        float wv[VEC];
        prms_load_vec<KernelStorage, VEC>(wv, w_lds + (size_t)n * KS_X + kg * VEC);
#    pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const float gpv = gp_rows[r * N + n];
#    pragma unroll
            for (int v = 0; v < VEC; v++) acc[r][v] += gpv * wv[v];
        }
    }

// Fused grad_r epilogue: grad_r * dr/dx = rs * x[m,k].
#    pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = rblk * MB_X + r4 * ROWS + r;
        if (row >= M) continue;
        const float rf   = fmaxf(prms_to_float(r_fwd[row]), 1e-10f);
        const float gr   = prms_to_float(grad_r[row]);
        const float norm = sqrtf((float)K) * (1.0f / rf - eps);
        const float rs   = (norm > 0.0f) ? gr * (-rf * rf) / (norm * sqrtf((float)K)) : 0.0f;

        float xv[VEC];
#    pragma unroll
        for (int v = 0; v < VEC; v++) xv[v] = prms_to_compute_float<KernelStorage>(x[(size_t)row * K + k0 + v]);
#    pragma unroll
        for (int v = 0; v < VEC; v++) acc[r][v] += rs * xv[v];
        prms_store_vec<T, VEC>(grad_x + (size_t)row * K + k0, acc[r]);
    }
}

#else  // no MMAC intrinsics on this arch — kernels are never launched at runtime.

template<typename T, typename W, typename KernelStorage>
static __global__ void __launch_bounds__(PRMS_THREADS)
    proj_rms_bwd_fused_mma_kernel(const T*, const T*, const T*, const W*, const T*, T*, W*, int, int, int, float) {}

#endif

namespace ftrain {
namespace {
template<typename T, typename W, typename KernelStorage>
static bool launch_proj_rms_bwd_single_t(hipStream_t stream, const T* grad_proj, const T* grad_r, const T* x,
                                         const W* w, const T* r_fwd, T* grad_x, W* grad_w, int M, int N, int K,
                                         float eps);

template<typename T, typename W, typename KernelStorage>
static bool launch_proj_rms_bwd_single_t(hipStream_t stream, const T* grad_proj, const T* grad_r, const T* x,
                                         const W* w, const T* r_fwd, T* grad_x, W* grad_w, int M, int N, int K,
                                         float eps) {
    constexpr int VEC  = (sizeof(KernelStorage) == 4) ? 4 : 8;
    constexpr int KS_X = 64 * VEC;
    if (N > 0 && N <= 32 && K % KS_X == 0) {
        constexpr int MB_X = 32;
        const int nbx      = (K / KS_X) * ((M + MB_X - 1) / MB_X);
        const int nbw      = ((N + 15) / 16) * (K / 64);
        dim3 grid(nbw + nbx);
        // grad_w uses two LDS buffers with 16 n-rows and four warps of k-rows;
        // grad_x needs [N][KS_X] + [32][N] floats.  Take the larger.
        constexpr int GW_MS_PAD = (sizeof(KernelStorage) == 4) ? 40 : 72;
        size_t shm_gw           = (size_t)2 * (16 * GW_MS_PAD + 4 * 16 * GW_MS_PAD) * sizeof(KernelStorage);
        size_t shm_gx           = (size_t)N * KS_X * sizeof(KernelStorage) + (size_t)MB_X * N * sizeof(float);
        size_t shm              = shm_gw > shm_gx ? shm_gw : shm_gx;
        hipLaunchKernelGGL((proj_rms_bwd_fused_mma_kernel<T, W, KernelStorage>), grid, PRMS_THREADS, shm, stream,
                           grad_proj, grad_r, x, w, r_fwd, grad_x, grad_w, M, N, K, eps);
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "proj_rms_bwd kernel launch failed: %s",
                                        hipGetErrorString(status));
            }
        }
        return true;
    }
    const int TX = 16, TY = 16;
    const int nkx = (K + TX - 1) / TX;
    const int nbx = nkx * ((M + TY - 1) / TY);
    const int nbw = nkx * ((N + TY - 1) / TY);
    dim3 grid(nbw + nbx);
    hipLaunchKernelGGL((proj_rms_bwd_fused_kernel<T, W, KernelStorage, 1>), grid, 256, 0, stream, grad_proj, grad_r, x,
                       w, r_fwd, grad_x, grad_w, M, N, K, eps);
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "proj_rms_bwd kernel launch failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}

template<typename T, typename W>
static bool launch_proj_rms_bwd_single(hipStream_t stream, const void* grad_proj, const void* grad_r, const void* x,
                                       const void* w, const void* r_fwd, void* grad_x, void* grad_w, int M, int N,
                                       int K, float eps) {
    {
        using KernelStorage = float;
        return launch_proj_rms_bwd_single_t<T, W, KernelStorage>(
            stream, static_cast<const T*>(grad_proj), static_cast<const T*>(grad_r), static_cast<const T*>(x),
            static_cast<const W*>(w), static_cast<const T*>(r_fwd), static_cast<T*>(grad_x), static_cast<W*>(grad_w), M,
            N, K, eps);
    }
}
}  // namespace

bool launchMixedFp32ProjRmsBwd(FTrainStream stream, const void* grad_proj, const void* grad_r, const void* x,
                               const void* weight, const void* r, void* grad_x, void* grad_weight, int rows,
                               int outputs, int reduction, float eps, FTrainNumericType x_type,
                               FTrainNumericType weight_type) {
    if (x_type == FTRAIN_NUMERIC_TYPE_BF16 && weight_type == FTRAIN_NUMERIC_TYPE_FP16) {
        return launch_proj_rms_bwd_single<__hip_bfloat16, __half>(stream, grad_proj, grad_r, x, weight, r, grad_x,
                                                                  grad_weight, rows, outputs, reduction, eps);
    }
    if (x_type == FTRAIN_NUMERIC_TYPE_BF16 && weight_type == FTRAIN_NUMERIC_TYPE_FP32) {
        return launch_proj_rms_bwd_single<__hip_bfloat16, float>(stream, grad_proj, grad_r, x, weight, r, grad_x,
                                                                 grad_weight, rows, outputs, reduction, eps);
    }
    if (x_type == FTRAIN_NUMERIC_TYPE_FP16 && weight_type == FTRAIN_NUMERIC_TYPE_BF16) {
        return launch_proj_rms_bwd_single<__half, __hip_bfloat16>(stream, grad_proj, grad_r, x, weight, r, grad_x,
                                                                  grad_weight, rows, outputs, reduction, eps);
    }
    if (x_type == FTRAIN_NUMERIC_TYPE_FP16 && weight_type == FTRAIN_NUMERIC_TYPE_FP32) {
        return launch_proj_rms_bwd_single<__half, float>(stream, grad_proj, grad_r, x, weight, r, grad_x, grad_weight,
                                                         rows, outputs, reduction, eps);
    }
    if (x_type == FTRAIN_NUMERIC_TYPE_FP32 && weight_type == FTRAIN_NUMERIC_TYPE_BF16) {
        return launch_proj_rms_bwd_single<float, __hip_bfloat16>(stream, grad_proj, grad_r, x, weight, r, grad_x,
                                                                 grad_weight, rows, outputs, reduction, eps);
    }
    if (x_type == FTRAIN_NUMERIC_TYPE_FP32 && weight_type == FTRAIN_NUMERIC_TYPE_FP16) {
        return launch_proj_rms_bwd_single<float, __half>(stream, grad_proj, grad_r, x, weight, r, grad_x, grad_weight,
                                                         rows, outputs, reduction, eps);
    }
    return false;
}

}  // namespace ftrain
