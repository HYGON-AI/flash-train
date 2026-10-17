// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <type_traits>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#    include <du_mma.h>
#endif

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/proj_rms_compute_h_bwd/mixed_fp32_proj_rms_compute_h_bwd.hpp"

static inline bool proj_rms_mma_supported() {
    int dev = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx928") != nullptr || strstr(prop.gcnArchName, "gfx936") != nullptr ||
           strstr(prop.gcnArchName, "gfx938") != nullptr;
}

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

template<typename T>
__device__ __forceinline__ static float prms_ch_dh(int c, int row, int n, int nres, const T* __restrict__ grad_h_pre,
                                                   const T* __restrict__ grad_h_post, const T* __restrict__ grad_h_res,
                                                   const T* __restrict__ h_pre, const T* __restrict__ h_post,
                                                   float compute_h_eps) {
    if (c < n) {
        const float sig = prms_to_float(h_pre[(size_t)row * n + c]) - compute_h_eps;
        return prms_to_float(grad_h_pre[(size_t)row * n + c]) * sig * (1.0f - sig);
    } else if (c < 2 * n) {
        const int cn    = c - n;
        const float sig = prms_to_float(h_post[(size_t)row * n + cn]) * 0.5f;
        return prms_to_float(grad_h_post[(size_t)row * n + cn]) * 2.0f * sig * (1.0f - sig);
    } else {
        return prms_to_float(grad_h_res[(size_t)row * nres + (c - 2 * n)]);
    }
}

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;

template<typename T, typename W, typename KernelStorage, bool PREPARE>
static __global__ void __launch_bounds__(PRMS_THREADS) proj_rms_ch_bwd_fused_mma_kernel(
    const T* __restrict__ grad_h_pre, const T* __restrict__ grad_h_post, const T* __restrict__ grad_h_res,
    const T* __restrict__ h_pre, const T* __restrict__ h_post, const T* __restrict__ proj, const T* __restrict__ r_ch,
    const T* __restrict__ grad_r_ch, const T* __restrict__ x, const W* __restrict__ weight,
    const float* __restrict__ alpha_pre, const float* __restrict__ alpha_post, const float* __restrict__ alpha_res,
    T* __restrict__ grad_x, W* __restrict__ grad_w, float* __restrict__ grad_bias, float* __restrict__ grad_alpha,
    float* __restrict__ gpv_scr, float* __restrict__ rpr_scr, float* __restrict__ gr_scr, int M, int N, int n, int nres,
    int K, float eps, float compute_h_eps) {
    constexpr int VEC      = (sizeof(KernelStorage) == 4) ? 4 : 8;
    constexpr int KS_X     = 64 * VEC;  // grad_x k-width per block
    constexpr int MB_X     = 32;        // grad_x rows per block
    constexpr int GW_KT    = 16;        // grad_w k-cols per warp
    constexpr int MK       = PrmsCfg<KernelStorage>::kfrag;
    constexpr int GW_MS    = (sizeof(KernelStorage) == 2) ? 96 : 32;  // grad_w M step per staging
    constexpr int GW_K     = 4 * GW_KT;                               // grad_w k-cols per block (64)
    constexpr int GW_NF    = GW_MS / MK;                              // m-fragments per buffer
    constexpr int GW_NT    = 16;                                      // grad_w n-rows per warp
    constexpr int GW_NTILE = 16;                                      // grad_w n-rows per block
    // Conflict-free LDS K... M stride for the MMA fragment loads:
    //   fp32 : bank(r) = (S*r + c) mod 32; S%32 == 10 -> 16 distinct row banks
    //   fp16/bf16: bank(r) = (S*r + c)/2 mod 32; S%32 == 2 -> 16 distinct rows
    constexpr int GW_PAD = (sizeof(KernelStorage) == 4) ? 10 : 2;
    constexpr int GW_S   = GW_MS + GW_PAD;  // padded M stride
    const int bid        = blockIdx.x;
    const int tid        = threadIdx.x;

    extern __shared__ char smem_raw[];

    const int nkx     = K / KS_X;
    const int nbx     = nkx * ((M + MB_X - 1) / MB_X);
    const int nbt     = (N + GW_NTILE - 1) / GW_NTILE;
    const int kkblock = K / GW_K;
    const int nbw     = nbt * kkblock;

    const float ap  = alpha_pre[0];
    const float apo = alpha_post[0];
    const float ar  = alpha_res[0];

    if constexpr (PREPARE) {
        constexpr int PH_MB = 32;
        const int row0      = bid * PH_MB;
        float* ph_gr        = (float*)smem_raw;  // [PH_MB]
        float* ph_bias      = ph_gr + PH_MB;     // [N]
        float* ph_alpha     = ph_bias + N;       // [3]
        for (int u = tid; u < PH_MB + N + 3; u += PRMS_THREADS) {
            if (u < PH_MB)
                ph_gr[u] = 0.0f;
            else if (u < PH_MB + N)
                ph_bias[u - PH_MB] = 0.0f;
            else
                ph_alpha[u - PH_MB - N] = 0.0f;
        }
        __syncthreads();
        for (int u = tid; u < PH_MB * N; u += PRMS_THREADS) {
            const int rr  = u / N;
            const int cc  = u - rr * N;
            const int row = row0 + rr;
            if (row >= M) continue;
            const float rpr = 1.0f / (prms_to_float(r_ch[row]) + eps);
            const float dh =
                prms_ch_dh(cc, row, n, nres, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post, compute_h_eps);
            const float alpha             = (cc < n) ? ap : ((cc < 2 * n) ? apo : ar);
            const float pv                = prms_to_float(proj[(size_t)row * N + cc]);
            gpv_scr[(size_t)row * N + cc] = dh * alpha * rpr;
            if (cc == 0) rpr_scr[row] = rpr;
            atomicAdd(&ph_gr[rr], dh * alpha * pv);
            atomicAdd(&ph_bias[cc], dh);
            atomicAdd(&ph_alpha[(cc < n) ? 0 : ((cc < 2 * n) ? 1 : 2)], dh * pv * rpr);
        }
        __syncthreads();
        if (tid == 0) {
            for (int r = 0; r < PH_MB && row0 + r < M; r++) gr_scr[row0 + r] = ph_gr[r];
            for (int c = 0; c < N; c++) atomicAdd(&grad_bias[c], ph_bias[c]);
            atomicAdd(&grad_alpha[0], ph_alpha[0]);
            atomicAdd(&grad_alpha[1], ph_alpha[1]);
            atomicAdd(&grad_alpha[2], ph_alpha[2]);
        }
        return;
    }

    const int b2 = bid;
    if (b2 < nbw) {
        constexpr int NTILE = GW_NTILE;
        const int nt        = b2 / kkblock;
        const int ktb       = b2 - nt * kkblock;
        const int n0        = nt * NTILE;
        const int warp      = tid >> 6;
        const int lane      = tid & 63;
        const int wn        = 0;  // n-tile of this warp
        const int k0        = ktb * GW_K + warp * GW_KT;

        constexpr int A_SZ   = NTILE * GW_S;              // a tile per buffer (padded)
        constexpr int B_SZ   = GW_KT * GW_S;              // b tile per warp per buffer
        KernelStorage* a_lds = (KernelStorage*)smem_raw;  // [2][NTILE n][GW_S]
        KernelStorage* b_lds = a_lds + (size_t)2 * A_SZ;  // [2][4][16k][GW_S]

        DUFragment<accumulator, 16, 16, MK, float> acc[GW_NF];
#    pragma unroll
        for (int r = 0; r < GW_NF; r++) du_fill_fragment(acc[r], 0.0f);

        const int nsteps = (M + GW_MS - 1) / GW_MS;

        // gp^T [16 n][GW_MS m] -> a_lds[buf][n*GW_S + m].  The score
        // epilogue (dh = d(h_act)/d(h_lin) * grad) is fused into the
        // staging, so grad_proj is never written to global memory.
        // Element-wise staging, two elements per thread: the independent
        // global loads of both elements are issued back-to-back so the
        // L2 latency of the dh epilogue overlaps.
        auto stage_a = [&](int s, int buf) {
            const int m0 = s * GW_MS;
#    pragma unroll 2
            for (int u = tid; u < NTILE * GW_MS; u += PRMS_THREADS) {
                const int mm  = u / NTILE;
                const int c   = n0 + u - mm * NTILE;
                const int g_m = m0 + mm;
                float gpv     = 0.0f;
                if (g_m < M && c < N) { gpv = gpv_scr[(size_t)g_m * N + c]; }
                a_lds[(size_t)buf * A_SZ + (size_t)(c - n0) * GW_S + mm] = prms_from_float<KernelStorage>(gpv);
            }
        };

        // x^T per warp [16 k][GW_MS m] -> b_lds[buf][warp][k*GW_S + m].
        auto stage_b_lds = [&](int s, int buf) {
            const int m0      = s * GW_MS;
            constexpr int GV  = 16 / (int)sizeof(KernelStorage);
            KernelStorage* bw = b_lds + (size_t)(buf * 4 + warp) * B_SZ;
            constexpr int BJ  = (sizeof(KernelStorage) == 4) ? (GW_MS / 16) : (GW_MS / 32);
#    pragma unroll
            for (int j = 0; j < BJ; j++) {
                int mm, kb;
                if constexpr (sizeof(KernelStorage) == 4) {
                    mm = (lane >> 1) + 32 * (j >> 1);
                    kb = (lane & 1) * 8 + (j & 1) * 4;
                } else {
                    mm = (lane >> 1) + 32 * j;
                    kb = (lane & 1) * 8;
                }
                const int g_m = m0 + mm;
                if (g_m + GV <= M) {
                    KernelStorage tmp[GV];
                    prms_copy_convert_vec<KernelStorage, T, GV>(tmp, x + (size_t)g_m * K + k0 + kb);
#    pragma unroll
                    for (int e = 0; e < GV; e++) bw[(size_t)(kb + e) * GW_S + mm] = tmp[e];
                } else {
#    pragma unroll
                    for (int e = 0; e < GV; e++) {
                        bw[(size_t)(kb + e) * GW_S + mm] =
                            (g_m < M) ? prms_convert<KernelStorage>(x[(size_t)g_m * K + k0 + kb + e])
                                      : prms_zero<KernelStorage>();
                    }
                }
            }
        };

        stage_b_lds(0, 0);
        stage_a(0, 0);
        __syncthreads();

#    pragma unroll 1
        for (int s = 0; s < nsteps; s++) {
            if (s + 1 < nsteps) {
                stage_a(s + 1, (s + 1) & 1);  // prefetch next step
                stage_b_lds(s + 1, (s + 1) & 1);
            }
            const KernelStorage* ab = a_lds + (size_t)(s & 1) * A_SZ;
            const KernelStorage* bb = b_lds + (size_t)((s & 1) * 4 + warp) * B_SZ;
#    pragma unroll
            for (int mc = 0; mc < GW_NF; mc++) {
                DUFragment<matrix_b, 16, 16, MK, KernelStorage, col_major> b;
                du_load_matrix_sync(b, bb + mc * MK, GW_S);
                DUFragment<matrix_a, 16, 16, MK, KernelStorage, row_major> a;
                du_load_matrix_sync(a, ab + (size_t)wn * GW_KT * GW_S + mc * MK, GW_S);
                du_mma_sync(acc[mc], a, b, acc[mc]);
            }
            __syncthreads();
        }

        float sum[4];
#    pragma unroll
        for (int r = 0; r < 4; r++) sum[r] = 0.0f;
#    pragma unroll
        for (int r = 0; r < 4; r++)
#    pragma unroll
            for (int a = 0; a < GW_NF; a++) sum[r] += acc[a].x[r];
        const int nn  = n0 + wn * GW_KT + (lane & 15);
        const int col = k0 + (lane >> 4);
        if (nn < N) {
            if (col < K) grad_w[(size_t)nn * K + col] = prms_from_float<W>(sum[0]);
            if (col + 4 < K) grad_w[(size_t)nn * K + col + 4] = prms_from_float<W>(sum[1]);
            if (col + 8 < K) grad_w[(size_t)nn * K + col + 8] = prms_from_float<W>(sum[2]);
            if (col + 12 < K) grad_w[(size_t)nn * K + col + 12] = prms_from_float<W>(sum[3]);
        }
        return;
    }

    // ---- grad_x: MB_X rows x KS_X cols, LDS-staged w + inline gp ----
    const int gx   = b2 - nbw;
    const int kx   = gx % nkx;
    const int rblk = gx / nkx;
    const int r4   = tid >> 6;
    const int kg   = tid & 63;
    const int k0   = kx * KS_X + kg * VEC;

    KernelStorage* w_lds = (KernelStorage*)smem_raw;            // [N][KS_X]
    float* gp_lds        = (float*)(w_lds + (size_t)N * KS_X);  // [MB_X][N]
    // All compute types read phase-0 precomputed gpv / gr_scr, avoiding
    // per-k-block nonlinear score recomputation and partial reductions.
    {
        constexpr int VPR = KS_X / VEC;
        for (int u = tid; u < N * VPR; u += PRMS_THREADS) {
            int nn = u / VPR;
            int vv = u - nn * VPR;
            prms_copy_convert_vec<KernelStorage, W, VEC>(w_lds + (size_t)nn * KS_X + vv * VEC,
                                                         weight + (size_t)nn * K + kx * KS_X + vv * VEC);
        }
        __syncthreads();
        for (int u = tid; u < MB_X * N; u += PRMS_THREADS) {
            const int rr  = u / N;
            const int nn  = u - rr * N;
            const int row = rblk * MB_X + rr;
            float gpv     = 0.0f;
            if (row < M) { gpv = gpv_scr[(size_t)row * N + nn]; }
            gp_lds[u] = gpv;
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
    for (int nn = 0; nn < N; nn++) {
        float wv[VEC];
        prms_load_vec<KernelStorage, VEC>(wv, w_lds + (size_t)nn * KS_X + kg * VEC);
#    pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const float gpv = gp_rows[r * N + nn];
#    pragma unroll
            for (int v = 0; v < VEC; v++) acc[r][v] += gpv * wv[v];
        }
    }

// Fused r epilogue: rs = dL/dr_pr * d(r_pr)/dx, with the loss-side
// grad_r_ch folded into the r_pr convention: dL/dr_pr = gr_h - g_r_ch/r_pr^2.
#    pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = rblk * MB_X + r4 * ROWS + r;
        if (row >= M) continue;
        const float rpr      = rpr_scr[row];
        const float rf       = fmaxf(rpr, 1e-10f);
        const float gr_h     = gr_scr[row];
        const float gr_total = gr_h - prms_to_float(grad_r_ch[row]) / (rf * rf);
        // Algebraically fold sqrt(K) * sqrt(K) into K.  This is the same
        // RMS derivative but removes the vector sqrt instruction from every
        // row of the mixed bf16/fp32 grad-x epilogue.
        const float rms_term = 1.0f / rf - eps;
        const float denom    = (float)K * rms_term;
        const float rs       = (denom > 0.0f) ? gr_total * (-rf * rf) / denom : 0.0f;

        float xv[VEC];
#    pragma unroll
        for (int v = 0; v < VEC; v++) xv[v] = prms_to_compute_float<KernelStorage>(x[(size_t)row * K + k0 + v]);
#    pragma unroll
        for (int v = 0; v < VEC; v++) acc[r][v] += rs * xv[v];
        prms_store_vec<T, VEC>(grad_x + (size_t)row * K + k0, acc[r]);
    }
}

#else  // no MMAC intrinsics on this arch — kernels are never launched at runtime.

template<typename T, typename W, typename KernelStorage, bool PREPARE>
static __global__ void __launch_bounds__(PRMS_THREADS)
    proj_rms_ch_bwd_fused_mma_kernel(const T*, const T*, const T*, const T*, const T*, const T*, const T*, const T*,
                                     const T*, const W*, const float*, const float*, const float*, T*, W*, float*,
                                     float*, float*, float*, float*, int, int, int, int, int, float, float) {}

#endif
template<typename T, typename W, typename KernelStorage>
static __global__ void __launch_bounds__(256)
    proj_rms_ch_bwd_portable_kernel(const T* __restrict__ grad_h_pre, const T* __restrict__ grad_h_post,
                                    const T* __restrict__ grad_h_res, const T* __restrict__ h_pre,
                                    const T* __restrict__ h_post, const T* __restrict__ proj,
                                    const T* __restrict__ r_ch, const T* __restrict__ grad_r_ch,
                                    const T* __restrict__ x, const W* __restrict__ weight,
                                    const float* __restrict__ alpha_pre, const float* __restrict__ alpha_post,
                                    const float* __restrict__ alpha_res, T* __restrict__ grad_x, W* __restrict__ grad_w,
                                    float* __restrict__ grad_bias, float* __restrict__ grad_alpha, int M, int N, int n,
                                    int nres, int K, float eps, float compute_h_eps) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int TX = 16, TY = 16;
    const int nkx   = (K + TX - 1) / TX;
    const int nbw   = nkx * ((N + TY - 1) / TY);
    const int txx   = tid % TX;
    const int tyy   = tid / TX;
    const float ap  = alpha_pre[0];
    const float apo = alpha_post[0];
    const float ar  = alpha_res[0];

    if (bid < nbw) {
        const int nt = bid / nkx;
        const int kt = bid - nt * nkx;
        const int nn = nt * TY + tyy;
        const int k  = kt * TX + txx;
        if (nn >= N || k >= K) return;
        float acc = 0.0f;
        for (int m = 0; m < M; m++) {
            const float dh =
                prms_ch_dh(nn, m, n, nres, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post, compute_h_eps);
            const float alpha  = (nn < n) ? ap : ((nn < 2 * n) ? apo : ar);
            const float rpr    = 1.0f / (prms_to_float(r_ch[m]) + eps);
            acc               += (dh * alpha * rpr) * prms_to_compute_float<KernelStorage>(x[(size_t)m * K + k]);
        }
        grad_w[(size_t)nn * K + k] = prms_from_float<W>(acc);
        return;
    }

    const int gx  = bid - nbw;
    const int kt  = gx % nkx;
    const int row = (gx / nkx) * TY + tyy;
    const int k   = kt * TX + txx;
    if (row >= M || k >= K) return;

    const float xv  = prms_to_float(x[(size_t)row * K + k]);
    const float rpr = 1.0f / (prms_to_float(r_ch[row]) + eps);
    const float rf  = fmaxf(rpr, 1e-10f);
    float gp_acc = 0.0f, gr_h = 0.0f;
    const T* prow  = proj + (size_t)row * N;
    const W* w_col = weight + k;
    for (int c = 0; c < N; c++) {
        const float dh = prms_ch_dh(c, row, n, nres, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post, compute_h_eps);
        const float alpha  = (c < n) ? ap : ((c < 2 * n) ? apo : ar);
        const float pv     = prms_to_float(prow[c]);
        gp_acc            += (dh * alpha * rpr) * prms_to_compute_float<KernelStorage>(w_col[(size_t)c * K]);
        gr_h              += dh * pv * alpha;
    }
    const float gr_total = gr_h - prms_to_float(grad_r_ch[row]) / (rf * rf);
    const float norm     = sqrtf((float)K) * (1.0f / rf - eps);
    float dr_acc         = 0.0f;
    if (norm > 0.0f) dr_acc = gr_total * (-rf * rf) * xv / (norm * sqrtf((float)K));
    grad_x[(size_t)row * K + k] = prms_from_float<T>(gp_acc + dr_acc);

    // Reductions: each row has exactly one kt == 0 block; within it, the
    // thread with txx == 0 owns the row's reduction.
    if (kt == 0 && txx == 0) {
        float ga0 = 0.0f, ga1 = 0.0f, ga2 = 0.0f;
        for (int c = 0; c < N; c++) {
            const float dh =
                prms_ch_dh(c, row, n, nres, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post, compute_h_eps);
            atomicAdd(&grad_bias[c], dh);
            const float da = dh * prms_to_float(prow[c]) * rpr;
            if (c < n)
                ga0 += da;
            else if (c < 2 * n)
                ga1 += da;
            else
                ga2 += da;
        }
        atomicAdd(&grad_alpha[0], ga0);
        atomicAdd(&grad_alpha[1], ga1);
        atomicAdd(&grad_alpha[2], ga2);
    }
}

template<typename KernelStorage>
static bool prms_ch_bwd_fast_path(int M, int N, int K) {
    constexpr int KS_X = sizeof(KernelStorage) == 4 ? 256 : 512;
    return M > 0 && N > 0 && N <= 32 && K > 0 && K % KS_X == 0 && proj_rms_mma_supported();
}

namespace ftrain {
namespace {
template<typename T, typename W, typename KernelStorage>
static bool launch_proj_rms_ch_bwd_single_t(hipStream_t stream, const T* grad_h_pre, const T* grad_h_post,
                                            const T* grad_h_res, const T* h_pre, const T* h_post, const T* proj,
                                            const T* r_ch, const T* grad_r_ch, const T* x, const W* w,
                                            const float* alpha_pre, const float* alpha_post, const float* alpha_res,
                                            T* grad_x, W* grad_w, float* grad_bias, float* grad_alpha, float* workspace,
                                            int M, int N, int n, int K, float eps, float compute_h_eps);
size_t proj_rms_ch_bwd_workspace_floats(int M, int N, int K);
template<typename T, typename W>
bool launch_proj_rms_ch_bwd_single(hipStream_t stream, const void* grad_h_pre, const void* grad_h_post,
                                   const void* grad_h_res, const void* h_pre, const void* h_post, const void* proj,
                                   const void* r_ch, const void* grad_r_ch, const void* x, const void* w,
                                   const float* alpha_pre, const float* alpha_post, const float* alpha_res,
                                   void* grad_x, void* grad_w, float* grad_bias, float* grad_alpha, float* workspace,
                                   int M, int N, int n, int K, float eps, float compute_h_eps);

template<typename T, typename W, typename KernelStorage>
static bool launch_proj_rms_ch_bwd_single_t(hipStream_t stream, const T* grad_h_pre, const T* grad_h_post,
                                            const T* grad_h_res, const T* h_pre, const T* h_post, const T* proj,
                                            const T* r_ch, const T* grad_r_ch, const T* x, const W* w,
                                            const float* alpha_pre, const float* alpha_post, const float* alpha_res,
                                            T* grad_x, W* grad_w, float* grad_bias, float* grad_alpha, float* workspace,
                                            int M, int N, int n, int K, float eps, float compute_h_eps) {
    const int nres     = N - 2 * n;
    constexpr int VEC  = (sizeof(KernelStorage) == 4) ? 4 : 8;
    constexpr int KS_X = 64 * VEC;
    if (prms_ch_bwd_fast_path<KernelStorage>(M, N, K)) {
        constexpr int MB_X = 32;
        const int nbx      = (K / KS_X) * ((M + MB_X - 1) / MB_X);
        const int nbw      = ((N + 15) / 16) * (K / 64);
        const int nphase   = (M + 31) / 32;
        float* gpv_scr     = workspace;
        float* rpr_scr     = gpv_scr + (size_t)M * N;
        float* gr_scr      = rpr_scr + M;
        dim3 grid(nbw + nbx);
        constexpr int GW_MS_L  = (sizeof(KernelStorage) == 4) ? 32 : 96;
        constexpr int GW_PAD_L = (sizeof(KernelStorage) == 4) ? 10 : 2;
        size_t shm_gw          = (size_t)2 * (16 + 4 * 16) * (GW_MS_L + GW_PAD_L) * sizeof(KernelStorage);
        size_t shm_gx = (size_t)N * KS_X * sizeof(KernelStorage) + (size_t)(MB_X * N + MB_X + N + 3) * sizeof(float);
        size_t shm    = shm_gw > shm_gx ? shm_gw : shm_gx;
        hipLaunchKernelGGL((proj_rms_ch_bwd_fused_mma_kernel<T, W, KernelStorage, true>), dim3(nphase), PRMS_THREADS,
                           (32 + N + 3) * sizeof(float), stream, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post,
                           proj, r_ch, grad_r_ch, x, w, alpha_pre, alpha_post, alpha_res, grad_x, grad_w, grad_bias,
                           grad_alpha, gpv_scr, rpr_scr, gr_scr, M, N, n, nres, K, eps, compute_h_eps);
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "proj_rms_compute_h_bwd kernel launch failed: %s",
                                        hipGetErrorString(status));
            }
        }
        hipLaunchKernelGGL((proj_rms_ch_bwd_fused_mma_kernel<T, W, KernelStorage, false>), grid, PRMS_THREADS, shm,
                           stream, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post, proj, r_ch, grad_r_ch, x, w,
                           alpha_pre, alpha_post, alpha_res, grad_x, grad_w, grad_bias, grad_alpha, gpv_scr, rpr_scr,
                           gr_scr, M, N, n, nres, K, eps, compute_h_eps);
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "proj_rms_compute_h_bwd kernel launch failed: %s",
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
    hipLaunchKernelGGL((proj_rms_ch_bwd_portable_kernel<T, W, KernelStorage>), grid, 256, 0, stream, grad_h_pre,
                       grad_h_post, grad_h_res, h_pre, h_post, proj, r_ch, grad_r_ch, x, w, alpha_pre, alpha_post,
                       alpha_res, grad_x, grad_w, grad_bias, grad_alpha, M, N, n, nres, K, eps, compute_h_eps);
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "proj_rms_compute_h_bwd kernel launch failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}

static size_t proj_rms_ch_bwd_workspace_floats(int M, int N, int K) {
    size_t count = 0;
    {
        using KernelStorage = float;
        if (prms_ch_bwd_fast_path<KernelStorage>(M, N, K)) count = (size_t)M * ((size_t)N + 2);
    }
    return count;
}

template<typename T, typename W>
static bool launch_proj_rms_ch_bwd_single(hipStream_t stream, const void* grad_h_pre, const void* grad_h_post,
                                          const void* grad_h_res, const void* h_pre, const void* h_post,
                                          const void* proj, const void* r_ch, const void* grad_r_ch, const void* x,
                                          const void* w, const float* alpha_pre, const float* alpha_post,
                                          const float* alpha_res, void* grad_x, void* grad_w, float* grad_bias,
                                          float* grad_alpha, float* workspace, int M, int N, int n, int K, float eps,
                                          float compute_h_eps) {
    {
        using KernelStorage = float;
        return launch_proj_rms_ch_bwd_single_t<T, W, KernelStorage>(
            stream, static_cast<const T*>(grad_h_pre), static_cast<const T*>(grad_h_post),
            static_cast<const T*>(grad_h_res), static_cast<const T*>(h_pre), static_cast<const T*>(h_post),
            static_cast<const T*>(proj), static_cast<const T*>(r_ch), static_cast<const T*>(grad_r_ch),
            static_cast<const T*>(x), static_cast<const W*>(w), alpha_pre, alpha_post, alpha_res,
            static_cast<T*>(grad_x), static_cast<W*>(grad_w), grad_bias, grad_alpha, workspace, M, N, n, K, eps,
            compute_h_eps);
    }
}
}  // namespace

std::uint64_t mixedFp32ProjRmsComputeHBwdWorkspaceFloats(int rows, int outputs, int reduction) {
    return static_cast<std::uint64_t>(proj_rms_ch_bwd_workspace_floats(rows, outputs, reduction));
}

bool launchMixedFp32ProjRmsComputeHBwd(FTrainStream stream, const void* grad_h_pre, const void* grad_h_post,
                                       const void* grad_h_res, const void* grad_r, const void* h_pre,
                                       const void* h_post, const void* proj, const void* r, const void* x,
                                       const void* weight, const float* alpha_pre, const float* alpha_post,
                                       const float* alpha_res, void* grad_x, void* grad_weight, float* grad_alpha,
                                       float* grad_bias, float* workspace, int rows, int outputs, int num_streams,
                                       int reduction, float eps, float compute_h_eps, FTrainNumericType x_type,
                                       FTrainNumericType weight_type) {
    if (x_type == FTRAIN_NUMERIC_TYPE_BF16 && weight_type == FTRAIN_NUMERIC_TYPE_FP16) {
        return launch_proj_rms_ch_bwd_single<__hip_bfloat16, __half>(
            stream, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post, proj, r, grad_r, x, weight, alpha_pre,
            alpha_post, alpha_res, grad_x, grad_weight, grad_bias, grad_alpha, workspace, rows, outputs, num_streams,
            reduction, eps, compute_h_eps);
    }
    if (x_type == FTRAIN_NUMERIC_TYPE_BF16 && weight_type == FTRAIN_NUMERIC_TYPE_FP32) {
        return launch_proj_rms_ch_bwd_single<__hip_bfloat16, float>(
            stream, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post, proj, r, grad_r, x, weight, alpha_pre,
            alpha_post, alpha_res, grad_x, grad_weight, grad_bias, grad_alpha, workspace, rows, outputs, num_streams,
            reduction, eps, compute_h_eps);
    }
    if (x_type == FTRAIN_NUMERIC_TYPE_FP16 && weight_type == FTRAIN_NUMERIC_TYPE_BF16) {
        return launch_proj_rms_ch_bwd_single<__half, __hip_bfloat16>(
            stream, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post, proj, r, grad_r, x, weight, alpha_pre,
            alpha_post, alpha_res, grad_x, grad_weight, grad_bias, grad_alpha, workspace, rows, outputs, num_streams,
            reduction, eps, compute_h_eps);
    }
    if (x_type == FTRAIN_NUMERIC_TYPE_FP16 && weight_type == FTRAIN_NUMERIC_TYPE_FP32) {
        return launch_proj_rms_ch_bwd_single<__half, float>(
            stream, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post, proj, r, grad_r, x, weight, alpha_pre,
            alpha_post, alpha_res, grad_x, grad_weight, grad_bias, grad_alpha, workspace, rows, outputs, num_streams,
            reduction, eps, compute_h_eps);
    }
    if (x_type == FTRAIN_NUMERIC_TYPE_FP32 && weight_type == FTRAIN_NUMERIC_TYPE_BF16) {
        return launch_proj_rms_ch_bwd_single<float, __hip_bfloat16>(
            stream, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post, proj, r, grad_r, x, weight, alpha_pre,
            alpha_post, alpha_res, grad_x, grad_weight, grad_bias, grad_alpha, workspace, rows, outputs, num_streams,
            reduction, eps, compute_h_eps);
    }
    if (x_type == FTRAIN_NUMERIC_TYPE_FP32 && weight_type == FTRAIN_NUMERIC_TYPE_FP16) {
        return launch_proj_rms_ch_bwd_single<float, __half>(
            stream, grad_h_pre, grad_h_post, grad_h_res, h_pre, h_post, proj, r, grad_r, x, weight, alpha_pre,
            alpha_post, alpha_res, grad_x, grad_weight, grad_bias, grad_alpha, workspace, rows, outputs, num_streams,
            reduction, eps, compute_h_eps);
    }
    return false;
}

void mixedFp32ProjRmsComputeHBwdZeroMemory(FTrainStream stream, void* memory, std::uint64_t bytes) {
    if (bytes == 0 || memory == nullptr) { return; }
    const hipError_t status = hipMemsetAsync(memory, 0, static_cast<std::size_t>(bytes), stream);
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Workspace zero-fill failed: %s", hipGetErrorString(status));
    }
}

}  // namespace ftrain
