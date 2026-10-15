// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// This implementation uses HIP warp synchronization intrinsics.
#define HIP_ENABLE_WARP_SYNC_BUILTINS 1

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <math.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#    include <du_mma.h>
#endif

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_attn_bwd/fp32_sparse_attn_bwd.hpp"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

static inline int dsa_shared_memory_limit() {
    int device = 0, limit = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&limit, hipDeviceAttributeMaxSharedMemoryPerBlock, device) != hipSuccess)
        return 0;
    return limit;
}

static inline bool dsa_shared_memory_fits(size_t bytes) {
    return bytes <= static_cast<size_t>(dsa_shared_memory_limit());
}

template<typename T>
__device__ __forceinline__ static float dsa_to_float(T value) {
    return static_cast<float>(value);
}

template<>
__device__ __forceinline__ float dsa_to_float<float>(float value) {
    return value;
}

template<typename T>
__device__ __forceinline__ static T dsa_from_float(float value) {
    return static_cast<T>(value);
}

template<>
__device__ __forceinline__ float dsa_from_float<float>(float value) {
    return value;
}

template<typename Compute>
struct DsaMmacTraits {
    static constexpr int k               = sizeof(Compute) == 4 ? 8 : 16;
    static constexpr int vector_elements = 16 / sizeof(Compute);
    static constexpr int dense_dchunk    = sizeof(Compute) == 4 ? 64 : 128;
    static constexpr int sparse_chunk    = sizeof(Compute) == 4 ? 16 : 32;
};

template<typename Compute>
__device__ __forceinline__ static Compute dsa_load_as(const void* data, size_t index) {
    return reinterpret_cast<const Compute*>(data)[index];
}

template<typename Compute>
__device__ __forceinline__ static float dsa_load_compute_float(const void* data, size_t index) {
    return dsa_to_float(dsa_load_as<Compute>(data, index));
}

template<typename Compute, int Count>
__device__ __forceinline__ static void dsa_copy_as(Compute* dst, const void* src, size_t index) {
    constexpr int bytes = Count * sizeof(Compute);
    if constexpr (bytes == 16) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(reinterpret_cast<const Compute*>(src) + index);
    } else if constexpr (bytes == 8) {
        *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(reinterpret_cast<const Compute*>(src) + index);
    } else {
#pragma unroll
        for (int i = 0; i < Count; ++i) { dst[i] = reinterpret_cast<const Compute*>(src)[index + i]; }
    }
}

template<typename Native>
__device__ __forceinline__ static void dsa_store_from_float(void* data, size_t index, float value) {
    reinterpret_cast<Native*>(data)[index] = dsa_from_float<Native>(value);
}

#define SAF_THREADS 256
#define SAF_HBLK 64
#define SAF_MAX_D 576

static inline bool sparse_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx928") != nullptr || strstr(prop.gcnArchName, "gfx936") != nullptr ||
           strstr(prop.gcnArchName, "gfx938") != nullptr;
}

static inline bool sparse_attn_fused_supported() { return sparse_mmac_supported(); }

// Adapted from fused_dsa/hip_dsa/kernels/sparse_attn_bwd.hip.

__device__ __forceinline__ static void saf_backward_normalization(float lse, float sink, float& maximum, float& inverse,
                                                                  float& p_sink) {
    if (lse == INFINITY)
        p_sink = 0.0f;
    else if (lse == sink)
        p_sink = 0.5f;
    else
        p_sink = 1.0f / (1.0f + expf(lse - sink));
    const float kv_lse = lse == -INFINITY ? 0.0f : lse;
    maximum            = fmaxf(kv_lse, sink);
    if (isnan(lse) || isnan(sink)) {
        maximum = NAN;
        inverse = 1.0f;
    } else if (isinf(maximum)) {
        inverse = 1.0f;
    } else {
        inverse = 1.0f / (expf(kv_lse - maximum) + expf(sink - maximum));
    }
}

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;

using saf_intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
extern __device__ saf_intx4 saf_raw_buffer_load_i32x4(saf_intx4 srsrc, int voffset, int soffset,
                                                      int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
extern __device__ int saf_raw_buffer_load_i32(saf_intx4 srsrc, int voffset, int soffset,
                                              int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");

#    if defined(__gfx936__) || defined(__gfx938__)

using saf_floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
#    endif

__device__ __forceinline__ static void saf_atomic_add_mmac_row(float* row16, int g, float x0, float x1, float x2,
                                                               float x3) {
#    if defined(__gfx936__) || defined(__gfx938__)
    const float p0 = __shfl_xor(x0, 16, 64);
    const float p1 = __shfl_xor(x1, 16, 64);
    const float p2 = __shfl_xor(x2, 16, 64);
    const float p3 = __shfl_xor(x3, 16, 64);
    const float y0 = (g & 1) ? p1 : x0;
    const float y1 = (g & 1) ? x1 : p0;
    const float y2 = (g & 1) ? p3 : x2;
    const float y3 = (g & 1) ? x3 : p2;
    const float q0 = __shfl_xor(y0, 32, 64);
    const float q1 = __shfl_xor(y1, 32, 64);
    const float q2 = __shfl_xor(y2, 32, 64);
    const float q3 = __shfl_xor(y3, 32, 64);
    saf_floatx4 v;
    v[0] = (g & 2) ? q2 : y0;
    v[1] = (g & 2) ? q3 : y1;
    v[2] = (g & 2) ? y2 : q0;
    v[3] = (g & 2) ? y3 : q1;
#        if defined(__gfx938__)
    __builtin_hcu_global_atomic_fadd_f32_x4(row16 + 4 * g, v);
#        else
    __builtin_amdgcn_global_atomic_fadd_v4f32((saf_floatx4*)(row16 + 4 * g), v);
#        endif
#    else
    float* p = row16 + g;
    atomicAdd(&p[0], x0);
    atomicAdd(&p[4], x1);
    atomicAdd(&p[8], x2);
    atomicAdd(&p[12], x3);
#    endif
}

#endif  // supported gfx archs

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

template<typename C, int NK, bool COMPACT_VALUES = false>
static __global__ void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_bwd_dtype_kernel(const void* __restrict__ q, const void* __restrict__ kv, const void* __restrict__ out,
                                 const void* __restrict__ dout, const float* __restrict__ lse,
                                 const float* __restrict__ attn_sink, const int* __restrict__ idxs,
                                 const int* __restrict__ topk_length, float* __restrict__ dkv_float,
                                 float* __restrict__ dsink, void* __restrict__ dq, void* __restrict__ dkv,
                                 int* __restrict__ done_count, int T, int H, int D_rt, int T_kv, int topk,
                                 float softmax_scale) {
    constexpr int KFRAG      = DsaMmacTraits<C>::k;
    constexpr int FRAG_ELEMS = sizeof(C) == 4 ? 2 : 4;
    constexpr int D          = NK * KFRAG;
    constexpr int VALUE_DIM  = COMPACT_VALUES ? 512 : D;
    constexpr int NOUT       = D / 16;
    constexpr int VEC        = DsaMmacTraits<C>::vector_elements;
    constexpr int PAD        = sizeof(C) == 4 ? 10 : 8;
    constexpr int KLD        = D + PAD;
    (void)T;
    (void)D_rt;

    const int t        = blockIdx.x;
    const int hb       = blockIdx.y * SAF_HBLK;
    const int lane     = threadIdx.x & 63;
    const int warp     = threadIdx.x >> 6;
    const int r        = lane & 15;
    const int g        = lane >> 4;
    const int local_h  = warp * 16 + r;
    const int h        = hb + local_h;
    const bool lane_ok = h < H;

    extern __shared__ char smem_raw[];
    C* k_lds     = reinterpret_cast<C*>(smem_raw);  // [16][KLD]
    int* idx_lds = reinterpret_cast<int*>(k_lds + (size_t)16 * KLD);
    C* pt_lds    = reinterpret_cast<C*>(idx_lds + topk);  // [16][64]
    C* dlt_lds   = pt_lds + 16 * SAF_HBLK;                // [16][64]
    C* dl_lds    = dlt_lds + 16 * SAF_HBLK;               // [4][16][16]
    C* qd_lds    = dl_lds + 4 * 256;                      // [4][2][16][16]

    for (int i = threadIdx.x; i < topk; i += SAF_THREADS) idx_lds[i] = idxs[(size_t)t * topk + i];
    __syncthreads();
    int n_valid = topk;
    if (topk_length) {
        const int n = topk_length[t];
        if (n < n_valid) n_valid = n > 0 ? n : 0;
    }

    float m = 0.0f, inv_l = 0.0f, p_sink = 0.0f, avg = 0.0f;
    if (lane_ok) {
        const float lh = lse[(size_t)t * H + h];
        const float sh = attn_sink[h];
        saf_backward_normalization(lh, sh, m, inv_l, p_sink);
        {
#    pragma unroll
            for (int kk = 0; kk < VALUE_DIM / KFRAG; ++kk) {
#    pragma unroll
                for (int e = 0; e < FRAG_ELEMS; ++e) {
                    const size_t at  = ((size_t)t * H + h) * VALUE_DIM + kk * KFRAG + g * FRAG_ELEMS + e;
                    avg             += dsa_load_compute_float<C>(out, at) * dsa_load_compute_float<C>(dout, at);
                }
            }
        }
    }
    avg += __shfl_xor(avg, 16);
    avg += __shfl_xor(avg, 32);

    DUFragment<accumulator, 16, 16, KFRAG, float> dqacc[NOUT];
#    pragma unroll
    for (int dc = 0; dc < NOUT; ++dc) du_fill_fragment(dqacc[dc], 0.0f);

    const int ntiles = CEIL_DIV(topk, 16);
    for (int kt = 0; kt < ntiles; ++kt) {
        int ck = topk - kt * 16;
        if (ck > 16) ck = 16;
        for (int i = threadIdx.x; i < 16 * KLD; i += SAF_THREADS) k_lds[i] = dsa_from_float<C>(0.0f);
        __syncthreads();
        const int vecs = CEIL_DIV(D, VEC);
        for (int u = threadIdx.x; u < ck * vecs; u += SAF_THREADS) {
            const int kr  = u / vecs;
            const int dv  = (u % vecs) * VEC;
            const int key = kt * 16 + kr;
            int idx       = idx_lds[key];
            if (key >= n_valid || idx < 0 || idx >= T_kv) continue;
            const size_t src = (size_t)idx * D + dv;
            if (dv + VEC <= D) {
                dsa_copy_as<C, VEC>(k_lds + (size_t)kr * KLD + dv, kv, src);
            } else {
                for (int e = 0; e < VEC && dv + e < D; ++e)
                    k_lds[(size_t)kr * KLD + dv + e] = dsa_load_as<C>(kv, src + e);
            }
        }
        __syncthreads();

        DUFragment<accumulator, 16, 16, KFRAG, float> sacc, dacc;
        du_fill_fragment(sacc, 0.0f);
        du_fill_fragment(dacc, 0.0f);
#    pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
            DUFragment<matrix_a, 16, 16, KFRAG, C, row_major> qfr, dfr;
#    pragma unroll
            for (int e = 0; e < FRAG_ELEMS; ++e) {
                const size_t at = ((size_t)t * H + (lane_ok ? h : H - 1)) * D + kk * KFRAG + g * FRAG_ELEMS + e;
                qfr.x[e]        = lane_ok ? dsa_load_as<C>(q, at) : dsa_from_float<C>(0.0f);
                const int column      = kk * KFRAG + g * FRAG_ELEMS + e;
                const size_t value_at = ((size_t)t * H + (lane_ok ? h : H - 1)) * VALUE_DIM + column;
                dfr.x[e] = lane_ok && column < VALUE_DIM ? dsa_load_as<C>(dout, value_at) : dsa_from_float<C>(0.0f);
            }
            DUFragment<matrix_b, 16, 16, KFRAG, C, col_major> kfr;
            du_load_matrix_sync(kfr, k_lds + kk * KFRAG, KLD);
            du_mma_sync(sacc, qfr, kfr, sacc);
            du_mma_sync(dacc, dfr, kfr, dacc);
        }

        const int c0 = kt * 16 + g;
        const int c1 = c0 + 4, c2 = c0 + 8, c3 = c0 + 12;
        const bool ok0 = c0 < n_valid && idx_lds[c0] >= 0 && idx_lds[c0] < T_kv;
        const bool ok1 = c1 < n_valid && idx_lds[c1] >= 0 && idx_lds[c1] < T_kv;
        const bool ok2 = c2 < n_valid && idx_lds[c2] >= 0 && idx_lds[c2] < T_kv;
        const bool ok3 = c3 < n_valid && idx_lds[c3] >= 0 && idx_lds[c3] < T_kv;
        const float p0 = ok0 && lane_ok ? expf(sacc.x[0] * softmax_scale - m) * inv_l : 0.0f;
        const float p1 = ok1 && lane_ok ? expf(sacc.x[1] * softmax_scale - m) * inv_l : 0.0f;
        const float p2 = ok2 && lane_ok ? expf(sacc.x[2] * softmax_scale - m) * inv_l : 0.0f;
        const float p3 = ok3 && lane_ok ? expf(sacc.x[3] * softmax_scale - m) * inv_l : 0.0f;
        const float dl0 =
            ok0 && lane_ok ? dsa_to_float(dsa_from_float<C>(p0 * (dacc.x[0] - avg))) * softmax_scale : 0.0f;
        const float dl1 =
            ok1 && lane_ok ? dsa_to_float(dsa_from_float<C>(p1 * (dacc.x[1] - avg))) * softmax_scale : 0.0f;
        const float dl2 =
            ok2 && lane_ok ? dsa_to_float(dsa_from_float<C>(p2 * (dacc.x[2] - avg))) * softmax_scale : 0.0f;
        const float dl3 =
            ok3 && lane_ok ? dsa_to_float(dsa_from_float<C>(p3 * (dacc.x[3] - avg))) * softmax_scale : 0.0f;

        C* dlw                                         = dl_lds + (size_t)warp * 256;
        dlw[(size_t)r * 16 + g]                        = dsa_from_float<C>(dl0);
        dlw[(size_t)r * 16 + g + 4]                    = dsa_from_float<C>(dl1);
        dlw[(size_t)r * 16 + g + 8]                    = dsa_from_float<C>(dl2);
        dlw[(size_t)r * 16 + g + 12]                   = dsa_from_float<C>(dl3);
        pt_lds[(size_t)(g + 0) * SAF_HBLK + local_h]   = dsa_from_float<C>(p0);
        pt_lds[(size_t)(g + 4) * SAF_HBLK + local_h]   = dsa_from_float<C>(p1);
        pt_lds[(size_t)(g + 8) * SAF_HBLK + local_h]   = dsa_from_float<C>(p2);
        pt_lds[(size_t)(g + 12) * SAF_HBLK + local_h]  = dsa_from_float<C>(p3);
        dlt_lds[(size_t)(g + 0) * SAF_HBLK + local_h]  = dsa_from_float<C>(dl0);
        dlt_lds[(size_t)(g + 4) * SAF_HBLK + local_h]  = dsa_from_float<C>(dl1);
        dlt_lds[(size_t)(g + 8) * SAF_HBLK + local_h]  = dsa_from_float<C>(dl2);
        dlt_lds[(size_t)(g + 12) * SAF_HBLK + local_h] = dsa_from_float<C>(dl3);
        __syncthreads();

#    pragma unroll
        for (int pk = 0; pk < 16; pk += KFRAG) {
            DUFragment<matrix_a, 16, 16, KFRAG, C, row_major> dlfr;
            du_load_matrix_sync(dlfr, dlw + pk, 16);
#    pragma unroll
            for (int dc = 0; dc < NOUT; ++dc) {
                DUFragment<matrix_b, 16, 16, KFRAG, C, row_major> kb;
                du_load_matrix_sync(kb, k_lds + (size_t)pk * KLD + dc * 16, KLD);
                du_mma_sync(dqacc[dc], dlfr, kb, dqacc[dc]);
            }
        }

        const int rounds = CEIL_DIV(NOUT, 4);
        for (int rd = 0; rd < rounds; ++rd) {
            const int dc = rd * 4 + warp;
            DUFragment<accumulator, 16, 16, KFRAG, float> kacc;
            du_fill_fragment(kacc, 0.0f);
            for (int hg = 0; hg < SAF_HBLK; hg += KFRAG) {
                C* qtile = qd_lds + (size_t)warp * 2 * 256;
                C* dtile = qtile + 256;
                for (int u = lane; u < KFRAG * 16; u += 64) {
                    const int hr  = u / 16;
                    const int col = u % 16;
                    const int gh  = hb + hg + hr;
                    if (dc < NOUT && gh < H) {
                        const size_t at              = ((size_t)t * H + gh) * D + dc * 16 + col;
                        qtile[(size_t)hr * 16 + col] = dsa_load_as<C>(q, at);
                        dtile[(size_t)hr * 16 + col] =
                            dc * 16 + col < VALUE_DIM
                                ? dsa_load_as<C>(dout, ((size_t)t * H + gh) * VALUE_DIM + dc * 16 + col)
                                : dsa_from_float<C>(0.0f);
                    } else {
                        qtile[(size_t)hr * 16 + col] = dsa_from_float<C>(0.0f);
                        dtile[(size_t)hr * 16 + col] = dsa_from_float<C>(0.0f);
                    }
                }
                __syncwarp();
                if (dc < NOUT) {
                    DUFragment<matrix_a, 16, 16, KFRAG, C, row_major> pa, da;
                    du_load_matrix_sync(pa, pt_lds + hg, SAF_HBLK);
                    du_load_matrix_sync(da, dlt_lds + hg, SAF_HBLK);
                    DUFragment<matrix_b, 16, 16, KFRAG, C, row_major> db, qb;
                    du_load_matrix_sync(db, dtile, 16);
                    du_load_matrix_sync(qb, qtile, 16);
                    du_mma_sync(kacc, pa, db, kacc);
                    du_mma_sync(kacc, da, qb, kacc);
                }
                __syncwarp();
            }
            if (dc < NOUT && r < ck) {
                const int idx = idx_lds[kt * 16 + r];
                if (idx >= 0 && idx < T_kv) {
                    float* dst = dkv_float + (size_t)idx * D + dc * 16;
                    saf_atomic_add_mmac_row(dst, g, kacc.x[0], kacc.x[1], kacc.x[2], kacc.x[3]);
                }
            }
        }
        __syncthreads();
    }

    if (lane_ok && g == 0) atomicAdd(dsink + h, -p_sink * avg);

    if (lane_ok) {
#    pragma unroll
        for (int dc = 0; dc < NOUT; ++dc) {
            const int col     = dc * 16 + g;
            const size_t base = ((size_t)t * H + h) * D + col;
            dsa_store_from_float<C>(dq, base, dqacc[dc].x[0]);
            dsa_store_from_float<C>(dq, base + 4, dqacc[dc].x[1]);
            dsa_store_from_float<C>(dq, base + 8, dqacc[dc].x[2]);
            dsa_store_from_float<C>(dq, base + 12, dqacc[dc].x[3]);
        }
    }

    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (threadIdx.x == 0) last = atomicAdd(done_count, 1) == gridDim.x * gridDim.y - 1;
    __syncthreads();
    if (last) {
        const size_t n = (size_t)T_kv * D;
        for (size_t i = threadIdx.x; i < n; i += SAF_THREADS) dsa_store_from_float<C>(dkv, i, dkv_float[i]);
    }
}

#else

template<typename C, int ND, bool COMPACT_VALUES = false>
static __global__ void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_bwd_dtype_kernel(const void*, const void*, const void*, const void*, const float*, const float*,
                                 const int*, const int*, float*, float*, void*, void*, int*, int, int, int, int, int,
                                 float) {}

#endif

static __global__ void saf_zero_f32(float* p, long long n) {
    const long long stride = (long long)(gridDim.x * blockDim.x);
    for (long long i = (long long)(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += stride) p[i] = 0.f;
}

namespace ftrain {
namespace {
template<typename C>
static bool launch_sparse_attn_bwd_dtype(hipStream_t stream, const void* q, const void* kv, const void* out,
                                         const void* dout, const float* lse, const float* attn_sink,
                                         const int* topk_idxs, const int* topk_length, void* dq, float* dkv_float,
                                         float* dsink, void* dkv, int* done_count, int T, int H, int D, int D_v,
                                         int T_kv, int topk, float softmax_scale);
bool launch_sparse_attn_bwd_fused(hipStream_t stream, const void* q, const void* kv, const void* out, const void* dout,
                                  const float* lse, const float* attn_sink, const int* topk_idxs,
                                  const int* topk_length, void* dq, float* dkv_float, float* dsink, void* dkv,
                                  int* done_count, int T, int H, int D, int D_v, int T_kv, int topk,
                                  float softmax_scale);
bool launch_sparse_attn_bwd_fused2(hipStream_t stream, const void* q, const void* kv, const void* out, const void* dout,
                                   const float* lse, const float* attn_sink, const int* topk_idxs,
                                   const int* topk_length, void* dq, float* dkv_float, float* dsink, void* dkv,
                                   int* done_count, void* pdl_ws, int T, int H, int D, int D_v, int T_kv, int topk,
                                   float softmax_scale);

template<typename C>
static bool launch_sparse_attn_bwd_dtype(hipStream_t stream, const void* q, const void* kv, const void* out,
                                         const void* dout, const float* lse, const float* attn_sink,
                                         const int* topk_idxs, const int* topk_length, void* dq, float* dkv_float,
                                         float* dsink, void* dkv, int* done_count, int T, int H, int D, int D_v,
                                         int T_kv, int topk, float softmax_scale) {
    constexpr int kfrag = DsaMmacTraits<C>::k;
    constexpr int pad   = sizeof(C) == 4 ? 10 : 8;
    const int kld       = D + pad;
    const size_t shm    = (size_t)(16 * kld + 5 * 1024) * sizeof(C) + (size_t)topk * sizeof(int);
    if (!dsa_shared_memory_fits(shm)) return false;
    dim3 grid(T, CEIL_DIV(H, SAF_HBLK));
    hipLaunchKernelGGL((saf_zero_f32), dim3(256), 256, 0, stream, dkv_float, (long long)T_kv * D);
#define SAB_DTYPE_CASE(NK)                                                                                          \
    case NK:                                                                                                        \
        hipLaunchKernelGGL((sparse_attn_bwd_dtype_kernel<C, NK>), grid, SAF_THREADS, shm, stream, q, kv, out, dout, \
                           lse, attn_sink, topk_idxs, topk_length, dkv_float, dsink, dq, dkv, done_count, T, H, D,  \
                           T_kv, topk, softmax_scale);                                                              \
        break
    if (D_v != D) {
        hipLaunchKernelGGL((sparse_attn_bwd_dtype_kernel<C, 576 / kfrag, true>), grid, SAF_THREADS, shm, stream, q, kv,
                           out, dout, lse, attn_sink, topk_idxs, topk_length, dkv_float, dsink, dq, dkv, done_count, T,
                           H, D, T_kv, topk, softmax_scale);
    } else if constexpr (kfrag == 8) {
        switch (D / kfrag) {
            SAB_DTYPE_CASE(8);
            SAB_DTYPE_CASE(16);
            SAB_DTYPE_CASE(32);
            SAB_DTYPE_CASE(56);
            SAB_DTYPE_CASE(64);
            SAB_DTYPE_CASE(72);
            default: return false;
        }
    } else {
        switch (D / kfrag) {
            SAB_DTYPE_CASE(4);
            SAB_DTYPE_CASE(8);
            SAB_DTYPE_CASE(16);
            SAB_DTYPE_CASE(28);
            SAB_DTYPE_CASE(32);
            SAB_DTYPE_CASE(36);
            default: return false;
        }
    }
#undef SAB_DTYPE_CASE
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "fp32_sparse_attn_bwd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}

static bool launch_sparse_attn_bwd_fused(hipStream_t stream, const void* q, const void* kv, const void* out,
                                         const void* dout, const float* lse, const float* attn_sink,
                                         const int* topk_idxs, const int* topk_length, void* dq, float* dkv_float,
                                         float* dsink, void* dkv, int* done_count, int T, int H, int D, int D_v,
                                         int T_kv, int topk, float softmax_scale) {
    if (!sparse_attn_fused_supported()) return false;
    if (D % 16 != 0 || D > SAF_MAX_D || (D_v != D && !(D == 576 && D_v == 512))) return false;
    // H may be any value >= 1: partial 64-head slabs are masked per-lane.
    if (H <= 0 || T <= 0 || topk <= 0) return false;

    {
        return launch_sparse_attn_bwd_dtype<float>(stream, q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length, dq,
                                                   dkv_float, dsink, dkv, done_count, T, H, D, D_v, T_kv, topk,
                                                   softmax_scale);
    }
}

static bool launch_sparse_attn_bwd_fused2(hipStream_t stream, const void* q, const void* kv, const void* out,
                                          const void* dout, const float* lse, const float* attn_sink,
                                          const int* topk_idxs, const int* topk_length, void* dq, float* dkv_float,
                                          float* dsink, void* dkv, int* done_count, void* pdl_ws, int T, int H, int D,
                                          int D_v, int T_kv, int topk, float softmax_scale) {
    if (!sparse_attn_fused_supported()) return false;
    return false;
}
}  // namespace

bool fp32SparseAttnBwdLaunchFits(int width, int columns) {
    if (width != 512 && width != 576) return false;
    return dsa_shared_memory_fits((16ULL * (width + 10) + 5120) * 4 + static_cast<std::size_t>(columns) * 4);
}

bool launchFp32SparseAttnBwdFused2(FTrainStream stream, const void* q, const void* kv, const void* out,
                                   const void* dout, const float* lse, const float* attn_sink, const int* topk_indices,
                                   const int* topk_length, void* dq, float* dkv_float, float* dsink, void* dkv,
                                   int* done_count, void* pdl_ws, int tokens, int heads, int head_dim,
                                   int backward_width, int kv_tokens, int selection_width, float softmax_scale) {
    return launch_sparse_attn_bwd_fused2(stream, q, kv, out, dout, lse, attn_sink, topk_indices, topk_length, dq,
                                         dkv_float, dsink, dkv, done_count, pdl_ws, tokens, heads, head_dim,
                                         backward_width, kv_tokens, selection_width, softmax_scale);
}

bool launchFp32SparseAttnBwdFused(FTrainStream stream, const void* q, const void* kv, const void* out, const void* dout,
                                  const float* lse, const float* attn_sink, const int* topk_indices,
                                  const int* topk_length, void* dq, float* dkv_float, float* dsink, void* dkv,
                                  int* done_count, int tokens, int heads, int head_dim, int backward_width,
                                  int kv_tokens, int selection_width, float softmax_scale) {
    return launch_sparse_attn_bwd_fused(stream, q, kv, out, dout, lse, attn_sink, topk_indices, topk_length, dq,
                                        dkv_float, dsink, dkv, done_count, tokens, heads, head_dim, backward_width,
                                        kv_tokens, selection_width, softmax_scale);
}

bool fp32SparseAttnBwdSparseAttnFusedSupported() { return sparse_attn_fused_supported(); }

void fp32SparseAttnBwdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    const hipError_t status = hipMemsetAsync(memory, 0, static_cast<std::size_t>(bytes), stream);
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Workspace zero-fill failed: %s", hipGetErrorString(status));
    }
}

}  // namespace ftrain
