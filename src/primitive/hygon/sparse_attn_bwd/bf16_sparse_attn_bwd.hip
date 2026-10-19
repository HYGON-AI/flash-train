// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

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
#include "flash_train/primitive/hygon/sparse_attn_bwd/bf16_sparse_attn_bwd.hpp"

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

namespace hip_dsa_numeric {

__device__ __forceinline__ static uint16_t bf16_rne(float value) { return __builtin_bit_cast(uint16_t, (__bf16)value); }

__device__ __forceinline__ static uint32_t bf16_pair(float lo, float hi) {
#if defined(__gfx938__)
    // Exact call shape from the local Hygon FlashMLA and FlashAttention sources.
    // gfx936 lacks hcu-bf16-insts; use the verified compiler lowering below.
    auto bits = __builtin_hcu_cvt_pk_bf16_f32(0, lo, 0, hi, 0);
    return __builtin_bit_cast(uint32_t, bits);
#else
    using Float2  = float __attribute__((ext_vector_type(2)));
    using BFloat2 = __bf16 __attribute__((ext_vector_type(2)));
    BFloat2 bits  = __builtin_convertvector((Float2){lo, hi}, BFloat2);
    return __builtin_bit_cast(uint32_t, bits);
#endif
}
}  // namespace hip_dsa_numeric

namespace hip_dsa_numeric {

__device__ __forceinline__ static float softmax_exp(float x) {
#pragma clang fp contract(off)
#if defined(__gfx936__) || defined(__gfx938__)
    // Keep DTK's rounded two-part log2(e) reduction. The reduced exp2
    // argument is near [-0.5,0.5] whenever expf has a finite nonzero result,
    // so its generic exp2 underflow adjustment is redundant. The original
    // final expf underflow/overflow cuts remain below. Implicit contraction
    // of (x*hi)-n would change last bits; explicit compensation FMAs remain.
    const float hi = __uint_as_float(0x3fb8aa3bu);
    const float lo = __uint_as_float(0x32a5705fu);
    const float y  = x * hi;
    const float n  = __builtin_rintf(y);
    float r        = __builtin_fmaf(x, hi, -y);
    r              = __builtin_fmaf(x, lo, r);
    const float f  = (y - n) + r;
    int exponent;
    // Match the native conversion used by DTK without an out-of-range C++
    // float-to-int conversion for nonfinite inputs (final cuts below handle them).
    asm("v_cvt_i32_f32 %0, %1" : "=v"(exponent) : "v"(n));
    float z = __builtin_amdgcn_ldexpf(__builtin_amdgcn_exp2f(f), exponent);
    // Preserve DTK's exact final range cuts; only the redundant exp2-input
    // correction is removed. Outside the finite output range, these cuts
    // determine the result. NaNs pass through the same arithmetic path.
    if (x < __uint_as_float(0xc2ce8ed0u)) z = 0.0f;
    if (x > __uint_as_float(0x42b17218u)) z = INFINITY;
    return z;
#else
    return expf(x);
#endif
}
}  // namespace hip_dsa_numeric

// Adapted from fused_dsa/hip_dsa/kernels/sparse_attn_bwd.hip.

inline bool sparse_attn_bwd_register_fused_shape(int T, int H, int D, int T_kv, int topk) {
    return D == 512 && (H == 64 || H == 128) && T >= 256 && T_kv % 8 == 0 && (topk == 128 || topk == 160);
}
typedef uint16_t bf16;

// Cached H128 dQ has an 8,448-byte K region. Keep index residency bounded
// so the extra LDS does not reduce the long-row fallback's capacity.
constexpr int SAF_DQ_INDEX_MAX_TOPK = 2048;

// Keep launch allocation and cached P/dS row layout identical.
template<int HC, int D>
constexpr int saf_dkv_ldm = HC + ((HC == 128 || (HC == 64 && D == 576)) ? 4 : 2);

// Match SM90 preprocess conventions without evaluating inf - inf. Delta is
// computed separately from the supplied O/dO, even for non-finite normalizers.
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

// Parallel last-N-block cast epilogue for dkv_float (f32 atomics) -> dkv
// (bf16).  Every block fences its dkv/dsink atomics before incrementing
// done_count, so once done_count reaches total all contributions are
// visible device-wide.  The last SAF_CAST_BLOCKS blocks to arrive each cast
// one 1/SAF_CAST_BLOCKS slice: the cast leaves the critical path of the
// kernel instead of serializing in the single final block.  prev is this
// block's atomicAdd return value (arrival order, 0-based).
__device__ __forceinline__ static void saf_multi_cast(const int* done_count, const float* dkv_float, bf16* dkv,
                                                      int T_kv, int D, int total, int prev) {
    constexpr int NC = 32;
    const int ncast  = (total < NC) ? total : NC;
    if (prev < total - ncast) return;
    const int slot = prev - (total - ncast);
    if (threadIdx.x == 0) {
        const volatile int* dc = (const volatile int*)done_count;
        while (dc[0] < total) {}
    }
    __syncthreads();
    const long n4      = ((long)T_kv * D) >> 2;  // D % 16 == 0
    const long per     = (n4 + ncast - 1) / ncast;
    const long s0      = (long)slot * per;
    const long s1      = (s0 + per < n4) ? s0 + per : n4;
    const float4* src4 = (const float4*)dkv_float;
    uint64_t* dst      = (uint64_t*)dkv;  // 4 bf16 packed per u64
    long i             = s0 + threadIdx.x;
    for (; i + 3 * (long)SAF_THREADS < s1; i += 4 * (long)SAF_THREADS) {
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const float4 v                 = src4[i + (long)u * SAF_THREADS];
            const uint64_t b0              = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(v.x);
            const uint64_t b1              = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(v.y);
            const uint64_t b2              = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(v.z);
            const uint64_t b3              = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(v.w);
            dst[i + (long)u * SAF_THREADS] = b0 | (b1 << 16) | (b2 << 32) | (b3 << 48);
        }
    }
    for (; i < s1; i += SAF_THREADS) {
        const float4 v    = src4[i];
        const uint64_t b0 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(v.x);
        const uint64_t b1 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(v.y);
        const uint64_t b2 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(v.z);
        const uint64_t b3 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(v.w);
        dst[i]            = b0 | (b1 << 16) | (b2 << 32) | (b3 << 48);
    }
}

// Fused2 dKV atomics use the MMAC-native interleaved column order inside the
// temporary fp32 buffer: each four-float group contains columns {g,g+4,
// g+8,g+12}.  Convert that private layout back to the public contiguous bf16
// layout while performing the existing final cast.  This keeps the x4 atomic
// path free of the lane-shuffle transpose used by the legacy helper.
__device__ __forceinline__ static void saf_multi_cast_swizzled(const int* done_count, const float* dkv_float, bf16* dkv,
                                                               int T_kv, int D, int total, int prev) {
    constexpr int NC = 32;
    const int ncast  = (total < NC) ? total : NC;
    if (prev < total - ncast) return;
    const int slot = prev - (total - ncast);
    if (threadIdx.x == 0) {
        const volatile int* dc = (const volatile int*)done_count;
        while (dc[0] < total) {}
    }
    __syncthreads();
    const long tiles   = (long)T_kv * (D >> 4);
    const long per     = (tiles + ncast - 1) / ncast;
    const long s0      = (long)slot * per;
    const long s1      = (s0 + per < tiles) ? s0 + per : tiles;
    const float4* src4 = (const float4*)dkv_float;
    uint64_t* dst      = (uint64_t*)dkv;
    for (long tile = s0 + threadIdx.x; tile < s1; tile += SAF_THREADS) {
        const long base4   = tile * 4;
        const float4 a0    = src4[base4 + 0];
        const float4 a1    = src4[base4 + 1];
        const float4 a2    = src4[base4 + 2];
        const float4 a3    = src4[base4 + 3];
        const uint64_t b00 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.x);
        const uint64_t b01 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.x);
        const uint64_t b02 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.x);
        const uint64_t b03 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.x);
        const uint64_t b10 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.y);
        const uint64_t b11 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.y);
        const uint64_t b12 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.y);
        const uint64_t b13 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.y);
        const uint64_t b20 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.z);
        const uint64_t b21 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.z);
        const uint64_t b22 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.z);
        const uint64_t b23 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.z);
        const uint64_t b30 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.w);
        const uint64_t b31 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.w);
        const uint64_t b32 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.w);
        const uint64_t b33 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.w);
        const long out8    = tile * 4;
        dst[out8 + 0]      = b00 | (b01 << 16) | (b02 << 32) | (b03 << 48);
        dst[out8 + 1]      = b10 | (b11 << 16) | (b12 << 32) | (b13 << 48);
        dst[out8 + 2]      = b20 | (b21 << 16) | (b22 << 32) | (b23 << 48);
        dst[out8 + 3]      = b30 | (b31 << 16) | (b32 << 32) | (b33 << 48);
    }
}

// Private accumulator layout [key/8, column/16, group, key%8, 4 values].
__device__ __forceinline__ static void saf_multi_cast_keypacked(const int* done_count, const float* dkv_float,
                                                                bf16* dkv, int T_kv, int D, int total, int prev) {
    constexpr int NC = 32;
    const int ncast  = (total < NC) ? total : NC;
    if (prev < total - ncast) return;
    const int slot = prev - (total - ncast);
    if (threadIdx.x == 0) {
        const volatile int* dc = (const volatile int*)done_count;
        while (dc[0] < total) {}
    }
    __syncthreads();
    const long tiles   = (long)T_kv * (D >> 4);
    const long per     = (tiles + ncast - 1) / ncast;
    const long s0      = (long)slot * per;
    const long s1      = (s0 + per < tiles) ? s0 + per : tiles;
    const float4* src4 = (const float4*)dkv_float;
    uint64_t* dst      = (uint64_t*)dkv;
    for (long tile = s0 + threadIdx.x; tile < s1; tile += SAF_THREADS) {
        const long key = tile / (D / 16), col = tile % (D / 16);
        const long base4   = ((key / 8) * (D / 16) + col) * (4 * 8) + key % 8;
        const float4 a0    = src4[base4 + 0];
        const float4 a1    = src4[base4 + 8];
        const float4 a2    = src4[base4 + 16];
        const float4 a3    = src4[base4 + 24];
        const uint64_t b00 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.x);
        const uint64_t b01 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.x);
        const uint64_t b02 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.x);
        const uint64_t b03 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.x);
        const uint64_t b10 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.y);
        const uint64_t b11 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.y);
        const uint64_t b12 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.y);
        const uint64_t b13 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.y);
        const uint64_t b20 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.z);
        const uint64_t b21 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.z);
        const uint64_t b22 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.z);
        const uint64_t b23 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.z);
        const uint64_t b30 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.w);
        const uint64_t b31 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.w);
        const uint64_t b32 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.w);
        const uint64_t b33 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.w);
        const long out8    = tile * 4;
        dst[out8 + 0]      = b00 | (b01 << 16) | (b02 << 32) | (b03 << 48);
        dst[out8 + 1]      = b10 | (b11 << 16) | (b12 << 32) | (b13 << 48);
        dst[out8 + 2]      = b20 | (b21 << 16) | (b22 << 32) | (b23 << 48);
        dst[out8 + 3]      = b30 | (b31 << 16) | (b32 << 32) | (b33 << 48);
    }
}

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;
typedef __hip_bfloat16 hbf16;
using saf_intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
extern __device__ saf_intx4 saf_raw_buffer_load_i32x4(saf_intx4 srsrc, int voffset, int soffset,
                                                      int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
extern __device__ int saf_raw_buffer_load_i32(saf_intx4 srsrc, int voffset, int soffset,
                                              int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");

// ---------------------------------------------------------------------------
// Streaming (non-temporal) helpers for the packed P/dS workspace.
//
// The (T, topk, H) packed P/dS buffer is written once by the stats pass.
// The dKV/dQ column CTAs stream through it with long reuse distances. Caching it
// in the 8 MB L2 evicts the small, heavily re-read buffers (KV rows for the
// stats QK gather, the BF16 KV rows for the dQ gather) and the FP32 dKV
// atomic accumulator, which costs a large amount of extra DRAM traffic.
// Marking only the workspace traffic non-temporal keeps those hot buffers
// resident.  glc_slc=3 is the encoding clang emits for
// __builtin_nontemporal_load/store (bit0=glc L1 bypass, bit1=slc L2 bypass);
// the accesses stay fully coherent, so results are unchanged.
typedef unsigned int __attribute__((ext_vector_type(4))) saf_u4e;

__device__ __forceinline__ static saf_u4e saf_nt_load_u4(const void* p) {
    return __builtin_nontemporal_load((const saf_u4e*)p);
}
__device__ __forceinline__ static unsigned int saf_nt_load_u32(const void* p) {
    return __builtin_nontemporal_load((const unsigned int*)p);
}
__device__ __forceinline__ static unsigned long long saf_nt_load_u64(const void* p) {
    return __builtin_nontemporal_load((const unsigned long long*)p);
}
__device__ __forceinline__ static void saf_nt_store_u32(void* p, unsigned int v) {
    __builtin_nontemporal_store(v, (unsigned int*)p);
}
#    if defined(__gfx936__) || defined(__gfx938__)
// Hygon gfx936/gfx938 expose a native 128-bit global FP32 atomic add.  The dKV
// epilogue below transposes each four-lane MMAC fragment so its output is
// contiguous and can use one x4 atomic instead of four scalar atomics.
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

// The fused2-only variant writes the MMAC's native interleaved four-column
// fragment directly.  Its caller uses saf_multi_cast_swizzled to restore the
// public contiguous layout after all block atomics have arrived.
__device__ __forceinline__ static void saf_atomic_add_mmac_row_swizzled(float* row16, int g, float x0, float x1,
                                                                        float x2, float x3) {
#    if defined(__gfx936__) || defined(__gfx938__)
    saf_floatx4 v;
    v[0] = x0;
    v[1] = x1;
    v[2] = x2;
    v[3] = x3;
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
// ---------------------------------------------------------------------------
// Single-kernel sparse_attn_bwd (dkv + dq fused, grid (T, nslab)).
//
// One block per (query row, 64-head slab) runs ONE pass over the row's topk
// keys.  No stats re-pass: the softmax normalization is recovered from the
// forward's KV-only lse and the sink logit — m = max(lse, sink),
// l = exp(lse-m) + exp(sink-m), P = exp(S*scale - m) / l — and the softmax
// expectation avg = sum_k P_k (dO . K_k) is evaluated as the cheap
// elementwise dot avg = dO . O of the forward output O (K == V, and the
// sink logit has no V so it drops out of avg).  Per-chunk work is therefore
// one S/DV recompute (2 MMACs per 16x16x16 tile) + the dq MMACs + the dkv
// MMACs:
//   P = exp(S*scale - m) * inv_l,  dl = scale * P * (DV - avg)
//   dkv = P^T @ dO + dl^T @ Q  (registers, one atomicAdd per (k, d)),
//   dq  = dl @ K              (registers, written straight out).
// dkv is accumulated in a caller-zeroed (T_kv, D) f32 buffer; the f32->bf16
// cast runs as a tiny separate epilogue (required for fp32-accurate atomic
// accumulation).  Peak memory stays flat: the only extra inputs are the
// fwd's out/lse, and no S/DV workspaces exist.  H may be any value >= 1
// (partial slabs are masked per-lane; dkv tiles are re-distributed across
// the active warps); for H > 64 the slab blocks run in parallel.
// ---------------------------------------------------------------------------

template<int ND, int TOPK = 0, bool COMPACT_VALUES = false>
__global__ static void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_bwd_fused_kernel(const bf16* __restrict__ q,           // (T, H, D)
                                 const bf16* __restrict__ kv,          // (T_kv, D)
                                 const bf16* __restrict__ out,         // (T, H, D) fwd output (bf16)
                                 const bf16* __restrict__ dout,        // (T, H, D)
                                 const float* __restrict__ lse,        // (T, H) KV-only logsumexp (fwd)
                                 const float* __restrict__ attn_sink,  // (H,)
                                 const int* __restrict__ idxs,         // (T, topk)
                                 const int* __restrict__ topk_length,
                                 float* __restrict__ dkv_float,  // (T_kv, D) f32 (zeroed by caller)
                                 float* __restrict__ dsink,      // (H,) f32 (zeroed by caller)
                                 bf16* __restrict__ dq,          // (T, H, D)
                                 bf16* __restrict__ dkv,         // (T_kv, D) bf16 output
                                 int* __restrict__ done_count,   // zeroed by caller; last-block cast
                                 int T, int H, int D_rt, int T_kv, int topk_rt, float softmax_scale) {
    constexpr int D         = ND * 16;
    constexpr int VALUE_DIM = COMPACT_VALUES ? 512 : D;
    const int topk          = (TOPK > 0) ? TOPK : topk_rt;
    (void)D_rt;
    const int t  = blockIdx.x;
    const int hb = blockIdx.y * SAF_HBLK;  // one (t, 64-head slab) per block
    if (hb >= H) return;
    const int lane = threadIdx.x & 63;
    const int warp = threadIdx.x >> 6;
    const int r0   = warp * 16;
    const int r    = lane & 15;
    const int g    = lane >> 4;
    const int KLD  = D + 8;

    extern __shared__ char smem_raw[];
    hbf16* k_lds  = (hbf16*)smem_raw;              // 16 keys x KLD
    hbf16* pt_lds = (hbf16*)(k_lds + 16 * KLD);    // P^T / dl^T: 4 warps x 256 x 2
    int* idx_lds  = (int*)(pt_lds + 4 * 256 * 2);  // this row's topk indices

    int n_valid_t = topk;
    if (topk_length != NULL) {
        int tl = topk_length[t];
        if (tl < n_valid_t) n_valid_t = tl > 0 ? tl : 0;
    }

    const hbf16* qrow = (const hbf16*)(q + (size_t)t * H * D);
    const hbf16* orow = (const hbf16*)(out + (size_t)t * H * VALUE_DIM);
    const hbf16* drow = (const hbf16*)(dout + (size_t)t * H * VALUE_DIM);
    const int* irow   = idxs + (size_t)t * topk;
    const int ntiles  = (topk + 15) >> 4;

    __shared__ int keys_all_valid;
    if (threadIdx.x == 0) keys_all_valid = (n_valid_t == topk);
    __syncthreads();
    for (int i = threadIdx.x; i < topk; i += SAF_THREADS) {
        const int idx = irow[i];
        idx_lds[i]    = idx;
        if (idx < 0 || idx >= T_kv) atomicExch(&keys_all_valid, 0);
    }
    __syncthreads();

    {
        const int h        = hb + r0 + r;
        const bool lane_ok = (h < H);
        const int hq       = hb + r0;  // first global Q row of this warp

        // Q/dout A-fragments: for D <= 128 (ND <= 8) cache this slab's
        // fragments once — the S/DV MMACs re-read the same 64-head slice
        // every key round.  For larger D the cache would overflow the
        // register file, so reload the two fragments per chunk instead
        // (extra global traffic is cheap next to the spill traffic it
        // avoids).
        constexpr int QD_CACHE = (ND <= 8) ? ND : 2;
        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> qf[QD_CACHE], df[QD_CACHE];
        auto load_qd = [&](int kk, DUFragment<matrix_a, 16, 16, 16, hbf16, row_major>& qfr,
                           DUFragment<matrix_a, 16, 16, 16, hbf16, row_major>& dfr) {
            const int row_h       = lane_ok ? h : (H - 1);  // clamp reads
            const hbf16* qp       = (const hbf16*)(qrow + (size_t)row_h * D + kk * 16);
            *(uint64_t*)&qfr.x[0] = *(const uint64_t*)(qp + (g << 2));
            const hbf16* dp       = (const hbf16*)(drow + (size_t)row_h * VALUE_DIM + kk * 16);
            if (kk * 16 < VALUE_DIM) {
                *(uint64_t*)&dfr.x[0] = *(const uint64_t*)(dp + (g << 2));
            } else {
                dfr.x[0] = dfr.x[1] = dfr.x[2] = dfr.x[3] = hbf16(0);
            }
            if (!lane_ok) {
                qfr.x[0] = qfr.x[1] = qfr.x[2] = qfr.x[3] = hbf16(0);
                dfr.x[0] = dfr.x[1] = dfr.x[2] = dfr.x[3] = hbf16(0);
            }
        };
        if constexpr (ND <= 8) {
#    pragma unroll
            for (int kk = 0; kk < ND; kk++) load_qd(kk, qf[kk], df[kk]);
        }

        // Reuse forward LSE and FP32 dO.O for both dS and d_sink.
        float m = 0.0f, inv_l = 0.0f, P_sink = 0.0f, avg = 0.0f;
        {
            if (lane_ok) {
                const float lse_h  = lse[(size_t)t * H + h];
                const float sink_h = attn_sink[h];
                saf_backward_normalization(lse_h, sink_h, m, inv_l, P_sink);
                {
                    const hbf16* orp = orow + (size_t)h * VALUE_DIM;
                    const hbf16* dr  = drow + (size_t)h * VALUE_DIM;
                    float p          = 0.0f;
#    pragma unroll
                    for (int kk = 0; kk < VALUE_DIM / 16; kk++) {
                        const int c  = kk * 16 + (g << 2);
                        p           += (float)orp[c + 0] * (float)dr[c + 0];
                        p           += (float)orp[c + 1] * (float)dr[c + 1];
                        p           += (float)orp[c + 2] * (float)dr[c + 2];
                        p           += (float)orp[c + 3] * (float)dr[c + 3];
                    }
                    avg = p;
                }
            }
            avg += __shfl_xor(avg, 16);
            avg += __shfl_xor(avg, 32);
        }

        auto ck_for = [&](int kkt) {
            int c = topk - kkt * 16;
            return (c > 16) ? 16 : (c < 0 ? 0 : c);
        };

        // Software-pipelined K staging.  The gather for chunk kt+1 issues
        // into private registers while chunk kt's S/DV MMACs run (the loads
        // are independent of k_lds, so the compiler cannot re-serialize
        // them); after the compute barrier the registers are committed to
        // k_lds.  This hides the gather latency behind the matrix work in a
        // rolled loop (a fully unrolled loop would blow the I-cache).
        constexpr int PREF_MAX = 2 * ((16 * (D / 8) + SAF_THREADS - 1) / SAF_THREADS);
        auto prefetch_keys     = [&](int kt, int ck, uint64_t* regs) -> int {
            int cnt = 0;
            if (keys_all_valid) {
                for (int u = threadIdx.x; u < ck * (D / 8); u += SAF_THREADS) {
                    int n               = u / (D / 8);
                    int d8              = u - n * (D / 8);
                    const int idx       = idx_lds[kt * 16 + n];
                    const uint64_t* src = (const uint64_t*)(kv + (size_t)idx * D + d8 * 8);
                    regs[2 * cnt]       = src[0];
                    regs[2 * cnt + 1]   = src[1];
                    cnt++;
                }
                return cnt;
            }
            for (int u = threadIdx.x; u < ck * (D / 8); u += SAF_THREADS) {
                int n               = u / (D / 8);
                int d8              = u - n * (D / 8);
                const int key       = kt * 16 + n;
                int idx             = idx_lds[key];
                const bool valid    = key < n_valid_t && idx >= 0 && idx < T_kv;
                const int safe_idx  = valid ? idx : 0;
                const uint64_t* src = (const uint64_t*)(kv + (size_t)safe_idx * D + d8 * 8);
                const uint64_t x0 = src[0], x1 = src[1];
                regs[2 * cnt]     = valid ? x0 : 0;
                regs[2 * cnt + 1] = valid ? x1 : 0;
                cnt++;
            }
            return cnt;
        };
        auto commit_keys = [&](int ck, const uint64_t* regs, int cnt) {
            for (int j = 0; j < cnt; j++) {
                int u         = threadIdx.x + j * SAF_THREADS;
                int n         = u / (D / 8);
                int d8        = u - n * (D / 8);
                uint64_t* dst = (uint64_t*)(k_lds + (size_t)n * KLD + d8 * 8);
                dst[0]        = regs[2 * j];
                dst[1]        = regs[2 * j + 1];
            }
            if (ck < 16) {
                for (int u = threadIdx.x; u < 16 * KLD; u += SAF_THREADS) {
                    if (u / KLD >= ck) k_lds[u] = hbf16(0);
                }
            }
        };

        DUFragment<accumulator, 16, 16, 16, float> sacc, dacc;

        // ---- Single pass: S/DV -> P/dl -> dq + dkv per chunk.  The P/dl
        // tiles are staged once (LDS, per-warp regions) and feed both the
        // dq MMACs (own warp's 16 heads) and the dkv MMACs (active warps
        // own contiguous 16-wide d-tile slices; one atomicAdd per (k, d)).
        // H may be any value >= 1: partial slabs keep their valid heads,
        // phantom heads stage 0. ----
        const int nsub = (H - hb + 15) >> 4;
        const int ns   = (nsub < 4) ? nsub : 4;

        // dq accumulator: 16 heads x all D per warp.  All ND fragments are
        // live across the whole key loop; the surrounding passes must leave
        // enough registers for this array to stay resident (otherwise the
        // compiler spills it to memory and dq slows down ~2x).
        DUFragment<accumulator, 16, 16, 16, float> dqacc[ND];
#    pragma unroll
        for (int d16 = 0; d16 < ND; d16++) du_fill_fragment(dqacc[d16], 0.0f);

        {
            uint64_t pref[PREF_MAX];
            int cnt = prefetch_keys(0, ck_for(0), pref);
            commit_keys(ck_for(0), pref, cnt);
        }
        __syncthreads();  // k_lds ready
#    pragma unroll 1
        for (int kt = 0; kt < ntiles; kt++) {
            uint64_t pref[PREF_MAX];
            int cnt       = 0;
            const int nkt = kt + 1;
            if (nkt < ntiles) cnt = prefetch_keys(nkt, ck_for(nkt), pref);
            const int ck = ck_for(kt);

            du_fill_fragment(sacc, 0.0f);
            du_fill_fragment(dacc, 0.0f);
            if (hq < H) {
                if constexpr (ND > 8) {
#    pragma unroll 1
                    for (int kk = 0; kk < ND; kk += 2) {
                        DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr0, bfr1;
                        du_load_matrix_sync(bfr0, k_lds + (size_t)kk * 16, KLD);
                        du_load_matrix_sync(bfr1, k_lds + (size_t)(kk + 1) * 16, KLD);
                        load_qd(kk, qf[0], df[0]);
                        load_qd(kk + 1, qf[1], df[1]);
                        du_mma_sync(sacc, qf[0], bfr0, sacc);
                        du_mma_sync(dacc, df[0], bfr0, dacc);
                        du_mma_sync(sacc, qf[1], bfr1, sacc);
                        du_mma_sync(dacc, df[1], bfr1, dacc);
                    }
                } else {
#    pragma unroll 1
                    for (int kk = 0; kk < ND; kk++) {
                        DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr;
                        du_load_matrix_sync(bfr, k_lds + (size_t)kk * 16, KLD);
                        du_mma_sync(sacc, qf[kk], bfr, sacc);
                        du_mma_sync(dacc, df[kk], bfr, dacc);
                    }
                }
            }
            {
                const int c0 = kt * 16 + g;
                const int c1 = c0 + 4, c2 = c0 + 8, c3 = c0 + 12;
                const bool ok0 = (c0 < n_valid_t) && (idx_lds[c0] >= 0) && (idx_lds[c0] < T_kv);
                const bool ok1 = (c1 < n_valid_t) && (idx_lds[c1] >= 0) && (idx_lds[c1] < T_kv);
                const bool ok2 = (c2 < n_valid_t) && (idx_lds[c2] >= 0) && (idx_lds[c2] < T_kv);
                const bool ok3 = (c3 < n_valid_t) && (idx_lds[c3] >= 0) && (idx_lds[c3] < T_kv);
                // Phantom heads (h >= H) get -inf so their exp(-inf - m) = 0.
                const float s0 = lane_ok ? sacc.x[0] * softmax_scale : -INFINITY;
                const float s1 = lane_ok ? sacc.x[1] * softmax_scale : -INFINITY;
                const float s2 = lane_ok ? sacc.x[2] * softmax_scale : -INFINITY;
                const float s3 = lane_ok ? sacc.x[3] * softmax_scale : -INFINITY;
                const float p0 = (ok0 && lane_ok) ? hip_dsa_numeric::softmax_exp(s0 - m) * inv_l : 0.0f;
                const float p1 = (ok1 && lane_ok) ? hip_dsa_numeric::softmax_exp(s1 - m) * inv_l : 0.0f;
                const float p2 = (ok2 && lane_ok) ? hip_dsa_numeric::softmax_exp(s2 - m) * inv_l : 0.0f;
                const float p3 = (ok3 && lane_ok) ? hip_dsa_numeric::softmax_exp(s3 - m) * inv_l : 0.0f;
                const float d0 = dacc.x[0], d1 = dacc.x[1];
                const float d2 = dacc.x[2], d3 = dacc.x[3];
                // Guard dl so phantom DV slots cannot turn 0 * stale/NaN
                // into NaN that poisons the P/dl tiles.
                const float dl0 =
                    (ok0 && lane_ok)
                        ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p0 * (d0 - avg)) << 16) * softmax_scale
                        : 0.0f;
                const float dl1 =
                    (ok1 && lane_ok)
                        ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p1 * (d1 - avg)) << 16) * softmax_scale
                        : 0.0f;
                const float dl2 =
                    (ok2 && lane_ok)
                        ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p2 * (d2 - avg)) << 16) * softmax_scale
                        : 0.0f;
                const float dl3 =
                    (ok3 && lane_ok)
                        ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p3 * (d3 - avg)) << 16) * softmax_scale
                        : 0.0f;
                // Stage P^T / dl^T (own warp region; phantom heads stage 0).
                hbf16* pw                     = pt_lds + warp * 256;
                hbf16* dw                     = pt_lds + 4 * 256 + warp * 256;
                pw[(size_t)(g + 0) * 16 + r]  = __float2bfloat16(lane_ok ? p0 : 0.0f);
                pw[(size_t)(g + 4) * 16 + r]  = __float2bfloat16(lane_ok ? p1 : 0.0f);
                pw[(size_t)(g + 8) * 16 + r]  = __float2bfloat16(lane_ok ? p2 : 0.0f);
                pw[(size_t)(g + 12) * 16 + r] = __float2bfloat16(lane_ok ? p3 : 0.0f);
                dw[(size_t)(g + 0) * 16 + r]  = __float2bfloat16(lane_ok ? dl0 : 0.0f);
                dw[(size_t)(g + 4) * 16 + r]  = __float2bfloat16(lane_ok ? dl1 : 0.0f);
                dw[(size_t)(g + 8) * 16 + r]  = __float2bfloat16(lane_ok ? dl2 : 0.0f);
                dw[(size_t)(g + 12) * 16 + r] = __float2bfloat16(lane_ok ? dl3 : 0.0f);
            }
            __syncthreads();  // pt_lds for all warps visible

            // dq: this warp's 16 heads x all D, accumulated in registers
            // (fully unrolled so the k_lds B-fragment loads pipeline across
            // the whole D range).
            if (hq < H) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, col_major> dlf;
                du_load_matrix_sync(dlf, pt_lds + 4 * 256 + warp * 256, 16);
#    pragma unroll
                for (int d16 = 0; d16 < ND; d16++) {
                    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bfr;
                    du_load_matrix_sync(bfr, k_lds + (size_t)d16 * 16, KLD);
                    du_mma_sync(dqacc[d16], dlf, bfr, dqacc[d16]);
                }
            }

            // dkv: active warps own contiguous 16-wide d-tile slices.
            // The s / t nests are unrolled explicitly (trip counts are
            // runtime, so #pragma unroll is only a hint) so the dO/Q
            // B-fragment global loads pipeline ahead of the MMACs instead
            // of stalling a rolled loop every 2 MMACs.
            if (warp < ns) {
                const int t0   = (warp * ND) / ns;
                const int t1   = ((warp + 1) * ND) / ns;
                const int kidx = idx_lds[kt * 16 + r];
                // kacc groups of 4 keep the accumulator array small enough
                // that dq's 28-fragment accumulator stays in registers.
                for (int tg = t0; tg < t1; tg += 4) {
                    const int t_hi = (tg + 4 < t1) ? tg + 4 : t1;
                    DUFragment<accumulator, 16, 16, 16, float> kacc[4];
#    pragma unroll
                    for (int j = 0; j < 4; j++) du_fill_fragment(kacc[j], 0.0f);
#    pragma unroll
                    for (int s = 0; s < 4; s++) {
                        const int h0s = hb + s * 16;
                        if (h0s >= H) break;
                        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa;
                        du_load_matrix_sync(pa, pt_lds + (size_t)s * 256, 16);
                        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> da;
                        du_load_matrix_sync(da, pt_lds + (size_t)4 * 256 + (size_t)s * 256, 16);
#    pragma unroll
                        for (int j = 0; j < 4; j++) {
                            const int t = tg + j;
                            if (t >= t_hi) break;
                            DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bb, qb;
                            const hbf16* bp  = drow + (size_t)h0s * VALUE_DIM + (t * 16 + r);
                            const hbf16* qp2 = qrow + (size_t)h0s * D + (t * 16 + r);
                            // Fragment rows are heads h0s + g*4 + j; rows past
                            // H (partial slab, H % 16 != 0) are masked to 0 so
                            // they cannot fault or feed stale data into the
                            // kacc MMAs (their P/dl rows are 0 anyway).
                            const int fr = h0s + (g << 2);
                            bb.x[0] = (fr < H && t * 16 + r < VALUE_DIM) ? bp[(size_t)(g << 2) * VALUE_DIM] : hbf16(0);
                            bb.x[1] = (fr + 1 < H && t * 16 + r < VALUE_DIM) ? bp[((size_t)(g << 2) + 1) * VALUE_DIM]
                                                                             : hbf16(0);
                            bb.x[2] = (fr + 2 < H && t * 16 + r < VALUE_DIM) ? bp[((size_t)(g << 2) + 2) * VALUE_DIM]
                                                                             : hbf16(0);
                            bb.x[3] = (fr + 3 < H && t * 16 + r < VALUE_DIM) ? bp[((size_t)(g << 2) + 3) * VALUE_DIM]
                                                                             : hbf16(0);
                            qb.x[0] = (fr < H) ? qp2[(size_t)(g << 2) * D] : hbf16(0);
                            qb.x[1] = (fr + 1 < H) ? qp2[((size_t)(g << 2) + 1) * D] : hbf16(0);
                            qb.x[2] = (fr + 2 < H) ? qp2[((size_t)(g << 2) + 2) * D] : hbf16(0);
                            qb.x[3] = (fr + 3 < H) ? qp2[((size_t)(g << 2) + 3) * D] : hbf16(0);
                            du_mma_sync(kacc[j], pa, bb, kacc[j]);
                            du_mma_sync(kacc[j], da, qb, kacc[j]);
                        }
                    }
                    if (r < ck && kidx >= 0 && kidx < T_kv) {
#    pragma unroll
                        for (int j = 0; j < 4; j++) {
                            const int t = tg + j;
                            if (t >= t_hi) break;
                            float* pk = dkv_float + (size_t)kidx * D + t * 16;
                            saf_atomic_add_mmac_row(pk, g, kacc[j].x[0], kacc[j].x[1], kacc[j].x[2], kacc[j].x[3]);
                        }
                    }
                }
            }
            __syncthreads();  // k_lds / pt_lds reads done; commit next chunk
            if (nkt < ntiles) commit_keys(ck_for(nkt), pref, cnt);
            __syncthreads();  // next chunk's k_lds visible
        }

        // cuDNN uses the supplied O in the FP32 dO.O reduction for d_sink.

        if (lane_ok && g == 0) { atomicAdd(&dsink[h], -P_sink * avg); }

        // ---- dq epilogue: this slab's 64 heads, coalesced-ish 16B stores ----
        if (lane_ok) {
            bf16* dqbase = dq + ((size_t)t * H + h) * D;
#    pragma unroll
            for (int d16 = 0; d16 < ND; d16++) {
                const int dcol = d16 * 16 + g;
                const float q0 = dqacc[d16].x[0];
                const float q1 = dqacc[d16].x[1];
                const float q2 = dqacc[d16].x[2];
                const float q3 = dqacc[d16].x[3];
                // Same gfx938 MMAC/bit-twiddle aliasing guard as the fwd epilogue.
                dqbase[dcol + 0]  = (q0 != q0) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(q0);
                dqbase[dcol + 4]  = (q1 != q1) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(q1);
                dqbase[dcol + 8]  = (q2 != q2) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(q2);
                dqbase[dcol + 12] = (q3 != q3) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(q3);
            }
        }

        // ---- Last-block cast: dkv_float (fp32 atomics) -> bf16 dkv.  All
        // blocks have committed their dkv/dsink atomics at this point, so
        // the last block to arrive casts the whole buffer in place: the
        // operator stays exactly one kernel launch. ----
        __threadfence();
        __syncthreads();
        __shared__ int fused_last;
        if (threadIdx.x == 0) fused_last = atomicAdd(done_count, 1);
        __syncthreads();
        saf_multi_cast(done_count, dkv_float, dkv, T_kv, D, gridDim.x * gridDim.y, fused_last);
    }
}
// ---------------------------------------------------------------------------
// Workspace sparse_attn_bwd for large D. The stats and D-sliced gradient
// phases exchange P/dl through block-owned global workspaces.
//
// Pass 1 caches Q/dO fragments, recomputes S/DV, and writes transposed bf16
// P/dl workspaces while accumulating dsink. Pass 2 consumes those workspaces
// for dkv atomics and accumulates dq one D slice at a time.
// The last gradient block performs the f32->bf16 dkv cast.
// ---------------------------------------------------------------------------
// PHASE=0 keeps the legacy single-launch path.  PHASE=1/2 split the exact
// same algorithm at the workspace boundary so the two phases receive
// independent register allocation; the hot D=512 shapes use this path.
template<int ND, int NDS, int TOPK = 0, int PHASE = 0, int GDKV = 5, bool SPLIT_ROLES = false, int DQ_WAVES = 4,
         bool DQ_BUFFER = false, bool DQ_KBUFFER = false, int FIXED_H = 0, bool FUSE_DKV = false,
         int DISPATCH_GUARD = 0>
__global__ static void __launch_bounds__((PHASE == 4 ? DQ_WAVES * 64 : SAF_THREADS), 1) sparse_attn_bwd_fused2_kernel(
    const bf16* __restrict__ q,           // (T, H, D)
    const bf16* __restrict__ kv,          // (T_kv, D)
    const bf16* __restrict__ out,         // (T, H, D) fwd output (bf16)
    const bf16* __restrict__ dout,        // (T, H, D)
    const float* __restrict__ lse,        // (T, H) KV-only logsumexp (fwd)
    const float* __restrict__ attn_sink,  // (H,)
    const int* __restrict__ idxs,         // (T, topk)
    const int* __restrict__ topk_length,
    float* __restrict__ dkv_float,  // (T_kv, D) f32; fused2 uses tile-swizzled private layout
    float* __restrict__ dsink,      // (H,) f32 (zeroed by caller)
    uint32_t* __restrict__ pdl_ws,  // (T, topk, H) packed: low=P, high=dl bf16
    bf16* __restrict__ dq,          // (T, H, D)
    bf16* __restrict__ dkv,         // (T_kv, D) bf16 output
    int* __restrict__ done_count,   // zeroed by caller; last-block cast
    int T, int H, int D_rt, int T_kv, int topk_rt, float softmax_scale, const int* dispatch_guard = nullptr) {
    // Internal stats specialization: expose the actual head stride to the
    // compiler while retaining head/key guards and the arithmetic order.
    if constexpr (DISPATCH_GUARD == 5 || DISPATCH_GUARD == 6) {
        if ((dispatch_guard[0] == 0) != (DISPATCH_GUARD == 6)) return;
    } else if constexpr (DISPATCH_GUARD != 0 && DISPATCH_GUARD != 4) {
        const bool fuse = (dispatch_guard[0] & 1) == 0 && (int64_t)dispatch_guard[1] * 4 >= (int64_t)T * topk_rt;
        if (fuse != (DISPATCH_GUARD > 0)) return;
        if constexpr (DISPATCH_GUARD > 0) {
            // Both kernels execute the same query/key arithmetic; the GPU
            // selects one private FP32 accumulator layout for the whole call.
            const bool packed = (dispatch_guard[0] & 2) == 0 && T_kv % 8 == 0;
            if (packed != (DISPATCH_GUARD == 2)) return;
        }
    }
    if constexpr (FIXED_H > 0) H = FIXED_H;

    // dQ shares one K tile across all 128 heads when launched with eight waves.
    // Other phases retain their four-wave ownership and resource budgets.
    constexpr int THREADS = PHASE == 4 ? DQ_WAVES * 64 : SAF_THREADS;
    constexpr int D       = ND * 16;
    constexpr int SLICE   = NDS * 16;
    static_assert(PHASE != 4 || (NDS % 8 == 0 && (16 * (SLICE / 8)) % THREADS == 0),
                  "dQ matrix reads require complete staging and eight-column-tile groups");
    const int topk = (TOPK > 0) ? TOPK : topk_rt;
    (void)D_rt;
    // Consecutive H128 stats CTAs share the same sparse K rows. Pair the
    // two head slabs in launch order for prompt L2 reuse, retaining the
    // four-wave resource budget of each CTA.
    const bool pair_heads = PHASE == 1 && ND == 32 && H == 128;
    const int linear      = blockIdx.y * gridDim.x + blockIdx.x;
    // Keep this query's column slices adjacent in the launch grid.
    // Each output still has one owner and preserves its original key/MMAC order.
    // Long D512/D576 traversals reuse P/dS across these column slices.
    constexpr int DQ_ORDER_GROUP = 1;
    constexpr bool FUSED_TAIL    = ND == 36 && TOPK == 2048 && FIXED_H == 64 && (GDKV == 7 || GDKV == 11);
    constexpr int DQ_NSLICES     = FUSED_TAIL ? 4 : (ND + NDS - 1) / NDS;
    // Pairing the two column-slice roles of one query in launch order lets the
    // second role read the same P/dS from L2.  That reuse does not depend on
    // the row length: the short CSA rows (topk 128/160) pay for the same
    // 2x P/dS read, so keep the pairing for them too.
    const bool dq_order =
        PHASE == 4 && (ND == 32 || ND == 36) && (H == 64 || H == 128) && topk >= 128 && T % DQ_ORDER_GROUP == 0;
    const int dq_linear = (int)blockIdx.z * gridDim.x + (int)blockIdx.x;
    const int dq_group  = dq_linear / (DQ_ORDER_GROUP * DQ_NSLICES);
    const int dq_within = dq_linear % (DQ_ORDER_GROUP * DQ_NSLICES);
    const int t         = dq_order ? dq_group * DQ_ORDER_GROUP + dq_within % DQ_ORDER_GROUP
                                   : (pair_heads ? (linear >> 1) : (int)blockIdx.x);
    const int hb        = (pair_heads ? (linear & 1) : (int)blockIdx.y) * (PHASE == 4 ? DQ_WAVES * 16 : SAF_HBLK);
    if (hb >= H) return;
    const int lane = threadIdx.x & 63;
    // Wave ownership is scalar; avoid keeping duplicate address state in VGPRs.
    const int warp = ((ND == 32 || ND == 36) && (PHASE == 1 || PHASE == 4))
                         ? __builtin_amdgcn_readfirstlane(threadIdx.x >> 6)
                         : threadIdx.x >> 6;
    const int r0   = warp * 16;
    const int r    = lane & 15;
    const int g    = lane >> 4;
    // In the split phase-2 schedule z=0 owns dKV atomics and z=1..NSL
    // each owns one dQ slice.  The phase-1 and legacy phase-2 launches use
    // z=0 for the single combined role.
    // PHASE=4 is the dQ-only consumer used alongside cached dKV.
    const int role =
        dq_order ? dq_within / DQ_ORDER_GROUP + 1 : (SPLIT_ROLES ? (int)blockIdx.z + (PHASE == 4 ? 1 : 0) : 0);
    // Make the dQ-only role a compile-time fact: z + 1 alone does not
    // let the compiler eliminate the unreachable dKV MMAC/atomic body.
    const bool dkv_role = PHASE != 4 && (!SPLIT_ROLES || role == 0);
    const bool dq_role  = PHASE == 4 || !SPLIT_ROLES || role > 0;
    // Stats B fragments read four BF16 values per lane. An eight-byte
    // row skew separates their LDS banks while preserving 16-byte global
    // transfers (the direct-to-LDS row base may be eight-byte aligned).
    // Other phases keep their original row/slice layouts.
    constexpr int CH = 16;
    const int KLD1   = PHASE == 1 ? D + 4 : D + 24;
    const int KLD2   = SLICE + 8;

    extern __shared__ char smem_raw[];
    // One aliased LDS region serves pass-1 K rows, pass-2 P/dl tiles and the
    // pass-2 K slice at non-overlapping times. Top-k indices are read from L2.
    hbf16* k_lds   = (hbf16*)smem_raw;                               // pass-1 K rows / aliased P/dl
    hbf16* pt2_lds = (hbf16*)(smem_raw + 4096);                      // pass-2 dl^T tiles (byte 6144..10240)
    hbf16* k2_lds  = (hbf16*)(smem_raw + (PHASE == 4 ? 0 : 10240));  // pass-2 K slice rows (after dl tiles)
    // In split phase-2 blocks the dQ roles do not execute the dKV staging
    // path, so the dead pass-1/P-dl region can hold a second K tile.  This
    // gives dQ a true two-buffer key pipeline without increasing the dynamic
    // LDS allocation: [K0][K1][dl] is packed inside the existing 21 KB slab.
    constexpr bool DQ_PIPE_LAYOUT = SPLIT_ROLES && (PHASE == 2);
    hbf16* dq_k0                  = DQ_PIPE_LAYOUT ? (hbf16*)smem_raw : k2_lds;
    hbf16* dq_k1                  = DQ_PIPE_LAYOUT ? dq_k0 + (size_t)16 * KLD2 : k2_lds;
    hbf16* dq_dl                  = DQ_PIPE_LAYOUT ? dq_k1 + (size_t)16 * KLD2 : pt2_lds + 4 * 256;

    int n_valid_t = topk;
    if (topk_length != NULL) {
        int tl = topk_length[t];
        if (tl < n_valid_t) n_valid_t = tl > 0 ? tl : 0;
    }

    const hbf16* qrow = (const hbf16*)(q + (size_t)t * H * D);
    // GLM phase 1 accepts compact 512-wide O/dO or legacy padded D-wide
    // values via D_rt. Q/K width and every MMAC contraction remain D.
    const int value_stride        = PHASE == 1 && ND == 36 ? D_rt : D;
    const hbf16* orow             = (const hbf16*)(out + (size_t)t * H * value_stride);
    const hbf16* drow             = (const hbf16*)(dout + (size_t)t * H * value_stride);
    const int* irow               = idxs + (size_t)t * topk;
    constexpr bool CACHE_DQ_INDEX = PHASE == 4 && ND == 32 && NDS == 16 && DQ_WAVES == 8;
    int* dq_idx_lds               = (int*)(smem_raw + 8448);
    if constexpr (CACHE_DQ_INDEX) {
        if (topk <= SAF_DQ_INDEX_MAX_TOPK) {
            // The four-wave path is faster without this extra LDS traffic.
            // Initialize physical tail indices too: the final K prefetch
            // reads them before applying the logical-length mask.
            for (int i = threadIdx.x; i < topk; i += THREADS) dq_idx_lds[i] = irow[i];
            __syncthreads();
        }
    }
    // Short, constant-stride stats traversals benefit from removing the
    // dependent global index load ahead of each K gather. Keep the complete
    // row bounded: larger rows increase LDS pressure and regress training.
    constexpr bool CACHE_STATS_INDEX = PHASE == 1 && ND == 32 && TOPK > 0 && TOPK <= 1024;
    int* stats_idx_lds               = (int*)(smem_raw + 16 * (D + 24) * 2);
    if constexpr (CACHE_STATS_INDEX) {
        for (int i = threadIdx.x; i < topk; i += THREADS) stats_idx_lds[i] = irow[i];
        __syncthreads();
    }
    auto load_index = [&](int i) -> int {
        if constexpr (CACHE_STATS_INDEX) return stats_idx_lds[i];
        if constexpr (CACHE_DQ_INDEX) {
            if (topk <= SAF_DQ_INDEX_MAX_TOPK) return dq_idx_lds[i];
        }
        return irow[i];
    };
    // Every backward consumer uses this same logical tile bound (including
    // cached dKV). Wholly masked suffix tiles need no workspace writes,
    // gathers or barriers. ck_for retains the physical stride and initializes
    // the final partial tile, so no consumer reads uninitialized P/dS there.
    const int ntiles = (n_valid_t + CH - 1) / CH;

    // One 16-bit mask per logical key tile replaces repeated global index
    // ballots, including rows whose physical sparse list contains holes.
    constexpr bool CACHE_TILE_MASK =
        (ND == 32 || (ND == 36 && TOPK == 2048)) && (PHASE == 1 || PHASE == 4) && TOPK > 0 && TOPK <= 2048;
    __shared__ uint16_t tile_masks[CACHE_TILE_MASK ? ((TOPK + 63) / 64) * 4 : 1];
    __shared__ int keys_all_valid;
    {
        // This flag covers the logical prefix. Compact rows can still contain
        // only valid indices; the final tile retains its explicit length mask.
        if (threadIdx.x == 0)
            keys_all_valid =
                ((PHASE == 1 || PHASE == 4) && (ND == 32 || (ND == 36 && TOPK == 2048))) || (n_valid_t == topk);
        __syncthreads();
        for (int i = threadIdx.x; i < n_valid_t; i += THREADS) {
            const int idx = load_index(i);
            if constexpr (CACHE_TILE_MASK) {
                const unsigned long long bits = __ballot(idx >= 0 && idx < T_kv);
                if ((lane & 15) == 0) tile_masks[i / 16] = (uint16_t)(bits >> ((lane / 16) * 16));
                const bool bad = __any(idx < 0 || idx >= T_kv);
                if (lane == 0 && bad) atomicExch(&keys_all_valid, 0);
            } else if (idx < 0 || idx >= T_kv)
                atomicExch(&keys_all_valid, 0);
        }
        __syncthreads();
    }

    // The CTA validity reduction is immutable after the preceding barrier.
    // Keep its result in a wave-uniform register across all key tiles.
    const bool all_keys_valid = __builtin_amdgcn_readfirstlane(keys_all_valid) != 0;
    const int h               = hb + r0 + r;
    const bool lane_ok        = (h < H);
    const int hq              = hb + r0;

    const int nsub = (H - hb + 15) >> 4;
    const int ns   = (nsub < 4) ? nsub : 4;
    auto ck_for    = [&](int kkt) {
        int c = topk - kkt * CH;
        return (c > CH) ? CH : (c < 0 ? 0 : c);
    };

    constexpr int PREF1      = 2 * ((16 * (D / 8) + THREADS - 1) / THREADS);
    constexpr int PREF2      = 2 * ((16 * (SLICE / 8) + THREADS - 1) / THREADS);
    const uint64_t dq_kaddr  = (uint64_t)kv;
    const saf_intx4 dq_kdesc = {
        (int)(uint32_t)dq_kaddr,
        (int)((dq_kaddr >> 32) | ((PHASE == 4 && (ND == 32 || ND == 36) && (NDS == 16 || (ND == 36 && NDS == 8)) &&
                                   DQ_WAVES == 4 && DQ_KBUFFER)
                                      ? 0u
                                      : 0x40000000u)),
        (int)0x80000000u, 0x00020000};
    constexpr bool DQ_INDEX_AHEAD =
        PHASE == 4 && (ND == 32 || ND == 36) && (NDS == 16 || (ND == 36 && NDS == 8)) && DQ_WAVES == 4 && DQ_KBUFFER;
    // NDS8 gathers one key per lane; NDS16 gathers two.
    int dq_future_index[(16 * (SLICE / 8) + THREADS - 1) / THREADS];
    bool dq_future_ready          = false;
    const uint64_t dq_index_addr  = (uint64_t)irow;
    const saf_intx4 dq_index_desc = {(int)(uint32_t)dq_index_addr, (int)(dq_index_addr >> 32), TOPK * 4, 0x00020000};
    auto prefetch_keys            = [&](int kt, int ck, int d8n, int stride, int col0, uint64_t* regs) -> int {
        if constexpr (PHASE == 1) {
            // All waves inspect the same 16 keys. An empty stats tile has
            // no LDS consumer: its P/dS outputs are explicitly zeroed below.
            // Return no payload, so commit_keys also skips its LDS writes;
            // retain both loop barriers and the next nonempty tile's gather.
            if (!all_keys_valid) {
                const int ti  = kt * CH + r;
                const int key = ti < n_valid_t ? load_index(ti) : -1;
                if (!__builtin_amdgcn_readfirstlane(__any(key >= 0 && key < T_kv))) return 0;
            }
        }

        if constexpr (PHASE == 4) {
            // Fixed slots permit register scalarization and unrolling.  Every
            // slot is initialized, including masked rows in the last tile.
            constexpr int count = (16 * (SLICE / 8) + THREADS - 1) / THREADS;
            // Resolve the complete index batch before loading K payloads.
            // Four-wave dQ needs two keys per lane; interleaving their index
            // and payload loads serializes the second gather on gfx93x.
            int gather_idx[count];
#    pragma unroll
            for (int j = 0; j < count; j++) {
                const int n = (threadIdx.x + j * THREADS) / d8n;
                if constexpr (DQ_INDEX_AHEAD) {
                    gather_idx[j] = dq_future_ready ? dq_future_index[j] : (n < ck ? load_index(kt * CH + n) : 0);
                } else
                    gather_idx[j] = n < ck ? load_index(kt * CH + n) : 0;
            }
            if constexpr (count == 2) asm volatile("" ::"v"(gather_idx[0]), "v"(gather_idx[1]) : "memory");
#    pragma unroll
            for (int j = 0; j < count; j++) {
                const int u = threadIdx.x + j * THREADS;
                const int n = u / d8n, d8 = u % d8n;
                const int key = kt * CH + n;
                const int idx = gather_idx[j];
                const bool valid =
                    n < ck && (key < n_valid_t && idx >= 0 && idx < T_kv && (ND % NDS == 0 || col0 + d8 * 8 < D));
                if constexpr (DQ_KBUFFER) {
                    // Invalid lanes use the descriptor's out-of-range zero fill,
                    // avoiding divergent masked-load PHIs and their early waits.
                    const unsigned off = valid ? ((unsigned)idx * D + col0 + d8 * 8) * 2u : 0x80000000u;
                    const saf_intx4 v  = saf_raw_buffer_load_i32x4(dq_kdesc, off, 0, 0);
                    regs[2 * j]        = ((const uint64_t*)&v)[0];
                    regs[2 * j + 1]    = ((const uint64_t*)&v)[1];
                } else {
                    const uint64_t* src = (const uint64_t*)(kv + (size_t)(valid ? idx : 0) * D + col0 + d8 * 8);
                    regs[2 * j]         = valid ? src[0] : 0;
                    regs[2 * j + 1]     = valid ? src[1] : 0;
                }
            }
            return count;
        }
        int cnt = 0;
        if (all_keys_valid) {
            for (int u = threadIdx.x; u < ck * d8n; u += THREADS) {
                int n               = u / d8n;
                int d8              = u - n * d8n;
                const int idx       = load_index(kt * CH + n);
                const uint64_t* src = (const uint64_t*)(kv + (size_t)idx * D + col0 + d8 * 8);
                regs[2 * cnt]       = src[0];
                regs[2 * cnt + 1]   = src[1];
                cnt++;
            }
            return cnt;
        }
        for (int u = threadIdx.x; u < ck * d8n; u += THREADS) {
            int n               = u / d8n;
            int d8              = u - n * d8n;
            const int key       = kt * CH + n;
            int idx             = load_index(key);
            const bool valid    = key < n_valid_t && idx >= 0 && idx < T_kv;
            const int safe_idx  = valid ? idx : 0;
            const uint64_t* src = (const uint64_t*)(kv + (size_t)safe_idx * D + col0 + d8 * 8);
            const uint64_t x0 = src[0], x1 = src[1];
            regs[2 * cnt]     = valid ? x0 : 0;
            regs[2 * cnt + 1] = valid ? x1 : 0;
            cnt++;
        }
        return cnt;
    };
    auto commit_keys = [&](int ck, int d8n, int stride, int nrows, hbf16* base, const uint64_t* regs, int cnt) {
#    pragma unroll
        for (int j = 0; j < (PHASE == 4 && (ND == 32 || ND == 36) ? (16 * (SLICE / 8) + THREADS - 1) / THREADS : cnt);
             j++) {
            int u  = threadIdx.x + j * THREADS;
            int n  = u / d8n;
            int d8 = u - n * d8n;
            // Matrix-read dQ uses contiguous [D/32][16 keys][32 columns] tiles.
            // Preserve aligned 16-byte writes; other phases retain row-major K.
            uint64_t* dst = (uint64_t*)(base + (PHASE == 4 ? (size_t)(d8 / 4) * 512 + n * 32 + (d8 % 4) * 8
                                                           : (size_t)n * stride + d8 * 8));
            dst[0]        = regs[2 * j];
            dst[1]        = regs[2 * j + 1];
        }
        // PHASE4 prefetch initializes all 16 rows, including masked/tail slots.
        if constexpr (PHASE == 4) return;
        if (ck < nrows) {
            for (int u = threadIdx.x; u < nrows * stride; u += THREADS) {
                if (u / stride >= ck) base[u] = hbf16(0);
            }
        }
    };
    if constexpr (PHASE == 0 || PHASE == 1) {
        // ---- Prologue: m/inv_l from lse/sink and avg = dO . O ----
        // The public value width is 512, with compact or padded O/dO storage.
        // Keep the RoPE columns in QK, but omit their zero dO products.
        constexpr int VALUE_ND   = PHASE == 1 && ND == 36 ? 32 : ND;
        constexpr bool REUSE_DOT = PHASE == 1 && (ND == 32 || ND == 36);
        bool avg_enabled         = false;
        float m = 0.0f, inv_l = 0.0f, P_sink = 0.0f, avg = 0.0f;
        if (lane_ok) {
            const float lse_h  = lse[(size_t)t * H + h];
            const float sink_h = attn_sink[h];
            saf_backward_normalization(lse_h, sink_h, m, inv_l, P_sink);
            {
                avg_enabled = true;
                if constexpr (!REUSE_DOT) {
                    const hbf16* orp = orow + (size_t)h * value_stride;
                    const hbf16* dr  = drow + (size_t)h * value_stride;
                    float p          = 0.0f;
#    pragma unroll
                    for (int kk = 0; kk < VALUE_ND; kk++) {
                        const int c  = kk * 16 + (g << 2);
                        p           += (float)orp[c + 0] * (float)dr[c + 0];
                        p           += (float)orp[c + 1] * (float)dr[c + 1];
                        p           += (float)orp[c + 2] * (float)dr[c + 2];
                        p           += (float)orp[c + 3] * (float)dr[c + 3];
                    }
                    avg = p;
                }
            }
        }
        if constexpr (!REUSE_DOT) {
            avg += __shfl_xor(avg, 16);
            avg += __shfl_xor(avg, 32);
        }

        // ---- Pass 1: S/DV -> P/dl -> dkv atomics + dl_ws ----
        constexpr int QD_CACHE = FUSE_DKV ? ND : VALUE_ND;  // dO cached in registers across chunks
        constexpr int QCACHE   = ND;                        // Q cached in registers across chunks
        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> qf[QCACHE], df[QD_CACHE];
        auto load_d = [&](int kk, DUFragment<matrix_a, 16, 16, 16, hbf16, row_major>& dfr) {
            const int row_h = lane_ok ? h : (H - 1);
            const hbf16* dp = (const hbf16*)(drow + (size_t)row_h * value_stride + kk * 16);
            if constexpr ((ND == 32 && FIXED_H == 64 && PHASE == 1 && TOPK == 1152 &&
                           (DISPATCH_GUARD == 3 || DISPATCH_GUARD == 4) && !FUSE_DKV) ||
                          (ND == 32 && FIXED_H == 128 && PHASE == 1 && TOPK == 1152 && DISPATCH_GUARD == 0 &&
                           !FUSE_DKV) ||
                          (ND == 36 && (FIXED_H == 32 || FIXED_H == 64) && TOPK == 2048 && PHASE == 1)) {
                const uint64_t address = (uint64_t)drow;
                const saf_u4e desc     = {(unsigned)address,
                                          (unsigned)(address >> 32) | ((value_stride * 2) << 16) | 0x40000000u,
                                          (unsigned)FIXED_H, 0x00020000u};
                auto packed = __builtin_amdgcn_buffer_load_dwordx2(desc, row_h, (kk * 16 + g * 4) * 2, false, false);
                *(uint64_t*)&dfr.x[0] = *(const uint64_t*)&packed;
            } else
                *(uint64_t*)&dfr.x[0] = *(const uint64_t*)(dp + (g << 2));
            if (!lane_ok) { dfr.x[0] = dfr.x[1] = dfr.x[2] = dfr.x[3] = hbf16(0); }
        };
        auto load_q = [&](int kk, DUFragment<matrix_a, 16, 16, 16, hbf16, row_major>& qfr) {
            const int row_h = lane_ok ? h : (H - 1);
            const hbf16* qp = (const hbf16*)(qrow + (size_t)row_h * D + kk * 16);
            if constexpr ((ND == 32 && FIXED_H == 64 && PHASE == 1 && TOPK == 1152 &&
                           (DISPATCH_GUARD == 3 || DISPATCH_GUARD == 4) && !FUSE_DKV) ||
                          (ND == 32 && FIXED_H == 128 && PHASE == 1 && TOPK == 1152 && DISPATCH_GUARD == 0 &&
                           !FUSE_DKV) ||
                          (ND == 36 && (FIXED_H == 32 || FIXED_H == 64) && TOPK == 2048 && PHASE == 1)) {
                const uint64_t address = (uint64_t)qrow;
                const saf_u4e desc     = {(unsigned)address, (unsigned)(address >> 32) | ((D * 2) << 16) | 0x40000000u,
                                          (unsigned)FIXED_H, 0x00020000u};
                auto packed = __builtin_amdgcn_buffer_load_dwordx2(desc, row_h, (kk * 16 + g * 4) * 2, false, false);
                *(uint64_t*)&qfr.x[0] = *(const uint64_t*)&packed;
            } else
                *(uint64_t*)&qfr.x[0] = *(const uint64_t*)(qp + (g << 2));
            if (!lane_ok) { qfr.x[0] = qfr.x[1] = qfr.x[2] = qfr.x[3] = hbf16(0); }
        };

        if constexpr (!REUSE_DOT) {
#    pragma unroll
            for (int kk = 0; kk < ND; kk++) load_q(kk, qf[kk]);
        }
        // dO has the same re-read pattern as Q (one A-fragment per 16-d tile,
        // re-read every key chunk); cache it once per block so pass-1 touches
        // dO from DRAM a single time instead of once per chunk.
        if constexpr ((REUSE_DOT && ND == 32 &&
                       ((TOPK == 128 || TOPK == 160) ||
                        (TOPK == 1152 && (DISPATCH_GUARD == 3 || DISPATCH_GUARD == 4) && !FUSE_DKV))) ||
                      (ND == 36 && (FIXED_H == 32 || FIXED_H == 64) && TOPK == 2048 && PHASE == 1)) {
            // Batch independent prologue loads; retain original FP32 product order.
            constexpr int DOT_BATCH = TOPK == 1152 || ND == 36 ? 4 : 8;
// Short CSA batching remains unchanged.
// Issue a small group of independent dO/O loads before consuming
// them. Preserve ascending kk/element FP32 dot accumulation.
#    pragma unroll
            for (int kk = 0; kk < VALUE_ND; kk += DOT_BATCH) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> ofr[DOT_BATCH];
#    pragma unroll
                for (int j = 0; j < DOT_BATCH; j++) {
                    load_d(kk + j, df[kk + j]);
                    if (avg_enabled) {
                        if constexpr ((ND == 32 && FIXED_H == 64 && PHASE == 1 && TOPK == 1152 &&
                                       (DISPATCH_GUARD == 3 || DISPATCH_GUARD == 4) && !FUSE_DKV) ||
                                      (ND == 32 && FIXED_H == 128 && PHASE == 1 && TOPK == 1152 &&
                                       DISPATCH_GUARD == 0 && !FUSE_DKV) ||
                                      (ND == 36 && (FIXED_H == 32 || FIXED_H == 64) && TOPK == 2048 && PHASE == 1)) {
                            const uint64_t address = (uint64_t)orow;
                            const saf_u4e desc = {(unsigned)address,
                                                  (unsigned)(address >> 32) | ((value_stride * 2) << 16) | 0x40000000u,
                                                  (unsigned)FIXED_H, 0x00020000u};
                            auto packed = __builtin_amdgcn_buffer_load_dwordx2(desc, h, ((kk + j) * 16 + g * 4) * 2,
                                                                               false, false);
                            *(uint64_t*)&ofr[j].x[0] = *(const uint64_t*)&packed;
                        } else
                            *(uint64_t*)&ofr[j].x[0] =
                                *(const uint64_t*)(orow + (size_t)h * value_stride + (kk + j) * 16 + g * 4);
                    }
                }
                if (avg_enabled) {
#    pragma unroll
                    for (int j = 0; j < DOT_BATCH; j++) {
                        avg += (float)ofr[j].x[0] * (float)df[kk + j].x[0];
                        avg += (float)ofr[j].x[1] * (float)df[kk + j].x[1];
                        avg += (float)ofr[j].x[2] * (float)df[kk + j].x[2];
                        avg += (float)ofr[j].x[3] * (float)df[kk + j].x[3];
                    }
                }
                asm volatile("" ::"v"(avg) : "memory");
            }
        } else {
#    pragma unroll
            for (int kk = 0; kk < VALUE_ND; kk++) {
                load_d(kk, df[kk]);
                if constexpr (REUSE_DOT) {
                    if (avg_enabled) {
                        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> ofr;
                        *(uint64_t*)&ofr.x[0] = *(const uint64_t*)(orow + (size_t)h * value_stride + kk * 16 + g * 4);
                        // Identical ascending kk/element FP32 accumulation as dO.O.
                        avg += (float)ofr.x[0] * (float)df[kk].x[0];
                        avg += (float)ofr.x[1] * (float)df[kk].x[1];
                        avg += (float)ofr.x[2] * (float)df[kk].x[2];
                        avg += (float)ofr.x[3] * (float)df[kk].x[3];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&df[kk].x[0]), "v"(avg) : "memory");
                }
            }
        }
        if constexpr (REUSE_DOT) {
            avg += __shfl_xor(avg, 16);
            avg += __shfl_xor(avg, 32);
            // Keep the Q loads after the dot to avoid extending their lifetimes.
            asm volatile("" ::"v"(avg) : "memory");
        }
        if constexpr (REUSE_DOT) {
#    pragma unroll
            for (int kk = 0; kk < ND; kk++) load_q(kk, qf[kk]);
        }

        DUFragment<accumulator, 16, 16, 16, float> sacc, dacc;
        DUFragment<accumulator, 16, 16, 16, float> tail_dq[4];
        if constexpr (FUSED_TAIL) {
#    pragma unroll
            for (int j = 0; j < 4; j++) du_fill_fragment(tail_dq[j], 0.f);
        }

        // D=512 stats keep Q and dO resident. Direct global-to-LDS transfers
        // avoid the additional next-tile VGPR payload (224 -> 198 VGPR on
        // gfx936) and its explicit LDS stores. Use the same wave64 path on
        // gfx936/gfx938. D=576 uses a second, eight-lane copy for the RoPE
        // tail; the other phases retain their register prefetch.
        constexpr bool DIRECT_STATS = PHASE == 1 && (ND == 32 || ND == 36);
        typedef int StatsDescriptor __attribute__((ext_vector_type(4)));
        const uint64_t kv_addr        = (uint64_t)kv;
        const StatsDescriptor kv_desc = {(int)(uint32_t)kv_addr, (int)((kv_addr >> 32) | 0x40000000u), (int)0x80000000u,
                                         0x00020000};
        constexpr bool STATS_INDEX_AHEAD =
            (PHASE == 1 && ND == 32 && TOPK == 1152 && (DISPATCH_GUARD == 3 || DISPATCH_GUARD == 4) && !FUSE_DKV) ||
            (ND == 36 && (FIXED_H == 32 || FIXED_H == 64) && TOPK == 2048 && PHASE == 1);
        const uint64_t stats_index_address     = (uint64_t)irow;
        const saf_intx4 stats_index_descriptor = {(int)(uint32_t)stats_index_address,
                                                  (int)((stats_index_address >> 32) | 0x40000000u), (int)0x80000000u,
                                                  0x00020000};
        auto gather_stats                      = [&](int kt, const int* future_keys = nullptr) {
            if constexpr (DIRECT_STATS) {
                // Entirely masked tiles have no matrix consumer. Preserve the
                // loop barriers and explicit zero P/dS stores below, but omit K.
                if (!all_keys_valid) {
                    if constexpr (CACHE_TILE_MASK) {
                        if (tile_masks[kt] == 0) return;
                    } else {
                        const int ki  = kt * CH + r;
                        const int key = ki < n_valid_t ? load_index(ki) : -1;
                        if (!__builtin_amdgcn_readfirstlane(__any(key >= 0 && key < T_kv))) return;
                    }
                }
                // One wave owns a key row; its 64 lanes transfer 16 bytes each.
                // Zero invalid rows so mixed-validity tiles cannot propagate NaN
                // from a masked key through the matrix products.
                // Load every index before issuing global-to-LDS transfers. A
                // later index load would otherwise wait for an earlier K transfer.
                int keys[4];
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int ki = kt * CH + warp + j * 4;
                    keys[j] =
                        STATS_INDEX_AHEAD && future_keys ? future_keys[j] : (ki < n_valid_t ? load_index(ki) : -1);
                }
                asm volatile("" ::"v"(keys[0]), "v"(keys[1]), "v"(keys[2]), "v"(keys[3]) : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n      = warp + j * 4;
                    const int ki     = kt * CH + n;
                    const int key    = keys[j];
                    const bool valid = ki < n_valid_t && (all_keys_valid || (key >= 0 && key < T_kv));
#    pragma unroll
                    for (int dc = 0; dc < D; dc += 512) {
                        if (dc + lane * 8 < D) {
                            if (valid) {
                                __builtin_amdgcn_raw_buffer_load_lds(
                                    kv_desc, (__attribute__((address_space(3))) int*)(k_lds + n * KLD1 + dc), 16,
                                    key * (D * 2) + dc * 2 + lane * 16, 0, 0, 0);
                            } else {
                                uint64_t* dst = (uint64_t*)(k_lds + n * KLD1 + dc + lane * 8);
                                dst[0] = dst[1] = 0;
                            }
                        }
                    }
                }
            }
        };
        constexpr bool STATS_LDS_PREFIX =
            PHASE == 1 && ND == 32 && TOPK == 1152 && (DISPATCH_GUARD == 3 || DISPATCH_GUARD == 4) && !FUSE_DKV;
        using StatsRead4 = unsigned __attribute__((ext_vector_type(4)));
        StatsRead4 stats_first_k[2];
        auto read_stats_prefix = [&]() {
#    pragma unroll
            for (int z = 0; z < 2; z++) {
                const unsigned address = (unsigned)(size_t)(k_lds + (lane & 15) * KLD1 + (lane >> 4) * 4 + z * 32);
                asm volatile("ds_read2_b64 %0, %1 offset0:0 offset1:4"
                             : "=v"(stats_first_k[z])
                             : "v"(address)
                             : "memory");
            }
        };
        if constexpr (DIRECT_STATS) {
            gather_stats(0);
            // All direct transfers must complete before another wave reads K.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        } else {
            uint64_t pref[PREF1];
            int cnt = prefetch_keys(0, ck_for(0), D / 8, KLD1, 0, pref);
            commit_keys(ck_for(0), D / 8, KLD1, CH, k_lds, pref, cnt);
        }
        __syncthreads();
        if constexpr (STATS_LDS_PREFIX) {
            if (ntiles > 0) read_stats_prefix();
        }
#    pragma unroll 1
        for (int kt = 0; kt < ntiles; kt++) {
            uint64_t pref[PREF1];
            int cnt       = 0;
            const int nkt = kt + 1;
            if constexpr (!DIRECT_STATS) {
                if (nkt < ntiles) cnt = prefetch_keys(nkt, ck_for(nkt), D / 8, KLD1, 0, pref);
            }
            const int ck = ck_for(kt);

            // The existing row-wide validity flag keeps fully populated rows
            // on the original path.  All four waves inspect the same 16 keys,
            // so an empty tile can bypass MMAC while retaining prefetch/barriers.
            const int ti       = kt * CH + r;
            const int tile_key = all_keys_valid ? 0 : (ti < n_valid_t ? load_index(ti) : -1);
            // The low 16 ballot bits describe the same key tile for every wave.
            // Reuse them for the four score columns instead of reloading each
            // index during softmax. ck also masks the final partial tile.
            // The CTA prologue checked the complete logical index prefix; holes still use tile_masks.
            constexpr bool STATS_VALID_FAST =
                (PHASE == 1 && ND == 32 && TOPK == 1152 && (DISPATCH_GUARD == 3 || DISPATCH_GUARD == 4) && !FUSE_DKV) ||
                (PHASE == 1 && ND == 36 && TOPK == 2048 && FIXED_H == 64 && FUSE_DKV);
            const unsigned tile_mask = STATS_VALID_FAST && all_keys_valid ? ((1u << min(ck, n_valid_t - kt * CH)) - 1u)
                                       : CACHE_TILE_MASK                  ? tile_masks[kt]
                                       : all_keys_valid
                                           ? ((1u << min(ck, n_valid_t - kt * CH)) - 1u)
                                           : ((unsigned)__ballot(tile_key >= 0 && tile_key < T_kv) & 0xffffu);
            const bool active        = tile_mask != 0;
            int next_stats_keys[4];
            if constexpr (STATS_INDEX_AHEAD) {
// LLVM-tracked vector loads overlap the current QK/dOV MMACs.
// The following barrier drains them before gather_stats uses them.
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int key_pos  = nkt * CH + warp + j * 4;
                    const unsigned off = key_pos < n_valid_t ? key_pos * 4u : 0x80000000u;
                    if constexpr (ND == 36)
                        next_stats_keys[j] = key_pos < n_valid_t ? load_index(key_pos) : -1;
                    else
                        next_stats_keys[j] = saf_raw_buffer_load_i32(stats_index_descriptor, off, 0, 0);
                }
            }
            if (active) {
                du_fill_fragment(sacc, 0.0f);
                du_fill_fragment(dacc, 0.0f);
                if (hq < H) {
                    if constexpr (PHASE == 1 && (ND == 32 || ND == 36)) {
                        // Pair adjacent 16-channel BF16 fragments in the original MMAC order.
                        // After waiting for the current pair, prefetch the next pair
                        // before the current QK/dOV work. Only consume prefetched
                        // registers after the next LGKM wait; no extra LDS is needed.
                        using Read4 = unsigned __attribute__((ext_vector_type(4)));
                        Read4 packed[ND / 2];
                        if constexpr (STATS_LDS_PREFIX) {
                            // Tie the asynchronous outputs to the wait so no register
                            // copy or reuse can precede completion.
                            asm volatile("s_waitcnt lgkmcnt(0)"
                                         : "+v"(stats_first_k[0]), "+v"(stats_first_k[1])::"memory");
                            packed[0] = stats_first_k[0];
                            packed[1] = stats_first_k[1];
                        } else {
#    pragma unroll
                            for (int z = 0; z < 2; ++z) {
                                unsigned address =
                                    (unsigned)(size_t)(k_lds + (lane & 15) * KLD1 + (lane >> 4) * 4 + z * 32);
                                asm volatile("ds_read2_b64 %0, %1 offset0:0 offset1:4"
                                             : "=v"(packed[z])
                                             : "v"(address)
                                             : "memory");
                            }
                        }
#    pragma unroll
                        for (int k = 0; k < ND / 2; k += 2) {
                            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#    pragma unroll
                            for (int z = 0; z < 2; ++z) {
                                if (k + 2 + z < ND / 2) {
                                    unsigned address = (unsigned)(size_t)(k_lds + (lane & 15) * KLD1 + (lane >> 4) * 4 +
                                                                          (k + 2 + z) * 32);
                                    asm volatile("ds_read2_b64 %0, %1 offset0:0 offset1:4"
                                                 : "=v"(packed[k + 2 + z])
                                                 : "v"(address)
                                                 : "memory");
                                }
                            }
#    pragma unroll
                            for (int z = 0; z < 2; ++z) {
                                if (k + z < ND / 2) {
                                    DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr0, bfr1;
                                    ((unsigned*)&bfr0.x[0])[0] = packed[k + z][0];
                                    ((unsigned*)&bfr0.x[0])[1] = packed[k + z][1];
                                    ((unsigned*)&bfr1.x[0])[0] = packed[k + z][2];
                                    ((unsigned*)&bfr1.x[0])[1] = packed[k + z][3];
                                    du_mma_sync(sacc, qf[2 * (k + z)], bfr0, sacc);
                                    if (2 * (k + z) < VALUE_ND) du_mma_sync(dacc, df[2 * (k + z)], bfr0, dacc);
                                    du_mma_sync(sacc, qf[2 * (k + z) + 1], bfr1, sacc);
                                    if (2 * (k + z) + 1 < VALUE_ND) du_mma_sync(dacc, df[2 * (k + z) + 1], bfr1, dacc);
                                }
                            }
                        }
                    } else {
#    pragma unroll
                        for (int kk = 0; kk < ND; kk += 2) {
                            DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr0, bfr1;
                            du_load_matrix_sync(bfr0, k_lds + (size_t)kk * 16, KLD1);
                            du_load_matrix_sync(bfr1, k_lds + (size_t)(kk + 1) * 16, KLD1);
                            du_mma_sync(sacc, qf[kk], bfr0, sacc);
                            if (kk < VALUE_ND) du_mma_sync(dacc, df[kk], bfr0, dacc);
                            du_mma_sync(sacc, qf[kk + 1], bfr1, sacc);
                            if (kk + 1 < VALUE_ND) du_mma_sync(dacc, df[kk + 1], bfr1, dacc);
                        }
                    }
                }
            }
            DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> tail_k[4];
            if constexpr (FUSED_TAIL) {
                if (active) {
                    if constexpr (GDKV == 11) {
                        using TailRead4 = unsigned __attribute__((ext_vector_type(4)));
                        TailRead4 tr[4];
#    pragma unroll
                        for (int j = 0; j < 4; j++) {
                            const unsigned addr = (unsigned)(size_t)(k_lds + (g * 4) * KLD1 + 512 + j * 16 + r);
                            asm volatile("ds_read_u16 %0, %4 offset:0\n\t"
                                     "ds_read_u16 %1, %4 offset:%5\n\t"
                                     "ds_read_u16 %2, %4 offset:%6\n\t"
                                     "ds_read_u16 %3, %4 offset:%7"
                          : "=&v"(tr[j][0]),"=&v"(tr[j][1]),"=&v"(tr[j][2]),"=&v"(tr[j][3])
                          : "v"(addr),"n"(KLD1*2),"n"(KLD1*4),"n"(KLD1*6) : "memory");
                        }
                        // Preserve every physical return until all sixteen LDS loads complete.
                        asm volatile("s_waitcnt lgkmcnt(0)"
                                     : "+v"(tr[0]), "+v"(tr[1]), "+v"(tr[2]), "+v"(tr[3])::"memory");
#    pragma unroll
                        for (int j = 0; j < 4; j++) {
                            ((uint32_t*)&tail_k[j].x[0])[0] = tr[j][0] | (tr[j][1] << 16);
                            ((uint32_t*)&tail_k[j].x[0])[1] = tr[j][2] | (tr[j][3] << 16);
                        }
                    } else {
#    pragma unroll
                        for (int j = 0; j < 4; j++) {
                            du_load_matrix_sync(tail_k[j], k_lds + 512 + j * 16, KLD1);
                            // Complete every old-tile LDS read before the global-to-LDS
                            // overwrite barrier. These operands remain live through dQ.
                            asm volatile("" ::"v"(*(const uint64_t*)&tail_k[j].x[0]) : "memory");
                        }
                    }
                }
            }
            if constexpr (DIRECT_STATS) {
                // All waves have finished reading K. Reuse the same LDS tile
                // while softmax and packed P/dS stores consume only registers.
                __syncthreads();
                if (nkt < ntiles) {
                    if constexpr (STATS_INDEX_AHEAD)
                        gather_stats(nkt, next_stats_keys);
                    else
                        gather_stats(nkt);
                }
            }
            if (active) {
                {
                    const bool ok0 = (tile_mask >> (g + 0)) & 1u;
                    const bool ok1 = (tile_mask >> (g + 4)) & 1u;
                    const bool ok2 = (tile_mask >> (g + 8)) & 1u;
                    const bool ok3 = (tile_mask >> (g + 12)) & 1u;
                    const float s0 = lane_ok ? sacc.x[0] * softmax_scale : -INFINITY;
                    const float s1 = lane_ok ? sacc.x[1] * softmax_scale : -INFINITY;
                    const float s2 = lane_ok ? sacc.x[2] * softmax_scale : -INFINITY;
                    const float s3 = lane_ok ? sacc.x[3] * softmax_scale : -INFINITY;
                    const float p0 = (ok0 && lane_ok) ? hip_dsa_numeric::softmax_exp(s0 - m) * inv_l : 0.0f;
                    const float p1 = (ok1 && lane_ok) ? hip_dsa_numeric::softmax_exp(s1 - m) * inv_l : 0.0f;
                    const float p2 = (ok2 && lane_ok) ? hip_dsa_numeric::softmax_exp(s2 - m) * inv_l : 0.0f;
                    const float p3 = (ok3 && lane_ok) ? hip_dsa_numeric::softmax_exp(s3 - m) * inv_l : 0.0f;
                    const float d0 = dacc.x[0], d1 = dacc.x[1];
                    const float d2 = dacc.x[2], d3 = dacc.x[3];
                    // CuTe SM90 rounds dS before scaling, then rounds the scaled MMA
                    // operand again. Keep FP32 P here; its BF16 copy is only for PV.
                    const float dl0 =
                        (ok0 && lane_ok) ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p0 * (d0 - avg)) << 16) *
                                               softmax_scale
                                         : 0.0f;
                    const float dl1 =
                        (ok1 && lane_ok) ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p1 * (d1 - avg)) << 16) *
                                               softmax_scale
                                         : 0.0f;
                    const float dl2 =
                        (ok2 && lane_ok) ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p2 * (d2 - avg)) << 16) *
                                               softmax_scale
                                         : 0.0f;
                    const float dl3 =
                        (ok3 && lane_ok) ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p3 * (d3 - avg)) << 16) *
                                               softmax_scale
                                         : 0.0f;

                    if constexpr (FUSED_TAIL) {
                        // Transpose the four lane groups' BF16 dS registers: the
                        // accumulator uses keys g+4*e, the A operand uses 4*g+e.
                        uint32_t v[4] = {hip_dsa_numeric::bf16_rne(dl0), hip_dsa_numeric::bf16_rne(dl1),
                                         hip_dsa_numeric::bf16_rne(dl2), hip_dsa_numeric::bf16_rne(dl3)},
                                 x[4], y[4];
#    pragma unroll
                        for (int e = 0; e < 4; e++) {
                            uint32_t peer = __shfl_xor(v[e ^ 1], 16);
                            x[e]          = ((g ^ e) & 1) ? peer : v[e];
                        }
#    pragma unroll
                        for (int e = 0; e < 4; e++) {
                            uint32_t peer = __shfl_xor(x[e ^ 2], 32);
                            y[e]          = ((g ^ e) & 2) ? peer : x[e];
                        }
                        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> ds;
#    pragma unroll
                        for (int e = 0; e < 4; e++) *(uint16_t*)&ds.x[e] = (uint16_t)y[e];
#    pragma unroll
                        for (int j = 0; j < 4; j++) du_mma_sync(tail_dq[j], ds, tail_k[j], tail_dq[j]);
                    }

                    // dl and P are stored transposed (T, topk, H): for a fixed key
                    // the 16 lane-heads write consecutive h (coalesced), and the
                    // dkv phase below re-reads them as 8B per-lane vectors.  Keys
                    // beyond ck / n_valid_t are zeroed (their K rows are zeroed
                    // too, so they cannot leak into dkv/dq).
                    // Wait for next K before issuing this tile's independent stores.
                    // The LDS barrier below then permits those stores to overlap the
                    // next QK/dOV contraction. No next-tile reader consumes P/dS here.
                    if constexpr (DIRECT_STATS) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    if constexpr (STATS_LDS_PREFIX) {
                        // Same second CTA barrier, now before independent P/dS stores.
                        // All K copies/invalid-row zeros are visible before the reads.
                        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
                        if (nkt < ntiles) read_stats_prefix();
                    }
                    uint32_t* pdl = pdl_ws + ((size_t)t * topk + kt * CH + g) * H + h;
                    if (lane_ok) {
                        // Streaming stores: the P/dS workspace is written once here and
                        // streamed by the dKV/dQ passes, so allocating it in L2 can
                        // evicts the KV rows this same kernel re-reads.
                        if (g + 0 < ck) {
                            const uint32_t packed = ok0 ? hip_dsa_numeric::bf16_pair(p0, dl0) : 0u;
                            saf_nt_store_u32(pdl + 0, packed);
                        }
                        if (g + 4 < ck) {
                            const uint32_t packed = ok1 ? hip_dsa_numeric::bf16_pair(p1, dl1) : 0u;
                            saf_nt_store_u32(pdl + 4 * (size_t)H, packed);
                        }
                        if (g + 8 < ck) {
                            const uint32_t packed = ok2 ? hip_dsa_numeric::bf16_pair(p2, dl2) : 0u;
                            saf_nt_store_u32(pdl + 8 * (size_t)H, packed);
                        }
                        if (g + 12 < ck) {
                            const uint32_t packed = ok3 ? hip_dsa_numeric::bf16_pair(p3, dl3) : 0u;
                            saf_nt_store_u32(pdl + 12 * (size_t)H, packed);
                        }
                    }
                    if constexpr (STATS_LDS_PREFIX)
                        asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(stats_first_k[0]), "+v"(stats_first_k[1])::"memory");
                }
            } else {
                if constexpr (DIRECT_STATS) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                if constexpr (STATS_LDS_PREFIX) {
                    // Same second CTA barrier, now before independent P/dS stores.
                    // All K copies/invalid-row zeros are visible before the reads.
                    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
                    if (nkt < ntiles) read_stats_prefix();
                }
                if (lane_ok) {
// Holes inside the logical traversal still need initialized
// workspace; only whole tiles beyond n_valid_t are omitted.
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int key = kt * CH + g + e * 4;
                        if (key < topk) saf_nt_store_u32(pdl_ws + ((size_t)t * topk + key) * H + h, 0u);
                    }
                }
                if constexpr (STATS_LDS_PREFIX)
                    asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(stats_first_k[0]), "+v"(stats_first_k[1])::"memory");
            }
            if constexpr (!DIRECT_STATS) __syncthreads();
            if constexpr (DIRECT_STATS) {
                // Same barrier as sparse_attn_mla_common.h::lds_barrier: all
                // waves already waited for their global-to-LDS K transfers.
                if constexpr (!STATS_LDS_PREFIX) asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
            } else {
                if (nkt < ntiles) commit_keys(ck_for(nkt), D / 8, KLD1, CH, k_lds, pref, cnt);
                __syncthreads();
            }
        }

        if constexpr (FUSED_TAIL) {
            if (lane_ok) {
#    pragma unroll
                for (int j = 0; j < 4; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        float x = tail_dq[j].x[e];
                        dq[((size_t)t * 64 + h) * 576 + 512 + j * 16 + g + 4 * e] =
                            (x != x) ? (uint16_t)0x7fc0 : hip_dsa_numeric::bf16_rne(x);
                    }
                }
            }
        }
        // Reuse the FP32 dO.O reduction, including the forward output rounding.

        if (lane_ok && g == 0) { atomicAdd(&dsink[h], -P_sink * avg); }

        if constexpr (FUSE_DKV && ND == 36) {
#    pragma unroll
            for (int kk = VALUE_ND; kk < ND; kk++) {
#    pragma unroll
                for (int e = 0; e < 4; e++) df[kk].x[e] = hbf16(0);
            }
        }
        if constexpr (FUSE_DKV) {
            static_assert(PHASE == 1 && (ND == 32 || ND == 36) && FIXED_H == 64);
            constexpr int CG = ND / 4;
            // One CTA owns all 64 heads of one query. Reuse the exact Q/dO
            // BF16 register bits. Exchange 128-column wave slices through a
            // 16-KiB LDS tile, replacing only already-consumed fragments.
            // Constant register indices avoid local-memory array addressing.
            constexpr int TP = CG * 16 + 4, TS = 16 * TP;
            hbf16* tq = (hbf16*)smem_raw;
            hbf16* td = tq + 2 * TS;
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[0 * TS + r * TP + j * 16 + g * 4 + e] = qf[0 * CG + j].x[e];
                        td[0 * TS + r * TP + j * 16 + g * 4 + e] = df[0 * CG + j].x[e];
                    }
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[0 * CG + j].x[e] = tq[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[0 * CG + j].x[e] = td[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[0 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[0 * CG + j].x[0])
                                 : "memory");
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[0 * TS + r * TP + j * 16 + g * 4 + e] = qf[1 * CG + j].x[e];
                        td[0 * TS + r * TP + j * 16 + g * 4 + e] = df[1 * CG + j].x[e];
                    }
                }
            }
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[1 * TS + r * TP + j * 16 + g * 4 + e] = qf[0 * CG + j].x[e];
                        td[1 * TS + r * TP + j * 16 + g * 4 + e] = df[0 * CG + j].x[e];
                    }
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[1 * CG + j].x[e] = tq[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[1 * CG + j].x[e] = td[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[1 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[1 * CG + j].x[0])
                                 : "memory");
                }
            }
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[0 * CG + j].x[e] = tq[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[0 * CG + j].x[e] = td[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[0 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[0 * CG + j].x[0])
                                 : "memory");
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[0 * TS + r * TP + j * 16 + g * 4 + e] = qf[2 * CG + j].x[e];
                        td[0 * TS + r * TP + j * 16 + g * 4 + e] = df[2 * CG + j].x[e];
                    }
                }
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[1 * TS + r * TP + j * 16 + g * 4 + e] = qf[0 * CG + j].x[e];
                        td[1 * TS + r * TP + j * 16 + g * 4 + e] = df[0 * CG + j].x[e];
                    }
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[2 * CG + j].x[e] = tq[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[2 * CG + j].x[e] = td[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[2 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[2 * CG + j].x[0])
                                 : "memory");
                }
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[0 * CG + j].x[e] = tq[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[0 * CG + j].x[e] = td[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[0 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[0 * CG + j].x[0])
                                 : "memory");
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[0 * TS + r * TP + j * 16 + g * 4 + e] = qf[3 * CG + j].x[e];
                        td[0 * TS + r * TP + j * 16 + g * 4 + e] = df[3 * CG + j].x[e];
                    }
                }
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[1 * TS + r * TP + j * 16 + g * 4 + e] = qf[0 * CG + j].x[e];
                        td[1 * TS + r * TP + j * 16 + g * 4 + e] = df[0 * CG + j].x[e];
                    }
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[3 * CG + j].x[e] = tq[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[3 * CG + j].x[e] = td[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[3 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[3 * CG + j].x[0])
                                 : "memory");
                }
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[0 * CG + j].x[e] = tq[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[0 * CG + j].x[e] = td[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[0 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[0 * CG + j].x[0])
                                 : "memory");
                }
            }
            __syncthreads();
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[0 * TS + r * TP + j * 16 + g * 4 + e] = qf[1 * CG + j].x[e];
                        td[0 * TS + r * TP + j * 16 + g * 4 + e] = df[1 * CG + j].x[e];
                    }
                }
            }
            __syncthreads();
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[1 * CG + j].x[e] = tq[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[1 * CG + j].x[e] = td[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[1 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[1 * CG + j].x[0])
                                 : "memory");
                }
            }
            __syncthreads();
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[0 * TS + r * TP + j * 16 + g * 4 + e] = qf[2 * CG + j].x[e];
                        td[0 * TS + r * TP + j * 16 + g * 4 + e] = df[2 * CG + j].x[e];
                    }
                }
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[1 * TS + r * TP + j * 16 + g * 4 + e] = qf[1 * CG + j].x[e];
                        td[1 * TS + r * TP + j * 16 + g * 4 + e] = df[1 * CG + j].x[e];
                    }
                }
            }
            __syncthreads();
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[2 * CG + j].x[e] = tq[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[2 * CG + j].x[e] = td[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[2 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[2 * CG + j].x[0])
                                 : "memory");
                }
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[1 * CG + j].x[e] = tq[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[1 * CG + j].x[e] = td[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[1 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[1 * CG + j].x[0])
                                 : "memory");
                }
            }
            __syncthreads();
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[0 * TS + r * TP + j * 16 + g * 4 + e] = qf[3 * CG + j].x[e];
                        td[0 * TS + r * TP + j * 16 + g * 4 + e] = df[3 * CG + j].x[e];
                    }
                }
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[1 * TS + r * TP + j * 16 + g * 4 + e] = qf[1 * CG + j].x[e];
                        td[1 * TS + r * TP + j * 16 + g * 4 + e] = df[1 * CG + j].x[e];
                    }
                }
            }
            __syncthreads();
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[3 * CG + j].x[e] = tq[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[3 * CG + j].x[e] = td[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[3 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[3 * CG + j].x[0])
                                 : "memory");
                }
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[1 * CG + j].x[e] = tq[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[1 * CG + j].x[e] = td[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[1 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[1 * CG + j].x[0])
                                 : "memory");
                }
            }
            __syncthreads();
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[0 * TS + r * TP + j * 16 + g * 4 + e] = qf[2 * CG + j].x[e];
                        td[0 * TS + r * TP + j * 16 + g * 4 + e] = df[2 * CG + j].x[e];
                    }
                }
            }
            __syncthreads();
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[2 * CG + j].x[e] = tq[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[2 * CG + j].x[e] = td[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[2 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[2 * CG + j].x[0])
                                 : "memory");
                }
            }
            __syncthreads();
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[0 * TS + r * TP + j * 16 + g * 4 + e] = qf[3 * CG + j].x[e];
                        td[0 * TS + r * TP + j * 16 + g * 4 + e] = df[3 * CG + j].x[e];
                    }
                }
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[1 * TS + r * TP + j * 16 + g * 4 + e] = qf[2 * CG + j].x[e];
                        td[1 * TS + r * TP + j * 16 + g * 4 + e] = df[2 * CG + j].x[e];
                    }
                }
            }
            __syncthreads();
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[3 * CG + j].x[e] = tq[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[3 * CG + j].x[e] = td[1 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[3 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[3 * CG + j].x[0])
                                 : "memory");
                }
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[2 * CG + j].x[e] = tq[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[2 * CG + j].x[e] = td[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[2 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[2 * CG + j].x[0])
                                 : "memory");
                }
            }
            __syncthreads();
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        tq[0 * TS + r * TP + j * 16 + g * 4 + e] = qf[3 * CG + j].x[e];
                        td[0 * TS + r * TP + j * 16 + g * 4 + e] = df[3 * CG + j].x[e];
                    }
                }
            }
            __syncthreads();
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < CG; j++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        qf[3 * CG + j].x[e] = tq[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                        df[3 * CG + j].x[e] = td[0 * TS + (g * 4 + e) * TP + j * 16 + r];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&qf[3 * CG + j].x[0]),
                                 "v"(*(const uint64_t*)&df[3 * CG + j].x[0])
                                 : "memory");
                }
            }
            __syncthreads();
            // Stats used non-temporal stores; make this CTA's packed workspace
            // visible before the cross-wave dKV reads. No other query is read.
            __threadfence_block();
            __syncthreads();
            if constexpr (GDKV == 11) {
                // qf/df now have the exact native dKV B-fragment orientation.
                saf_u4e* packed_qd = (saf_u4e*)(pdl_ws + (size_t)T * 2048 * 64 + (size_t)T * 4096);
#    pragma unroll
                for (int hg = 0; hg < 4; hg++) {
#    pragma unroll
                    for (int j = 0; j < CG; j++) {
                        saf_u4e bits;
                        bits[0] = ((uint32_t*)&qf[hg * CG + j].x[0])[0];
                        bits[1] = ((uint32_t*)&qf[hg * CG + j].x[0])[1];
                        bits[2] = ((uint32_t*)&df[hg * CG + j].x[0])[0];
                        bits[3] = ((uint32_t*)&df[hg * CG + j].x[0])[1];
                        __builtin_nontemporal_store(bits,
                                                    packed_qd + (((size_t)t * 36 + warp * 9 + j) * 4 + hg) * 64 + lane);
                    }
                }
            } else {
                constexpr int PLD = 66, PTILE = 16 * PLD;
                hbf16* pd = (hbf16*)smem_raw;
                // Stats and Q/dO transpose have finished: reuse their LDS for indices.
                // 2*16*66 BF16 values + 3072 ints = 16512 B, within the existing
                // 64*68*sizeof(float) allocation. Larger K keeps global index reads.
                const bool cache_dkv_indices = topk <= 3072;
                int* cached_indices          = (int*)(pd + 2 * PTILE);
                if (cache_dkv_indices) {
                    for (int i = threadIdx.x; i < n_valid_t; i += 256) cached_indices[i] = idxs[(size_t)t * topk + i];
                    __syncthreads();
                }
                // With eight column tiles per wave, dKV has the same MMAC work
                // as the existing H128 cached prefetch path. Retain next P/dS in
                // registers while consuming the current LDS tile.
                const int u = threadIdx.x, row = u / 16, head = (u % 16) * 4;
                saf_u4e packed;
                int carried_key =
                    r < n_valid_t ? (cache_dkv_indices ? cached_indices[r] : idxs[(size_t)t * topk + r]) : -1;
                auto prefetch_pd = [&](int next, int next_key) {
                    if (!__builtin_amdgcn_readfirstlane(__any(next_key >= 0 && next_key < T_kv))) return;
                    // A nonempty ballot implies next < n_valid_t <= topk. With a
                    // constant physical K divisible by 16, this whole tile is in
                    // bounds. Logical key/length masks remain in the consumers.
                    if constexpr (TOPK > 0 && TOPK % 16 == 0)
                        packed = saf_nt_load_u4(pdl_ws + ((size_t)t * topk + next + row) * 64 + head);
                    else
                        packed = next + row < topk
                                     ? saf_nt_load_u4(pdl_ws + ((size_t)t * topk + next + row) * 64 + head)
                                     : (saf_u4e){0, 0, 0, 0};
                };
                prefetch_pd(0, carried_key);
#    pragma unroll 1
                for (int kt = 0; kt < n_valid_t; kt += 16) {
                    const int key = carried_key;
                    carried_key   = kt + 16 + r < n_valid_t ? (cache_dkv_indices ? cached_indices[kt + 16 + r]
                                                                                 : idxs[(size_t)t * topk + kt + 16 + r])
                                                            : -1;
                    if (!__builtin_amdgcn_readfirstlane(__any(key >= 0 && key < T_kv))) {
                        prefetch_pd(kt + 16, carried_key);
                        continue;
                    }
                    uint32_t* pp    = (uint32_t*)(pd + row * PLD + head);
                    uint32_t* dd    = (uint32_t*)(pd + PTILE + row * PLD + head);
                    const saf_u4e x = packed;
                    pp[0]           = (x.x & 65535) | (x.y << 16);
                    pp[1]           = (x.z & 65535) | (x.w << 16);
                    dd[0]           = (x.x >> 16) | (x.y & 0xffff0000);
                    dd[1]           = (x.z >> 16) | (x.w & 0xffff0000);
                    __syncthreads();
                    prefetch_pd(kt + 16, carried_key);
                    // Bound GLM atomic return live ranges; retain complete return waits.
                    constexpr int ATOMIC_BATCH = ND == 36 ? 3 : CG;
                    saf_floatx4 atomic_values[ATOMIC_BATCH];
                    constexpr int KEY_PACK = DISPATCH_GUARD == 2 ? 8 : 1;
                    const uint64_t element_offset =
                        KEY_PACK == 8
                            ? (((uint64_t)(uint32_t)key / 8 * ND + warp * CG) * 128 + g * 32 + ((uint32_t)key % 8) * 4)
                            : ((uint64_t)(uint32_t)key * D + warp * (CG * 16) + 4 * g);
                    uint64_t atomic_address    = (uint64_t)dkv_float + element_offset * sizeof(float);
                    const uint64_t atomic_mask = __ballot(key >= 0 && key < T_kv);
                    DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[4], da[4];
#    pragma unroll
                    for (int hg = 0; hg < 4; hg++) {
                        du_load_matrix_sync(pa[hg], pd + hg * 16, PLD);
                        du_load_matrix_sync(da[hg], pd + PTILE + hg * 16, PLD);
                    }
#    pragma unroll
                    for (int j = 0; j < CG; j++) {
                        DUFragment<accumulator, 16, 16, 16, float> acc;
                        du_fill_fragment(acc, 0.f);
#    pragma unroll
                        for (int hg = 0; hg < 4; hg++) {
                            DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> qb, db;
#    pragma unroll
                            for (int e = 0; e < 4; e++) {
                                qb.x[e] = qf[hg * CG + j].x[e];
                                db.x[e] = df[hg * CG + j].x[e];
                            }
                            du_mma_sync(acc, pa[hg], db, acc);
                            du_mma_sync(acc, da[hg], qb, acc);
                        }
                        atomic_values[j % ATOMIC_BATCH] = {acc.x[0], acc.x[1], acc.x[2], acc.x[3]};
                        // Preserve real atomic return/value registers until completion.
                        // Mask only the atomic, keeping C++ control flow uniform.
                        uint64_t saved_exec;
                        asm volatile("s_and_saveexec_b64 %1, %3\n\t"
                             "global_atomic_add_f32_x4 %0, %2, %0, off offset:%4 glc\n\t"
                             "s_mov_b64 exec, %1"
                    : "+v"(atomic_values[j%ATOMIC_BATCH]), "=&s"(saved_exec), "+v"(atomic_address)
                    : "s"(atomic_mask), "n"(j*64*KEY_PACK) : "exec", "memory");
                        if constexpr (ND == 36) {
                            if ((j + 1) % ATOMIC_BATCH == 0 || j + 1 == CG) {
                                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#    pragma unroll
                                for (int z = 0; z < ATOMIC_BATCH; z++)
                                    asm volatile("" ::"v"(atomic_values[z]), "v"(atomic_address) : "memory");
                            }
                        }
                    }
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#    pragma unroll
                    for (int j = 0; j < ATOMIC_BATCH; j++)
                        asm volatile("" ::"v"(atomic_values[j % ATOMIC_BATCH]), "v"(atomic_address) : "memory");
                    __syncthreads();
                }
                __threadfence();
                __syncthreads();
                __shared__ int fused_previous;
                if (threadIdx.x == 0) fused_previous = atomicAdd(done_count, 1);
                __syncthreads();
                if constexpr (DISPATCH_GUARD == 2)
                    saf_multi_cast_keypacked(done_count, dkv_float, dkv, T_kv, D, T, fused_previous);
                else
                    saf_multi_cast_swizzled(done_count, dkv_float, dkv, T_kv, D, T, fused_previous);
            }
        }
    }  // stats phase

    // Separate phase launches are ordered by their HIP stream. Only the
    // one-code-object fallback needs an intra-kernel workspace hand-off.
    if constexpr (PHASE == 0) {
        __threadfence();
        __syncthreads();
    }

    if constexpr (PHASE != 1) {
        // ---- dkv: P/dl tiles for G chunks are staged from this block's own
        // p_ws / dl_ws scratch (written by pass 1 above) into LDS, then the
        // dO/Q B-fragments are register-cached per (d-tile group, head
        // sub-block) and reused across the G chunks (the old layout re-read
        // dO/Q from global once per chunk).  Atomics stay per key-chunk, so
        // nothing else changes; dO/Q global loads drop ~Gx. ----
        if (dkv_role) {
            constexpr int G = GDKV;  // P/dl chunk pairs resident in LDS
            // A 66-element row keeps the cooperative uint32 staging aligned and
            // fits five P/dl chunk pairs in a 21120-byte aliased LDS region.
            constexpr int PLDM    = 66;
            constexpr int PTILE   = 16 * PLDM;  // bf16 per P or dl tile
            const int dkv_hb      = hb;
            const int dkv_hgroups = 4;
            const int dkv_h4      = 32;
            // Partial 64-head slabs may have fewer than four active waves.  Split
            // the D tiles across those waves so every column is covered; using a
            // fixed divisor of four dropped the upper D tiles for H < 49.
            const int dkv_ns = ns;
            const int t0     = (warp * ND) / dkv_ns;
            const int t1     = ((warp + 1) * ND) / dkv_ns;
            // Stage G chunks of P/dl (16 keys x 64 heads, padded ldm) into the
            // pass-1 K region, which is dead after pass 1.  Tiles beyond the
            // chunk count / ck rows are zeroed (their K rows are zeroed too).
            saf_intx4 pdl_desc;
            // Rebase the descriptor per query.  A Pro workspace already exceeds
            // 2 GiB at T=4096 and 4 GiB at T=8192; a whole-workspace voffset
            // would hit the descriptor bound or wrap.  soffset is a byte offset,
            // not the high 32 bits of a 64-bit address.
            const uint64_t pdl_addr = (uint64_t)(pdl_ws + (size_t)t * topk * H);
            pdl_desc[0]             = __builtin_amdgcn_readfirstlane((int)(uint32_t)(pdl_addr & 0xffffffffu));
            pdl_desc[1]             = __builtin_amdgcn_readfirstlane((int)((uint32_t)(pdl_addr >> 32) | 0x40000000u));
            pdl_desc[2]             = __builtin_amdgcn_readfirstlane((int)0x80000000u);
            pdl_desc[3]             = __builtin_amdgcn_readfirstlane((int)0x00020000u);
            for (int ktG = 0; ktG < ntiles; ktG += G) {
                hbf16* pd = k_lds;
                // A full slab has an even, contiguous head range.  Load four
                // packed P/dl words (four heads) with one 16B buffer instruction,
                // unpack them into two P and two dl dwords, and write both LDS
                // planes.  The old scalar path remains for partial slabs.
                const bool full_slab = H >= dkv_hb + 2 * dkv_h4;
                if (full_slab) {
                    const int hp = (dkv_h4 + 1) >> 1;
                    for (int u = threadIdx.x; u < G * 16 * hp; u += THREADS) {
                        const int c    = u / (16 * hp);
                        const int rem  = u - c * (16 * hp);
                        const int key  = rem / hp;
                        const int slot = (rem - key * hp) << 1;
                        const int kk   = ktG + c;
                        uint32_t* dstp = (uint32_t*)(pd + c * (2 * PTILE) + key * PLDM + slot * 2);
                        uint32_t* dstd = (uint32_t*)(pd + c * (2 * PTILE) + PTILE + key * PLDM + slot * 2);
                        if (kk < ntiles && key < ck_for(kk)) {
                            const uint64_t boff = (((size_t)kk * CH + key) * H + dkv_hb + slot * 2) * sizeof(uint32_t);
                            const int voff      = (int)(uint32_t)boff;
                            const saf_intx4 x   = saf_raw_buffer_load_i32x4(pdl_desc, voff, 0, 3);
                            const uint32_t p0   = (uint32_t)x[0];
                            const uint32_t p1   = (uint32_t)x[1];
                            const uint32_t p2   = (uint32_t)x[2];
                            const uint32_t p3   = (uint32_t)x[3];
                            dstp[0]             = (p0 & 0xffffu) | ((p1 & 0xffffu) << 16);
                            dstd[0]             = (p0 >> 16) | ((p1 >> 16) << 16);
                            if (slot + 1 < dkv_h4) {
                                dstp[1] = (p2 & 0xffffu) | ((p3 & 0xffffu) << 16);
                                dstd[1] = (p2 >> 16) | ((p3 >> 16) << 16);
                            }
                        } else {
                            dstp[0] = dstd[0] = 0;
                            if (slot + 1 < dkv_h4) dstp[1] = dstd[1] = 0;
                        }
                    }
                } else {
                    // 4B staging stores cover 2 bf16 heads each; the tile keeps
                    // 64 heads per key (local slab head = hb + head4*2).
                    for (int u = threadIdx.x; u < G * 2 * 16 * dkv_h4; u += THREADS) {
                        const int c   = u / (2 * 16 * dkv_h4);
                        const int rem = u - c * (2 * 16 * dkv_h4);
                        const int arr = rem / (16 * dkv_h4);
                        const int key = (rem % (16 * dkv_h4)) / dkv_h4;
                        const int h4  = rem % dkv_h4;
                        const int kk  = ktG + c;
                        uint32_t* dst = (uint32_t*)(pd + c * (2 * PTILE) + arr * PTILE + key * PLDM + h4 * 2);
                        if (kk < ntiles && key < ck_for(kk)) {
                            const int hs        = dkv_hb + h4 * 2;
                            const uint32_t* src = pdl_ws + ((size_t)t * topk + kk * CH + key) * H + hs;
                            if (hs + 1 < H) {
                                const uint64_t x     = saf_nt_load_u64((const void*)src);
                                const uint32_t ppack = (x & 0xFFFFu) | (((x >> 32) & 0xFFFFu) << 16);
                                const uint32_t dpack = ((x >> 16) & 0xFFFFu) | (((x >> 48) & 0xFFFFu) << 16);
                                *dst                 = (arr == 0) ? ppack : dpack;
                            } else if (hs < H) {
                                const uint32_t x = saf_nt_load_u32((const void*)src);
                                *dst             = (arr == 0) ? (x & 0xFFFFu) : (x >> 16);
                            } else {
                                *dst = 0;
                            }
                        } else {
                            *dst = 0;
                        }
                    }
                }
                __syncthreads();  // P/dl tiles visible to all warps
                if (warp < ns) {
                    for (int tg = t0; tg < t1; tg += 4) {
                        const int t_hi = (tg + 4 < t1) ? tg + 4 : t1;
                        DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bb[8][4], qb[8][4];
#    pragma unroll
                        for (int s = 0; s < dkv_hgroups; s++) {
                            const int h0s = dkv_hb + s * 16;
                            if (h0s >= H) break;
#    pragma unroll
                            for (int j = 0; j < 4; j++) {
                                const int tt = tg + j;
                                if (tt >= t_hi) break;
                                const hbf16* bp  = drow + (size_t)h0s * value_stride + (tt * 16 + r);
                                const hbf16* qp2 = qrow + (size_t)h0s * D + (tt * 16 + r);
                                const int fr     = h0s + (g << 2);
                                bb[s][j].x[0]    = (fr < H) ? bp[(size_t)(g << 2) * D] : hbf16(0);
                                bb[s][j].x[1]    = (fr + 1 < H) ? bp[((size_t)(g << 2) + 1) * D] : hbf16(0);
                                bb[s][j].x[2]    = (fr + 2 < H) ? bp[((size_t)(g << 2) + 2) * D] : hbf16(0);
                                bb[s][j].x[3]    = (fr + 3 < H) ? bp[((size_t)(g << 2) + 3) * D] : hbf16(0);
                                qb[s][j].x[0]    = (fr < H) ? qp2[(size_t)(g << 2) * D] : hbf16(0);
                                qb[s][j].x[1]    = (fr + 1 < H) ? qp2[((size_t)(g << 2) + 1) * D] : hbf16(0);
                                qb[s][j].x[2]    = (fr + 2 < H) ? qp2[((size_t)(g << 2) + 2) * D] : hbf16(0);
                                qb[s][j].x[3]    = (fr + 3 < H) ? qp2[((size_t)(g << 2) + 3) * D] : hbf16(0);
                            }
                        }
#    pragma unroll
                        for (int c = 0; c < G; c++) {
                            const int kt = ktG + c;
                            if (kt >= ntiles) break;
                            const int ck = ck_for(kt);
                            DUFragment<accumulator, 16, 16, 16, float> kacc[4];
#    pragma unroll
                            for (int j = 0; j < 4; j++) du_fill_fragment(kacc[j], 0.0f);
#    pragma unroll
                            for (int s = 0; s < dkv_hgroups; s++) {
                                const int h0s = dkv_hb + s * 16;
                                if (h0s >= H) break;
                                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa, da;
                                du_load_matrix_sync(pa, pd + c * (2 * PTILE) + s * 16, PLDM);
                                du_load_matrix_sync(da, pd + c * (2 * PTILE) + PTILE + s * 16, PLDM);
#    pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    if (tg + j < t_hi) {
                                        du_mma_sync(kacc[j], pa, bb[s][j], kacc[j]);
                                        du_mma_sync(kacc[j], da, qb[s][j], kacc[j]);
                                    }
                                }
                            }
                            const int kidx = (r < ck) ? load_index(kt * CH + r) : -1;
                            if (r < ck && kidx >= 0 && kidx < T_kv) {
#    pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    if (tg + j < t_hi) {
                                        float* pk = dkv_float + (size_t)kidx * D + (tg + j) * 16;
#    if defined(__gfx936__) || defined(__gfx938__)
                                        // MMAC lanes g=0..3 own columns
                                        // {g,g+4,g+8,g+12}.  The fused2 private
                                        // workspace stores exactly this interleaved
                                        // order, so one native x4 atomic can commit
                                        // the fragment without a lane permutation;
                                        // the final cast restores contiguous D.
                                        saf_atomic_add_mmac_row_swizzled(pk, g, kacc[j].x[0], kacc[j].x[1],
                                                                         kacc[j].x[2], kacc[j].x[3]);
#    else
                                        pk += g;
                                        atomicAdd(&pk[0], kacc[j].x[0]);
                                        atomicAdd(&pk[4], kacc[j].x[1]);
                                        atomicAdd(&pk[8], kacc[j].x[2]);
                                        atomicAdd(&pk[12], kacc[j].x[3]);
#    endif
                                    }
                                }
                            }
                        }
                    }
                }
                __syncthreads();  // P/dl tiles consumed; next group may overwrite
            }
        }  // dkv_role
        // ---- Pass 2: dq, D-sliced (ND/NDS slice passes over the chunk loop) ----
        constexpr int NSL = (ND + NDS - 1) / NDS;
        if (dq_role) {
            // Relocate the descriptor for every query using a 64-bit base. Only
            // the bounded offset within that query uses buffer addressing; the
            // full workspace can cross 2/4 GiB. Keep flat loads for a single
            // query whose workspace itself exceeds the descriptor's range.
            // DQ_BUFFER is selected by the HIP launcher, so the common loop
            // carries neither a range branch nor a live fallback address.
            const uint32_t* dq_pdl_query = pdl_ws + (size_t)t * topk * H;
            const uint64_t dq_pdl_addr   = (uint64_t)dq_pdl_query;
            const saf_intx4 dq_pdl_desc  = {(int)(uint32_t)dq_pdl_addr,
                                            (int)((dq_pdl_addr >> 32) | (DQ_INDEX_AHEAD ? 0u : 0x40000000u)),
                                            (int)0x80000000u, 0x00020000};
            // Split-role blocks execute exactly one slice.  The legacy block
            // executes all slices serially, preserving the old A/B path.
            const int sl_begin = SPLIT_ROLES ? (role - 1) : 0;
            const int sl_end   = SPLIT_ROLES ? role : NSL;
#    pragma unroll 1
            for (int sl = sl_begin; sl < sl_end; sl++) {
                const int D0 = sl * SLICE;
                if constexpr (DQ_INDEX_AHEAD) dq_future_ready = false;
                DUFragment<accumulator, 16, 16, 16, float> dqacc[NDS];
#    pragma unroll
                for (int d16 = 0; d16 < NDS; d16++) du_fill_fragment(dqacc[d16], 0.0f);

                {
                    uint64_t pref[PREF2];
                    int cnt = prefetch_keys(0, ck_for(0), SLICE / 8, KLD2, D0, pref);
                    commit_keys(ck_for(0), SLICE / 8, KLD2, 16, k2_lds, pref, cnt);
                }
                __syncthreads();
#    pragma unroll 1
                for (int kt = 0; kt < ntiles; kt++) {
                    uint64_t pref[PREF2];
                    int cnt       = 0;
                    const int nkt = kt + 1;
                    if constexpr (DQ_INDEX_AHEAD) {
                        const int next_ck = ck_for(nkt);
#    pragma unroll
                        for (int j = 0; j < (16 * (SLICE / 8) + THREADS - 1) / THREADS; j++) {
                            const int n = (threadIdx.x + j * THREADS) / (SLICE / 8);
                            if constexpr (TOPK > 0 && TOPK <= 1152) {
                                // Constant physical row bound makes out-of-range
                                // future entries return zero without an EXEC branch.
                                dq_future_index[j] = saf_raw_buffer_load_i32(dq_index_desc, (nkt * CH + n) * 4, 0, 0);
                            } else
                                dq_future_index[j] = n < next_ck ? load_index(nkt * CH + n) : 0;
                        }
                        dq_future_ready = true;
                    }
                    if constexpr (!DQ_KBUFFER) {
                        if (nkt < ntiles) cnt = prefetch_keys(nkt, ck_for(nkt), SLICE / 8, KLD2, D0, pref);
                    }
                    const int ck = ck_for(kt);

                    // dl for this chunk from the block's own scratch; phantom
                    // heads read 0 (their dl slots were never written).  For
                    // CH < 16 chunks the tile rows beyond ck are zeroed (their
                    // K rows are zeroed too, so they cannot leak into dq).
                    if constexpr (PHASE != 4) {
                        const int koff = kt * CH;
                        const int c0 = koff + g, c1 = c0 + 4, c2 = c0 + 8, c3 = c0 + 12;
                        const uint32_t* pdlrow = pdl_ws + ((size_t)t * topk + koff) * H + h;
                        const float dl0 = (lane_ok && c0 < topk)
                                              ? __bfloat162float((bf16)(saf_nt_load_u32(pdlrow + (size_t)g * H) >> 16))
                                              : 0.0f;
                        const float dl1 =
                            (lane_ok && c1 < topk)
                                ? __bfloat162float((bf16)(saf_nt_load_u32(pdlrow + (size_t)(g + 4) * H) >> 16))
                                : 0.0f;
                        const float dl2 =
                            (lane_ok && c2 < topk)
                                ? __bfloat162float((bf16)(saf_nt_load_u32(pdlrow + (size_t)(g + 8) * H) >> 16))
                                : 0.0f;
                        const float dl3 =
                            (lane_ok && c3 < topk)
                                ? __bfloat162float((bf16)(saf_nt_load_u32(pdlrow + (size_t)(g + 12) * H) >> 16))
                                : 0.0f;
                        hbf16* dw                     = pt2_lds + 4 * 256 + warp * 256;
                        dw[(size_t)(g + 0) * 16 + r]  = __float2bfloat16(dl0);
                        dw[(size_t)(g + 4) * 16 + r]  = __float2bfloat16(dl1);
                        dw[(size_t)(g + 8) * 16 + r]  = __float2bfloat16(dl2);
                        dw[(size_t)(g + 12) * 16 + r] = __float2bfloat16(dl3);
                    }
                    // Each wave writes and consumes only its own 16x16 dl tile
                    // (pt2_lds + 4*256 + warp*256).  A block-wide barrier here
                    // serialized otherwise independent waves and showed up as a
                    // large SQ_WAIT_INST_LDS component in phase 2.  The compiler
                    // emits the required same-wave LDS dependency wait between
                    // these stores and du_load_matrix_sync; the block barrier
                    // below remains for the cooperatively staged K slice.

                    const int ti       = kt * CH + r;
                    const int tile_key = all_keys_valid ? 0 : (ti < n_valid_t ? load_index(ti) : -1);
                    constexpr bool DQ_VALID_FAST =
                        (PHASE == 4 && ND == 32 && TOPK == 1152 && NDS == 16 && DQ_WAVES == 4 && DQ_KBUFFER) ||
                        (PHASE == 4 && ND == 36 && TOPK == 2048 && FIXED_H == 64 && NDS == 8 && DQ_WAVES == 4 &&
                         DQ_KBUFFER);
                    // The CTA-wide prologue already proved every logical key is
                    // valid. Partial final tiles still keep per-key load guards.
                    const bool active =
                        DQ_VALID_FAST && all_keys_valid
                            ? true
                            : (CACHE_TILE_MASK ? tile_masks[kt] != 0
                                               : all_keys_valid || __builtin_amdgcn_readfirstlane(
                                                                       __any(tile_key >= 0 && tile_key < T_kv)));

                    if (active && hq < H) {
                        DUFragment<matrix_a, 16, 16, 16, hbf16, col_major> dlf;
                        if constexpr (PHASE == 4) {
// DTK du_mma.hpp A/col-major fragment mapping:
// head=lane%16, key=4*(lane/16)+e. The workspace
// already has this logical layout. Load the BF16
// bits directly, without a round-trip through LDS.
#    pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                const int key = kt * CH + 4 * g + e;
                                // Normal-cache reads retain this query's P/dS for its
                                // other dQ slice. Producer/consumer kernel ordering is unchanged.
                                const unsigned offset = ((unsigned)key * H + h) * 4u;
                                const uint32_t packed =
                                    (lane_ok && key < topk)
                                        ? (DQ_BUFFER ? (uint32_t)saf_raw_buffer_load_i32(
                                                           dq_pdl_desc, offset, 0,
                                                           ((ND == 32 && (TOPK == 1024 || TOPK == 1152)) ||
                                                            (ND == 36 && TOPK == 2048))
                                                               ? 0
                                                               : 3)
                                                     : saf_nt_load_u32(dq_pdl_query + (size_t)key * H + h))
                                        : 0;
                                // Let LLVM schedule all four loads and insert
                                // their dependency waits before unpacking dS.
                                *(uint16_t*)&dlf.x[e] = (uint16_t)(packed >> 16);
                            }
                        } else {
                            du_load_matrix_sync(dlf, pt2_lds + 4 * 256 + warp * 256, 16);
                        }
                        if constexpr (DQ_KBUFFER) { cnt = prefetch_keys(nkt, ck_for(nkt), SLICE / 8, KLD2, D0, pref); }
                        if constexpr (PHASE == 4) {
                            // Each matrix read returns even/odd column fragments for
                            // a 16-key x 32-column tile. Issue four independent reads
                            // before waiting, then consume the packed BF16 operands.
                            // Both gfx936/gfx938 emit this source-backed DS opcode.
                            typedef unsigned int mat4 __attribute__((ext_vector_type(4)));
#    pragma unroll
                            for (int d16 = 0; d16 < NDS; d16 += 8) {
                                // D576's final 256-column slab contains only 64
                                // real columns. Keep the existing grid/pipeline,
                                // but omit LDS reads and MMACs for its zero tail.
                                if constexpr (ND % NDS != 0) {
                                    if (D0 + d16 * 16 >= D) break;
                                }
                                mat4 packed[4];
#    pragma unroll
                                for (int z = 0; z < 4; z++) {
                                    if constexpr (ND % NDS != 0) {
                                        if (D0 + (d16 + 2 * z) * 16 >= D) break;
                                    }
                                    const unsigned addr = (unsigned)(size_t)(k2_lds + (d16 / 2 + z) * 512 +
                                                                             (lane / 4) * 32 + (lane % 4) * 8);
                                    asm volatile("ds_read_m32x16_b16_alt %0, %1"
                                                 : "=v"(packed[z])
                                                 : "v"(addr)
                                                 : "memory");
                                }
                                // Drain one fragment at a time instead of waiting
                                // for the whole group. `lgkmcnt(3-z)` releases the
                                // z-th read (0 = oldest) as soon as it lands, so the
                                // MMAC pair for z runs while reads z+1..3 are still
                                // in flight. Reads, MMAC count, order and operands
                                // are unchanged; only the wait granularity is finer.
                                // The short D576 tail keeps the original full drain.
                                if constexpr (ND % NDS != 0) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#    pragma unroll
                                for (int z = 0; z < 4; z++) {
                                    if constexpr (ND % NDS != 0) {
                                        if (D0 + (d16 + 2 * z) * 16 >= D) break;
                                    } else {
                                        asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - z) : "memory");
                                    }
                                    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                                    ((uint32_t*)&b0.x[0])[0] = packed[z][0];
                                    ((uint32_t*)&b0.x[0])[1] = packed[z][1];
                                    ((uint32_t*)&b1.x[0])[0] = packed[z][2];
                                    ((uint32_t*)&b1.x[0])[1] = packed[z][3];
                                    du_mma_sync(dqacc[d16 + 2 * z], dlf, b0, dqacc[d16 + 2 * z]);
                                    du_mma_sync(dqacc[d16 + 2 * z + 1], dlf, b1, dqacc[d16 + 2 * z + 1]);
                                }
                            }
                        } else {
#    pragma unroll
                            for (int d16 = 0; d16 + 1 < NDS; d16 += 2) {
                                DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                                du_load_matrix_sync(b0, k2_lds + (size_t)d16 * 16, KLD2);
                                du_load_matrix_sync(b1, k2_lds + (size_t)(d16 + 1) * 16, KLD2);
                                du_mma_sync(dqacc[d16], dlf, b0, dqacc[d16]);
                                du_mma_sync(dqacc[d16 + 1], dlf, b1, dqacc[d16 + 1]);
                            }
                        }
                        if constexpr ((NDS & 1) != 0) {
                            DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b;
                            du_load_matrix_sync(b, k2_lds + (size_t)(NDS - 1) * 16, KLD2);
                            du_mma_sync(dqacc[NDS - 1], dlf, b, dqacc[NDS - 1]);
                        }
                    }
                    if constexpr (DQ_KBUFFER) {
                        if (!(active && hq < H)) cnt = prefetch_keys(nkt, ck_for(nkt), SLICE / 8, KLD2, D0, pref);
                    }
                    __syncthreads();
                    if (nkt < ntiles) commit_keys(ck_for(nkt), SLICE / 8, KLD2, 16, k2_lds, pref, cnt);
                    __syncthreads();
                }

                // dq epilogue: this block's head slab, slice columns only.
                if (lane_ok) {
                    bf16* dqbase = dq + ((size_t)t * H + h) * D + D0;
                    if constexpr (PHASE == 4) {
#    pragma unroll
                        for (int d16 = 0; d16 < NDS; d16 += 2) {
                            const int dcol = (d16 / 2) * 32 + g * 8;
                            if constexpr (ND % NDS != 0) {
                                if (D0 + dcol >= D) continue;
                            }
                            // Transpose already-rounded pairs within each head,
                            // preserving NaN canonicalization and every output bit.
                            uint32_t v[4], x[4], y[4];
#    pragma unroll
                            for (int z = 0; z < 4; ++z) {
                                const float a = dqacc[d16].x[z], b = dqacc[d16 + 1].x[z];
                                const uint32_t lo = (a != a) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(a);
                                const uint32_t hi = (b != b) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(b);
                                v[z]              = lo | (hi << 16);
                            }
#    pragma unroll
                            for (int z = 0; z < 4; z++) {
                                uint32_t peer = __shfl_xor(v[z ^ 1], 16);
                                x[z]          = ((g ^ z) & 1) ? peer : v[z];
                            }
#    pragma unroll
                            for (int z = 0; z < 4; z++) {
                                uint32_t peer = __shfl_xor(x[z ^ 2], 32);
                                y[z]          = ((g ^ z) & 2) ? peer : x[z];
                            }
                            __builtin_nontemporal_store((saf_u4e){y[0], y[1], y[2], y[3]}, (saf_u4e*)(dqbase + dcol));
                        }
                    } else {
#    pragma unroll
                        for (int d16 = 0; d16 < NDS; d16++) {
                            // DS-read ALT interleaves the even/odd output columns.
                            const int dcol = PHASE == 4 ? (d16 / 2) * 32 + (d16 % 2) + g * 2 : d16 * 16 + g;
                            if constexpr (ND % NDS != 0) {
                                if (D0 + dcol >= D) continue;
                            }
                            const float q0   = dqacc[d16].x[0];
                            const float q1   = dqacc[d16].x[1];
                            const float q2   = dqacc[d16].x[2];
                            const float q3   = dqacc[d16].x[3];
                            dqbase[dcol + 0] = (q0 != q0) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(q0);
                            dqbase[dcol + (PHASE == 4 ? 8 : 4)] =
                                (q1 != q1) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(q1);
                            dqbase[dcol + (PHASE == 4 ? 16 : 8)] =
                                (q2 != q2) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(q2);
                            dqbase[dcol + (PHASE == 4 ? 24 : 12)] =
                                (q3 != q3) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(q3);
                        }
                    }
                }
            }
        }  // dq_role
        // The last gradient block casts dkv_float after every block has
        // committed its dkv/dsink atomics.
        if (dkv_role) {
            __threadfence();
            __syncthreads();
            __shared__ int fused_last;
            if (threadIdx.x == 0) fused_last = atomicAdd(done_count, 1);
            __syncthreads();
            // Split-role dQ blocks do not participate in done_count; the dKV
            // role count remains grid.x * grid.y.
            const int dkv_blocks = (int)(gridDim.x * gridDim.y);
            saf_multi_cast_swizzled(done_count, dkv_float, dkv, T_kv, D, dkv_blocks, fused_last);
        }
    }  // PHASE != 1
}

// Short-row full backward fusion. Packed BF16 P/dS stay in per-lane VGPRs.
// Stats, dKV, dQ, and the final dKV cast share one compute launch.

__device__ __forceinline__ static unsigned saf_lds_load_u32(const void* p) { return *(const unsigned*)p; }
__device__ __forceinline__ static unsigned long long saf_lds_load_u64(const void* p) {
    return *(const unsigned long long*)p;
}

template<int ND, int NDS, int TOPK = 0, int PHASE = 0, int GDKV = 5, bool SPLIT_ROLES = false, int DQ_WAVES = 4,
         bool DQ_BUFFER = false, bool DQ_KBUFFER = false, int FIXED_H = 0, bool FUSE_DKV = false,
         int DISPATCH_GUARD = 0>
__global__ static void __launch_bounds__((PHASE == 4 ? DQ_WAVES * 64 : SAF_THREADS), 1)
    sparse_attn_bwd_full_regs_kernel(
        const bf16* __restrict__ q,           // (T, H, D)
        const bf16* __restrict__ kv,          // (T_kv, D)
        const bf16* __restrict__ out,         // (T, H, D) fwd output (bf16)
        const bf16* __restrict__ dout,        // (T, H, D)
        const float* __restrict__ lse,        // (T, H) KV-only logsumexp (fwd)
        const float* __restrict__ attn_sink,  // (H,)
        const int* __restrict__ idxs,         // (T, topk)
        const int* __restrict__ topk_length,
        float* __restrict__ dkv_float,         // (T_kv, D) f32; fused2 uses tile-swizzled private layout
        float* __restrict__ dsink,             // (H,) f32 (zeroed by caller)
        uint32_t* __restrict__ unused_pdl_ws,  // (T, topk, H) packed: low=P, high=dl bf16
        bf16* __restrict__ dq,                 // (T, H, D)
        bf16* __restrict__ dkv,                // (T_kv, D) bf16 output
        int* __restrict__ done_count,          // zeroed by caller; last-block cast
        int T, int H, int D_rt, int T_kv, int topk_rt, float softmax_scale, const int* dispatch_guard = nullptr) {
    // Internal stats specialization: expose the actual head stride to the
    // compiler while retaining head/key guards and the arithmetic order.
    if constexpr (DISPATCH_GUARD != 0) {
        const bool fuse = (dispatch_guard[0] & 1) == 0 && (int64_t)dispatch_guard[1] * 4 >= (int64_t)T * topk_rt;
        if (fuse != (DISPATCH_GUARD > 0)) return;
        if constexpr (DISPATCH_GUARD > 0) {
            // Both kernels execute the same query/key arithmetic; the GPU
            // selects one private FP32 accumulator layout for the whole call.
            const bool packed = (dispatch_guard[0] & 2) == 0 && T_kv % 8 == 0;
            if (packed != (DISPATCH_GUARD == 2)) return;
        }
    }
    if constexpr (FIXED_H > 0) H = FIXED_H;

    // dQ shares one K tile across all 128 heads when launched with eight waves.
    // Other phases retain their four-wave ownership and resource budgets.
    constexpr int THREADS = PHASE == 4 ? DQ_WAVES * 64 : SAF_THREADS;
    constexpr int D       = ND * 16;
    constexpr int SLICE   = NDS * 16;
    static_assert(PHASE != 4 || (NDS % 8 == 0 && (16 * (SLICE / 8)) % THREADS == 0),
                  "dQ matrix reads require complete staging and eight-column-tile groups");
    const int topk = (TOPK > 0) ? TOPK : topk_rt;
    (void)D_rt;
    // Consecutive H128 stats CTAs share the same sparse K rows. Pair the
    // two head slabs in launch order for prompt L2 reuse, retaining the
    // four-wave resource budget of each CTA.
    const bool pair_heads = PHASE == 1 && ND == 32 && H == 128;
    const int linear      = blockIdx.y * gridDim.x + blockIdx.x;
    // Keep this query's column slices adjacent in the launch grid.
    // Each output still has one owner and preserves its original key/MMAC order.
    // Long D512/D576 traversals reuse P/dS across these column slices.
    constexpr int DQ_ORDER_GROUP = 1;
    constexpr int DQ_NSLICES     = (ND + NDS - 1) / NDS;
    // Pairing the two column-slice roles of one query in launch order lets the
    // second role read the same P/dS from L2.  That reuse does not depend on
    // the row length: the short CSA rows (topk 128/160) pay for the same
    // 2x P/dS read, so keep the pairing for them too.
    const bool dq_order =
        PHASE == 4 && (ND == 32 || ND == 36) && (H == 64 || H == 128) && topk >= 128 && T % DQ_ORDER_GROUP == 0;
    const int dq_linear = (int)blockIdx.z * gridDim.x + (int)blockIdx.x;
    const int dq_group  = dq_linear / (DQ_ORDER_GROUP * DQ_NSLICES);
    const int dq_within = dq_linear % (DQ_ORDER_GROUP * DQ_NSLICES);
    const int t         = dq_order ? dq_group * DQ_ORDER_GROUP + dq_within % DQ_ORDER_GROUP
                                   : (pair_heads ? (linear >> 1) : (int)blockIdx.x);
    const int hb        = (pair_heads ? (linear & 1) : (int)blockIdx.y) * (PHASE == 4 ? DQ_WAVES * 16 : SAF_HBLK);
    if (hb >= H) return;
    const int lane = threadIdx.x & 63;
    // Wave ownership is scalar; avoid keeping duplicate address state in VGPRs.
    const int warp = ((ND == 32 || ND == 36) && (PHASE == 1 || PHASE == 4))
                         ? __builtin_amdgcn_readfirstlane(threadIdx.x >> 6)
                         : threadIdx.x >> 6;
    const int r0   = warp * 16;
    const int r    = lane & 15;
    const int g    = lane >> 4;
    // In the split phase-2 schedule z=0 owns dKV atomics and z=1..NSL
    // each owns one dQ slice.  The phase-1 and legacy phase-2 launches use
    // z=0 for the single combined role.
    // PHASE=4 is the dQ-only consumer used alongside cached dKV.
    const int role =
        dq_order ? dq_within / DQ_ORDER_GROUP + 1 : (SPLIT_ROLES ? (int)blockIdx.z + (PHASE == 4 ? 1 : 0) : 0);
    // Make the dQ-only role a compile-time fact: z + 1 alone does not
    // let the compiler eliminate the unreachable dKV MMAC/atomic body.
    const bool dkv_role = false;
    const bool dq_role  = PHASE == 4 || !SPLIT_ROLES || role > 0;
    // Stats B fragments read four BF16 values per lane. An eight-byte
    // row skew separates their LDS banks while preserving 16-byte global
    // transfers (the direct-to-LDS row base may be eight-byte aligned).
    // Other phases keep their original row/slice layouts.
    constexpr int CH = 16;
    const int KLD1   = PHASE == 1 ? D + 4 : D + 24;
    const int KLD2   = SLICE + 8;

    extern __shared__ char smem_raw[];
    uint32_t* pdl_ws = unused_pdl_ws;
    using PDVector   = uint32_t __attribute__((ext_vector_type(32)));
    using PDTail     = uint32_t __attribute__((ext_vector_type(8)));
    PDVector pd_regs = {};
    PDTail pd_tail   = {};
    static_assert(PHASE == 1 && FUSE_DKV && (FIXED_H == 64 || FIXED_H == 128) && (TOPK == 128 || TOPK == 160));
    // One aliased LDS region serves pass-1 K rows, pass-2 P/dl tiles and the
    // pass-2 K slice at non-overlapping times. Top-k indices are read from L2.
    hbf16* k_lds   = (hbf16*)smem_raw;                               // pass-1 K rows / aliased P/dl
    hbf16* pt2_lds = (hbf16*)(smem_raw + 4096);                      // pass-2 dl^T tiles (byte 6144..10240)
    hbf16* k2_lds  = (hbf16*)(smem_raw + (PHASE == 4 ? 0 : 10240));  // pass-2 K slice rows (after dl tiles)
    // In split phase-2 blocks the dQ roles do not execute the dKV staging
    // path, so the dead pass-1/P-dl region can hold a second K tile.  This
    // gives dQ a true two-buffer key pipeline without increasing the dynamic
    // LDS allocation: [K0][K1][dl] is packed inside the existing 21 KB slab.
    constexpr bool DQ_PIPE_LAYOUT = SPLIT_ROLES && (PHASE == 2);
    hbf16* dq_k0                  = DQ_PIPE_LAYOUT ? (hbf16*)smem_raw : k2_lds;
    hbf16* dq_k1                  = DQ_PIPE_LAYOUT ? dq_k0 + (size_t)16 * KLD2 : k2_lds;
    hbf16* dq_dl                  = DQ_PIPE_LAYOUT ? dq_k1 + (size_t)16 * KLD2 : pt2_lds + 4 * 256;

    int n_valid_t = topk;
    if (topk_length != NULL) {
        int tl = topk_length[t];
        if (tl < n_valid_t) n_valid_t = tl > 0 ? tl : 0;
    }

    const hbf16* qrow = (const hbf16*)(q + (size_t)t * H * D);
    // GLM phase 1 accepts compact 512-wide O/dO or legacy padded D-wide
    // values via D_rt. Q/K width and every MMAC contraction remain D.
    const int value_stride        = PHASE == 1 && ND == 36 ? D_rt : D;
    const hbf16* orow             = (const hbf16*)(out + (size_t)t * H * value_stride);
    const hbf16* drow             = (const hbf16*)(dout + (size_t)t * H * value_stride);
    const int* irow               = idxs + (size_t)t * topk;
    constexpr bool CACHE_DQ_INDEX = PHASE == 4 && ND == 32 && NDS == 16 && DQ_WAVES == 8;
    int* dq_idx_lds               = (int*)(smem_raw + 8448);
    if constexpr (CACHE_DQ_INDEX) {
        if (topk <= SAF_DQ_INDEX_MAX_TOPK) {
            // The four-wave path is faster without this extra LDS traffic.
            // Initialize physical tail indices too: the final K prefetch
            // reads them before applying the logical-length mask.
            for (int i = threadIdx.x; i < topk; i += THREADS) dq_idx_lds[i] = irow[i];
            __syncthreads();
        }
    }
    // Short, constant-stride stats traversals benefit from removing the
    // dependent global index load ahead of each K gather. Keep the complete
    // row bounded: larger rows increase LDS pressure and regress training.
    constexpr bool CACHE_STATS_INDEX = PHASE == 1 && ND == 32 && TOPK > 0 && TOPK <= 1024;
    int* stats_idx_lds               = (int*)(smem_raw + 19456);
    if constexpr (CACHE_STATS_INDEX) {
        for (int i = threadIdx.x; i < topk; i += THREADS) stats_idx_lds[i] = irow[i];
        __syncthreads();
    }
    auto load_index = [&](int i) -> int {
        if constexpr (CACHE_STATS_INDEX) return stats_idx_lds[i];
        if constexpr (CACHE_DQ_INDEX) {
            if (topk <= SAF_DQ_INDEX_MAX_TOPK) return dq_idx_lds[i];
        }
        return irow[i];
    };
    // Every backward consumer uses this same logical tile bound (including
    // cached dKV). Wholly masked suffix tiles need no workspace writes,
    // gathers or barriers. ck_for retains the physical stride and initializes
    // the final partial tile, so no consumer reads uninitialized P/dS there.
    const int ntiles = (n_valid_t + CH - 1) / CH;

    // One 16-bit mask per logical key tile replaces repeated global index
    // ballots, including rows whose physical sparse list contains holes.
    constexpr bool CACHE_TILE_MASK = ND == 32 && (PHASE == 1 || PHASE == 4) && TOPK > 0 && TOPK <= 2048;
    __shared__ uint16_t tile_masks[CACHE_TILE_MASK ? ((TOPK + 63) / 64) * 4 : 1];
    __shared__ int keys_all_valid;
    {
        // This flag covers the logical prefix. Compact rows can still contain
        // only valid indices; the final tile retains its explicit length mask.
        if (threadIdx.x == 0) keys_all_valid = ((PHASE == 1 || PHASE == 4) && ND == 32) || (n_valid_t == topk);
        __syncthreads();
        for (int i = threadIdx.x; i < n_valid_t; i += THREADS) {
            const int idx = load_index(i);
            if constexpr (CACHE_TILE_MASK) {
                const unsigned long long bits = __ballot(idx >= 0 && idx < T_kv);
                if ((lane & 15) == 0) tile_masks[i / 16] = (uint16_t)(bits >> ((lane / 16) * 16));
                const bool bad = __any(idx < 0 || idx >= T_kv);
                if (lane == 0 && bad) atomicExch(&keys_all_valid, 0);
            } else if (idx < 0 || idx >= T_kv)
                atomicExch(&keys_all_valid, 0);
        }
        __syncthreads();
    }

    // The CTA validity reduction is immutable after the preceding barrier.
    // Keep its result in a wave-uniform register across all key tiles.
    const bool all_keys_valid = __builtin_amdgcn_readfirstlane(keys_all_valid) != 0;
    const int h               = hb + r0 + r;
    const bool lane_ok        = (h < H);
    const int hq              = hb + r0;

    const int nsub = (H - hb + 15) >> 4;
    const int ns   = (nsub < 4) ? nsub : 4;
    auto ck_for    = [&](int kkt) {
        int c = topk - kkt * CH;
        return (c > CH) ? CH : (c < 0 ? 0 : c);
    };

    constexpr int PREF1      = 2 * ((16 * (D / 8) + THREADS - 1) / THREADS);
    constexpr int PREF2      = 2 * ((16 * (SLICE / 8) + THREADS - 1) / THREADS);
    const uint64_t dq_kaddr  = (uint64_t)kv;
    const saf_intx4 dq_kdesc = {
        (int)(uint32_t)dq_kaddr,
        (int)((dq_kaddr >> 32) |
              ((PHASE == 4 && (ND == 32 || ND == 36) && NDS == 16 && DQ_WAVES == 4 && DQ_KBUFFER) ? 0u : 0x40000000u)),
        (int)0x80000000u, 0x00020000};
    constexpr bool DQ_INDEX_AHEAD = PHASE == 4 && (ND == 32 || ND == 36) && NDS == 16 && DQ_WAVES == 4 && DQ_KBUFFER;
    int dq_future_index[2];
    bool dq_future_ready          = false;
    const uint64_t dq_index_addr  = (uint64_t)irow;
    const saf_intx4 dq_index_desc = {(int)(uint32_t)dq_index_addr, (int)(dq_index_addr >> 32), TOPK * 4, 0x00020000};
    auto prefetch_keys            = [&](int kt, int ck, int d8n, int stride, int col0, uint64_t* regs) -> int {
        if constexpr (PHASE == 1) {
            // All waves inspect the same 16 keys. An empty stats tile has
            // no LDS consumer: its P/dS outputs are explicitly zeroed below.
            // Return no payload, so commit_keys also skips its LDS writes;
            // retain both loop barriers and the next nonempty tile's gather.
            if (!all_keys_valid) {
                const int ti  = kt * CH + r;
                const int key = ti < n_valid_t ? load_index(ti) : -1;
                if (!__builtin_amdgcn_readfirstlane(__any(key >= 0 && key < T_kv))) return 0;
            }
        }

        if constexpr (PHASE == 4) {
            // Fixed slots permit register scalarization and unrolling.  Every
            // slot is initialized, including masked rows in the last tile.
            constexpr int count = (16 * (SLICE / 8) + THREADS - 1) / THREADS;
            // Resolve the complete index batch before loading K payloads.
            // Four-wave dQ needs two keys per lane; interleaving their index
            // and payload loads serializes the second gather on gfx93x.
            int gather_idx[count];
#    pragma unroll
            for (int j = 0; j < count; j++) {
                const int n = (threadIdx.x + j * THREADS) / d8n;
                if constexpr (DQ_INDEX_AHEAD) {
                    gather_idx[j] = dq_future_ready ? dq_future_index[j] : (n < ck ? load_index(kt * CH + n) : 0);
                } else
                    gather_idx[j] = n < ck ? load_index(kt * CH + n) : 0;
            }
            if constexpr (count == 2) asm volatile("" ::"v"(gather_idx[0]), "v"(gather_idx[1]) : "memory");
#    pragma unroll
            for (int j = 0; j < count; j++) {
                const int u = threadIdx.x + j * THREADS;
                const int n = u / d8n, d8 = u % d8n;
                const int key = kt * CH + n;
                const int idx = gather_idx[j];
                const bool valid =
                    n < ck && (key < n_valid_t && idx >= 0 && idx < T_kv && (ND % NDS == 0 || col0 + d8 * 8 < D));
                if constexpr (DQ_KBUFFER) {
                    // Invalid lanes use the descriptor's out-of-range zero fill,
                    // avoiding divergent masked-load PHIs and their early waits.
                    const unsigned off = valid ? ((unsigned)idx * D + col0 + d8 * 8) * 2u : 0x80000000u;
                    const saf_intx4 v  = saf_raw_buffer_load_i32x4(dq_kdesc, off, 0, 0);
                    regs[2 * j]        = ((const uint64_t*)&v)[0];
                    regs[2 * j + 1]    = ((const uint64_t*)&v)[1];
                } else {
                    const uint64_t* src = (const uint64_t*)(kv + (size_t)(valid ? idx : 0) * D + col0 + d8 * 8);
                    regs[2 * j]         = valid ? src[0] : 0;
                    regs[2 * j + 1]     = valid ? src[1] : 0;
                }
            }
            return count;
        }
        int cnt = 0;
        if (all_keys_valid && n_valid_t == topk) {
            for (int u = threadIdx.x; u < ck * d8n; u += THREADS) {
                int n               = u / d8n;
                int d8              = u - n * d8n;
                const int idx       = load_index(kt * CH + n);
                const uint64_t* src = (const uint64_t*)(kv + (size_t)idx * D + col0 + d8 * 8);
                regs[2 * cnt]       = src[0];
                regs[2 * cnt + 1]   = src[1];
                cnt++;
            }
            return cnt;
        }
        for (int u = threadIdx.x; u < ck * d8n; u += THREADS) {
            int n               = u / d8n;
            int d8              = u - n * d8n;
            const int key       = kt * CH + n;
            int idx             = load_index(key);
            const bool valid    = key < n_valid_t && idx >= 0 && idx < T_kv;
            const int safe_idx  = valid ? idx : 0;
            const uint64_t* src = (const uint64_t*)(kv + (size_t)safe_idx * D + col0 + d8 * 8);
            const uint64_t x0 = src[0], x1 = src[1];
            regs[2 * cnt]     = valid ? x0 : 0;
            regs[2 * cnt + 1] = valid ? x1 : 0;
            cnt++;
        }
        return cnt;
    };
    auto commit_keys = [&](int ck, int d8n, int stride, int nrows, hbf16* base, const uint64_t* regs, int cnt) {
#    pragma unroll
        for (int j = 0; j < (PHASE == 4 && (ND == 32 || ND == 36) ? (16 * (SLICE / 8) + THREADS - 1) / THREADS : cnt);
             j++) {
            int u  = threadIdx.x + j * THREADS;
            int n  = u / d8n;
            int d8 = u - n * d8n;
            // Matrix-read dQ uses contiguous [D/32][16 keys][32 columns] tiles.
            // Preserve aligned 16-byte writes; other phases retain row-major K.
            uint64_t* dst = (uint64_t*)(base + (PHASE == 4 ? (size_t)(d8 / 4) * 512 + n * 32 + (d8 % 4) * 8
                                                           : (size_t)n * stride + d8 * 8));
            dst[0]        = regs[2 * j];
            dst[1]        = regs[2 * j + 1];
        }
        // PHASE4 prefetch initializes all 16 rows, including masked/tail slots.
        if constexpr (PHASE == 4) return;
        if (ck < nrows) {
            for (int u = threadIdx.x; u < nrows * stride; u += THREADS) {
                if (u / stride >= ck) base[u] = hbf16(0);
            }
        }
    };
    uint64_t dq_load_masks[PREF2 / 2];
    auto prefetch_dq_keys = [&](int kt, int ck, int d8n, int stride, int col0, uint64_t* regs) -> int {
        {
            // Fixed slots permit register scalarization and unrolling.  Every
            // slot is initialized, including masked rows in the last tile.
            constexpr int count = (16 * (SLICE / 8) + THREADS - 1) / THREADS;
            // Resolve the complete index batch before loading K payloads.
            // Four-wave dQ needs two keys per lane; interleaving their index
            // and payload loads serializes the second gather on gfx93x.
            int gather_idx[count];
#    pragma unroll
            for (int j = 0; j < count; j++) {
                const int n = (threadIdx.x + j * THREADS) / d8n;
                if constexpr (DQ_INDEX_AHEAD) {
                    gather_idx[j] = dq_future_ready ? dq_future_index[j] : (n < ck ? load_index(kt * CH + n) : 0);
                } else
                    gather_idx[j] = n < ck ? load_index(kt * CH + n) : 0;
            }
            if constexpr (count == 2) asm volatile("" ::"v"(gather_idx[0]), "v"(gather_idx[1]) : "memory");
#    pragma unroll
            for (int j = 0; j < count; j++) {
                const int u = threadIdx.x + j * THREADS;
                const int n = u / d8n, d8 = u % d8n;
                const int key = kt * CH + n;
                const int idx = gather_idx[j];
                const bool valid =
                    n < ck && (key < n_valid_t && idx >= 0 && idx < T_kv && (ND % NDS == 0 || col0 + d8 * 8 < D));
                if constexpr (DQ_KBUFFER) {
                    // Invalid lanes use the descriptor's out-of-range zero fill,
                    // avoiding divergent masked-load PHIs and their early waits.
                    const unsigned off = valid ? ((unsigned)idx * D + col0 + d8 * 8) * 2u : 0x80000000u;
                    const saf_intx4 v  = saf_raw_buffer_load_i32x4(dq_kdesc, off, 0, 0);
                    regs[2 * j]        = ((const uint64_t*)&v)[0];
                    regs[2 * j + 1]    = ((const uint64_t*)&v)[1];
                } else {
                    const uint64_t* src = (const uint64_t*)(kv + (size_t)(valid ? idx : 0) * D + col0 + d8 * 8);
                    // Keep the payload load independent of masked-value selection.
                    // The opaque integer mask prevents conversion back to a divergent
                    // load/PHI with an early VMEM drain. Empty KV remains load-free.
                    dq_load_masks[j] = valid ? ~uint64_t(0) : uint64_t(0);
                    saf_u4e payload  = {0, 0, 0, 0};
                    if (T_kv > 0) payload = *(const saf_u4e*)src;
                    // Do not consume the prefetched VMEM results until LDS commit.
                    regs[2 * j]     = ((const uint64_t*)&payload)[0];
                    regs[2 * j + 1] = ((const uint64_t*)&payload)[1];
                }
            }
            return count;
        }
    };
    auto commit_dq_keys = [&](int ck, int d8n, int stride, int nrows, hbf16* base, const uint64_t* regs, int cnt) {
#    pragma unroll
        for (int j = 0; j < (true && (ND == 32 || ND == 36) ? (16 * (SLICE / 8) + THREADS - 1) / THREADS : cnt); j++) {
            int u  = threadIdx.x + j * THREADS;
            int n  = u / d8n;
            int d8 = u - n * d8n;
            // Matrix-read dQ uses contiguous [D/32][16 keys][32 columns] tiles.
            // Preserve aligned 16-byte writes; other phases retain row-major K.
            uint64_t* dst = (uint64_t*)(base + ((size_t)(d8 / 4) * 512 + n * 32 + (d8 % 4) * 8));
            uint64_t mask = dq_load_masks[j];
            // This barrier depends only on the mask, never on the payload.
            // Place masked consumption after the current tile's MMACs/barrier.
            asm volatile("" : "+v"(mask)::"memory");
            dst[0] = regs[2 * j] & mask;
            dst[1] = regs[2 * j + 1] & mask;
        }
        // PHASE4 prefetch initializes all 16 rows, including masked/tail slots.
        { return; }
    };
    if constexpr (PHASE == 0 || PHASE == 1) {
        // ---- Prologue: m/inv_l from lse/sink and avg = dO . O ----
        // The public value width is 512, with compact or padded O/dO storage.
        // Keep the RoPE columns in QK, but omit their zero dO products.
        constexpr int VALUE_ND   = PHASE == 1 && ND == 36 ? 32 : ND;
        constexpr bool REUSE_DOT = PHASE == 1 && (ND == 32 || ND == 36);
        bool avg_enabled         = false;
        float m = 0.0f, inv_l = 0.0f, P_sink = 0.0f, avg = 0.0f;
        if (lane_ok) {
            const float lse_h  = lse[(size_t)t * H + h];
            const float sink_h = attn_sink[h];
            saf_backward_normalization(lse_h, sink_h, m, inv_l, P_sink);
            {
                avg_enabled = true;
                if constexpr (!REUSE_DOT) {
                    const hbf16* orp = orow + (size_t)h * value_stride;
                    const hbf16* dr  = drow + (size_t)h * value_stride;
                    float p          = 0.0f;
#    pragma unroll
                    for (int kk = 0; kk < VALUE_ND; kk++) {
                        const int c  = kk * 16 + (g << 2);
                        p           += (float)orp[c + 0] * (float)dr[c + 0];
                        p           += (float)orp[c + 1] * (float)dr[c + 1];
                        p           += (float)orp[c + 2] * (float)dr[c + 2];
                        p           += (float)orp[c + 3] * (float)dr[c + 3];
                    }
                    avg = p;
                }
            }
        }
        if constexpr (!REUSE_DOT) {
            avg += __shfl_xor(avg, 16);
            avg += __shfl_xor(avg, 32);
        }

        // ---- Pass 1: S/DV -> P/dl -> dkv atomics + dl_ws ----
        constexpr int QD_CACHE = VALUE_ND;  // dO cached in registers across chunks
        constexpr int QCACHE   = ND;        // Q cached in registers across chunks
        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> qf[QCACHE], df[QD_CACHE];
        auto load_d = [&](int kk, DUFragment<matrix_a, 16, 16, 16, hbf16, row_major>& dfr) {
            const int row_h = lane_ok ? h : (H - 1);
            const hbf16* dp = (const hbf16*)(drow + (size_t)row_h * value_stride + kk * 16);
            if constexpr (ND == 32 && FIXED_H == 64) {
                const uint64_t address = (uint64_t)drow;
                const saf_u4e desc = {(unsigned)address, (unsigned)(address >> 32) | ((D * 2) << 16) | 0x40000000u, 64u,
                                      0x00020000u};
                auto packed = __builtin_amdgcn_buffer_load_dwordx2(desc, row_h, (kk * 16 + g * 4) * 2, false, false);
                *(uint64_t*)&dfr.x[0] = *(const uint64_t*)&packed;
            } else {
                *(uint64_t*)&dfr.x[0] = *(const uint64_t*)(dp + (g << 2));
            }
            if (!lane_ok) { dfr.x[0] = dfr.x[1] = dfr.x[2] = dfr.x[3] = hbf16(0); }
        };
        auto load_q = [&](int kk, DUFragment<matrix_a, 16, 16, 16, hbf16, row_major>& qfr) {
            const int row_h = lane_ok ? h : (H - 1);
            const hbf16* qp = (const hbf16*)(qrow + (size_t)row_h * D + kk * 16);
            if constexpr (ND == 32 && FIXED_H == 64) {
                // Same raw 64-bit BF16 fragment, now through the FlashMLA buffer
                // builtin (csrc/extension/utils.h:562) and strided Q descriptor.
                const uint64_t address = (uint64_t)qrow;
                const saf_u4e desc = {(unsigned)address, (unsigned)(address >> 32) | ((D * 2) << 16) | 0x40000000u, 64u,
                                      0x00020000u};
                auto packed = __builtin_amdgcn_buffer_load_dwordx2(desc, row_h, (kk * 16 + g * 4) * 2, false, false);
                *(uint64_t*)&qfr.x[0] = *(const uint64_t*)&packed;
            } else {
                *(uint64_t*)&qfr.x[0] = *(const uint64_t*)(qp + (g << 2));
            }
            if (!lane_ok) { qfr.x[0] = qfr.x[1] = qfr.x[2] = qfr.x[3] = hbf16(0); }
        };

        if constexpr (!REUSE_DOT) {
#    pragma unroll
            for (int kk = 0; kk < ND; kk++) load_q(kk, qf[kk]);
        }
        // dO has the same re-read pattern as Q (one A-fragment per 16-d tile,
        // re-read every key chunk); cache it once per block so pass-1 touches
        // dO from DRAM a single time instead of once per chunk.
        if constexpr (REUSE_DOT && ND == 32 && (TOPK == 128 || TOPK == 160)) {
// Short CSA rows expose the prologue latency most strongly.
// Issue a small group of independent dO/O loads before consuming
// them. Preserve ascending kk/element FP32 dot accumulation.
#    pragma unroll
            for (int kk = 0; kk < VALUE_ND; kk += 8) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> ofr[8];
#    pragma unroll
                for (int j = 0; j < 8; j++) {
                    load_d(kk + j, df[kk + j]);
                    if (avg_enabled) {
                        if constexpr (FIXED_H == 64) {
                            const uint64_t address = (uint64_t)orow;
                            const saf_u4e desc     = {(unsigned)address,
                                                      (unsigned)(address >> 32) | ((D * 2) << 16) | 0x40000000u, 64u,
                                                      0x00020000u};
                            auto packed = __builtin_amdgcn_buffer_load_dwordx2(desc, h, ((kk + j) * 16 + g * 4) * 2,
                                                                               false, false);
                            *(uint64_t*)&ofr[j].x[0] = *(const uint64_t*)&packed;
                        } else
                            *(uint64_t*)&ofr[j].x[0] =
                                *(const uint64_t*)(orow + (size_t)h * value_stride + (kk + j) * 16 + g * 4);
                    }
                }
                if (avg_enabled) {
#    pragma unroll
                    for (int j = 0; j < 8; j++) {
                        avg += (float)ofr[j].x[0] * (float)df[kk + j].x[0];
                        avg += (float)ofr[j].x[1] * (float)df[kk + j].x[1];
                        avg += (float)ofr[j].x[2] * (float)df[kk + j].x[2];
                        avg += (float)ofr[j].x[3] * (float)df[kk + j].x[3];
                    }
                }
                asm volatile("" ::"v"(avg) : "memory");
            }
        } else {
#    pragma unroll
            for (int kk = 0; kk < VALUE_ND; kk++) {
                load_d(kk, df[kk]);
                if constexpr (REUSE_DOT) {
                    if (avg_enabled) {
                        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> ofr;
                        *(uint64_t*)&ofr.x[0] = *(const uint64_t*)(orow + (size_t)h * value_stride + kk * 16 + g * 4);
                        // Identical ascending kk/element FP32 accumulation as dO.O.
                        avg += (float)ofr.x[0] * (float)df[kk].x[0];
                        avg += (float)ofr.x[1] * (float)df[kk].x[1];
                        avg += (float)ofr.x[2] * (float)df[kk].x[2];
                        avg += (float)ofr.x[3] * (float)df[kk].x[3];
                    }
                    asm volatile("" ::"v"(*(const uint64_t*)&df[kk].x[0]), "v"(avg) : "memory");
                }
            }
        }
        if constexpr (REUSE_DOT) {
            avg += __shfl_xor(avg, 16);
            avg += __shfl_xor(avg, 32);
            // Keep the Q loads after the dot to avoid extending their lifetimes.
            asm volatile("" ::"v"(avg) : "memory");
        }
        if constexpr (REUSE_DOT) {
#    pragma unroll
            for (int kk = 0; kk < ND; kk++) load_q(kk, qf[kk]);
        }

        DUFragment<accumulator, 16, 16, 16, float> sacc, dacc;

        // D=512 stats keep Q and dO resident. Direct global-to-LDS transfers
        // avoid the additional next-tile VGPR payload (224 -> 198 VGPR on
        // gfx936) and its explicit LDS stores. Use the same wave64 path on
        // gfx936/gfx938. D=576 uses a second, eight-lane copy for the RoPE
        // tail; the other phases retain their register prefetch.
        constexpr bool DIRECT_STATS = true;
        typedef int StatsDescriptor __attribute__((ext_vector_type(4)));
        const uint64_t kv_addr        = (uint64_t)kv;
        const StatsDescriptor kv_desc = {(int)(uint32_t)kv_addr, (int)(kv_addr >> 32), (int)0x80000000u, 0x00020000};
        auto gather_stats             = [&](int kt) {
            if constexpr (DIRECT_STATS) {
                // Entirely masked tiles have no matrix consumer. Preserve the
                // loop barriers and explicit zero P/dS stores below, but omit K.
                if (!all_keys_valid) {
                    if constexpr (CACHE_TILE_MASK) {
                        if (tile_masks[kt] == 0) return;
                    } else {
                        const int ki  = kt * CH + r;
                        const int key = ki < n_valid_t ? load_index(ki) : -1;
                        if (!__builtin_amdgcn_readfirstlane(__any(key >= 0 && key < T_kv))) return;
                    }
                }
                // One wave owns a key row; its 64 lanes transfer 16 bytes each.
                // Zero invalid rows so mixed-validity tiles cannot propagate NaN
                // from a masked key through the matrix products.
                // Load every index before issuing global-to-LDS transfers. A
                // later index load would otherwise wait for an earlier K transfer.
                int keys[4];
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int ki = kt * CH + warp + j * 4;
                    keys[j]      = ki < n_valid_t ? load_index(ki) : -1;
                }
                asm volatile("" ::"v"(keys[0]), "v"(keys[1]), "v"(keys[2]), "v"(keys[3]) : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n      = warp + j * 4;
                    const int ki     = kt * CH + n;
                    const int key    = keys[j];
                    const bool valid = ki < n_valid_t && (all_keys_valid || (key >= 0 && key < T_kv));
#    pragma unroll
                    for (int dc = 0; dc < D; dc += 512) {
                        if (dc + lane * 8 < D) {
                            if (valid) {
                                __builtin_amdgcn_raw_buffer_load_lds(
                                    kv_desc, (__attribute__((address_space(3))) int*)(k_lds + n * KLD1 + dc), 16,
                                    key * (D * 2) + dc * 2 + lane * 16, 0, 0, 0);
                            } else {
                                uint64_t* dst = (uint64_t*)(k_lds + n * KLD1 + dc + lane * 8);
                                dst[0] = dst[1] = 0;
                            }
                        }
                    }
                }
            }
        };
        if constexpr (DIRECT_STATS) {
            gather_stats(0);
            // All direct transfers must complete before another wave reads K.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        } else {
            uint64_t pref[PREF1];
            int cnt = prefetch_keys(0, ck_for(0), D / 8, KLD1, 0, pref);
            commit_keys(ck_for(0), D / 8, KLD1, CH, k_lds, pref, cnt);
        }
        __syncthreads();
#    pragma unroll 1
        for (int kt = 0; kt < ntiles; kt++) {
            uint32_t pd_current[4] = {};
            uint64_t pref[PREF1];
            int cnt       = 0;
            const int nkt = kt + 1;
            if constexpr (!DIRECT_STATS) {
                if (nkt < ntiles) cnt = prefetch_keys(nkt, ck_for(nkt), D / 8, KLD1, 0, pref);
            }
            const int ck = ck_for(kt);

            // The existing row-wide validity flag keeps fully populated rows
            // on the original path.  All four waves inspect the same 16 keys,
            // so an empty tile can bypass MMAC while retaining prefetch/barriers.
            const int ti       = kt * CH + r;
            const int tile_key = all_keys_valid ? 0 : (ti < n_valid_t ? load_index(ti) : -1);
            // The low 16 ballot bits describe the same key tile for every wave.
            // Reuse them for the four score columns instead of reloading each
            // index during softmax. ck also masks the final partial tile.
            const unsigned tile_mask = CACHE_TILE_MASK ? tile_masks[kt]
                                       : all_keys_valid
                                           ? ((1u << min(ck, n_valid_t - kt * CH)) - 1u)
                                           : ((unsigned)__ballot(tile_key >= 0 && tile_key < T_kv) & 0xffffu);
            const bool active        = tile_mask != 0;
            if (active) {
                du_fill_fragment(sacc, 0.0f);
                du_fill_fragment(dacc, 0.0f);
                if (hq < H) {
                    if constexpr (PHASE == 1 && (ND == 32 || ND == 36)) {
                        // Pair adjacent 16-channel BF16 fragments in the original MMAC order.
                        // After waiting for the current pair, prefetch the next pair
                        // before the current QK/dOV work. Only consume prefetched
                        // registers after the next LGKM wait; no extra LDS is needed.
                        using Read4 = unsigned __attribute__((ext_vector_type(4)));
                        Read4 packed[ND / 2];
#    pragma unroll
                        for (int z = 0; z < 2; ++z) {
                            unsigned address =
                                (unsigned)(size_t)(k_lds + (lane & 15) * KLD1 + (lane >> 4) * 4 + z * 32);
                            asm volatile("ds_read2_b64 %0, %1 offset0:0 offset1:4"
                                         : "=v"(packed[z])
                                         : "v"(address)
                                         : "memory");
                        }
#    pragma unroll
                        for (int k = 0; k < ND / 2; k += 2) {
                            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#    pragma unroll
                            for (int z = 0; z < 2; ++z) {
                                if (k + 2 + z < ND / 2) {
                                    unsigned address = (unsigned)(size_t)(k_lds + (lane & 15) * KLD1 + (lane >> 4) * 4 +
                                                                          (k + 2 + z) * 32);
                                    asm volatile("ds_read2_b64 %0, %1 offset0:0 offset1:4"
                                                 : "=v"(packed[k + 2 + z])
                                                 : "v"(address)
                                                 : "memory");
                                }
                            }
#    pragma unroll
                            for (int z = 0; z < 2; ++z) {
                                if (k + z < ND / 2) {
                                    DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr0, bfr1;
                                    ((unsigned*)&bfr0.x[0])[0] = packed[k + z][0];
                                    ((unsigned*)&bfr0.x[0])[1] = packed[k + z][1];
                                    ((unsigned*)&bfr1.x[0])[0] = packed[k + z][2];
                                    ((unsigned*)&bfr1.x[0])[1] = packed[k + z][3];
                                    du_mma_sync(sacc, qf[2 * (k + z)], bfr0, sacc);
                                    if (2 * (k + z) < VALUE_ND) du_mma_sync(dacc, df[2 * (k + z)], bfr0, dacc);
                                    du_mma_sync(sacc, qf[2 * (k + z) + 1], bfr1, sacc);
                                    if (2 * (k + z) + 1 < VALUE_ND) du_mma_sync(dacc, df[2 * (k + z) + 1], bfr1, dacc);
                                }
                            }
                        }
                    } else {
#    pragma unroll
                        for (int kk = 0; kk < ND; kk += 2) {
                            DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr0, bfr1;
                            du_load_matrix_sync(bfr0, k_lds + (size_t)kk * 16, KLD1);
                            du_load_matrix_sync(bfr1, k_lds + (size_t)(kk + 1) * 16, KLD1);
                            du_mma_sync(sacc, qf[kk], bfr0, sacc);
                            if (kk < VALUE_ND) du_mma_sync(dacc, df[kk], bfr0, dacc);
                            du_mma_sync(sacc, qf[kk + 1], bfr1, sacc);
                            if (kk + 1 < VALUE_ND) du_mma_sync(dacc, df[kk + 1], bfr1, dacc);
                        }
                    }
                }
            }
            if constexpr (DIRECT_STATS) {
                // All waves have finished reading K. Reuse the same LDS tile
                // while softmax and packed P/dS stores consume only registers.
                __syncthreads();
                if (nkt < ntiles) gather_stats(nkt);
            }
            if (active) {
                {
                    const bool ok0 = (tile_mask >> (g + 0)) & 1u;
                    const bool ok1 = (tile_mask >> (g + 4)) & 1u;
                    const bool ok2 = (tile_mask >> (g + 8)) & 1u;
                    const bool ok3 = (tile_mask >> (g + 12)) & 1u;
                    const float s0 = lane_ok ? sacc.x[0] * softmax_scale : -INFINITY;
                    const float s1 = lane_ok ? sacc.x[1] * softmax_scale : -INFINITY;
                    const float s2 = lane_ok ? sacc.x[2] * softmax_scale : -INFINITY;
                    const float s3 = lane_ok ? sacc.x[3] * softmax_scale : -INFINITY;
                    const float p0 = (ok0 && lane_ok) ? hip_dsa_numeric::softmax_exp(s0 - m) * inv_l : 0.0f;
                    const float p1 = (ok1 && lane_ok) ? hip_dsa_numeric::softmax_exp(s1 - m) * inv_l : 0.0f;
                    const float p2 = (ok2 && lane_ok) ? hip_dsa_numeric::softmax_exp(s2 - m) * inv_l : 0.0f;
                    const float p3 = (ok3 && lane_ok) ? hip_dsa_numeric::softmax_exp(s3 - m) * inv_l : 0.0f;
                    const float d0 = dacc.x[0], d1 = dacc.x[1];
                    const float d2 = dacc.x[2], d3 = dacc.x[3];
                    // CuTe SM90 rounds dS before scaling, then rounds the scaled MMA
                    // operand again. Keep FP32 P here; its BF16 copy is only for PV.
                    const float dl0 =
                        (ok0 && lane_ok) ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p0 * (d0 - avg)) << 16) *
                                               softmax_scale
                                         : 0.0f;
                    const float dl1 =
                        (ok1 && lane_ok) ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p1 * (d1 - avg)) << 16) *
                                               softmax_scale
                                         : 0.0f;
                    const float dl2 =
                        (ok2 && lane_ok) ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p2 * (d2 - avg)) << 16) *
                                               softmax_scale
                                         : 0.0f;
                    const float dl3 =
                        (ok3 && lane_ok) ? __uint_as_float((uint32_t)hip_dsa_numeric::bf16_rne(p3 * (d3 - avg)) << 16) *
                                               softmax_scale
                                         : 0.0f;

                    // dl and P are stored transposed (T, topk, H): for a fixed key
                    // the 16 lane-heads write consecutive h (coalesced), and the
                    // dkv phase below re-reads them as 8B per-lane vectors.  Keys
                    // beyond ck / n_valid_t are zeroed (their K rows are zeroed
                    // too, so they cannot leak into dkv/dq).
                    // Wait for next K before issuing this tile's independent stores.
                    // The LDS barrier below then permits those stores to overlap the
                    // next QK/dOV contraction. No next-tile reader consumes P/dS here.
                    if constexpr (DIRECT_STATS) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

                    if (lane_ok) {
                        // Streaming stores: the P/dS workspace is written once here and
                        // streamed by the dKV/dQ passes, so allocating it in L2 can
                        // evicts the KV rows this same kernel re-reads.
                        if (g + 0 < ck) {
                            const uint32_t packed = ok0 ? hip_dsa_numeric::bf16_pair(p0, dl0) : 0u;
                            pd_current[0]         = packed;
                        }
                        if (g + 4 < ck) {
                            const uint32_t packed = ok1 ? hip_dsa_numeric::bf16_pair(p1, dl1) : 0u;
                            pd_current[1]         = packed;
                        }
                        if (g + 8 < ck) {
                            const uint32_t packed = ok2 ? hip_dsa_numeric::bf16_pair(p2, dl2) : 0u;
                            pd_current[2]         = packed;
                        }
                        if (g + 12 < ck) {
                            const uint32_t packed = ok3 ? hip_dsa_numeric::bf16_pair(p3, dl3) : 0u;
                            pd_current[3]         = packed;
                        }
                    }
                }
            } else {
                if constexpr (DIRECT_STATS) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                if (lane_ok) {
// Holes inside the logical traversal still need initialized
// workspace; only whole tiles beyond n_valid_t are omitted.
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int key = kt * CH + g + e * 4;
                        pd_current[e] = 0u;
                    }
                }
            }
            if constexpr (TOPK == 128) {
                const int store_index = __builtin_amdgcn_readfirstlane(kt) * 4;
#    pragma unroll
                for (int e = 0; e < 4; e++) {
                    if constexpr (TOPK == 160) {
                        if (store_index >= 32)
                            pd_tail[store_index - 32 + e] = pd_current[e];
                        else
                            pd_regs[store_index + e] = pd_current[e];
                    } else
                        pd_regs[store_index + e] = pd_current[e];
                }
            } else {
#    pragma unroll
                for (int slot = 0; slot < TOPK / 16; slot++) {
#    pragma unroll
                    for (int e = 0; e < 4; e++) {
                        if (kt == slot) {
                            if (slot < 8)
                                pd_regs[slot * 4 + e] = pd_current[e];
                            else if constexpr (TOPK == 160)
                                pd_tail[(slot - 8) * 4 + e] = pd_current[e];
                        }
                    }
                }
            }
            if constexpr (!DIRECT_STATS) __syncthreads();
            if constexpr (DIRECT_STATS) {
                // Same barrier as sparse_attn_mla_common.h::lds_barrier: all
                // waves already waited for their global-to-LDS K transfers.
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                __syncthreads();
            } else {
                if (nkt < ntiles) commit_keys(ck_for(nkt), D / 8, KLD1, CH, k_lds, pref, cnt);
                __syncthreads();
            }
        }

        // Reuse the FP32 dO.O reduction, including the forward output rounding.

        if (lane_ok && g == 0) { atomicAdd(&dsink[h], -P_sink * avg); }

        if constexpr (FUSE_DKV) {
            static_assert(PHASE == 1 && ND == 32 && (FIXED_H == 64 || FIXED_H == 128));
            // One CTA owns all 64 heads of one query. Reuse the exact Q/dO
            // BF16 register bits. Exchange 128-column wave slices through a
            // 16-KiB LDS tile, replacing only already-consumed fragments.
            // Constant register indices avoid local-memory array addressing.
            // Four wave-permutation rounds; aligned rows enable native LDS transpose.
            constexpr int TP = 136, TS = 16 * TP;
            hbf16* tx = (hbf16*)smem_raw;
            using Ex4 = unsigned __attribute__((ext_vector_type(4)));
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 0 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[0 * 8 + j].x[0];
            }
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 1 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[1 * 8 + j].x[0];
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 2 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[2 * 8 + j].x[0];
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 3 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[3 * 8 + j].x[0];
            }
            __syncthreads();
            if (warp == 0) {
                const unsigned adr = (unsigned)(size_t)(tx + 0 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[0 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[0 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 1) {
                const unsigned adr = (unsigned)(size_t)(tx + 1 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[1 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[1 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 2) {
                const unsigned adr = (unsigned)(size_t)(tx + 2 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[2 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[2 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 3) {
                const unsigned adr = (unsigned)(size_t)(tx + 3 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[3 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[3 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 0 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[0 * 8 + j].x[0];
            }
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 1 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[1 * 8 + j].x[0];
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 2 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[2 * 8 + j].x[0];
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 3 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[3 * 8 + j].x[0];
            }
            __syncthreads();
            if (warp == 0) {
                const unsigned adr = (unsigned)(size_t)(tx + 0 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[0 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[0 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 1) {
                const unsigned adr = (unsigned)(size_t)(tx + 1 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[1 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[1 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 2) {
                const unsigned adr = (unsigned)(size_t)(tx + 2 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[2 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[2 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 3) {
                const unsigned adr = (unsigned)(size_t)(tx + 3 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[3 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[3 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 0 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[1 * 8 + j].x[0];
            }
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 1 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[0 * 8 + j].x[0];
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 2 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[3 * 8 + j].x[0];
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 3 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[2 * 8 + j].x[0];
            }
            __syncthreads();
            if (warp == 0) {
                const unsigned adr = (unsigned)(size_t)(tx + 1 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[1 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[1 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 1) {
                const unsigned adr = (unsigned)(size_t)(tx + 0 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[0 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[0 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 2) {
                const unsigned adr = (unsigned)(size_t)(tx + 3 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[3 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[3 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 3) {
                const unsigned adr = (unsigned)(size_t)(tx + 2 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[2 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[2 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 0 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[1 * 8 + j].x[0];
            }
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 1 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[0 * 8 + j].x[0];
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 2 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[3 * 8 + j].x[0];
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 3 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[2 * 8 + j].x[0];
            }
            __syncthreads();
            if (warp == 0) {
                const unsigned adr = (unsigned)(size_t)(tx + 1 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[1 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[1 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 1) {
                const unsigned adr = (unsigned)(size_t)(tx + 0 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[0 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[0 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 2) {
                const unsigned adr = (unsigned)(size_t)(tx + 3 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[3 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[3 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 3) {
                const unsigned adr = (unsigned)(size_t)(tx + 2 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[2 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[2 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 0 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[2 * 8 + j].x[0];
            }
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 1 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[3 * 8 + j].x[0];
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 2 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[0 * 8 + j].x[0];
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 3 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[1 * 8 + j].x[0];
            }
            __syncthreads();
            if (warp == 0) {
                const unsigned adr = (unsigned)(size_t)(tx + 2 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[2 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[2 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 1) {
                const unsigned adr = (unsigned)(size_t)(tx + 3 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[3 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[3 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 2) {
                const unsigned adr = (unsigned)(size_t)(tx + 0 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[0 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[0 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 3) {
                const unsigned adr = (unsigned)(size_t)(tx + 1 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[1 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[1 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 0 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[2 * 8 + j].x[0];
            }
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 1 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[3 * 8 + j].x[0];
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 2 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[0 * 8 + j].x[0];
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 3 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[1 * 8 + j].x[0];
            }
            __syncthreads();
            if (warp == 0) {
                const unsigned adr = (unsigned)(size_t)(tx + 2 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[2 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[2 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 1) {
                const unsigned adr = (unsigned)(size_t)(tx + 3 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[3 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[3 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 2) {
                const unsigned adr = (unsigned)(size_t)(tx + 0 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[0 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[0 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 3) {
                const unsigned adr = (unsigned)(size_t)(tx + 1 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[1 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[1 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 0 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[3 * 8 + j].x[0];
            }
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 1 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[2 * 8 + j].x[0];
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 2 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[1 * 8 + j].x[0];
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 3 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&qf[0 * 8 + j].x[0];
            }
            __syncthreads();
            if (warp == 0) {
                const unsigned adr = (unsigned)(size_t)(tx + 3 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[3 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[3 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 1) {
                const unsigned adr = (unsigned)(size_t)(tx + 2 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[2 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[2 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 2) {
                const unsigned adr = (unsigned)(size_t)(tx + 1 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[1 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[1 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 3) {
                const unsigned adr = (unsigned)(size_t)(tx + 0 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&qf[0 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&qf[0 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            __syncthreads();
            if (warp == 0) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 0 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[3 * 8 + j].x[0];
            }
            if (warp == 1) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 1 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[2 * 8 + j].x[0];
            }
            if (warp == 2) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 2 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[1 * 8 + j].x[0];
            }
            if (warp == 3) {
#    pragma unroll
                for (int j = 0; j < 8; j++)
                    *(uint64_t*)(tx + 3 * TS + r * TP + j * 16 + g * 4) = *(uint64_t*)&df[0 * 8 + j].x[0];
            }
            __syncthreads();
            if (warp == 0) {
                const unsigned adr = (unsigned)(size_t)(tx + 3 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[3 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[3 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 1) {
                const unsigned adr = (unsigned)(size_t)(tx + 2 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[2 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[2 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 2) {
                const unsigned adr = (unsigned)(size_t)(tx + 1 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[1 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[1 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            if (warp == 3) {
                const unsigned adr = (unsigned)(size_t)(tx + 0 * TS + (lane / 4) * TP + (lane % 4) * 8);
                Ex4 mat[4];
#    pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile("ds_read_m32x16_b16 %0, %1 offset:%2"
                                 : "=v"(mat[j])
                                 : "v"(adr), "n"(j * 64)
                                 : "memory");
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - j) : "memory");
                    *(uint64_t*)&df[0 * 8 + j * 2].x[0]     = ((uint64_t*)&mat[j])[0];
                    *(uint64_t*)&df[0 * 8 + j * 2 + 1].x[0] = ((uint64_t*)&mat[j])[1];
                }
            }
            __syncthreads();
            // Stats used non-temporal stores; make this CTA's packed workspace
            // visible before the cross-wave dKV reads. No other query is read.
            __threadfence_block();
            __syncthreads();
            constexpr int PLD = 66, PTILE = 16 * PLD;
            hbf16* pd = (hbf16*)smem_raw;
            // Stats and Q/dO transpose have finished: reuse their LDS for indices.
            // 2*16*66 BF16 values + 3072 ints = 16512 B, within the existing
            // 64*68*sizeof(float) allocation. Larger K keeps global index reads.
            const bool cache_dkv_indices = topk <= 3072;
            int* cached_indices          = (int*)(pd + 2 * PTILE);
            if (cache_dkv_indices) {
                for (int i = threadIdx.x; i < n_valid_t; i += 256) cached_indices[i] = idxs[(size_t)t * topk + i];
                __syncthreads();
            }
            // With eight column tiles per wave, dKV has the same MMAC work
            // as the existing H128 cached prefetch path. Retain next P/dS in
            // registers while consuming the current LDS tile.
            const int u = threadIdx.x, row = u / 16, head = (u % 16) * 4;
            int carried_key = r < n_valid_t ? (cache_dkv_indices ? cached_indices[r] : idxs[(size_t)t * topk + r]) : -1;
#    pragma unroll 1
            for (int kt = 0; kt < n_valid_t; kt += 16) {
                const int key = carried_key;
                carried_key   = kt + 16 + r < n_valid_t ? (cache_dkv_indices ? cached_indices[kt + 16 + r]
                                                                             : idxs[(size_t)t * topk + kt + 16 + r])
                                                        : -1;
                if (!__builtin_amdgcn_readfirstlane(__any(key >= 0 && key < T_kv))) { continue; }
                uint32_t pd_current[4] = {};
                const int pd_index     = __builtin_amdgcn_readfirstlane(kt / 16) * 4;
#    pragma unroll
                for (int e = 0; e < 4; e++) {
                    if constexpr (TOPK == 160) {
                        if (pd_index >= 32)
                            pd_current[e] = pd_tail[pd_index - 32 + e];
                        else
                            pd_current[e] = pd_regs[pd_index + e];
                    } else
                        pd_current[e] = pd_regs[pd_index + e];
                }
#    pragma unroll
                for (int e = 0; e < 4; e++) {
                    ((uint16_t*)pd)[(g + e * 4) * PLD + (h - hb)]         = (uint16_t)pd_current[e];
                    ((uint16_t*)pd)[PTILE + (g + e * 4) * PLD + (h - hb)] = (uint16_t)(pd_current[e] >> 16);
                }
                __syncthreads();
                saf_floatx4 atomic_values[8];
                constexpr int KEY_PACK = 8;
                const uint64_t element_offset =
                    KEY_PACK == 8
                        ? (((uint64_t)(uint32_t)key / 8 * 32 + warp * 8) * 128 + g * 32 + ((uint32_t)key % 8) * 4)
                        : ((uint64_t)(uint32_t)key * 512 + warp * 128 + 4 * g);
                uint64_t atomic_address    = (uint64_t)dkv_float + element_offset * sizeof(float);
                const uint64_t atomic_mask = __ballot(key >= 0 && key < T_kv);
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[4], da[4];
#    pragma unroll
                for (int hg = 0; hg < 4; hg++) {
                    du_load_matrix_sync(pa[hg], pd + hg * 16, PLD);
                    du_load_matrix_sync(da[hg], pd + PTILE + hg * 16, PLD);
                }
#    pragma unroll
                for (int j = 0; j < 8; j++) {
                    DUFragment<accumulator, 16, 16, 16, float> acc;
                    du_fill_fragment(acc, 0.f);
#    pragma unroll
                    for (int hg = 0; hg < 4; hg++) {
                        DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> qb, db;
#    pragma unroll
                        for (int e = 0; e < 4; e++) {
                            qb.x[e] = qf[hg * 8 + j].x[e];
                            db.x[e] = df[hg * 8 + j].x[e];
                        }
                        du_mma_sync(acc, pa[hg], db, acc);
                        du_mma_sync(acc, da[hg], qb, acc);
                    }
                    atomic_values[j] = {acc.x[0], acc.x[1], acc.x[2], acc.x[3]};
                    // Preserve real atomic return/value registers until completion.
                    // Mask only the atomic, keeping C++ control flow uniform.
                    uint64_t saved_exec;
                    asm volatile("s_and_saveexec_b64 %1, %3\n\t"
                             "global_atomic_add_f32_x4 %0, %2, %0, off offset:%4 glc\n\t"
                             "s_mov_b64 exec, %1"
                    : "+v"(atomic_values[j]), "=&s"(saved_exec), "+v"(atomic_address)
                    : "s"(atomic_mask), "n"(j*64*KEY_PACK) : "exec", "memory");
                }
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#    pragma unroll
                for (int j = 0; j < 8; j++) asm volatile("" ::"v"(atomic_values[j]), "v"(atomic_address) : "memory");
                __syncthreads();
            }
        }
    }  // stats phase

    // Separate phase launches are ordered by their HIP stream. Only the
    // one-code-object fallback needs an intra-kernel workspace hand-off.
    if constexpr (PHASE == 0) {
        __threadfence();
        __syncthreads();
    }

    {
        // ---- dkv: P/dl tiles for G chunks are staged from this block's own
        // p_ws / dl_ws scratch (written by pass 1 above) into LDS, then the
        // dO/Q B-fragments are register-cached per (d-tile group, head
        // sub-block) and reused across the G chunks (the old layout re-read
        // dO/Q from global once per chunk).  Atomics stay per key-chunk, so
        // nothing else changes; dO/Q global loads drop ~Gx. ----
        if (dkv_role) {
            constexpr int G = GDKV;  // P/dl chunk pairs resident in LDS
            // A 66-element row keeps the cooperative uint32 staging aligned and
            // fits five P/dl chunk pairs in a 21120-byte aliased LDS region.
            constexpr int PLDM    = 66;
            constexpr int PTILE   = 16 * PLDM;  // bf16 per P or dl tile
            const int dkv_hb      = hb;
            const int dkv_hgroups = 4;
            const int dkv_h4      = 32;
            // Partial 64-head slabs may have fewer than four active waves.  Split
            // the D tiles across those waves so every column is covered; using a
            // fixed divisor of four dropped the upper D tiles for H < 49.
            const int dkv_ns = ns;
            const int t0     = (warp * ND) / dkv_ns;
            const int t1     = ((warp + 1) * ND) / dkv_ns;
            // Stage G chunks of P/dl (16 keys x 64 heads, padded ldm) into the
            // pass-1 K region, which is dead after pass 1.  Tiles beyond the
            // chunk count / ck rows are zeroed (their K rows are zeroed too).
            saf_intx4 pdl_desc;
            // Rebase the descriptor per query.  A Pro workspace already exceeds
            // 2 GiB at T=4096 and 4 GiB at T=8192; a whole-workspace voffset
            // would hit the descriptor bound or wrap.  soffset is a byte offset,
            // not the high 32 bits of a 64-bit address.
            const uint64_t pdl_addr = (uint64_t)(pdl_ws);
            pdl_desc[0]             = __builtin_amdgcn_readfirstlane((int)(uint32_t)(pdl_addr & 0xffffffffu));
            pdl_desc[1]             = __builtin_amdgcn_readfirstlane((int)((uint32_t)(pdl_addr >> 32) | 0x40000000u));
            pdl_desc[2]             = __builtin_amdgcn_readfirstlane((int)0x80000000u);
            pdl_desc[3]             = __builtin_amdgcn_readfirstlane((int)0x00020000u);
            for (int ktG = 0; ktG < ntiles; ktG += G) {
                hbf16* pd = k_lds;
                // A full slab has an even, contiguous head range.  Load four
                // packed P/dl words (four heads) with one 16B buffer instruction,
                // unpack them into two P and two dl dwords, and write both LDS
                // planes.  The old scalar path remains for partial slabs.
                const bool full_slab = H >= dkv_hb + 2 * dkv_h4;
                if (full_slab) {
                    const int hp = (dkv_h4 + 1) >> 1;
                    for (int u = threadIdx.x; u < G * 16 * hp; u += THREADS) {
                        const int c    = u / (16 * hp);
                        const int rem  = u - c * (16 * hp);
                        const int key  = rem / hp;
                        const int slot = (rem - key * hp) << 1;
                        const int kk   = ktG + c;
                        uint32_t* dstp = (uint32_t*)(pd + c * (2 * PTILE) + key * PLDM + slot * 2);
                        uint32_t* dstd = (uint32_t*)(pd + c * (2 * PTILE) + PTILE + key * PLDM + slot * 2);
                        if (kk < ntiles && key < ck_for(kk)) {
                            const uint64_t boff = (((size_t)kk * CH + key) * H + dkv_hb + slot * 2) * sizeof(uint32_t);
                            const int voff      = (int)(uint32_t)boff;
                            const saf_intx4 x   = saf_raw_buffer_load_i32x4(pdl_desc, voff, 0, 3);
                            const uint32_t p0   = (uint32_t)x[0];
                            const uint32_t p1   = (uint32_t)x[1];
                            const uint32_t p2   = (uint32_t)x[2];
                            const uint32_t p3   = (uint32_t)x[3];
                            dstp[0]             = (p0 & 0xffffu) | ((p1 & 0xffffu) << 16);
                            dstd[0]             = (p0 >> 16) | ((p1 >> 16) << 16);
                            if (slot + 1 < dkv_h4) {
                                dstp[1] = (p2 & 0xffffu) | ((p3 & 0xffffu) << 16);
                                dstd[1] = (p2 >> 16) | ((p3 >> 16) << 16);
                            }
                        } else {
                            dstp[0] = dstd[0] = 0;
                            if (slot + 1 < dkv_h4) dstp[1] = dstd[1] = 0;
                        }
                    }
                } else {
                    // 4B staging stores cover 2 bf16 heads each; the tile keeps
                    // 64 heads per key (local slab head = hb + head4*2).
                    for (int u = threadIdx.x; u < G * 2 * 16 * dkv_h4; u += THREADS) {
                        const int c   = u / (2 * 16 * dkv_h4);
                        const int rem = u - c * (2 * 16 * dkv_h4);
                        const int arr = rem / (16 * dkv_h4);
                        const int key = (rem % (16 * dkv_h4)) / dkv_h4;
                        const int h4  = rem % dkv_h4;
                        const int kk  = ktG + c;
                        uint32_t* dst = (uint32_t*)(pd + c * (2 * PTILE) + arr * PTILE + key * PLDM + h4 * 2);
                        if (kk < ntiles && key < ck_for(kk)) {
                            const int hs        = dkv_hb + h4 * 2;
                            const uint32_t* src = pdl_ws + (kk * CH + key) * H + hs;
                            if (hs + 1 < H) {
                                const uint64_t x     = saf_lds_load_u64((const void*)src);
                                const uint32_t ppack = (x & 0xFFFFu) | (((x >> 32) & 0xFFFFu) << 16);
                                const uint32_t dpack = ((x >> 16) & 0xFFFFu) | (((x >> 48) & 0xFFFFu) << 16);
                                *dst                 = (arr == 0) ? ppack : dpack;
                            } else if (hs < H) {
                                const uint32_t x = saf_lds_load_u32((const void*)src);
                                *dst             = (arr == 0) ? (x & 0xFFFFu) : (x >> 16);
                            } else {
                                *dst = 0;
                            }
                        } else {
                            *dst = 0;
                        }
                    }
                }
                __syncthreads();  // P/dl tiles visible to all warps
                if (warp < ns) {
                    for (int tg = t0; tg < t1; tg += 4) {
                        const int t_hi = (tg + 4 < t1) ? tg + 4 : t1;
                        DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bb[8][4], qb[8][4];
#    pragma unroll
                        for (int s = 0; s < dkv_hgroups; s++) {
                            const int h0s = dkv_hb + s * 16;
                            if (h0s >= H) break;
#    pragma unroll
                            for (int j = 0; j < 4; j++) {
                                const int tt = tg + j;
                                if (tt >= t_hi) break;
                                const hbf16* bp  = drow + (size_t)h0s * value_stride + (tt * 16 + r);
                                const hbf16* qp2 = qrow + (size_t)h0s * D + (tt * 16 + r);
                                const int fr     = h0s + (g << 2);
                                bb[s][j].x[0]    = (fr < H) ? bp[(size_t)(g << 2) * D] : hbf16(0);
                                bb[s][j].x[1]    = (fr + 1 < H) ? bp[((size_t)(g << 2) + 1) * D] : hbf16(0);
                                bb[s][j].x[2]    = (fr + 2 < H) ? bp[((size_t)(g << 2) + 2) * D] : hbf16(0);
                                bb[s][j].x[3]    = (fr + 3 < H) ? bp[((size_t)(g << 2) + 3) * D] : hbf16(0);
                                qb[s][j].x[0]    = (fr < H) ? qp2[(size_t)(g << 2) * D] : hbf16(0);
                                qb[s][j].x[1]    = (fr + 1 < H) ? qp2[((size_t)(g << 2) + 1) * D] : hbf16(0);
                                qb[s][j].x[2]    = (fr + 2 < H) ? qp2[((size_t)(g << 2) + 2) * D] : hbf16(0);
                                qb[s][j].x[3]    = (fr + 3 < H) ? qp2[((size_t)(g << 2) + 3) * D] : hbf16(0);
                            }
                        }
#    pragma unroll
                        for (int c = 0; c < G; c++) {
                            const int kt = ktG + c;
                            if (kt >= ntiles) break;
                            const int ck = ck_for(kt);
                            DUFragment<accumulator, 16, 16, 16, float> kacc[4];
#    pragma unroll
                            for (int j = 0; j < 4; j++) du_fill_fragment(kacc[j], 0.0f);
#    pragma unroll
                            for (int s = 0; s < dkv_hgroups; s++) {
                                const int h0s = dkv_hb + s * 16;
                                if (h0s >= H) break;
                                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa, da;
                                du_load_matrix_sync(pa, pd + c * (2 * PTILE) + s * 16, PLDM);
                                du_load_matrix_sync(da, pd + c * (2 * PTILE) + PTILE + s * 16, PLDM);
#    pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    if (tg + j < t_hi) {
                                        du_mma_sync(kacc[j], pa, bb[s][j], kacc[j]);
                                        du_mma_sync(kacc[j], da, qb[s][j], kacc[j]);
                                    }
                                }
                            }
                            const int kidx = (r < ck) ? load_index(kt * CH + r) : -1;
                            if (r < ck && kidx >= 0 && kidx < T_kv) {
#    pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    if (tg + j < t_hi) {
                                        float* pk = dkv_float + (size_t)kidx * D + (tg + j) * 16;
#    if defined(__gfx936__) || defined(__gfx938__)
                                        // MMAC lanes g=0..3 own columns
                                        // {g,g+4,g+8,g+12}.  The fused2 private
                                        // workspace stores exactly this interleaved
                                        // order, so one native x4 atomic can commit
                                        // the fragment without a lane permutation;
                                        // the final cast restores contiguous D.
                                        saf_atomic_add_mmac_row_swizzled(pk, g, kacc[j].x[0], kacc[j].x[1],
                                                                         kacc[j].x[2], kacc[j].x[3]);
#    else
                                        pk += g;
                                        atomicAdd(&pk[0], kacc[j].x[0]);
                                        atomicAdd(&pk[4], kacc[j].x[1]);
                                        atomicAdd(&pk[8], kacc[j].x[2]);
                                        atomicAdd(&pk[12], kacc[j].x[3]);
#    endif
                                    }
                                }
                            }
                        }
                    }
                }
                __syncthreads();  // P/dl tiles consumed; next group may overwrite
            }
        }  // dkv_role
        // After dKV, its Q/dO staging and cached indices are dead. Reuse the
        // first 16 KiB for two 8-KiB K slices. The immutable stats indices at
        // byte 19456 remain separate. Each wave transfers one 16-key x 32-col
        // tile: lane/4 selects key, lane%4 selects an aligned 16-byte vector.
        constexpr bool SHORT_DQ_DIRECT = FIXED_H == 64;
        const uint64_t direct_addr     = (uint64_t)kv;
        const saf_intx4 direct_desc    = {(int)(uint32_t)direct_addr, (int)(direct_addr >> 32), (int)0x80000000u,
                                          0x00020000};
        auto gather_short_dq           = [&](int kt, int col0, hbf16* destination) {
            const int ki     = kt * 16 + lane / 4;
            const int key    = ki < n_valid_t ? load_index(ki) : -1;
            const bool valid = ki < n_valid_t && key >= 0 && key < T_kv;
#    pragma unroll
            for (int j = 0; j < 2; j++) {
                const int tile = warp + j * 4;
                hbf16* dst     = destination + tile * 512;
                // Descriptor range ends at 2 GiB; the invalid offset is outside
                // that range. Validate zero-fill for holes, tails and empty rows.
                const unsigned offset =
                    valid ? (unsigned)key * (D * 2) + (col0 + tile * 32) * 2 + (lane % 4) * 16 : 0x80000000u;
                __builtin_amdgcn_raw_buffer_load_lds(direct_desc, (__attribute__((address_space(3))) int*)dst, 16,
                                                     offset, 0, 0, 0);
            }
        };
        // ---- Pass 2: dq, D-sliced (ND/NDS slice passes over the chunk loop) ----
        constexpr int NSL = (ND + NDS - 1) / NDS;
        if (dq_role) {
#    pragma unroll
            for (int slot = 0; slot < TOPK / 16; slot++) {
                uint32_t v[4], x[4], y[4];
#    pragma unroll
                for (int e = 0; e < 4; e++) {
                    if (slot < 8)
                        v[e] = pd_regs[slot * 4 + e];
                    else if constexpr (TOPK == 160)
                        v[e] = pd_tail[(slot - 8) * 4 + e];
                }
#    pragma unroll
                for (int e = 0; e < 4; e++) {
                    const uint32_t peer = __shfl_xor(v[e ^ 1], 16);
                    x[e]                = ((g ^ e) & 1) ? peer : v[e];
                }
#    pragma unroll
                for (int e = 0; e < 4; e++) {
                    const uint32_t peer = __shfl_xor(x[e ^ 2], 32);
                    y[e]                = ((g ^ e) & 2) ? peer : x[e];
                }
#    pragma unroll
                for (int e = 0; e < 4; e++) {
                    if (slot < 8)
                        pd_regs[slot * 4 + e] = y[e];
                    else if constexpr (TOPK == 160)
                        pd_tail[(slot - 8) * 4 + e] = y[e];
                }
            }

            // Relocate the descriptor for every query using a 64-bit base. Only
            // the bounded offset within that query uses buffer addressing; the
            // full workspace can cross 2/4 GiB. Keep flat loads for a single
            // query whose workspace itself exceeds the descriptor's range.
            // DQ_BUFFER is selected by the HIP launcher, so the common loop
            // carries neither a range branch nor a live fallback address.
            const uint32_t* dq_pdl_query = pdl_ws;
            const uint64_t dq_pdl_addr   = (uint64_t)dq_pdl_query;
            const saf_intx4 dq_pdl_desc  = {(int)(uint32_t)dq_pdl_addr,
                                            (int)((dq_pdl_addr >> 32) | (DQ_INDEX_AHEAD ? 0u : 0x40000000u)),
                                            (int)0x80000000u, 0x00020000};
            // Split-role blocks execute exactly one slice.  The legacy block
            // executes all slices serially, preserving the old A/B path.
            const int sl_begin = SPLIT_ROLES ? (role - 1) : 0;
            const int sl_end   = SPLIT_ROLES ? role : NSL;
#    pragma unroll 1
            for (int sl = sl_begin; sl < sl_end; sl++) {
                const int D0 = sl * SLICE;
                if constexpr (DQ_INDEX_AHEAD) dq_future_ready = false;
                DUFragment<accumulator, 16, 16, 16, float> dqacc[NDS];
#    pragma unroll
                for (int d16 = 0; d16 < NDS; d16++) du_fill_fragment(dqacc[d16], 0.0f);

                if constexpr (SHORT_DQ_DIRECT) {
                    gather_short_dq(0, D0, (hbf16*)smem_raw);
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                } else {
                    uint64_t pref[PREF2];
                    int cnt = prefetch_dq_keys(0, ck_for(0), SLICE / 8, KLD2, D0, pref);
                    commit_dq_keys(ck_for(0), SLICE / 8, KLD2, 16, k2_lds, pref, cnt);
                }
                __syncthreads();
#    pragma unroll 1
                for (int kt = 0; kt < ntiles; kt++) {
                    uint64_t pref[PREF2];
                    int cnt       = 0;
                    const int nkt = kt + 1;
                    if constexpr (DQ_INDEX_AHEAD) {
                        const int next_ck = ck_for(nkt);
#    pragma unroll
                        for (int j = 0; j < 2; j++) {
                            const int n = (threadIdx.x + j * THREADS) / (SLICE / 8);
                            if constexpr (TOPK > 0 && TOPK <= 1152) {
                                // Constant physical row bound makes out-of-range
                                // future entries return zero without an EXEC branch.
                                dq_future_index[j] = saf_raw_buffer_load_i32(dq_index_desc, (nkt * CH + n) * 4, 0, 0);
                            } else
                                dq_future_index[j] = n < next_ck ? load_index(nkt * CH + n) : 0;
                        }
                        dq_future_ready = true;
                    }
                    unsigned direct_active = 0;
                    if constexpr (SHORT_DQ_DIRECT) {
                        // Read the immutable mask before issuing next-buffer LDS
                        // writes. Otherwise LLVM drains VMEM ahead of this LDS read.
                        direct_active =
                            __builtin_amdgcn_readfirstlane((unsigned)(all_keys_valid || tile_masks[kt] != 0));
                        asm volatile("" : "+s"(direct_active)::"memory");
                        k2_lds = (hbf16*)(smem_raw + (kt & 1) * 8192);
                        if (nkt < ntiles) gather_short_dq(nkt, D0, (hbf16*)(smem_raw + (nkt & 1) * 8192));
                    } else if constexpr (!DQ_KBUFFER) {
                        if (nkt < ntiles) cnt = prefetch_dq_keys(nkt, ck_for(nkt), SLICE / 8, KLD2, D0, pref);
                    }
                    const int ck = ck_for(kt);

                    // dl for this chunk from the block's own scratch; phantom
                    // heads read 0 (their dl slots were never written).  For
                    // CH < 16 chunks the tile rows beyond ck are zeroed (their
                    // K rows are zeroed too, so they cannot leak into dq).
                    const int ti       = kt * CH + r;
                    const int tile_key = all_keys_valid ? 0 : (ti < n_valid_t ? load_index(ti) : -1);
                    const bool active =
                        SHORT_DQ_DIRECT ? direct_active != 0
                                        : (CACHE_TILE_MASK ? tile_masks[kt] != 0
                                                           : all_keys_valid || __builtin_amdgcn_readfirstlane(__any(
                                                                                   tile_key >= 0 && tile_key < T_kv)));
                    if (active && hq < H) {
                        DUFragment<matrix_a, 16, 16, 16, hbf16, col_major> dlf;
                        uint32_t pd_current[4] = {};
                        const int pd_index     = __builtin_amdgcn_readfirstlane(kt) * 4;
#    pragma unroll
                        for (int e = 0; e < 4; e++) {
                            if constexpr (TOPK == 160) {
                                if (pd_index >= 32)
                                    pd_current[e] = pd_tail[pd_index - 32 + e];
                                else
                                    pd_current[e] = pd_regs[pd_index + e];
                            } else
                                pd_current[e] = pd_regs[pd_index + e];
                        }
#    pragma unroll
                        for (int e = 0; e < 4; e++) *(uint16_t*)&dlf.x[e] = (uint16_t)(pd_current[e] >> 16);
                        {
                            // Each matrix read returns even/odd column fragments for
                            // a 16-key x 32-column tile. Issue four independent reads
                            // before waiting, then consume the packed BF16 operands.
                            // Both gfx936/gfx938 emit this source-backed DS opcode.
                            typedef unsigned int mat4 __attribute__((ext_vector_type(4)));
#    pragma unroll
                            for (int d16 = 0; d16 < NDS; d16 += 8) {
                                // D576's final 256-column slab contains only 64
                                // real columns. Keep the existing grid/pipeline,
                                // but omit LDS reads and MMACs for its zero tail.
                                if constexpr (ND % NDS != 0) {
                                    if (D0 + d16 * 16 >= D) break;
                                }
                                mat4 packed[4];
#    pragma unroll
                                for (int z = 0; z < 4; z++) {
                                    if constexpr (ND % NDS != 0) {
                                        if (D0 + (d16 + 2 * z) * 16 >= D) break;
                                    }
                                    const unsigned addr = (unsigned)(size_t)(k2_lds + (d16 / 2 + z) * 512 +
                                                                             (lane / 4) * 32 + (lane % 4) * 8);
                                    asm volatile("ds_read_m32x16_b16_alt %0, %1"
                                                 : "=v"(packed[z])
                                                 : "v"(addr)
                                                 : "memory");
                                }
                                // Drain one fragment at a time instead of waiting
                                // for the whole group. `lgkmcnt(3-z)` releases the
                                // z-th read (0 = oldest) as soon as it lands, so the
                                // MMAC pair for z runs while reads z+1..3 are still
                                // in flight. Reads, MMAC count, order and operands
                                // are unchanged; only the wait granularity is finer.
                                // The short D576 tail keeps the original full drain.
                                if constexpr (ND % NDS != 0) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#    pragma unroll
                                for (int z = 0; z < 4; z++) {
                                    if constexpr (ND % NDS != 0) {
                                        if (D0 + (d16 + 2 * z) * 16 >= D) break;
                                    } else {
                                        asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(3 - z) : "memory");
                                    }
                                    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                                    ((uint32_t*)&b0.x[0])[0] = packed[z][0];
                                    ((uint32_t*)&b0.x[0])[1] = packed[z][1];
                                    ((uint32_t*)&b1.x[0])[0] = packed[z][2];
                                    ((uint32_t*)&b1.x[0])[1] = packed[z][3];
                                    du_mma_sync(dqacc[d16 + 2 * z], dlf, b0, dqacc[d16 + 2 * z]);
                                    du_mma_sync(dqacc[d16 + 2 * z + 1], dlf, b1, dqacc[d16 + 2 * z + 1]);
                                }
                            }
                        }
                        if constexpr ((NDS & 1) != 0) {
                            DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b;
                            du_load_matrix_sync(b, k2_lds + (size_t)(NDS - 1) * 16, KLD2);
                            du_mma_sync(dqacc[NDS - 1], dlf, b, dqacc[NDS - 1]);
                        }
                    }
                    if constexpr (DQ_KBUFFER) {
                        if (!(active && hq < H)) cnt = prefetch_dq_keys(nkt, ck_for(nkt), SLICE / 8, KLD2, D0, pref);
                    }
                    if constexpr (SHORT_DQ_DIRECT) {
                        // Finish next-buffer transfers and current-buffer reads
                        // before the CTA swaps their roles. One CTA barrier serves
                        // both handoffs; no pending VGPR payload is involved.
                        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                        __syncthreads();
                    } else {
                        __syncthreads();
                        if (nkt < ntiles) commit_dq_keys(ck_for(nkt), SLICE / 8, KLD2, 16, k2_lds, pref, cnt);
                        __syncthreads();
                    }
                }

                // dq epilogue: this block's head slab, slice columns only.
                if (lane_ok) {
                    bf16* dqbase = dq + ((size_t)t * H + h) * D + D0;
                    {
#    pragma unroll
                        for (int d16 = 0; d16 < NDS; d16 += 2) {
                            const int dcol = (d16 / 2) * 32 + g * 8;
                            if constexpr (ND % NDS != 0) {
                                if (D0 + dcol >= D) continue;
                            }
                            // Transpose already-rounded pairs within each head,
                            // preserving NaN canonicalization and every output bit.
                            uint32_t v[4], x[4], y[4];
#    pragma unroll
                            for (int z = 0; z < 4; ++z) {
                                const float a = dqacc[d16].x[z], b = dqacc[d16 + 1].x[z];
                                const uint32_t lo = (a != a) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(a);
                                const uint32_t hi = (b != b) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(b);
                                v[z]              = lo | (hi << 16);
                            }
#    pragma unroll
                            for (int z = 0; z < 4; z++) {
                                uint32_t peer = __shfl_xor(v[z ^ 1], 16);
                                x[z]          = ((g ^ z) & 1) ? peer : v[z];
                            }
#    pragma unroll
                            for (int z = 0; z < 4; z++) {
                                uint32_t peer = __shfl_xor(x[z ^ 2], 32);
                                y[z]          = ((g ^ z) & 2) ? peer : x[z];
                            }
                            __builtin_nontemporal_store((saf_u4e){y[0], y[1], y[2], y[3]}, (saf_u4e*)(dqbase + dcol));
                        }
                    }
                }
            }
        }  // dq_role
        // The last gradient block casts dkv_float after every block has
        // committed its dkv/dsink atomics.
        if (dkv_role) {
            __threadfence();
            __syncthreads();
            __shared__ int fused_last;
            if (threadIdx.x == 0) fused_last = atomicAdd(done_count, 1);
            __syncthreads();
            // Split-role dQ blocks do not participate in done_count; the dKV
            // role count remains grid.x * grid.y.
            const int dkv_blocks = (int)(gridDim.x * gridDim.y);
            saf_multi_cast_swizzled(done_count, dkv_float, dkv, T_kv, D, dkv_blocks, fused_last);
        }
        __threadfence();
        __syncthreads();
        __shared__ int fused_previous;
        if (threadIdx.x == 0) fused_previous = atomicAdd(done_count, 1);
        __syncthreads();
        { saf_multi_cast_keypacked(done_count, dkv_float, dkv, T_kv, 512, T * (FIXED_H / 64), fused_previous); }
    }  // PHASE != 1
}

// Pair queries only when their current index tiles match exactly. Mismatched
// tiles retain two independent contractions and scatters; duplicates are legal.
__global__ static void __launch_bounds__(256, 1)
    sparse_dkv_glm_pair(const bf16* q, const bf16* dout, const uint32_t* pdl, const int* idx, const int* lengths,
                        float* dkvf, bf16* dkv, int* done, int T, int K, int S, int V) {
#    if defined(__gfx936__) || defined(__gfx938__)
    constexpr int H = 32, D = 576, NC = 9, PLD = 36, PT = 16 * PLD;
    const int qb = blockIdx.x * 2, lane = threadIdx.x & 63, r = lane & 15, g = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), dc0 = wave * NC;
    const int n = min(2, T - qb);
    int valid[2];
#        pragma unroll
    for (int qi = 0; qi < 2; qi++) valid[qi] = qi < n ? (lengths ? max(0, min(K, lengths[qb + qi])) : K) : 0;
    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> qc[2][2][NC], dc[2][2][NC];
#        pragma unroll
    for (int qi = 0; qi < 2; qi++) {
#        pragma unroll
        for (int hg = 0; hg < 2; hg++) {
#        pragma unroll
            for (int j = 0; j < NC; j++) {
#        pragma unroll
                for (int e = 0; e < 4; e++) {
                    int h = hg * 16 + g * 4 + e, col = (dc0 + j) * 16 + r;
                    qc[qi][hg][j].x[e] = qi < n ? ((const hbf16*)q)[((size_t)(qb + qi) * H + h) * D + col] : hbf16(0);
                    dc[qi][hg][j].x[e] =
                        (qi < n && col < V) ? ((const hbf16*)dout)[((size_t)(qb + qi) * H + h) * V + col] : hbf16(0);
                }
                asm volatile("" ::"v"(*(const uint64_t*)&qc[qi][hg][j].x[0]), "v"(*(const uint64_t*)&dc[qi][hg][j].x[0])
                             : "memory");
            }
        }
    }
    __shared__ hbf16 pd[4 * PT];
    for (int kt = 0; kt < max(valid[0], valid[1]); kt += 16) {
        int key[2];
#        pragma unroll
        for (int qi = 0; qi < 2; qi++) {
            key[qi] = kt + r < valid[qi] ? idx[(size_t)(qb + qi) * K + kt + r] : -1;
            if (key[qi] < 0 || key[qi] >= S) key[qi] = -1;
        }
        bool merge = __builtin_amdgcn_readfirstlane(__all(key[0] == key[1]));
        bool any0  = __builtin_amdgcn_readfirstlane(__any(key[0] >= 0));
        bool any1  = __builtin_amdgcn_readfirstlane(__any(key[1] >= 0));
        if (!any0 && !any1) continue;
#        pragma unroll
        for (int qi = 0; qi < 2; qi++) {
            if (threadIdx.x < 128) {
                const int row = threadIdx.x / 8, h = (threadIdx.x % 8) * 4;
                int ix    = kt + row < valid[qi] ? idx[(size_t)(qb + qi) * K + kt + row] : -1;
                saf_u4e x = {0, 0, 0, 0};
                if (ix >= 0 && ix < S) x = saf_nt_load_u4(pdl + ((size_t)(qb + qi) * K + kt + row) * H + h);
                uint32_t* p = (uint32_t*)(pd + qi * 2 * PT + row * PLD + h);
                uint32_t* d = (uint32_t*)(pd + qi * 2 * PT + PT + row * PLD + h);
                p[0]        = (x.x & 65535) | (x.y << 16);
                p[1]        = (x.z & 65535) | (x.w << 16);
                d[0]        = (x.x >> 16) | (x.y & 0xffff0000);
                d[1]        = (x.z >> 16) | (x.w & 0xffff0000);
            }
        }
        __syncthreads();
        DUFragment<accumulator, 16, 16, 16, float> acc[NC];
#        pragma unroll
        for (int j = 0; j < NC; j++) du_fill_fragment(acc[j], 0.f);
#        pragma unroll
        for (int qi = 0; qi < 2; qi++) {
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[2], da[2];
#        pragma unroll
            for (int hg = 0; hg < 2; hg++) {
                du_load_matrix_sync(pa[hg], pd + qi * 2 * PT + hg * 16, PLD);
                du_load_matrix_sync(da[hg], pd + qi * 2 * PT + PT + hg * 16, PLD);
            }
#        pragma unroll
            for (int j = 0; j < NC; j++) {
#        pragma unroll
                for (int hg = 0; hg < 2; hg++) {
                    du_mma_sync(acc[j], pa[hg], dc[qi][hg][j], acc[j]);
                    du_mma_sync(acc[j], da[hg], qc[qi][hg][j], acc[j]);
                }
            }
            if (!merge || qi == 1) {
                const uint64_t mask = __ballot(key[qi] >= 0);
                uint64_t address    = (uint64_t)dkvf + ((uint64_t)(uint32_t)key[qi] * D + dc0 * 16 + g * 4) * 4;
                saf_floatx4 values[NC];
#        pragma unroll
                for (int j = 0; j < NC; j++) {
                    values[j] = {acc[j].x[0], acc[j].x[1], acc[j].x[2], acc[j].x[3]};
                    uint64_t saved;
                    asm volatile(
                        "s_and_saveexec_b64 %1, %3\n\tglobal_atomic_add_f32_x4 %2, %0, off offset:%4\n\ts_mov_b64 "
                        "exec, %1"
                        : "+v"(values[j]), "=&s"(saved), "+v"(address)
                        : "s"(mask), "n"(j * 64)
                        : "exec", "memory");
                }
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#        pragma unroll
                for (int j = 0; j < NC; j++) {
                    asm volatile("" ::"v"(values[j]), "v"(address) : "memory");
                    du_fill_fragment(acc[j], 0.f);
                }
            }
        }
        __syncthreads();
    }
    __threadfence();
    __syncthreads();
    __shared__ int previous;
    if (threadIdx.x == 0) previous = atomicAdd(done, 1);
    __syncthreads();
    saf_multi_cast_swizzled(done, dkvf, dkv, S, D, (T + 1) / 2, previous);
#    endif
}

template<bool PADDED_DISPATCH = false, int HC = 64>
__global__ static void stage130_build_inverse(const int* idx, const int* lens, int* inverse, int* guard, int T) {
    if constexpr (PADDED_DISPATCH)
        if ((guard[2] == 0) != (HC == 32)) return;
    const int t = blockIdx.x, n = lens ? max(0, min(2048, lens[t])) : 2048;
    for (int p = threadIdx.x; p < n; p += blockDim.x) {
        const int key = idx[(size_t)t * 2048 + p];
        if (key >= 0 && key < 4096 && atomicCAS(inverse + (size_t)t * 4096 + key, 0, p + 1) != 0)
            atomicOr(guard, 1 << 28);
    }
}
__global__ static void sparse_dkv_cast_no_gridwait_kernel(const float* dkv_float, bf16* dkv, int T_kv, int D) {
    const float4* src4 = (const float4*)dkv_float;
    uint64_t* dst      = (uint64_t*)dkv;
    for (long tile  = (long)blockIdx.x * blockDim.x + threadIdx.x; tile < (long)T_kv * (D >> 4);
         tile      += (long)gridDim.x * blockDim.x) {
        const long base4   = tile * 4;
        const float4 a0    = src4[base4 + 0];
        const float4 a1    = src4[base4 + 1];
        const float4 a2    = src4[base4 + 2];
        const float4 a3    = src4[base4 + 3];
        const uint64_t b00 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.x);
        const uint64_t b01 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.x);
        const uint64_t b02 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.x);
        const uint64_t b03 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.x);
        const uint64_t b10 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.y);
        const uint64_t b11 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.y);
        const uint64_t b12 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.y);
        const uint64_t b13 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.y);
        const uint64_t b20 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.z);
        const uint64_t b21 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.z);
        const uint64_t b22 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.z);
        const uint64_t b23 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.z);
        const uint64_t b30 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a0.w);
        const uint64_t b31 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a1.w);
        const uint64_t b32 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a2.w);
        const uint64_t b33 = (uint64_t)(uint16_t)hip_dsa_numeric::bf16_rne(a3.w);
        const long out8    = tile * 4;
        dst[out8 + 0]      = b00 | (b01 << 16) | (b02 << 32) | (b03 << 48);
        dst[out8 + 1]      = b10 | (b11 << 16) | (b12 << 32) | (b13 << 48);
        dst[out8 + 2]      = b20 | (b21 << 16) | (b22 << 32) | (b23 << 48);
        dst[out8 + 3]      = b30 | (b31 << 16) | (b32 << 32) | (b33 << 48);
    }
}
template<int NT, int KB, int QG = 16, bool PADDED_DISPATCH = false, int HC = 64>
__global__ __launch_bounds__(NT, 1) static void sparse_dkv_key_shared(const uint32_t* pd, const int* inv,
                                                                      const int* guard, float* dkvf, int T) {
#    if defined(__gfx936__) || defined(__gfx938__)
    if constexpr (PADDED_DISPATCH)
        if ((guard[2] == 0) != (HC == 32)) return;
    if (guard[0] & (1 << 28)) return;
    const int lane = threadIdx.x & 63, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), r = lane & 15,
              g      = lane >> 4;
    constexpr int NW = NT / 64, BASE = 36 / NW, EXTRA = 36 % NW, MAXNC = BASE + (EXTRA > 0), KR = KB * 16, PLD = HC + 4,
                  PS  = KR * PLD;
    const int nc      = wave < EXTRA ? BASE + 1 : BASE,
              column  = wave < EXTRA ? wave * (BASE + 1) : EXTRA * (BASE + 1) + (wave - EXTRA) * BASE;
    const int keybase = blockIdx.x * KR, qbegin = blockIdx.y * QG;
    // Runtime inverse only: if every query/key in this CTA is absent, dKV is exactly zero.
    // This uniform early return precedes all accumulator initialization and loop PHIs.
    __shared__ int group_active;
    if (threadIdx.x == 0) group_active = 0;
    __syncthreads();
    unsigned found = 0;
    for (int i = threadIdx.x; i < min(QG, T - qbegin) * KR; i += NT)
        found |= (keybase + i % KR < 4096 && inv[(size_t)(qbegin + i / KR) * 4096 + keybase + i % KR] != 0);
    const bool wave_active = __any(found);
    if (lane == 0 && wave_active) atomicOr(&group_active, 1);
    __syncthreads();
    if (!group_active) return;

    extern __shared__ char raw[];
    hbf16* pl                = (hbf16*)raw;
    hbf16* dl                = pl + PS;
    const saf_u4e* packed_qd = (const saf_u4e*)(pd + (size_t)T * 2048 * 64 + (size_t)T * 4096);
    DUFragment<accumulator, 16, 16, 16, float> acc[KB][MAXNC];
#        pragma unroll
    for (int k = 0; k < KB; k++) {
#        pragma unroll
        for (int j = 0; j < MAXNC; j++) du_fill_fragment(acc[k][j], 0.f);
    }
#        pragma unroll 1
    for (int t = qbegin; t < min(T, qbegin + QG); t++) {
        // Each quarter-wave owns one key row. Direct inverse reads trade
        // repeated cache-line accesses for the per-query inverse LDS barrier.
        constexpr int PD_LOADS = (KR * (HC / 4) + NT - 1) / NT;
        int positions[PD_LOADS];
#        pragma unroll
        for (int j = 0; j < PD_LOADS; j++) {
            const int i = threadIdx.x + j * NT, row = i / (HC / 4);
            positions[j] = (i < KR * (HC / 4) && keybase + row < 4096) ? inv[(size_t)t * 4096 + keybase + row] : 0;
        }
        __builtin_amdgcn_sched_barrier(0);
        saf_u4e staged_pd[PD_LOADS];
#        pragma unroll
        for (int j = 0; j < PD_LOADS; j++) {
            const int i = threadIdx.x + j * NT, row = i / (HC / 4), head = (i % (HC / 4)) * 4;
            staged_pd[j] = {0, 0, 0, 0};
            if (i < KR * (HC / 4)) {
                const int pos = positions[j];
                if (pos) staged_pd[j] = *(const saf_u4e*)(pd + ((size_t)t * 2048 + pos - 1) * HC + head);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
#        pragma unroll
        for (int j = 0; j < PD_LOADS; j++) {
            const int i = threadIdx.x + j * NT, row = i / (HC / 4), head = (i % (HC / 4)) * 4;
            if (i < KR * (HC / 4)) {
                saf_u4e v    = staged_pd[j];
                uint32_t* pp = (uint32_t*)(pl + row * PLD + head);
                uint32_t* dd = (uint32_t*)(dl + row * PLD + head);
                pp[0]        = (v.x & 65535) | (v.y << 16);
                pp[1]        = (v.z & 65535) | (v.w << 16);
                dd[0]        = (v.x >> 16) | (v.y & 0xffff0000);
                dd[1]        = (v.z >> 16) | (v.w & 0xffff0000);
            }
        }
        __syncthreads();
#        pragma unroll 1
        for (int hg = 0; hg < HC / 16; hg++) {
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[KB], da[KB];
#        pragma unroll
            for (int k = 0; k < KB; k++) {
                du_load_matrix_sync(pa[k], pl + k * 16 * PLD + hg * 16, PLD);
                du_load_matrix_sync(da[k], dl + k * 16 * PLD + hg * 16, PLD);
            }
            saf_u4e operands[MAXNC];
#        pragma unroll
            for (int j = 0; j < MAXNC; j++) {
                operands[j] = {0, 0, 0, 0};
                if (j < nc) operands[j] = packed_qd[(((size_t)t * 36 + column + j) * 4 + hg) * 64 + lane];
            }
            __builtin_amdgcn_sched_barrier(0);
#        pragma unroll
            for (int j = 0; j < MAXNC; j++)
                if (j < nc) {
                    saf_u4e bits = operands[j];
                    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> qb, db;
                    ((uint32_t*)&qb.x[0])[0] = bits[0];
                    ((uint32_t*)&qb.x[0])[1] = bits[1];
                    ((uint32_t*)&db.x[0])[0] = bits[2];
                    ((uint32_t*)&db.x[0])[1] = bits[3];
#        pragma unroll
                    for (int k = 0; k < KB; k++) {
                        du_mma_sync(acc[k][j], pa[k], db, acc[k][j]);
                        du_mma_sync(acc[k][j], da[k], qb, acc[k][j]);
                    }
                }
        }
        __syncthreads();  // All P/dS readers finish before the next query rewrites LDS.
    }
#        pragma unroll
    for (int k = 0; k < KB; k++)
        if (keybase + k * 16 + r < 4096) {
            uint64_t addr       = (uint64_t)(dkvf + (size_t)(keybase + k * 16 + r) * 576 + column * 16 + g * 4);
            saf_floatx4 vals[3] = {
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}
            };
#        pragma unroll
            for (int j = 0; j < MAXNC; j++)
                if (j < nc) {
                    vals[j % 3] = {acc[k][j].x[0], acc[k][j].x[1], acc[k][j].x[2], acc[k][j].x[3]};
                    asm volatile("global_atomic_add_f32_x4 %0, %1, %0, off offset:%2 glc"
                                 : "+v"(vals[j % 3]), "+v"(addr)
                                 : "n"(j * 64)
                                 : "memory");
                    if ((j + 1) % 3 == 0 || j + 1 == nc) {
                        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#        pragma unroll
                        for (int z = 0; z < 3; z++) asm volatile("" ::"v"(vals[z]), "v"(addr) : "memory");
                    }
                }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#        pragma unroll
            for (int z = 0; z < 3; z++) asm volatile("" ::"v"(vals[z]), "v"(addr) : "memory");
        }
    // Stream-ordered output conversion follows; no cross-CTA polling.
#    endif
}
// One query, HC heads, 64*NC output columns per block.
// D=512 uses NC=4. D=576 defaults to NC=3; verified local H64 input
// uses NC=9 so one block retains all 576 columns and reads P/dS once.
// Retain Q/dO across the complete sparse traversal; for H=128 reduce
// both head slabs before the FP32 atomic commit. Public output stays BF16.
// Each wave caches Q/dO for its 16*NC columns across the complete key loop.
template<int HC = 128, int NC = 4, int D = 512, bool PREFETCH = true, bool IDX_LDS = true, bool GUARDED = false,
         bool DUP_GUARD = false, int HEAD_MODE = 0>
__global__ static void __launch_bounds__(256, 1)
    sparse_dkv_cached_kernel(const bf16* q, const bf16* dout, const uint32_t* pdl, const int* idx, const int* lengths,
                             float* dkvf, bf16* dkv, int* done, int T, int H, int K, int Tkv,
                             const int* dispatch_guard = nullptr, int value_stride = D) {
    if constexpr (HEAD_MODE != 0)
        if ((done[3] == 0) != (HEAD_MODE == 32)) return;
    if constexpr (DUP_GUARD)
        if (!(dispatch_guard[0] & (1 << 28))) return;
    if constexpr (GUARDED) {
        const bool fuse = (dispatch_guard[0] & 1) == 0 && (int64_t)dispatch_guard[1] * 4 >= (int64_t)T * K;
        // The wide D576 variant is selected by the same whole-call locality
        // predicate that makes the guarded narrow variant return.
        if constexpr (HC == 64 && D == 576 && NC == 9) {
            if (!fuse) return;
        } else {
            if (fuse) return;
        }
    }

    constexpr int HG = HC / 16, PLDM = saf_dkv_ldm<HC, D>, TILE = 16 * PLDM;
    // Prefetch the next packed P/dS tile during the current contraction.
    constexpr bool PREFETCH_PD = PREFETCH && ((HC == 128 && D == 512) || (HC == 64 && D == 576));
    const int t = blockIdx.x, ds = blockIdx.y;
    // A compact GLM dO has 512 physical columns. Materialize its zero
    // tail in registers, retaining the original MMAC and atomic order.
    const int dvalue = D == 576 ? value_stride : D;
    const int lane   = threadIdx.x & 63;
    const int wave   = PREFETCH_PD ? __builtin_amdgcn_readfirstlane(threadIdx.x >> 6) : threadIdx.x >> 6;
    const int r = lane & 15, g = lane >> 4;
    const int dc0   = (ds * 4 + wave) * NC;
    const int valid = lengths ? max(0, min(K, lengths[t])) : K;
    extern __shared__ hbf16 sm[];
    // The top-k index row is K int32 per query.  Reading it with one global load
    // per 16-key tile puts a full-latency, L2-thrashing access (T*K*4 bytes)
    // directly on the tile ballot's critical path. Preload the row cooperatively
    // so subsequent ballots consume LDS-resident indices. IDX_LDS is
    // false when the row cannot fit in the 64 KiB LDS budget, in which case the
    // launcher allocates no index region and the per-tile global fetch is kept.
    int* saf_idx_lds = IDX_LDS ? (int*)(sm + 2 * TILE) : nullptr;
    if constexpr (IDX_LDS) {
        for (int u = threadIdx.x; u < valid; u += 256) saf_idx_lds[u] = idx[(size_t)t * K + u];
        __syncthreads();
    }
    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> qq[HG][NC], dd[HG][NC];
#    pragma unroll
    for (int s = 0; s < HG; s++) {
#    pragma unroll
        for (int j = 0; j < NC; j++) {
#    pragma unroll
            for (int e = 0; e < 4; e++) {
                const int h    = s * 16 + g * 4 + e;
                const size_t a = ((size_t)t * H + h) * D + (dc0 + j) * 16 + r;
                qq[s][j].x[e]  = ((HC != 16 || h < H) && (D % (64 * NC) == 0 || (dc0 + j) * 16 + r < D))
                                     ? ((const hbf16*)q)[a]
                                     : hbf16(0);
                if constexpr (D == 576)
                    dd[s][j].x[e] = ((HC != 16 || h < H) && (dc0 + j) * 16 + r < dvalue)
                                        ? ((const hbf16*)dout)[((size_t)t * H + h) * dvalue + (dc0 + j) * 16 + r]
                                        : hbf16(0);
                else
                    dd[s][j].x[e] = ((HC != 16 || h < H) && (D % (64 * NC) == 0 || (dc0 + j) * 16 + r < D))
                                        ? ((const hbf16*)dout)[a]
                                        : hbf16(0);
            }
            if constexpr (PREFETCH_PD) {
                // Finish packing this Q/dO pair before starting the next global loads.
                // Otherwise LLVM overlaps the whole prologue and spills 19 VGPRs in
                // the H128 prefetch path. This compiler dependency emits no instruction;
                // both gfx936/gfx938 retain the same cached fragments and MMAC order.
                asm volatile("" ::"v"(*(const uint64_t*)&qq[s][j].x[0]), "v"(*(const uint64_t*)&dd[s][j].x[0])
                             : "memory");
            }
        }
    }
    // Streaming P/dS fetches: column CTAs reread this large workspace, which
    // would otherwise evict the FP32 dKV atomic accumulator from the 8 MB L2.
    saf_u4e packed[PREFETCH_PD ? (16 * (HC / 4) + 255) / 256 : 1];
    int carried_key = -1;
    if constexpr (PREFETCH_PD) carried_key = r < valid ? (IDX_LDS ? saf_idx_lds[r] : idx[(size_t)t * K + r]) : -1;
    auto prefetch_pd = [&](int next, int next_key) {
        if constexpr (PREFETCH_PD) {
            // P/dS for empty tiles need not be initialized by the stats pass.
            // Match the traversal's CTA-uniform ballot before reading that tile.
            if (!__builtin_amdgcn_readfirstlane(__any(next_key >= 0 && next_key < Tkv))) return;
#    pragma unroll
            for (int z = 0; z < (16 * (HC / 4) + 255) / 256; z++) {
                const int u = threadIdx.x + z * 256, key = u / (HC / 4), h = (u % (HC / 4)) * 4;
                packed[z] = (u < 16 * (HC / 4) && next + key < K)
                                ? saf_nt_load_u4((const void*)(pdl + ((size_t)t * K + next + key) * H + h))
                                : (saf_u4e){0, 0, 0, 0};
            }
        }
    };
    prefetch_pd(0, carried_key);
// Match stats/dQ's logical tile bound. Stats leaves wholly masked suffix
// tiles unwritten; the last partial tile is still fully initialized.
// P/dS loads are naturally contiguous over heads; Q/dO never reload.
#    pragma unroll 1
    for (int kt = 0; kt < valid; kt += 16) {
        const int key = PREFETCH_PD
                            ? carried_key
                            : (kt + r < valid ? (IDX_LDS ? saf_idx_lds[kt + r] : idx[(size_t)t * K + kt + r]) : -1);
        if constexpr (PREFETCH_PD)
            carried_key =
                kt + 16 + r < valid ? (IDX_LDS ? saf_idx_lds[kt + 16 + r] : idx[(size_t)t * K + kt + 16 + r]) : -1;
        // r spans the same 16 selections in every wave.  This branch is CTA-
        // uniform, including empty rows and holes, and precedes all LDS barriers.
        if (!__builtin_amdgcn_readfirstlane(__any(key >= 0 && key < Tkv))) {
            // Empty tiles still advance the prefetch to preserve holes/lengths.
            prefetch_pd(kt + 16, carried_key);
            continue;
        }
        if constexpr (PREFETCH_PD) {
#    pragma unroll
            for (int z = 0; z < (16 * (HC / 4) + 255) / 256; z++) {
                const int u = threadIdx.x + z * 256;
                if (u < 16 * (HC / 4)) {
                    const int key = u / (HC / 4), h = (u % (HC / 4)) * 4;
                    uint32_t* p     = (uint32_t*)(sm + key * PLDM + h);
                    uint32_t* d     = (uint32_t*)(sm + TILE + key * PLDM + h);
                    const saf_u4e x = packed[z];
                    p[0]            = (x.x & 65535) | (x.y << 16);
                    p[1]            = (x.z & 65535) | (x.w << 16);
                    d[0]            = (x.x >> 16) | (x.y & 0xffff0000);
                    d[1]            = (x.z >> 16) | (x.w & 0xffff0000);
                }
            }
        } else {
            for (int u = threadIdx.x; u < 16 * (HC / 4); u += 256) {
                const int key = u / (HC / 4), h = (u % (HC / 4)) * 4;
                uint32_t* p = (uint32_t*)(sm + key * PLDM + h);
                uint32_t* d = (uint32_t*)(sm + TILE + key * PLDM + h);
                if (kt + key < K && (HC != 16 || h + 3 < H)) {
                    const saf_u4e x = saf_nt_load_u4((const void*)(pdl + ((size_t)t * K + kt + key) * H + h));
                    p[0]            = (x.x & 65535) | (x.y << 16);
                    p[1]            = (x.z & 65535) | (x.w << 16);
                    d[0]            = (x.x >> 16) | (x.y & 0xffff0000);
                    d[1]            = (x.z >> 16) | (x.w & 0xffff0000);
                } else {
                    p[0] = p[1] = d[0] = d[1] = 0;
                }
            }
        }
        __syncthreads();
        // Reuse registers while the current tile remains in LDS. The existing
        // end-of-tile barrier protects LDS before the next iteration overwrites it.
        prefetch_pd(kt + 16, carried_key);
#    if defined(__gfx936__) || defined(__gfx938__)
        constexpr bool COLUMN_ATOMICS = (HC == 64 && D == 576) || (HC == 128 && D == 512 && PREFETCH && IDX_LDS);
#    else
        constexpr bool COLUMN_ATOMICS = false;
#    endif
        if constexpr (COLUMN_ATOMICS && ((HC == 128 && D == 512) || (HC == 64 && D == 576 && NC == 9))) {
            saf_floatx4 atomic_values[NC];
            uint64_t atomic_address = (uint64_t)dkvf + ((uint64_t)(uint32_t)key * D + dc0 * 16 + 4 * g) * sizeof(float);
            const uint64_t atomic_mask = __ballot(key >= 0 && key < Tkv);
            // Cache each P/dS fragment once. Scatter each completed column while
            // computing the next, preserving the per-output FP32 accumulation order.
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[HG], da[HG];
#    pragma unroll
            for (int s = 0; s < HG; s++) {
                du_load_matrix_sync(pa[s], sm + s * 16, PLDM);
                du_load_matrix_sync(da[s], sm + TILE + s * 16, PLDM);
            }
#    pragma unroll
            for (int j = 0; j < NC; j++) {
                DUFragment<accumulator, 16, 16, 16, float> acc;
                du_fill_fragment(acc, 0.f);
#    pragma unroll
                for (int s = 0; s < HG; s++) {
                    du_mma_sync(acc, pa[s], dd[s][j], acc);
                    du_mma_sync(acc, da[s], qq[s][j], acc);
                }
                atomic_values[j] = {acc.x[0], acc.x[1], acc.x[2], acc.x[3]};
                // Integer address arithmetic remains defined for masked invalid keys.
                // All VMEM operands stay live until the completion wait below.
                uint64_t saved_exec;
                // Keep C++ control flow uniform: no asynchronous return-value PHI
                // copies may execute before the final wait. Only the atomic is masked;
                // LDS fragment reads and MMAC execute with the original full-wave EXEC.
                asm volatile("s_and_saveexec_b64 %1, %3\n\t"
                   "global_atomic_add_f32_x4 %0, %2, %0, off offset:%4 glc\n\t"
                   "s_mov_b64 exec, %1"
       : "+v"(atomic_values[j]), "=&s"(saved_exec), "+v"(atomic_address)
       : "s"(atomic_mask), "n"(j*64) : "exec", "memory");
            }
            // All native atomic return registers must be complete before reuse.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#    pragma unroll
            for (int j = 0; j < NC; j++) asm volatile("" ::"v"(atomic_values[j]), "v"(atomic_address) : "memory");
        } else if constexpr (COLUMN_ATOMICS) {
            saf_floatx4 atomic_returns[NC], atomic_values[NC];
            uint64_t atomic_addresses[NC];
            const uint64_t atomic_mask = __ballot(key >= 0 && key < Tkv);
            // Cache each P/dS fragment once. Scatter each completed column while
            // computing the next, preserving the per-output FP32 accumulation order.
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[HG], da[HG];
#    pragma unroll
            for (int s = 0; s < HG; s++) {
                du_load_matrix_sync(pa[s], sm + s * 16, PLDM);
                du_load_matrix_sync(da[s], sm + TILE + s * 16, PLDM);
            }
#    pragma unroll
            for (int j = 0; j < NC; j++) {
                DUFragment<accumulator, 16, 16, 16, float> acc;
                du_fill_fragment(acc, 0.f);
#    pragma unroll
                for (int s = 0; s < HG; s++) {
                    du_mma_sync(acc, pa[s], dd[s][j], acc);
                    du_mma_sync(acc, da[s], qq[s][j], acc);
                }
                atomic_values[j] = {acc.x[0], acc.x[1], acc.x[2], acc.x[3]};
                atomic_addresses[j] =
                    (uint64_t)dkvf + ((uint64_t)(uint32_t)key * D + (dc0 + j) * 16 + 4 * g) * sizeof(float);
                // Integer address arithmetic remains defined for masked invalid keys.
                // All VMEM operands stay live until the completion wait below.
                uint64_t saved_exec;
                // Keep C++ control flow uniform: no asynchronous return-value PHI
                // copies may execute before the final wait. Only the atomic is masked;
                // LDS fragment reads and MMAC execute with the original full-wave EXEC.
                asm volatile("s_and_saveexec_b64 %1, %4\n\t"
                   "global_atomic_add_f32_x4 %0, %2, %3, off glc\n\t"
                   "s_mov_b64 exec, %1"
       : "=&v"(atomic_returns[j]), "=&s"(saved_exec),
         "+v"(atomic_addresses[j]), "+v"(atomic_values[j])
       : "s"(atomic_mask) : "exec", "memory");
            }
            // All native atomic return registers must be complete before reuse.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#    pragma unroll
            for (int j = 0; j < NC; j++)
                asm volatile("" ::"v"(atomic_returns[j]), "v"(atomic_values[j]), "v"(atomic_addresses[j]) : "memory");
        } else {
            DUFragment<accumulator, 16, 16, 16, float> acc[NC];
#    pragma unroll
            for (int j = 0; j < NC; j++) du_fill_fragment(acc[j], 0.f);
#    pragma unroll
            for (int s = 0; s < HG; s++) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa, da;
                du_load_matrix_sync(pa, sm + s * 16, PLDM);
                du_load_matrix_sync(da, sm + TILE + s * 16, PLDM);
#    pragma unroll
                for (int j = 0; j < NC; j++) {
                    du_mma_sync(acc[j], pa, dd[s][j], acc[j]);
                    du_mma_sync(acc[j], da, qq[s][j], acc[j]);
                }
            }

#    if defined(__gfx936__) || defined(__gfx938__)
            // Preserve every asynchronous physical return until the full completion wait.
            // Uniform C++ flow avoids compiler-generated return-value PHI copies.
            saf_floatx4 atomic_values[NC];
            uint64_t atomic_address = (uint64_t)dkvf + ((uint64_t)(uint32_t)key * D + dc0 * 16 + 4 * g) * sizeof(float);
#        pragma unroll
            for (int j = 0; j < NC; j++) {
                atomic_values[j]    = {acc[j].x[0], acc[j].x[1], acc[j].x[2], acc[j].x[3]};
                const uint64_t mask = __ballot(key >= 0 && key < Tkv && (D % (64 * NC) == 0 || (dc0 + j) * 16 < D));
                uint64_t saved_exec;
                asm volatile("s_and_saveexec_b64 %1, %3\n\t"
                "global_atomic_add_f32_x4 %0, %2, %0, off offset:%4 glc\n\t"
                "s_mov_b64 exec, %1"
    : "+v"(atomic_values[j]), "=&s"(saved_exec), "+v"(atomic_address)
    : "s"(mask), "n"(j*64) : "exec", "memory");
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#        pragma unroll
            for (int j = 0; j < NC; j++) asm volatile("" ::"v"(atomic_values[j]), "v"(atomic_address) : "memory");
#    else
            if (key >= 0 && key < Tkv) {
#        pragma unroll
                for (int j = 0; j < NC; j++)
                    if (D % (64 * NC) == 0 || (dc0 + j) * 16 < D)
                        saf_atomic_add_mmac_row_swizzled(dkvf + (size_t)key * D + (dc0 + j) * 16, g, acc[j].x[0],
                                                         acc[j].x[1], acc[j].x[2], acc[j].x[3]);
            }
#    endif
        }
        __syncthreads();
    }
    if constexpr (HEAD_MODE != 0) return;  // final stream-ordered cast handles both batches
    __threadfence();
    __syncthreads();
    __shared__ int prev;
    if (threadIdx.x == 0) prev = atomicAdd(done, 1);
    __syncthreads();
    saf_multi_cast_swizzled(done, dkvf, dkv, Tkv, D, T * gridDim.y, prev);
}

#else
__global__ static void sparse_dkv_glm_pair(const bf16*, const bf16*, const uint32_t*, const int*, const int*, float*,
                                           bf16*, int*, int, int, int, int) {}
template<bool PADDED_DISPATCH = false, int HC = 64>
__global__ static void stage130_build_inverse(const int*, const int*, int*, int*, int) {}
template<int NT, int KB, int QG = 16, bool PADDED_DISPATCH = false, int HC = 64>
__global__ static void sparse_dkv_key_shared(const uint32_t*, const int*, const int*, float*, int) {}
__global__ static void sparse_dkv_cast_no_gridwait_kernel(const float*, bf16*, int, int) {}

template<int ND, int TOPK = 0, bool COMPACT_VALUES = false>
__global__ static void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_bwd_fused_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kv, const bf16* __restrict__ out,
                                 const bf16* __restrict__ dout, const float* __restrict__ lse,
                                 const float* __restrict__ attn_sink, const int* __restrict__ idxs,
                                 const int* __restrict__ topk_length, float* __restrict__ dkv_float,
                                 float* __restrict__ dsink, bf16* __restrict__ dq, bf16* __restrict__ dkv,
                                 int* __restrict__ done_count, int T, int H, int D_rt, int T_kv, int topk_rt,
                                 float softmax_scale) {}

template<int ND, int NDS, int TOPK = 0, int PHASE = 0, int GDKV = 5, bool SPLIT_ROLES = false, int DQ_WAVES = 4,
         bool DQ_BUFFER = false, bool DQ_KBUFFER = false, int FIXED_H = 0, bool FUSE_DKV = false,
         int DISPATCH_GUARD = 0>
__global__ static void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_bwd_fused2_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kv, const bf16* __restrict__ out,
                                  const bf16* __restrict__ dout, const float* __restrict__ lse,
                                  const float* __restrict__ attn_sink, const int* __restrict__ idxs,
                                  const int* __restrict__ topk_length, float* __restrict__ dkv_float,
                                  float* __restrict__ dsink, uint32_t* __restrict__ pdl_ws, bf16* __restrict__ dq,
                                  bf16* __restrict__ dkv, int* __restrict__ done_count, int T, int H, int D_rt,
                                  int T_kv, int topk_rt, float softmax_scale, const int* dispatch_guard = nullptr) {}

template<int ND, int NDS, int TOPK = 0, int PHASE = 0, int GDKV = 5, bool SPLIT_ROLES = false, int DQ_WAVES = 4,
         bool DQ_BUFFER = false, bool DQ_KBUFFER = false, int FIXED_H = 0, bool FUSE_DKV = false,
         int DISPATCH_GUARD = 0>
__global__ static void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_bwd_full_regs_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kv,
                                     const bf16* __restrict__ out, const bf16* __restrict__ dout,
                                     const float* __restrict__ lse, const float* __restrict__ attn_sink,
                                     const int* __restrict__ idxs, const int* __restrict__ topk_length,
                                     float* __restrict__ dkv_float, float* __restrict__ dsink,
                                     uint32_t* __restrict__ pdl_ws, bf16* __restrict__ dq, bf16* __restrict__ dkv,
                                     int* __restrict__ done_count, int T, int H, int D_rt, int T_kv, int topk_rt,
                                     float softmax_scale, const int* dispatch_guard = nullptr) {}

template<int HC = 128, int NC = 4, int D = 512, bool PREFETCH = true, bool IDX_LDS = true, bool GUARDED = false,
         bool DUP_GUARD = false, int HEAD_MODE = 0>
__global__ static void sparse_dkv_cached_kernel(const bf16*, const bf16*, const uint32_t*, const int*, const int*,
                                                float*, bf16*, int*, int, int, int, int, const int* = nullptr,
                                                int = D) {}

#endif

// Two actual queries contribute to one dKV accumulator before the atomic.
// The GPU classifier has proved the exact compactify mapping for every row.
// Q, dO, P and dS retain their original BF16 bits; no query or key is dropped.
__global__ __launch_bounds__(256, 1) static void sparse_dkv_grouped2_compact(const bf16* q, const bf16* dout,
                                                                             const uint32_t* pdl, float* dkv_float,
                                                                             bf16* dkv, int* done, const int* guard) {
#if defined(__gfx936__) || defined(__gfx938__)
    if (guard[0] != 2 || (int64_t)guard[1] * 4 < (int64_t)4096 * 1152) return;
    constexpr int T = 4096, H = 64, D = 512, K = 1152, S = 5120, PLD = 68, PT = 16 * PLD;
    const int qb = ((int)blockIdx.x / 2) * 2, cb = ((int)blockIdx.x % 2) * 256;
    const int lane = threadIdx.x & 63, r = lane & 15, g = lane >> 4;
    const int warp    = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int maxcomp = (qb + 2) / 4, first = max(0, qb - 127), union_count = maxcomp + (qb + 2 - first);
    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> qcache[2][4][4], dcache[2][4][4];
#    pragma unroll
    for (int qi = 0; qi < 2; qi++) {
#    pragma unroll
        for (int hg = 0; hg < 4; hg++) {
#    pragma unroll
            for (int dc = 0; dc < 4; dc++) {
#    pragma unroll
                for (int e = 0; e < 4; e++) {
                    const size_t off = ((size_t)(qb + qi) * H + hg * 16 + g * 4 + e) * D + cb + warp * 64 + dc * 16 + r;
                    qcache[qi][hg][dc].x[e] = ((const hbf16*)q)[off];
                    dcache[qi][hg][dc].x[e] = ((const hbf16*)dout)[off];
                }
            }
// Issue four Q/dO fragment pairs before constraining their uses.
// Preserve all source values, fragment layout and MMAC order.
#    pragma unroll
            for (int dc = 0; dc < 4; dc++) {
                asm volatile("" ::"v"(*(const uint64_t*)&qcache[qi][hg][dc].x[0]),
                             "v"(*(const uint64_t*)&dcache[qi][hg][dc].x[0])
                             : "memory");
            }
        }
    }
    __shared__ hbf16 pd[4 * PT];
    saf_u4e pref[2];
    auto prefetch_pd = [&](int pos) {
        const int load_slot = pos + (int)threadIdx.x / 16, head = ((int)threadIdx.x % 16) * 4;
        const int load_key = load_slot < maxcomp ? T + load_slot : first + load_slot - maxcomp;
#    pragma unroll
        for (int qi = 0; qi < 2; qi++) {
            const int qt = qb + qi, nc = (qt + 1) / 4, window = min(qt + 1, 128), wf = qt + 1 - window;
            const bool is_comp = load_key >= T;
            const bool valid =
                load_slot < union_count && (is_comp ? load_key < T + nc : load_key >= wf && load_key <= qt);
            int source   = is_comp ? load_key - T : 1024 + load_key - wf;
            const int li = source & 255, seg = source >> 8, full = nc / 256, rem = nc % 256;
            const int physical  = full * li + min(li, rem) + min(li, window) + (seg == 4 ? full + (li < rem) : seg);
            const uint64_t mask = __ballot(valid);
            uint64_t address    = (uint64_t)(pdl + ((size_t)qt * K + (valid ? physical : 0)) * H + head);
            pref[qi]            = (saf_u4e){0, 0, 0, 0};
            uint64_t saved_exec;
            // Same global-load opcode/cache flags as saf_nt_load_u4's verified ISA.
            // Wait explicitly at the LDS commit, keeping this payload live across MMAC.
            asm volatile("s_and_saveexec_b64 %1, %3\n\t"
                         "global_load_dwordx4 %0, %2, off glc slc\n\t"
                         "s_mov_b64 exec, %1"
                : "+v"(pref[qi]),"=&s"(saved_exec)
                : "v"(address),"s"(mask):"exec","memory");
        }
    };
    prefetch_pd(0);
    for (int pos = 0; pos < union_count; pos += 16) {
        DUFragment<accumulator, 16, 16, 16, float> acc[4];
        du_fill_fragment(acc[0], 0.f);
        du_fill_fragment(acc[1], 0.f);
        du_fill_fragment(acc[2], 0.f);
        du_fill_fragment(acc[3], 0.f);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        const int head = ((int)threadIdx.x % 16) * 4;
#    pragma unroll
        for (int qi = 0; qi < 2; qi++) {
            const saf_u4e packed = pref[qi];
            const int row        = (int)threadIdx.x / 16;
            uint32_t* pp         = (uint32_t*)(pd + qi * 2 * PT + row * PLD + head);
            uint32_t* dd         = (uint32_t*)(pd + qi * 2 * PT + PT + row * PLD + head);
            pp[0]                = (packed.x & 65535) | (packed.y << 16);
            pp[1]                = (packed.z & 65535) | (packed.w << 16);
            dd[0]                = (packed.x >> 16) | (packed.y & 0xffff0000);
            dd[1]                = (packed.z >> 16) | (packed.w & 0xffff0000);
        }
        __syncthreads();
        if (pos + 16 < union_count) prefetch_pd(pos + 16);
#    pragma unroll
        for (int qi = 0; qi < 2; qi++) {
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[4], da[4];
#    pragma unroll
            for (int hg = 0; hg < 4; hg++) {
                du_load_matrix_sync(pa[hg], pd + qi * 2 * PT + hg * 16, PLD);
                du_load_matrix_sync(da[hg], pd + qi * 2 * PT + PT + hg * 16, PLD);
            }
#    pragma unroll
            for (int dc = 0; dc < 4; dc++) {
#    pragma unroll
                for (int hg = 0; hg < 4; hg++) {
                    du_mma_sync(acc[dc], pa[hg], dcache[qi][hg][dc], acc[dc]);
                    du_mma_sync(acc[dc], da[hg], qcache[qi][hg][dc], acc[dc]);
                }
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
        const int slot = pos + r, key = slot < maxcomp ? T + slot : first + slot - maxcomp;
        const uint64_t mask = __ballot(slot < union_count);
        uint64_t address    = (uint64_t)dkv_float + ((uint64_t)key * D + cb + warp * 64 + g * 4) * sizeof(float);
        saf_floatx4 values[4];
#    pragma unroll
        for (int dc = 0; dc < 4; dc++) {
            values[dc] = {acc[dc].x[0], acc[dc].x[1], acc[dc].x[2], acc[dc].x[3]};
            uint64_t saved_exec;
            asm volatile("s_and_saveexec_b64 %1, %3\n\t"
                         "global_atomic_add_f32_x4 %2, %0, off offset:%4\n\t"
                         "s_mov_b64 exec, %1"
                : "+v"(values[dc]),"=&s"(saved_exec),"+v"(address)
                : "s"(mask),"n"(dc*64):"exec","memory");
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#    pragma unroll
        for (int dc = 0; dc < 4; dc++) asm volatile("" ::"v"(values[dc]), "v"(address) : "memory");
    }
    __threadfence();
    __syncthreads();
    __shared__ int previous;
    if (threadIdx.x == 0) previous = atomicAdd(done, 1);
    __syncthreads();
    saf_multi_cast_swizzled(done, dkv_float, dkv, S, D, 4096, previous);
#endif
}

#if defined(__gfx936__) || defined(__gfx938__)
template<int HC = 128, int NC = 4, int D = 512, bool PREFETCH = true, bool IDX_LDS = true, bool GUARDED = false>
__device__ __forceinline__ static void sparse_dkv_cached_dispatch_body(
    const bf16* q, const bf16* dout, const uint32_t* pdl, const int* idx, const int* lengths, float* dkvf, bf16* dkv,
    int* done, int T, int H, int K, int Tkv, const int* dispatch_guard = nullptr, int value_stride = D) {
    if constexpr (GUARDED) {
        const bool fuse = (dispatch_guard[0] & 1) == 0 && (int64_t)dispatch_guard[1] * 4 >= (int64_t)T * K;
        // The wide D576 variant is selected by the same whole-call locality
        // predicate that makes the guarded narrow variant return.
        if constexpr (HC == 64 && D == 576 && NC == 9) {
            if (!fuse) return;
        } else {
            if (fuse) return;
        }
    }

    constexpr int HG = HC / 16, PLDM = saf_dkv_ldm<HC, D>, TILE = 16 * PLDM;
    // Prefetch the next packed P/dS tile during the current contraction.
    constexpr bool PREFETCH_PD = PREFETCH && ((HC == 128 && D == 512) || (HC == 64 && D == 576));
    const int t = blockIdx.x, ds = blockIdx.y;
    // A compact GLM dO has 512 physical columns. Materialize its zero
    // tail in registers, retaining the original MMAC and atomic order.
    const int dvalue = D == 576 ? value_stride : D;
    const int lane   = threadIdx.x & 63;
    const int wave   = PREFETCH_PD ? __builtin_amdgcn_readfirstlane(threadIdx.x >> 6) : threadIdx.x >> 6;
    const int r = lane & 15, g = lane >> 4;
    const int dc0   = (ds * 4 + wave) * NC;
    const int valid = lengths ? max(0, min(K, lengths[t])) : K;
    extern __shared__ hbf16 sm[];
    // The top-k index row is K int32 per query.  Reading it with one global load
    // per 16-key tile puts a full-latency, L2-thrashing access (T*K*4 bytes)
    // directly on the tile ballot's critical path. Preload the row cooperatively
    // so subsequent ballots consume LDS-resident indices. IDX_LDS is
    // false when the row cannot fit in the 64 KiB LDS budget, in which case the
    // launcher allocates no index region and the per-tile global fetch is kept.
    int* saf_idx_lds = IDX_LDS ? (int*)(sm + 2 * TILE) : nullptr;
    if constexpr (IDX_LDS) {
        for (int u = threadIdx.x; u < valid; u += 256) saf_idx_lds[u] = idx[(size_t)t * K + u];
        __syncthreads();
    }
    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> qq[HG][NC], dd[HG][NC];
#    pragma unroll
    for (int s = 0; s < HG; s++) {
#    pragma unroll
        for (int j = 0; j < NC; j++) {
#    pragma unroll
            for (int e = 0; e < 4; e++) {
                const int h    = s * 16 + g * 4 + e;
                const size_t a = ((size_t)t * H + h) * D + (dc0 + j) * 16 + r;
                qq[s][j].x[e]  = ((HC != 16 || h < H) && (D % (64 * NC) == 0 || (dc0 + j) * 16 + r < D))
                                     ? ((const hbf16*)q)[a]
                                     : hbf16(0);
                if constexpr (D == 576)
                    dd[s][j].x[e] = ((HC != 16 || h < H) && (dc0 + j) * 16 + r < dvalue)
                                        ? ((const hbf16*)dout)[((size_t)t * H + h) * dvalue + (dc0 + j) * 16 + r]
                                        : hbf16(0);
                else
                    dd[s][j].x[e] = ((HC != 16 || h < H) && (D % (64 * NC) == 0 || (dc0 + j) * 16 + r < D))
                                        ? ((const hbf16*)dout)[a]
                                        : hbf16(0);
            }
            if constexpr (PREFETCH_PD) {
                // Finish packing this Q/dO pair before starting the next global loads.
                // Otherwise LLVM overlaps the whole prologue and spills 19 VGPRs in
                // the H128 prefetch path. This compiler dependency emits no instruction;
                // both gfx936/gfx938 retain the same cached fragments and MMAC order.
                asm volatile("" ::"v"(*(const uint64_t*)&qq[s][j].x[0]), "v"(*(const uint64_t*)&dd[s][j].x[0])
                             : "memory");
            }
        }
    }
    // Streaming P/dS fetches: column CTAs reread this large workspace, which
    // would otherwise evict the FP32 dKV atomic accumulator from the 8 MB L2.
    saf_u4e packed[PREFETCH_PD ? (16 * (HC / 4) + 255) / 256 : 1];
    int carried_key = -1;
    if constexpr (PREFETCH_PD) carried_key = r < valid ? (IDX_LDS ? saf_idx_lds[r] : idx[(size_t)t * K + r]) : -1;
    auto prefetch_pd = [&](int next, int next_key) {
        if constexpr (PREFETCH_PD) {
            // P/dS for empty tiles need not be initialized by the stats pass.
            // Match the traversal's CTA-uniform ballot before reading that tile.
            if (!__builtin_amdgcn_readfirstlane(__any(next_key >= 0 && next_key < Tkv))) return;
#    pragma unroll
            for (int z = 0; z < (16 * (HC / 4) + 255) / 256; z++) {
                const int u = threadIdx.x + z * 256, key = u / (HC / 4), h = (u % (HC / 4)) * 4;
                packed[z] = (u < 16 * (HC / 4) && next + key < K)
                                ? saf_nt_load_u4((const void*)(pdl + ((size_t)t * K + next + key) * H + h))
                                : (saf_u4e){0, 0, 0, 0};
            }
        }
    };
    prefetch_pd(0, carried_key);
// Match stats/dQ's logical tile bound. Stats leaves wholly masked suffix
// tiles unwritten; the last partial tile is still fully initialized.
// P/dS loads are naturally contiguous over heads; Q/dO never reload.
#    pragma unroll 1
    for (int kt = 0; kt < valid; kt += 16) {
        const int key = PREFETCH_PD
                            ? carried_key
                            : (kt + r < valid ? (IDX_LDS ? saf_idx_lds[kt + r] : idx[(size_t)t * K + kt + r]) : -1);
        if constexpr (PREFETCH_PD)
            carried_key =
                kt + 16 + r < valid ? (IDX_LDS ? saf_idx_lds[kt + 16 + r] : idx[(size_t)t * K + kt + 16 + r]) : -1;
        // r spans the same 16 selections in every wave.  This branch is CTA-
        // uniform, including empty rows and holes, and precedes all LDS barriers.
        if (!__builtin_amdgcn_readfirstlane(__any(key >= 0 && key < Tkv))) {
            // Empty tiles still advance the prefetch to preserve holes/lengths.
            prefetch_pd(kt + 16, carried_key);
            continue;
        }
        if constexpr (PREFETCH_PD) {
#    pragma unroll
            for (int z = 0; z < (16 * (HC / 4) + 255) / 256; z++) {
                const int u = threadIdx.x + z * 256;
                if (u < 16 * (HC / 4)) {
                    const int key = u / (HC / 4), h = (u % (HC / 4)) * 4;
                    uint32_t* p     = (uint32_t*)(sm + key * PLDM + h);
                    uint32_t* d     = (uint32_t*)(sm + TILE + key * PLDM + h);
                    const saf_u4e x = packed[z];
                    p[0]            = (x.x & 65535) | (x.y << 16);
                    p[1]            = (x.z & 65535) | (x.w << 16);
                    d[0]            = (x.x >> 16) | (x.y & 0xffff0000);
                    d[1]            = (x.z >> 16) | (x.w & 0xffff0000);
                }
            }
        } else {
            for (int u = threadIdx.x; u < 16 * (HC / 4); u += 256) {
                const int key = u / (HC / 4), h = (u % (HC / 4)) * 4;
                uint32_t* p = (uint32_t*)(sm + key * PLDM + h);
                uint32_t* d = (uint32_t*)(sm + TILE + key * PLDM + h);
                if (kt + key < K && (HC != 16 || h + 3 < H)) {
                    const saf_u4e x = saf_nt_load_u4((const void*)(pdl + ((size_t)t * K + kt + key) * H + h));
                    p[0]            = (x.x & 65535) | (x.y << 16);
                    p[1]            = (x.z & 65535) | (x.w << 16);
                    d[0]            = (x.x >> 16) | (x.y & 0xffff0000);
                    d[1]            = (x.z >> 16) | (x.w & 0xffff0000);
                } else {
                    p[0] = p[1] = d[0] = d[1] = 0;
                }
            }
        }
        __syncthreads();
        // Reuse registers while the current tile remains in LDS. The existing
        // end-of-tile barrier protects LDS before the next iteration overwrites it.
        prefetch_pd(kt + 16, carried_key);
#    if defined(__gfx936__) || defined(__gfx938__)
        constexpr bool COLUMN_ATOMICS = (HC == 64 && D == 576) || (HC == 128 && D == 512 && PREFETCH && IDX_LDS);
#    else
        constexpr bool COLUMN_ATOMICS = false;
#    endif
        if constexpr (COLUMN_ATOMICS && ((HC == 128 && D == 512) || (HC == 64 && D == 576 && NC == 9))) {
            saf_floatx4 atomic_values[NC];
            uint64_t atomic_address = (uint64_t)dkvf + ((uint64_t)(uint32_t)key * D + dc0 * 16 + 4 * g) * sizeof(float);
            const uint64_t atomic_mask = __ballot(key >= 0 && key < Tkv);
            // Cache each P/dS fragment once. Scatter each completed column while
            // computing the next, preserving the per-output FP32 accumulation order.
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[HG], da[HG];
#    pragma unroll
            for (int s = 0; s < HG; s++) {
                du_load_matrix_sync(pa[s], sm + s * 16, PLDM);
                du_load_matrix_sync(da[s], sm + TILE + s * 16, PLDM);
            }
#    pragma unroll
            for (int j = 0; j < NC; j++) {
                DUFragment<accumulator, 16, 16, 16, float> acc;
                du_fill_fragment(acc, 0.f);
#    pragma unroll
                for (int s = 0; s < HG; s++) {
                    du_mma_sync(acc, pa[s], dd[s][j], acc);
                    du_mma_sync(acc, da[s], qq[s][j], acc);
                }
                atomic_values[j] = {acc.x[0], acc.x[1], acc.x[2], acc.x[3]};
                // Integer address arithmetic remains defined for masked invalid keys.
                // All VMEM operands stay live until the completion wait below.
                uint64_t saved_exec;
                // Keep C++ control flow uniform: no asynchronous return-value PHI
                // copies may execute before the final wait. Only the atomic is masked;
                // LDS fragment reads and MMAC execute with the original full-wave EXEC.
                asm volatile("s_and_saveexec_b64 %1, %3\n\t"
                   "global_atomic_add_f32_x4 %0, %2, %0, off offset:%4 glc\n\t"
                   "s_mov_b64 exec, %1"
       : "+v"(atomic_values[j]), "=&s"(saved_exec), "+v"(atomic_address)
       : "s"(atomic_mask), "n"(j*64) : "exec", "memory");
            }
            // All native atomic return registers must be complete before reuse.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#    pragma unroll
            for (int j = 0; j < NC; j++) asm volatile("" ::"v"(atomic_values[j]), "v"(atomic_address) : "memory");
        } else if constexpr (COLUMN_ATOMICS) {
            saf_floatx4 atomic_returns[NC], atomic_values[NC];
            uint64_t atomic_addresses[NC];
            const uint64_t atomic_mask = __ballot(key >= 0 && key < Tkv);
            // Cache each P/dS fragment once. Scatter each completed column while
            // computing the next, preserving the per-output FP32 accumulation order.
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[HG], da[HG];
#    pragma unroll
            for (int s = 0; s < HG; s++) {
                du_load_matrix_sync(pa[s], sm + s * 16, PLDM);
                du_load_matrix_sync(da[s], sm + TILE + s * 16, PLDM);
            }
#    pragma unroll
            for (int j = 0; j < NC; j++) {
                DUFragment<accumulator, 16, 16, 16, float> acc;
                du_fill_fragment(acc, 0.f);
#    pragma unroll
                for (int s = 0; s < HG; s++) {
                    du_mma_sync(acc, pa[s], dd[s][j], acc);
                    du_mma_sync(acc, da[s], qq[s][j], acc);
                }
                atomic_values[j] = {acc.x[0], acc.x[1], acc.x[2], acc.x[3]};
                atomic_addresses[j] =
                    (uint64_t)dkvf + ((uint64_t)(uint32_t)key * D + (dc0 + j) * 16 + 4 * g) * sizeof(float);
                // Integer address arithmetic remains defined for masked invalid keys.
                // All VMEM operands stay live until the completion wait below.
                uint64_t saved_exec;
                // Keep C++ control flow uniform: no asynchronous return-value PHI
                // copies may execute before the final wait. Only the atomic is masked;
                // LDS fragment reads and MMAC execute with the original full-wave EXEC.
                asm volatile("s_and_saveexec_b64 %1, %4\n\t"
                   "global_atomic_add_f32_x4 %0, %2, %3, off glc\n\t"
                   "s_mov_b64 exec, %1"
       : "=&v"(atomic_returns[j]), "=&s"(saved_exec),
         "+v"(atomic_addresses[j]), "+v"(atomic_values[j])
       : "s"(atomic_mask) : "exec", "memory");
            }
            // All native atomic return registers must be complete before reuse.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#    pragma unroll
            for (int j = 0; j < NC; j++)
                asm volatile("" ::"v"(atomic_returns[j]), "v"(atomic_values[j]), "v"(atomic_addresses[j]) : "memory");
        } else {
            DUFragment<accumulator, 16, 16, 16, float> acc[NC];
#    pragma unroll
            for (int j = 0; j < NC; j++) du_fill_fragment(acc[j], 0.f);
#    pragma unroll
            for (int s = 0; s < HG; s++) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa, da;
                du_load_matrix_sync(pa, sm + s * 16, PLDM);
                du_load_matrix_sync(da, sm + TILE + s * 16, PLDM);
#    pragma unroll
                for (int j = 0; j < NC; j++) {
                    du_mma_sync(acc[j], pa, dd[s][j], acc[j]);
                    du_mma_sync(acc[j], da, qq[s][j], acc[j]);
                }
            }

#    if defined(__gfx936__) || defined(__gfx938__)
            // Preserve every asynchronous physical return until the full completion wait.
            // Uniform C++ flow avoids compiler-generated return-value PHI copies.
            saf_floatx4 atomic_values[NC];
            uint64_t atomic_address = (uint64_t)dkvf + ((uint64_t)(uint32_t)key * D + dc0 * 16 + 4 * g) * sizeof(float);
#        pragma unroll
            for (int j = 0; j < NC; j++) {
                atomic_values[j]    = {acc[j].x[0], acc[j].x[1], acc[j].x[2], acc[j].x[3]};
                const uint64_t mask = __ballot(key >= 0 && key < Tkv && (D % (64 * NC) == 0 || (dc0 + j) * 16 < D));
                uint64_t saved_exec;
                asm volatile("s_and_saveexec_b64 %1, %3\n\t"
                "global_atomic_add_f32_x4 %0, %2, %0, off offset:%4 glc\n\t"
                "s_mov_b64 exec, %1"
    : "+v"(atomic_values[j]), "=&s"(saved_exec), "+v"(atomic_address)
    : "s"(mask), "n"(j*64) : "exec", "memory");
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#        pragma unroll
            for (int j = 0; j < NC; j++) asm volatile("" ::"v"(atomic_values[j]), "v"(atomic_address) : "memory");
#    else
            if (key >= 0 && key < Tkv) {
#        pragma unroll
                for (int j = 0; j < NC; j++)
                    if (D % (64 * NC) == 0 || (dc0 + j) * 16 < D)
                        saf_atomic_add_mmac_row_swizzled(dkvf + (size_t)key * D + (dc0 + j) * 16, g, acc[j].x[0],
                                                         acc[j].x[1], acc[j].x[2], acc[j].x[3]);
            }
#    endif
        }
        __syncthreads();
    }
    __threadfence();
    __syncthreads();
    __shared__ int prev;
    if (threadIdx.x == 0) prev = atomicAdd(done, 1);
    __syncthreads();
    saf_multi_cast_swizzled(done, dkvf, dkv, Tkv, D, T * gridDim.y, prev);
}

__device__ __forceinline__ static void sparse_dkv_grouped2_dispatch_body(const bf16* q, const bf16* dout,
                                                                         const uint32_t* pdl, float* dkv_float,
                                                                         bf16* dkv, int* done, const int* guard) {
#    if defined(__gfx936__) || defined(__gfx938__)
    if (guard[0] != 2 || (int64_t)guard[1] * 4 < (int64_t)4096 * 1152) return;
    constexpr int T = 4096, H = 64, D = 512, K = 1152, S = 5120, PLD = 68, PT = 16 * PLD;
    const int qb = ((int)blockIdx.x / 2) * 2, cb = ((int)blockIdx.x % 2) * 256;
    const int lane = threadIdx.x & 63, r = lane & 15, g = lane >> 4;
    const int warp    = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int maxcomp = (qb + 2) / 4, first = max(0, qb - 127), union_count = maxcomp + (qb + 2 - first);
    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> qcache[2][4][4], dcache[2][4][4];
#        pragma unroll
    for (int qi = 0; qi < 2; qi++) {
#        pragma unroll
        for (int hg = 0; hg < 4; hg++) {
#        pragma unroll
            for (int dc = 0; dc < 4; dc++) {
#        pragma unroll
                for (int e = 0; e < 4; e++) {
                    const size_t off = ((size_t)(qb + qi) * H + hg * 16 + g * 4 + e) * D + cb + warp * 64 + dc * 16 + r;
                    qcache[qi][hg][dc].x[e] = ((const hbf16*)q)[off];
                    dcache[qi][hg][dc].x[e] = ((const hbf16*)dout)[off];
                }
            }
// Issue four Q/dO fragment pairs before constraining their uses.
// Preserve all source values, fragment layout and MMAC order.
#        pragma unroll
            for (int dc = 0; dc < 4; dc++) {
                asm volatile("" ::"v"(*(const uint64_t*)&qcache[qi][hg][dc].x[0]),
                             "v"(*(const uint64_t*)&dcache[qi][hg][dc].x[0])
                             : "memory");
            }
        }
    }
    extern __shared__ hbf16 pd[];
    saf_u4e pref[2];
    auto prefetch_pd = [&](int pos) {
        const int load_slot = pos + (int)threadIdx.x / 16, head = ((int)threadIdx.x % 16) * 4;
        const int load_key = load_slot < maxcomp ? T + load_slot : first + load_slot - maxcomp;
#        pragma unroll
        for (int qi = 0; qi < 2; qi++) {
            const int qt = qb + qi, nc = (qt + 1) / 4, window = min(qt + 1, 128), wf = qt + 1 - window;
            const bool is_comp = load_key >= T;
            const bool valid =
                load_slot < union_count && (is_comp ? load_key < T + nc : load_key >= wf && load_key <= qt);
            int source   = is_comp ? load_key - T : 1024 + load_key - wf;
            const int li = source & 255, seg = source >> 8, full = nc / 256, rem = nc % 256;
            const int physical  = full * li + min(li, rem) + min(li, window) + (seg == 4 ? full + (li < rem) : seg);
            const uint64_t mask = __ballot(valid);
            uint64_t address    = (uint64_t)(pdl + ((size_t)qt * K + (valid ? physical : 0)) * H + head);
            pref[qi]            = (saf_u4e){0, 0, 0, 0};
            uint64_t saved_exec;
            // Same global-load opcode/cache flags as saf_nt_load_u4's verified ISA.
            // Wait explicitly at the LDS commit, keeping this payload live across MMAC.
            asm volatile("s_and_saveexec_b64 %1, %3\n\t"
                         "global_load_dwordx4 %0, %2, off glc slc\n\t"
                         "s_mov_b64 exec, %1"
                : "+v"(pref[qi]),"=&s"(saved_exec)
                : "v"(address),"s"(mask):"exec","memory");
        }
    };
    prefetch_pd(0);
    for (int pos = 0; pos < union_count; pos += 16) {
        DUFragment<accumulator, 16, 16, 16, float> acc[4];
        du_fill_fragment(acc[0], 0.f);
        du_fill_fragment(acc[1], 0.f);
        du_fill_fragment(acc[2], 0.f);
        du_fill_fragment(acc[3], 0.f);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        const int head = ((int)threadIdx.x % 16) * 4;
#        pragma unroll
        for (int qi = 0; qi < 2; qi++) {
            const saf_u4e packed = pref[qi];
            const int row        = (int)threadIdx.x / 16;
            uint32_t* pp         = (uint32_t*)(pd + qi * 2 * PT + row * PLD + head);
            uint32_t* dd         = (uint32_t*)(pd + qi * 2 * PT + PT + row * PLD + head);
            pp[0]                = (packed.x & 65535) | (packed.y << 16);
            pp[1]                = (packed.z & 65535) | (packed.w << 16);
            dd[0]                = (packed.x >> 16) | (packed.y & 0xffff0000);
            dd[1]                = (packed.z >> 16) | (packed.w & 0xffff0000);
        }
        __syncthreads();
        if (pos + 16 < union_count) prefetch_pd(pos + 16);
#        pragma unroll
        for (int qi = 0; qi < 2; qi++) {
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[4], da[4];
#        pragma unroll
            for (int hg = 0; hg < 4; hg++) {
                du_load_matrix_sync(pa[hg], pd + qi * 2 * PT + hg * 16, PLD);
                du_load_matrix_sync(da[hg], pd + qi * 2 * PT + PT + hg * 16, PLD);
            }
#        pragma unroll
            for (int dc = 0; dc < 4; dc++) {
#        pragma unroll
                for (int hg = 0; hg < 4; hg++) {
                    du_mma_sync(acc[dc], pa[hg], dcache[qi][hg][dc], acc[dc]);
                    du_mma_sync(acc[dc], da[hg], qcache[qi][hg][dc], acc[dc]);
                }
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
        const int slot = pos + r, key = slot < maxcomp ? T + slot : first + slot - maxcomp;
        const uint64_t mask = __ballot(slot < union_count);
        uint64_t address    = (uint64_t)dkv_float + ((uint64_t)key * D + cb + warp * 64 + g * 4) * sizeof(float);
        saf_floatx4 values[4];
#        pragma unroll
        for (int dc = 0; dc < 4; dc++) {
            values[dc] = {acc[dc].x[0], acc[dc].x[1], acc[dc].x[2], acc[dc].x[3]};
            uint64_t saved_exec;
            asm volatile("s_and_saveexec_b64 %1, %3\n\t"
                         "global_atomic_add_f32_x4 %2, %0, off offset:%4\n\t"
                         "s_mov_b64 exec, %1"
                : "+v"(values[dc]),"=&s"(saved_exec),"+v"(address)
                : "s"(mask),"n"(dc*64):"exec","memory");
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#        pragma unroll
        for (int dc = 0; dc < 4; dc++) asm volatile("" ::"v"(values[dc]), "v"(address) : "memory");
    }
    __threadfence();
    __syncthreads();
    __shared__ int previous;
    if (threadIdx.x == 0) previous = atomicAdd(done, 1);
    __syncthreads();
    saf_multi_cast_swizzled(done, dkv_float, dkv, S, D, 4096, previous);
#    endif
}
#endif

// Mutually exclusive dKV paths share dynamic LDS and one physical launch.
// Canonical grouped2 uses only y=0; fallback uses the original (T,2) grid.
// All completion counts and original atomic waits remain inside their bodies.
__global__ __launch_bounds__(256, 1) static void sparse_dkv_dispatch_compact64(const bf16* q, const bf16* dout,
                                                                               const uint32_t* pdl, const int* idx,
                                                                               const int* lengths, float* dkvf,
                                                                               bf16* dkv, int* done, const int* guard) {
#if defined(__gfx936__) || defined(__gfx938__)
    const bool compact = guard[0] == 2 && (int64_t)guard[1] * 4 >= (int64_t)4096 * 1152;
    if (compact) {
        if (blockIdx.y == 0) sparse_dkv_grouped2_dispatch_body(q, dout, pdl, dkvf, dkv, done, guard);
    } else {
        sparse_dkv_cached_dispatch_body<64, 4, 512, true, true, false>(q, dout, pdl, idx, lengths, dkvf, dkv, done,
                                                                       4096, 64, 1152, 5120, guard, 512);
    }
#endif
}

// Classify memory locality without changing any query's indices or values.
// At most two contiguous runs, stable addresses between adjacent rows, and
// sufficient total work. This check never merges queries, keys or gradients.
template<bool CLASSIFY_KEY_PACK = false>
__global__ static void saf_classify_compact_only(const int* idx, const int* lens, int* guard, int T, int S, int K,
                                                 float* dkv_zero, long long dkv_n) {
    // Zero the fp32 dKV accumulator as part of this launch.  Every backward path
    // already runs this kernel (or, for the non-fused paths, runs it harmlessly
    // with an unused guard), and the stores overlap the index reads below.  A
    // standalone torch fill of the same buffer measured ~70 us per call under the
    // 8-GPU training load because it is launch/latency bound.
    if (dkv_zero != nullptr) {
        const long long n4     = dkv_n >> 2;
        const long long stride = (long long)(gridDim.x * blockDim.x);
        float4* v4             = (float4*)dkv_zero;
        for (long long i = (long long)(blockIdx.x * blockDim.x + threadIdx.x); i < n4; i += stride)
            v4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        const long long base = n4 << 2;
        for (long long i = (long long)(blockIdx.x * blockDim.x + threadIdx.x); i < dkv_n - base; i += stride)
            dkv_zero[base + i] = 0.f;
    }
    // This helper is used only for explicit lengths, T4096/S5120/K1152.
    // For this exact shape, the original classifier rejects the whole call
    // when any row is not canonical. If all rows are canonical, row4095 has
    // more than two runs, so guard bit1 is necessarily set. Thus contiguous
    // and adjacent-row tests cannot change the selected dKV path here.
    const int lane       = (int)threadIdx.x & 63;
    const int wave       = __builtin_amdgcn_readfirstlane((int)threadIdx.x >> 6);
    const int t          = 4 * (int)blockIdx.x + wave;
    const int n          = max(0, min(K, lens[t]));
    const int compressed = min((t + 1) / 4, 1024), window = min(t + 1, 128), first = t + 1 - window;
    const int full = compressed / 256, rem = compressed % 256;
    int bad = n != compressed + window;
    // One wave validates one row. A 16-byte aligned vector covers four
    // consecutive physical indices; every logical tail element is still
    // masked. K1152 and the row stride are divisible by four.
    for (int pos = 0; pos < n; pos += 256) {
        const int k = pos + lane * 4;
        if (k < n) {
            const int4 packed   = *(const int4*)(idx + (size_t)t * K + k);
            const int values[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int j = 0; j < 4; j++)
                if (k + j < n) {
                    const int x = values[j];
                    int source;
                    if (x >= T && x < T + compressed)
                        source = x - T;
                    else if (x >= first && x <= t)
                        source = 1024 + x - first;
                    else {
                        source = 0;
                        bad    = 1;
                    }
                    const int ilane = source & 255, segment = source >> 8;
                    const int expected  = full * ilane + min(ilane, rem) + min(ilane, window) +
                                          (segment == 4 ? full + (ilane < rem) : segment);
                    bad                |= expected != k + j;
                }
        }
    }
    bad = __any(bad);
    // Every row still participates; no CTA-wide shared reduction or barrier.
    if (lane == 0) {
        const int flags = (bad ? 1 : 0) | (t == T - 1 ? 2 : 0);
        if (flags) atomicOr(guard, flags);
        if (t == 0) guard[1] = 4 * (1023 * 1024 / 2) + 1024 + 128 * 129 / 2 + (4096 - 128) * 128;
    }
}

template<bool CLASSIFY_KEY_PACK = false>
__global__ static void saf_classify_local_runs(const int* idx, const int* lens, int* guard, int T, int S, int K,
                                               float* dkv_zero, long long dkv_n) {
    // Zero the fp32 dKV accumulator as part of this launch.  Every backward path
    // already runs this kernel (or, for the non-fused paths, runs it harmlessly
    // with an unused guard), and the stores overlap the index reads below.  A
    // standalone torch fill of the same buffer measured ~70 us per call under the
    // 8-GPU training load because it is launch/latency bound.
    if (dkv_zero != nullptr) {
        const long long n4     = dkv_n >> 2;
        const long long stride = (long long)(gridDim.x * blockDim.x);
        float4* v4             = (float4*)dkv_zero;
        for (long long i = (long long)(blockIdx.x * blockDim.x + threadIdx.x); i < n4; i += stride)
            v4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        const long long base = n4 << 2;
        for (long long i = (long long)(blockIdx.x * blockDim.x + threadIdx.x); i < dkv_n - base; i += stride)
            dkv_zero[base + i] = 0.f;
    }
    int t = blockIdx.x, n = lens ? max(0, min(K, lens[t])) : K;
    int runs = 0, valid = 0, unstable = 0;
    // Recognize the exact existing 256-thread compactify order. This only
    // selects a kernel; indices, order, values and gradient ownership stay intact.
    const int compressed = min((t + 1) / 4, 1024), window = min(t + 1, 128), first = t + 1 - window;
    const bool compact_shape = lens && K == 1152 && T % 4 == 0 && T <= 4096 && S - T == T / 4;
    int compact_bad          = !(compact_shape && n == compressed + window);
    const int full = compressed / 256, rem = compressed % 256;
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        int x = idx[(size_t)t * K + k];
        if (compact_shape) {
            int source;
            if (x >= T && x < T + compressed)
                source = x - T;
            else if (x >= first && x <= t)
                source = 1024 + x - first;
            else {
                source      = 0;
                compact_bad = 1;
            }
            const int lane = source & 255, segment = source >> 8;
            const int expected =
                full * lane + min(lane, rem) + min(lane, window) + (segment == 4 ? full + (lane < rem) : segment);
            compact_bad |= expected != k;
        }
        if (x >= 0 && x < S) {
            valid++;
            if (t > 0) {
                int pn = lens ? max(0, min(K, lens[t - 1])) : K;
                int px = idx[(size_t)(t - 1) * K + k];
                if (k < pn && px >= 0 && px < S) unstable |= (x - px > 1 || px - x > 1);
            }
            int prev  = k ? idx[(size_t)t * K + k - 1] : -1;
            runs     += k == 0 || prev < 0 || prev >= S || x != prev + 1;
        }
    }
    for (int off = 32; off; off /= 2) {
        runs        += __shfl_down(runs, off);
        valid       += __shfl_down(valid, off);
        unstable    |= __shfl_down(unstable, off);
        compact_bad |= __shfl_down(compact_bad, off);
    }
    __shared__ int rr[4], vv[4], uu[4], bb[4];
    if ((threadIdx.x & 63) == 0) {
        rr[threadIdx.x >> 6] = runs;
        vv[threadIdx.x >> 6] = valid;
        uu[threadIdx.x >> 6] = unstable;
        bb[threadIdx.x >> 6] = compact_bad;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        runs                  = rr[0] + rr[1] + rr[2] + rr[3];
        valid                 = vv[0] + vv[1] + vv[2] + vv[3];
        const bool contiguous = runs <= 2 && !(uu[0] || uu[1] || uu[2] || uu[3]);
        const bool compact    = !(bb[0] || bb[1] || bb[2] || bb[3]);
        int flags             = !contiguous && !compact ? 1 : 0;
        // Exact-shape grouped2 ignores explicit indices after classification.
        // Every row must satisfy the compact mapping, including rows that also
        // look contiguous: OR-ing flags cannot distinguish a mixed whole call.
        // Reject the whole fused call if any such row violates that mapping.
        if (T == 4096 && compact_shape && !compact) flags |= 1;
        // Bit 0 rejects fusion; bit 1 retains the legacy accumulator layout.
        // Compactify order is valid for fusion but lacks adjacent-key coalescing.
        if constexpr (CLASSIFY_KEY_PACK) {
            if (!contiguous) flags |= 2;
        }
        if (flags) atomicOr(guard, flags);
        atomicAdd(guard + 1, valid);
    }
}

// Zeroing helper for the cold (non-fused2) backward launchers: they keep a
// separate launch because they have no classifier to fuse with.
__global__ static void saf_zero_f32(float* p, long long n) {
    const long long stride = (long long)(gridDim.x * blockDim.x);
    for (long long i = (long long)(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += stride) p[i] = 0.f;
}

// Small-D and mixed-dtype backward launchers.

// ---------------------------------------------------------------------------
// Single-kernel sparse_attn_bwd launcher: one grid-(T, nslab) kernel runs
// the dkv / dq pass AND the f32->bf16 dkv cast (last-block pattern), so the
// whole backward is exactly one kernel launch.
// dkv_float (T_kv x D f32), dsink and done_count must be zeroed by the
// caller on the stream.
// ---------------------------------------------------------------------------
static bool launch_sparse_attn_bwd_fused(hipStream_t stream, const void* q, const void* kv, const void* out,
                                         const void* dout, const float* lse, const float* attn_sink,
                                         const int* topk_idxs, const int* topk_length, void* dq, float* dkv_float,
                                         float* dsink, void* dkv, int* done_count, int T, int H, int D, int D_v,
                                         int T_kv, int topk, float softmax_scale) {
    if (!sparse_attn_fused_supported()) return false;
    if (D % 16 != 0 || D > SAF_MAX_D || (D_v != D && !(D == 576 && D_v == 512))) return false;
    // H may be any value >= 1: partial 64-head slabs are masked per-lane.
    if (H <= 0 || T <= 0 || topk <= 0) return false;

    const int max_lds = dsa_shared_memory_limit();
    const int KLD     = D + 8;
    // 16 staged keys x KLD bf16 + P^T / dl^T tiles (2 x 4 warps x 256 bf16)
    // + this row's topk indices.
    size_t shm = (size_t)16 * KLD * 2 + 2 * (size_t)4 * 256 * 2 + (size_t)topk * 4;
    if (shm > (size_t)max_lds) return false;
    hipLaunchKernelGGL((saf_zero_f32), dim3(256), 256, 0, stream, dkv_float, (long long)T_kv * D);

    // One block per (query row, 64-head slab): H > 64 slabs run in parallel.
    dim3 grid(T, (H + SAF_HBLK - 1) / SAF_HBLK);
    const int nd_ = D / 16;
#define SAB_FUSED(NDV)                                                                                               \
    hipLaunchKernelGGL((sparse_attn_bwd_fused_kernel<NDV, 0>), grid, SAF_THREADS, shm, stream, (const bf16*)q,       \
                       (const bf16*)kv, (const bf16*)out, (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length, \
                       dkv_float, dsink, (bf16*)dq, (bf16*)dkv, done_count, T, H, D, T_kv, topk, softmax_scale)
#define SAB_FUSED_T(NDV, TOPK)                                                                                       \
    hipLaunchKernelGGL((sparse_attn_bwd_fused_kernel<NDV, TOPK>), grid, SAF_THREADS, shm, stream, (const bf16*)q,    \
                       (const bf16*)kv, (const bf16*)out, (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length, \
                       dkv_float, dsink, (bf16*)dq, (bf16*)dkv, done_count, T, H, D, T_kv, topk, softmax_scale)
#define SAB_CASE(NDV)                                 \
    case NDV:                                         \
        switch (topk) {                               \
            case 64: SAB_FUSED_T(NDV, 64); break;     \
            case 128: SAB_FUSED_T(NDV, 128); break;   \
            case 256: SAB_FUSED_T(NDV, 256); break;   \
            case 512: SAB_FUSED_T(NDV, 512); break;   \
            case 1024: SAB_FUSED_T(NDV, 1024); break; \
            case 2048: SAB_FUSED_T(NDV, 2048); break; \
            default: SAB_FUSED(NDV); break;           \
        }                                             \
        break
    if (D_v != D) {
        hipLaunchKernelGGL((sparse_attn_bwd_fused_kernel<36, 0, true>), grid, SAF_THREADS, shm, stream, (const bf16*)q,
                           (const bf16*)kv, (const bf16*)out, (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length,
                           dkv_float, dsink, (bf16*)dq, (bf16*)dkv, done_count, T, H, D, T_kv, topk, softmax_scale);
    } else
        switch (nd_) {
            SAB_CASE(4);   // D = 64
            SAB_CASE(8);   // D = 128
            SAB_CASE(16);  // D = 256
            SAB_CASE(28);  // D = 448
            SAB_CASE(32);  // D = 512
            SAB_CASE(36);  // D = 576
            default: return false;
        }
#undef SAB_FUSED
#undef SAB_FUSED_T
#undef SAB_CASE
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_bwd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}

// Large-D backward workspace, stats, dQ and cached dKV launchers.

// Compact values preserve the public 512-channel O/dO storage. Only the
// cached GLM path supports this layout; every other path keeps padding.
static bool sparse_attn_bwd_compact_values_supported(int H, int topk, bool cached_backward) {
    if (!(H == 8 || H == 16 || H == 32 || H == 64 || H == 128) || topk < 128) return false;
    int device = 0;
    hipGetDevice(&device);
    hipDeviceProp_t_v2 prop;
    hipGetDeviceProperties_v2(&prop, device);
    return (strstr(prop.gcnArchName, "gfx936") || strstr(prop.gcnArchName, "gfx938")) && cached_backward;
}

// ---------------------------------------------------------------------------
// Large-D sparse_attn_bwd launcher (fused2).  Hot D=512 shapes launch the
// stats and gradient phases separately; other shapes retain one grid-(T,
// nslab) launch. The caller allocates packed P/dl workspace (T, topk, H).
// ---------------------------------------------------------------------------
// Stream-ordered initializer for the register-fused path. The following
// compute launch observes all zeros; no grid-wide in-kernel barrier is needed.
__global__ static void saf_zero_register_fused(float* dkv, long long n, float* dsink, int H, int* counters) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (long long i = tid; i < n / 4; i += (long long)gridDim.x * blockDim.x)
        ((float4*)dkv)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (blockIdx.x == 0) {
        if (threadIdx.x < H) dsink[threadIdx.x] = 0.f;
        if (threadIdx.x < 3) counters[threadIdx.x] = 0;
    }
}

// Exact GPU-side padding contract. A single nonzero/NaN tail operand,
// nonzero LSE, non--inf sink, nonfinite KV or scale selects full H64 work.
// Nothing is inferred from TP, a sampled row, or a previous Graph replay.
__global__ static void saf_pack_padded_qd(const bf16* q, const bf16* dout, uint4* dst, const int* flag, int T) {
    if (flag[0]) return;
    const int t = blockIdx.x, lane = threadIdx.x & 63, hg = (threadIdx.x >> 6) & 1;
    const int h = hg * 16 + (lane >> 4) * 4;
    for (int coltile = (threadIdx.x >> 7); coltile < 36; coltile += 2) {
        const int col = coltile * 16 + (lane & 15);
        uint32_t qb[4], db[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            qb[e] = q[((size_t)t * 32 + h + e) * 576 + col];
            db[e] = col < 512 ? dout[((size_t)t * 32 + h + e) * 512 + col] : 0;
        }
        dst[(((size_t)t * 36 + coltile) * 4 + hg) * 64 + lane] = {qb[0] | (qb[1] << 16), qb[2] | (qb[3] << 16),
                                                                  db[0] | (db[1] << 16), db[2] | (db[3] << 16)};
    }
}
// Cheap rejection for ordinary dense H64; acceptance still requires the
// complete per-call scan below, including every query and every Graph replay.
__global__ static void saf_reject_dense_h64(const bf16* q, const float* sink, int* flag) {
    unsigned bad = 0;
    if (threadIdx.x < 32) bad |= __float_as_uint(sink[32 + threadIdx.x]) != 0xff800000u;
    bad |= (q[32 * 576 + threadIdx.x] & 0x7fff) != 0;
    if (__any(bad) && (threadIdx.x & 63) == 0) atomicOr(flag, 1);
}
__global__ static void saf_check_padded_h32(const bf16* q, const bf16* out, const bf16* dout, const float* lse,
                                            const float* sink, int* flag, int T) {
    if (flag[0]) return;
    const int t  = blockIdx.x;
    unsigned bad = 0;
    for (int i = threadIdx.x; i < 32 * 576 / 8; i += blockDim.x) {
        const uint4 x  = ((const uint4*)q)[((size_t)t * 64 + 32) * 576 / 8 + i];
        bad           |= (x.x | x.y | x.z | x.w) & 0x7fff7fff;
    }
    for (int i = threadIdx.x; i < 32 * 512 / 8; i += blockDim.x) {
        const uint4 x  = ((const uint4*)out)[((size_t)t * 64 + 32) * 512 / 8 + i];
        const uint4 y  = ((const uint4*)dout)[((size_t)t * 64 + 32) * 512 / 8 + i];
        bad           |= (x.x | x.y | x.z | x.w | y.x | y.y | y.z | y.w) & 0x7fff7fff;
    }
    if (threadIdx.x < 32) {
        bad |= (__float_as_uint(lse[(size_t)t * 64 + 32 + threadIdx.x]) & 0x7fffffff);
        bad |= __float_as_uint(sink[32 + threadIdx.x]) != 0xff800000u;
    }
    if (__any(bad) && (threadIdx.x & 63) == 0) atomicOr(flag, 1);
}
__global__ static void saf_check_finite_kv(const bf16* kv, int* flag, int n, float scale) {
    if (flag[0]) return;
    unsigned bad = !isfinite(scale);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        bad |= (kv[i] & 0x7f80) == 0x7f80;
    if (__any(bad) && (threadIdx.x & 63) == 0) atomicOr(flag, 1);
}
__global__ static void saf_pack_padded_h32(const bf16* q, const bf16* out, const bf16* dout, const float* lse, bf16* cq,
                                           bf16* co, bf16* cd, float* cl, const int* flag, int T) {
    if (flag[0]) return;
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < 32 * 576 / 8; i += blockDim.x)
        ((uint4*)cq)[(size_t)t * 32 * 576 / 8 + i] = ((const uint4*)q)[(size_t)t * 64 * 576 / 8 + i];
    for (int i = threadIdx.x; i < 32 * 512 / 8; i += blockDim.x) {
        ((uint4*)co)[(size_t)t * 32 * 512 / 8 + i] = ((const uint4*)out)[(size_t)t * 64 * 512 / 8 + i];
        ((uint4*)cd)[(size_t)t * 32 * 512 / 8 + i] = ((const uint4*)dout)[(size_t)t * 64 * 512 / 8 + i];
    }
    if (threadIdx.x < 32) cl[(size_t)t * 32 + threadIdx.x] = lse[(size_t)t * 64 + threadIdx.x];
}
__global__ static void saf_scatter_padded_h32(const bf16* compact, bf16* dq, const int* flag, int T) {
    if (flag[0]) return;
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < 64 * 576 / 8; i += blockDim.x) {
        uint4 v = {0, 0, 0, 0};
        if (i < 32 * 576 / 8) v = ((const uint4*)compact)[(size_t)t * 32 * 576 / 8 + i];
        ((uint4*)dq)[(size_t)t * 64 * 576 / 8 + i] = v;
    }
}

static bool launch_sparse_attn_bwd_fused2(hipStream_t stream, const void* q, const void* kv, const void* out,
                                          const void* dout, const float* lse, const float* attn_sink,
                                          const int* topk_idxs, const int* topk_length, void* dq, float* dkv_float,
                                          float* dsink, void* dkv, int* done_count, void* pdl_ws, int T, int H, int D,
                                          int D_v, int T_kv, int topk, float softmax_scale, bool full_workspace,
                                          bool cached_backward) {
    if (!sparse_attn_fused_supported()) return false;

    if (D % 64 != 0 || D > SAF_MAX_D ||
        (D_v != D && !(D == 576 && D_v == 512 && sparse_attn_bwd_compact_values_supported(H, topk, cached_backward))))
        return false;
    if (H <= 0 || T <= 0 || topk <= 0) return false;

    const int nd_  = D / 16;
    const int nds_ = nd_ / 4;  // 4 slice passes; NDS = ND/4 d-tiles each
    // The workspace is zeroed here rather than by a separate fill kernel: every
    // path below accumulates into dkv_float, so this classifier launch is
    // unconditional (its guard is simply unused by the non-fused paths).
    {
        const long long dkv_n = (long long)T_kv * D;
        int* guard            = done_count + 1;
        if (T == 4096 && H == 64 && D == 512 && T_kv == 5120 && topk == 1152) {
            // Initialize the two small buffers before the classifier atomics.
            hipLaunchKernelGGL(saf_zero_register_fused, dim3(1), 256, 0, stream, dkv_float, 0LL, dsink, H, done_count);
        }
        if (sparse_attn_bwd_register_fused_shape(T, H, D, T_kv, topk)) {
            hipLaunchKernelGGL(saf_zero_register_fused, dim3(256), 256, 0, stream, dkv_float, dkv_n, dsink, H,
                               done_count);
        } else if (T == 4096 && H == 64 && D == 512 && T_kv == 5120 && topk == 1152 && topk_length) {
            hipLaunchKernelGGL((saf_classify_compact_only<true>), dim3(T / 4), SAF_THREADS, 0, stream, topk_idxs,
                               topk_length, guard, T, T_kv, topk, dkv_float, dkv_n);
        } else if (nd_ == 36) {
            hipLaunchKernelGGL((saf_classify_local_runs<false>), dim3(T), SAF_THREADS, 0, stream, topk_idxs,
                               topk_length, guard, T, T_kv, topk, dkv_float, dkv_n);
        } else {
            hipLaunchKernelGGL((saf_classify_local_runs<true>), dim3(T), SAF_THREADS, 0, stream, topk_idxs, topk_length,
                               guard, T, T_kv, topk, dkv_float, dkv_n);
        }
    }
    // The preceding launch initializes the shared FP32 accumulator.
    // Short rows keep P/dS in registers across all gradient contractions.
    if (sparse_attn_bwd_register_fused_shape(T, H, D, T_kv, topk)) {
#define FULL_REGS_LAUNCH(K, HEADS)                                                                                    \
    hipLaunchKernelGGL((sparse_attn_bwd_full_regs_kernel<32, 16, K, 1, 5, false, 4, false, false, HEADS, true, 0>),   \
                       dim3(T, HEADS / 64), SAF_THREADS, 20480, stream, (const bf16*)q, (const bf16*)kv,              \
                       (const bf16*)out, (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length, dkv_float, dsink, \
                       (uint32_t*)pdl_ws, (bf16*)dq, (bf16*)dkv, done_count, T, H, D, T_kv, topk, softmax_scale,      \
                       nullptr)
        if (H == 64) {
            if (topk == 128) {
                FULL_REGS_LAUNCH(128, 64);
            } else {
                FULL_REGS_LAUNCH(160, 64);
            }
        } else {
            if (topk == 128) {
                FULL_REGS_LAUNCH(128, 128);
            } else {
                FULL_REGS_LAUNCH(160, 128);
            }
        }
#undef FULL_REGS_LAUNCH
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_bwd HIP call failed: %s",
                                        hipGetErrorString(status));
            }
        }
        return true;
    }
    // topk*4 bytes of the 64 KiB LDS budget hold the preloaded index row.
    // 8192 keys leave the P/dS tiles and the attribute limit comfortable;
    // larger rows (only synthetic single-query workspaces) keep the per-tile
    // global fetch.
    const int DSA_IDX_LDS_MAX = 8192;
    // Reserve the original K-row capacity so the stats index cache keeps
    // its existing offset. Phase 1 now packs K rows with stride D+4;
    // the legacy combined phases still use D+24.
    const int KLD1 = D + 24;
    // One region is shared by the pass-1 K rows / aliased P/dl tiles and
    // the pass-2 dl/K regions (see the kernel prologue).  A full 16-row
    // tile uses 17152 bytes for D=512 and avoids partial MMAC tiles.
    const int ch_ = 16;
    size_t shm    = (size_t)ch_ * KLD1 * 2;
    // The dkv phase aliases five P/dl chunk pairs over the pass-1 K region.
    // 21120 B still permits three resident blocks in a 64 KiB LDS/CU.
    const size_t dkv_shm = (size_t)5 * 2 * 16 * 66 * 2;
    if (shm < dkv_shm) shm = dkv_shm;
    // Keep the common allocation at the phase-2 requirement.  Phase 1 has a
    // smaller K-only footprint, so it uses stats_shm directly.
    // Match CACHE_STATS_INDEX in the constant-TOPK phase-1 specializations.
    // Runtime K, long training rows and D576 keep the existing LDS budget.
    const bool cache_stats_index = nd_ == 32 && (topk == 128 || topk == 160 || topk == 512 || topk == 1024);
    const size_t stats_shm       = (size_t)ch_ * KLD1 * 2 + (cache_stats_index ? (size_t)topk * sizeof(int) : 0);
    if (nd_ % nds_ != 0) return false;

    dim3 grid(T, (H + SAF_HBLK - 1) / SAF_HBLK);
    // Split phase 2 uses one dKV role and one role per dQ slice.
    dim3 grid_split8(T, (H + SAF_HBLK - 1) / SAF_HBLK, 5);
    dim3 grid_split16(T, (H + SAF_HBLK - 1) / SAF_HBLK, 3);
#define SAB2_LAUNCH(NDV, NDSV)                                                                                        \
    hipLaunchKernelGGL((sparse_attn_bwd_fused2_kernel<NDV, NDSV, 0>), grid, SAF_THREADS, shm, stream, (const bf16*)q, \
                       (const bf16*)kv, (const bf16*)out, (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length,  \
                       dkv_float, dsink, (uint32_t*)pdl_ws, (bf16*)dq, (bf16*)dkv, done_count, T, H, D, T_kv, topk,   \
                       softmax_scale)
#define SAB2_LAUNCH_T(NDV, NDSV, TOPK)                                                                                 \
    hipLaunchKernelGGL((sparse_attn_bwd_fused2_kernel<NDV, NDSV, TOPK>), grid, SAF_THREADS, shm, stream,               \
                       (const bf16*)q, (const bf16*)kv, (const bf16*)out, (const bf16*)dout, lse, attn_sink,           \
                       topk_idxs, topk_length, dkv_float, dsink, (uint32_t*)pdl_ws, (bf16*)dq, (bf16*)dkv, done_count, \
                       T, H, D, T_kv, topk, softmax_scale)
#define SAB2_LAUNCH_PHASE_H(NDV, NDSV, TOPKV, PHASEV, HV)                                                            \
    do {                                                                                                             \
        (void)hipFuncSetAttribute(                                                                                   \
            (const void*)sparse_attn_bwd_fused2_kernel<NDV, NDSV, TOPKV, PHASEV, 5, false, 4, false, false, HV>,     \
            hipFuncAttributeMaxDynamicSharedMemorySize, (int)stats_shm);                                             \
        hipLaunchKernelGGL((sparse_attn_bwd_fused2_kernel<NDV, NDSV, TOPKV, PHASEV, 5, false, 4, false, false, HV>), \
                           grid, SAF_THREADS, stats_shm, stream, (const bf16*)q, (const bf16*)kv, (const bf16*)out,  \
                           (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length, dkv_float, dsink,              \
                           (uint32_t*)pdl_ws, (bf16*)dq, (bf16*)dkv, done_count, T, H,                               \
                           (NDV == 36 && PHASEV == 1 ? D_v : D), T_kv, topk, softmax_scale);                         \
    } while (0)
#define SAB2_LAUNCH_PHASE(NDV, NDSV, TOPKV, PHASEV)             \
    do {                                                        \
        if ((PHASEV) == 1 && H == 64) {                         \
            SAB2_LAUNCH_PHASE_H(NDV, NDSV, TOPKV, PHASEV, 64);  \
        } else if ((PHASEV) == 1 && H == 128) {                 \
            SAB2_LAUNCH_PHASE_H(NDV, NDSV, TOPKV, PHASEV, 128); \
        } else {                                                \
            SAB2_LAUNCH_PHASE_H(NDV, NDSV, TOPKV, PHASEV, 0);   \
        }                                                       \
    } while (0)
    // GLM latent attention keeps 576 Q/K channels and 512 value channels.
    // Full-width values use the general contraction below; these kernels
    // omit the nonexistent value columns while retaining all Q/K gradients.
    if (nd_ == 36 && D_v == 512 && topk >= 128 && (H == 8 || H == 16 || H == 32 || H == 64 || H == 128)) {
        int device = 0;
        hipGetDevice(&device);
        hipDeviceProp_t_v2 prop;
        hipGetDeviceProperties_v2(&prop, device);
        const bool cached_arch = strstr(prop.gcnArchName, "gfx936") || strstr(prop.gcnArchName, "gfx938");
        if (cached_arch && cached_backward) {
            if ((T == 4096 || T == 4097) && T_kv == 4096 && H == 64 && topk == 2048 && D_v == 512) {
                {
                    const hipError_t status = hipMemsetAsync(done_count + 3, 0, sizeof(int), stream);
                    if (status != hipSuccess) {
                        throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                "bf16_sparse_attn_bwd HIP call failed: %s", hipGetErrorString(status));
                    }
                }
                hipLaunchKernelGGL(saf_reject_dense_h64, dim3(1), 256, 0, stream, (const bf16*)q, attn_sink,
                                   done_count + 3);
                hipLaunchKernelGGL(saf_check_padded_h32, dim3(T), 256, 0, stream, (const bf16*)q, (const bf16*)out,
                                   (const bf16*)dout, lse, attn_sink, done_count + 3, T);
                hipLaunchKernelGGL(saf_check_finite_kv, dim3(128), 256, 0, stream, (const bf16*)kv, done_count + 3,
                                   T_kv * D, softmax_scale);
                // Keep all heads and keys, but reuse the workspace across two
                // query batches. dKV/dsink accumulate across both batches;
                // dQ writes disjoint output rows. Stream order protects reuse.
                // Split at 2048 so 32-query consumer groups retain their layout.
                const int batch_count = full_workspace ? 1 : 2;
                for (int batch = 0; batch < batch_count; ++batch) {
                    const int begin    = batch * 2048;
                    const int rows     = full_workspace ? T : (batch == 0 ? 2048 : T - begin);
                    const bf16* qb     = (const bf16*)q + (size_t)begin * H * D;
                    const bf16* ob     = (const bf16*)out + (size_t)begin * H * D_v;
                    const bf16* db     = (const bf16*)dout + (size_t)begin * H * D_v;
                    bf16* dqb          = (bf16*)dq + (size_t)begin * H * D;
                    const float* lb    = lse + (size_t)begin * H;
                    const int* ib      = topk_idxs + (size_t)begin * topk;
                    const int* lengths = topk_length ? topk_length + begin : nullptr;
                    // Mutually exclusive layouts share the same allocation.
                    // Compact H32 P/dS plus Q/O/dO/dQ/LSE occupy < 0.8 GiB.
                    bf16* cq         = (bf16*)((uint32_t*)pdl_ws + (size_t)rows * topk * 32);
                    bf16* co         = cq + (size_t)rows * 32 * D;
                    bf16* cd         = co + (size_t)rows * 32 * D_v;
                    bf16* compact_dq = cd + (size_t)rows * 32 * D_v;
                    float* cl        = (float*)(compact_dq + (size_t)rows * 32 * D);
                    hipLaunchKernelGGL(saf_pack_padded_h32, dim3(rows), 256, 0, stream, qb, ob, db, lb, cq, co, cd, cl,
                                       done_count + 3, rows);
                    hipLaunchKernelGGL(
                        (sparse_attn_bwd_fused2_kernel<36, 9, 2048, 1, 5, false, 4, false, false, 32, false, 6>),
                        dim3(rows, 1), SAF_THREADS, stats_shm, stream, cq, (const bf16*)kv, co, cd, cl, attn_sink, ib,
                        lengths, dkv_float, dsink, (uint32_t*)pdl_ws, compact_dq, (bf16*)dkv, done_count, rows, 32, D_v,
                        T_kv, topk, softmax_scale, done_count + 3);
                    int* inv32 = (int*)((uint32_t*)pdl_ws + (size_t)rows * topk * 64);
                    {
                        const hipError_t status = hipMemsetAsync(inv32, 0, (size_t)rows * T_kv * sizeof(int), stream);
                        if (status != hipSuccess) {
                            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                    "bf16_sparse_attn_bwd HIP call failed: %s",
                                                    hipGetErrorString(status));
                        }
                    }
                    {
                        const hipError_t status = hipMemsetAsync(done_count + 1, 0, sizeof(int), stream);
                        if (status != hipSuccess) {
                            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                    "bf16_sparse_attn_bwd HIP call failed: %s",
                                                    hipGetErrorString(status));
                        }
                    }
                    hipLaunchKernelGGL((stage130_build_inverse<true, 32>), dim3(rows), 256, 0, stream, ib, lengths,
                                       inv32, done_count + 1, rows);
                    hipLaunchKernelGGL(saf_pack_padded_qd, dim3(rows), 256, 0, stream, cq, cd,
                                       (uint4*)(inv32 + (size_t)rows * 4096), done_count + 3, rows);
                    hipLaunchKernelGGL((sparse_dkv_key_shared<768, 6, 32, true, 32>), dim3(43, (rows + 31) / 32), 768,
                                       13824, stream, (const uint32_t*)pdl_ws, inv32, done_count + 1, dkv_float, rows);
                    hipLaunchKernelGGL((sparse_dkv_cached_kernel<32, 9, 576, true, true, false, true, 32>),
                                       dim3(rows, 1), SAF_THREADS,
                                       (size_t)(2 * 16 * saf_dkv_ldm<32, 576> * 2) + (size_t)topk * 4, stream, cq, cd,
                                       (const uint32_t*)pdl_ws, ib, lengths, dkv_float, (bf16*)dkv, done_count, rows,
                                       32, topk, T_kv, done_count + 1, D_v);
                    hipLaunchKernelGGL(
                        (sparse_attn_bwd_fused2_kernel<36, 16, 2048, 4, 5, true, 4, true, true, 32, false, 6>),
                        dim3(rows, 1, 3), 256, 8448, stream, cq, (const bf16*)kv, co, cd, cl, attn_sink, ib, lengths,
                        dkv_float, dsink, (uint32_t*)pdl_ws, compact_dq, (bf16*)dkv, done_count, rows, 32, D, T_kv,
                        topk, softmax_scale, done_count + 3);
                    hipLaunchKernelGGL(saf_scatter_padded_h32, dim3(rows), 256, 0, stream, compact_dq, dqb,
                                       done_count + 3, rows);
                    int* inverse = (int*)((uint32_t*)pdl_ws + (size_t)rows * topk * H);
                    // The mutually exclusive H32/H64 branches share the inverse
                    // cleared above; neither requires a second memset.
                    hipLaunchKernelGGL((stage130_build_inverse<true>), dim3(rows), 256, 0, stream, ib, lengths, inverse,
                                       done_count + 1, rows);
                    hipLaunchKernelGGL(
                        (sparse_attn_bwd_fused2_kernel<36, 9, 2048, 1, 11, false, 4, false, false, 64, true, 5>),
                        dim3(rows, 1), SAF_THREADS, stats_shm, stream, qb, (const bf16*)kv, ob, db, lb, attn_sink, ib,
                        lengths, dkv_float, dsink, (uint32_t*)pdl_ws, dqb, (bf16*)dkv, done_count, rows, H, D_v, T_kv,
                        topk, softmax_scale, done_count + 3);
                    hipLaunchKernelGGL((sparse_dkv_key_shared<768, 6, 32, true>), dim3(43, (rows + 31) / 32), 768,
                                       26112, stream, (const uint32_t*)pdl_ws, inverse, done_count + 1, dkv_float,
                                       rows);
                    // Duplicate indices are checked separately in each batch;
                    // the cached fallback accumulates into the same FP32 dKV.
                    hipLaunchKernelGGL((sparse_dkv_cached_kernel<64, 9, 576, true, true, false, true, 64>),
                                       dim3(rows, 1), SAF_THREADS,
                                       (size_t)(2 * 16 * saf_dkv_ldm<64, 576> * 2) + (size_t)topk * 4, stream, qb, db,
                                       (const uint32_t*)pdl_ws, ib, lengths, dkv_float, (bf16*)dkv, done_count, rows, H,
                                       topk, T_kv, done_count + 1, D_v);
                    hipLaunchKernelGGL(
                        (sparse_attn_bwd_fused2_kernel<36, 8, 2048, 4, 7, true, 4, true, true, 64, false, 5>),
                        dim3(rows, 1, 4), 256, 4352, stream, qb, (const bf16*)kv, ob, db, lb, attn_sink, ib, lengths,
                        dkv_float, dsink, (uint32_t*)pdl_ws, dqb, (bf16*)dkv, done_count, rows, H, D, T_kv, topk,
                        softmax_scale, done_count + 3);
                    {
                        const hipError_t status = hipGetLastError();
                        if (status != hipSuccess) {
                            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                    "bf16_sparse_attn_bwd HIP call failed: %s",
                                                    hipGetErrorString(status));
                        }
                    }
                }
                hipLaunchKernelGGL(sparse_dkv_cast_no_gridwait_kernel, dim3(128), 256, 0, stream, dkv_float, (bf16*)dkv,
                                   T_kv, D);
                {
                    const hipError_t status = hipGetLastError();
                    if (status != hipSuccess) {
                        throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                "bf16_sparse_attn_bwd HIP call failed: %s", hipGetErrorString(status));
                    }
                }
                return true;
            }
            if ((T == 4096 || T == 4097) && T_kv == 4096 && H == 64 && topk == 2048) {
                SAB2_LAUNCH_PHASE_H(36, 9, 2048, 1, 64);
            } else if ((T == 4096 || T == 4097) && T_kv == 4096 && H == 32 && topk == 2048) {
                SAB2_LAUNCH_PHASE_H(36, 9, 2048, 1, 32);
            } else {
                SAB2_LAUNCH_PHASE(36, 9, 0, 1);
            }
            {
                const hipError_t status = hipGetLastError();
                if (status != hipSuccess) {
                    throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_bwd HIP call failed: %s",
                                            hipGetErrorString(status));
                }
            }
#define GLM_DKV(HC)                                                                                                    \
    do {                                                                                                               \
        if (topk <= DSA_IDX_LDS_MAX) {                                                                                 \
            hipLaunchKernelGGL((sparse_dkv_cached_kernel<HC, 3, 576, true, true>), dim3(T, 3), SAF_THREADS,            \
                               (size_t)(2 * 16 * saf_dkv_ldm<HC, 576> * 2) + (size_t)topk * 4, stream, (const bf16*)q, \
                               (const bf16*)dout, (const uint32_t*)pdl_ws, topk_idxs, topk_length, dkv_float,          \
                               (bf16*)dkv, done_count, T, H, topk, T_kv, nullptr, D_v);                                \
        } else {                                                                                                       \
            hipLaunchKernelGGL((sparse_dkv_cached_kernel<HC, 3, 576, true, false>), dim3(T, 3), SAF_THREADS,           \
                               (size_t)(2 * 16 * saf_dkv_ldm<HC, 576> * 2), stream, (const bf16*)q, (const bf16*)dout, \
                               (const uint32_t*)pdl_ws, topk_idxs, topk_length, dkv_float, (bf16*)dkv, done_count, T,  \
                               H, topk, T_kv, nullptr, D_v);                                                           \
        }                                                                                                              \
    } while (0)
            // This GLM branch is guarded by H in {8, 16, 32, 64, 128}.
            switch (H) {
                case 8: GLM_DKV(16); break;
                case 16: GLM_DKV(16); break;
                case 32:
                    if ((T == 4096 || T == 4097) && T_kv == 4096 && topk == 2048) {
                        hipLaunchKernelGGL(sparse_dkv_glm_pair, dim3((T + 1) / 2), SAF_THREADS, 0, stream,
                                           (const bf16*)q, (const bf16*)dout, (const uint32_t*)pdl_ws, topk_idxs,
                                           topk_length, dkv_float, (bf16*)dkv, done_count, T, topk, T_kv, D_v);
                    } else {
                        GLM_DKV(32);
                    }
                    break;
                case 64:
                    // GLM5.2 TP=1: reuse Q/dO across all 576 columns in one CTA.
                    if ((T == 4096 || T == 4097) && T_kv == 4096 && topk == 2048) {
                        hipLaunchKernelGGL((sparse_dkv_cached_kernel<64, 9, 576, true, true>), dim3(T, 1), SAF_THREADS,
                                           (size_t)(2 * 16 * saf_dkv_ldm<64, 576> * 2) + (size_t)topk * 4, stream,
                                           (const bf16*)q, (const bf16*)dout, (const uint32_t*)pdl_ws, topk_idxs,
                                           topk_length, dkv_float, (bf16*)dkv, done_count, T, H, topk, T_kv, nullptr,
                                           D_v);
                        break;
                    }
                    // Retain all 576 Q/dO columns in one CTA only when the
                    // current indices have verified locality. The classifier
                    // reruns inside Graph replay; dense random input keeps NC=3.
                    if (T >= 2048 && topk >= 1024 && topk <= DSA_IDX_LDS_MAX && T_kv >= topk &&
                        (int64_t)T * topk <= 0x7fffffffLL) {
                        int* guard           = done_count + 1;  // launched at the top of this launcher
                        const size_t dkv_shm = 2 * 16 * saf_dkv_ldm<64, 576> * 2 + (size_t)topk * 4;
                        hipLaunchKernelGGL((sparse_dkv_cached_kernel<64, 9, 576, true, true, true>), dim3(T, 1),
                                           SAF_THREADS, dkv_shm, stream, (const bf16*)q, (const bf16*)dout,
                                           (const uint32_t*)pdl_ws, topk_idxs, topk_length, dkv_float, (bf16*)dkv,
                                           done_count, T, H, topk, T_kv, guard, D_v);
                        hipLaunchKernelGGL((sparse_dkv_cached_kernel<64, 3, 576, true, true, true>), dim3(T, 3),
                                           SAF_THREADS, dkv_shm, stream, (const bf16*)q, (const bf16*)dout,
                                           (const uint32_t*)pdl_ws, topk_idxs, topk_length, dkv_float, (bf16*)dkv,
                                           done_count, T, H, topk, T_kv, guard, D_v);
                    } else {
                        GLM_DKV(64);
                    }
                    break;
                case 128: GLM_DKV(128); break;
            }
#undef GLM_DKV
            {
                const hipError_t status = hipGetLastError();
                if (status != hipSuccess) {
                    throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_bwd HIP call failed: %s",
                                            hipGetErrorString(status));
                }
            }
#define GLM_DQ_IMPL(WAVES, RAW, RAWK)                                                                                  \
    hipLaunchKernelGGL((sparse_attn_bwd_fused2_kernel<36, 16, 0, 4, 5, true, WAVES, RAW, RAWK>), dim3(T, 1, 3),        \
                       (WAVES) * 64, 8448, stream, (const bf16*)q, (const bf16*)kv, (const bf16*)out,                  \
                       (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length, dkv_float, dsink, (uint32_t*)pdl_ws, \
                       (bf16*)dq, (bf16*)dkv, done_count, T, H, D, T_kv, topk, softmax_scale)
#define GLM_DQ_PICK_K(WAVES, RAW)                                                 \
    do {                                                                          \
        if ((WAVES) == 4 && (uint64_t)T_kv * D * sizeof(bf16) <= 0x80000000ull) { \
            GLM_DQ_IMPL(WAVES, RAW, true);                                        \
        } else {                                                                  \
            GLM_DQ_IMPL(WAVES, RAW, false);                                       \
        }                                                                         \
    } while (0)
#define GLM_DQ(WAVES)                                  \
    do {                                               \
        if ((uint64_t)topk * H * 4 <= 0x80000000ull) { \
            GLM_DQ_PICK_K(WAVES, true);                \
        } else {                                       \
            GLM_DQ_PICK_K(WAVES, false);               \
        }                                              \
    } while (0)
            // Keep every H64 head and all 576 columns; the last 128-column slab is masked.
            if ((T == 4096 || T == 4097) && T_kv == 4096 && H == 64 && topk == 2048) {
                hipLaunchKernelGGL((sparse_attn_bwd_fused2_kernel<36, 8, 2048, 4, 5, true, 4, true, true, 64>),
                                   dim3(T, 1, 5), 256, 4352, stream, (const bf16*)q, (const bf16*)kv, (const bf16*)out,
                                   (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length, dkv_float, dsink,
                                   (uint32_t*)pdl_ws, (bf16*)dq, (bf16*)dkv, done_count, T, H, D, T_kv, topk,
                                   softmax_scale);
            } else if ((T == 4096 || T == 4097) && T_kv == 4096 && (H == 64 || H == 32) && topk == 2048) {
                hipLaunchKernelGGL((sparse_attn_bwd_fused2_kernel<36, 16, 2048, 4, 5, true, 4, true, true>),
                                   dim3(T, 1, 3), 256, 8448, stream, (const bf16*)q, (const bf16*)kv, (const bf16*)out,
                                   (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length, dkv_float, dsink,
                                   (uint32_t*)pdl_ws, (bf16*)dq, (bf16*)dkv, done_count, T, H, D, T_kv, topk,
                                   softmax_scale);
            } else if (H == 128) {
                GLM_DQ(8);
            } else {
                GLM_DQ(4);
            }
#undef GLM_DQ
#undef GLM_DQ_PICK_K
#undef GLM_DQ_IMPL
            {
                const hipError_t status = hipGetLastError();
                if (status != hipSuccess) {
                    throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_bwd HIP call failed: %s",
                                            hipGetErrorString(status));
                }
            }
            return true;
        }
    }
    // Keep D=512 on split stats/gradients across top-K boundaries.
    // TOPK=0 handles tails and arbitrary runtime K (including 640); it
    // uses the same cached gradient kernels as the named specializations.
    // The ratio-128 training path has 32 compressed keys plus a 128-key
    // window. For sufficiently many queries, cached dQ/dKV also beat the
    // combined kernel below K=256; all three phases support runtime tails.
    const bool cached_training_shape = T >= 256 && topk >= 128 && (H == 64 || H == 128);
    if (nd_ == 32 && (topk >= 256 || cached_training_shape)) {
        // Select cached dKV/dQ on both deployment architectures. The cached
        // override remains available to validate the generic fallback.
        static int arch_device = -1;
        static int is_gfx936   = 0;
        static int is_gfx938   = 0;
        int current_dev        = 0;
        hipGetDevice(&current_dev);
        if (current_dev != arch_device) {
            hipDeviceProp_t_v2 prop;
            hipGetDeviceProperties_v2(&prop, current_dev);
            is_gfx936   = (strstr(prop.gcnArchName, "gfx936") != nullptr) ? 1 : 0;
            is_gfx938   = (strstr(prop.gcnArchName, "gfx938") != nullptr) ? 1 : 0;
            arch_device = current_dev;
        }
        // Both code objects use verified wave64 MMAC and native FP32 atomics.
        const bool cached_arch   = is_gfx936 || is_gfx938;
        const bool wide_dq_slice = cached_arch || H <= 64;
        const bool sab2_cached   = cached_arch && (H == 64 || H == 128) && cached_backward;
#define SAB2_P2_LAUNCH(NDSV, TOPKV, SHMV, GV, GRIDV)                                                                  \
    do {                                                                                                              \
        (void)hipFuncSetAttribute((const void*)sparse_attn_bwd_fused2_kernel<32, NDSV, TOPKV, 2, GV, true>,           \
                                  hipFuncAttributeMaxDynamicSharedMemorySize, (int)(SHMV));                           \
        hipLaunchKernelGGL((sparse_attn_bwd_fused2_kernel<32, NDSV, TOPKV, 2, GV, true>), GRIDV, SAF_THREADS,         \
                           (size_t)(SHMV), stream, (const bf16*)q, (const bf16*)kv, (const bf16*)out,                 \
                           (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length, dkv_float, dsink,               \
                           (uint32_t*)pdl_ws, (bf16*)dq, (bf16*)dkv, done_count, T, H, D, T_kv, topk, softmax_scale); \
    } while (0)
#define SAB2_P2_RUN(TOPKV)                                   \
    do {                                                     \
        if (wide_dq_slice) {                                 \
            SAB2_P2_LAUNCH(16, TOPKV, shm, 5, grid_split16); \
        } else {                                             \
            SAB2_P2_LAUNCH(8, TOPKV, 14592, 3, grid_split8); \
        }                                                    \
    } while (0)
// Cached dKV and dQ use separate kernels so their register and LDS
// budgets are independent. NDS16 dQ needs only its 8,448-byte K slab.
#define SAB2_CACHED_DKV(HC, PREFETCH, IDXLDS)                                                                         \
    do {                                                                                                              \
        /* P/dS tiles + the preloaded top-k index row (topk int32). */                                                \
        const int cached_shm = 2 * 16 * saf_dkv_ldm<HC, 512> * 2 + ((IDXLDS) ? topk * 4 : 0);                         \
        (void)hipFuncSetAttribute((const void*)sparse_dkv_cached_kernel<HC, 4, 512, PREFETCH, IDXLDS>,                \
                                  hipFuncAttributeMaxDynamicSharedMemorySize, cached_shm);                            \
        hipLaunchKernelGGL((sparse_dkv_cached_kernel<HC, 4, 512, PREFETCH, IDXLDS>), dim3(T, 2), SAF_THREADS,         \
                           cached_shm, stream, (const bf16*)q, (const bf16*)dout, (const uint32_t*)pdl_ws, topk_idxs, \
                           topk_length, dkv_float, (bf16*)dkv, done_count, T, H, topk, T_kv);                         \
    } while (0)
#define SAB2_CACHED_DKV_PICK(HC, PREFETCH)        \
    do {                                          \
        if (topk <= DSA_IDX_LDS_MAX) {            \
            SAB2_CACHED_DKV(HC, PREFETCH, true);  \
        } else {                                  \
            SAB2_CACHED_DKV(HC, PREFETCH, false); \
        }                                         \
    } while (0)
#define SAB2_CACHED_DQ_IMPL(TOPKV, WAVES, RAW, RAWK)                                                             \
    do {                                                                                                         \
        const size_t dq_shm = 8448 + (((WAVES) == 8 && topk <= SAF_DQ_INDEX_MAX_TOPK) ? (size_t)topk * 4 : 0);   \
        (void)hipFuncSetAttribute(                                                                               \
            (const void*)sparse_attn_bwd_fused2_kernel<32, 16, TOPKV, 4, 5, true, WAVES, RAW, RAWK>,             \
            hipFuncAttributeMaxDynamicSharedMemorySize, (int)dq_shm);                                            \
        hipLaunchKernelGGL((sparse_attn_bwd_fused2_kernel<32, 16, TOPKV, 4, 5, true, WAVES, RAW, RAWK>),         \
                           dim3(T, (H + (WAVES) * 16 - 1) / ((WAVES) * 16), 2), (WAVES) * 64, dq_shm, stream,    \
                           (const bf16*)q, (const bf16*)kv, (const bf16*)out, (const bf16*)dout, lse, attn_sink, \
                           topk_idxs, topk_length, dkv_float, dsink, (uint32_t*)pdl_ws, (bf16*)dq, (bf16*)dkv,   \
                           done_count, T, H, D, T_kv, topk, softmax_scale);                                      \
    } while (0)
#define SAB2_CACHED_DQ_PICK_K(TOPKV, WAVES, RAW)                                  \
    do {                                                                          \
        if ((WAVES) == 4 && (uint64_t)T_kv * D * sizeof(bf16) <= 0x80000000ull) { \
            SAB2_CACHED_DQ_IMPL(TOPKV, WAVES, RAW, true);                         \
        } else {                                                                  \
            SAB2_CACHED_DQ_IMPL(TOPKV, WAVES, RAW, false);                        \
        }                                                                         \
    } while (0)
#define SAB2_CACHED_DQ(TOPKV, WAVES)                          \
    do {                                                      \
        if constexpr (TOPKV > 0) {                            \
            SAB2_CACHED_DQ_PICK_K(TOPKV, WAVES, true);        \
        } else if ((uint64_t)topk * H * 4 <= 0x80000000ull) { \
            SAB2_CACHED_DQ_PICK_K(TOPKV, WAVES, true);        \
        } else {                                              \
            SAB2_CACHED_DQ_PICK_K(TOPKV, WAVES, false);       \
        }                                                     \
    } while (0)
#define SAB2_P2_CACHED(TOPKV)                                                                                     \
    do {                                                                                                          \
        if (H == 128) {                                                                                           \
            /* Short traversals cannot amortize prefetch register pressure. */                                    \
            if (topk >= 1024) {                                                                                   \
                SAB2_CACHED_DKV_PICK(128, true);                                                                  \
            } else {                                                                                              \
                SAB2_CACHED_DKV_PICK(128, false);                                                                 \
            }                                                                                                     \
        } else {                                                                                                  \
            SAB2_CACHED_DKV_PICK(64, true);                                                                       \
        }                                                                                                         \
        {                                                                                                         \
            const hipError_t status = hipGetLastError();                                                          \
            if (status != hipSuccess) {                                                                           \
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_bwd HIP call failed: %s", \
                                        hipGetErrorString(status));                                               \
            }                                                                                                     \
        }                                                                                                         \
        if (H == 128) {                                                                                           \
            SAB2_CACHED_DQ(TOPKV, 8);                                                                             \
        } else {                                                                                                  \
            SAB2_CACHED_DQ(TOPKV, 4);                                                                             \
        }                                                                                                         \
    } while (0)
#define SAB2_P2_DISPATCH(TOPKV)    \
    do {                           \
        if (sab2_cached)           \
            SAB2_P2_CACHED(TOPKV); \
        else                       \
            SAB2_P2_RUN(TOPKV);    \
    } while (0)
        // Fusion reuses Q/dO within each query. The GPU verifies index locality
        // on every invocation, including graph replay. It
        // chooses one whole-call path; it never combines different queries.
        const bool local_fusion_shape = sab2_cached && H == 64 && T >= 2048 && T_kv >= topk &&
                                        topk <= DSA_IDX_LDS_MAX && (int64_t)T * topk <= 0x7fffffffLL;
#define SAB2_LOCAL_FUSION(TOPKV)                                                                                       \
    do {                                                                                                               \
        int* guard = done_count + 1; /* launched at the top of this launcher */                                        \
        if ((TOPKV) == 1152 && T == 4096 && T_kv == 5120) {                                                            \
            hipLaunchKernelGGL(                                                                                        \
                (sparse_attn_bwd_fused2_kernel<32, 8, TOPKV, 1, 5, false, 4, false, false, 64, false, 3>), dim3(T, 1), \
                SAF_THREADS, 64 * 68 * 4 + ((TOPKV) > 0 && (TOPKV) <= 1024 ? topk * 4 : 0), stream, (const bf16*)q,    \
                (const bf16*)kv, (const bf16*)out, (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length,          \
                dkv_float, dsink, (uint32_t*)pdl_ws, nullptr, (bf16*)dkv, done_count, T, H, D, T_kv, topk,             \
                softmax_scale, guard);                                                                                 \
            hipLaunchKernelGGL(sparse_dkv_grouped2_compact, dim3(T), 256, 0, stream, (const bf16*)q,                   \
                               (const bf16*)dout, (const uint32_t*)pdl_ws, dkv_float, (bf16*)dkv, done_count, guard);  \
        } else {                                                                                                       \
            hipLaunchKernelGGL(                                                                                        \
                (sparse_attn_bwd_fused2_kernel<32, 8, TOPKV, 1, 5, false, 4, false, false, 64, true, 1>), dim3(T, 1),  \
                SAF_THREADS, 64 * 68 * 4 + ((TOPKV) > 0 && (TOPKV) <= 1024 ? topk * 4 : 0), stream, (const bf16*)q,    \
                (const bf16*)kv, (const bf16*)out, (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length,          \
                dkv_float, dsink, (uint32_t*)pdl_ws, nullptr, (bf16*)dkv, done_count, T, H, D, T_kv, topk,             \
                softmax_scale, guard);                                                                                 \
        }                                                                                                              \
        if (T_kv % 8 == 0) {                                                                                           \
            hipLaunchKernelGGL(                                                                                        \
                (sparse_attn_bwd_fused2_kernel<32, 8, TOPKV, 1, 5, false, 4, false, false, 64, true, 2>), dim3(T, 1),  \
                SAF_THREADS, 64 * 68 * 4 + ((TOPKV) > 0 && (TOPKV) <= 1024 ? topk * 4 : 0), stream, (const bf16*)q,    \
                (const bf16*)kv, (const bf16*)out, (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length,          \
                dkv_float, dsink, (uint32_t*)pdl_ws, nullptr, (bf16*)dkv, done_count, T, H, D, T_kv, topk,             \
                softmax_scale, guard);                                                                                 \
        }                                                                                                              \
        hipLaunchKernelGGL(                                                                                            \
            (sparse_attn_bwd_fused2_kernel<32, 8, TOPKV, 1, 5, false, 4, false, false, 64, false, -1>), dim3(T, 1),    \
            SAF_THREADS, 16 * (D + 24) * 2 + ((TOPKV) > 0 && (TOPKV) <= 1024 ? topk * 4 : 0), stream, (const bf16*)q,  \
            (const bf16*)kv, (const bf16*)out, (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length, nullptr,     \
            dsink, (uint32_t*)pdl_ws, nullptr, nullptr, nullptr, T, H, D, T_kv, topk, softmax_scale, guard);           \
        hipLaunchKernelGGL((sparse_dkv_cached_kernel<64, 4, 512, true, true, true>), dim3(T, 2), SAF_THREADS,          \
                           2 * 16 * 66 * 2 + topk * 4, stream, (const bf16*)q, (const bf16*)dout,                      \
                           (const uint32_t*)pdl_ws, topk_idxs, topk_length, dkv_float, (bf16*)dkv, done_count, T, H,   \
                           topk, T_kv, guard);                                                                         \
        {                                                                                                              \
            const hipError_t status = hipGetLastError();                                                               \
            if (status != hipSuccess) {                                                                                \
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_bwd HIP call failed: %s",      \
                                        hipGetErrorString(status));                                                    \
            }                                                                                                          \
        }                                                                                                              \
        SAB2_CACHED_DQ(TOPKV, 4);                                                                                      \
    } while (0)
        // With explicit lengths in the exact grouped2 shape, the strict
        // classifier accepts fusion only if every row has canonical compact
        // order. Canonical row4095 necessarily sets bit1 (not contiguous), so
        // the key-packed guard2 path is impossible. Run one common stats pass
        // for both accepted compact input and arbitrary fallback input.
        // Grouped2 and guarded cached-dKV still select the correct whole-call
        // path on device, including after dynamic Graph input changes.
        if (local_fusion_shape && topk == 1152 && T == 4096 && T_kv == 5120 && topk_length) {
            int* guard = done_count + 1;
            hipLaunchKernelGGL((sparse_attn_bwd_fused2_kernel<32, 8, 1152, 1, 5, false, 4, false, false, 64, false, 4>),
                               dim3(T, 1), SAF_THREADS, 64 * 68 * 4, stream, (const bf16*)q, (const bf16*)kv,
                               (const bf16*)out, (const bf16*)dout, lse, attn_sink, topk_idxs, topk_length, dkv_float,
                               dsink, (uint32_t*)pdl_ws, nullptr, (bf16*)dkv, done_count, T, H, D, T_kv, topk,
                               softmax_scale, guard);
            hipLaunchKernelGGL(sparse_dkv_dispatch_compact64, dim3(T, 2), 256, 2 * 16 * 66 * 2 + topk * 4, stream,
                               (const bf16*)q, (const bf16*)dout, (const uint32_t*)pdl_ws, topk_idxs, topk_length,
                               dkv_float, (bf16*)dkv, done_count, guard);
            SAB2_CACHED_DQ(1152, 4);
            {
                const hipError_t status = hipGetLastError();
                if (status != hipSuccess) {
                    throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_bwd HIP call failed: %s",
                                            hipGetErrorString(status));
                }
            }
            return true;
        }
        // TOPK specializes the physical index/workspace stride. Logical row
        // lengths remain runtime-clamped inside each phase, including empty
        // rows and partial tiles, so compact rows can use the same kernels.
        switch (topk) {
            case 128:
                if (local_fusion_shape) {
                    SAB2_LOCAL_FUSION(128);
                    break;
                }
                SAB2_LAUNCH_PHASE(32, 8, 128, 1);
                {
                    const hipError_t status = hipGetLastError();
                    if (status != hipSuccess) {
                        throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                "bf16_sparse_attn_bwd HIP call failed: %s", hipGetErrorString(status));
                    }
                }
                SAB2_P2_DISPATCH(128);
                break;
            case 160:
                if (local_fusion_shape) {
                    SAB2_LOCAL_FUSION(160);
                    break;
                }
                SAB2_LAUNCH_PHASE(32, 8, 160, 1);
                {
                    const hipError_t status = hipGetLastError();
                    if (status != hipSuccess) {
                        throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                "bf16_sparse_attn_bwd HIP call failed: %s", hipGetErrorString(status));
                    }
                }
                SAB2_P2_DISPATCH(160);
                break;
            case 512:
                SAB2_LAUNCH_PHASE(32, 8, 512, 1);
                {
                    const hipError_t status = hipGetLastError();
                    if (status != hipSuccess) {
                        throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                "bf16_sparse_attn_bwd HIP call failed: %s", hipGetErrorString(status));
                    }
                }
                SAB2_P2_DISPATCH(512);
                break;
            default:
                if (local_fusion_shape && topk >= 1024) {
                    SAB2_LOCAL_FUSION(0);
                    break;
                }
                SAB2_LAUNCH_PHASE(32, 8, 0, 1);
                {
                    const hipError_t status = hipGetLastError();
                    if (status != hipSuccess) {
                        throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                "bf16_sparse_attn_bwd HIP call failed: %s", hipGetErrorString(status));
                    }
                }
                SAB2_P2_DISPATCH(0);
                break;
            case 1024:
                if (local_fusion_shape) {
                    SAB2_LOCAL_FUSION(1024);
                    break;
                }
                SAB2_LAUNCH_PHASE(32, 8, 1024, 1);
                {
                    const hipError_t status = hipGetLastError();
                    if (status != hipSuccess) {
                        throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                "bf16_sparse_attn_bwd HIP call failed: %s", hipGetErrorString(status));
                    }
                }
                SAB2_P2_DISPATCH(1024);
                break;
            case 1152:
                if (local_fusion_shape) {
                    SAB2_LOCAL_FUSION(1152);
                    break;
                }
                SAB2_LAUNCH_PHASE(32, 8, 1152, 1);
                {
                    const hipError_t status = hipGetLastError();
                    if (status != hipSuccess) {
                        throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                "bf16_sparse_attn_bwd HIP call failed: %s", hipGetErrorString(status));
                    }
                }
                SAB2_P2_DISPATCH(1152);
                break;
        }
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_bwd HIP call failed: %s",
                                        hipGetErrorString(status));
            }
        }
        return true;
    }
#undef SAB2_LOCAL_FUSION
#undef SAB2_P2_RUN
#undef SAB2_CACHED_DKV
#undef SAB2_CACHED_DKV_PICK
#undef SAB2_CACHED_DQ
#undef SAB2_CACHED_DQ_PICK_K
#undef SAB2_CACHED_DQ_IMPL
#undef SAB2_P2_CACHED
#undef SAB2_P2_DISPATCH
#undef SAB2_P2_LAUNCH
#undef SAB2_LAUNCH_PHASE_H
#undef SAB2_LAUNCH_PHASE
#define SAB2_CASE(NDV, NDSV)                                  \
    case NDV:                                                 \
        switch (topk) {                                       \
            case 64: SAB2_LAUNCH_T(NDV, NDSV, 64); break;     \
            case 128: SAB2_LAUNCH_T(NDV, NDSV, 128); break;   \
            case 256: SAB2_LAUNCH_T(NDV, NDSV, 256); break;   \
            case 512: SAB2_LAUNCH_T(NDV, NDSV, 512); break;   \
            case 1024: SAB2_LAUNCH_T(NDV, NDSV, 1024); break; \
            case 2048: SAB2_LAUNCH_T(NDV, NDSV, 2048); break; \
            default: SAB2_LAUNCH(NDV, NDSV); break;           \
        }                                                     \
        break
    switch (nd_) {
        SAB2_CASE(16, 4);  // D = 256, 4 slice passes
        SAB2_CASE(28, 7);  // D = 448, 4 slice passes
        SAB2_CASE(32, 8);  // D = 512, 4 slice passes
        SAB2_CASE(36, 9);  // D = 576, 4 slice passes
        default: return false;
    }
#undef SAB2_LAUNCH
#undef SAB2_LAUNCH_T
#undef SAB2_CASE
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_bwd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}

namespace ftrain {

bool bf16SparseAttnBwdLaunchFits(int width, int columns, int heads) {
    if (width != 512 && width != 576) return false;
    // Packed stats and gradients reuse fixed tiles. Cached indices are
    // staged only up to 8192; larger selections stream from global memory.
    const std::size_t stats =
        16ULL * (width + 24) * 2 +
        ((width == 512 && (columns == 128 || columns == 160 || columns == 512 || columns == 1024)) ? columns * 4 : 0);
    const std::size_t gradient = 5 * 2 * 16 * 66 * 2;
    const std::size_t cached =
        (heads == 16 || heads == 32 || heads == 64 || heads == 128)
            ? 2ULL * 16 * (heads + ((heads == 128 || (heads == 64 && width == 576)) ? 4 : 2)) * 2 +
                  (columns <= 8192 ? columns * 4 : 0)
            : 0;
    return dsa_shared_memory_fits(stats) && dsa_shared_memory_fits(gradient) && dsa_shared_memory_fits(cached);
}

bool bf16SparseAttnBwdSparseAttnBwdCompactValuesSupported(int heads, int selection_width, bool cached_backward) {
    return sparse_attn_bwd_compact_values_supported(heads, selection_width, cached_backward);
}

bool launchBf16SparseAttnBwdFused2(FTrainStream stream, const void* q, const void* kv, const void* out,
                                   const void* dout, const float* lse, const float* attn_sink, const int* topk_indices,
                                   const int* topk_length, void* dq, float* dkv_float, float* dsink, void* dkv,
                                   int* done_count, void* pdl_ws, int tokens, int heads, int head_dim,
                                   int backward_width, int kv_tokens, int selection_width, float softmax_scale,
                                   bool full_workspace, bool cached_backward) {
    return launch_sparse_attn_bwd_fused2(stream, q, kv, out, dout, lse, attn_sink, topk_indices, topk_length, dq,
                                         dkv_float, dsink, dkv, done_count, pdl_ws, tokens, heads, head_dim,
                                         backward_width, kv_tokens, selection_width, softmax_scale, full_workspace,
                                         cached_backward);
}

bool launchBf16SparseAttnBwdFused(FTrainStream stream, const void* q, const void* kv, const void* out, const void* dout,
                                  const float* lse, const float* attn_sink, const int* topk_indices,
                                  const int* topk_length, void* dq, float* dkv_float, float* dsink, void* dkv,
                                  int* done_count, int tokens, int heads, int head_dim, int backward_width,
                                  int kv_tokens, int selection_width, float softmax_scale) {
    return launch_sparse_attn_bwd_fused(stream, q, kv, out, dout, lse, attn_sink, topk_indices, topk_length, dq,
                                        dkv_float, dsink, dkv, done_count, tokens, heads, head_dim, backward_width,
                                        kv_tokens, selection_width, softmax_scale);
}

bool bf16SparseAttnBwdSparseAttnFusedSupported() { return sparse_attn_fused_supported(); }

void bf16SparseAttnBwdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    const hipError_t status = hipMemsetAsync(memory, 0, static_cast<std::size_t>(bytes), stream);
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Workspace zero-fill failed: %s", hipGetErrorString(status));
    }
}

}  // namespace ftrain
