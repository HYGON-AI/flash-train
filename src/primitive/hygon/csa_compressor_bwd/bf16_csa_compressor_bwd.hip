// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include "flash_train/error.hpp"
#include <hip/hip_bf16.h>
#include <math.h>
#include <climits>
#include <stdint.h>

namespace hip_dsa_numeric {
// Standard BF16 round-to-nearest-even. The compiler handles finite values,
// subnormals, infinities and quieting NaNs; no arithmetic is performed in BF16.
__device__ __forceinline__ uint16_t bf16_rne(float value) { return __builtin_bit_cast(uint16_t, (__bf16)value); }

// Two independent FP32 -> BF16 conversions, followed by a bitwise 32-bit
// payload store. This is not a conversion of a BF16 pair to a numeric float.
__device__ __forceinline__ uint32_t bf16_pair(float lo, float hi) {
#if defined(__gfx938__)
    // DCC 26.04 does not select the packed instruction from convertvector.
    // gfx936 lacks this instruction and uses the compiler lowering below.
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

namespace {
typedef uint16_t csc_bf16;

__device__ __forceinline__ float csc_bf16_to_float(uint16_t v) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.u = (uint32_t)v << 16;
    return conv.f;
}

// Round-to-nearest-even fp32 -> bf16 (matches torch's bf16 cast).
__device__ __forceinline__ uint16_t csc_float_to_bf16_rne(float f) { return hip_dsa_numeric::bf16_rne(f); }

// Pinned round-to-nearest fp32 multiply / fused-multiply-add (no contraction).
__device__ __forceinline__ float csc_fmul_rn(float a, float b) {
    float r;
    asm volatile("v_mul_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}

__device__ __forceinline__ float csc_ffma_rn(float a, float b, float c) {
    float r;
    asm volatile("v_fma_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}

// The contiguous (D=1) Torch softmax uses descending xor-tree sums.
template<int N>
__device__ __forceinline__ float csc_contiguous_sum(float (&v)[N]) {
#pragma unroll
    for (int offset = N / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int i = 0; i < offset; ++i) v[i] += v[i + offset];
    }
    return v[0];
}

// Segment owning compressed block bb (binary search over cu_comp).
__device__ __forceinline__ int csc_segment_of_block(const int* __restrict__ cuc, int n_seq, int bb) {
    int lo = 0, hi = n_seq;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (cuc[mid] <= bb)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

// Zero only slots no window consumes. Each slot has exactly one CTA owner,
// disjoint from every gradient store, so caller buffers may be uninitialized.
__device__ __forceinline__ void csc_zero_unconsumed(csc_bf16* gkv, csc_bf16* gs, const int* cu, const int* cuc, int bb,
                                                    int nb_total, int n_seq, int ratio, int d, int coff, int j,
                                                    int total_tokens) {
    if (j >= d || bb >= nb_total) return;
    const int W     = coff * d;
    auto zero_token = [&](int t, bool both) {
        const size_t off = (size_t)t * W + j;
        gkv[off]         = 0;
        gs[off]          = 0;
        if (both && coff == 2) {
            gkv[off + d] = 0;
            gs[off + d]  = 0;
        }
    };
    if (bb == 0) {
        for (int s = 0; s < n_seq; ++s) {
            const int begin = cu[s], end = cu[s + 1];
            if (cuc[s] == cuc[s + 1] && begin >= 0 && end >= begin && end <= total_tokens)
                for (int t = begin; t < end; ++t) zero_token(t, true);
        }
    }
    const int pad0 = cu[n_seq];
    if (pad0 >= 0 && pad0 <= total_tokens)
        for (int64_t t = (int64_t)pad0 + bb; t < total_tokens; t += nb_total) zero_token((int)t, true);
    if (bb < cuc[n_seq]) {
        const int seq   = csc_segment_of_block(cuc, n_seq, bb);
        const int begin = cu[seq], end = cu[seq + 1];
        const int64_t tok = (int64_t)begin + ((int64_t)bb - cuc[seq]) * ratio;
        if (begin >= 0 && end >= begin && end <= total_tokens && tok >= begin && tok + ratio <= end &&
            (int64_t)cuc[seq + 1] - cuc[seq] == (end - begin) / ratio && bb + 1 == cuc[seq + 1]) {
            if (coff == 2)
                for (int k = 0; k < ratio; ++k) zero_token((int)tok + k, false);
            for (int t = (int)tok + ratio; t < end; ++t) zero_token(t, true);
        }
    }
}

struct CscWin {
    float s;  // score + ape
    float u;  // kv
    int off;  // flat offset of the consumed element (or -1)
};

__device__ __forceinline__ CscWin csc_window_elem(const csc_bf16* __restrict__ kv, const csc_bf16* __restrict__ score,
                                                  const float* __restrict__ ape, int tok0, int ratio, int d, int W,
                                                  int j, int k, int bis, bool coff2) {
    CscWin w;
    if (coff2 && k < ratio) {
        if (bis > 0) {
            w.off = (tok0 - ratio + k) * W + j;
            w.s   = csc_bf16_to_float(score[w.off]) + ape[k * W + j];
            w.u   = csc_bf16_to_float(kv[w.off]);
        } else {
            // Invalid first-in-segment window: constant -inf score (no APE
            // add), zero kv -- exactly like the eager overlap transform.
            w.s   = -INFINITY;
            w.u   = 0.0f;
            w.off = -1;
        }
    } else {
        const int col     = coff2 ? (d + j) : j;
        w.off             = (coff2 ? (tok0 + (k - ratio)) : (tok0 + k)) * W + col;
        const int ape_row = (k >= ratio) ? (k - ratio) : k;
        w.s               = csc_bf16_to_float(score[w.off]) + ape[ape_row * W + col];
        w.u               = csc_bf16_to_float(kv[w.off]);
    }
    return w;
}


__global__ void __launch_bounds__(256)
    csa_compressor_bwd_kernel(const csc_bf16* __restrict__ kv, const csc_bf16* __restrict__ score,
                              const float* __restrict__ ape, const int* __restrict__ cu, const int* __restrict__ cuc,
                              const csc_bf16* __restrict__ go,  // (nb_total * head_dim) bf16
                              csc_bf16* __restrict__ gkv,       // (total_tokens * W) bf16 (fully overwritten)
                              csc_bf16* __restrict__ gs,        // (total_tokens * W) bf16 (fully overwritten)
                              float* __restrict__ gape,         // (ratio * W) fp32 (pre-zeroed)
                              int nb_total, int n_seq, int ratio, int d, int coff, int W, int total_tokens) {
    const int bb = blockIdx.x;
    const int j  = blockIdx.y * blockDim.x + threadIdx.x;
    if (bb >= nb_total || j >= d) return;

    csc_zero_unconsumed(gkv, gs, cu, cuc, bb, nb_total, n_seq, ratio, d, coff, j, total_tokens);
    const int nb_valid = max(0, min(nb_total, cuc[n_seq]));
    if (bb >= nb_valid) return;  // padding rows: incoming grads ignored

    int seq_idx         = csc_segment_of_block(cuc, n_seq, bb);
    const int64_t bis   = (int64_t)bb - cuc[seq_idx];
    const int64_t begin = cu[seq_idx], end = cu[seq_idx + 1];
    const int64_t tok = begin + (int64_t)bis * ratio;
    const bool window_valid =
        cuc[seq_idx] >= 0 && begin >= 0 && end >= begin && end <= total_tokens && tok >= begin &&
        tok + ratio <= total_tokens &&
        (bb >= nb_valid || (tok + ratio <= end && (int64_t)cuc[seq_idx + 1] - cuc[seq_idx] == (end - begin) / ratio));
    const int tok0 = window_valid ? (int)tok : 0;
    if (!window_valid) return;
    const int win    = (coff == 2) ? 2 * ratio : ratio;
    const bool coff2 = (coff == 2);

    float mx = -INFINITY;
    for (int k = 0; k < win; k++) {
        const CscWin w = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, k, bis, coff2);
        if (w.s > mx) mx = w.s;
    }
    float den = 0.0f;
    for (int k = 0; k < win; k++) {
        const CscWin w  = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, k, bis, coff2);
        den            += expf(w.s - mx);
    }

    const float go_val = csc_bf16_to_float(go[(size_t)bb * d + j]);

    // Same expression tree as torch's softmax backward (fp32 intermediates):
    //   dp_k = go * kv_k ; S = serial sum of ROUNDED dp_k * p_k ;
    //   ds_k = fma(p_k, -S, round(dp_k * p_k)) ; dkv_k = go * p_k.
    float S = 0.0f;
    for (int k = 0; k < win; k++) {
        const CscWin w = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, k, bis, coff2);
        if (w.off >= 0) {
            const float p   = expf(w.s - mx) / den;
            const float dp  = go_val * w.u;
            S              += csc_fmul_rn(dp, p);
        }
    }

    for (int k = 0; k < win; k++) {
        const CscWin w = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, k, bis, coff2);
        if (w.off < 0) continue;  // invalid window slot: no write, no dAPE
        const float p     = expf(w.s - mx) / den;
        const float dp    = go_val * w.u;
        const float ds    = csc_ffma_rn(p, -S, csc_fmul_rn(dp, p));
        gkv[w.off]        = (csc_bf16)csc_float_to_bf16_rne(go_val * p);
        gs[w.off]         = (csc_bf16)csc_float_to_bf16_rne(ds);
        const int col     = (coff2 && k < ratio) ? j : (coff2 ? (d + j) : j);
        const int ape_row = (k >= ratio) ? (k - ratio) : k;
        atomicAdd(&gape[ape_row * W + col], ds);
    }
}

// ratio128/coff1: eight logical groups of32 dimensions per CTA; hardware
// wavefronts remain64 lanes. Each group computes16 window entries, then
// group0 accumulates den, weighted output and S in original index order.
// The17.5KiB LDS tile permits more resident CTAs than the64-dimension tile.


// Cache short windows in registers. Arithmetic sums retain increasing-k order.
template<int Capacity, bool Backward, bool Overlap>
__global__ void __launch_bounds__(128)
    csa_compressor_cached_kernel(const csc_bf16* __restrict__ kv, const csc_bf16* __restrict__ score,
                                 const float* __restrict__ ape, const int* __restrict__ cu, const int* __restrict__ cuc,
                                 csc_bf16* __restrict__ out, const csc_bf16* __restrict__ go,
                                 csc_bf16* __restrict__ gkv, csc_bf16* __restrict__ gs, float* __restrict__ gape,
                                 int nb_total, int n_seq, int ratio, int d, int W, int total_tokens) {
    const int bb = blockIdx.x;
    const int j  = blockIdx.y * blockDim.x + threadIdx.x;
    if (bb >= nb_total || j >= d) return;
    if constexpr (Backward)
        csc_zero_unconsumed(gkv, gs, cu, cuc, bb, nb_total, n_seq, ratio, d, Overlap ? 2 : 1, j, total_tokens);
    const int nb_valid = max(0, min(nb_total, cuc[n_seq]));
    if (Backward && bb >= nb_valid) return;
    const int seq       = bb < nb_valid ? csc_segment_of_block(cuc, n_seq, bb) : 0;
    const int64_t bis   = bb < nb_valid ? (int64_t)bb - cuc[seq] : 0;
    const int64_t begin = cu[seq], end = cu[seq + 1];
    const int64_t tok = begin + bis * ratio;
    const bool valid =
        cuc[seq] >= 0 && begin >= 0 && end >= begin && end <= total_tokens && tok >= begin &&
        tok + ratio <= total_tokens &&
        (bb >= nb_valid || (tok + ratio <= end && (int64_t)cuc[seq + 1] - cuc[seq] == (end - begin) / ratio));
    if (!valid) {
        if (!Backward) out[(size_t)bb * d + j] = 0;
        return;
    }
    const int tok0 = (int)tok;
    if constexpr (Capacity == 1 && !Overlap) {
        // A one-element softmax is exactly one. Gradients of score and APE
        // remain in the caller-zeroed buffers; no contended zero atomics.
        if (Backward)
            gkv[(size_t)tok0 * W + j] = go[(size_t)bb * d + j];
        else
            out[(size_t)bb * d + j] = kv[(size_t)tok0 * W + j];
        return;
    }
    const int win = (Overlap ? 2 : 1) * ratio;
    float p[Capacity], u[Capacity];
    float mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < Capacity; ++k) {
        CscWin v = {-INFINITY, 0.f, -1};
        if (k < win) v = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, k, bis, Overlap);
        p[k] = v.s;
        u[k] = v.u;
        mx   = fmaxf(mx, v.s);
    }
    float den = 0.f;
#pragma unroll
    for (int k = 0; k < Capacity; ++k) {
        p[k] = expf(p[k] - mx);
        if (k < win) den += p[k];
    }
    if (d == 1 && win == Capacity) {
        float terms[Capacity];
#pragma unroll
        for (int k = 0; k < Capacity; ++k) terms[k] = p[k];
        den = csc_contiguous_sum(terms);
    }
    const float g = Backward ? csc_bf16_to_float(go[(size_t)bb * d + j]) : 0.f;
    float sum     = 0.f, terms[Capacity];
#pragma unroll
    for (int k = 0; k < Capacity; ++k) {
        p[k] /= den;
        if (k < win) {
            const float value  = Backward ? csc_fmul_rn(g, u[k]) : u[k];
            terms[k]           = csc_fmul_rn(value, p[k]);
            sum               += terms[k];
        }
    }
    if constexpr (Backward) {
        if (d == 1 && win == Capacity) sum = csc_contiguous_sum(terms);
#pragma unroll
        for (int k = 0; k < Capacity; ++k) {
            if (k < win && !(Overlap && k < ratio && bis == 0)) {
                const int col    = Overlap && k >= ratio ? d + j : j;
                const int row    = Overlap && k >= ratio ? k - ratio : k;
                const int token  = Overlap && k < ratio ? tok0 - ratio + k : tok0 + row;
                const size_t off = (size_t)token * W + col;
                const float dp   = csc_fmul_rn(g, u[k]);
                const float ds   = csc_ffma_rn(p[k], -sum, csc_fmul_rn(dp, p[k]));
                gkv[off]         = csc_float_to_bf16_rne(g * p[k]);
                gs[off]          = csc_float_to_bf16_rne(ds);
                atomicAdd(gape + (size_t)row * W + col, ds);
            }
        }
    } else {
        out[(size_t)bb * d + j] = csc_float_to_bf16_rne(sum);
    }
}

template<int Capacity, bool Backward, bool OVERLAP>
static void csc_launch_cached(hipStream_t stream, const void* kv, const void* score, const float* ape, const int* cu,
                              const int* cuc, void* out, const void* go, void* gkv, void* gs, float* gape, int nb_total,
                              int n_seq, int ratio, int d, int coff, int total_tokens) {
#define CSC_CACHED(O)                                                                                             \
    hipLaunchKernelGGL((csa_compressor_cached_kernel<Capacity, Backward, O>), dim3(nb_total, (d + 127) / 128),    \
                       dim3(128), 0, stream, (const csc_bf16*)kv, (const csc_bf16*)score, ape, cu, cuc,           \
                       (csc_bf16*)out, (const csc_bf16*)go, (csc_bf16*)gkv, (csc_bf16*)gs, gape, nb_total, n_seq, \
                       ratio, d, coff* d, total_tokens)
    if constexpr (OVERLAP) {
        CSC_CACHED(true);
    } else {
        CSC_CACHED(false);
    }
#undef CSC_CACHED
}

// Generalize the r128 cooperative algorithm to bounded window buckets.
// Comparison-only max is parallel; FP32 den/output/S sums keep k order.
template<int Capacity, int Dim, bool Backward, bool Overlap>
__global__ void __launch_bounds__(256)
    csa_compressor_tiled_kernel(const csc_bf16* __restrict__ kv, const csc_bf16* __restrict__ score,
                                const float* __restrict__ ape, const int* __restrict__ cu, const int* __restrict__ cuc,
                                csc_bf16* __restrict__ out, const csc_bf16* __restrict__ go, csc_bf16* __restrict__ gkv,
                                csc_bf16* __restrict__ gs, float* __restrict__ gape, int nb_total, int n_seq, int ratio,
                                int d, int W, int total_tokens) {
    constexpr int Groups = 256 / Dim;
    constexpr int Items  = (Capacity + Groups - 1) / Groups;
    const int lane = threadIdx.x % Dim, group = threadIdx.x / Dim;
    const int bb = blockIdx.x, j = blockIdx.y * Dim + lane;
    const bool active = j < d;
    if constexpr (Backward) {
        if (group == 0 && active)
            csc_zero_unconsumed(gkv, gs, cu, cuc, bb, nb_total, n_seq, ratio, d, Overlap ? 2 : 1, j, total_tokens);
    }
    const int nb_valid = max(0, min(nb_total, cuc[n_seq]));
    if (bb >= nb_total || (Backward && bb >= nb_valid)) return;
    const int seq       = bb < nb_valid ? csc_segment_of_block(cuc, n_seq, bb) : 0;
    const int64_t bis   = bb < nb_valid ? (int64_t)bb - cuc[seq] : 0;
    const int64_t begin = cu[seq], end = cu[seq + 1];
    const int64_t tok = begin + bis * ratio;
    const bool valid =
        cuc[seq] >= 0 && begin >= 0 && end >= begin && end <= total_tokens && tok >= begin &&
        tok + ratio <= total_tokens &&
        (bb >= nb_valid || (tok + ratio <= end && (int64_t)cuc[seq + 1] - cuc[seq] == (end - begin) / ratio));
    if (!valid) {
        if (!Backward && active && group == 0) out[(size_t)bb * d + j] = 0;
        return;
    }
    const int tok0 = (int)tok;
    const int win  = (Overlap ? 2 : 1) * ratio;
    __shared__ float temp[Capacity * Dim], part_max[Groups * Dim];
    __shared__ float den_shared[Dim], sum_shared[Dim];
    float p[Items], mx = -INFINITY;
#pragma unroll
    for (int z = 0; z < Items; ++z) {
        const int k = group + z * Groups;
        float s     = -INFINITY;
        if (active && k < win) s = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, k, bis, Overlap).s;
        p[z] = s;
        mx   = fmaxf(mx, s);
    }
    part_max[group * Dim + lane] = mx;
    __syncthreads();
#pragma unroll
    for (int g = 0; g < Groups; ++g) mx = fmaxf(mx, part_max[g * Dim + lane]);
#pragma unroll
    for (int z = 0; z < Items; ++z) {
        const int k = group + z * Groups;
        p[z]        = active && k < win ? expf(p[z] - mx) : 0.f;
        if (k < Capacity) temp[k * Dim + lane] = p[z];
    }
    __syncthreads();
    if (group == 0) {
        float den = 0.f;
#pragma unroll 8
        for (int k = 0; k < win; ++k) den += temp[k * Dim + lane];
        den_shared[lane] = den;
    }
    __syncthreads();
    const float den = den_shared[lane];
    const float g   = Backward && active ? csc_bf16_to_float(go[(size_t)bb * d + j]) : 0.f;
#pragma unroll
    for (int z = 0; z < Items; ++z) {
        const int k = group + z * Groups;
        float u     = 0.f;
        if (active && k < win) u = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, k, bis, Overlap).u;
        p[z] /= den;
        if (k < Capacity) temp[k * Dim + lane] = csc_fmul_rn(Backward ? csc_fmul_rn(g, u) : u, p[z]);
    }
    __syncthreads();
    if (group == 0) {
        float sum = 0.f;
#pragma unroll 8
        for (int k = 0; k < win; ++k) sum += temp[k * Dim + lane];
        if (Backward)
            sum_shared[lane] = sum;
        else if (active)
            out[(size_t)bb * d + j] = csc_float_to_bf16_rne(sum);
    }
    if constexpr (Backward) {
        __syncthreads();
        const float sum = sum_shared[lane];
        if (active) {
#pragma unroll
            for (int z = 0; z < Items; ++z) {
                const int k = group + z * Groups;
                if (k < win && !(Overlap && k < ratio && bis == 0)) {
                    const int col    = Overlap && k >= ratio ? d + j : j;
                    const int row    = Overlap && k >= ratio ? k - ratio : k;
                    const int token  = Overlap && k < ratio ? tok0 - ratio + k : tok0 + row;
                    const size_t off = (size_t)token * W + col;
                    const float dp   = csc_fmul_rn(g, csc_bf16_to_float(kv[off]));
                    const float ds   = csc_ffma_rn(p[z], -sum, csc_fmul_rn(dp, p[z]));
                    gkv[off]         = csc_float_to_bf16_rne(g * p[z]);
                    gs[off]          = csc_float_to_bf16_rne(ds);
                    atomicAdd(gape + (size_t)row * W + col, ds);
                }
            }
        }
    }
}

template<int Capacity, int Dim, bool Backward, bool OVERLAP>
static void csc_launch_tiled(hipStream_t stream, const void* kv, const void* score, const float* ape, const int* cu,
                             const int* cuc, void* out, const void* go, void* gkv, void* gs, float* gape, int nb_total,
                             int n_seq, int ratio, int d, int coff, int total_tokens) {
#define CSC_TILED(O)                                                                                                   \
    hipLaunchKernelGGL((csa_compressor_tiled_kernel<Capacity, Dim, Backward, O>), dim3(nb_total, (d + Dim - 1) / Dim), \
                       dim3(256), 0, stream, (const csc_bf16*)kv, (const csc_bf16*)score, ape, cu, cuc,                \
                       (csc_bf16*)out, (const csc_bf16*)go, (csc_bf16*)gkv, (csc_bf16*)gs, gape, nb_total, n_seq,      \
                       ratio, d, coff* d, total_tokens)
    if constexpr (OVERLAP) {
        CSC_TILED(true);
    } else {
        CSC_TILED(false);
    }
#undef CSC_TILED
}

// Bound LDS for arbitrarily large windows. Each slab is produced in parallel
// and consumed in increasing-k order, retaining the FP32 reduction contract.
template<bool Backward, bool Overlap>
__global__ void __launch_bounds__(256)
    csa_compressor_streamed_kernel(const csc_bf16* __restrict__ kv, const csc_bf16* __restrict__ score,
                                   const float* __restrict__ ape, const int* __restrict__ cu,
                                   const int* __restrict__ cuc, csc_bf16* __restrict__ out,
                                   const csc_bf16* __restrict__ go, csc_bf16* __restrict__ gkv,
                                   csc_bf16* __restrict__ gs, float* __restrict__ gape, int nb_total, int n_seq,
                                   int ratio, int d, int W, int total_tokens) {
    constexpr int Dim = 8, Groups = 32, Chunk = 256, Items = Chunk / Groups;
    const int lane = threadIdx.x % Dim, group = threadIdx.x / Dim;
    const int bb = blockIdx.x, j = blockIdx.y * Dim + lane;
    const bool active = j < d;
    if constexpr (Backward) {
        if (group == 0 && active)
            csc_zero_unconsumed(gkv, gs, cu, cuc, bb, nb_total, n_seq, ratio, d, Overlap ? 2 : 1, j, total_tokens);
    }
    const int nb_valid = max(0, min(nb_total, cuc[n_seq]));
    if (bb >= nb_total || (Backward && bb >= nb_valid)) return;
    const int seq       = bb < nb_valid ? csc_segment_of_block(cuc, n_seq, bb) : 0;
    const int64_t bis   = bb < nb_valid ? (int64_t)bb - cuc[seq] : 0;
    const int64_t begin = cu[seq], end = cu[seq + 1];
    const int64_t tok = begin + bis * ratio;
    const bool valid =
        cuc[seq] >= 0 && begin >= 0 && end >= begin && end <= total_tokens && tok >= begin &&
        tok + ratio <= total_tokens &&
        (bb >= nb_valid || (tok + ratio <= end && (int64_t)cuc[seq + 1] - cuc[seq] == (end - begin) / ratio));
    if (!valid) {
        if (!Backward && active && group == 0) out[(size_t)bb * d + j] = 0;
        return;
    }
    const int tok0 = (int)tok;
    const int win  = (Overlap ? 2 : 1) * ratio;
    __shared__ float temp[Chunk * Dim], part_max[Groups * Dim];
    __shared__ float den_shared[Dim], sum_shared[Dim];
    float mx = -INFINITY;
    for (int64_t k = group; k < win; k += Groups) {
        const float s =
            active ? csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, (int)k, bis, Overlap).s : -INFINITY;
        mx = fmaxf(mx, s);
    }
    part_max[group * Dim + lane] = mx;
    __syncthreads();
#pragma unroll
    for (int g = 0; g < Groups; ++g) mx = fmaxf(mx, part_max[g * Dim + lane]);
    float den = 0.f;
    for (int64_t tile = 0; tile < win; tile += Chunk) {
#pragma unroll
        for (int z = 0; z < Items; ++z) {
            const int local = group * Items + z;
            const int64_t k = tile + local;
            float s         = -INFINITY;
            if (active && k < win) s = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, (int)k, bis, Overlap).s;
            temp[local * Dim + lane] = expf(s - mx);
        }
        __syncthreads();
        if (group == 0) {
#pragma unroll 8
            for (int k = 0; k < min((int64_t)Chunk, win - tile); ++k) den += temp[k * Dim + lane];
        }
        __syncthreads();
    }
    if (group == 0) den_shared[lane] = den;
    __syncthreads();
    den           = den_shared[lane];
    const float g = Backward && active ? csc_bf16_to_float(go[(size_t)bb * d + j]) : 0.f;
    float sum     = 0.f;
    for (int64_t tile = 0; tile < win; tile += Chunk) {
#pragma unroll
        for (int z = 0; z < Items; ++z) {
            const int local = group * Items + z;
            const int64_t k = tile + local;
            CscWin v        = {-INFINITY, 0.f, -1};
            if (active && k < win) v = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, (int)k, bis, Overlap);
            const float p            = expf(v.s - mx) / den;
            temp[local * Dim + lane] = csc_fmul_rn(Backward ? csc_fmul_rn(g, v.u) : v.u, p);
        }
        __syncthreads();
        if (group == 0) {
#pragma unroll 8
            for (int k = 0; k < min((int64_t)Chunk, win - tile); ++k) sum += temp[k * Dim + lane];
        }
        __syncthreads();
    }
    if (group == 0) {
        if (Backward)
            sum_shared[lane] = sum;
        else if (active)
            out[(size_t)bb * d + j] = csc_float_to_bf16_rne(sum);
    }
    if constexpr (Backward) {
        __syncthreads();
        sum = sum_shared[lane];
        if (active)
            for (int64_t k = group; k < win; k += Groups) {
                const CscWin v = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, (int)k, bis, Overlap);
                if (v.off < 0) continue;
                const float p  = expf(v.s - mx) / den;
                const float dp = csc_fmul_rn(g, v.u);
                const float ds = csc_ffma_rn(p, -sum, csc_fmul_rn(dp, p));
                gkv[v.off]     = csc_float_to_bf16_rne(g * p);
                gs[v.off]      = csc_float_to_bf16_rne(ds);
                const int col  = Overlap && k >= ratio ? d + j : j;
                const int row  = Overlap && k >= ratio ? (int)(k - ratio) : (int)k;
                atomicAdd(gape + (size_t)row * W + col, ds);
            }
    }
}

// The cooperative path favors scarce output rows. Filled large-D grids
// can favor the original serial window kernel: more resident waves, no LDS.

// Experiment: fixed r128 cooperative chunks; only partials use LDS.
template<int Dim, bool Backward, bool Overlap, bool WindowMajor = false>
__global__ void __launch_bounds__(256)
    csa_compressor_r128_kernel(const csc_bf16* __restrict__ kv, const csc_bf16* __restrict__ score,
                               const float* __restrict__ ape, const int* __restrict__ cu, const int* __restrict__ cuc,
                               csc_bf16* __restrict__ out, const csc_bf16* __restrict__ go, csc_bf16* __restrict__ gkv,
                               csc_bf16* __restrict__ gs, float* __restrict__ gape, int nb_total, int n_seq, int d,
                               int total_tokens) {
    constexpr int Ratio = 128, WFactor = Overlap ? 2 : 1;
    constexpr int Groups = 256 / Dim, Items = Ratio * WFactor / Groups;
    const int lane = threadIdx.x % Dim, group = threadIdx.x / Dim;
    // Adjacent forward CTAs visit adjacent dimensions of the same window.
    // Keep the backward traversal (and atomic accumulation order) unchanged.
    const int bb = WindowMajor && !Backward ? blockIdx.y : blockIdx.x;
    const int j = (WindowMajor && !Backward ? blockIdx.x : blockIdx.y) * Dim + lane, W = WFactor * d;
    const bool active = j < d;
    if constexpr (Backward) {
        if (group == 0 && active)
            csc_zero_unconsumed(gkv, gs, cu, cuc, bb, nb_total, n_seq, Ratio, d, WFactor, j, total_tokens);
    }
    const int nb_valid = max(0, min(nb_total, cuc[n_seq]));
    if (bb >= nb_total || (Backward && bb >= nb_valid)) return;
    const int seq       = bb < nb_valid ? csc_segment_of_block(cuc, n_seq, bb) : 0;
    const int64_t bis   = bb < nb_valid ? (int64_t)bb - cuc[seq] : 0;
    const int64_t begin = cu[seq], end = cu[seq + 1], tok = begin + bis * Ratio;
    const bool valid =
        cuc[seq] >= 0 && begin >= 0 && end >= begin && end <= total_tokens && tok >= begin &&
        tok + Ratio <= total_tokens &&
        (bb >= nb_valid || (tok + Ratio <= end && (int64_t)cuc[seq + 1] - cuc[seq] == (end - begin) / Ratio));
    if (!valid) {
        if (!Backward && group == 0 && active) out[(size_t)bb * d + j] = 0;
        return;
    }
    const int tok0 = (int)tok;
    __shared__ float maxima[256], denoms[256], sums[256];
    float p[Items], mx = -INFINITY;
#pragma unroll
    for (int z = 0; z < Items; ++z) {
        const int k = group * Items + z;
        p[z]        = active ? csc_window_elem(kv, score, ape, tok0, Ratio, d, W, j, k, bis, Overlap).s : -INFINITY;
        mx          = fmaxf(mx, p[z]);
    }
    maxima[threadIdx.x] = mx;
    __syncthreads();
#pragma unroll
    for (int q = 0; q < Groups; ++q) mx = fmaxf(mx, maxima[q * Dim + lane]);
    float den = 0.f;
#pragma unroll
    for (int z = 0; z < Items; ++z) {
        const float delta = p[z] - mx;
        // Native exp flushes FP32 subnormals on gfx938. Keep precise tails,
        // including large-KV cases where a tiny probability has a visible result.
        p[z]  = (delta > -80.f || delta == -INFINITY) ? __expf(delta) : expf(delta);
        den  += p[z];
    }
    denoms[threadIdx.x] = den;
    __syncthreads();
    den = 0.f;
#pragma unroll
    for (int q = 0; q < Groups; ++q) den += denoms[q * Dim + lane];
    const float g = Backward && active ? csc_bf16_to_float(go[(size_t)bb * d + j]) : 0.f;
    float sum     = 0.f;
#pragma unroll
    for (int z = 0; z < Items; ++z) {
        const int k    = group * Items + z;
        p[z]          /= den;
        const float u  = active ? csc_window_elem(kv, score, ape, tok0, Ratio, d, W, j, k, bis, Overlap).u : 0.f;
        sum           += csc_fmul_rn(Backward ? csc_fmul_rn(g, u) : u, p[z]);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    sum = 0.f;
#pragma unroll
    for (int q = 0; q < Groups; ++q) sum += sums[q * Dim + lane];
    if constexpr (Backward) {
        if (active) {
#pragma unroll
            for (int z = 0; z < Items; ++z) {
                const int k = group * Items + z;
                if (!(Overlap && k < Ratio && bis == 0)) {
                    const CscWin v = csc_window_elem(kv, score, ape, tok0, Ratio, d, W, j, k, bis, Overlap);
                    const float dp = csc_fmul_rn(g, v.u);
                    const float ds = csc_ffma_rn(p[z], -sum, csc_fmul_rn(dp, p[z]));
                    gkv[v.off]     = csc_float_to_bf16_rne(g * p[z]);
                    gs[v.off]      = csc_float_to_bf16_rne(ds);
                    const int col  = Overlap && k >= Ratio ? d + j : j;
                    atomicAdd(gape + (size_t)(k % Ratio) * W + col, ds);
                }
            }
        }
    } else if (group == 0 && active)
        out[(size_t)bb * d + j] = csc_float_to_bf16_rne(sum);
}

template<int Dim, bool Backward, bool WindowMajor, bool OVERLAP>
static void csc_launch_r128_dim(hipStream_t stream, const void* kv, const void* score, const float* ape, const int* cu,
                                const int* cuc, void* out, const void* go, void* gkv, void* gs, float* gape,
                                int nb_total, int n_seq, int d, int coff, int total_tokens) {
    const dim3 grid =
        WindowMajor && !Backward ? dim3((d + Dim - 1) / Dim, nb_total) : dim3(nb_total, (d + Dim - 1) / Dim);
#define CSC_R128(O)                                                                                                    \
    hipLaunchKernelGGL((csa_compressor_r128_kernel<Dim, Backward, O, WindowMajor>), grid, dim3(256), 0, stream,        \
                       (const csc_bf16*)kv, (const csc_bf16*)score, ape, cu, cuc, (csc_bf16*)out, (const csc_bf16*)go, \
                       (csc_bf16*)gkv, (csc_bf16*)gs, gape, nb_total, n_seq, d, total_tokens)
    if constexpr (OVERLAP) {
        CSC_R128(true);
    } else {
        CSC_R128(false);
    }
#undef CSC_R128
}

// The wider backward tile avoids excessive CTA/atomic pressure on filled D512 grids.

// Coarsen independent output rows, retaining the per-row arithmetic exactly.
template<int Ratio, bool Overlap, int Rows>
__global__ void __launch_bounds__(128)
    csa_compressor_short_bwd_kernel(const csc_bf16* __restrict__ kv, const csc_bf16* __restrict__ score,
                                    const float* __restrict__ ape, const int* __restrict__ cu,
                                    const int* __restrict__ cuc, const csc_bf16* __restrict__ go,
                                    csc_bf16* __restrict__ gkv, csc_bf16* __restrict__ gs, float* __restrict__ gape,
                                    int nb_total, int n_seq, int d, int total_tokens) {
    constexpr int Win = Ratio * (Overlap ? 2 : 1);
    const int j = blockIdx.y * 128 + threadIdx.x, W = (Overlap ? 2 : 1) * d;
    if (j >= d) return;
    float da[Win]      = {};
    const int nb_valid = max(0, min(nb_total, cuc[n_seq]));
#pragma unroll 1
    for (int rr = 0; rr < Rows; ++rr) {
        const int bb = blockIdx.x * Rows + rr;
        if (bb >= nb_total) break;
        csc_zero_unconsumed(gkv, gs, cu, cuc, bb, nb_total, n_seq, Ratio, d, Overlap ? 2 : 1, j, total_tokens);
        if (bb >= nb_valid) continue;
        const int seq       = csc_segment_of_block(cuc, n_seq, bb);
        const int64_t bis   = (int64_t)bb - cuc[seq];
        const int64_t begin = cu[seq], end = cu[seq + 1], tok = begin + bis * Ratio;
        const bool valid = cuc[seq] >= 0 && begin >= 0 && end >= begin && end <= total_tokens && tok >= begin &&
                           tok + Ratio <= end && (int64_t)cuc[seq + 1] - cuc[seq] == (end - begin) / Ratio;
        if (!valid) continue;
        const int tok0 = (int)tok;
        float p[Win], u[Win], mx = -INFINITY;
#pragma unroll
        for (int k = 0; k < Win; ++k) {
            const CscWin v = csc_window_elem(kv, score, ape, tok0, Ratio, d, W, j, k, bis, Overlap);
            p[k]           = v.s;
            u[k]           = v.u;
            mx             = fmaxf(mx, v.s);
        }
        float den = 0.f;
#pragma unroll
        for (int k = 0; k < Win; ++k) {
            p[k]  = expf(p[k] - mx);
            den  += p[k];
        }
        if (d == 1) {
            float terms[Win];
#pragma unroll
            for (int k = 0; k < Win; ++k) terms[k] = p[k];
            den = csc_contiguous_sum(terms);
        }
        const float g = csc_bf16_to_float(go[(size_t)bb * d + j]);
        float sum     = 0.f, terms[Win];
#pragma unroll
        for (int k = 0; k < Win; ++k) {
            p[k]     /= den;
            terms[k]  = csc_fmul_rn(csc_fmul_rn(g, u[k]), p[k]);
            sum      += terms[k];
        }
        if (d == 1) sum = csc_contiguous_sum(terms);
#pragma unroll
        for (int k = 0; k < Win; ++k) {
            if (!(Overlap && k < Ratio && bis == 0)) {
                const int col = Overlap && k >= Ratio ? d + j : j;
                const int row = k % Ratio, token = Overlap && k < Ratio ? tok0 - Ratio + k : tok0 + row;
                const size_t off  = (size_t)token * W + col;
                const float ds    = csc_ffma_rn(p[k], -sum, csc_fmul_rn(csc_fmul_rn(g, u[k]), p[k]));
                gkv[off]          = csc_float_to_bf16_rne(g * p[k]);
                gs[off]           = csc_float_to_bf16_rne(ds);
                da[k]            += ds;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < Win; ++k) {
        const int col = Overlap && k >= Ratio ? d + j : j;
        atomicAdd(gape + (size_t)(k % Ratio) * W + col, da[k]);
    }
}

}  // namespace
#include "flash_train/primitive/hygon/csa_compressor_bwd/bf16_csa_compressor_bwd.hpp"

static __global__ void csa_zero_gradients(unsigned char* memory, std::uint64_t bytes) {
    const std::uint64_t index   = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride  = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    const std::uint64_t vectors = bytes / sizeof(uint4);
    for (std::uint64_t i = index; i < vectors; i += stride) {
        reinterpret_cast<uint4*>(memory)[i] = make_uint4(0, 0, 0, 0);
    }
    if (index < bytes % sizeof(uint4)) { memory[vectors * sizeof(uint4) + index] = 0; }
}

namespace ftrain {

void bf16CSACompressorBwdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    hipError_t status;
    if (reinterpret_cast<std::uintptr_t>(memory) % alignof(uint4) == 0) {
        const std::uint64_t needed = (bytes + 4095) / 4096;
        const unsigned blocks      = static_cast<unsigned>(needed < 4096 ? needed : 4096);
        hipLaunchKernelGGL(csa_zero_gradients, dim3(blocks), dim3(256), 0, stream, static_cast<unsigned char*>(memory),
                           bytes);
        status = hipGetLastError();
    } else {
        status = hipMemsetAsync(memory, 0, static_cast<std::size_t>(bytes), stream);
    }
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Workspace zero-fill failed: %s", hipGetErrorString(status));
    }
}

}  // namespace ftrain

namespace ftrain {

template<class Implementation, int CAPACITY, int DIM, bool OVERLAP>
bool launchBf16CSACompressorBwd(FTrainStream stream, const void* kv, const void* score, const float* ape,
                                const int* cu_seqlens, const int* cu_seqlens_comp, const void* grad_compressed,
                                void* grad_kv, void* grad_score, float* grad_ape, int total_comp, int num_sequences,
                                int ratio, int head_dim, int coff, int total_tokens) {
    const int width = coff * head_dim;
    const dim3 grid(total_comp, (head_dim + 255) / 256);
    if constexpr (Implementation::kSerial) {
hipLaunchKernelGGL(csa_compressor_bwd_kernel, grid, 256, 0, stream, (const csc_bf16*)kv,
                               (const csc_bf16*)score, ape, cu_seqlens, cu_seqlens_comp,
                               (const csc_bf16*)grad_compressed, (csc_bf16*)grad_kv, (csc_bf16*)grad_score, grad_ape,
                               total_comp, num_sequences, ratio, head_dim, coff, width, total_tokens);
}else if constexpr (Implementation::kCached) {
    csc_launch_cached<CAPACITY, true, OVERLAP>(stream, kv, score, ape, cu_seqlens, cu_seqlens_comp, nullptr,
                                               grad_compressed, grad_kv, grad_score, grad_ape, total_comp,
                                               num_sequences, ratio, head_dim, coff, total_tokens);
}else if constexpr (Implementation::kTiled) {
    csc_launch_tiled<CAPACITY, DIM, true, OVERLAP>(stream, kv, score, ape, cu_seqlens, cu_seqlens_comp, nullptr,
                                                   grad_compressed, grad_kv, grad_score, grad_ape, total_comp,
                                                   num_sequences, ratio, head_dim, coff, total_tokens);
}else if constexpr (Implementation::kR128) {
    csc_launch_r128_dim<16, true, false, OVERLAP>(stream, kv, score, ape, cu_seqlens, cu_seqlens_comp, nullptr,
                                                  grad_compressed, grad_kv, grad_score, grad_ape, total_comp,
                                                  num_sequences, head_dim, coff, total_tokens);
}else if constexpr (Implementation::kR128Wide) {
    csc_launch_r128_dim<32, true, false, OVERLAP>(stream, kv, score, ape, cu_seqlens, cu_seqlens_comp, nullptr,
                                                  grad_compressed, grad_kv, grad_score, grad_ape, total_comp,
                                                  num_sequences, head_dim, coff, total_tokens);
}else if constexpr (Implementation::kShortRows1) {
    if constexpr (OVERLAP) {
        hipLaunchKernelGGL((csa_compressor_short_bwd_kernel<4, true, 1>),
                           dim3((total_comp + 1 - 1) / 1, (head_dim + 127) / 128), 128, 0, stream, (const csc_bf16*)kv,
                           (const csc_bf16*)score, ape, cu_seqlens, cu_seqlens_comp, (const csc_bf16*)grad_compressed,
                           (csc_bf16*)grad_kv, (csc_bf16*)grad_score, grad_ape, total_comp, num_sequences, head_dim,
                           total_tokens);
    } else {
        hipLaunchKernelGGL((csa_compressor_short_bwd_kernel<4, false, 1>),
                           dim3((total_comp + 1 - 1) / 1, (head_dim + 127) / 128), 128, 0, stream, (const csc_bf16*)kv,
                           (const csc_bf16*)score, ape, cu_seqlens, cu_seqlens_comp, (const csc_bf16*)grad_compressed,
                           (csc_bf16*)grad_kv, (csc_bf16*)grad_score, grad_ape, total_comp, num_sequences, head_dim,
                           total_tokens);
    }
}else if constexpr (Implementation::kShortRows4) {
    if constexpr (OVERLAP) {
        hipLaunchKernelGGL((csa_compressor_short_bwd_kernel<4, true, 4>),
                           dim3((total_comp + 4 - 1) / 4, (head_dim + 127) / 128), 128, 0, stream, (const csc_bf16*)kv,
                           (const csc_bf16*)score, ape, cu_seqlens, cu_seqlens_comp, (const csc_bf16*)grad_compressed,
                           (csc_bf16*)grad_kv, (csc_bf16*)grad_score, grad_ape, total_comp, num_sequences, head_dim,
                           total_tokens);
    } else {
        hipLaunchKernelGGL((csa_compressor_short_bwd_kernel<4, false, 4>),
                           dim3((total_comp + 4 - 1) / 4, (head_dim + 127) / 128), 128, 0, stream, (const csc_bf16*)kv,
                           (const csc_bf16*)score, ape, cu_seqlens, cu_seqlens_comp, (const csc_bf16*)grad_compressed,
                           (csc_bf16*)grad_kv, (csc_bf16*)grad_score, grad_ape, total_comp, num_sequences, head_dim,
                           total_tokens);
    }
}
    const auto status = hipGetLastError();
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "CSA launch failed: %s", hipGetErrorString(status));
    }
    return true;
}

// Explicit fixed launch instances used by the C++ configuration code.
#define FTRAIN_INSTANTIATE_LAUNCH(Implementation, CAPACITY, DIM, OVERLAP)                                          \
    template bool launchBf16CSACompressorBwd<Implementation, CAPACITY, DIM, OVERLAP>(                              \
        FTrainStream stream, const void* kv, const void* score, const float* ape, const int* cu_seqlens,           \
        const int* cu_seqlens_comp, const void* grad_compressed, void* grad_kv, void* grad_score, float* grad_ape, \
        int total_comp, int num_sequences, int ratio, int head_dim, int coff, int total_tokens);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwd, 1, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwd, 1, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdCached, 4, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdCached, 4, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdCachedCapacity16, 16, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdCachedCapacity16, 16, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdCachedCapacity32, 32, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdCachedCapacity32, 32, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdCachedCapacity64, 64, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdCachedCapacity64, 64, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdCachedCapacity8, 8, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdCachedCapacity8, 8, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdR128, 1, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdR128, 1, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdR128Wide, 1, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdR128Wide, 1, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdShortRows1, 1, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdShortRows1, 1, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdShortRows4, 1, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdShortRows4, 1, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiled, 64, 32, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiled, 64, 32, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity128, 128, 32, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity128, 128, 32, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity160, 160, 32, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity160, 160, 32, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity256, 256, 32, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity256, 256, 32, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity256Dim16, 256, 16, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity256Dim16, 256, 16, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity256Dim4, 256, 4, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity256Dim4, 256, 4, true);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity256Dim8, 256, 8, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16CSACompressorBwdTiledCapacity256Dim8, 256, 8, true);
#undef FTRAIN_INSTANTIATE_LAUNCH

}  // namespace ftrain
