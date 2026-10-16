// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <math.h>

#include <hip/hip_runtime.h>

#include "flash_train/primitive/hygon/csa_compressor_fwd/bf16_csa_compressor_fwd.hpp"

namespace hip_dsa_numeric {

__device__ __forceinline__ static uint16_t bf16_rne(float value) { return __builtin_bit_cast(uint16_t, (__bf16)value); }

}  // namespace hip_dsa_numeric

// Adapted from fused_dsa/hip_dsa/kernels/csa_compressor.hip.

typedef uint16_t csc_bf16;

__device__ __forceinline__ static float csc_bf16_to_float(uint16_t v) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.u = (uint32_t)v << 16;
    return conv.f;
}

__device__ __forceinline__ static uint16_t csc_float_to_bf16_rne(float f) { return hip_dsa_numeric::bf16_rne(f); }

__device__ __forceinline__ static float csc_fmul_rn(float a, float b) {
    float r;
    asm volatile("v_mul_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}

__device__ __forceinline__ static int csc_segment_of_block(const int* __restrict__ cuc, int n_seq, int bb) {
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

struct CscWin {
    float s;  // score + ape
    float u;  // kv
    int off;  // flat offset of the consumed element (or -1)
};

__device__ __forceinline__ static CscWin csc_window_elem(const csc_bf16* __restrict__ kv,
                                                         const csc_bf16* __restrict__ score,
                                                         const float* __restrict__ ape, int tok0, int ratio, int d,
                                                         int W, int j, int k, int bis, bool coff2) {
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
        w.off             = (coff2 ? (tok0 + k - ratio) : (tok0 + k)) * W + col;
        const int ape_row = (k >= ratio) ? (k - ratio) : k;
        w.s               = csc_bf16_to_float(score[w.off]) + ape[ape_row * W + col];
        w.u               = csc_bf16_to_float(kv[w.off]);
    }
    return w;
}

static __global__ void __launch_bounds__(256)
    csa_compressor_fwd_kernel(const csc_bf16* __restrict__ kv,     // (total_tokens * W) bf16
                              const csc_bf16* __restrict__ score,  // (total_tokens * W) bf16
                              const float* __restrict__ ape,       // (ratio * W) fp32
                              const int* __restrict__ cu,          // (n_seq + 1)
                              const int* __restrict__ cuc,         // (n_seq + 1)
                              csc_bf16* __restrict__ out,          // (nb_total * head_dim) bf16
                              int nb_total, int n_seq, int ratio, int d, int coff, int W, int total_tokens) {
    const int bb = blockIdx.x;
    const int j  = blockIdx.y * blockDim.x + threadIdx.x;
    if (bb >= nb_total || j >= d) return;

    const int nb_valid = max(0, min(nb_total, cuc[n_seq]));
    int seq_idx        = 0;
    int64_t bis        = 0;
    if (bb < nb_valid) {
        seq_idx = csc_segment_of_block(cuc, n_seq, bb);
        bis     = (int64_t)bb - cuc[seq_idx];
    }
    const int64_t begin = cu[seq_idx], end = cu[seq_idx + 1];
    const int64_t tok = begin + (int64_t)bis * ratio;
    const bool window_valid =
        cuc[seq_idx] >= 0 && begin >= 0 && end >= begin && end <= total_tokens && tok >= begin &&
        tok + ratio <= total_tokens &&
        (bb >= nb_valid || (tok + ratio <= end && (int64_t)cuc[seq_idx + 1] - cuc[seq_idx] == (end - begin) / ratio));
    const int tok0 = window_valid ? (int)tok : 0;
    if (!window_valid) {
        out[(size_t)bb * d + j] = csc_float_to_bf16_rne(0.0f);
        return;
    }
    const int win    = (coff == 2) ? 2 * ratio : ratio;
    const bool coff2 = (coff == 2);

    // Serial max over the window (same order as the eager softmax).
    float mx = -INFINITY;
    for (int k = 0; k < win; k++) {
        const CscWin w = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, k, bis, coff2);
        if (w.s > mx) mx = w.s;
    }
    // Serial exp-sum (den).
    float den = 0.0f;
    for (int k = 0; k < win; k++) {
        const CscWin w  = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, k, bis, coff2);
        den            += expf(w.s - mx);
    }
    // Serial gated weighted sum: acc += mul_rn(kv, exp/den).
    float acc = 0.0f;
    for (int k = 0; k < win; k++) {
        const CscWin w  = csc_window_elem(kv, score, ape, tok0, ratio, d, W, j, k, bis, coff2);
        acc            += csc_fmul_rn(w.u, expf(w.s - mx) / den);
    }
    out[(size_t)bb * d + j] = (csc_bf16)csc_float_to_bf16_rne(acc);
}

namespace ftrain {
namespace {
bool launch_csa_compressor_fwd(hipStream_t stream, const void* kv, const void* score, const float* ape, const int* cu,
                               const int* cuc, void* out, int nb_total, int n_seq, int ratio, int head_dim, int coff,
                               int total_tokens);

static bool launch_csa_compressor_fwd(hipStream_t stream, const void* kv, const void* score, const float* ape,
                                      const int* cu, const int* cuc, void* out, int nb_total, int n_seq, int ratio,
                                      int head_dim, int coff, int total_tokens) {
    if (nb_total <= 0 || n_seq <= 0 || head_dim <= 0 || ratio <= 0 || coff < 1) return false;
    const int W       = coff * head_dim;
    const int threads = 256;
    dim3 grid((unsigned)nb_total, (unsigned)((head_dim + threads - 1) / threads));
    hipLaunchKernelGGL(csa_compressor_fwd_kernel, grid, threads, 0, stream, (const csc_bf16*)kv, (const csc_bf16*)score,
                       ape, cu, cuc, (csc_bf16*)out, nb_total, n_seq, ratio, head_dim, coff, W, total_tokens);
    return hipGetLastError() == hipSuccess;
}
}  // namespace

bool launchBf16CSACompressorFwd(FTrainStream stream, const void* kv, const void* score, const float* ape,
                                const int* cu_seqlens, const int* cu_seqlens_comp, void* compressed, int total_comp,
                                int num_sequences, int ratio, int head_dim, int coff, int total_tokens) {
    return launch_csa_compressor_fwd(stream, kv, score, ape, cu_seqlens, cu_seqlens_comp, compressed, total_comp,
                                     num_sequences, ratio, head_dim, coff, total_tokens);
}

}  // namespace ftrain
