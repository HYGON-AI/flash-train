// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <math.h>

#include <hip/hip_runtime.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/csa_compressor_bwd/bf16_csa_compressor_bwd.hpp"

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

__device__ __forceinline__ static float csc_ffma_rn(float a, float b, float c) {
    float r;
    asm volatile("v_fma_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
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
    csa_compressor_bwd_kernel(const csc_bf16* __restrict__ kv, const csc_bf16* __restrict__ score,
                              const float* __restrict__ ape, const int* __restrict__ cu, const int* __restrict__ cuc,
                              const csc_bf16* __restrict__ go,  // (nb_total * head_dim) bf16
                              csc_bf16* __restrict__ gkv,       // (total_tokens * W) bf16 (pre-zeroed)
                              csc_bf16* __restrict__ gs,        // (total_tokens * W) bf16 (pre-zeroed)
                              float* __restrict__ gape,         // (ratio * W) fp32 (pre-zeroed)
                              int nb_total, int n_seq, int ratio, int d, int coff, int W, int total_tokens) {
    const int bb = blockIdx.x;
    const int j  = blockIdx.y * blockDim.x + threadIdx.x;
    if (bb >= nb_total || j >= d) return;

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

// Vector stores match the original Torch fill kernels; avoid the slower byte-memset path.
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
namespace {

static bool launch_csa_compressor_bwd(hipStream_t stream, const void* kv, const void* score, const float* ape,
                                      const int* cu, const int* cuc, const void* go, void* gkv, void* gs, float* gape,
                                      int nb_total, int n_seq, int ratio, int head_dim, int coff, int total_tokens) {
    if (nb_total <= 0 || n_seq <= 0 || head_dim <= 0 || ratio <= 0 || coff < 1) return false;
    const int W       = coff * head_dim;
    const int threads = 256;
    dim3 grid((unsigned)nb_total, (unsigned)((head_dim + threads - 1) / threads));
    hipLaunchKernelGGL(csa_compressor_bwd_kernel, grid, threads, 0, stream, (const csc_bf16*)kv, (const csc_bf16*)score,
                       ape, cu, cuc, (const csc_bf16*)go, (csc_bf16*)gkv, (csc_bf16*)gs, gape, nb_total, n_seq, ratio,
                       head_dim, coff, W, total_tokens);
    return hipGetLastError() == hipSuccess;
}
}  // namespace

bool launchBf16CSACompressorBwd(FTrainStream stream, const void* kv, const void* score, const float* ape,
                                const int* cu_seqlens, const int* cu_seqlens_comp, const void* grad_compressed,
                                void* grad_kv, void* grad_score, float* grad_ape, int total_comp, int num_sequences,
                                int ratio, int head_dim, int coff, int total_tokens) {
    return launch_csa_compressor_bwd(stream, kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_compressed, grad_kv,
                                     grad_score, grad_ape, total_comp, num_sequences, ratio, head_dim, coff,
                                     total_tokens);
}

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
