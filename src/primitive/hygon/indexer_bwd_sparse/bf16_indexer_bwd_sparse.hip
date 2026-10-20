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
#include "flash_train/primitive/hygon/indexer_bwd_sparse/bf16_indexer_bwd_sparse.hpp"

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

static inline bool dense_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx928") != nullptr || strstr(prop.gcnArchName, "gfx936") != nullptr ||
           strstr(prop.gcnArchName, "gfx938") != nullptr;
}

// Adapted from fused_dsa/hip_dsa/kernels/indexer_bwd.hip.

/**
 * indexer_bwd.hip
 * HIP kernels for DSA indexer backward (sparse and dense).
 *
 * Fused MMAC compute implementations (mixed inputs are normalized by bindings):
 *   - indexer_bwd_sparse: one kernel computes provided predict -> score gradient -> dq/dw/dk.
 *   - indexer_bwd_dense: fused P1/P2/dq/dk pipeline (see dense_mmac.hip).
 * Non-fused kernels were removed; the bindings error out on unsupported shapes.
 */

typedef uint16_t bf16;

#define MAX_H 128
#define MAX_D 128      // indexer head_dim for DSv4
#define IDXB_CHUNK 16  // 16-key chunk: half the K arena -> more CTAs/CU
#define IDXB_THREADS 256

// ---------------------------------------------------------------------------
// bf16 conversion + vectorized load helpers
// ---------------------------------------------------------------------------

__device__ static inline float bf16_to_float(uint16_t v) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.u = (uint32_t)v << 16;
    return conv.f;
}

__device__ __forceinline__ static uint16_t float_to_bf16(float f) { return hip_dsa_numeric::bf16_rne(f); }

#if defined(__HIP_PLATFORM_AMD__) && __has_builtin(__builtin_amdgcn_flat_atomic_fadd_f32)

#else

#endif

// FP32 scratch may retain MMAC column order; public dK is restored at cast.
template<bool SWIZZLED>
__device__ __forceinline__ static float4 idxb_cast_load(const float4* src, size_t i) {
#if defined(__gfx936__) || defined(__gfx938__)
    if constexpr (SWIZZLED) {
        const float* p = (const float*)src + (i / 4) * 16 + (i % 4);
        return make_float4(p[0], p[4], p[8], p[12]);
    }
#endif
    return src[i];
}

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;
typedef __hip_bfloat16 hbf16;

#    if defined(__gfx936__) || defined(__gfx938__)
using idxb_floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
#    endif

// MMAC accumulator lanes g=0..3 own columns {g,g+4,g+8,g+12}.  Transpose
// four participating lanes into contiguous four-float groups before the
// native x4 global atomic.  gfx928 retains the scalar path.
__device__ __forceinline__ static void idxb_atomic_add_mmac_row(float* row16, int g, float x0, float x1, float x2,
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
    idxb_floatx4 v;
    v[0] = (g & 2) ? q2 : y0;
    v[1] = (g & 2) ? q3 : y1;
    v[2] = (g & 2) ? y2 : q0;
    v[3] = (g & 2) ? y3 : q1;
#        if defined(__gfx938__)
    __builtin_hcu_global_atomic_fadd_f32_x4(row16 + 4 * g, v);
#        else
    __builtin_amdgcn_global_atomic_fadd_v4f32((idxb_floatx4*)(row16 + 4 * g), v);
#        endif
#    else
    float* p = row16 + g;
    atomicAdd(&p[0], x0);
    atomicAdd(&p[4], x1);
    atomicAdd(&p[8], x2);
    atomicAdd(&p[12], x3);
#    endif
}

template<bool SWIZZLED>
__device__ __forceinline__ static void idxb_atomic_add_sparse(float* row16, int g, float x0, float x1, float x2,
                                                              float x3) {
#    if defined(__gfx936__) || defined(__gfx938__)
    if constexpr (SWIZZLED) {
        idxb_floatx4 v = {x0, x1, x2, x3};
#        if defined(__gfx938__)
        __builtin_hcu_global_atomic_fadd_f32_x4(row16 + 4 * g, v);
#        else
        __builtin_amdgcn_global_atomic_fadd_v4f32((idxb_floatx4*)(row16 + 4 * g), v);
#        endif
        return;
    }
#    endif
    idxb_atomic_add_mmac_row(row16, g, x0, x1, x2, x3);
}

// Exact HIP BF16 RNE, including NaN/Inf payload handling, without divergent branches.
__device__ __forceinline__ static hbf16 idxb_rne(float f) { return __ushort_as_bfloat16(hip_dsa_numeric::bf16_rne(f)); }

// The sparse API supplies probabilities aligned with its top-k slots.
// Match cuDNN ScoreGradSm90: 128 thread-local sums followed by the same
// ascending serial reduction. Keep predict read-only and overwrite target
// only after every thread has finished reading it.
template<bool SCORE_GLOBAL = false>
__device__ __forceinline__ static void indexer_sparse_score_grad(float* trow, const float* predict, float* score,
                                                                 float* red, int topk, float c) {
    // Match cuDNN SM90's 128-thread score_grad: index_score already
    // contains the forward probabilities. No second score/softmax pass.
    const float eps = expf(-100.f);
    if (threadIdx.x < 128) {
        float local_sum = 0.f;
        for (int pos = threadIdx.x; pos < topk; pos += 128) {
            const float target_eff  = fmaxf(trow[pos], eps);
            const float pp          = predict[pos];
            const float gate        = pp >= eps ? 1.f : 0.f;
            local_sum              += -target_eff * gate * c;
        }
        red[threadIdx.x] = local_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.f;
#    pragma unroll
        for (int i = 0; i < 128; i++) sum += red[i];
        red[0] = sum;
    }
    __syncthreads();
    const float sum_grad = red[0];
    if constexpr (SCORE_GLOBAL) {
        float values[8];
#    pragma unroll
        for (int j = 0; j < 8; j++) {
            const int pos = threadIdx.x + j * IDXB_THREADS;
            if (pos < topk) {
                const float target_eff = fmaxf(trow[pos], eps);
                const float pp         = predict[pos];
                const float gate       = pp >= eps ? 1.f : 0.f;
                const float gi         = -target_eff * gate * c;
                values[j]              = gi - pp * sum_grad;
            }
        }
        __syncthreads();
#    pragma unroll
        for (int j = 0; j < 8; j++) {
            const int pos = threadIdx.x + j * IDXB_THREADS;
            if (pos < topk) trow[pos] = values[j];
        }
        __syncthreads();
    } else {
        for (int pos = threadIdx.x; pos < topk; pos += IDXB_THREADS) {
            const float target_eff = fmaxf(trow[pos], eps);
            const float pp         = predict[pos];
            const float gate       = pp >= eps ? 1.f : 0.f;
            const float gi         = -target_eff * gate * c;
            score[pos]             = gi - pp * sum_grad;
        }
        __syncthreads();
        for (int pos = threadIdx.x; pos < topk; pos += IDXB_THREADS) trow[pos] = score[pos];
        __syncthreads();
    }
}

// Keep host scalar instantiations unchanged; the tensor specialization reads
// the current device value on every launch, including graph replay.
__device__ __forceinline__ static float idxb_grad_scale(float value) { return value; }
__device__ __forceinline__ static float idxb_grad_scale(const float* value) { return *value; }

// H17..32/D128 needs one-block launch bounds to avoid register spills.
// The paired layout alone still spills under the old two-block bound.
// NW==4 (the D128 fast path) instead benefits from two blocks: hipcc budgets
// 65536/(256*2)=128 VGPRs, which lands 132->128, spills only 16 B/workitem and
// raises resident CTAs from ~4.7 to ~5.55 (+20% waves, -4% kernel time);
// three blocks budget 85 VGPRs and cost +250% from spills.
template<int D, int NW, bool FULL_TILES = false, bool SCORE_GLOBAL = false, bool IDX_GLOBAL = false,
         typename GradScale = float>
__global__ static void __launch_bounds__(IDXB_THREADS,
                                         ((D == 128 && (NW == 2 || (NW == 4 && SCORE_GLOBAL && IDX_GLOBAL))) ? 1 : 2))
    indexer_bwd_sparse_mmac_kernel(const bf16* __restrict__ q,         // (B, Sq, Hq, D)
                                   const bf16* __restrict__ w,         // (B, Sq, Hq)
                                   const bf16* __restrict__ k,         // (B, Sk, D)
                                   float* __restrict__ target,         // (B, Sq, topk) IN-PLACE: overwritten
                                                                       // with grad_signal (cuDNN contract)
                                   const float* __restrict__ predict,  // (B, Sq, topk), read-only probabilities
                                   const int* __restrict__ idxs,       // (B, Sq, topk)
                                   bf16* __restrict__ dq,              // (B, Sq, Hq, D) bf16
                                   bf16* __restrict__ dw,              // (B, Sq, Hq) bf16
                                   float* __restrict__ dk_float,       // (B, Sk, D) f32 (zeroed by caller)
                                   bf16* __restrict__ dk,              // (B, Sk, D) bf16 output
                                   int* __restrict__ done_count,       // zeroed by caller; last-block cast
                                   int B, int Sq, int Sk, int Hq, int topk, float sm_scale, float loss_coeff,
                                   GradScale grad_loss, bool topk_indices_global) {
#    if defined(__gfx936__) || defined(__gfx938__)
    constexpr bool FAST = D == 128 && (NW == 1 || NW == 2 || NW == 4);
#    else
    constexpr bool FAST = false;
#    endif
    // dQ uses the register dS fragment on this measured path, leaving
    // dK as the only reader of mask_lds. Orient that private tile for dK.
    constexpr bool TRANSPOSE_MASK = FAST;
    constexpr int ND              = D / 16;
    // K LDS row stride: D+8 (=136 for D=128) was the measured winner of a
    // stride sweep {D, D+1, D+4, D+8} (18.18ms best at sweep time), despite
    // leaving a 2-way conflict pattern on the 16-row B-tile loads.
    constexpr int KLD    = D + 8;
    constexpr int n_warp = NW;  // warps with data
    // Head waves own score/dQ/dW tiles. All four waves can independently
    // own dK column tiles even when TP leaves only one or two head tiles.
    constexpr int dk_warps = D == 128 ? 4 : NW;
    constexpr int n_dc_w   = (ND + dk_warps - 1) / dk_warps;
    const int row          = blockIdx.x;
    if (row >= B * Sq) return;
    const int b    = row / Sq;
    const int lane = threadIdx.x & 63;
    const int warp = threadIdx.x >> 6;
    const int r0   = warp * 16;
    const int r    = lane & 15;
    const int g    = lane >> 4;
    const int h    = r0 + r;
    // ``grad_scale = loss_coeff / (B * Sq)``: the KL loss is a *mean* over
    // query rows, so the per-row gradient carries the 1/row_count factor.
    // Matches cuDNN's indexer_backward_wrapper, which folds the same divisor
    // into its runtime ``grad_scale`` (see cudnn DSA indexer_backward/api.py).
    const float c         = (loss_coeff / (float)(B * Sq)) * idxb_grad_scale(grad_loss);
    const bool warp_ok    = (warp < n_warp);
    const bool dk_warp_ok = (warp < dk_warps);

    extern __shared__ char smem_raw[];
    hbf16* k_lds = (hbf16*)smem_raw;  // (IDXB_CHUNK, KLD)
    const int* idx_lds =
        IDX_GLOBAL ? idxs + (size_t)row * topk : (const int*)(k_lds + (size_t)IDXB_CHUNK * KLD);  // [topk]
    float* score = SCORE_GLOBAL ? target + (size_t)row * topk : (float*)(idx_lds + topk);         // [topk]
    float* red   = (float*)(k_lds + (size_t)IDXB_CHUNK * KLD) + (IDX_GLOBAL ? 0 : topk) +
                   (SCORE_GLOBAL ? 0 : topk);  // [IDXB_THREADS]
    // Mask tiles: 4 warps x 2 ntiles x (16 heads x 16 keys).  Row stride
    // 20 bf16 (40B) walks 16 distinct LDS bank groups instead of aliasing
    // 4 rows per bank (stride 16 bf16 = 32B rows).
    hbf16* mask_lds = (hbf16*)(red + IDXB_THREADS);                      // [4][ntiles][16*20]
    float* w_lds    = (float*)(mask_lds + 4 * (IDXB_CHUNK / 16) * 320);  // [Hq]

    const int* idx_row = idxs + (size_t)row * topk;
    float* trow        = target + (size_t)row * topk;

    // Bound both passes by the last valid selection, without assuming that
    // invalid keys form a contiguous suffix.  Empty rows perform no MMAC.
    int last_valid = 0;
    // ---- Load idxs / weights; Q is read directly from global (L1-cached) ----
    for (int i = threadIdx.x; i < topk; i += IDXB_THREADS) {
        if constexpr (!IDX_GLOBAL) ((int*)idx_lds)[i] = idx_row[i];
        if constexpr ((NW == 4 || (NW == 2 && FULL_TILES && SCORE_GLOBAL))) {
            if (idx_row[i] >= 0 && idx_row[i] < (topk_indices_global ? B * Sk : Sk))
                last_valid = max(last_valid, i + 1);
        }
        if constexpr (!SCORE_GLOBAL) score[i] = 0.0f;
    }
    if constexpr ((NW == 4 || (NW == 2 && FULL_TILES && SCORE_GLOBAL))) {
#    pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) last_valid = max(last_valid, __shfl_xor(last_valid, offset));
        if (lane == 0) red[warp] = (float)last_valid;
    }
    for (int i = threadIdx.x; i < Hq; i += IDXB_THREADS) w_lds[i] = bf16_to_float(w[row * Hq + i]);
    __syncthreads();

    const float w_h = (h < Hq) ? w_lds[h] : 0.0f;
    // Keep the original tight-register schedule for fewer than four head
    // waves: the variable traversal bound regressed small H=32 workloads.
    // NW counts head waves, not block waves: IDXB_THREADS is always 256 here.
    // All four lane-0 threads wrote red[0..3], including NW=2 (outside warp_ok).
    const int last_key = (NW == 4 || (NW == 2 && FULL_TILES && SCORE_GLOBAL))
                             ? (int)fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]))
                             : topk;
    const int n_chunks = (last_key + IDXB_CHUNK - 1) / IDXB_CHUNK;
    // All waves must consume the bound before score_grad reuses red[0..127].
    if constexpr (NW == 4 || (NW == 2 && FULL_TILES && SCORE_GLOBAL)) { __syncthreads(); }

    // The score-MMA Q operand (h x d fragment, 8B per d-tile) is
    // chunk-invariant: cache it in registers once per block so the per-chunk
    // global re-loads disappear (same idea as the dk qdk cache).
    uint64_t qa64[ND];
    if (warp_ok) {
#    pragma unroll
        for (int kk = 0; kk < ND; kk++) {
            const int qrow  = (r0 + r < Hq) ? (r0 + r) : (Hq - 1);
            const hbf16* qp = (const hbf16*)(q + ((size_t)row * Hq + qrow) * D + kk * 16);
            uint64_t v      = *(const uint64_t*)(qp + (g << 2));
            if (r0 + r >= Hq) v = 0;
            qa64[kk] = v;
        }
    }

    // Keep one LDS K tile. The next tile stays in registers while MMAC
    // consumes the current tile, then commits after all readers finish.
    // This exact mapping is used only by D=128 / four-wave kernels.
    constexpr bool SHARED_INDEX = D == 128 && NW == 4 && FULL_TILES && SCORE_GLOBAL && IDX_GLOBAL && IDXB_CHUNK == 16;
    int current_idx = -1, future_idx = -1;
    if constexpr (SHARED_INDEX) {
        current_idx = r < topk ? idx_lds[r] : -1;
        future_idx  = 16 + r < topk ? idx_lds[16 + r] : -1;
    }
    auto prefetch_k = [&](int ch, uint4* regs, int shared_idx) {
#    pragma unroll
        for (int j = 0; j < IDXB_CHUNK / 16; j++) {
            const int u     = threadIdx.x + j * IDXB_THREADS;
            const int row_k = u / 16, d8 = u % 16, ki = ch * IDXB_CHUNK + row_k;
            int ix;
            if constexpr (SHARED_INDEX)
                ix = __shfl(shared_idx, warp * 4 + lane / 16, 64);
            else
                ix = ki < topk ? idx_lds[ki] : 0;
            // Match cuDNN sparse gather: an invalid slot stages a zero key.
            // Nonzero probabilities at invalid slots must not read key zero.
            regs[j] = (ki < topk && ix >= 0 && ix < (topk_indices_global ? B * Sk : Sk))
                          ? *((const uint4*)(k + ((size_t)(topk_indices_global ? 0 : b) * Sk + ix) * D + d8 * 8))
                          : make_uint4(0, 0, 0, 0);
        }
    };
    auto commit_k = [&](const uint4* regs) {
#    pragma unroll
        for (int j = 0; j < IDXB_CHUNK / 16; j++) {
            const int u                                        = threadIdx.x + j * IDXB_THREADS;
            *((uint4*)(k_lds + (u / 16) * KLD + (u % 16) * 8)) = regs[j];
        }
    };
    indexer_sparse_score_grad<SCORE_GLOBAL>(trow, predict + (size_t)row * topk, score, red, topk, c);

    // ---- Pass 2: grads ----
    DUFragment<accumulator, 16, 16, 16, float> dqacc[ND];
#    pragma unroll
    for (int dc = 0; dc < ND; dc++) du_fill_fragment(dqacc[dc], 0.0f);
    float dwacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // The dk Q operand (matrix_b, h x d) is chunk-invariant: cache each
    // lane's 4 strided elements for all head groups / d-tiles once per block
    // instead of re-loading them in every chunk (measured: the per-chunk
    // scalar strided Q loads dominate the dk pass).
    hbf16 qdk[n_dc_w][n_warp][4];
    if (dk_warp_ok) {
#    pragma unroll
        for (int dl = 0; dl < n_dc_w; dl++) {
            const int dc = warp + dl * dk_warps;
            if (dc < ND) {
#    pragma unroll
                for (int hc = 0; hc < n_warp; hc++) {
                    const int hg0    = hc * 16 + (g << 2);
                    const int hcl    = (hg0 < Hq) ? hg0 : (Hq - 1);
                    const hbf16* qp2 = (const hbf16*)(q + ((size_t)row * Hq + hcl) * D + dc * 16 + r);
                    qdk[dl][hc][0]   = (hg0 < Hq) ? qp2[0] : hbf16(0);
                    qdk[dl][hc][1]   = (hg0 + 1 < Hq) ? qp2[(size_t)D] : hbf16(0);
                    qdk[dl][hc][2]   = (hg0 + 2 < Hq) ? qp2[2 * (size_t)D] : hbf16(0);
                    qdk[dl][hc][3]   = (hg0 + 3 < Hq) ? qp2[3 * (size_t)D] : hbf16(0);
                }
            }
        }
    }

    if constexpr (FAST) {
        uint4 first_k[2];
        prefetch_k(0, first_k, current_idx);
        commit_k(first_k);
        __syncthreads();
    }
    for (int chunk = 0; chunk < n_chunks; chunk++) {
        // The full-tile dispatch requires K % 32 == 0. last_key <= K,
        // so every executed chunk owns 32 physical slots. Holes and
        // arbitrary/repeated key indices retain their original checks.
        const int ck     = FULL_TILES ? IDXB_CHUNK : min(IDXB_CHUNK, topk - chunk * IDXB_CHUNK);
        const int ntiles = FULL_TILES ? (IDXB_CHUNK / 16) : (ck + 15) >> 4;

        int next_idx = -1, after_idx = -1;
        if constexpr (SHARED_INDEX) {
            next_idx      = future_idx;
            const int pos = (chunk + 2) * 16 + r;
            if (pos < topk) after_idx = idx_lds[pos];
        }
        uint4 next_k[2];
        if constexpr (FAST) {
            if (chunk + 1 < n_chunks) prefetch_k(chunk + 1, next_k, next_idx);
        } else {
            // Gather K chunk again (same layout as pass 1).
            for (int u = threadIdx.x; u < ck * (D / 8); u += IDXB_THREADS) {
                int kk    = u / (D / 8);
                int d8    = u - kk * (D / 8);
                int kv    = idx_lds[chunk * IDXB_CHUNK + kk];
                bf16* dst = (bf16*)(k_lds + (size_t)kk * KLD + d8 * 8);
                if (kv >= 0 && kv < (topk_indices_global ? B * Sk : Sk)) {
                    const bf16* src   = k + ((size_t)(topk_indices_global ? 0 : b) * Sk + (size_t)kv) * D + d8 * 8;
                    ((float2*)dst)[0] = ((const float2*)src)[0];
                    ((float2*)dst)[1] = ((const float2*)src)[1];
                } else {
                    ((uint4*)dst)[0] = make_uint4(0, 0, 0, 0);
                }
            }
            if (ck < IDXB_CHUNK) {
                for (int u = threadIdx.x; u < (IDXB_CHUNK - ck) * D; u += IDXB_THREADS) {
                    int kk                      = ck + u / D;
                    int d                       = u - (u / D) * D;
                    k_lds[(size_t)kk * KLD + d] = hbf16(0);
                }
            }
            // The whole block cooperatively fills k_lds above and every warp
            // then reads *other* threads' rows through the QK matrix loads, so
            // this barrier is required.  Removing it (a leftover probe edit in
            // the d26e9fd handoff commit) made the dq/dw/dk of every non-FAST
            // instance (D != 128, or D == 128 with NW 1/3) depend on warp
            // timing and intermittently exceed tolerance.
            __syncthreads();
        }

        // One lane-distributed score read per 16-key tile. These immutable
        // FP32 grad signals are shared by the 16 heads of every wave.
        // Prefetch before QK; broadcast bits to the original accumulator layout.
        float score_prefetch[2];
        if constexpr (FAST && SCORE_GLOBAL && FULL_TILES) {
#    pragma unroll
            for (int nt = 0; nt < 2; nt++) score_prefetch[nt] = score[chunk * IDXB_CHUNK + nt * 16 + r];
        }
        DUFragment<accumulator, 16, 16, 16, float> acc[2];
#    pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            if (nt < ntiles) du_fill_fragment(acc[nt], 0.0f);
        }
        if (warp_ok) {
            for (int kk = 0; kk < ND; kk++) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> a;
                *(uint64_t*)&a.x[0] = qa64[kk];
#    pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    if (nt < ntiles) {
                        DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bf;
                        du_load_matrix_sync(bf, k_lds + (size_t)(nt * 16) * KLD + kk * 16, KLD);
                        du_mma_sync(acc[nt], a, bf, acc[nt]);
                    }
                }
            }
        }

        uint16_t grad_c[2][4];
        // Store the per-warp (16 h x 16 k) rounded dS tiles to LDS so the
        // transposed (k x h) operand for the dk MMA can be built.
        if (warp_ok) {
#    pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                if (nt < ntiles) {
                    hbf16* mt = mask_lds + ((size_t)warp * (IDXB_CHUNK / 16) + nt) * 320;
#    pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int col = g + 4 * j;
                        grad_c[nt][j] = 0;
                        if (col < ck) {
                            // CuTe SM90: dS = BF16(grad_signal * w * relu_gate).
                            // Q/K MMAs accumulate FP32; scale is applied after them.
                            const int pos = chunk * IDXB_CHUNK + nt * 16 + col;
                            float dsc;
                            if constexpr (FAST && SCORE_GLOBAL && FULL_TILES)
                                dsc = __shfl(score_prefetch[nt], g + 4 * j, 64);
                            else
                                dsc = pos < topk ? score[pos] : 0.0f;
                            const float dot = acc[nt].x[j] * sm_scale;
                            const hbf16 ds  = idxb_rne(dot > 0.0f ? dsc * w_h : 0.0f);
                            grad_c[nt][j]   = *(const uint16_t*)&ds;
                            if constexpr (TRANSPOSE_MASK)
                                mt[(size_t)col * 20 + r] = ds;
                            else
                                mt[(size_t)r * 20 + col] = ds;
                            dwacc[j] += dsc * (dot > 0.0f ? dot : 0.0f);
                        }
                    }
                }
            }
        }
        __syncthreads();

        if constexpr (D == 128 && NW == 4 && FULL_TILES && SCORE_GLOBAL && IDX_GLOBAL && IDXB_CHUNK == 16) {
            idxb_floatx4 av0 = {0, 0, 0, 0}, av1 = {0, 0, 0, 0};
            // ---- dk: acc += A(k x h) @ Q(h x d); direct per-warp atomics ----
            // Each warp computes the full h-sum for its own disjoint set of
            // (nt, dc) tiles, so no cross-warp LDS staging / extra barriers are
            // needed; Q is read from global (L1-cached) as the matrix_b operand.
            {
                DUFragment<accumulator, 16, 16, 16, float> dkacc[2][n_dc_w];
#    pragma unroll
                for (int nt = 0; nt < 2; nt++) {
#    pragma unroll
                    for (int dl = 0; dl < n_dc_w; dl++) du_fill_fragment(dkacc[nt][dl], 0.0f);
                }
#    pragma unroll
                for (int hc = 0; hc < n_warp; hc++) {
                    const int hbase = hc * 16;
#    pragma unroll
                    for (int nt = 0; nt < 2; nt++) {
                        if (nt < ntiles) {
                            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> adk;
                            const hbf16* mt = mask_lds + ((size_t)hc * (IDXB_CHUNK / 16) + nt) * 320;
                            if constexpr (TRANSPOSE_MASK) {
                                // Four adjacent heads occupy one aligned 8B
                                // vector. Preserve the rounded dS bits exactly.
                                *(uint64_t*)&adk.x[0] = *(const uint64_t*)(mt + (size_t)r * 20 + g * 4);
                            } else {
#    pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    // A element (row = t = r, col = h) of (t x h).
                                    // Tile already holds w_h * relu(S) in bf16
                                    // (phantom-head rows are zero), so load directly.
                                    adk.x[j] = mt[(size_t)(g * 4 + j) * 20 + r];
                                }
                            }
#    pragma unroll
                            for (int dl = 0; dl < n_dc_w; dl++) {
                                const int dc = warp + dl * dk_warps;
                                if (dc < ND) {
                                    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bf;
                                    // Phantom heads (h >= Hq, e.g. H=8 with a
                                    // 16-head tile) must NOT read past the Hq
                                    // block of q (out-of-bounds at the last row)
                                    // and their contributions must be zero: the
                                    // adk operand already carries a zero weight
                                    // there, so 0 * stale (possibly NaN) would
                                    // otherwise poison the whole dk row.
                                    const int hg0    = hbase + (g << 2);
                                    const int hcl    = (hg0 < Hq) ? hg0 : (Hq - 1);
                                    const hbf16* qp2 = (const hbf16*)(q + ((size_t)row * Hq + hcl) * D + dc * 16 + r);
                                    bf.x[0]          = (hg0 < Hq) ? qdk[dl][hc][0] : hbf16(0);
                                    bf.x[1]          = (hg0 + 1 < Hq) ? qdk[dl][hc][1] : hbf16(0);
                                    bf.x[2]          = (hg0 + 2 < Hq) ? qdk[dl][hc][2] : hbf16(0);
                                    bf.x[3]          = (hg0 + 3 < Hq) ? qdk[dl][hc][3] : hbf16(0);

                                    du_mma_sync(dkacc[nt][dl], adk, bf, dkacc[nt][dl]);
                                }
                            }
                        }
                    }
                }
                // Stage both destination indices before the first atomic. The
                // global-index path otherwise waits for prior atomic returns to
                // load the second index. Keep compiler-managed return lifetimes.
                int prefetched_kidx[2];
                if constexpr (FAST && NW == 4) {
#    pragma unroll
                    for (int nt = 0; nt < 2; ++nt)
                        prefetched_kidx[nt] =
                            SHARED_INDEX ? current_idx : (nt < ntiles ? idx_lds[chunk * IDXB_CHUNK + nt * 16 + r] : -1);
                    asm volatile("" ::"v"(prefetched_kidx[0]), "v"(prefetched_kidx[1]) : "memory");
                }
                av0 = {sm_scale * dkacc[0][0].x[0], sm_scale * dkacc[0][0].x[1], sm_scale * dkacc[0][0].x[2],
                       sm_scale * dkacc[0][0].x[3]};
                av1 = {sm_scale * dkacc[0][1].x[0], sm_scale * dkacc[0][1].x[1], sm_scale * dkacc[0][1].x[2],
                       sm_scale * dkacc[0][1].x[3]};
                uint64_t address = (uint64_t)dk_float +
                                   (((uint64_t)(topk_indices_global ? 0 : b) * Sk + (uint32_t)prefetched_kidx[0]) * D +
                                    warp * 16 + 4 * g) *
                                       sizeof(float);
                const uint64_t mask = __ballot(r < ck && prefetched_kidx[0] >= 0 &&
                                               prefetched_kidx[0] < (topk_indices_global ? B * Sk : Sk));
                uint64_t saved;
                asm volatile("s_and_saveexec_b64 %[saved], %[mask]\n\t"
                         "global_atomic_add_f32_x4 %[v0], %[addr], %[v0], off glc\n\t"
                         "global_atomic_add_f32_x4 %[v1], %[addr], %[v1], off offset:256 glc\n\t"
                         "s_mov_b64 exec, %[saved]"
                 : [v0] "+v"(av0),[v1] "+v"(av1),[saved] "=&s"(saved),[addr] "+v"(address)
                 : [mask] "s"(mask) : "exec","memory");
            }
            // ---- dq: acc += A(h x k) @ K(k x d); dw: dscore*relu ----
            if (warp_ok) {
#    pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    if (nt < ntiles) {
                        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> adq;
                        const int kbase = chunk * IDXB_CHUNK + nt * 16;
                        hbf16* mt0      = mask_lds + ((size_t)warp * (IDXB_CHUNK / 16) + nt) * 320;
                        if constexpr (FAST) {
                            // The stored dK tile and register dQ operand contain
                            // exactly the same rounded dS values, as in CuTe.
                            const uint16_t* eb = grad_c[nt];
                            unsigned a         = unsigned(eb[0]) | (unsigned(eb[1]) << 16);
                            unsigned b         = unsigned(eb[2]) | (unsigned(eb[3]) << 16);
                            unsigned ap = __shfl_xor(a, 16), bp = __shfl_xor(b, 16);
                            unsigned lo    = (g & 1) ? (ap >> 16) | (a & 0xffff0000u) : (a & 0xffffu) | (ap << 16);
                            unsigned hi    = (g & 1) ? (bp >> 16) | (b & 0xffff0000u) : (b & 0xffffu) | (bp << 16);
                            unsigned other = __shfl_xor((g & 2) ? lo : hi, 32);
                            if (g & 2)
                                lo = other;
                            else
                                hi = other;
                            ((unsigned*)&adq.x[0])[0] = lo;
                            ((unsigned*)&adq.x[0])[1] = hi;
                        } else {
#    pragma unroll
                            for (int j = 0; j < 4; j++) {
                                // matrix_a fragment element (row = r, col = g*4+j) of the
                                // (h x t) A operand. The LDS tile holds BF16(grad_signal *
                                // w_h * ReLU_gate), shared with the dK contraction.
                                const int col = g * 4 + j;
                                adq.x[j]      = (col < ck) ? mt0[(size_t)r * 20 + col] : hbf16(0);
                            }
                        }
                        if constexpr (FAST) {
                            typedef unsigned int mat4 __attribute__((ext_vector_type(4)));
                            mat4 packed[4];
#    pragma unroll
                            for (int z = 0; z < 4; z++) {
                                const unsigned addr = (unsigned)(size_t)(k_lds + (size_t)(nt * 16) * KLD + z * 32 +
                                                                         (lane / 4) * KLD + (lane % 4) * 8);
                                asm volatile("ds_read_m32x16_b16_alt %0, %1" : "=v"(packed[z]) : "v"(addr) : "memory");
                            }
                            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#    pragma unroll
                            for (int z = 0; z < 4; z++) {
                                DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                                ((uint32_t*)&b0.x[0])[0] = packed[z][0];
                                ((uint32_t*)&b0.x[0])[1] = packed[z][1];
                                ((uint32_t*)&b1.x[0])[0] = packed[z][2];
                                ((uint32_t*)&b1.x[0])[1] = packed[z][3];
                                du_mma_sync(dqacc[2 * z], adq, b0, dqacc[2 * z]);
                                du_mma_sync(dqacc[2 * z + 1], adq, b1, dqacc[2 * z + 1]);
                            }
                        } else {
#    pragma unroll
                            for (int dc = 0; dc < ND; dc++) {
                                DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bf;
                                du_load_matrix_sync(bf, k_lds + (size_t)(nt * 16) * KLD + dc * 16, KLD);
                                du_mma_sync(dqacc[dc], adq, bf, dqacc[dc]);
                            }
                        }
                    }
                }
            }

            asm volatile("s_waitcnt vmcnt(0)" : "+v"(av0), "+v"(av1)::"memory");
        } else {
            // ---- dq: acc += A(h x k) @ K(k x d); dw: dscore*relu ----
            if (warp_ok) {
#    pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    if (nt < ntiles) {
                        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> adq;
                        const int kbase = chunk * IDXB_CHUNK + nt * 16;
                        hbf16* mt0      = mask_lds + ((size_t)warp * (IDXB_CHUNK / 16) + nt) * 320;
                        if constexpr (FAST) {
                            // The stored dK tile and register dQ operand contain
                            // exactly the same rounded dS values, as in CuTe.
                            const uint16_t* eb = grad_c[nt];
                            unsigned a         = unsigned(eb[0]) | (unsigned(eb[1]) << 16);
                            unsigned b         = unsigned(eb[2]) | (unsigned(eb[3]) << 16);
                            unsigned ap = __shfl_xor(a, 16), bp = __shfl_xor(b, 16);
                            unsigned lo    = (g & 1) ? (ap >> 16) | (a & 0xffff0000u) : (a & 0xffffu) | (ap << 16);
                            unsigned hi    = (g & 1) ? (bp >> 16) | (b & 0xffff0000u) : (b & 0xffffu) | (bp << 16);
                            unsigned other = __shfl_xor((g & 2) ? lo : hi, 32);
                            if (g & 2)
                                lo = other;
                            else
                                hi = other;
                            ((unsigned*)&adq.x[0])[0] = lo;
                            ((unsigned*)&adq.x[0])[1] = hi;
                        } else {
#    pragma unroll
                            for (int j = 0; j < 4; j++) {
                                // matrix_a fragment element (row = r, col = g*4+j) of the
                                // (h x t) A operand. The LDS tile holds BF16(grad_signal *
                                // w_h * ReLU_gate), shared with the dK contraction.
                                const int col = g * 4 + j;
                                adq.x[j]      = (col < ck) ? mt0[(size_t)r * 20 + col] : hbf16(0);
                            }
                        }
                        if constexpr (FAST) {
                            typedef unsigned int mat4 __attribute__((ext_vector_type(4)));
                            mat4 packed[4];
#    pragma unroll
                            for (int z = 0; z < 4; z++) {
                                const unsigned addr = (unsigned)(size_t)(k_lds + (size_t)(nt * 16) * KLD + z * 32 +
                                                                         (lane / 4) * KLD + (lane % 4) * 8);
                                asm volatile("ds_read_m32x16_b16_alt %0, %1" : "=v"(packed[z]) : "v"(addr) : "memory");
                            }
                            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#    pragma unroll
                            for (int z = 0; z < 4; z++) {
                                DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                                ((uint32_t*)&b0.x[0])[0] = packed[z][0];
                                ((uint32_t*)&b0.x[0])[1] = packed[z][1];
                                ((uint32_t*)&b1.x[0])[0] = packed[z][2];
                                ((uint32_t*)&b1.x[0])[1] = packed[z][3];
                                du_mma_sync(dqacc[2 * z], adq, b0, dqacc[2 * z]);
                                du_mma_sync(dqacc[2 * z + 1], adq, b1, dqacc[2 * z + 1]);
                            }
                        } else {
#    pragma unroll
                            for (int dc = 0; dc < ND; dc++) {
                                DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bf;
                                du_load_matrix_sync(bf, k_lds + (size_t)(nt * 16) * KLD + dc * 16, KLD);
                                du_mma_sync(dqacc[dc], adq, bf, dqacc[dc]);
                            }
                        }
                    }
                }
            }

            // ---- dk: acc += A(k x h) @ Q(h x d); direct per-warp atomics ----
            // Each warp computes the full h-sum for its own disjoint set of
            // (nt, dc) tiles, so no cross-warp LDS staging / extra barriers are
            // needed; Q is read from global (L1-cached) as the matrix_b operand.
            if (dk_warp_ok) {
                DUFragment<accumulator, 16, 16, 16, float> dkacc[2][n_dc_w];
#    pragma unroll
                for (int nt = 0; nt < 2; nt++) {
#    pragma unroll
                    for (int dl = 0; dl < n_dc_w; dl++) du_fill_fragment(dkacc[nt][dl], 0.0f);
                }
#    pragma unroll
                for (int hc = 0; hc < n_warp; hc++) {
                    const int hbase = hc * 16;
#    pragma unroll
                    for (int nt = 0; nt < 2; nt++) {
                        if (nt < ntiles) {
                            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> adk;
                            const hbf16* mt = mask_lds + ((size_t)hc * (IDXB_CHUNK / 16) + nt) * 320;
                            if constexpr (TRANSPOSE_MASK) {
                                // Four adjacent heads occupy one aligned 8B
                                // vector. Preserve the rounded dS bits exactly.
                                *(uint64_t*)&adk.x[0] = *(const uint64_t*)(mt + (size_t)r * 20 + g * 4);
                            } else {
#    pragma unroll
                                for (int j = 0; j < 4; j++) {
                                    // A element (row = t = r, col = h) of (t x h).
                                    // Tile already holds w_h * relu(S) in bf16
                                    // (phantom-head rows are zero), so load directly.
                                    adk.x[j] = mt[(size_t)(g * 4 + j) * 20 + r];
                                }
                            }
#    pragma unroll
                            for (int dl = 0; dl < n_dc_w; dl++) {
                                const int dc = warp + dl * dk_warps;
                                if (dc < ND) {
                                    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bf;
                                    // Phantom heads (h >= Hq, e.g. H=8 with a
                                    // 16-head tile) must NOT read past the Hq
                                    // block of q (out-of-bounds at the last row)
                                    // and their contributions must be zero: the
                                    // adk operand already carries a zero weight
                                    // there, so 0 * stale (possibly NaN) would
                                    // otherwise poison the whole dk row.
                                    const int hg0    = hbase + (g << 2);
                                    const int hcl    = (hg0 < Hq) ? hg0 : (Hq - 1);
                                    const hbf16* qp2 = (const hbf16*)(q + ((size_t)row * Hq + hcl) * D + dc * 16 + r);
                                    bf.x[0]          = (hg0 < Hq) ? qdk[dl][hc][0] : hbf16(0);
                                    bf.x[1]          = (hg0 + 1 < Hq) ? qdk[dl][hc][1] : hbf16(0);
                                    bf.x[2]          = (hg0 + 2 < Hq) ? qdk[dl][hc][2] : hbf16(0);
                                    bf.x[3]          = (hg0 + 3 < Hq) ? qdk[dl][hc][3] : hbf16(0);

                                    du_mma_sync(dkacc[nt][dl], adk, bf, dkacc[nt][dl]);
                                }
                            }
                        }
                    }
                }
                // Stage both destination indices before the first atomic. The
                // global-index path otherwise waits for prior atomic returns to
                // load the second index. Keep compiler-managed return lifetimes.
                int prefetched_kidx[2];
                if constexpr (FAST && NW == 4) {
#    pragma unroll
                    for (int nt = 0; nt < 2; ++nt)
                        prefetched_kidx[nt] =
                            SHARED_INDEX ? current_idx : (nt < ntiles ? idx_lds[chunk * IDXB_CHUNK + nt * 16 + r] : -1);
                    asm volatile("" ::"v"(prefetched_kidx[0]), "v"(prefetched_kidx[1]) : "memory");
                }
// Flush this warp's disjoint tiles directly to the f32 workspace
// (each (k, d) gets exactly one atomicAdd per block).
#    pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    if (nt < ntiles) {
                        const int kidx =
                            (FAST && NW == 4) ? prefetched_kidx[nt] : idx_lds[chunk * IDXB_CHUNK + nt * 16 + r];
                        if (r < ck && kidx >= 0 && kidx < (topk_indices_global ? B * Sk : Sk)) {
                            const float sc = sm_scale;
                            float* pk      = dk_float + ((size_t)(topk_indices_global ? 0 : b) * Sk + kidx) * D;
#    pragma unroll
                            for (int dl = 0; dl < n_dc_w; dl++) {
                                const int dc = warp + dl * dk_warps;
                                if (dc < ND) {
                                    float* pk2 = pk + dc * 16;
                                    idxb_atomic_add_sparse<FAST>(pk2, g, sc * dkacc[nt][dl].x[0],
                                                                 sc * dkacc[nt][dl].x[1], sc * dkacc[nt][dl].x[2],
                                                                 sc * dkacc[nt][dl].x[3]);
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        if constexpr (FAST) {
            if (chunk + 1 < n_chunks) commit_k(next_k);
            __syncthreads();
            if constexpr (SHARED_INDEX) {
                current_idx = next_idx;
                future_idx  = after_idx;
            }
        }
    }

    // ---- dq / dw epilogues ----
    if (h < Hq) {
        bf16* dqbase = dq + (size_t)row * Hq * D + (size_t)h * D;
        if constexpr (FAST) {
// ALT B loads produce adjacent output columns in pairs.
// Preserve the final conversion, then exchange pairs for 128-bit stores.
#    pragma unroll
            for (int dc = 0; dc < ND; dc += 2) {
                uint32_t v[4], x[4], y[4];
#    pragma unroll
                for (int z = 0; z < 4; z++)
                    v[z] = hip_dsa_numeric::bf16_pair(sm_scale * dqacc[dc].x[z], sm_scale * dqacc[dc + 1].x[z]);
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
                *(uint4*)(dqbase + (dc / 2) * 32 + g * 8) = make_uint4(y[0], y[1], y[2], y[3]);
            }
        } else {
#    pragma unroll
            for (int dc = 0; dc < ND; dc++) {
                const int col    = dc * 16 + g;
                dqbase[col]      = float_to_bf16(sm_scale * dqacc[dc].x[0]);
                dqbase[col + 4]  = float_to_bf16(sm_scale * dqacc[dc].x[1]);
                dqbase[col + 8]  = float_to_bf16(sm_scale * dqacc[dc].x[2]);
                dqbase[col + 12] = float_to_bf16(sm_scale * dqacc[dc].x[3]);
            }
        }
        float v  = (dwacc[0] + dwacc[1] + dwacc[2] + dwacc[3]);
        v       += __shfl_xor(v, 16);
        v       += __shfl_xor(v, 32);
        if (g == 0) dw[row * Hq + h] = float_to_bf16(v);
    }

    if (done_count == nullptr) return;
    // ---- Last-block cast: dk_float (fp32 atomics) -> bf16 dk.  All blocks
    // have committed their dk atomics at this point (per-chunk __syncthreads
    // above), so the last block to arrive casts the whole buffer in place:
    // the operator stays exactly one kernel launch. ----
    __threadfence();
    __syncthreads();
    __shared__ int fused_last;
    if (threadIdx.x == 0) fused_last = (atomicAdd(done_count, 1) == gridDim.x - 1);
    __syncthreads();
    if (fused_last) {
        // Vectorized + unrolled cast: the single last block must sweep the
        // whole (B*Sk*D) f32 buffer, and a scalar dependent load->store loop
        // is latency-bound (~0.9ms for 1M elements).  float4 loads with an
        // 8-way unrolled stride keep 8 independent loads in flight per lane.
        const long n4      = ((long)B * Sk * D) >> 2;
        const float4* src4 = (const float4*)dk_float;
        uint64_t* dst      = (uint64_t*)dk;
        long i             = threadIdx.x;
        for (; i + 7 * (long)IDXB_THREADS < n4; i += 8 * (long)IDXB_THREADS) {
#    pragma unroll
            for (int u = 0; u < 8; u++) {
                const float4 v                  = idxb_cast_load<FAST>(src4, i + (long)u * IDXB_THREADS);
                const uint64_t b0               = (uint64_t)(uint16_t)float_to_bf16(v.x);
                const uint64_t b1               = (uint64_t)(uint16_t)float_to_bf16(v.y);
                const uint64_t b2               = (uint64_t)(uint16_t)float_to_bf16(v.z);
                const uint64_t b3               = (uint64_t)(uint16_t)float_to_bf16(v.w);
                dst[i + (long)u * IDXB_THREADS] = b0 | (b1 << 16) | (b2 << 32) | (b3 << 48);
            }
        }
        for (; i < n4; i += IDXB_THREADS) {
            const float4 v    = idxb_cast_load<FAST>(src4, i);
            const uint64_t b0 = (uint64_t)(uint16_t)float_to_bf16(v.x);
            const uint64_t b1 = (uint64_t)(uint16_t)float_to_bf16(v.y);
            const uint64_t b2 = (uint64_t)(uint16_t)float_to_bf16(v.z);
            const uint64_t b3 = (uint64_t)(uint16_t)float_to_bf16(v.w);
            dst[i]            = b0 | (b1 << 16) | (b2 << 32) | (b3 << 48);
        }
    }
}

#else  // no MMAC intrinsics on this arch

template<int D, int NW, bool FULL_TILES = false, bool SCORE_GLOBAL = false, bool IDX_GLOBAL = false,
         typename GradScale = float>
__global__ static void __launch_bounds__(IDXB_THREADS,
                                         ((D == 128 && (NW == 2 || (NW == 4 && SCORE_GLOBAL && IDX_GLOBAL))) ? 1 : 2))
    indexer_bwd_sparse_mmac_kernel(const bf16*, const bf16*, const bf16*, float*, const float*, const int*, bf16*,
                                   bf16*, float*, bf16*, int*, int, int, int, int, int, float, float, GradScale, bool) {
}

#endif

template<typename GradScale>
__global__ __launch_bounds__(128, 1) static void indexer_bwd_sparse_two_wave_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ w, const bf16* __restrict__ k, float* __restrict__ target,
    const float* __restrict__ predict, const int* __restrict__ idxs, bf16* __restrict__ dq, bf16* __restrict__ dw,
    float* __restrict__ dk_float, bf16* __restrict__ dk, int* done_count, int B, int Sq, int Sk, int Hq, int topk,
    float sm_scale, float loss_coeff, GradScale grad_loss, bool topk_indices_global) {
#if defined(__gfx936__) || defined(__gfx938__)
    constexpr int D = 128, KLD = 136, ND = 8;
    const int row = blockIdx.x, b = row / Sq, lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    const int r = lane & 15, g = lane >> 4;
    extern __shared__ char raw[];
    hbf16* klds        = (hbf16*)raw;
    float* red         = (float*)(klds + 16 * KLD);
    hbf16* mask        = (hbf16*)(red + 128);
    const int* indices = idxs + (size_t)row * topk;
    float* signal      = target + (size_t)row * topk;
    const float* prob  = predict + (size_t)row * topk;
    int last           = 0;
    for (int i = threadIdx.x; i < topk; i += 128) {
        int ix = indices[i];
        if (ix >= 0 && ix < (topk_indices_global ? B * Sk : Sk)) last = max(last, i + 1);
    }
#    pragma unroll
    for (int off = 32; off; off >>= 1) last = max(last, __shfl_xor(last, off));
    if (lane == 0) red[warp] = (float)last;
    __syncthreads();
    const int chunks = ((int)fmaxf(red[0], red[1]) + 15) / 16;
    // Reproduce the original SM90-compatible128-thread ascending reduction.
    const float c = loss_coeff / (float)(B * Sq) * idxb_grad_scale(grad_loss), eps = expf(-100.f);
    float sum = 0.f;
    for (int pos = threadIdx.x; pos < topk; pos += 128) {
        const float t = fmaxf(signal[pos], eps), p = prob[pos];
        sum += -t * (p >= eps ? 1.f : 0.f) * c;
    }
    // Every wave has consumed red[0:2] before this storage is reused.
    __syncthreads();
    red[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
#    pragma unroll
        for (int i = 0; i < 128; i++) total += red[i];
        red[0] = total;
    }
    __syncthreads();
    const float total = red[0];
    float scores[16];
#    pragma unroll
    for (int j = 0; j < 16; j++) {
        const int pos = threadIdx.x + j * 128;
        if (pos < topk) {
            const float t = fmaxf(signal[pos], eps), p = prob[pos];
            scores[j] = -t * (p >= eps ? 1.f : 0.f) * c - p * total;
        }
    }
    __syncthreads();
#    pragma unroll
    for (int j = 0; j < 16; j++)
        if (threadIdx.x + j * 128 < topk) signal[threadIdx.x + j * 128] = scores[j];
    __syncthreads();
    uint64_t qa[2][ND];
    float wh[2];
#    pragma unroll
    for (int hs = 0; hs < 2; hs++) {
        const int h = hs * 32 + warp * 16 + r;
        wh[hs]      = bf16_to_float(w[(size_t)row * 64 + h]);
#    pragma unroll
        for (int dc = 0; dc < ND; dc++)
            qa[hs][dc] = *(const uint64_t*)(q + ((size_t)row * 64 + h) * D + dc * 16 + g * 4);
    }
    uint32_t qdk[4][4][2];
#    pragma unroll
    for (int dl = 0; dl < 4; dl++) {
        const int dc = warp + dl * 2;
#    pragma unroll
        for (int hc = 0; hc < 4; hc++) {
            const int h    = hc * 16 + g * 4;
            const bf16* qp = q + ((size_t)row * 64 + h) * D + dc * 16 + r;
#    pragma unroll
            for (int e = 0; e < 2; e++) {
                uint32_t v = (uint32_t)qp[(2 * e) * D] | ((uint32_t)qp[(2 * e + 1) * D] << 16);
                asm volatile("" : "+v"(v));
                qdk[dl][hc][e] = v;
            }
        }
    }
    DUFragment<accumulator, 16, 16, 16, float> dqacc[2][ND];
    float dwacc[2][4];
#    pragma unroll
    for (int hs = 0; hs < 2; hs++) {
#    pragma unroll
        for (int dc = 0; dc < ND; dc++) du_fill_fragment(dqacc[hs][dc], 0.f);
#    pragma unroll
        for (int j = 0; j < 4; j++) dwacc[hs][j] = 0.f;
    }
    auto prefetch = [&](int shared_idx, uint4* regs) {
#    pragma unroll
        for (int j = 0; j < 2; j++) {
            const int u = threadIdx.x + j * 128, ix = __shfl(shared_idx, warp * 4 + g + j * 8, 64);
            regs[j] = (ix >= 0 && ix < (topk_indices_global ? B * Sk : Sk))
                          ? *(const uint4*)(k + ((size_t)(topk_indices_global ? 0 : b) * Sk + ix) * D + (u % 16) * 8)
                          : make_uint4(0, 0, 0, 0);
        }
    };
    auto commit = [&](const uint4* regs) {
#    pragma unroll
        for (int j = 0; j < 2; j++) {
            const int u                                     = threadIdx.x + j * 128;
            *(uint4*)(klds + (u / 16) * KLD + (u % 16) * 8) = regs[j];
        }
    };
    int current_idx = indices[r], future_idx = 16 + r < topk ? indices[16 + r] : -1;
    float carried_score = signal[r];
    uint4 initial[2];
    prefetch(current_idx, initial);
    commit(initial);
    __syncthreads();
    for (int chunk = 0; chunk < chunks; chunk++) {
        const int next_idx = future_idx, pos = (chunk + 2) * 16 + r;
        const int after_idx = pos < topk ? indices[pos] : -1;
        uint4 next_k[2];
        if (chunk + 1 < chunks) prefetch(next_idx, next_k);
        const float score      = carried_score;
        const float next_score = (chunk + 1) * 16 + r < topk ? signal[(chunk + 1) * 16 + r] : 0.f;
        uint16_t grad[2][4];
#    pragma unroll
        for (int hs = 0; hs < 2; hs++) {
            DUFragment<accumulator, 16, 16, 16, float> dot;
            du_fill_fragment(dot, 0.f);
#    pragma unroll
            for (int dc = 0; dc < ND; dc++) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> a;
                DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> kb;
                *(uint64_t*)a.x = qa[hs][dc];
                du_load_matrix_sync(kb, klds + dc * 16, KLD);
                du_mma_sync(dot, a, kb, dot);
            }
            hbf16* mt = mask + (hs * 2 + warp) * 320;
#    pragma unroll
            for (int j = 0; j < 4; j++) {
                const int col  = g + 4 * j;
                const float ds = __shfl(score, col, 64), v = dot.x[j] * sm_scale;
                hbf16 rounded     = idxb_rne(v > 0.f ? ds * wh[hs] : 0.f);
                grad[hs][j]       = *(uint16_t*)&rounded;
                mt[col * 20 + r]  = rounded;
                dwacc[hs][j]     += ds * (v > 0.f ? v : 0.f);
            }
        }
        __syncthreads();
        DUFragment<accumulator, 16, 16, 16, float> dkacc[4];
#    pragma unroll
        for (int dl = 0; dl < 4; dl++) du_fill_fragment(dkacc[dl], 0.f);
#    pragma unroll
        for (int hc = 0; hc < 4; hc++) {
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> a;
            *(uint64_t*)a.x = *(const uint64_t*)(mask + hc * 320 + r * 20 + g * 4);
#    pragma unroll
            for (int dl = 0; dl < 4; dl++) {
                DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> qb;
                ((uint32_t*)qb.x)[0] = qdk[dl][hc][0];
                ((uint32_t*)qb.x)[1] = qdk[dl][hc][1];
                du_mma_sync(dkacc[dl], a, qb, dkacc[dl]);
            }
        }
        idxb_floatx4 av[4];
#    pragma unroll
        for (int dl = 0; dl < 4; dl++)
            av[dl] = {sm_scale * dkacc[dl].x[0], sm_scale * dkacc[dl].x[1], sm_scale * dkacc[dl].x[2],
                      sm_scale * dkacc[dl].x[3]};
        uint64_t address =
            (uint64_t)dk_float +
            (((uint64_t)(topk_indices_global ? 0 : b) * Sk + (uint32_t)current_idx) * D + warp * 16 + 4 * g) *
                sizeof(float);
        const uint64_t live = __ballot(current_idx >= 0 && current_idx < (topk_indices_global ? B * Sk : Sk));
        uint64_t saved;
        asm volatile("s_and_saveexec_b64 %[saved], %[live]\n\t"
               "global_atomic_add_f32_x4 %[v0], %[addr], %[v0], off glc\n\t"
               "global_atomic_add_f32_x4 %[v1], %[addr], %[v1], off offset:128 glc\n\t"
               "global_atomic_add_f32_x4 %[v2], %[addr], %[v2], off offset:256 glc\n\t"
               "global_atomic_add_f32_x4 %[v3], %[addr], %[v3], off offset:384 glc\n\t"
               "s_mov_b64 exec, %[saved]"
      :[v0] "+v"(av[0]),[v1] "+v"(av[1]),[v2] "+v"(av[2]),[v3] "+v"(av[3]),[addr] "+v"(address),[saved] "=&s"(saved)
      :[live] "s"(live):"exec","memory");
#    pragma unroll
        for (int hs = 0; hs < 2; hs++) {
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> adq;
            unsigned a  = unsigned(grad[hs][0]) | (unsigned(grad[hs][1]) << 16),
                     b  = unsigned(grad[hs][2]) | (unsigned(grad[hs][3]) << 16);
            unsigned ap = __shfl_xor(a, 16), bp = __shfl_xor(b, 16);
            unsigned lo    = (g & 1) ? (ap >> 16) | (a & 0xffff0000u) : (a & 0xffffu) | (ap << 16);
            unsigned hi    = (g & 1) ? (bp >> 16) | (b & 0xffff0000u) : (b & 0xffffu) | (bp << 16);
            unsigned other = __shfl_xor((g & 2) ? lo : hi, 32);
            if (g & 2)
                lo = other;
            else
                hi = other;
            ((unsigned*)adq.x)[0] = lo;
            ((unsigned*)adq.x)[1] = hi;
            typedef unsigned int mat4 __attribute__((ext_vector_type(4)));
#    pragma unroll
            for (int zb = 0; zb < 4; zb += 2) {
                mat4 packed[2];
#    pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int z         = zb + j;
                    const unsigned addr = (unsigned)(size_t)(klds + z * 32 + (lane / 4) * KLD + (lane % 4) * 8);
                    asm volatile("ds_read_m32x16_b16_alt %0, %1" : "=v"(packed[j]) : "v"(addr) : "memory");
                }
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#    pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int z = zb + j;
                    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                    ((uint32_t*)b0.x)[0] = packed[j][0];
                    ((uint32_t*)b0.x)[1] = packed[j][1];
                    ((uint32_t*)b1.x)[0] = packed[j][2];
                    ((uint32_t*)b1.x)[1] = packed[j][3];
                    du_mma_sync(dqacc[hs][2 * z], adq, b0, dqacc[hs][2 * z]);
                    du_mma_sync(dqacc[hs][2 * z + 1], adq, b1, dqacc[hs][2 * z + 1]);
                }
            }
        }
        asm volatile("s_waitcnt vmcnt(0)" : "+v"(av[0]), "+v"(av[1]), "+v"(av[2]), "+v"(av[3])::"memory");
        __syncthreads();
        if (chunk + 1 < chunks) commit(next_k);
        __syncthreads();
        current_idx   = next_idx;
        future_idx    = after_idx;
        carried_score = next_score;
    }
#    pragma unroll
    for (int hs = 0; hs < 2; hs++) {
        const int h = hs * 32 + warp * 16 + r;
        bf16* dest  = dq + ((size_t)row * 64 + h) * D;
#    pragma unroll
        for (int dc = 0; dc < ND; dc += 2) {
            uint32_t v[4], x[4], y[4];
#    pragma unroll
            for (int z = 0; z < 4; z++)
                v[z] = hip_dsa_numeric::bf16_pair(sm_scale * dqacc[hs][dc].x[z], sm_scale * dqacc[hs][dc + 1].x[z]);
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
            *(uint4*)(dest + (dc / 2) * 32 + g * 8) = make_uint4(y[0], y[1], y[2], y[3]);
        }
        float v  = (dwacc[hs][0] + dwacc[hs][1] + dwacc[hs][2] + dwacc[hs][3]);
        v       += __shfl_xor(v, 16);
        v       += __shfl_xor(v, 32);
        if (g == 0) dw[(size_t)row * 64 + h] = float_to_bf16(v);
    }
    if (done_count == nullptr) return;
    __threadfence();
    __syncthreads();
    __shared__ int last_block;
    if (threadIdx.x == 0) last_block = atomicAdd(done_count, 1) == gridDim.x - 1;
    __syncthreads();
    if (last_block) {
        const size_t n4 = (size_t)B * Sk * D / 4;
        for (size_t i = threadIdx.x; i < n4; i += 128) {
            const float4 v     = idxb_cast_load<true>((const float4*)dk_float, i);
            ((uint64_t*)dk)[i] = (uint64_t)float_to_bf16(v.x) | ((uint64_t)float_to_bf16(v.y) << 16) |
                                 ((uint64_t)float_to_bf16(v.z) << 32) | ((uint64_t)float_to_bf16(v.w) << 48);
        }
    }
#endif
}

static bool indexer_bwd_sparse_mmac_shm_fits(int Hq, int D, int topk) {
    if (D <= 0 || D % 16 != 0 || D > 128 || Hq > 64 || Hq < 1 || topk > 2048) return false;
    const int KLD = D + 8;
    size_t shm    = (size_t)IDXB_CHUNK * KLD * 2 + (size_t)topk * 4  // idx
                    + (size_t)topk * 4                               // score
                    + (size_t)IDXB_THREADS * 4                       // red
                    + (size_t)4 * (IDXB_CHUNK / 16) * 320 * 2        // mask tiles
                    + (size_t)Hq * 4;                                // w_lds
    return shm <= 65536;
}

// Large dk outputs should not be converted by one last CTA.
template<bool SWIZZLED = false>
__global__ static void indexer_dk_cast_kernel(const float* src, bf16* dst, size_t n4) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        const float4 v        = idxb_cast_load<SWIZZLED>((const float4*)src, i);
        const uint64_t packed = (uint64_t)float_to_bf16(v.x) | ((uint64_t)float_to_bf16(v.y) << 16) |
                                ((uint64_t)float_to_bf16(v.z) << 32) | ((uint64_t)float_to_bf16(v.w) << 48);
        ((uint64_t*)dst)[i]   = packed;
    }
}

template<typename GradScale>
static bool launch_indexer_bwd_sparse_mmac(hipStream_t stream, const void* index_q, const void* weights,
                                           const void* index_k, float* attn_score, const float* index_score,
                                           const int* topk_indices, void* dq, void* dw, void* dk_float, void* dk,
                                           int* done_count, int B, int Sq, int Sk, int Hq, int D, int topk,
                                           float sm_scale, float loss_coeff, GradScale grad_loss,
                                           bool topk_indices_global) {
    const size_t dk_elements = (size_t)B * Sk * D;
    // A 128K-element accumulator already benefits from distributing the
    // final cast across CTAs; smaller outputs keep the fused last-CTA path.
    const bool parallel_cast = dk_elements >= (1u << 17);
    int* kernel_done_count   = parallel_cast ? nullptr : done_count;
    const int KLD            = D + 8;
    size_t shm = (size_t)IDXB_CHUNK * KLD * 2 + (size_t)topk * 4 + (size_t)topk * 4 + (size_t)IDXB_THREADS * 4 +
                 (size_t)4 * (IDXB_CHUNK / 16) * 320 * 2 + (size_t)Hq * 4;
    dim3 grid(B * Sq);
    const int nw = (Hq + 15) / 16;
    if (nw < 1 || nw > 4) return false;
    if (D == 128 && Hq == 32 && topk == 2048 && Sq == 4096 && Sk == 4096) {
        shm -= (size_t)topk * (sizeof(float) + sizeof(int));
        {
            const hipError_t status =
                hipFuncSetAttribute((const void*)indexer_bwd_sparse_mmac_kernel<128, 2, true, true, true, GradScale>,
                                    hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipFuncSetAttribute failed: %s",
                                        hipGetErrorString(status));
            }
        }
        hipLaunchKernelGGL((indexer_bwd_sparse_mmac_kernel<128, 2, true, true, true, GradScale>), grid, IDXB_THREADS,
                           shm, stream, (const bf16*)index_q, (const bf16*)weights, (const bf16*)index_k, attn_score,
                           index_score, topk_indices, (bf16*)dq, (bf16*)dw, (float*)dk_float, (bf16*)dk,
                           kernel_done_count, B, Sq, Sk, Hq, topk, sm_scale, loss_coeff, grad_loss,
                           topk_indices_global);
    } else if (D == 128 && Hq == 64 && ((topk == 1024 && Sq >= 2048) || (topk == 2048 && Sq == 4096 && Sk == 4096))) {
        // The existing target output holds grad_signal after the read barrier.
        // Read immutable indices through cache as well, keeping LDS below
        // the four-CTA residency threshold. Other top-k sizes retain
        // the measured faster LDS-score path. No user configuration required.
        shm -= (size_t)topk * (sizeof(float) + sizeof(int));
        {
            const hipError_t status = hipFuncSetAttribute((const void*)indexer_bwd_sparse_two_wave_kernel<GradScale>,
                                                          hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipFuncSetAttribute failed: %s",
                                        hipGetErrorString(status));
            }
        }
        hipLaunchKernelGGL((indexer_bwd_sparse_two_wave_kernel<GradScale>), grid, 128, shm, stream,
                           (const bf16*)index_q, (const bf16*)weights, (const bf16*)index_k, attn_score, index_score,
                           topk_indices, (bf16*)dq, (bf16*)dw, (float*)dk_float, (bf16*)dk, kernel_done_count, B, Sq,
                           Sk, Hq, topk, sm_scale, loss_coeff, grad_loss, topk_indices_global);
    } else if (D == 128 && Hq == 64 && topk % IDXB_CHUNK == 0) {
        {
            const hipError_t status =
                hipFuncSetAttribute((const void*)indexer_bwd_sparse_mmac_kernel<128, 4, true, false, false, GradScale>,
                                    hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipFuncSetAttribute failed: %s",
                                        hipGetErrorString(status));
            }
        }
        hipLaunchKernelGGL((indexer_bwd_sparse_mmac_kernel<128, 4, true, false, false, GradScale>), grid, IDXB_THREADS,
                           shm, stream, (const bf16*)index_q, (const bf16*)weights, (const bf16*)index_k, attn_score,
                           index_score, topk_indices, (bf16*)dq, (bf16*)dw, (float*)dk_float, (bf16*)dk,
                           kernel_done_count, B, Sq, Sk, Hq, topk, sm_scale, loss_coeff, grad_loss,
                           topk_indices_global);
    } else {
#define IDXB_SETATTR(DD, NWW)                                                                                         \
    do {                                                                                                              \
        const hipError_t status =                                                                                     \
            hipFuncSetAttribute((const void*)indexer_bwd_sparse_mmac_kernel<DD, NWW, false, false, false, GradScale>, \
                                hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);                                \
        if (status != hipSuccess) {                                                                                   \
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipFuncSetAttribute failed: %s",                   \
                                    hipGetErrorString(status));                                                       \
        }                                                                                                             \
    } while (0)
#define IDXB_LAUNCH(DD, NWW)                                                                                          \
    hipLaunchKernelGGL((indexer_bwd_sparse_mmac_kernel<DD, NWW, false, false, false, GradScale>), grid, IDXB_THREADS, \
                       shm, stream, (const bf16*)index_q, (const bf16*)weights, (const bf16*)index_k, attn_score,     \
                       index_score, topk_indices, (bf16*)dq, (bf16*)dw, (float*)dk_float, (bf16*)dk,                  \
                       kernel_done_count, B, Sq, Sk, Hq, topk, sm_scale, loss_coeff, grad_loss, topk_indices_global)
        switch (D) {
#define IDXB_CASE(DD)                \
    case DD:                         \
        switch (nw) {                \
            case 1:                  \
                IDXB_SETATTR(DD, 1); \
                IDXB_LAUNCH(DD, 1);  \
                break;               \
            case 2:                  \
                IDXB_SETATTR(DD, 2); \
                IDXB_LAUNCH(DD, 2);  \
                break;               \
            case 3:                  \
                IDXB_SETATTR(DD, 3); \
                IDXB_LAUNCH(DD, 3);  \
                break;               \
            case 4:                  \
                IDXB_SETATTR(DD, 4); \
                IDXB_LAUNCH(DD, 4);  \
                break;               \
            default: return false;   \
        }                            \
        break;
            IDXB_CASE(16);
            IDXB_CASE(32);
            IDXB_CASE(48);
            IDXB_CASE(64);
            IDXB_CASE(80);
            IDXB_CASE(96);
            IDXB_CASE(112);
            IDXB_CASE(128);
#undef IDXB_CASE
            // Never report a successful launch when no specialization ran.
            default: return false;
        }
#undef IDXB_SETATTR
#undef IDXB_LAUNCH
    }
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_indexer_bwd_sparse HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    if (parallel_cast) {
        const size_t n4 = dk_elements / 4;
        if (D == 128 && (nw == 1 || nw == 2 || nw == 4)) {
            hipLaunchKernelGGL((indexer_dk_cast_kernel<true>), dim3((n4 + 255) / 256), 256, 0, stream,
                               (const float*)dk_float, (bf16*)dk, n4);
        } else {
            hipLaunchKernelGGL((indexer_dk_cast_kernel<false>), dim3((n4 + 255) / 256), 256, 0, stream,
                               (const float*)dk_float, (bf16*)dk, n4);
        }
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_indexer_bwd_sparse HIP call failed: %s",
                                        hipGetErrorString(status));
            }
        }
    }
    return true;
}

// Dense BF16 backward lives with the native sparse/dense variants above.
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#    if defined(__gfx936__) || defined(__gfx938__)
using dm_floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
#    endif

// ---------------------------------------------------------------------------
// Dense backward consumes the forward FP32 score/LSE to form the KL gradient.
// Each query then streams K once to recover per-head ReLU scores, accumulate
// dW in FP32, and contract BF16 dS with K/Q. dK accumulates in FP32 before
// the last block converts it to the requested output dtype.
// ---------------------------------------------------------------------------

// Load ck K-rows of chunk `chunk` from a packed/offset K base (krow) — the
// THD / per-segment analogue of dm_load_k (no per-batch Sk stride).

// Generic dtype path for native fp16/fp32 MMAC.  All-bf16 calls use the
// established optimized kernels above.

#else

#endif
// ---------------------------------------------------------------------------
// Dense indexer backward: ONE fused kernel for the whole (BSHD or THD) batch.
// dk_float (dk_rows_total x D f32) and done_count (1 int) are zeroed by the
// caller on the stream; the kernel's last block casts dk_float -> dk (bf16)
// in place, so the operator is exactly one kernel launch with no global
// dscore/A/partial workspaces and no persistent hipMalloc buffers.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Launch functions
// ---------------------------------------------------------------------------
static bool launch_indexer_bwd_sparse(hipStream_t stream, const void* index_q, const void* weights, const void* index_k,
                                      float* attn_score, const float* index_score, const int* topk_indices, void* dq,
                                      void* dw, void* dk, float* dk_float, int* done_count, int B, int Sq, int Sk,
                                      int H, int D, int topk, float sm_scale, float loss_coeff, float grad_loss,
                                      bool topk_indices_global, const float* grad_loss_device) {
    if (!(dense_mmac_supported() && indexer_bwd_sparse_mmac_shm_fits(H, D, topk))) { return false; }
    if (grad_loss_device != nullptr) {
        return launch_indexer_bwd_sparse_mmac(stream, index_q, weights, index_k, attn_score, index_score, topk_indices,
                                              dq, dw, dk_float, dk, done_count, B, Sq, Sk, H, D, topk, sm_scale,
                                              loss_coeff, grad_loss_device, topk_indices_global);
    }
    return launch_indexer_bwd_sparse_mmac(stream, index_q, weights, index_k, attn_score, index_score, topk_indices, dq,
                                          dw, dk_float, dk, done_count, B, Sq, Sk, H, D, topk, sm_scale, loss_coeff,
                                          grad_loss, topk_indices_global);
}

// ---------------------------------------------------------------------------
// THD (packed variable-length) dense indexer backward
//
// Same math as the BSHD dense backward, over packed rows:
//   index_q : (total_q, H, D), weights : (total_q, H), index_k : (total_k, D)
//   attn_score / raw_scores : (total_q, max_sk), attn_l1norm : (total_q,)
//   dq : (total_q, H, D), dw : (total_q, H), dk : (total_k, D)
// Column ``k_idx`` of row ``row`` maps to packed K row ``cu_seqlens_kv[b]+k_idx``.
// ---------------------------------------------------------------------------

namespace ftrain {

bool bf16IndexerBwdSparseLaunchFits(int heads, int width, int columns, int queries) {
    if (columns <= 0 || columns > 2048) return false;
    std::size_t bytes =
        16ULL * (width + 8) * 2 + static_cast<std::size_t>(columns) * 8 + 256 * 4 + 4 * 320 * 2 + heads * 4;
    if (width == 128 && heads == 64 && columns == 1024 && queries >= 2048) bytes -= columns * 8;
    return dsa_shared_memory_fits(bytes);
}

bool launchBf16IndexerBwdSparse(FTrainStream stream, const void* index_q, const void* weights, const void* index_k,
                                float* attn_score, const float* index_score, const int* topk_indices, void* dq,
                                void* dw, void* dk, float* dk_float, int* done_count, int batch, int query_length,
                                int key_length, int heads, int head_dim, int selection_width, float sm_scale,
                                float loss_coeff, float grad_loss, bool topk_indices_global,
                                const float* grad_loss_device) {
    return launch_indexer_bwd_sparse(stream, index_q, weights, index_k, attn_score, index_score, topk_indices, dq, dw,
                                     dk, dk_float, done_count, batch, query_length, key_length, heads, head_dim,
                                     selection_width, sm_scale, loss_coeff, grad_loss, topk_indices_global,
                                     grad_loss_device);
}

bool bf16IndexerBwdSparseDenseMmacSupported() { return dense_mmac_supported(); }

void bf16IndexerBwdSparseZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    const hipError_t status = hipMemsetAsync(memory, 0, static_cast<std::size_t>(bytes), stream);
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Workspace zero-fill failed: %s", hipGetErrorString(status));
    }
}

}  // namespace ftrain
