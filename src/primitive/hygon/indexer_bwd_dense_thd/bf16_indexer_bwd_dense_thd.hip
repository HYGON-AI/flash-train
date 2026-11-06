// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <math.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx936__) || defined(__gfx938__)
#    include <du_mma.h>
#endif

#include "flash_train/error.hpp"
#include <stdint.h>

namespace hip_dsa_numeric {
// Small counter/accumulator resets are cheaper as a stream-ordered kernel
// than a graph memcpy/memset node. Large or byte-aligned regions keep HIP's memset.
static __global__ void zero_workspace_words(uint32_t* memory, uint64_t words) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < words) memory[i] = 0;
}
static inline hipError_t zero_workspace(void* memory, uint64_t bytes, hipStream_t stream) {
    if (bytes == 0) return hipSuccess;
    if (bytes <= 65536 && bytes % 4 == 0 && (reinterpret_cast<uintptr_t>(memory) & 3) == 0) {
        hipLaunchKernelGGL(zero_workspace_words, dim3((bytes + 1023) / 1024), dim3(256), 0, stream,
                           static_cast<uint32_t*>(memory), bytes / 4);
        return hipGetLastError();
    }
    return hipMemsetAsync(memory, 0, bytes, stream);
}


}  // namespace hip_dsa_numeric
#include "flash_train/primitive/hygon/indexer_bwd_dense_thd/bf16_indexer_bwd_dense_thd.hpp"

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

__device__ __forceinline__ static void dsa_dense_score_grad(const float* target, float target_norm, float* logits,
                                                            float logsumexp, float* score, float* reduction,
                                                            int columns, int valid, float scale) {
    const float eps = expf(-100.0f);
    float local     = 0.0f;
    if (threadIdx.x < 128) {
        for (int i = threadIdx.x; i < valid; i += 128) {
            const float t          = target[i] / (target_norm + 1e-10f);
            const float effective  = t >= eps ? t : eps;
            const float gate       = logits[i] - logsumexp >= -100.0f ? 1.0f : 0.0f;
            local                 += -effective * gate * scale;
        }
        for (int offset = 16; offset; offset >>= 1) local += __shfl_down(local, offset, 32);
        if ((threadIdx.x & 31) == 0) reduction[threadIdx.x >> 5] = local;
    }
    __syncthreads();
    const float sum = reduction[0] + reduction[1] + reduction[2] + reduction[3];
    for (int i = threadIdx.x; i < columns; i += blockDim.x) {
        float value = 0.0f;
        if (i < valid) {
            const float log_p     = logits[i] - logsumexp;
            const float t         = target[i] / (target_norm + 1e-10f);
            const float effective = t >= eps ? t : eps;
            const float gate      = log_p >= -100.0f ? 1.0f : 0.0f;
            value                 = -effective * gate * scale - expf(log_p) * sum;
        }
        score[i]  = value;
        logits[i] = value;
    }
    __syncthreads();
}

#define DENSE_CHUNK 32
#define DENSE_THREADS 256
#define DENSE_DCHUNK 128

struct ThdRowMeta {
    int batch_id;    // segment (batch) index owning this row
    int pos_in_seq;  // row position inside its segment
    int kv_offset;   // first packed KV row of the segment
    int seqlen_kv;   // number of packed KV rows of the segment
};

__device__ __forceinline__ static int thd_batch_of_row(const int* __restrict__ cu_seqlens_q, int n_seqs, int row) {
    int lo = 0, hi = n_seqs;  // invariant: cu[lo] <= row < cu[hi]
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (cu_seqlens_q[mid] <= row)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

__device__ __forceinline__ static ThdRowMeta thd_row_meta(const int* __restrict__ cu_seqlens_q,
                                                          const int* __restrict__ cu_seqlens_kv, int n_seqs, int row,
                                                          int total_k) {
    ThdRowMeta m;
    m.batch_id   = thd_batch_of_row(cu_seqlens_q, n_seqs, row);
    const int qs = cu_seqlens_q[m.batch_id], qe = cu_seqlens_q[m.batch_id + 1];
    const int ks = cu_seqlens_kv[m.batch_id], ke = cu_seqlens_kv[m.batch_id + 1];
    const bool valid = qs >= 0 && qs <= row && row < qe && ks >= 0 && ks <= ke && ke <= total_k;
    m.pos_in_seq     = valid ? row - qs : 0;
    m.kv_offset      = valid ? ks : 0;
    m.seqlen_kv      = valid ? ke - ks : 0;
    return m;
}

__device__ __forceinline__ static int thd_col_limit(const ThdRowMeta& m, int ratio) {
    int lim = (int)(((int64_t)m.pos_in_seq + 1) / ratio);
    if (lim > m.seqlen_kv) lim = m.seqlen_kv;
    return lim;
}

__device__ __forceinline__ static int thd_col_limit(const ThdRowMeta& m, int ratio,
                                                    const int* __restrict__ q_causal_offsets) {
    const int offset = q_causal_offsets == nullptr ? 0 : q_causal_offsets[m.batch_id];
    int64_t lim      = ((int64_t)offset + m.pos_in_seq + 1) / ratio;
    if (lim < 0) lim = 0;
    if (lim > m.seqlen_kv) lim = m.seqlen_kv;
    return lim;
}

// Adapted from fused_dsa/hip_dsa/kernels/indexer_bwd.hip.

typedef uint16_t bf16;

#define IDXB_CHUNK 16  // 16-key chunk: half the K arena -> more CTAs/CU
#define IDXB_THREADS 256

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

#if defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;
typedef __hip_bfloat16 hbf16;

#else  // no MMAC intrinsics on this arch

#endif

#if defined(__gfx936__) || defined(__gfx938__)
using dm_floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

__device__ __forceinline__ static void dm_atomic_add_mmac_row(float* row16, int g, float x0, float x1, float x2,
                                                              float x3) {
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
    dm_floatx4 v;
    v[0] = (g & 2) ? q2 : y0;
    v[1] = (g & 2) ? q3 : y1;
    v[2] = (g & 2) ? y2 : q0;
    v[3] = (g & 2) ? y3 : q1;
#        if defined(__gfx938__)
    __builtin_hcu_global_atomic_fadd_f32_x4(row16 + 4 * g, v);
#        else
    __builtin_amdgcn_global_atomic_fadd_v4f32((dm_floatx4*)(row16 + 4 * g), v);
#        endif
}

__device__ __forceinline__ static void dm_load_k_from(hbf16* k_lds, const bf16* __restrict__ krow, int chunk, int ck,
                                                      int D, int KLD) {
    for (int u = threadIdx.x; u < ck * (D / 8); u += IDXB_THREADS) {
        int n             = u / (D / 8);
        int d8            = u % (D / 8);
        const bf16* src   = krow + (size_t)(chunk + n) * D + d8 * 8;
        bf16* dst         = (bf16*)(k_lds + (size_t)n * KLD + d8 * 8);
        ((float2*)dst)[0] = ((const float2*)src)[0];
        ((float2*)dst)[1] = ((const float2*)src)[1];
    }
}

template<int D>
static __global__ void __launch_bounds__(IDXB_THREADS)
    dense_indexer_bwd_fused_kernel(const bf16* __restrict__ q,            // BSHD: (B, Sq, Hq, D); THD: (total_q, Hq, D)
                                   const bf16* __restrict__ w,            // (rows, Hq)
                                   const bf16* __restrict__ k,            // BSHD: (B, Sk, D);  THD: (total_k, D)
                                   const float* __restrict__ attn_score,  // (rows, score_stride), read-only
                                   const float* __restrict__ attn_l1norm,  // (rows,)
                                   float* __restrict__ index_score,        // (rows, score_stride) IN-PLACE:
                                                                           // overwritten with grad_signal (cuDNN
                                                                           // dense contract, PredictRaw).
                                   const float* __restrict__ index_lse,    // supplied FP32 normalization
                                   bf16* __restrict__ dq,                  // (rows, Hq, D)
                                   bf16* __restrict__ dw,                  // (rows, Hq)
                                   float* __restrict__ dk_float,           // (dk_rows_total, D) f32, zeroed by caller
                                   bf16* __restrict__ dk,                  // (dk_rows_total, D) bf16 output
                                   int* __restrict__ done_count,           // zeroed by caller; last-block cast sync
                                   const int* __restrict__ cu_q,           // THD: (n_seqs + 1); NULL for BSHD
                                   const int* __restrict__ cu_kv,          // THD: (n_seqs + 1); NULL for BSHD
                                   const int* __restrict__ q_causal_offsets,  // THD: (n_seqs), optional
                                   int rows, int Sq, int Sk, int Hq, int D_rt, int n_seqs, int ratio, float sm_scale,
                                   float loss_coeff, float grad_loss, int score_stride, int dk_rows_total) {
    constexpr bool PACK_DQ = D % 32 == 0;
    constexpr int nd   = D / 16;  // d-tiles (8 for D=128)
    constexpr int KLD  = D + 8;
    constexpr int ALDS = IDXB_CHUNK + 2;
    // dk d-column split: when D divides into four 16-col tile groups
    // (D=64/128), each warp owns D/4 columns, so every (k, d) element is
    // flushed to the global dk accumulator exactly once per block instead
    // of once per head-warp (4x fewer global dk atomics).  dq keeps the
    // full D per warp (see pass 2).
    constexpr bool dsplit = (nd % 4) == 0;
    constexpr int ndq     = dsplit ? nd / 4 : nd;  // d-tiles per warp for dk
    (void)D_rt;
    const int row    = blockIdx.x;
    const int lane   = threadIdx.x & 63;
    const int warp   = threadIdx.x >> 6;
    const int g      = lane >> 4;
    const int r      = lane & 15;
    const int Hq_pad = (Hq + 15) & ~15;
    const int r0     = warp * 16;
    const int h      = r0 + r;
    const int d0     = dsplit ? (warp * ndq) * 16 : 0;  // this warp's d-column base

    int kv_offset = 0;
    int col_limit = 0;
    if (cu_q != nullptr) {
        ThdRowMeta m = thd_row_meta(cu_q, cu_kv, n_seqs, row, dk_rows_total);
        kv_offset    = m.kv_offset;
        col_limit    = thd_col_limit(m, ratio, q_causal_offsets);
    } else {
        const int sq = row % Sq;
        kv_offset    = (row / Sq) * Sk;
        col_limit    = (sq + 1) / ratio;
        if (col_limit > Sk) col_limit = Sk;
    }
    if (col_limit > score_stride) col_limit = score_stride;
    const int n_valid = col_limit;
    const bf16* krow  = k + (size_t)kv_offset * D;

    extern __shared__ char smem_raw[];
    hbf16* k_lds = (hbf16*)smem_raw;                            // [IDXB_CHUNK][KLD]
    float* score = (float*)(k_lds + (size_t)IDXB_CHUNK * KLD);  // [Sk]
    float* red   = score + Sk;                                  // [IDXB_THREADS]
    hbf16* a_lds = (hbf16*)(red + IDXB_THREADS);                // [Hq_pad][ALDS]

    const float w_h = (h < Hq) ? bf16_to_float(w[(size_t)row * Hq + h]) : 0.0f;

    // Cache Q fragments for the per-head scores and this warp's dK slice.
    constexpr int hgmax = dsplit ? 4 : 1;  // head-groups contracted by dk
    DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> af[nd];
    // bf[hg][dc]: Q fragments (head-group hg, this warp's d-tile dc) for the
    // dk MMAs.  With the d-split every warp must contract ALL head groups
    // (dk[k, d] sums over every h), so all valid groups are hoisted.
    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bf[hgmax][ndq];
    // Q fragments are loaded per-lane with rows past Hq clamped to row Hq-1
    // and zeroed: a plain du_load_matrix_sync of a 16-row fragment would
    // read past the head block (out-of-bounds at the last query row) and
    // feed stale/NaN data into the dk contraction (0 * stale = NaN).
    const int h_lane = r0 + (lane & 15);
    const int qrow_h = (h_lane < Hq) ? h_lane : (Hq - 1);
    if (r0 < Hq) {
        const hbf16* qrow = (const hbf16*)(q + (size_t)row * Hq * D + (size_t)qrow_h * D);
#    pragma unroll
        for (int kk = 0; kk < nd; kk++) {
            *(uint64_t*)&af[kk].x[0] = *(const uint64_t*)(qrow + kk * 16 + (g << 2));
            if (h_lane >= Hq) af[kk].x[0] = af[kk].x[1] = af[kk].x[2] = af[kk].x[3] = hbf16(0);
        }
    }
// bf[] is consumed by the dk MMAs of EVERY warp (each warp owns a d
// slice), including phantom-head warps, so hoist it unconditionally
// (rows past Hq clamped + zeroed, same as af[]).
#    pragma unroll
    for (int hg = 0; hg < hgmax; hg++) {
        const int hrow = dsplit ? hg * 16 : r0;
        if (hrow < Hq) {
            const int fr    = hrow + (g << 2);
            const hbf16* qh = (const hbf16*)(q + ((size_t)row * Hq + (size_t)hrow) * D + (size_t)(d0 + r));
#    pragma unroll
            for (int kk = 0; kk < ndq; kk++) {
                const hbf16* qc = qh + kk * 16;
                bf[hg][kk].x[0] = (fr < Hq) ? qc[(size_t)(g << 2) * D] : hbf16(0);
                bf[hg][kk].x[1] = (fr + 1 < Hq) ? qc[((size_t)(g << 2) + 1) * D] : hbf16(0);
                bf[hg][kk].x[2] = (fr + 2 < Hq) ? qc[((size_t)(g << 2) + 2) * D] : hbf16(0);
                bf[hg][kk].x[3] = (fr + 3 < Hq) ? qc[((size_t)(g << 2) + 3) * D] : hbf16(0);
            }
        }
    }

    for (int i = threadIdx.x; i < Sk; i += IDXB_THREADS) score[i] = 0.0f;
    // Zero k_lds once at kernel start: partial tail chunks leave rows beyond
    // ck untouched, and both MMA passes read all 16 rows of a B fragment.
    // Stale LDS (e.g. an INF/NaN bit pattern left by a previous kernel)
    // would turn 0 * stale into NaN and poison dq/dk accumulators.
    for (int i = threadIdx.x; i < IDXB_CHUNK * KLD; i += IDXB_THREADS) k_lds[i] = hbf16(0);
    // Zero a_lds with ALL threads: phantom-head rows (h >= Hq, e.g. H=8 with
    // Hq_pad=16) are never written by the per-chunk A build, but the dk MMAs
    // contract over all 16 rows of the A fragment.  Stale phantom rows (an
    // INF/NaN bit pattern left by a previous kernel) would corrupt every dk
    // column; dq is row-independent and would stay clean, masking the bug.
    for (int i = threadIdx.x; i < Hq_pad * ALDS; i += IDXB_THREADS) a_lds[i] = hbf16(0);
    __syncthreads();

    dsa_dense_score_grad(attn_score + (size_t)row * score_stride, attn_l1norm[row],
                         index_score + (size_t)row * score_stride, index_lse[row], score, red, Sk, n_valid,
                         (loss_coeff / (float)rows) * grad_loss);

    // ---- Per-head ReLU derivative, dW, dQ and dK per K chunk.
    // dq keeps the full D per warp (each (h, d) element belongs to the warp
    // owning head h); only dk is d-split across warps (its partials over
    // disjoint d-slices are flushed once per (k, d) element).
    DUFragment<accumulator, 16, 16, 16, float> dqacc[nd];
    DUFragment<accumulator, 16, 16, 16, float> dkacc[ndq];
#    pragma unroll
    for (int dc = 0; dc < nd; dc++) du_fill_fragment(dqacc[dc], 0.0f);
#    pragma unroll
    for (int dc = 0; dc < ndq; dc++) du_fill_fragment(dkacc[dc], 0.0f);
    float my_dw = 0.0f;

    for (int chunk = 0; chunk < col_limit; chunk += IDXB_CHUNK) {
        int ck = col_limit - chunk;
        if (ck > IDXB_CHUNK) ck = IDXB_CHUNK;
        const int ntiles = (ck + 15) >> 4;

        dm_load_k_from(k_lds, krow, chunk, ck, D, KLD);
        __syncthreads();

        // Recompute the per-head scores for this chunk.
        constexpr int MAXT2 = IDXB_CHUNK / 16;
        DUFragment<accumulator, 16, 16, 16, float> acc[MAXT2];
#    pragma unroll
        for (int nt = 0; nt < MAXT2; nt++) {
            if (nt < ntiles) du_fill_fragment(acc[nt], 0.0f);
        }
        for (int kk = 0; kk < nd; kk++) {
#    pragma unroll
            for (int nt = 0; nt < MAXT2; nt++) {
                if (nt < ntiles) {
                    DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr;
                    du_load_matrix_sync(bfr, k_lds + (size_t)(nt * 16) * KLD + kk * 16, KLD);
                    if (r0 < Hq) du_mma_sync(acc[nt], af[kk], bfr, acc[nt]);
                }
            }
        }

        // Build the bf16 A tile (head-major LDS: a_lds[h * ALDS + k]) and
        // accumulate the dw partial.  acc.x[i] = C[r][g + 4*i]: (h, k),
        // i.e. lane group g owns K columns g, g+4, g+8, g+12 of the tile
        // (same mapping as dm_score_chunk_epilogue and the dq/dk K tiles).
        if (r0 < Hq) {
            if (h < Hq) {
#    pragma unroll
                for (int nt = 0; nt < MAXT2; nt++) {
                    if (nt < ntiles) {
#    pragma unroll
                        for (int i = 0; i < 4; i++) {
                            const int kk = nt * 16 + g + 4 * i;
                            if (kk < ck) {
                                const float s                = acc[nt].x[i] * sm_scale;
                                const float pre              = (s > 0.0f) ? s : 0.0f;
                                const float dsc              = score[chunk + kk];
                                a_lds[(size_t)h * ALDS + kk] = (pre > 0.0f) ? float_to_bf16(dsc * w_h) : (bf16)0;
                                // CuTe keeps the ReLU score and dW product in FP32.
                                my_dw += dsc * pre;
                            }
                        }
                    }
                }
            }
            // Zero the tail columns [ck, ALDS) so partial-tile MMACs in dq/dk
            // never multiply stale A from the previous chunk.
            for (int kk = ck + g; kk < ALDS; kk += 4) a_lds[(size_t)h * ALDS + kk] = hbf16(0);
        }
        // Block-wide sync: every warp's A rows (and the zeroed phantom rows)
        // are visible before dq/dk contract over a_lds.  Must be outside the
        // r0 < Hq guard -- a conditional __syncthreads__ is UB.
        __syncthreads();

        // dq: dqacc[h, d] += A'[h, k] @ K[k, d]  (A head-major, rows h) --
        // only the warp owning head h touches (h, :), so valid warps only.
        // dk: dkacc[k, d] += A[k, h] @ Q[h, d]  (A col-major, rows k).
        //      The d-split makes each warp the sole writer of its d slice,
        //      so ALL warps (even phantom-head ones) must run the dk MMAs
        //      and contract every valid head group of a_lds.
        const int nksub = (ck + 15) >> 4;
        for (int ksub = 0; ksub < nksub; ksub++) {
            const int kk16 = ksub * 16;
#    pragma unroll
            for (int dc = 0; dc < ndq; dc++) du_fill_fragment(dkacc[dc], 0.0f);
            if (r0 < Hq) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> afr;
                du_load_matrix_sync(afr, a_lds + (size_t)r0 * ALDS + kk16, ALDS);
                if constexpr (PACK_DQ) {
                    typedef unsigned int mat4 __attribute__((ext_vector_type(4)));
#    pragma unroll
                    for (int dc = 0; dc < nd; dc += 2) {
                        const unsigned addr = (unsigned)(size_t)(k_lds + (size_t)kk16 * KLD + dc * 16 +
                                                                 (lane / 4) * KLD + (lane % 4) * 8);
                        mat4 packed;
                        asm volatile("ds_read_m32x16_b16_alt %0, %1" : "=v"(packed) : "v"(addr) : "memory");
                        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                        DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                        *(uint64_t*)&b0.x[0] = ((uint64_t*)&packed)[0];
                        *(uint64_t*)&b1.x[0] = ((uint64_t*)&packed)[1];
                        du_mma_sync(dqacc[dc], afr, b0, dqacc[dc]);
                        du_mma_sync(dqacc[dc + 1], afr, b1, dqacc[dc + 1]);
                    }
                } else {
#    pragma unroll
                    for (int dc = 0; dc < nd; dc++) {
                        DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bfr;
                        du_load_matrix_sync(bfr, k_lds + (size_t)kk16 * KLD + dc * 16, KLD);
                        du_mma_sync(dqacc[dc], afr, bfr, dqacc[dc]);
                    }
                }
            }
#    pragma unroll
            for (int hg = 0; hg < hgmax; hg++) {
                const int hrow = dsplit ? hg * 16 : r0;
                if (hrow < Hq) {
                    DUFragment<matrix_a, 16, 16, 16, hbf16, col_major> afc;
                    du_load_matrix_sync(afc, a_lds + (size_t)hrow * ALDS + kk16, ALDS);
#    pragma unroll
                    for (int dc = 0; dc < ndq; dc++) { du_mma_sync(dkacc[dc], afc, bf[hg][dc], dkacc[dc]); }
                }
            }
            // Flush this k-subtile's 16 k rows to the f32 accumulator.
            // With the d-split each (k, d) element is owned by exactly
            // one warp, so this is one atomic per element per block.
            if (kk16 + r < ck) {
                const int krow = kv_offset + chunk + kk16 + r;
#    pragma unroll
                for (int dc = 0; dc < ndq; dc++) {
                    float* pk = dk_float + (size_t)krow * D + d0 + dc * 16;
                    dm_atomic_add_mmac_row(pk, g, sm_scale * dkacc[dc].x[0], sm_scale * dkacc[dc].x[1],
                                           sm_scale * dkacc[dc].x[2], sm_scale * dkacc[dc].x[3]);
                }
            }
        }
        __syncthreads();
    }

    // ---- dq epilogue (all heads / d columns written exactly once). ----
    if (h < Hq) {
        bf16* dqbase = dq + ((size_t)row * Hq + h) * D;
        if constexpr (PACK_DQ) {
// A 32-bit word carries two BF16 payloads; no FP32 arithmetic is
// replaced by BF16, and sm_scale is still evaluated in FP32.
#    pragma unroll
            for (int dc = 0; dc < nd; dc += 2) {
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    *(uint32_t*)(dqbase + (dc / 2) * 32 + g * 2 + j * 8) =
                        hip_dsa_numeric::bf16_pair(sm_scale * dqacc[dc].x[j], sm_scale * dqacc[dc + 1].x[j]);
                }
            }
        } else {
#    pragma unroll
            for (int dc = 0; dc < nd; dc++) {
                const int col    = dc * 16 + g;
                dqbase[col]      = float_to_bf16(sm_scale * dqacc[dc].x[0]);
                dqbase[col + 4]  = float_to_bf16(sm_scale * dqacc[dc].x[1]);
                dqbase[col + 8]  = float_to_bf16(sm_scale * dqacc[dc].x[2]);
                dqbase[col + 12] = float_to_bf16(sm_scale * dqacc[dc].x[3]);
            }
        }
    }

    // ---- dw epilogue ----
    my_dw += __shfl_xor(my_dw, 16);
    my_dw += __shfl_xor(my_dw, 32);
    if (g == 0 && h < Hq) dw[(size_t)row * Hq + h] = float_to_bf16(my_dw);

    // ---- Last-block cast: dk_float (fp32 atomics) -> bf16 dk. ----
    __threadfence();
    __syncthreads();
    __shared__ int fused_last;
    if (threadIdx.x == 0) fused_last = (atomicAdd(done_count, 1) == gridDim.x - 1);
    __syncthreads();
    if (fused_last) {
        const long n = (long)dk_rows_total * D;
        // Only this block runs the cast, so cover the whole buffer with a
        // plain thread-stride loop (starting at threadIdx.x, not blockIdx.x).
        for (long i = threadIdx.x; i < n / 4; i += IDXB_THREADS) {
            const float4 v   = ((const float4*)dk_float)[i];
            const uint64_t a = float_to_bf16(v.x), b = float_to_bf16(v.y);
            const uint64_t c = float_to_bf16(v.z), d = float_to_bf16(v.w);
            ((uint64_t*)dk)[i] = a | (b << 16) | (c << 32) | (d << 48);
        }
    }
}

#else
template<int D>
static __global__ void __launch_bounds__(IDXB_THREADS)
    dense_indexer_bwd_fused_kernel(const bf16* __restrict__ q, const bf16* __restrict__ w, const bf16* __restrict__ k,
                                   const float* __restrict__ attn_score, const float* __restrict__ attn_l1norm,
                                   float* __restrict__ index_score, const float* __restrict__ index_lse,
                                   bf16* __restrict__ dq, bf16* __restrict__ dw, float* __restrict__ dk_float,
                                   bf16* __restrict__ dk, int* __restrict__ done_count, const int* __restrict__ cu_q,
                                   const int* __restrict__ cu_kv, const int* __restrict__ q_causal_offsets, int rows,
                                   int Sq, int Sk, int Hq, int D_rt, int n_seqs, int ratio, float sm_scale,
                                   float loss_coeff, float grad_loss, int score_stride, int dk_rows_total) {}

#endif

// Adapted from fused_dsa/hip_dsa/kernels/indexer_bwd.hip.
// Included after the common dense backward device helpers.
typedef __hip_bfloat16 hbf16;
#if (defined(__gfx936__) || defined(__gfx938__))
using idxb_floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
#endif

// MMAC accumulator lanes g=0..3 own columns {g,g+4,g+8,g+12}.  Transpose
// four participating lanes into contiguous four-float groups before the
// native x4 global atomic on the supported gfx936/gfx938 targets.
__device__ __forceinline__ void idxb_atomic_add_mmac_row(float* row16, int g, float x0, float x1, float x2, float x3) {
#if (defined(__gfx936__) || defined(__gfx938__))
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
    __builtin_amdgcn_global_atomic_fadd_v4f32((idxb_floatx4*)(row16 + 4 * g), v);
#else
    float* p = row16 + g;
    atomicAdd(&p[0], x0);
    atomicAdd(&p[4], x1);
    atomicAdd(&p[8], x2);
    atomicAdd(&p[12], x3);
#endif
}

template<bool SWIZZLED>
__device__ __forceinline__ void idxb_atomic_add_sparse(float* row16, int g, float x0, float x1, float x2, float x3) {
#if (defined(__gfx936__) || defined(__gfx938__))
    if constexpr (SWIZZLED) {
        idxb_floatx4 v = {x0, x1, x2, x3};
        __builtin_amdgcn_global_atomic_fadd_v4f32((idxb_floatx4*)(row16 + 4 * g), v);
        return;
    }
#endif
    idxb_atomic_add_mmac_row(row16, g, x0, x1, x2, x3);
}

// Exact HIP BF16 RNE, including NaN/Inf payload handling, without divergent branches.
__device__ __forceinline__ hbf16 idxb_rne(float f) { return __ushort_as_bfloat16(hip_dsa_numeric::bf16_rne(f)); }

template<bool SWIZZLED>
__device__ __forceinline__ float4 idxb_cast_load(const float4* src, size_t i) {
#if (defined(__gfx936__) || defined(__gfx938__))
    if constexpr (SWIZZLED) {
        const float* p = (const float*)src + (i / 4) * 16 + (i % 4);
        return make_float4(p[0], p[4], p[8], p[12]);
    }
#endif
    return src[i];
}
template<bool SWIZZLED = false>
static __global__ void indexer_dk_cast_kernel(const float* src, bf16* dst, size_t n4) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        const float4 v        = idxb_cast_load<SWIZZLED>((const float4*)src, i);
        const uint64_t packed = (uint64_t)float_to_bf16(v.x) | ((uint64_t)float_to_bf16(v.y) << 16) |
                                ((uint64_t)float_to_bf16(v.z) << 32) | ((uint64_t)float_to_bf16(v.w) << 48);
        ((uint64_t*)dst)[i]   = packed;
    }
}

template<int D>
static __global__ void __launch_bounds__(IDXB_THREADS)
    dense_bwd_norm_fallback_kernel(const bf16* __restrict__ q,            // BSHD: (B, Sq, Hq, D); THD: (total_q, Hq, D)
                                   const bf16* __restrict__ w,            // (rows, Hq)
                                   const bf16* __restrict__ k,            // BSHD: (B, Sk, D);  THD: (total_k, D)
                                   const float* __restrict__ attn_score,  // (rows, score_stride), read-only
                                   const float* __restrict__ attn_l1norm,  // (rows,)
                                   float* __restrict__ index_score,        // (rows, score_stride) IN-PLACE:
                                                                           // overwritten with grad_signal (cuDNN
                                                                           // dense contract, PredictRaw).
                                   const float* __restrict__ index_lse,    // supplied FP32 normalization
                                   bf16* __restrict__ dq,                  // (rows, Hq, D)
                                   bf16* __restrict__ dw,                  // (rows, Hq)
                                   float* __restrict__ dk_float,           // (dk_rows_total, D) f32, zeroed by caller
                                   bf16* __restrict__ dk,                  // (dk_rows_total, D) bf16 output
                                   int* __restrict__ done_count,           // zeroed by caller; last-block cast sync
                                   const int* __restrict__ cu_q,           // THD: (n_seqs + 1); NULL for BSHD
                                   const int* __restrict__ cu_kv,          // THD: (n_seqs + 1); NULL for BSHD
                                   const int* __restrict__ q_causal_offsets,  // THD: (n_seqs), optional
                                   int rows, int Sq, int Sk, int Hq, int D_rt, int n_seqs, int ratio, float sm_scale,
                                   float loss_coeff, float grad_loss, int score_stride, int dk_rows_total) {
#if (defined(__gfx936__) || defined(__gfx938__))

    constexpr bool PACK_DQ = D % 32 == 0;
    constexpr int nd       = D / 16;  // d-tiles (8 for D=128)
    constexpr int KLD      = D + 8;
    constexpr int ALDS     = IDXB_CHUNK + 2;
    // dk d-column split: when D divides into four 16-col tile groups
    // (D=64/128), each warp owns D/4 columns, so every (k, d) element is
    // flushed to the global dk accumulator exactly once per block instead
    // of once per head-warp (4x fewer global dk atomics).  dq keeps the
    // full D per warp (see pass 2).
    constexpr bool dsplit = (nd % 4) == 0;
    constexpr int ndq     = dsplit ? nd / 4 : nd;  // d-tiles per warp for dk
    (void)D_rt;
    const int row = blockIdx.x;
    if (isfinite(attn_l1norm[row]) && attn_l1norm[row] > 1e-10f) return;
    const int lane   = threadIdx.x & 63;
    const int warp   = threadIdx.x >> 6;
    const int g      = lane >> 4;
    const int r      = lane & 15;
    const int Hq_pad = (Hq + 15) & ~15;
    const int r0     = warp * 16;
    const int h      = r0 + r;
    const int d0     = dsplit ? (warp * ndq) * 16 : 0;  // this warp's d-column base

    int kv_offset = 0;
    int col_limit = 0;
    if (cu_q != nullptr) {
        ThdRowMeta m = thd_row_meta(cu_q, cu_kv, n_seqs, row, dk_rows_total);
        kv_offset    = m.kv_offset;
        col_limit    = thd_col_limit(m, ratio, q_causal_offsets);
    } else {
        const int sq = row % Sq;
        kv_offset    = (row / Sq) * Sk;
        col_limit    = (sq + 1) / ratio;
        if (col_limit > Sk) col_limit = Sk;
    }
    if (col_limit > score_stride) col_limit = score_stride;
    const int n_valid = col_limit;
    const bf16* krow  = k + (size_t)kv_offset * D;

    extern __shared__ char smem_raw[];
    hbf16* k_lds = (hbf16*)smem_raw;                            // [IDXB_CHUNK][KLD]
    float* score = (float*)(k_lds + (size_t)IDXB_CHUNK * KLD);  // [Sk]
    float* red   = score + Sk;                                  // [IDXB_THREADS]
    hbf16* a_lds = (hbf16*)(red + IDXB_THREADS);                // [Hq_pad][ALDS]

    const float w_h = (h < Hq) ? bf16_to_float(w[(size_t)row * Hq + h]) : 0.0f;

    // Cache Q fragments for the per-head scores and this warp's dK slice.
    constexpr int hgmax = dsplit ? 4 : 1;  // head-groups contracted by dk
    DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> af[nd];
    // bf[hg][dc]: Q fragments (head-group hg, this warp's d-tile dc) for the
    // dk MMAs.  With the d-split every warp must contract ALL head groups
    // (dk[k, d] sums over every h), so all valid groups are hoisted.
    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bf[hgmax][ndq];
    // Q fragments are loaded per-lane with rows past Hq clamped to row Hq-1
    // and zeroed: a plain du_load_matrix_sync of a 16-row fragment would
    // read past the head block (out-of-bounds at the last query row) and
    // feed stale/NaN data into the dk contraction (0 * stale = NaN).
    const int h_lane = r0 + (lane & 15);
    const int qrow_h = (h_lane < Hq) ? h_lane : (Hq - 1);
    if (r0 < Hq) {
        const hbf16* qrow = (const hbf16*)(q + (size_t)row * Hq * D + (size_t)qrow_h * D);
#    pragma unroll
        for (int kk = 0; kk < nd; kk++) {
            *(uint64_t*)&af[kk].x[0] = *(const uint64_t*)(qrow + kk * 16 + (g << 2));
            if (h_lane >= Hq) af[kk].x[0] = af[kk].x[1] = af[kk].x[2] = af[kk].x[3] = hbf16(0);
        }
    }
// bf[] is consumed by the dk MMAs of EVERY warp (each warp owns a d
// slice), including phantom-head warps, so hoist it unconditionally
// (rows past Hq clamped + zeroed, same as af[]).
#    pragma unroll
    for (int hg = 0; hg < hgmax; hg++) {
        const int hrow = dsplit ? hg * 16 : r0;
        if (hrow < Hq) {
            const int fr    = hrow + (g << 2);
            const hbf16* qh = (const hbf16*)(q + ((size_t)row * Hq + (size_t)hrow) * D + (size_t)(d0 + r));
#    pragma unroll
            for (int kk = 0; kk < ndq; kk++) {
                const hbf16* qc = qh + kk * 16;
                bf[hg][kk].x[0] = (fr < Hq) ? qc[(size_t)(g << 2) * D] : hbf16(0);
                bf[hg][kk].x[1] = (fr + 1 < Hq) ? qc[((size_t)(g << 2) + 1) * D] : hbf16(0);
                bf[hg][kk].x[2] = (fr + 2 < Hq) ? qc[((size_t)(g << 2) + 2) * D] : hbf16(0);
                bf[hg][kk].x[3] = (fr + 3 < Hq) ? qc[((size_t)(g << 2) + 3) * D] : hbf16(0);
            }
        }
    }

    for (int i = threadIdx.x; i < Sk; i += IDXB_THREADS) score[i] = 0.0f;
    // Zero k_lds once at kernel start: partial tail chunks leave rows beyond
    // ck untouched, and both MMA passes read all 16 rows of a B fragment.
    // Stale LDS (e.g. an INF/NaN bit pattern left by a previous kernel)
    // would turn 0 * stale into NaN and poison dq/dk accumulators.
    for (int i = threadIdx.x; i < IDXB_CHUNK * KLD; i += IDXB_THREADS) k_lds[i] = hbf16(0);
    // Zero a_lds with ALL threads: phantom-head rows (h >= Hq, e.g. H=8 with
    // Hq_pad=16) are never written by the per-chunk A build, but the dk MMAs
    // contract over all 16 rows of the A fragment.  Stale phantom rows (an
    // INF/NaN bit pattern left by a previous kernel) would corrupt every dk
    // column; dq is row-independent and would stay clean, masking the bug.
    for (int i = threadIdx.x; i < Hq_pad * ALDS; i += IDXB_THREADS) a_lds[i] = hbf16(0);
    __syncthreads();

    dsa_dense_score_grad(attn_score + (size_t)row * score_stride, attn_l1norm[row],
                         index_score + (size_t)row * score_stride, index_lse[row], score, red, Sk, n_valid,
                         (loss_coeff / (float)rows) * grad_loss);

    // ---- Per-head ReLU derivative, dW, dQ and dK per K chunk.
    // dq keeps the full D per warp (each (h, d) element belongs to the warp
    // owning head h); only dk is d-split across warps (its partials over
    // disjoint d-slices are flushed once per (k, d) element).
    DUFragment<accumulator, 16, 16, 16, float> dqacc[nd];
    DUFragment<accumulator, 16, 16, 16, float> dkacc[ndq];
#    pragma unroll
    for (int dc = 0; dc < nd; dc++) du_fill_fragment(dqacc[dc], 0.0f);
#    pragma unroll
    for (int dc = 0; dc < ndq; dc++) du_fill_fragment(dkacc[dc], 0.0f);
    float my_dw = 0.0f;

    for (int chunk = 0; chunk < col_limit; chunk += IDXB_CHUNK) {
        int ck = col_limit - chunk;
        if (ck > IDXB_CHUNK) ck = IDXB_CHUNK;
        const int ntiles = (ck + 15) >> 4;

        dm_load_k_from(k_lds, krow, chunk, ck, D, KLD);
        __syncthreads();

        // Recompute the per-head scores for this chunk.
        constexpr int MAXT2 = IDXB_CHUNK / 16;
        DUFragment<accumulator, 16, 16, 16, float> acc[MAXT2];
#    pragma unroll
        for (int nt = 0; nt < MAXT2; nt++) {
            if (nt < ntiles) du_fill_fragment(acc[nt], 0.0f);
        }
        for (int kk = 0; kk < nd; kk++) {
#    pragma unroll
            for (int nt = 0; nt < MAXT2; nt++) {
                if (nt < ntiles) {
                    DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr;
                    du_load_matrix_sync(bfr, k_lds + (size_t)(nt * 16) * KLD + kk * 16, KLD);
                    if (r0 < Hq) du_mma_sync(acc[nt], af[kk], bfr, acc[nt]);
                }
            }
        }

        // Build the bf16 A tile (head-major LDS: a_lds[h * ALDS + k]) and
        // accumulate the dw partial.  acc.x[i] = C[r][g + 4*i]: (h, k),
        // i.e. lane group g owns K columns g, g+4, g+8, g+12 of the tile
        // (same mapping as dm_score_chunk_epilogue and the dq/dk K tiles).
        if (r0 < Hq) {
            if (h < Hq) {
#    pragma unroll
                for (int nt = 0; nt < MAXT2; nt++) {
                    if (nt < ntiles) {
#    pragma unroll
                        for (int i = 0; i < 4; i++) {
                            const int kk = nt * 16 + g + 4 * i;
                            if (kk < ck) {
                                const float s                = acc[nt].x[i] * sm_scale;
                                const float pre              = (s > 0.0f) ? s : 0.0f;
                                const float dsc              = score[chunk + kk];
                                a_lds[(size_t)h * ALDS + kk] = (pre > 0.0f) ? float_to_bf16(dsc * w_h) : (bf16)0;
                                // CuTe keeps the ReLU score and dW product in FP32.
                                my_dw += dsc * pre;
                            }
                        }
                    }
                }
            }
            // Zero the tail columns [ck, ALDS) so partial-tile MMACs in dq/dk
            // never multiply stale A from the previous chunk.
            for (int kk = ck + g; kk < ALDS; kk += 4) a_lds[(size_t)h * ALDS + kk] = hbf16(0);
        }
        // Block-wide sync: every warp's A rows (and the zeroed phantom rows)
        // are visible before dq/dk contract over a_lds.  Must be outside the
        // r0 < Hq guard -- a conditional __syncthreads__ is UB.
        __syncthreads();

        // dq: dqacc[h, d] += A'[h, k] @ K[k, d]  (A head-major, rows h) --
        // only the warp owning head h touches (h, :), so valid warps only.
        // dk: dkacc[k, d] += A[k, h] @ Q[h, d]  (A col-major, rows k).
        //      The d-split makes each warp the sole writer of its d slice,
        //      so ALL warps (even phantom-head ones) must run the dk MMAs
        //      and contract every valid head group of a_lds.
        const int nksub = (ck + 15) >> 4;
        for (int ksub = 0; ksub < nksub; ksub++) {
            const int kk16 = ksub * 16;
#    pragma unroll
            for (int dc = 0; dc < ndq; dc++) du_fill_fragment(dkacc[dc], 0.0f);
            if (r0 < Hq) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> afr;
                du_load_matrix_sync(afr, a_lds + (size_t)r0 * ALDS + kk16, ALDS);
                if constexpr (PACK_DQ) {
                    typedef unsigned int mat4 __attribute__((ext_vector_type(4)));
#    pragma unroll
                    for (int dc = 0; dc < nd; dc += 2) {
                        const unsigned addr = (unsigned)(size_t)(k_lds + (size_t)kk16 * KLD + dc * 16 +
                                                                 (lane / 4) * KLD + (lane % 4) * 8);
                        mat4 packed;
                        asm volatile("ds_read_m32x16_b16_alt %0, %1" : "=v"(packed) : "v"(addr) : "memory");
                        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                        DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                        *(uint64_t*)&b0.x[0] = ((uint64_t*)&packed)[0];
                        *(uint64_t*)&b1.x[0] = ((uint64_t*)&packed)[1];
                        du_mma_sync(dqacc[dc], afr, b0, dqacc[dc]);
                        du_mma_sync(dqacc[dc + 1], afr, b1, dqacc[dc + 1]);
                    }
                } else {
#    pragma unroll
                    for (int dc = 0; dc < nd; dc++) {
                        DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bfr;
                        du_load_matrix_sync(bfr, k_lds + (size_t)kk16 * KLD + dc * 16, KLD);
                        du_mma_sync(dqacc[dc], afr, bfr, dqacc[dc]);
                    }
                }
            }
#    pragma unroll
            for (int hg = 0; hg < hgmax; hg++) {
                const int hrow = dsplit ? hg * 16 : r0;
                if (hrow < Hq) {
                    DUFragment<matrix_a, 16, 16, 16, hbf16, col_major> afc;
                    du_load_matrix_sync(afc, a_lds + (size_t)hrow * ALDS + kk16, ALDS);
#    pragma unroll
                    for (int dc = 0; dc < ndq; dc++) { du_mma_sync(dkacc[dc], afc, bf[hg][dc], dkacc[dc]); }
                }
            }
            // Flush this k-subtile's 16 k rows to the f32 accumulator.
            // With the d-split each (k, d) element is owned by exactly
            // one warp, so this is one atomic per element per block.
            if (kk16 + r < ck) {
                const int krow = kv_offset + chunk + kk16 + r;
#    pragma unroll
                for (int dc = 0; dc < ndq; dc++) {
                    float* pk = dk_float + (size_t)krow * D + d0 + dc * 16;
                    idxb_atomic_add_sparse<true>(pk, g, sm_scale * dkacc[dc].x[0], sm_scale * dkacc[dc].x[1],
                                                 sm_scale * dkacc[dc].x[2], sm_scale * dkacc[dc].x[3]);
                }
            }
        }
        __syncthreads();
    }

    // ---- dq epilogue (all heads / d columns written exactly once). ----
    if (h < Hq) {
        bf16* dqbase = dq + ((size_t)row * Hq + h) * D;
        if constexpr (PACK_DQ) {
// A 32-bit word carries two BF16 payloads; no FP32 arithmetic is
// replaced by BF16, and sm_scale is still evaluated in FP32.
#    pragma unroll
            for (int dc = 0; dc < nd; dc += 2) {
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    *(uint32_t*)(dqbase + (dc / 2) * 32 + g * 2 + j * 8) =
                        hip_dsa_numeric::bf16_pair(sm_scale * dqacc[dc].x[j], sm_scale * dqacc[dc + 1].x[j]);
                }
            }
        } else {
#    pragma unroll
            for (int dc = 0; dc < nd; dc++) {
                const int col    = dc * 16 + g;
                dqbase[col]      = float_to_bf16(sm_scale * dqacc[dc].x[0]);
                dqbase[col + 4]  = float_to_bf16(sm_scale * dqacc[dc].x[1]);
                dqbase[col + 8]  = float_to_bf16(sm_scale * dqacc[dc].x[2]);
                dqbase[col + 12] = float_to_bf16(sm_scale * dqacc[dc].x[3]);
            }
        }
    }

    // ---- dw epilogue ----
    my_dw += __shfl_xor(my_dw, 16);
    my_dw += __shfl_xor(my_dw, 32);
    if (g == 0 && h < Hq) dw[(size_t)row * Hq + h] = float_to_bf16(my_dw);

#endif
}

// Equal-prefix pair: every query retains its original K tail zeros, MMAC
// accumulation order and BF16 rounding. Only the final FP32 dK sum is local.
template<bool THD>
__device__ __forceinline__ bool dense_pair64_eligible(int row, int rows, int Sq, int Sk, int stride, int totalK,
                                                      int ratio, const int* cuq, const int* cuk, const int* offsets,
                                                      int nseq, const float* norm, int& kbase, int& valid) {
#if (defined(__gfx936__) || defined(__gfx938__))
    // Packed multi-segment layouts keep the established regular kernel.
    if constexpr (THD)
        if (nseq != 1) return false;
    if (ratio < 2 || (ratio & 1) || row < 0 || row + 1 >= rows) return false;
    if (!isfinite(norm[row]) || norm[row] <= 1e-10f || !isfinite(norm[row + 1]) || norm[row + 1] <= 1e-10f)
        return false;
    if constexpr (THD) {
        const ThdRowMeta m0 = thd_row_meta(cuq, cuk, nseq, row, totalK),
                         m1 = thd_row_meta(cuq, cuk, nseq, row + 1, totalK);
        if (!(m0.pos_in_seq & 1) || m0.batch_id != m1.batch_id || m1.pos_in_seq != m0.pos_in_seq + 1 ||
            m0.kv_offset != m1.kv_offset)
            return false;
        const int v0 = min(stride, thd_col_limit(m0, ratio, offsets)),
                  v1 = min(stride, thd_col_limit(m1, ratio, offsets));
        if (v0 != v1) return false;
        kbase = m0.kv_offset;
        valid = v0;
        return true;
    } else {
        if (!(row % Sq & 1) || row / Sq != (row + 1) / Sq) return false;
        const int v0 = min(Sk, int(((long long)(row % Sq) + 1) / ratio)),
                  v1 = min(Sk, int(((long long)((row + 1) % Sq) + 1) / ratio));
        if (v0 != v1) return false;
        kbase = row / Sq * Sk;
        valid = v0;
        return true;
    }
#else
    return false;
#endif
}
template<bool THD = false>
__device__ __forceinline__ bool dense_single64_eligible(int row, int rows, int Sq, int Sk, int stride, int totalK,
                                                        int ratio, const int* cuq, const int* cuk, const int* offsets,
                                                        int nseq, const float* norm, int& kbase, int& valid) {
#if (defined(__gfx936__) || defined(__gfx938__))
    if (ratio < 2 || (ratio & 1) || row < 0 || row >= rows) return false;
    if (!isfinite(norm[row]) || norm[row] <= 1e-10f) return false;
    if constexpr (THD) {
        if (nseq != 1) return false;
        const int qs = cuq[0], qe = cuq[1];
        if (qs < 0 || qe < qs || qe > rows || row < qs || row >= qe) return false;
        const int pos = row - qs;
        if (pos != 0 && !(row == qe - 1 && (pos & 1))) return false;
        const ThdRowMeta m = thd_row_meta(cuq, cuk, nseq, row, totalK);
        kbase              = m.kv_offset;
        valid              = min(stride, thd_col_limit(m, ratio, offsets));
        return true;
    } else {
        const int pos = row % Sq;
        if (pos != 0 && !(pos == Sq - 1 && (pos & 1))) return false;
        kbase = row / Sq * Sk;
        valid = min(Sk, int(((long long)pos + 1) / ratio));
        return true;
    }
#else
    return false;
#endif
}
template<int Count>
struct DensePairRealQueryCount {
    static constexpr int value = Count;
};
template<bool THD = false>
__global__ __launch_bounds__(256, 1) void dense_bwd_pair64_equal_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ w, const bf16* __restrict__ k, float* __restrict__ target,
    const float* __restrict__ predict, const int* __restrict__ idxs, bf16* __restrict__ dq, bf16* __restrict__ dw,
    float* __restrict__ dk_float, bf16* __restrict__ dk, int* done_count, int B, int Sq, int Sk, int Hq, int topk,
    float sm_scale, float loss_coeff, float grad_loss, bool topk_indices_global, const int* cuq, const int* cuk,
    const int* offsets, int nseq, int rows, int totalK, int ratio, const int* flags, int kparts, const float* norm) {
#if (defined(__gfx936__) || defined(__gfx938__))
    constexpr int D = 128, KLD = 136, ND = 8;
    // Slot0 handles the first singleton. Remaining odd pair starts include
    // an even-length final singleton. Never read/write a fictitious second row.
    const int slot = blockIdx.x, lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    int row;
    if constexpr (THD) {
        if (nseq != 1) return;
        const long long mapped = (long long)cuq[0] + (slot == 0 ? 0 : 2LL * slot - 1);
        row                    = (mapped >= 0 && mapped < rows) ? (int)mapped : -1;
    } else {
        const int np = Sq / 2 + 1;
        if (np <= 0) return;
        const int local = slot % np;
        row             = (slot / np) * Sq + (local == 0 ? 0 : 2 * local - 1);
    }

    const int r = lane & 15, g = lane >> 4;
    int kbase = 0, valid = 0;
    const bool singleton = dense_single64_eligible<THD>(row, rows, Sq, Sk, topk, totalK, ratio, cuq, cuk, offsets, nseq,
                                                        norm, kbase, valid);
    if (!singleton && !dense_pair64_eligible<THD>(row, rows, Sq, Sk, topk, totalK, ratio, cuq, cuk, offsets, nseq, norm,
                                                  kbase, valid))
        return;
    auto compute = [&](auto mode) __attribute__((always_inline)) {
        constexpr int count = decltype(mode)::value;
        extern __shared__ char raw[];
        hbf16* klds            = (hbf16*)raw;
        float* red             = (float*)(klds + 16 * KLD);
        hbf16* mask            = (hbf16*)(red + 128);
        const int chunks       = (valid + 15) / 16;
        const float* signal[2] = {target + (size_t)row * topk, target + (size_t)(row + (count == 2)) * topk};
        uint64_t qa[2][ND];
        float wh[2];
#    pragma unroll
        for (int hs = 0; hs < 2; hs++) {
            const int h = warp * 16 + r;
            wh[hs]      = hs < count ? bf16_to_float(w[(size_t)(row + hs) * 64 + h]) : 0.f;
#    pragma unroll
            for (int dc = 0; dc < ND; dc++)
                qa[hs][dc] =
                    hs < count ? *(const uint64_t*)(q + ((size_t)(row + hs) * 64 + h) * D + dc * 16 + g * 4) : 0;
        }
        uint32_t qdk[2][2][4][2];
#    pragma unroll
        for (int qq = 0; qq < 2; qq++) {
// Same H64 BF16 Q coordinates, no conversion or contraction change.
// Keep a group of independent raw halfword reads live before packing;
// this removes the old one-word opaque dependency at each global read.
#    pragma unroll
            for (int dl = 0; dl < 2; dl++) {
                const int dc = warp + dl * 4;
#    pragma unroll
                for (int hb = 0; hb < 4; hb += 2) {
                    uint32_t lo[4], hi[4];
#    pragma unroll
                    for (int z = 0; z < 4; z++) {
                        const int hc = hb + z / 2, e = z % 2, h = hc * 16 + g * 4;
                        const bf16* qp = q + ((size_t)(row + (qq < count ? qq : 0)) * 64 + h) * D + dc * 16 + r;
                        lo[z]          = qq < count ? (uint32_t)qp[(2 * e) * D] : 0;
                        hi[z]          = qq < count ? (uint32_t)qp[(2 * e + 1) * D] : 0;
                    }
                    asm volatile(""
                                 : "+v"(lo[0]), "+v"(hi[0]), "+v"(lo[1]), "+v"(hi[1]), "+v"(lo[2]), "+v"(hi[2]),
                                   "+v"(lo[3]), "+v"(hi[3]));
#    pragma unroll
                    for (int z = 0; z < 4; z++) qdk[qq][dl][hb + z / 2][z % 2] = lo[z] | (hi[z] << 16);
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
            for (int j = 0; j < 1; j++) {
                const int u = threadIdx.x + j * 256, ix = __shfl(shared_idx, warp * 4 + g + j * 16, 64);
                regs[j] = (ix >= 0 && ix < totalK) ? *(const uint4*)(k + (size_t)ix * D + (u % 16) * 8)
                                                   : make_uint4(0, 0, 0, 0);
            }
        };
        auto commit = [&](const uint4* regs) {
#    pragma unroll
            for (int j = 0; j < 1; j++) {
                const int u                                     = threadIdx.x + j * 256;
                *(uint4*)(klds + (u / 16) * KLD + (u % 16) * 8) = regs[j];
            }
        };
        int current_idx = r < valid ? kbase + r : -1, future_idx = 16 + r < valid ? kbase + 16 + r : -1;
        float carried_score[2] = {r < valid ? signal[0][r] : 0.f, count == 2 && r < valid ? signal[1][r] : 0.f};
        uint4 initial[1];
        prefetch(current_idx, initial);
        commit(initial);
        __syncthreads();
        for (int chunk = 0; chunk < chunks; chunk++) {
            const int next_idx = future_idx, pos = (chunk + 2) * 16 + r;
            const int after_idx = pos < valid ? kbase + pos : -1;
            uint4 next_k[1];
            if (chunk + 1 < chunks) prefetch(next_idx, next_k);
            const float score[2]      = {carried_score[0], carried_score[1]};
            const float next_score[2] = {
                (chunk + 1) * 16 + r < valid ? signal[0][(chunk + 1) * 16 + r] : 0.f,
                count == 2 && (chunk + 1) * 16 + r < valid ? signal[1][(chunk + 1) * 16 + r] : 0.f};
            uint16_t grad[2][4];
            // K is read-only between commit barriers. Share each raw col-major
            // fragment across both head groups; each dot[hs] keeps ascending DC order.
            DUFragment<accumulator, 16, 16, 16, float> dot[2];
#    pragma unroll
            for (int hs = 0; hs < 2; hs++) du_fill_fragment(dot[hs], 0.f);
#    pragma unroll
            for (int dc = 0; dc < ND; dc++) {
                DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> kb;
                du_load_matrix_sync(kb, klds + dc * 16, KLD);
#    pragma unroll
                for (int hs = 0; hs < 2; hs++) {
                    DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> a;
                    *(uint64_t*)a.x = qa[hs][dc];
                    du_mma_sync(dot[hs], a, kb, dot[hs]);
                }
            }
#    pragma unroll
            for (int hs = 0; hs < 2; hs++) {
                hbf16* mt = mask + (hs * 4 + warp) * 320;
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int col  = g + 4 * j;
                    const float ds = __shfl(score[hs], col, 64), v = dot[hs].x[j] * sm_scale;
                    hbf16 rounded     = idxb_rne(v > 0.f ? ds * wh[hs] : 0.f);
                    grad[hs][j]       = *(uint16_t*)&rounded;
                    mt[col * 20 + r]  = rounded;
                    dwacc[hs][j]     += ds * (v > 0.f ? v : 0.f);
                }
            }
            __syncthreads();
            DUFragment<accumulator, 16, 16, 16, float> dkacc[2][2];
#    pragma unroll
            for (int qq = 0; qq < 2; qq++) {
#    pragma unroll
                for (int dl = 0; dl < 2; dl++) du_fill_fragment(dkacc[qq][dl], 0.f);
#    pragma unroll
                for (int hc = 0; hc < 4; hc++) {
                    DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> a;
                    *(uint64_t*)a.x = *(const uint64_t*)(mask + (qq * 4 + hc) * 320 + r * 20 + g * 4);
#    pragma unroll
                    for (int dl = 0; dl < 2; dl++) {
                        DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> qb;
                        ((uint32_t*)qb.x)[0] = qdk[qq][dl][hc][0];
                        ((uint32_t*)qb.x)[1] = qdk[qq][dl][hc][1];
                        du_mma_sync(dkacc[qq][dl], a, qb, dkacc[qq][dl]);
                    }
                }
            }
            idxb_floatx4 av[2];
#    pragma unroll
            for (int dl = 0; dl < 2; dl++) {
                idxb_floatx4 v0 = {sm_scale * dkacc[0][dl].x[0], sm_scale * dkacc[0][dl].x[1],
                                   sm_scale * dkacc[0][dl].x[2], sm_scale * dkacc[0][dl].x[3]};
                idxb_floatx4 v1 = {sm_scale * dkacc[1][dl].x[0], sm_scale * dkacc[1][dl].x[1],
                                   sm_scale * dkacc[1][dl].x[2], sm_scale * dkacc[1][dl].x[3]};
                asm volatile("" : "+v"(v0), "+v"(v1));
                av[dl] = count == 1 ? v0 : v0 + v1;
            }
            uint64_t address =
                (uint64_t)dk_float + (((uint64_t)(uint32_t)current_idx) * D + warp * 16 + 4 * g) * sizeof(float);
            const uint64_t live = __ballot(current_idx >= 0 && current_idx < totalK);
            uint64_t saved;
            // Source-backed no-return x4 atomic, probed on gfx936; no global-load
            // consumer follows until the explicit vmcnt(0) below.
            asm volatile("s_and_saveexec_b64 %[saved], %[live]\n\t"
               "global_atomic_add_f32_x4 %[addr], %[v0], off\n\t"
               "global_atomic_add_f32_x4 %[addr], %[v1], off offset:256\n\t"
               "s_mov_b64 exec, %[saved]"
      :[saved] "=&s"(saved)
      :[v0] "v"(av[0]),[v1] "v"(av[1]),[addr] "v"(address),[live] "s"(live):"exec","memory");
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> adq[2];
#    pragma unroll
            for (int hs = 0; hs < 2; hs++) {
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
                ((unsigned*)adq[hs].x)[0] = lo;
                ((unsigned*)adq[hs].x)[1] = hi;
            }
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
                // Existing source-backed vector-output wait retains every physical
                // return until both matrix reads complete. No guessed nonzero wait.
                asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(packed[0]), "+v"(packed[1])::"memory");
#    pragma unroll
                for (int hs = 0; hs < 2; hs++) {
#    pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const int z = zb + j;
                        DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                        ((uint32_t*)b0.x)[0] = packed[j][0];
                        ((uint32_t*)b0.x)[1] = packed[j][1];
                        ((uint32_t*)b1.x)[0] = packed[j][2];
                        ((uint32_t*)b1.x)[1] = packed[j][3];
                        du_mma_sync(dqacc[hs][2 * z], adq[hs], b0, dqacc[hs][2 * z]);
                        du_mma_sync(dqacc[hs][2 * z + 1], adq[hs], b1, dqacc[hs][2 * z + 1]);
                    }
                }
            }

            asm volatile("s_waitcnt vmcnt(0)" : "+v"(av[0]), "+v"(av[1]), "+v"(address)::"memory");
            __syncthreads();
            if (chunk + 1 < chunks) commit(next_k);
            __syncthreads();
            current_idx      = next_idx;
            future_idx       = after_idx;
            carried_score[0] = next_score[0];
            carried_score[1] = next_score[1];
        }
#    pragma unroll
        for (int hs = 0; hs < 2; hs++)
            if (hs < count) {
                const int h = warp * 16 + r;
                bf16* dest  = dq + ((size_t)(row + hs) * 64 + h) * D;
#    pragma unroll
                for (int dc = 0; dc < ND; dc += 2) {
                    uint32_t v[4], x[4], y[4];
#    pragma unroll
                    for (int z = 0; z < 4; z++)
                        v[z] = hip_dsa_numeric::bf16_pair(sm_scale * dqacc[hs][dc].x[z],
                                                          sm_scale * dqacc[hs][dc + 1].x[z]);
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
                if (g == 0) dw[(size_t)(row + hs) * 64 + h] = float_to_bf16(v);
            }
    };
    if (singleton)
        compute(DensePairRealQueryCount<1>{});
    else
        compute(DensePairRealQueryCount<2>{});

#endif
}

template<bool THD = false>
__global__ __launch_bounds__(128, 1) void dense_bwd_two_wave_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ w, const bf16* __restrict__ k, float* __restrict__ target,
    const float* __restrict__ predict, const int* __restrict__ idxs, bf16* __restrict__ dq, bf16* __restrict__ dw,
    float* __restrict__ dk_float, bf16* __restrict__ dk, int* done_count, int B, int Sq, int Sk, int Hq, int topk,
    float sm_scale, float loss_coeff, float grad_loss, bool topk_indices_global, const int* cuq, const int* cuk,
    const int* offsets, int nseq, int rows, int totalK, int ratio, const int* flags, int kparts, const float* norm) {
#if (defined(__gfx936__) || defined(__gfx938__))
    constexpr int D = 128, KLD = 136, ND = 8;
    const int row = blockIdx.x, lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    const int r = lane & 15, g = lane >> 4;
    int pairbase = 0, pairvalid = 0;
    if (dense_single64_eligible<THD>(row, rows, Sq, Sk, topk, totalK, ratio, cuq, cuk, offsets, nseq, norm, pairbase,
                                     pairvalid) ||
        dense_pair64_eligible<THD>(row, rows, Sq, Sk, topk, totalK, ratio, cuq, cuk, offsets, nseq, norm, pairbase,
                                   pairvalid) ||
        dense_pair64_eligible<THD>(row - 1, rows, Sq, Sk, topk, totalK, ratio, cuq, cuk, offsets, nseq, norm, pairbase,
                                   pairvalid))
        return;
    if (!isfinite(norm[row]) || norm[row] <= 1e-10f) return;
    bool safe = false;
    int pair  = row & ~1;
    if (flags && pair + 1 < rows && pair / Sq == (pair + 1) / Sq && flags[16 + pair] == 0 && flags[17 + pair] == 0) {
        safe = true;
        for (int part = 0; part < kparts; part++) safe &= flags[part] == 0;
    }
    if (safe) return;
    extern __shared__ char raw[];
    hbf16* klds = (hbf16*)raw;
    float* red  = (float*)(klds + 16 * KLD);
    hbf16* mask = (hbf16*)(red + 128);
    int kbase = 0, valid = 0;
    if constexpr (THD) {
        ThdRowMeta meta = thd_row_meta(cuq, cuk, nseq, row, totalK);
        kbase           = meta.kv_offset;
        valid           = min(topk, thd_col_limit(meta, ratio, offsets));
    } else {
        kbase = (row / Sq) * Sk;
        valid = min(Sk, int(((long long)(row % Sq) + 1) / ratio));
    }
    const int chunks    = (valid + 15) / 16;
    const float* signal = target + (size_t)row * topk;
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
// Same H64 BF16 Q coordinates, no conversion or contraction change.
// Keep a group of independent raw halfword reads live before packing;
// this removes the old one-word opaque dependency at each global read.
#    pragma unroll
    for (int dl = 0; dl < 4; dl++) {
        const int dc = warp + dl * 2;
#    pragma unroll
        for (int hb = 0; hb < 4; hb += 2) {
            uint32_t lo[4], hi[4];
#    pragma unroll
            for (int z = 0; z < 4; z++) {
                const int hc = hb + z / 2, e = z % 2, h = hc * 16 + g * 4;
                const bf16* qp = q + ((size_t)row * 64 + h) * D + dc * 16 + r;
                lo[z]          = (uint32_t)qp[(2 * e) * D];
                hi[z]          = (uint32_t)qp[(2 * e + 1) * D];
            }
            asm volatile(""
                         : "+v"(lo[0]), "+v"(hi[0]), "+v"(lo[1]), "+v"(hi[1]), "+v"(lo[2]), "+v"(hi[2]), "+v"(lo[3]),
                           "+v"(hi[3]));
#    pragma unroll
            for (int z = 0; z < 4; z++) qdk[dl][hb + z / 2][z % 2] = lo[z] | (hi[z] << 16);
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
            regs[j] =
                (ix >= 0 && ix < totalK) ? *(const uint4*)(k + (size_t)ix * D + (u % 16) * 8) : make_uint4(0, 0, 0, 0);
        }
    };
    auto commit = [&](const uint4* regs) {
#    pragma unroll
        for (int j = 0; j < 2; j++) {
            const int u                                     = threadIdx.x + j * 128;
            *(uint4*)(klds + (u / 16) * KLD + (u % 16) * 8) = regs[j];
        }
    };
    int current_idx = r < valid ? kbase + r : -1, future_idx = 16 + r < valid ? kbase + 16 + r : -1;
    float carried_score = r < valid ? signal[r] : 0.f;
    uint4 initial[2];
    prefetch(current_idx, initial);
    commit(initial);
    __syncthreads();
    for (int chunk = 0; chunk < chunks; chunk++) {
        const int next_idx = future_idx, pos = (chunk + 2) * 16 + r;
        const int after_idx = pos < valid ? kbase + pos : -1;
        uint4 next_k[2];
        if (chunk + 1 < chunks) prefetch(next_idx, next_k);
        const float score      = carried_score;
        const float next_score = (chunk + 1) * 16 + r < valid ? signal[(chunk + 1) * 16 + r] : 0.f;
        uint16_t grad[2][4];
        // K is read-only between commit barriers. Share each raw col-major
        // fragment across both head groups; each dot[hs] keeps ascending DC order.
        DUFragment<accumulator, 16, 16, 16, float> dot[2];
#    pragma unroll
        for (int hs = 0; hs < 2; hs++) du_fill_fragment(dot[hs], 0.f);
#    pragma unroll
        for (int dc = 0; dc < ND; dc++) {
            DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> kb;
            du_load_matrix_sync(kb, klds + dc * 16, KLD);
#    pragma unroll
            for (int hs = 0; hs < 2; hs++) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> a;
                *(uint64_t*)a.x = qa[hs][dc];
                du_mma_sync(dot[hs], a, kb, dot[hs]);
            }
        }
#    pragma unroll
        for (int hs = 0; hs < 2; hs++) {
            hbf16* mt = mask + (hs * 2 + warp) * 320;
#    pragma unroll
            for (int j = 0; j < 4; j++) {
                const int col  = g + 4 * j;
                const float ds = __shfl(score, col, 64), v = dot[hs].x[j] * sm_scale;
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
            (uint64_t)dk_float + (((uint64_t)(uint32_t)current_idx) * D + warp * 16 + 4 * g) * sizeof(float);
        const uint64_t live = __ballot(current_idx >= 0 && current_idx < totalK);
        uint64_t saved;
        // Source-backed no-return x4 atomic, probed on gfx936; no global-load
        // consumer follows until the explicit vmcnt(0) below.
        asm volatile("s_and_saveexec_b64 %[saved], %[live]\n\t"
               "global_atomic_add_f32_x4 %[addr], %[v0], off\n\t"
               "global_atomic_add_f32_x4 %[addr], %[v1], off offset:128\n\t"
               "global_atomic_add_f32_x4 %[addr], %[v2], off offset:256\n\t"
               "global_atomic_add_f32_x4 %[addr], %[v3], off offset:384\n\t"
               "s_mov_b64 exec, %[saved]"
      :[saved] "=&s"(saved)
      :[v0] "v"(av[0]),[v1] "v"(av[1]),[v2] "v"(av[2]),[v3] "v"(av[3]),[addr] "v"(address),[live] "s"(live):"exec","memory");
        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> adq[2];
#    pragma unroll
        for (int hs = 0; hs < 2; hs++) {
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
            ((unsigned*)adq[hs].x)[0] = lo;
            ((unsigned*)adq[hs].x)[1] = hi;
        }
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
            // Existing source-backed vector-output wait retains every physical
            // return until both matrix reads complete. No guessed nonzero wait.
            asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(packed[0]), "+v"(packed[1])::"memory");
#    pragma unroll
            for (int hs = 0; hs < 2; hs++) {
#    pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int z = zb + j;
                    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                    ((uint32_t*)b0.x)[0] = packed[j][0];
                    ((uint32_t*)b0.x)[1] = packed[j][1];
                    ((uint32_t*)b1.x)[0] = packed[j][2];
                    ((uint32_t*)b1.x)[1] = packed[j][3];
                    du_mma_sync(dqacc[hs][2 * z], adq[hs], b0, dqacc[hs][2 * z]);
                    du_mma_sync(dqacc[hs][2 * z + 1], adq[hs], b1, dqacc[hs][2 * z + 1]);
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
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
#endif
}

template<bool THD = false>
__global__ __launch_bounds__(128, 1) void dense_bwd_pair32_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ w, const bf16* __restrict__ k, float* __restrict__ target,
    const float* __restrict__ predict, const int* __restrict__ idxs, bf16* __restrict__ dq, bf16* __restrict__ dw,
    float* __restrict__ dk_float, bf16* __restrict__ dk, int* done_count, int B, int Sq, int Sk, int Hq, int topk,
    float sm_scale, float loss_coeff, float grad_loss, bool topk_indices_global, const int* cuq, const int* cuk,
    const int* offsets, int nseq, int rows, int totalK, int ratio, const int* flags, int kparts) {
#if (defined(__gfx936__) || defined(__gfx938__))
    constexpr int D = 128, KLD = 136, ND = 8;
    const int row = blockIdx.x * 2, lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    const int r = lane & 15, g = lane >> 4;
    bool safe = false;
    if (flags && row + 1 < rows && row / Sq == (row + 1) / Sq && flags[16 + row] == 0 && flags[17 + row] == 0) {
        safe = true;
        for (int part = 0; part < kparts; part++) safe &= flags[part] == 0;
    }
    if (!safe) return;
    extern __shared__ char raw[];
    hbf16* klds            = (hbf16*)raw;
    float* red             = (float*)(klds + 16 * KLD);
    hbf16* mask            = (hbf16*)(red + 128);
    const int kbase        = (row / Sq) * Sk;
    const int valid[2]     = {min(Sk, int(((long long)(row % Sq) + 1) / ratio)),
                              min(Sk, int(((long long)((row + 1) % Sq) + 1) / ratio))};
    const int chunks       = (valid[1] + 15) / 16;
    const float* signal[2] = {target + (size_t)row * topk, target + (size_t)(row + 1) * topk};
    uint64_t qa[2][ND];
    float wh[2];
#    pragma unroll
    for (int hs = 0; hs < 2; hs++) {
        const int h = warp * 16 + r;
        wh[hs]      = bf16_to_float(w[(size_t)(row + hs) * 64 + h]);
#    pragma unroll
        for (int dc = 0; dc < ND; dc++)
            qa[hs][dc] = *(const uint64_t*)(q + ((size_t)(row + hs) * 64 + h) * D + dc * 16 + g * 4);
    }
    uint32_t qdk[2][4][2][2];
#    pragma unroll
    for (int qq = 0; qq < 2; qq++) {
#    pragma unroll
        for (int dl = 0; dl < 4; dl++) {
            const int dc = warp + dl * 2;
#    pragma unroll
            for (int hc = 0; hc < 2; hc++) {
                const int h    = hc * 16 + g * 4;
                const bf16* qp = q + ((size_t)(row + qq) * 64 + h) * D + dc * 16 + r;
#    pragma unroll
                for (int e = 0; e < 2; e++) {
                    uint32_t v = (uint32_t)qp[(2 * e) * D] | ((uint32_t)qp[(2 * e + 1) * D] << 16);
                    asm volatile("" : "+v"(v));
                    qdk[qq][dl][hc][e] = v;
                }
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
            regs[j] =
                (ix >= 0 && ix < totalK) ? *(const uint4*)(k + (size_t)ix * D + (u % 16) * 8) : make_uint4(0, 0, 0, 0);
        }
    };
    auto commit = [&](const uint4* regs) {
#    pragma unroll
        for (int j = 0; j < 2; j++) {
            const int u                                     = threadIdx.x + j * 128;
            *(uint4*)(klds + (u / 16) * KLD + (u % 16) * 8) = regs[j];
        }
    };
    int current_idx = r < valid[1] ? kbase + r : -1, future_idx = 16 + r < valid[1] ? kbase + 16 + r : -1;
    float carried_score[2] = {r < valid[0] ? signal[0][r] : 0.f, r < valid[1] ? signal[1][r] : 0.f};
    uint4 initial[2];
    prefetch(current_idx, initial);
    commit(initial);
    __syncthreads();
    for (int chunk = 0; chunk < chunks; chunk++) {
        const int next_idx = future_idx, pos = (chunk + 2) * 16 + r;
        const int after_idx = pos < valid[1] ? kbase + pos : -1;
        uint4 next_k[2];
        if (chunk + 1 < chunks) prefetch(next_idx, next_k);
        const float next_score[2] = {(chunk + 1) * 16 + r < valid[0] ? signal[0][(chunk + 1) * 16 + r] : 0.f,
                                     (chunk + 1) * 16 + r < valid[1] ? signal[1][(chunk + 1) * 16 + r] : 0.f};
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
                const float ds = __shfl(carried_score[hs], col, 64), v = dot.x[j] * sm_scale;
                const bool live   = chunk * 16 + col < valid[hs];
                hbf16 rounded     = idxb_rne(live && v > 0.f ? ds * wh[hs] : 0.f);
                grad[hs][j]       = *(uint16_t*)&rounded;
                mt[col * 20 + r]  = rounded;
                dwacc[hs][j]     += live ? ds * (v > 0.f ? v : 0.f) : 0.f;
            }
        }
        __syncthreads();
        // The fast proof bounds all operands and fixes scale=1. Accumulate the
        // two queries as one64-head contraction; dS was rounded perquery already.
        DUFragment<accumulator, 16, 16, 16, float> dkacc[4];
#    pragma unroll
        for (int dl = 0; dl < 4; dl++) du_fill_fragment(dkacc[dl], 0.f);
#    pragma unroll
        for (int qq = 0; qq < 2; qq++) {
#    pragma unroll
            for (int hc = 0; hc < 2; hc++) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> a;
                *(uint64_t*)a.x = *(const uint64_t*)(mask + (qq * 2 + hc) * 320 + r * 20 + g * 4);
#    pragma unroll
                for (int dl = 0; dl < 4; dl++) {
                    DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> qb;
                    ((uint32_t*)qb.x)[0] = qdk[qq][dl][hc][0];
                    ((uint32_t*)qb.x)[1] = qdk[qq][dl][hc][1];
                    du_mma_sync(dkacc[dl], a, qb, dkacc[dl]);
                }
            }
        }
        idxb_floatx4 av[4];
#    pragma unroll
        for (int dl = 0; dl < 4; dl++) av[dl] = {dkacc[dl].x[0], dkacc[dl].x[1], dkacc[dl].x[2], dkacc[dl].x[3]};
        uint64_t address =
            (uint64_t)dk_float + (((uint64_t)(uint32_t)current_idx) * D + warp * 16 + 4 * g) * sizeof(float);
        const uint64_t live = __ballot(current_idx >= 0 && current_idx < totalK);
        uint64_t saved;
        // Source-backed no-return x4 atomic, probed on gfx936; no global-load
        // consumer follows until the explicit vmcnt(0) below.
        asm volatile("s_and_saveexec_b64 %[saved], %[live]\n\t"
               "global_atomic_add_f32_x4 %[addr], %[v0], off\n\t"
               "global_atomic_add_f32_x4 %[addr], %[v1], off offset:128\n\t"
               "global_atomic_add_f32_x4 %[addr], %[v2], off offset:256\n\t"
               "global_atomic_add_f32_x4 %[addr], %[v3], off offset:384\n\t"
               "s_mov_b64 exec, %[saved]"
      :[saved] "=&s"(saved)
      :[v0] "v"(av[0]),[v1] "v"(av[1]),[v2] "v"(av[2]),[v3] "v"(av[3]),[addr] "v"(address),[live] "s"(live):"exec","memory");
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
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        if (chunk + 1 < chunks) commit(next_k);
        __syncthreads();
        current_idx      = next_idx;
        future_idx       = after_idx;
        carried_score[0] = next_score[0];
        carried_score[1] = next_score[1];
    }
#    pragma unroll
    for (int hs = 0; hs < 2; hs++) {
        const int h = warp * 16 + r;
        bf16* dest  = dq + ((size_t)(row + hs) * 64 + h) * D;
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
        if (g == 0) dw[(size_t)(row + hs) * 64 + h] = float_to_bf16(v);
    }
    const bf16 z = float_to_bf16(sm_scale * 0.f);
#    pragma unroll
    for (int qq = 0; qq < 2; qq++) {
        for (int u = threadIdx.x; u < 32 * 128; u += 128) dq[(size_t)(row + qq) * 64 * 128 + 32 * 128 + u] = z;
        if (threadIdx.x < 32) dw[(size_t)(row + qq) * 64 + 32 + threadIdx.x] = float_to_bf16(0.f);
    }
#endif
}

template<bool THD = false>
static __global__ void dense_score_grad_prepass(const float* target, const float* norm, float* logits, const float* lse,
                                                const int* cuq, const int* cuk, const int* offsets, int nseq, int Sq,
                                                int Sk, int stride, int totalK, int rows, int ratio, float loss,
                                                float grad, const bf16* q, const bf16* w, const bf16* k, float scale,
                                                int* flags, int kparts) {
    __shared__ float red[4];
    int row = blockIdx.x, valid;
    if constexpr (THD) {
        ThdRowMeta meta = thd_row_meta(cuq, cuk, nseq, row, totalK);
        valid           = min(stride, thd_col_limit(meta, ratio, offsets));
    } else
        valid = min(Sk, int(((long long)(row % Sq) + 1) / ratio));
    float* signal = logits + (size_t)row * stride;
    if (isfinite(norm[row]) && norm[row] > 1e-10f)
        dsa_dense_score_grad(target + (size_t)row * stride, norm[row], signal, lse[row], signal, red, stride, valid,
                             (loss / (float)rows) * grad);
    if (flags) {
        bool bad           = scale != 1.f || !isfinite(norm[row]) || norm[row] <= 1e-10f;
        const uint64_t* qp = (const uint64_t*)(q + ((size_t)row * 64 + 32) * 128);
        for (int i = threadIdx.x; i < 32 * 128 / 4; i += 256) bad |= (qp[i] & 0x7fff7fff7fff7fffULL) != 0;
        if (threadIdx.x < 32) {
            bad |= (w[(size_t)row * 64 + 32 + threadIdx.x] & 0x7fff) != 0;
            bad |= (w[(size_t)row * 64 + threadIdx.x] & 0x7fff) > 0x4100;
        }
        const unsigned* lowerq = (const unsigned*)(q + (size_t)row * 64 * 128);
        for (int i = threadIdx.x; i < 32 * 128 / 2; i += 256) {
            unsigned x  = lowerq[i];
            bad        |= (x & 0x7fffu) > 0x4100u || ((x >> 16) & 0x7fffu) > 0x4100u;
        }
        for (int i = threadIdx.x; i < stride; i += 256) bad |= !isfinite(signal[i]) || fabsf(signal[i]) > .01f;
        __shared__ int proof[4];
        int wb = __ballot(bad) != 0;
        if ((threadIdx.x & 63) == 0) proof[threadIdx.x >> 6] = wb;
        __syncthreads();
        if (threadIdx.x == 0) flags[16 + row] = proof[0] | proof[1] | proof[2] | proof[3];
        if (blockIdx.x < kparts) {
            bool badk          = false;
            const unsigned* kp = (const unsigned*)k;
            for (size_t i  = (size_t)blockIdx.x * 256 + threadIdx.x; i < (size_t)totalK * 64;
                 i        += (size_t)kparts * 256) {
                unsigned x  = kp[i];
                badk       |= (x & 0x7fffu) > 0x4100u || ((x >> 16) & 0x7fffu) > 0x4100u;
            }
            __syncthreads();
            wb = __ballot(badk) != 0;
            if ((threadIdx.x & 63) == 0) proof[threadIdx.x >> 6] = wb;
            __syncthreads();
            if (threadIdx.x == 0) flags[blockIdx.x] = proof[0] | proof[1] | proof[2] | proof[3];
        }
    }
}
// File-local launch check: observe each stage before a later HIP call can clear its error.
#define INDEXER_BWD_DENSE_THD_CHECKED_LAUNCH(kernel, ...)                                                            \
    do {                                                                                                             \
        hipLaunchKernelGGL(kernel, __VA_ARGS__);                                                                     \
        const hipError_t launch_status = hipGetLastError();                                                          \
        if (launch_status != hipSuccess) {                                                                           \
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_indexer_bwd_dense_thd %s launch failed: %s", \
                                    #kernel, hipGetErrorString(launch_status));                                      \
        }                                                                                                            \
    } while (0)

template<bool HAS_PAIR, bool HAS_PROOF>
static bool launch_dense_two_wave(hipStream_t stream, const void* q, const void* w, const void* k, const float* target,
                                  const float* norm, float* score, const float* lse, void* dq, void* dw, void* dk,
                                  float* dkf, const int* cuq, const int* cuk, const int* offsets, int nseq, int rows,
                                  int Sq, int Sk, int totalK, int stride, int ratio, float scale, float loss,
                                  float grad) {
    const int shm       = 16 * 136 * 2 + 128 * 4 + 4 * 320 * 2;
    const int pair64shm = 16 * 136 * 2 + 128 * 4 + 8 * 320 * 2;
    const int lds32     = 16 * 136 * 2 + 128 * 4 + 4 * 320 * 2;
    // DK output is dead until the stream-ordered final cast. Its storage is
    // temporary proof flags, with no extra allocation and no initialization race.
    int* flags      = HAS_PROOF ? (int*)dk : nullptr;
    int kparts      = min(rows, 16);
    {
#define PRE(T)                                                                                                     \
    INDEXER_BWD_DENSE_THD_CHECKED_LAUNCH((dense_score_grad_prepass<T>), dim3(rows), dim3(256), 0, stream, target,  \
                                         norm, score, lse, cuq, cuk, offsets, nseq, Sq, Sk, stride, totalK, rows,  \
                                         ratio, loss, grad, (const bf16*)q, (const bf16*)w, (const bf16*)k, scale, \
                                         flags, kparts)
#define MM(T)                                                                                                          \
    INDEXER_BWD_DENSE_THD_CHECKED_LAUNCH((dense_bwd_two_wave_kernel<T>), dim3(rows), dim3(128), shm, stream,           \
                                         (const bf16*)q, (const bf16*)w, (const bf16*)k, score, (const float*)nullptr, \
                                         (const int*)nullptr, (bf16*)dq, (bf16*)dw, dkf, (bf16*)dk, (int*)nullptr, 1,  \
                                         Sq, Sk, 64, stride, scale, loss, grad, true, cuq, cuk, offsets, nseq, rows,   \
                                         totalK, ratio, flags, kparts, norm)
#define PAIR64(T)                                                                                                     \
    if constexpr (HAS_PAIR)                                                                                           \
    INDEXER_BWD_DENSE_THD_CHECKED_LAUNCH(                                                                             \
        (dense_bwd_pair64_equal_kernel<T>), dim3((T) ? rows / 2 + 1 : (rows / Sq) * (Sq / 2 + 1)), dim3(256),         \
        pair64shm, stream, (const bf16*)q, (const bf16*)w, (const bf16*)k, score, (const float*)nullptr,              \
        (const int*)nullptr, (bf16*)dq, (bf16*)dw, dkf, (bf16*)dk, (int*)nullptr, 1, Sq, Sk, 64, stride, scale, loss, \
        grad, true, cuq, cuk, offsets, nseq, rows, totalK, ratio, flags, kparts, norm)
#define FAST(T)                                                                                                        \
    INDEXER_BWD_DENSE_THD_CHECKED_LAUNCH((dense_bwd_pair32_kernel<T>), dim3((rows + 1) / 2), dim3(128), lds32, stream, \
                                         (const bf16*)q, (const bf16*)w, (const bf16*)k, score, (const float*)nullptr, \
                                         (const int*)nullptr, (bf16*)dq, (bf16*)dw, dkf, (bf16*)dk, (int*)nullptr, 1,  \
                                         Sq, Sk, 64, stride, scale, loss, grad, true, cuq, cuk, offsets, nseq, rows,   \
                                         totalK, ratio, flags, kparts)
        if constexpr (true) {
            PRE(true);
            PAIR64(true);
            MM(true);
        } else {
            PRE(false);
            if constexpr (HAS_PROOF) { FAST(false); }
            PAIR64(false);
            MM(false);
        }
#undef PRE
#undef MM
#undef PAIR64
#undef FAST
        const size_t fallback_shm = (size_t)IDXB_CHUNK * 136 * sizeof(bf16) +
                                    (size_t)(Sk + IDXB_THREADS) * sizeof(float) +
                                    (size_t)64 * (IDXB_CHUNK + 2) * sizeof(bf16);
        {
            const auto err = hipFuncSetAttribute((const void*)dense_bwd_norm_fallback_kernel<128>,
                                                 hipFuncAttributeMaxDynamicSharedMemorySize, (int)fallback_shm);
            if (err != hipSuccess)
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "dense backward LDS: %s", hipGetErrorString(err));
        }
        INDEXER_BWD_DENSE_THD_CHECKED_LAUNCH(
            (dense_bwd_norm_fallback_kernel<128>), dim3(rows), dim3(IDXB_THREADS), fallback_shm, stream, (const bf16*)q,
            (const bf16*)w, (const bf16*)k, target, norm, score, lse, (bf16*)dq, (bf16*)dw, dkf, (bf16*)dk,
            (int*)nullptr, cuq, cuk, offsets, rows, Sq, Sk, 64, 128, nseq, ratio, scale, loss, grad, stride, totalK);
    }
    size_t n4 = (size_t)totalK * 128 / 4;
    INDEXER_BWD_DENSE_THD_CHECKED_LAUNCH((indexer_dk_cast_kernel<true>), dim3((n4 + 255) / 256), dim3(256), 0, stream,
                                         dkf, (bf16*)dk, n4);

    return true;
}
#undef INDEXER_BWD_DENSE_THD_CHECKED_LAUNCH

namespace ftrain {
namespace {
template<int WIDTH>
static bool launch_indexer_bwd_dense_mmac_full(hipStream_t stream, const void* q, const void* w, const void* k,
                                               const float* attn_score, const float* attn_l1norm, float* index_score,
                                               const float* index_lse, void* dq, void* dw, void* dk, float* dk_float,
                                               int* done_count, const int* cu_q, const int* cu_kv,
                                               const int* q_causal_offsets, int rows, int Sq, int Sk, int Hq, int D,
                                               int n_seqs, int ratio, float sm_scale, float loss_coeff, float grad_loss,
                                               int score_stride, int dk_rows_total);

template<int WIDTH>
static bool launch_indexer_bwd_dense_mmac_full(hipStream_t stream, const void* q, const void* w, const void* k,
                                               const float* attn_score, const float* attn_l1norm, float* index_score,
                                               const float* index_lse, void* dq, void* dw, void* dk, float* dk_float,
                                               int* done_count, const int* cu_q, const int* cu_kv,
                                               const int* q_causal_offsets, int rows, int Sq, int Sk, int Hq, int D,
                                               int n_seqs, int ratio, float sm_scale, float loss_coeff, float grad_loss,
                                               int score_stride, int dk_rows_total) {
    // This implementation has BF16 pointer types and BF16 MMAC fragments.
    // Never reinterpret FP32/FP16 storage, even if called without the binding.

    if (score_stride <= 0) score_stride = Sk;
    const int Hq_pad = (Hq + 15) & ~15;
    const size_t shm = (size_t)IDXB_CHUNK * (D + 8) * sizeof(bf16) + (size_t)(Sk + IDXB_THREADS) * sizeof(float) +
                       (size_t)Hq_pad * (IDXB_CHUNK + 2) * sizeof(bf16);
    {
        const hipError_t status = hipFuncSetAttribute((const void*)dense_indexer_bwd_fused_kernel<WIDTH>,
                                                      hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipFuncSetAttribute failed: %s",
                                    hipGetErrorString(status));
        }
    }
    hipLaunchKernelGGL((dense_indexer_bwd_fused_kernel<WIDTH>), rows, IDXB_THREADS, shm, stream, (const bf16*)q,
                       (const bf16*)w, (const bf16*)k, attn_score, attn_l1norm, index_score, index_lse, (bf16*)dq,
                       (bf16*)dw, dk_float, (bf16*)dk, done_count, cu_q, cu_kv, q_causal_offsets, rows, Sq, Sk, Hq, D,
                       n_seqs, ratio, sm_scale, loss_coeff, grad_loss, score_stride, dk_rows_total);
    // hipLaunchKernelGGL is a void launch macro; check the selected case here
    // before returning success. Execution stays asynchronous on this stream.
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_indexer_bwd_dense_thd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}

template<int WIDTH>
static bool launch_indexer_bwd_dense_thd(hipStream_t stream, const void* index_q, const void* weights,
                                         const void* index_k, const float* attn_score, const float* attn_l1norm,
                                         float* index_score, const float* index_lse, void* dq, void* dw, void* dk,
                                         float* dk_float, int* done_count, const int* cu_seqlens_q,
                                         const int* cu_seqlens_kv, const int* q_causal_offsets, int total_q,
                                         int total_k, int max_sk, int H, int D, int n_seqs, int ratio, float sm_scale,
                                         float loss_coeff, float grad_loss) {
    // Fused single-kernel path (MMAC tensor core) for the whole packed batch:
    // score + softmax + A + dq + dk in ONE launch.  No non-fused fallback
    // and no per-segment launch loop.
    if (!launch_indexer_bwd_dense_mmac_full<WIDTH>(
            stream, index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse, dq, dw, dk, dk_float,
            done_count, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, /*rows=*/total_q, /*Sq=*/1, /*Sk=*/max_sk, H, D,
            n_seqs, ratio, sm_scale, loss_coeff, grad_loss, /*score_stride=*/max_sk, /*dk_rows_total=*/total_k))
        return false;
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_indexer_bwd_dense_thd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}
}  // namespace

void bf16IndexerBwdDenseThdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    const hipError_t status = hip_dsa_numeric::zero_workspace(memory, bytes, stream);
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Workspace zero-fill failed: %s", hipGetErrorString(status));
    }
}

}  // namespace ftrain

namespace ftrain {

template<class Implementation, int WIDTH, bool HAS_PAIR, bool HAS_PROOF>
bool launchBf16IndexerBwdDenseThd(FTrainStream stream, const void* index_q, const void* weights, const void* index_k,
                                  const float* attn_score, const float* attn_l1norm, float* index_score,
                                  const float* index_lse, void* dq, void* dw, void* dk, float* dk_float,
                                  int* done_count, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                  const int* q_causal_offsets, int total_q, int total_k, int max_seqlen_kv, int heads,
                                  int head_dim, int num_sequences, int ratio, float sm_scale, float loss_coeff,
                                  float grad_loss) {
    if constexpr (Implementation::kTwoWave) {
        return launch_dense_two_wave<HAS_PAIR, HAS_PROOF>(
            stream, index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse, dq, dw, dk, dk_float,
            cu_seqlens_q, cu_seqlens_k, q_causal_offsets, num_sequences, total_q, 1, max_seqlen_kv, total_k,
            max_seqlen_kv, ratio, sm_scale, loss_coeff, grad_loss);
    }
    return launch_indexer_bwd_dense_thd<WIDTH>(stream, index_q, weights, index_k, attn_score, attn_l1norm, index_score,
                                               index_lse, dq, dw, dk, dk_float, done_count, cu_seqlens_q, cu_seqlens_k,
                                               q_causal_offsets, total_q, total_k, max_seqlen_kv, heads, head_dim,
                                               num_sequences, ratio, sm_scale, loss_coeff, grad_loss);
}

// Explicit fixed launch instances used by the C++ configuration code.
#define FTRAIN_INSTANTIATE_LAUNCH(Implementation, WIDTH, HAS_PAIR, HAS_PROOF)                                        \
    template bool launchBf16IndexerBwdDenseThd<Implementation, WIDTH, HAS_PAIR, HAS_PROOF>(                          \
        FTrainStream stream, const void* index_q, const void* weights, const void* index_k, const float* attn_score, \
        const float* attn_l1norm, float* index_score, const float* index_lse, void* dq, void* dw, void* dk,          \
        float* dk_float, int* done_count, const int* cu_seqlens_q, const int* cu_seqlens_k,                          \
        const int* q_causal_offsets, int total_q, int total_k, int max_seqlen_kv, int heads, int head_dim,           \
        int num_sequences, int ratio, float sm_scale, float loss_coeff, float grad_loss);
FTRAIN_INSTANTIATE_LAUNCH(Bf16IndexerBwdDenseThd, 112, false, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16IndexerBwdDenseThd, 128, false, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16IndexerBwdDenseThd, 16, false, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16IndexerBwdDenseThd, 32, false, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16IndexerBwdDenseThd, 48, false, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16IndexerBwdDenseThd, 64, false, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16IndexerBwdDenseThd, 80, false, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16IndexerBwdDenseThd, 96, false, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16IndexerBwdDenseThdTwoWave, 128, false, false);
FTRAIN_INSTANTIATE_LAUNCH(Bf16IndexerBwdDenseThdTwoWavePair, 128, true, false);
#undef FTRAIN_INSTANTIATE_LAUNCH

}  // namespace ftrain
