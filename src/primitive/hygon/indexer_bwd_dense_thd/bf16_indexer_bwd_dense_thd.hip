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
#include "flash_train/primitive/hygon/indexer_bwd_dense_thd/bf16_indexer_bwd_dense_thd.hpp"

static inline int dsa_shared_memory_limit() {
    int device = 0, limit = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&limit, hipDeviceAttributeMaxSharedMemoryPerBlock, device) != hipSuccess)
        return 0;
    return limit;
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

static inline bool dense_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx928") != nullptr || strstr(prop.gcnArchName, "gfx936") != nullptr ||
           strstr(prop.gcnArchName, "gfx938") != nullptr;
}

static size_t dense_mmac_shm(int Hq, int D, int Sk, bool bwd) {
    int Hq_pad  = (Hq + 15) & ~15;
    int dd      = bwd ? D : (D < DENSE_DCHUNK ? D : DENSE_DCHUNK);
    size_t sz   = (size_t)DENSE_CHUNK * (dd + 8) * 2;  // k_lds
    sz         += (size_t)Sk * 4;                      // score
    if (bwd) {
        // K, score-gradient, reduction and BF16 dS tiles; Q stays in registers.
        sz += DENSE_THREADS * 4;                       // red
        sz += (size_t)Hq_pad * (DENSE_CHUNK + 2) * 2;  // a_lds (ALDS = 34)
    } else {
        sz += (size_t)Hq_pad * (dd + 8) * 2;  // q_lds (forward kernels)
        sz += DENSE_THREADS * 4;              // red (denom)
    }
    return sz;
}

static inline bool dense_mmac_shm_fits(int Hq, int D, int Sk, bool bwd) {
    const int max_lds = dsa_shared_memory_limit();
    if (!bwd) return dense_mmac_shm(Hq, D, Sk, false) <= (size_t)max_lds;
    int Hq_pad = (Hq + 15) & ~15;
    return dense_mmac_shm(Hq, D, Sk, true) <= (size_t)max_lds;
}

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

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;
typedef __hip_bfloat16 hbf16;

#else  // no MMAC intrinsics on this arch

#endif

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#    if defined(__gfx936__) || defined(__gfx938__)
using dm_floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
#    endif

__device__ __forceinline__ static void dm_atomic_add_mmac_row(float* row16, int g, float x0, float x1, float x2,
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
#    else
    float* p = row16 + g;
    atomicAdd(&p[0], x0);
    atomicAdd(&p[4], x1);
    atomicAdd(&p[8], x2);
    atomicAdd(&p[12], x3);
#    endif
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
#    if defined(__gfx936__) || defined(__gfx938__)
    constexpr bool PACK_DQ = D % 32 == 0;
#    else
    constexpr bool PACK_DQ = false;
#    endif
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

namespace ftrain {
namespace {
static bool launch_indexer_bwd_dense_mmac_full(hipStream_t stream, const void* q, const void* w, const void* k,
                                               const float* attn_score, const float* attn_l1norm, float* index_score,
                                               const float* index_lse, void* dq, void* dw, void* dk, float* dk_float,
                                               int* done_count, const int* cu_q, const int* cu_kv,
                                               const int* q_causal_offsets, int rows, int Sq, int Sk, int Hq, int D,
                                               int n_seqs, int ratio, float sm_scale, float loss_coeff, float grad_loss,
                                               int score_stride, int dk_rows_total);
bool launch_indexer_bwd_dense_thd(hipStream_t stream, const void* index_q, const void* weights, const void* index_k,
                                  const float* attn_score, const float* attn_l1norm, float* index_score,
                                  const float* index_lse, void* dq, void* dw, void* dk, float* dk_float,
                                  int* done_count, const int* cu_seqlens_q, const int* cu_seqlens_kv,
                                  const int* q_causal_offsets, int total_q, int total_k, int max_sk, int H, int D,
                                  int n_seqs, int ratio, float sm_scale, float loss_coeff, float grad_loss);

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
    switch (D) {
        case 16:
            (void)hipFuncSetAttribute((const void*)dense_indexer_bwd_fused_kernel<16>,
                                      hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            hipLaunchKernelGGL((dense_indexer_bwd_fused_kernel<16>), rows, IDXB_THREADS, shm, stream, (const bf16*)q,
                               (const bf16*)w, (const bf16*)k, attn_score, attn_l1norm, index_score, index_lse,
                               (bf16*)dq, (bf16*)dw, dk_float, (bf16*)dk, done_count, cu_q, cu_kv, q_causal_offsets,
                               rows, Sq, Sk, Hq, D, n_seqs, ratio, sm_scale, loss_coeff, grad_loss, score_stride,
                               dk_rows_total);
            break;
        case 32:
            (void)hipFuncSetAttribute((const void*)dense_indexer_bwd_fused_kernel<32>,
                                      hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            hipLaunchKernelGGL((dense_indexer_bwd_fused_kernel<32>), rows, IDXB_THREADS, shm, stream, (const bf16*)q,
                               (const bf16*)w, (const bf16*)k, attn_score, attn_l1norm, index_score, index_lse,
                               (bf16*)dq, (bf16*)dw, dk_float, (bf16*)dk, done_count, cu_q, cu_kv, q_causal_offsets,
                               rows, Sq, Sk, Hq, D, n_seqs, ratio, sm_scale, loss_coeff, grad_loss, score_stride,
                               dk_rows_total);
            break;
        case 48:
            (void)hipFuncSetAttribute((const void*)dense_indexer_bwd_fused_kernel<48>,
                                      hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            hipLaunchKernelGGL((dense_indexer_bwd_fused_kernel<48>), rows, IDXB_THREADS, shm, stream, (const bf16*)q,
                               (const bf16*)w, (const bf16*)k, attn_score, attn_l1norm, index_score, index_lse,
                               (bf16*)dq, (bf16*)dw, dk_float, (bf16*)dk, done_count, cu_q, cu_kv, q_causal_offsets,
                               rows, Sq, Sk, Hq, D, n_seqs, ratio, sm_scale, loss_coeff, grad_loss, score_stride,
                               dk_rows_total);
            break;
        case 64:
            (void)hipFuncSetAttribute((const void*)dense_indexer_bwd_fused_kernel<64>,
                                      hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            hipLaunchKernelGGL((dense_indexer_bwd_fused_kernel<64>), rows, IDXB_THREADS, shm, stream, (const bf16*)q,
                               (const bf16*)w, (const bf16*)k, attn_score, attn_l1norm, index_score, index_lse,
                               (bf16*)dq, (bf16*)dw, dk_float, (bf16*)dk, done_count, cu_q, cu_kv, q_causal_offsets,
                               rows, Sq, Sk, Hq, D, n_seqs, ratio, sm_scale, loss_coeff, grad_loss, score_stride,
                               dk_rows_total);
            break;
        case 80:
            (void)hipFuncSetAttribute((const void*)dense_indexer_bwd_fused_kernel<80>,
                                      hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            hipLaunchKernelGGL((dense_indexer_bwd_fused_kernel<80>), rows, IDXB_THREADS, shm, stream, (const bf16*)q,
                               (const bf16*)w, (const bf16*)k, attn_score, attn_l1norm, index_score, index_lse,
                               (bf16*)dq, (bf16*)dw, dk_float, (bf16*)dk, done_count, cu_q, cu_kv, q_causal_offsets,
                               rows, Sq, Sk, Hq, D, n_seqs, ratio, sm_scale, loss_coeff, grad_loss, score_stride,
                               dk_rows_total);
            break;
        case 96:
            (void)hipFuncSetAttribute((const void*)dense_indexer_bwd_fused_kernel<96>,
                                      hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            hipLaunchKernelGGL((dense_indexer_bwd_fused_kernel<96>), rows, IDXB_THREADS, shm, stream, (const bf16*)q,
                               (const bf16*)w, (const bf16*)k, attn_score, attn_l1norm, index_score, index_lse,
                               (bf16*)dq, (bf16*)dw, dk_float, (bf16*)dk, done_count, cu_q, cu_kv, q_causal_offsets,
                               rows, Sq, Sk, Hq, D, n_seqs, ratio, sm_scale, loss_coeff, grad_loss, score_stride,
                               dk_rows_total);
            break;
        case 112:
            (void)hipFuncSetAttribute((const void*)dense_indexer_bwd_fused_kernel<112>,
                                      hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            hipLaunchKernelGGL((dense_indexer_bwd_fused_kernel<112>), rows, IDXB_THREADS, shm, stream, (const bf16*)q,
                               (const bf16*)w, (const bf16*)k, attn_score, attn_l1norm, index_score, index_lse,
                               (bf16*)dq, (bf16*)dw, dk_float, (bf16*)dk, done_count, cu_q, cu_kv, q_causal_offsets,
                               rows, Sq, Sk, Hq, D, n_seqs, ratio, sm_scale, loss_coeff, grad_loss, score_stride,
                               dk_rows_total);
            break;
        case 128:
            (void)hipFuncSetAttribute((const void*)dense_indexer_bwd_fused_kernel<128>,
                                      hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            hipLaunchKernelGGL((dense_indexer_bwd_fused_kernel<128>), rows, IDXB_THREADS, shm, stream, (const bf16*)q,
                               (const bf16*)w, (const bf16*)k, attn_score, attn_l1norm, index_score, index_lse,
                               (bf16*)dq, (bf16*)dw, dk_float, (bf16*)dk, done_count, cu_q, cu_kv, q_causal_offsets,
                               rows, Sq, Sk, Hq, D, n_seqs, ratio, sm_scale, loss_coeff, grad_loss, score_stride,
                               dk_rows_total);
            break;
        default:
            fprintf(stderr,
                    "[indexer_bwd_dense_mmac_full] unsupported D=%d " "(fused path requires D in {16, 32, 48, 64, 80, 96, 112, 128})\n", D);
            return false;
    }
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_indexer_bwd_dense_thd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}

static bool launch_indexer_bwd_dense_thd(hipStream_t stream, const void* index_q, const void* weights,
                                         const void* index_k, const float* attn_score, const float* attn_l1norm,
                                         float* index_score, const float* index_lse, void* dq, void* dw, void* dk,
                                         float* dk_float, int* done_count, const int* cu_seqlens_q,
                                         const int* cu_seqlens_kv, const int* q_causal_offsets, int total_q,
                                         int total_k, int max_sk, int H, int D, int n_seqs, int ratio, float sm_scale,
                                         float loss_coeff, float grad_loss) {
    if (D <= 0 || D % 16 != 0 || D > 128 || H < 1 || H > 64 || !dense_mmac_supported()) return false;

    // Fused single-kernel path (MMAC tensor core) for the whole packed batch:
    // score + softmax + A + dq + dk in ONE launch.  No non-fused fallback
    // and no per-segment launch loop.
    if (!(D % 16 == 0 && D <= 128 && H <= 64 && dense_mmac_supported() && dense_mmac_shm_fits(H, D, max_sk, true))) {
        return false;
    }
    if (!launch_indexer_bwd_dense_mmac_full(
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

bool bf16IndexerBwdDenseThdLaunchFits(int heads, int width, int columns, int queries) {
    (void)queries;
    return dense_mmac_shm_fits(heads, width, columns, true);
}

bool launchBf16IndexerBwdDenseThd(FTrainStream stream, const void* index_q, const void* weights, const void* index_k,
                                  const float* attn_score, const float* attn_l1norm, float* index_score,
                                  const float* index_lse, void* dq, void* dw, void* dk, float* dk_float,
                                  int* done_count, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                  const int* q_causal_offsets, int total_q, int total_k, int max_seqlen_kv, int heads,
                                  int head_dim, int num_sequences, int ratio, float sm_scale, float loss_coeff,
                                  float grad_loss) {
    return launch_indexer_bwd_dense_thd(stream, index_q, weights, index_k, attn_score, attn_l1norm, index_score,
                                        index_lse, dq, dw, dk, dk_float, done_count, cu_seqlens_q, cu_seqlens_k,
                                        q_causal_offsets, total_q, total_k, max_seqlen_kv, heads, head_dim,
                                        num_sequences, ratio, sm_scale, loss_coeff, grad_loss);
}

bool bf16IndexerBwdDenseThdDenseMmacSupported() { return dense_mmac_supported(); }

void bf16IndexerBwdDenseThdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    const hipError_t status = hipMemsetAsync(memory, 0, static_cast<std::size_t>(bytes), stream);
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Workspace zero-fill failed: %s", hipGetErrorString(status));
    }
}

}  // namespace ftrain
