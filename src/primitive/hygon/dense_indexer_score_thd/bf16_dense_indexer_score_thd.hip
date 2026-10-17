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
#include "flash_train/primitive/hygon/dense_indexer_score_thd/bf16_dense_indexer_score_thd.hpp"

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

#define DENSE_CHUNK 32
#define DENSE_THREADS 256
#define DENSE_DCHUNK 128
#define DENSE_MAX_TILES 36  // D <= 576 (DSv4 absorbed MLA)

static inline bool dense_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx928") != nullptr || strstr(prop.gcnArchName, "gfx936") != nullptr ||
           strstr(prop.gcnArchName, "gfx938") != nullptr;
}

static size_t dense_mmac_shm_largeD_tpl(int Hq, int D, int Sk) {
    int Hq_pad  = (Hq + 15) & ~15;
    int dd      = (D < DENSE_DCHUNK ? D : DENSE_DCHUNK) + 8;
    int qk_rows = (Hq_pad > 2 * DENSE_CHUNK) ? Hq_pad : 2 * DENSE_CHUNK;
    return (size_t)Sk * 4 + (size_t)DENSE_THREADS * 4 + (size_t)qk_rows * dd * 2;
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

static inline bool dense_mmac_storage_fits(int heads, int width, int keys, bool packed) {
    if (heads <= 0 || heads > 128 || width <= 0 || width > 576 || width % 16 || keys <= 0) return false;
    if (!dense_mmac_shm_fits(heads, width, keys, false)) return false;

    if (width > 128 && width != 256 && width != 448 && width != 512 && width != 576) return false;
    return packed || (width <= 128 && heads <= 64) ||
           dsa_shared_memory_fits(dense_mmac_shm_largeD_tpl(heads, width, keys));
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

// Adapted from fused_dsa/hip_dsa/kernels/dense_mmac.hip.

typedef uint16_t bf16;

#define DENSE_CHUNK 32
#define DENSE_THREADS 256

#define DENSE_DCHUNK 128
#define DENSE_MAX_TILES 36  // D <= 576 (DSv4 absorbed MLA)

#define DM_MODE2_NACC 2

__device__ static inline float bf16_to_float_dm(uint16_t v) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.u = (uint32_t)v << 16;
    return conv.f;
}

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;
typedef __hip_bfloat16 hbf16;

__device__ __forceinline__ static void dm_zero_bf16(hbf16* dst, int n) {
    char* p  = reinterpret_cast<char*>(dst);
    // Advance to the next 16-byte boundary before issuing uint4 stores.
    int lead = (int)((16u - ((uintptr_t)p & 15u)) & 15u) >> 1;
    if (lead > n) lead = n;
    for (int i = threadIdx.x; i < lead; i += DENSE_THREADS) dst[i] = hbf16(0);
    const int nvec   = (n - lead) >> 3;
    uint4* v         = reinterpret_cast<uint4*>(p + (size_t)lead * 2);
    const uint4 zero = make_uint4(0, 0, 0, 0);
    for (int i = threadIdx.x; i < nvec; i += DENSE_THREADS) v[i] = zero;
    for (int i = lead + (nvec << 3) + threadIdx.x; i < n; i += DENSE_THREADS) dst[i] = hbf16(0);
}

template<int MODE>
__device__ __forceinline__ static float dm_epi(float dot, float head_a, float scale) {
    if (MODE == 2) return expf(dot * scale - head_a);
    float d = dot * scale;
    // Match torch.relu's NaN propagation (relu(NaN) == NaN) so degenerate
    // NaN inputs behave identically to the eager reference instead of
    // collapsing NaN to 0.
    return (d > 0.0f ? d : (d != d ? d : 0.0f)) * head_a;
}

template<int MODE>
__device__ __forceinline__ static float dm_epi_acc(const DUFragment<accumulator, 16, 16, 16, float>* acc, int j,
                                                   int nacc, float head_a, float scale) {
    if (MODE == 2 && nacc > 1) {
        double s = 0.0;
#    pragma unroll
        for (int g = 0; g < DM_MODE2_NACC; g++)
            if (g < nacc) s += (double)acc[g].x[j];
        return expf((float)(s * (double)scale - (double)head_a));
    }
    return dm_epi<MODE>(acc[0].x[j], head_a, scale);
}

template<int MODE>
__device__ __forceinline__ static float dense_xor(float x, int off) {
    int bits = __builtin_bit_cast(int, x), y;
    if constexpr (MODE > 0) {
        if (off == 1) {
            y = __builtin_amdgcn_mov_dpp(bits, 0xb1, 0xf, 0xf, true);
            return __builtin_bit_cast(float, y);
        }
        if (off == 2) {
            y = __builtin_amdgcn_mov_dpp(bits, 0x4e, 0xf, 0xf, true);
            return __builtin_bit_cast(float, y);
        }
    }
    if constexpr (MODE == 2) {
        if (off == 8) {
            y = __builtin_amdgcn_mov_dpp(bits, 0x128, 0xf, 0xf, true);
            return __builtin_bit_cast(float, y);
        }
        if (off == 4) {
            y = __builtin_amdgcn_update_dpp(bits, bits, 0x104, 0xf, 0x5, true);
            y = __builtin_amdgcn_update_dpp(y, bits, 0x114, 0xf, 0xa, true);
            return __builtin_bit_cast(float, y);
        }
    }
    return __shfl_xor(x, off);
}

template<int MODE, bool USE_DPP = false>
__device__ __forceinline__ static void dm_score_chunk_epilogue(const DUFragment<accumulator, 16, 16, 16, float>* acc,
                                                               int ntiles, int ck, int chunk, int lane, int nacc,
                                                               float head_a, float scale, float* score,
                                                               float* part = nullptr, int warp = 0) {
#    pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        if (nt < ntiles) {
            const DUFragment<accumulator, 16, 16, 16, float>* ac = acc + nt * nacc;
            float v0                                             = dm_epi_acc<MODE>(ac, 0, nacc, head_a, scale);
            float v1                                             = dm_epi_acc<MODE>(ac, 1, nacc, head_a, scale);
            float v2                                             = dm_epi_acc<MODE>(ac, 2, nacc, head_a, scale);
            float v3                                             = dm_epi_acc<MODE>(ac, 3, nacc, head_a, scale);
#    pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                if constexpr (USE_DPP)
                    v0 += dense_xor<2>(v0, off);
                else
                    v0 += __shfl_xor(v0, off);
                if constexpr (USE_DPP)
                    v1 += dense_xor<2>(v1, off);
                else
                    v1 += __shfl_xor(v1, off);
                if constexpr (USE_DPP)
                    v2 += dense_xor<2>(v2, off);
                else
                    v2 += __shfl_xor(v2, off);
                if constexpr (USE_DPP)
                    v3 += dense_xor<2>(v3, off);
                else
                    v3 += __shfl_xor(v3, off);
            }
            if ((lane & 15) == 0) {
                int g   = lane >> 4;
                int col = nt * 16 + g;
                if (part) {
                    // Per-warp private merge: no LDS atomics, no cross-warp
                    // serialization; the caller sums the four slices.
                    float* pd = part + (size_t)warp * 32;
                    if (col < ck) pd[col] = v0;
                    if (col + 4 < ck) pd[col + 4] = v1;
                    if (col + 8 < ck) pd[col + 8] = v2;
                    if (col + 12 < ck) pd[col + 12] = v3;
                } else {
                    if (col < ck) atomicAdd(&score[chunk + col], v0);
                    if (col + 4 < ck) atomicAdd(&score[chunk + col + 4], v1);
                    if (col + 8 < ck) atomicAdd(&score[chunk + col + 8], v2);
                    if (col + 12 < ck) atomicAdd(&score[chunk + col + 12], v3);
                }
            }
        }
    }
}

__device__ __forceinline__ static void dm_stage_q_chunk(const bf16* __restrict__ q, int row, int Hq, int D, int d0,
                                                        int dchunk, int Hq_pad, int QLD, hbf16* q_lds) {
    const hbf16* qrow = (const hbf16*)(q + (size_t)row * Hq * D + d0);
    for (int h = threadIdx.x; h < Hq; h += DENSE_THREADS) {
        for (int d = 0; d < dchunk; d += 8) {
            float2* dst       = (float2*)(q_lds + (size_t)h * QLD + d);
            const float2* src = (const float2*)(qrow + (size_t)h * D + d);
            dst[0]            = src[0];
            dst[1]            = src[1];
        }
    }
}

template<int MODE, int DIM = 0>
static __global__ void __launch_bounds__(DENSE_THREADS)
    dense_score_mmac_thd_kernel(const bf16* __restrict__ q,                // (total_q, Hq, D)
                                const bf16* __restrict__ k,                // (total_k, D)  (Hkv == 1)
                                const bf16* __restrict__ w,                // (total_q, Hq) [MODE 0/1]
                                const float* __restrict__ lse,             // (total_q, Hq) [MODE 2]
                                const int* __restrict__ cu_q,              // (n_seqs + 1)
                                const int* __restrict__ cu_kv,             // (n_seqs + 1)
                                const int* __restrict__ q_causal_offsets,  // (n_seqs), optional
                                float* __restrict__ out,                   // (total_q, max_sk)
                                float* __restrict__ denom,                 // (total_q,)     [MODE 1/2]
                                int total_q, int total_k, int max_sk, int Hq, int D, int n_seqs, int ratio,
                                float scale) {
    const int row    = blockIdx.x;
    const int lane   = threadIdx.x & 63;
    const int warp   = threadIdx.x >> 6;
    const int Hq_pad = (Hq + 15) & ~15;
    // DIM > 0 keeps the hoisted A fragments statically indexed (registers);
    // Every dispatch is specialized; runtime-D spill instantiations are removed.
    static_assert(DIM > 0, "Runtime-D spill kernels are not supported");
    constexpr int Dd = DIM;
    const int QLD    = (Dd < DENSE_DCHUNK ? Dd : DENSE_DCHUNK) + 8;
    const int KLD    = QLD;

    ThdRowMeta m  = thd_row_meta(cu_q, cu_kv, n_seqs, row, total_k);
    int col_limit = thd_col_limit(m, ratio, q_causal_offsets);
    if (col_limit > max_sk) col_limit = max_sk;

    extern __shared__ char smem_raw[];
    hbf16* q_lds = (hbf16*)smem_raw;
    hbf16* k_lds = q_lds + (size_t)Hq_pad * QLD;
    float* score = (float*)(k_lds + (size_t)DENSE_CHUNK * KLD);
    float* red   = score + max_sk;

    for (int i = threadIdx.x; i < max_sk; i += DENSE_THREADS) score[i] = 0.0f;
    dm_zero_bf16(q_lds, Hq_pad * QLD);
    // Zero k_lds once at kernel start (partial tail chunk rows stay zero).
    dm_zero_bf16(k_lds, DENSE_CHUNK * KLD);
    __syncthreads();

    // Heads run in 64-head passes (4 warps x 16 heads); full-D A fragments
    // are hoisted per pass (D-chunked Q staging), same as the large-D BSHD
    // kernel.
    const int n_hpass = (Hq_pad + 63) / 64;
    for (int hp = 0; hp < n_hpass; hp++) {
        const int r0 = hp * 64 + warp * 16;

        const int h  = hp * 64 + warp * 16 + (lane & 15);
        float head_a = 0.0f;
        if (MODE == 2)
            head_a = (h < Hq) ? lse[(size_t)row * Hq + h] : INFINITY;
        else
            head_a = (h < Hq) ? bf16_to_float_dm(w[(size_t)row * Hq + h]) : 0.0f;

        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> af[DENSE_MAX_TILES];
#    pragma unroll
        for (int d0 = 0; d0 < Dd; d0 += DENSE_DCHUNK) {
            int dchunk = Dd - d0;
            if (dchunk > DENSE_DCHUNK) dchunk = DENSE_DCHUNK;
            int qld = dchunk + 8;
            dm_stage_q_chunk(q, row, Hq, D, d0, dchunk, Hq_pad, qld, q_lds);
            __syncthreads();
#    pragma unroll
            for (int kk = 0; kk < dchunk / 16; kk++) {
                if (r0 < Hq_pad) du_load_matrix_sync(af[(d0 + kk * 16) / 16], q_lds + (size_t)r0 * qld + kk * 16, qld);
            }
            __syncthreads();
        }

        for (int chunk = 0; chunk < col_limit; chunk += DENSE_CHUNK) {
            int ck = col_limit - chunk;
            if (ck > DENSE_CHUNK) ck = DENSE_CHUNK;
            int ntiles     = (ck + 15) >> 4;
            const int nacc = (MODE == 2 && Dd > DENSE_DCHUNK) ? DM_MODE2_NACC : 1;
            DUFragment<accumulator, 16, 16, 16, float> acc[2 * DM_MODE2_NACC];
#    pragma unroll
            for (int i = 0; i < 2 * DM_MODE2_NACC; i++)
                if (i < 2 * nacc) du_fill_fragment(acc[i], 0.0f);

#    pragma unroll
            for (int d0 = 0; d0 < Dd; d0 += DENSE_DCHUNK) {
                int dchunk = Dd - d0;
                if (dchunk > DENSE_DCHUNK) dchunk = DENSE_DCHUNK;
                int kld = dchunk + 8;
                // THD K rows: packed (kv_offset + chunk + n) with global row
                // stride D (no per-batch Sk stride).
                for (int u = threadIdx.x; u < ck * (dchunk / 8); u += DENSE_THREADS) {
                    int n           = u / (dchunk / 8);
                    int d8          = u % (dchunk / 8);
                    const bf16* src = k + (size_t)(m.kv_offset + chunk + n) * D + d0 + d8 * 8;
                    // Store with the SAME per-chunk stride the B fragments
                    // are loaded with below (kld); KLD is 136 for full
                    // 128-col chunks but 72 for a partial tail chunk
                    // (e.g. D=448 -> 384 + 64), so using KLD here would
                    // misalign the tail chunk and corrupt every score.
                    bf16* dst         = (bf16*)(k_lds + (size_t)n * kld + d8 * 8);
                    ((float2*)dst)[0] = ((const float2*)src)[0];
                    ((float2*)dst)[1] = ((const float2*)src)[1];
                }
                __syncthreads();
#    pragma unroll
                for (int kk = 0; kk < dchunk / 16; kk++) {
                    int t = (d0 + kk * 16) / 16;
#    pragma unroll
                    for (int nt = 0; nt < 2; nt++) {
                        if (nt < ntiles) {
                            DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr;
                            du_load_matrix_sync(bfr, k_lds + (size_t)(nt * 16) * kld + kk * 16, kld);
                            if (r0 < Hq_pad)
                                du_mma_sync(acc[nt * nacc + (t & (nacc - 1))], af[t], bfr,
                                            acc[nt * nacc + (t & (nacc - 1))]);
                        }
                    }
                }
                __syncthreads();
            }
            dm_score_chunk_epilogue<MODE>(acc, ntiles, ck, chunk, lane, nacc, head_a, scale, score);
        }
    }
    __syncthreads();

    // Write out with the THD causal mask.  cuDNN's score kernels prefill
    // masked/padded positions with -inf for every mode and every ratio;
    // Megatron's ``_kl_loss_from_dense_scores`` gates on ``isfinite`` of the
    // index scores, so masked entries must be non-finite even for ratio == 1.
    for (int i = threadIdx.x; i < max_sk; i += DENSE_THREADS) {
        float v = score[i];
        if (i >= col_limit) v = -INFINITY;
        out[(size_t)row * max_sk + i] = v;
    }
    if (MODE == 0) return;

    if (MODE == 1) {
        float m = -INFINITY;
        for (int i = threadIdx.x; i < col_limit; i += DENSE_THREADS) m = fmaxf(m, score[i]);
        red[threadIdx.x] = m;
        __syncthreads();
        for (int st = DENSE_THREADS / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
            __syncthreads();
        }
        float mx = red[0];
        __syncthreads();
        float s = 0.0f;
        for (int i = threadIdx.x; i < col_limit; i += DENSE_THREADS) s += expf(score[i] - mx);
        red[threadIdx.x] = s;
        __syncthreads();
        for (int st = DENSE_THREADS / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
            __syncthreads();
        }
        if (threadIdx.x == 0) denom[row] = (col_limit == 0) ? -INFINITY : mx + logf(red[0]);
    } else {
        float s = 0.0f;
        for (int i = threadIdx.x; i < col_limit; i += DENSE_THREADS) s += score[i];
        red[threadIdx.x] = s;
        __syncthreads();
        for (int st = DENSE_THREADS / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
            __syncthreads();
        }
        if (threadIdx.x == 0) denom[row] = red[0];
    }
}

#else  // gfx906 / gfx926 / others: no MMAC intrinsics — never launched at runtime.

template<int MODE, int DIM>
static __global__ void __launch_bounds__(DENSE_THREADS)
    dense_score_mmac_thd_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ w,
                                const float* __restrict__ lse, const int* __restrict__ cu_q,
                                const int* __restrict__ cu_kv, const int* __restrict__ q_causal_offsets,
                                float* __restrict__ out, float* __restrict__ denom, int total_q, int total_k,
                                int max_sk, int Hq, int D, int n_seqs, int ratio, float scale) {}

#endif  // supported gfx archs

namespace ftrain {
namespace {

static void launch_dense_score_thd_dispatch(hipStream_t stream, const void* q, const void* k, const void* w_or_lse,
                                            float* out, float* denom, const int* cu_q, const int* cu_kv,
                                            const int* q_causal_offsets, int total_q, int total_k, int max_sk, int Hq,
                                            int D, int n_seqs, int ratio, float scale, int mode);
void launch_dense_indexer_score_thd_mmac(hipStream_t stream, const void* q, const void* k, const void* w, float* out,
                                         float* denom, const int* cu_q, const int* cu_kv, const int* q_causal_offsets,
                                         int total_q, int total_k, int max_sk, int Hq, int D, int n_seqs, int ratio,
                                         float sm_scale);
bool launch_dense_indexer_score_thd(hipStream_t stream, const void* q, const void* k, const void* w, float* out,
                                    float* denom, const int* cu_seqlens_q, const int* cu_seqlens_kv,
                                    const int* q_causal_offsets, int total_q, int total_k, int max_sk, int Hq, int Hkv,
                                    int D, int n_seqs, int ratio, float sm_scale);

static void launch_dense_score_thd_dispatch(hipStream_t stream, const void* q, const void* k, const void* w_or_lse,
                                            float* out, float* denom, const int* cu_q, const int* cu_kv,
                                            const int* q_causal_offsets, int total_q, int total_k, int max_sk, int Hq,
                                            int D, int n_seqs, int ratio, float scale, int mode) {
    size_t shm        = dense_mmac_shm(Hq, D, max_sk, false);
    const int thd_dim = D;

    if (mode == 0) {
        switch (thd_dim) {
            case 16:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 16>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            case 32:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 32>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            case 48:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 48>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            case 64:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 64>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            case 80:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 80>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            case 96:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 96>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            case 112:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 112>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            case 128:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 128>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            case 256:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 256>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            case 448:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 448>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            case 512:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 512>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            case 576:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<0, 576>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, nullptr, total_q, total_k, max_sk, Hq, D, n_seqs, ratio,
                                   scale);
                break;
            default:
                throw ftrain::Exception(
                    FTRAIN_STATUS_UNSUPPORTED,
                    "dense score: unsupported head dimension for parallel specialized kernel; "
                    "supported D: 16..128 in steps of 16, 256, 448, 512, 576; "
                    "runtime-D spill fallback has been removed");
        }
    } else if (mode == 1) {
        switch (thd_dim) {
            case 16:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 16>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 32:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 32>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 48:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 48>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 64:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 64>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 80:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 80>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 96:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 96>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 112:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 112>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 128:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 128>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 256:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 256>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 448:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 448>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 512:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 512>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 576:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<1, 576>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, (const bf16*)w_or_lse, nullptr, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            default:
                throw ftrain::Exception(
                    FTRAIN_STATUS_UNSUPPORTED,
                    "dense score: unsupported head dimension for parallel specialized kernel; "
                    "supported D: 16..128 in steps of 16, 256, 448, 512, 576; "
                    "runtime-D spill fallback has been removed");
        }
    } else {
        switch (thd_dim) {
            case 16:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 16>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 32:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 32>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 48:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 48>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 64:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 64>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 80:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 80>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 96:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 96>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 112:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 112>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 128:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 128>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 256:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 256>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 448:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 448>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 512:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 512>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            case 576:
                hipLaunchKernelGGL((dense_score_mmac_thd_kernel<2, 576>), total_q, DENSE_THREADS, shm, stream,
                                   (const bf16*)q, (const bf16*)k, nullptr, (const float*)w_or_lse, cu_q, cu_kv,
                                   q_causal_offsets, out, denom, total_q, total_k, max_sk, Hq, D, n_seqs, ratio, scale);
                break;
            default:
                throw ftrain::Exception(
                    FTRAIN_STATUS_UNSUPPORTED,
                    "dense score: unsupported head dimension for parallel specialized kernel; "
                    "supported D: 16..128 in steps of 16, 256, 448, 512, 576; "
                    "runtime-D spill fallback has been removed");
        }
    }
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_dense_indexer_score_thd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
}

static void launch_dense_indexer_score_thd_mmac(hipStream_t stream, const void* q, const void* k, const void* w,
                                                float* out, float* denom, const int* cu_q, const int* cu_kv,
                                                const int* q_causal_offsets, int total_q, int total_k, int max_sk,
                                                int Hq, int D, int n_seqs, int ratio, float sm_scale) {
    launch_dense_score_thd_dispatch(stream, q, k, w, out, denom, cu_q, cu_kv, q_causal_offsets, total_q, total_k,
                                    max_sk, Hq, D, n_seqs, ratio, sm_scale, 1);
}

static bool launch_dense_indexer_score_thd(hipStream_t stream, const void* q, const void* k, const void* w, float* out,
                                           float* denom, const int* cu_seqlens_q, const int* cu_seqlens_kv,
                                           const int* q_causal_offsets, int total_q, int total_k, int max_sk, int Hq,
                                           int Hkv, int D, int n_seqs, int ratio, float sm_scale) {
    // Fused single-kernel path (MMAC tensor core) only; no non-fused
    // fallback and no per-segment launch loop.
    if (!(Hkv == 1 && dense_mmac_supported() &&
          dense_mmac_storage_fits(Hq, D, max_sk, true))) {
        return false;
    }
    launch_dense_indexer_score_thd_mmac(stream, q, k, w, out, denom, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets,
                                        total_q, total_k, max_sk, Hq, D, n_seqs, ratio, sm_scale);
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_dense_indexer_score_thd HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}
}  // namespace

bool bf16DenseIndexerScoreThdLaunchFits(int heads, int width, int keys) {
    return dense_mmac_storage_fits(heads, width, keys, true);
}

bool launchBf16DenseIndexerScoreThd(FTrainStream stream, const void* q, const void* k, const void* w, float* out,
                                    float* denom, const int* cu_seqlens_q, const int* cu_seqlens_kv,
                                    const int* q_causal_offsets, int total_q, int total_k, int max_seqlen_kv,
                                    int query_heads, int kv_heads, int head_dim, int num_sequences, int ratio,
                                    float sm_scale) {
    return launch_dense_indexer_score_thd(stream, q, k, w, out, denom, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets,
                                          total_q, total_k, max_seqlen_kv, query_heads, kv_heads, head_dim,
                                          num_sequences, ratio, sm_scale);
}

bool bf16DenseIndexerScoreThdDenseMmacSupported() { return dense_mmac_supported(); }

}  // namespace ftrain
