// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <math.h>
#include <stdexcept>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#    include <du_mma.h>
#endif

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/dense_attn_score/bf16_dense_attn_score.hpp"

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

static inline size_t dense_mmac_shm_v2(int D, int Sk) {
    return (size_t)DENSE_CHUNK * (D + 8) * 2 + (size_t)Sk * 4 + DENSE_THREADS * 4;
}

static inline bool dense_mmac_storage_fits(int heads, int width, int keys, bool packed) {
    if (heads <= 0 || heads > 128 || width <= 0 || width > 576 || width % 16 || keys <= 0) return false;
    if (!dense_mmac_shm_fits(heads, width, keys, false)) return false;

    if (width > 128 && width != 256 && width != 448 && width != 512 && width != 576) return false;
    return packed || (width <= 128 && heads <= 64) ||
           dsa_shared_memory_fits(dense_mmac_shm_largeD_tpl(heads, width, keys));
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

template<int MODE>
static __global__ void __launch_bounds__(DENSE_THREADS)
    dense_score_mmac_kernel(const bf16* __restrict__ q,     // (B, Sq, Hq, D)
                            const bf16* __restrict__ k,     // (B, Sk, 1, D)
                            const bf16* __restrict__ w,     // (B, Sq, Hq)   [MODE 0/1]
                            const float* __restrict__ lse,  // (B, Sq, Hq)   [MODE 2]
                            float* __restrict__ out,        // (B, Sq, Sk)
                            float* __restrict__ denom,      // (B, Sq)       [MODE 1/2]
                            int B, int Sq, int Sk, int Hq, int D, int ratio, float scale, int score_stride);

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;
typedef __hip_bfloat16 hbf16;

__device__ __forceinline__ static void dm_zero_bf16(hbf16* dst, int n) {
    char* p  = reinterpret_cast<char*>(dst);
    int lead = (int)(((uintptr_t)p & 15u) >> 1);  // bf16 elements to 16B alignment
    if (lead > n) lead = n;
    for (int i = threadIdx.x; i < lead; i += DENSE_THREADS) dst[i] = hbf16(0);
    const int nvec   = (n - lead) >> 3;
    uint4* v         = reinterpret_cast<uint4*>(p + (size_t)lead * 2);
    const uint4 zero = make_uint4(0, 0, 0, 0);
    for (int i = threadIdx.x; i < nvec; i += DENSE_THREADS) v[i] = zero;
    for (int i = lead + (nvec << 3) + threadIdx.x; i < n; i += DENSE_THREADS) dst[i] = hbf16(0);
}

__device__ __forceinline__ static void dm_mma_chunk(const hbf16* q_lds, const hbf16* k_lds, int r0, int Hq_pad, int QLD,
                                                    int KLD, int D, int ntiles,
                                                    DUFragment<accumulator, 16, 16, 16, float>* acc) {
#    pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        if (nt < ntiles) du_fill_fragment(acc[nt], 0.0f);
    }
    for (int kk = 0; kk < D / 16; kk++) {
        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> a;
        if (r0 < Hq_pad) { du_load_matrix_sync(a, q_lds + (size_t)r0 * QLD + kk * 16, QLD); }
#    pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            if (nt < ntiles) {
                DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr;
                du_load_matrix_sync(bfr, k_lds + (size_t)(nt * 16) * KLD + kk * 16, KLD);
                if (r0 < Hq_pad) du_mma_sync(acc[nt], a, bfr, acc[nt]);
            }
        }
    }
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

__device__ __forceinline__ static void dm_load_k(hbf16* k_lds, const bf16* __restrict__ k, int b, int Sk, int chunk,
                                                 int ck, int D, int KLD) {
    for (int u = threadIdx.x; u < ck * (D / 8); u += DENSE_THREADS) {
        int n             = u / (D / 8);
        int d8            = u % (D / 8);
        const bf16* src   = k + (size_t)b * (size_t)Sk * D + (size_t)(chunk + n) * D + d8 * 8;
        bf16* dst         = (bf16*)(k_lds + (size_t)n * KLD + d8 * 8);
        ((float2*)dst)[0] = ((const float2*)src)[0];
        ((float2*)dst)[1] = ((const float2*)src)[1];
    }
}

__device__ __forceinline__ static void dm_stage_q(const bf16* __restrict__ q, int row, int Hq, int D, int Hq_pad,
                                                  int QLD, hbf16* q_lds) {
    const hbf16* qrow = (const hbf16*)(q + (size_t)row * Hq * D);
    dm_zero_bf16(q_lds, Hq_pad * QLD);
    __syncthreads();
    for (int h = threadIdx.x; h < Hq; h += DENSE_THREADS) {
        for (int d = 0; d < D; d += 8) {
            float2* dst       = (float2*)(q_lds + (size_t)h * QLD + d);
            const float2* src = (const float2*)(qrow + (size_t)h * D + d);
            dst[0]            = src[0];
            dst[1]            = src[1];
        }
    }
    __syncthreads();
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

__device__ __forceinline__ static void dm_load_k_chunk(hbf16* k_lds, const bf16* __restrict__ k, int b, int Sk,
                                                       int chunk, int ck, int D, int d0, int dchunk, int KLD) {
    for (int u = threadIdx.x; u < ck * (dchunk / 8); u += DENSE_THREADS) {
        int n             = u / (dchunk / 8);
        int d8            = u % (dchunk / 8);
        const bf16* src   = k + (size_t)b * (size_t)Sk * D + (size_t)(chunk + n) * D + d0 + d8 * 8;
        bf16* dst         = (bf16*)(k_lds + (size_t)n * KLD + d8 * 8);
        ((float2*)dst)[0] = ((const float2*)src)[0];
        ((float2*)dst)[1] = ((const float2*)src)[1];
    }
}

template<int MODE>
static __global__ void __launch_bounds__(DENSE_THREADS)
    dense_score_mmac_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ w,
                            const float* __restrict__ lse, float* __restrict__ out, float* __restrict__ denom, int B,
                            int Sq, int Sk, int Hq, int D, int ratio, float scale, int score_stride) {
    const int row    = blockIdx.x;
    const int b      = row / Sq;
    const int sq     = row - b * Sq;
    const int lane   = threadIdx.x & 63;
    const int warp   = threadIdx.x >> 6;
    const int Hq_pad = (Hq + 15) & ~15;
    const int QLD    = D + 8;
    const int KLD    = D + 8;
    const int r0     = warp * 16;

    int col_limit = (sq + 1) / ratio;
    if (col_limit > Sk) col_limit = Sk;

    extern __shared__ char smem_raw[];
    hbf16* q_lds = (hbf16*)smem_raw;
    hbf16* k_lds = q_lds + (size_t)Hq_pad * QLD;
    float* score = (float*)(k_lds + (size_t)DENSE_CHUNK * KLD);
    float* red   = score + Sk;

    // Per-lane head epilogue scalar.
    const int h  = warp * 16 + (lane & 15);
    float head_a = 0.0f;
    if (MODE == 2)
        head_a = (h < Hq) ? lse[(size_t)row * Hq + h] : INFINITY;
    else
        head_a = (h < Hq) ? bf16_to_float_dm(w[(size_t)row * Hq + h]) : 0.0f;

    for (int i = threadIdx.x; i < Sk; i += DENSE_THREADS) score[i] = 0.0f;
    // Zero k_lds once at kernel start: partial tail chunks leave rows beyond
    // ck untouched, and the MMA reads all 16 rows of a B fragment.  Without
    // this, stale LDS from a previous kernel (possibly an INF/NaN bit
    // pattern) would poison every accumulator via 0 * stale = NaN.
    dm_zero_bf16(k_lds, DENSE_CHUNK * KLD);
    __syncthreads();

    // Q row staged once (this kernel runs for D <= DENSE_DCHUNK only).
    dm_stage_q(q, row, Hq, D, Hq_pad, QLD, q_lds);

    for (int chunk = 0; chunk < col_limit; chunk += DENSE_CHUNK) {
        int ck = col_limit - chunk;
        if (ck > DENSE_CHUNK) ck = DENSE_CHUNK;

        dm_load_k(k_lds, k, b, Sk, chunk, ck, D, KLD);
        __syncthreads();

        int ntiles = (ck + 15) >> 4;
        DUFragment<accumulator, 16, 16, 16, float> acc[2];
        dm_mma_chunk(q_lds, k_lds, r0, Hq_pad, QLD, KLD, D, ntiles, acc);
        dm_score_chunk_epilogue<MODE>(acc, ntiles, ck, chunk, lane, 1, head_a, scale, score);
        __syncthreads();
    }
    __syncthreads();

    // Write out with the ratio-causal mask.  cuDNN's score kernels prefill
    // masked positions with -inf for every mode (indexer scores, indexer LSE
    // scores and attention scores alike); Megatron's KL-loss path gates on
    // ``torch.isfinite(index_score)``, so masked entries must be non-finite.
    for (int i = threadIdx.x; i < Sk; i += DENSE_THREADS) {
        float v = score[i];
        if (i >= col_limit) v = -INFINITY;
        out[(size_t)row * score_stride + i] = v;
    }
    if (MODE == 0) return;

    // Block reduction for denom (MODE 1: logsumexp, MODE 2: plain sum).
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

template<int MODE, int DK>
static __global__ void __launch_bounds__(DENSE_THREADS)
    dense_score_mmac_kernel_v2(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ w,
                               const float* __restrict__ lse, float* __restrict__ out, float* __restrict__ denom, int B,
                               int Sq, int Sk, int Hq, int D, int ratio, float scale, int score_stride) {
    const int row    = blockIdx.x;
    const int b      = row / Sq;
    const int sq     = row - b * Sq;
    const int lane   = threadIdx.x & 63;
    const int warp   = threadIdx.x >> 6;
    const int Hq_pad = (Hq + 15) & ~15;
    const int KLD    = D + 8;
    const int r0     = warp * 16;

    int col_limit = (sq + 1) / ratio;
    if (col_limit > Sk) col_limit = Sk;

    extern __shared__ char smem_raw[];
    hbf16* k_lds = (hbf16*)smem_raw;  // [DENSE_CHUNK][KLD]
    float* score = (float*)(k_lds + (size_t)DENSE_CHUNK * KLD);
    float* red   = score + Sk;

    const int h  = warp * 16 + (lane & 15);
    float head_a = 0.0f;
    if (MODE == 2)
        head_a = (h < Hq) ? lse[(size_t)row * Hq + h] : INFINITY;
    else
        head_a = (h < Hq) ? bf16_to_float_dm(w[(size_t)row * Hq + h]) : 0.0f;

    for (int i = threadIdx.x; i < Sk; i += DENSE_THREADS) score[i] = 0.0f;
    // Zero k_lds once at kernel start (partial tail chunk rows stay zero).
    dm_zero_bf16(k_lds, DENSE_CHUNK * KLD);
    __syncthreads();

    // Hoist the full-D A fragments (this warp's 16 heads) straight from
    // global.  Per-lane loads with rows past Hq clamped to row Hq-1 and
    // zeroed: a plain du_load_matrix_sync of a 16-row fragment would read
    // past the head block when Hq % 16 != 0 (out-of-bounds at the last
    // query row) and feed stale/NaN data into the MMA epilogue.
    DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> af[DK];
    if (r0 < Hq) {
        const int h_lane  = r0 + (lane & 15);
        const int qrow_h  = (h_lane < Hq) ? h_lane : (Hq - 1);
        const int g4      = (lane >> 4) << 2;
        const hbf16* qrow = (const hbf16*)(q + (size_t)row * Hq * D + (size_t)qrow_h * D);
#    pragma unroll
        for (int kk = 0; kk < DK; kk++) {
            *(uint64_t*)&af[kk].x[0] = *(const uint64_t*)(qrow + kk * 16 + g4);
            if (h_lane >= Hq) af[kk].x[0] = af[kk].x[1] = af[kk].x[2] = af[kk].x[3] = hbf16(0);
        }
    }

    for (int chunk = 0; chunk < col_limit; chunk += DENSE_CHUNK) {
        int ck = col_limit - chunk;
        if (ck > DENSE_CHUNK) ck = DENSE_CHUNK;
        int ntiles = (ck + 15) >> 4;
        DUFragment<accumulator, 16, 16, 16, float> acc[2];
#    pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            if (nt < ntiles) du_fill_fragment(acc[nt], 0.0f);
        }

        dm_load_k(k_lds, k, b, Sk, chunk, ck, D, KLD);
        __syncthreads();
#    pragma unroll
        for (int kk = 0; kk < DK; kk++) {
#    pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                if (nt < ntiles) {
                    DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr;
                    du_load_matrix_sync(bfr, k_lds + (size_t)(nt * 16) * KLD + kk * 16, KLD);
                    if (r0 < Hq) du_mma_sync(acc[nt], af[kk], bfr, acc[nt]);
                }
            }
        }
        __syncthreads();
        dm_score_chunk_epilogue<MODE>(acc, ntiles, ck, chunk, lane, 1, head_a, scale, score);
    }
    __syncthreads();

    // Write out with the ratio-causal mask.
    for (int i = threadIdx.x; i < Sk; i += DENSE_THREADS) {
        float v = score[i];
        if (i >= col_limit) v = -INFINITY;
        out[(size_t)row * score_stride + i] = v;
    }
    if (MODE == 0) return;

    // Block reduction for denom (MODE 1: logsumexp, MODE 2: plain sum).
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

template<int MODE, int DK, bool FULL_HEADS = false, bool SCORE_RING = false>
static __global__ void __launch_bounds__(DENSE_THREADS)
    dense_indexer_prefetch_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ w,
                                  const float* __restrict__ lse, float* __restrict__ out, float* __restrict__ denom,
                                  int B, int Sq, int Sk, int Hq, int D, int ratio, float scale, int score_stride) {
    static_assert(!SCORE_RING || MODE == 0);
    D = DK * 16;
    if constexpr (FULL_HEADS) Hq = 64;
    const int row    = blockIdx.x;
    const int b      = row / Sq;
    const int sq     = row - b * Sq;
    const int lane   = threadIdx.x & 63;
    const int warp   = threadIdx.x >> 6;
    const int Hq_pad = (Hq + 15) & ~15;
    const int KLD    = D + 8;
    const int r0     = warp * 16;

    int col_limit = (sq + 1) / ratio;
    if (col_limit > Sk) col_limit = Sk;

    extern __shared__ char smem_raw[];
    hbf16* k_lds = (hbf16*)smem_raw;  // [DENSE_CHUNK][KLD]
    float* score = (float*)(k_lds + (size_t)DENSE_CHUNK * KLD);
    float* red   = score + (SCORE_RING ? DENSE_CHUNK : Sk);
    // MODE==0/SCORE_RING retires a whole 32-column chunk at a time, so the four
    // warps' partial column sums can live in four private slices (aliasing the
    // red scratch, which MODE==0 never reaches) and be merged with three adds at
    // the flush.  This replaces 128 LDS atomicAdds per chunk.
    float* score_part = (SCORE_RING && MODE == 0) ? red : nullptr;

    const int h  = warp * 16 + (lane & 15);
    float head_a = 0.0f;
    if (MODE == 2)
        head_a = (FULL_HEADS || h < Hq) ? lse[(size_t)row * Hq + h] : INFINITY;
    else
        head_a = (FULL_HEADS || h < Hq) ? bf16_to_float_dm(w[(size_t)row * Hq + h]) : 0.0f;

    for (int i = threadIdx.x; i < (SCORE_RING ? DENSE_CHUNK : Sk); i += DENSE_THREADS) score[i] = 0.0f;
    // Zero k_lds once at kernel start (partial tail chunk rows stay zero).
    dm_zero_bf16(k_lds, DENSE_CHUNK * KLD);
    __syncthreads();

    // Hoist the full-D A fragments (this warp's 16 heads) straight from
    // global.  Per-lane loads with rows past Hq clamped to row Hq-1 and
    // zeroed: a plain du_load_matrix_sync of a 16-row fragment would read
    // past the head block when Hq % 16 != 0 (out-of-bounds at the last
    // query row) and feed stale/NaN data into the MMA epilogue.
    DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> af[DK];
    if (FULL_HEADS || r0 < Hq) {
        const int h_lane  = r0 + (lane & 15);
        const int qrow_h  = (FULL_HEADS || h_lane < Hq) ? h_lane : (Hq - 1);
        const int g4      = (lane >> 4) << 2;
        const hbf16* qrow = (const hbf16*)(q + (size_t)row * Hq * D + (size_t)qrow_h * D);
#    pragma unroll
        for (int kk = 0; kk < DK; kk++) {
            *(uint64_t*)&af[kk].x[0] = *(const uint64_t*)(qrow + kk * 16 + g4);
            if (!FULL_HEADS && h_lane >= Hq) af[kk].x[0] = af[kk].x[1] = af[kk].x[2] = af[kk].x[3] = hbf16(0);
        }
    }

    if (col_limit > 0) dm_load_k(k_lds, k, b, Sk, 0, min(DENSE_CHUNK, col_limit), D, KLD);
    __syncthreads();
    for (int chunk = 0; chunk < col_limit; chunk += DENSE_CHUNK) {
        int ck = col_limit - chunk;
        if (ck > DENSE_CHUNK) ck = DENSE_CHUNK;
        int ntiles = (ck + 15) >> 4;
        DUFragment<accumulator, 16, 16, 16, float> acc[2];
#    pragma unroll
        for (int nt = 0; nt < 2; nt++) { du_fill_fragment(acc[nt], 0.0f); }

        // Fetch next K into registers while current K stays immutable in LDS.
        // Each complete wave participates; scalar loop bounds preserve tails.
        using KPack = unsigned int __attribute__((ext_vector_type(4)));
        KPack next_k[DK / 4];
        const int next_ck = max(0, min(DENSE_CHUNK, col_limit - chunk - DENSE_CHUNK));
#    pragma unroll
        for (int z = 0; z < DK / 4; z++) {
            const int u = threadIdx.x + z * DENSE_THREADS, n = u / (DK * 2), d8 = u % (DK * 2);
            next_k[z] = n < next_ck ? *(const KPack*)(k + ((size_t)b * Sk + chunk + DENSE_CHUNK + n) * D + d8 * 8)
                                    : KPack{0, 0, 0, 0};
        }
#    pragma unroll
        for (int kk = 0; kk < DK; kk++) {
#    pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                {  // Execute a full 32-column MMAC tile; epilogue retains the exact valid-key mask.
                    DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr;
                    du_load_matrix_sync(bfr, k_lds + (size_t)(nt * 16) * KLD + kk * 16, KLD);
                    if (FULL_HEADS || r0 < Hq) du_mma_sync(acc[nt], af[kk], bfr, acc[nt]);
                }
            }
        }
        // Current LDS readers must finish before reusing their tile. This
        // LDS-only barrier leaves independent global K reads outstanding.
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
        if constexpr (SCORE_RING) {
            // Existing barrier also completes the prior chunk's LDS atomics.
            // Current scores still reside in acc: flush and clear the retired
            // score tile before the existing post-K-publication barrier.
            if (chunk > 0 && threadIdx.x < DENSE_CHUNK) {
                const int c = threadIdx.x;
                out[(size_t)row * score_stride + chunk - DENSE_CHUNK + c] =
                    score_part[c] + score_part[DENSE_CHUNK + c] + score_part[2 * DENSE_CHUNK + c] +
                    score_part[3 * DENSE_CHUNK + c];
            }
        }
        if (next_ck > 0) {
#    pragma unroll
            for (int z = 0; z < DK / 4; z++) {
                const int u = threadIdx.x + z * DENSE_THREADS, n = u / (DK * 2), d8 = u % (DK * 2);
                *(KPack*)(k_lds + (size_t)n * KLD + d8 * 8) = next_k[z];
            }
        }
        __syncthreads();
        dm_score_chunk_epilogue<MODE, true>(acc, ntiles, ck, SCORE_RING ? 0 : chunk, lane, 1, head_a, scale, score,
                                            score_part, warp);
    }
    __syncthreads();

    if constexpr (SCORE_RING) {
        if (col_limit > 0) {
            const int last = ((col_limit - 1) / DENSE_CHUNK) * DENSE_CHUNK;
            if (threadIdx.x < (col_limit - last)) {
                const int c                                = threadIdx.x;
                out[(size_t)row * score_stride + last + c] = score_part[c] + score_part[DENSE_CHUNK + c] +
                                                             score_part[2 * DENSE_CHUNK + c] +
                                                             score_part[3 * DENSE_CHUNK + c];
            }
        }
        for (int i = threadIdx.x; i < Sk; i += DENSE_THREADS)
            if (i >= col_limit) out[(size_t)row * score_stride + i] = -INFINITY;
    } else {
        // Write out with the ratio-causal mask.
        for (int i = threadIdx.x; i < Sk; i += DENSE_THREADS) {
            float v = score[i];
            if (i >= col_limit) v = -INFINITY;
            out[(size_t)row * score_stride + i] = v;
        }
    }
    if (MODE == 0) return;

    // Block reduction for denom (MODE 1: logsumexp, MODE 2: plain sum).
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

template<int MODE, int DIM = 0>
static __global__ void __launch_bounds__(DENSE_THREADS)
    dense_score_mmac_largeD_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ w,
                                   const float* __restrict__ lse, float* __restrict__ out, float* __restrict__ denom,
                                   int B, int Sq, int Sk, int Hq, int D, int ratio, float scale, int score_stride) {
    const int row    = blockIdx.x;
    const int b      = row / Sq;
    const int sq     = row - b * Sq;
    const int lane   = threadIdx.x & 63;
    const int warp   = threadIdx.x >> 6;
    const int Hq_pad = (Hq + 15) & ~15;
    // DIM > 0 specializes the D-chunk loops at compile time so the hoisted
    // A fragments are indexed statically and stay in registers; the runtime
    // version (DIM == 0) forces dynamic fragment indexing and register
    // spills of the whole af[] array to private (scratch) memory.
    static_assert(DIM > 0, "Runtime-D spill kernels are not supported");
    constexpr int Dd = DIM;
    const int QLD    = (Dd < DENSE_DCHUNK ? Dd : DENSE_DCHUNK) + 8;
    const int KLD    = QLD;
    int col_limit    = (sq + 1) / ratio;
    if (col_limit > Sk) col_limit = Sk;

    extern __shared__ char smem_raw[];
    float* score;
    float* red;
    hbf16* q_lds;
    hbf16* k_lds;
    if constexpr (DIM > 0) {
        // Specialized path: score/red first, then one region that first
        // stages Q (rows Hq_pad x QLD) and, after the A fragments are
        // hoisted, becomes the double-buffered K staging area.  This keeps
        // the dynamic LDS footprint <= 32 KB for Hq <= 64 so 2 blocks/CU
        // are resident, and lets K loads overlap the MMAC issue stream.
        score = (float*)smem_raw;
        red   = score + Sk;
        q_lds = (hbf16*)(red + DENSE_THREADS);
        k_lds = q_lds + (size_t)DENSE_CHUNK * KLD;
    }

    for (int i = threadIdx.x; i < Sk; i += DENSE_THREADS) score[i] = 0.0f;
    if constexpr (DIM > 0) {
        const int qk_rows = (Hq_pad > 2 * DENSE_CHUNK) ? Hq_pad : 2 * DENSE_CHUNK;
        dm_zero_bf16(q_lds, qk_rows * KLD);
    }
    __syncthreads();

    // Heads run in 64-head passes (4 warps x 16 heads); the full-D A
    // fragments for each pass are hoisted into registers (re-staging the
    // D-chunked Q tile per pass — Q/K are L2-resident for the DSv4 shapes).
    // score[] accumulates across passes via atomicAdd.
    const int n_hpass = (Hq_pad + 63) / 64;
    for (int hp = 0; hp < n_hpass; hp++) {
        const int r0 = hp * 64 + warp * 16;

        const int h  = hp * 64 + warp * 16 + (lane & 15);
        float head_a = 0.0f;
        if (MODE == 2)
            head_a = (h < Hq) ? lse[(size_t)row * Hq + h] : INFINITY;
        else
            head_a = (h < Hq) ? bf16_to_float_dm(w[(size_t)row * Hq + h]) : 0.0f;

        // Hoist the full-D A fragments (this pass's 16 heads) into registers.
        DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> af[DENSE_MAX_TILES];
#    pragma unroll
        for (int d0 = 0; d0 < Dd; d0 += DENSE_DCHUNK) {
            int dchunk = Dd - d0;
            if (dchunk > DENSE_DCHUNK) dchunk = DENSE_DCHUNK;
            int qld = dchunk + 8;
            dm_stage_q_chunk(q, row, Hq, D, d0, dchunk, Hq_pad, qld, q_lds);
            __syncthreads();  // q_lds must be fully staged before the af loads
#    pragma unroll
            for (int kk = 0; kk < dchunk / 16; kk++) {
                if (r0 < Hq_pad) du_load_matrix_sync(af[(d0 + kk * 16) / 16], q_lds + (size_t)r0 * qld + kk * 16, qld);
            }
            __syncthreads();
        }

        if constexpr (DIM > 0) {
            // Re-zero the staging region (now double-buffered K) once per
            // head pass: the Q staging above overwrote it, and partial tail
            // K chunks leave rows >= ck untouched.
            const int qk_halfs = 2 * DENSE_CHUNK * KLD;
            dm_zero_bf16(q_lds, qk_halfs);
            __syncthreads();

            const int nd = (Dd + DENSE_DCHUNK - 1) / DENSE_DCHUNK;
            for (int chunk = 0; chunk < col_limit; chunk += DENSE_CHUNK) {
                int ck = col_limit - chunk;
                if (ck > DENSE_CHUNK) ck = DENSE_CHUNK;
                int ntiles     = (ck + 15) >> 4;
                const int nacc = (MODE == 2 && Dd > DENSE_DCHUNK) ? DM_MODE2_NACC : 1;
                DUFragment<accumulator, 16, 16, 16, float> acc[2 * DM_MODE2_NACC];
#    pragma unroll
                for (int i = 0; i < 2 * DM_MODE2_NACC; i++)
                    if (i < 2 * nacc) du_fill_fragment(acc[i], 0.0f);

                // Stage the first D sub-chunk, then per sub-chunk prefetch
                // the next sub-chunk into the other buffer before running
                // the MMACs: one barrier per sub-chunk and load/MMAC overlap.
                int dchunk0 = (Dd < DENSE_DCHUNK) ? Dd : DENSE_DCHUNK;
                dm_load_k_chunk(q_lds, k, b, Sk, chunk, ck, D, 0, dchunk0, KLD);
                __syncthreads();
#    pragma unroll
                for (int di = 0; di < nd; di++) {
                    const int d0     = di * DENSE_DCHUNK;
                    const int dchunk = ((Dd - d0) < DENSE_DCHUNK) ? (Dd - d0) : DENSE_DCHUNK;
                    if (di + 1 < nd) {
                        const int d1      = d0 + DENSE_DCHUNK;
                        const int dchunk1 = ((Dd - d1) < DENSE_DCHUNK) ? (Dd - d1) : DENSE_DCHUNK;
                        // Prefetch into the buffer opposite the one the MMACs
                        // read this iteration; both prefetch stores and MMAC
                        // reads of the current buffer are fenced by the
                        // end-of-iteration barrier.
                        dm_load_k_chunk(((di & 1) == 0) ? k_lds : q_lds, k, b, Sk, chunk, ck, D, d1, dchunk1, KLD);
                    }
                    const hbf16* kcur = ((di & 1) == 0) ? q_lds : k_lds;
#    pragma unroll
                    for (int kk = 0; kk < dchunk / 16; kk++) {
                        const int t = (d0 + kk * 16) / 16;
#    pragma unroll
                        for (int nt = 0; nt < 2; nt++) {
                            if (nt < ntiles) {
                                DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr;
                                du_load_matrix_sync(bfr, kcur + (size_t)(nt * 16) * KLD + kk * 16, KLD);
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
    }
    __syncthreads();

    // Write out with the ratio-causal mask.
    for (int i = threadIdx.x; i < Sk; i += DENSE_THREADS) {
        float v = score[i];
        if (i >= col_limit) v = -INFINITY;
        out[(size_t)row * score_stride + i] = v;
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

template<int MODE>
static __global__ void __launch_bounds__(DENSE_THREADS)
    dense_score_mmac_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ w,
                            const float* __restrict__ lse, float* __restrict__ out, float* __restrict__ denom, int B,
                            int Sq, int Sk, int Hq, int D, int ratio, float scale, int score_stride) {}

template<int MODE, int DIM>
static __global__ void __launch_bounds__(DENSE_THREADS)
    dense_score_mmac_largeD_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ w,
                                   const float* __restrict__ lse, float* __restrict__ out, float* __restrict__ denom,
                                   int B, int Sq, int Sk, int Hq, int D, int ratio, float scale, int score_stride) {}

template<int MODE, int DK, bool FULL_HEADS = false, bool SCORE_RING = false>
static __global__ void dense_indexer_prefetch_kernel(const bf16* q, const bf16* k, const bf16* w, const float* lse,
                                                     float* out, float* denom, int B, int Sq, int Sk, int Hq, int D,
                                                     int ratio, float scale, int score_stride) {}

template<int MODE, int DK>
static __global__ void __launch_bounds__(DENSE_THREADS)
    dense_score_mmac_kernel_v2(const bf16*, const bf16*, const bf16*, const float*, float*, float*, int, int, int, int,
                               int, int, float, int) {}

#endif  // supported gfx archs

namespace ftrain {
namespace {

template<int MODE>
static inline void launch_dense_score_mmac_dispatch(hipStream_t stream, const void* q, const void* k, const void* w,
                                                    const float* lse, float* out, float* denom, int B, int Sq, int Sk,
                                                    int Hq, int D, int ratio, float scale, int score_stride = 0);
void launch_dense_attn_score_mmac(hipStream_t stream, const void* q, const void* k, const float* lse, float* out,
                                  float* denom, int B, int Sq, int Sk, int Hq, int D, int ratio, float sm_scale,
                                  int score_stride = 0);
bool launch_dense_attn_score(hipStream_t stream, const void* q, const void* k, const float* lse, float* out,
                             float* denom, int B, int Sq, int Sk, int Hq, int Hkv, int D, int ratio,
                             float softmax_scale);

template<int MODE>
static inline void launch_dense_score_mmac_dispatch(hipStream_t stream, const void* q, const void* k, const void* w,
                                                    const float* lse, float* out, float* denom, int B, int Sq, int Sk,
                                                    int Hq, int D, int ratio, float scale, int score_stride) {
    if (score_stride <= 0) score_stride = Sk;

    // Hq > 64 (DSv4-pro: 128 heads) runs the large-D kernel, which processes
    // heads in 64-head passes; the v2/v1 kernels cover Hq <= 64 only.
    if (Hq > 64) {
        switch (D) {
            case 16:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 16>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 32:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 32>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 48:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 48>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 64:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 64>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 80:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 80>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 96:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 96>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 112:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 112>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 128:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 128>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 256:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 256>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 448:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 448>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 512:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 512>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 576:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 576>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            default:
                throw std::runtime_error("dense score: unsupported head dimension for parallel specialized kernel; "
                                     "supported D: 16..128 in steps of 16, 256, 448, 512, 576; "
                                     "runtime-D spill fallback has been removed");
        }
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_dense_attn_score HIP call failed: %s",
                                        hipGetErrorString(status));
            }
        }
        return;
    }
    // v2 wins once the A-fragment hoist is amortized over enough key chunks;
    // the small preset (Sk=128) is faster with the staged-Q v1 kernel.
    // Hq % 16 == 0 required: v2 hoists the A fragments straight from global
    // memory, and for Hq not a multiple of 16 the load would read rows past
    // the head block (out-of-bounds at the last row of q).  The staged-Q v1
    // kernel zero-pads those rows in LDS and is safe for any Hq.
    if (D <= DENSE_DCHUNK && D % 16 == 0 && Hq <= 64 && Hq % 16 == 0 && Sk >= 256 && (D == 64 || D == 128)) {
        size_t shm = dense_mmac_shm_v2(D, Sk);
        if constexpr (MODE == 0) {
            // MODE0 has no row denominator: retire each32-key score tile
            // at existing barriers and reuse128B of LDS. The large dense
            // indexer shapes gain residency without extra launches/barriers.
            if (D == 128 && Hq == 64 && Sk >= 1024) {
                shm = (size_t)DENSE_CHUNK * (D + 8) * 2 + DENSE_CHUNK * sizeof(float) +
                      4 * (size_t)DENSE_CHUNK * sizeof(float);
                hipLaunchKernelGGL((dense_indexer_prefetch_kernel<0, 8, true, true>), B * Sq, DENSE_THREADS, shm,
                                   stream, (const bf16*)q, (const bf16*)k, (const bf16*)w, lse, out, denom, B, Sq, Sk,
                                   Hq, D, ratio, scale, score_stride);
                {
                    const hipError_t status = hipGetLastError();
                    if (status != hipSuccess) {
                        throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                                "bf16_dense_attn_score HIP call failed: %s", hipGetErrorString(status));
                    }
                }
                return;
            }
#define DM_LAUNCH_PREFETCH(DK, FULL)                                                                             \
    hipLaunchKernelGGL((dense_indexer_prefetch_kernel<0, DK, FULL>), B* Sq, DENSE_THREADS, shm, stream,          \
                       (const bf16*)q, (const bf16*)k, (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, \
                       scale, score_stride)
            if (D == 128) {
                if (Hq == 64) {
                    DM_LAUNCH_PREFETCH(8, true);
                } else {
                    DM_LAUNCH_PREFETCH(8, false);
                }
            } else {
                if (Hq == 64) {
                    DM_LAUNCH_PREFETCH(4, true);
                } else {
                    DM_LAUNCH_PREFETCH(4, false);
                }
            }
#undef DM_LAUNCH_PREFETCH
            {
                const hipError_t status = hipGetLastError();
                if (status != hipSuccess) {
                    throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_dense_attn_score HIP call failed: %s",
                                            hipGetErrorString(status));
                }
            }
            return;
        }

        if (D == 128) {
            hipLaunchKernelGGL((dense_score_mmac_kernel_v2<MODE, 8>), B * Sq, DENSE_THREADS, shm, stream,
                               (const bf16*)q, (const bf16*)k, (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio,
                               scale, score_stride);
        } else {
            hipLaunchKernelGGL((dense_score_mmac_kernel_v2<MODE, 4>), B * Sq, DENSE_THREADS, shm, stream,
                               (const bf16*)q, (const bf16*)k, (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio,
                               scale, score_stride);
        }
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_dense_attn_score HIP call failed: %s",
                                        hipGetErrorString(status));
            }
        }
        return;
    }
    if (D > DENSE_DCHUNK) {
        switch (D) {
            case 256:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 256>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 448:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 448>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 512:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 512>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            case 576:
                hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, 576>), B * Sq, DENSE_THREADS,
                                   dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k,
                                   (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
                break;
            default:
                throw std::runtime_error("dense score: unsupported head dimension for parallel specialized kernel; "
                                     "supported D: 16..128 in steps of 16, 256, 448, 512, 576; "
                                     "runtime-D spill fallback has been removed");
        }
    } else {
        hipLaunchKernelGGL((dense_score_mmac_kernel<MODE>), B * Sq, DENSE_THREADS, dense_mmac_shm(Hq, D, Sk, false),
                           stream, (const bf16*)q, (const bf16*)k, (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D,
                           ratio, scale, score_stride);
    }
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_dense_attn_score HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
}

static void launch_dense_attn_score_mmac(hipStream_t stream, const void* q, const void* k, const float* lse, float* out,
                                         float* denom, int B, int Sq, int Sk, int Hq, int D, int ratio, float sm_scale,
                                         int score_stride) {
    launch_dense_score_mmac_dispatch<2>(stream, q, k, nullptr, lse, out, denom, B, Sq, Sk, Hq, D, ratio, sm_scale,
                                        score_stride);
}

static bool launch_dense_attn_score(hipStream_t stream, const void* q, const void* k, const float* lse, float* out,
                                    float* denom, int B, int Sq, int Sk, int Hq, int Hkv, int D, int ratio,
                                    float softmax_scale) {
    // Fused single-kernel path (MMAC tensor core) only; no non-fused
    // fallback and no hipblas GEMM anywhere in this operator.
    if (!(Hkv == 1 && D % 16 == 0 && D <= 576 && Hq <= 128 && dense_mmac_supported() &&
          dense_mmac_shm_fits(Hq, D, Sk, false))) {
        return false;
    }
    launch_dense_attn_score_mmac(stream, q, k, lse, out, denom, B, Sq, Sk, Hq, D, ratio, softmax_scale);
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_dense_attn_score HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}
}  // namespace

bool bf16DenseAttnScoreLaunchFits(int heads, int width, int keys) {
    return dense_mmac_storage_fits(heads, width, keys, false);
}

bool bf16DenseAttnScoreDenseMmacSupported() { return dense_mmac_supported(); }

bool launchBf16DenseAttnScore(FTrainStream stream, const void* q, const void* k, const float* lse, float* out,
                              float* denom, int batch, int query_length, int key_length, int query_heads, int kv_heads,
                              int head_dim, int ratio, float softmax_scale) {
    return launch_dense_attn_score(stream, q, k, lse, out, denom, batch, query_length, key_length, query_heads,
                                   kv_heads, head_dim, ratio, softmax_scale);
}

}  // namespace ftrain
