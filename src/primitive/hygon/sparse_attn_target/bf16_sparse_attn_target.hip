// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <math.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

// DTK du_mma.h defines a non-inline host function; keep vendor fragments local.
namespace {
#include <du_mma.h>
}

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#    include <du_mma.h>
#endif

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_attn_target/bf16_sparse_attn_target.hpp"

static inline int dsa_shared_memory_limit() {
    int device = 0, limit = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&limit, hipDeviceAttributeMaxSharedMemoryPerBlock, device) != hipSuccess)
        return 0;
    return limit;
}

#define SPARSE_NWARPS 4
#define SPARSE_CHUNK 16
#define SPARSE_CSLICE 256

static inline bool sparse_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx928") != nullptr || strstr(prop.gcnArchName, "gfx936") != nullptr ||
           strstr(prop.gcnArchName, "gfx938") != nullptr;
}

static inline int sparse_mmac_max_lds() {
    const int max_lds = dsa_shared_memory_limit();
    return max_lds;
}

static inline bool sparse_mmac_shm_fits(int Hq, int D, int topk) {
    const int max_lds = sparse_mmac_max_lds();
    const int Hq_pad  = (Hq + 15) & ~15;
    size_t shm_lds    = (size_t)Hq_pad * (D + 8) * 2 + SPARSE_CHUNK * (D + 8) * 2 + (size_t)SPARSE_CSLICE * 8;
    if (Hq <= 64 && shm_lds + 2048 <= (size_t)max_lds) return true;
    const int dk = D / 16;
    size_t shm_g = (size_t)SPARSE_CHUNK * (D + 8) * 2 + (size_t)SPARSE_CSLICE * 8;
    return Hq > 0 && Hq <= 128 && shm_g + 2048 <= (size_t)max_lds &&
           (dk == 4 || dk == 8 || dk == 16 || dk == 28 || dk == 32 || dk == 36);
}

namespace hip_dsa_mla64 {
using bf = __hip_bfloat16;
#if defined(__gfx936__) || defined(__gfx938__)
using namespace du::dumma;
using AF = DUFragment<matrix_a, 16, 16, 16, bf, row_major>;
using BC = DUFragment<matrix_b, 16, 16, 16, bf, col_major>;

using CF   = DUFragment<accumulator, 16, 16, 16, float>;
using Desc = int __attribute__((ext_vector_type(4)));

__device__ __forceinline__ static void bar() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
__device__ __forceinline__ static void waitn(int n) {
    if (n == 3) asm volatile("s_waitcnt vmcnt(3)" ::: "memory");
    if (n == 2) asm volatile("s_waitcnt vmcnt(2)" ::: "memory");
    if (n == 1) asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
    if (n == 0) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    bar();
}
__device__ __forceinline__ static void ld(Desc desc, int off, bf* dst) {
    asm volatile(
        "s_mov_b32 m0, %1\n\ts_nop 0\n\tbuffer_load_dwordx4 %0, %2, 0 " "offen lds" ::"v"(off),
        "s"((unsigned)(size_t)dst), "s"(desc)
        : "m0", "memory");
}
#endif
}  // namespace hip_dsa_mla64

// Adapted from fused_dsa/hip_dsa/kernels/sparse_attn_target.hip.

// Adapted from fused_dsa/hip_dsa/kernels/sparse_mmac.hip.

typedef uint16_t bf16;

#define SPARSE_NWARPS 4
#define SPARSE_CHUNK 16

#define SPARSE_CSLICE 256

__device__ static inline float bf16_to_float_sm(uint16_t v) {
    union {
        float f;
        uint32_t u;
    } conv;
    conv.u = (uint32_t)v << 16;
    return conv.f;
}

template<int MODE>
static __global__ void __launch_bounds__(256)
    sparse_gather_mmac_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ w,
                              const float* __restrict__ lse, const int* __restrict__ idxs, float* __restrict__ raw,
                              int* __restrict__ row_count, int n_slices, int B, int Sq, int Sk, int Hq, int D, int topk,
                              float scale, bool topk_indices_global);

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;
typedef __hip_bfloat16 hbf16;

template<int MODE>
static __global__ void __launch_bounds__(256)
    sparse_gather_mmac_kernel(const bf16* __restrict__ q,                 // (B, Sq, Hq, D)
                              const bf16* __restrict__ k,                 // (B, Sk, D)
                              const bf16* __restrict__ w,                 // (B, Sq, Hq)  [MODE 0]
                              const float* __restrict__ lse,              // (B, Sq, Hq)  [MODE 1]
                              const int* __restrict__ idxs,               // (B*Sq, topk)
                              float* __restrict__ raw,                    // (B*Sq, topk)
                              int* __restrict__ row_count, int n_slices,  // fused-norm last-block sync
                              int B, int Sq, int Sk, int Hq, int D, int topk, float scale, bool topk_indices_global) {
    const int row    = blockIdx.x;
    const int slice  = blockIdx.y;
    const int b      = row / Sq;
    const int lane   = threadIdx.x & 63;
    const int warp   = threadIdx.x >> 6;
    const int Hq_pad = (Hq + 15) & ~15;
    const int t0     = slice * SPARSE_CSLICE;  // first topk column of this block
    const int ck_all = min(SPARSE_CSLICE, topk - t0);
    // Padded LDS row strides: unpadded D bf16 rows put every fragment row on
    // the same LDS bank (16-way conflicts).  D+8 keeps rows 16B-aligned and
    // spreads rows across banks (2-way worst case).
    const int QLD = D + 8;
    const int KLD = D + 8;

    extern __shared__ char smem_raw[];
    hbf16* q_lds = (hbf16*)smem_raw;
    hbf16* k_lds = q_lds + Hq_pad * QLD;
    int* idx_lds = (int*)(k_lds + SPARSE_CHUNK * KLD);
    float* score = (float*)(idx_lds + SPARSE_CSLICE);

    // Stage Q into LDS (zero-padded rows beyond Hq).
    const hbf16* qrow = (const hbf16*)(q + (size_t)row * Hq * D);
    for (int i = threadIdx.x; i < Hq_pad * QLD; i += 256) q_lds[i] = hbf16(0);
    __syncthreads();
    for (int h = threadIdx.x; h < Hq; h += 256)
        for (int d = 0; d < D; d += 8) {
            float2* dst       = (float2*)(q_lds + (size_t)h * QLD + d);
            const float2* src = (const float2*)(qrow + (size_t)h * D + d);
            dst[0]            = src[0];
            dst[1]            = src[1];
        }
    // Stage this slice's topk indices in LDS so the gather address chain is
    // LDS-fast.  Each block owns a disjoint column range, so the epilogue
    // needs no cross-block atomics and the grid is n_rows * ceil(topk/slice).
    for (int t = threadIdx.x; t < ck_all; t += 256) idx_lds[t] = idxs[(size_t)row * topk + t0 + t];
    for (int t = threadIdx.x; t < ck_all; t += 256) score[t] = 0.0f;
    __syncthreads();

    // Per-lane head value for the epilogue (rows owned by this warp).
    const int h  = warp * 16 + (lane & 15);
    float head_a = 0.0f;
    if (MODE == 1) {
        head_a = (h < Hq) ? lse[(size_t)row * Hq + h] : INFINITY;
    } else {
        head_a = (h < Hq) ? bf16_to_float_sm(w[(size_t)row * Hq + h]) : 0.0f;
    }
    const int r0 = warp * 16;

    for (int chunk = 0; chunk < ck_all; chunk += SPARSE_CHUNK) {
        int ck = ck_all - chunk;
        if (ck > SPARSE_CHUNK) ck = SPARSE_CHUNK;

        // Gather K chunk into LDS: k_lds[n * KLD + d] = K[b][idx[chunk+n]][d]
        if (ck < SPARSE_CHUNK) {
            for (int i = threadIdx.x; i < (SPARSE_CHUNK - ck) * KLD; i += 256) k_lds[(size_t)ck * KLD + i] = hbf16(0);
        }
        for (int u = threadIdx.x; u < ck * (D / 8); u += 256) {
            int n   = u / (D / 8);
            int d8  = u % (D / 8);
            int idx = idx_lds[chunk + n];
            if (idx < 0 || idx >= (topk_indices_global ? B * Sk : Sk)) idx = 0;
            const bf16* src   = k + ((size_t)(topk_indices_global ? 0 : b) * Sk + idx) * D + d8 * 8;
            bf16* dst         = (bf16*)(k_lds + (size_t)n * KLD + d8 * 8);
            ((float2*)dst)[0] = ((const float2*)src)[0];
            ((float2*)dst)[1] = ((const float2*)src)[1];
        }
        __syncthreads();

        int ntiles = (ck + 15) >> 4;
        // Keep the tile loop bounds compile-time constant (2 tiles max) so
        // acc[] stays in registers; the tail chunk is masked by the epilogue
        // bounds checks (k_lds is zero-padded, so extra MMACs are harmless).
        DUFragment<accumulator, 16, 16, 16, float> acc[2];
#    pragma unroll
        for (int nt = 0; nt < 2; nt++)
            if (nt < ntiles) du_fill_fragment(acc[nt], 0.0f);

        // K-loop over the D dimension (16 per MMAC step).
        for (int kk = 0; kk < D / 16; kk++) {
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> a;
            if (r0 < Hq_pad) { du_load_matrix_sync(a, q_lds + (size_t)r0 * QLD + kk * 16, QLD); }
#    pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                if (nt < ntiles) {
                    // k_lds is (key, d) row-major, so load B (K x N) col_major:
                    // element (k, n) = p[n*ldm + k] = K[nt*16+n][kk*16+k].
                    DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr;
                    du_load_matrix_sync(bfr, k_lds + (size_t)(nt * 16) * KLD + kk * 16, KLD);
                    if (r0 < Hq_pad) { du_mma_sync(acc[nt], a, bfr, acc[nt]); }
                }
            }
        }

// Epilogue: per-element f(), reduce over the 16 rows, add to score[].
#    pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            if (nt >= ntiles) continue;
            float v0, v1, v2, v3;
            if (MODE == 1) {
                v0 = expf(acc[nt].x[0] * scale - head_a);
                v1 = expf(acc[nt].x[1] * scale - head_a);
                v2 = expf(acc[nt].x[2] * scale - head_a);
                v3 = expf(acc[nt].x[3] * scale - head_a);
            } else {
                float d0 = acc[nt].x[0] * scale;
                float d1 = acc[nt].x[1] * scale;
                float d2 = acc[nt].x[2] * scale;
                float d3 = acc[nt].x[3] * scale;
                v0       = (d0 > 0.0f ? d0 : 0.0f) * head_a;
                v1       = (d1 > 0.0f ? d1 : 0.0f) * head_a;
                v2       = (d2 > 0.0f ? d2 : 0.0f) * head_a;
                v3       = (d3 > 0.0f ? d3 : 0.0f) * head_a;
            }
#    pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                v0 += __shfl_xor(v0, off);
                v1 += __shfl_xor(v1, off);
                v2 += __shfl_xor(v2, off);
                v3 += __shfl_xor(v3, off);
            }
            if ((lane & 15) == 0) {
                // Four warps accumulate into the same score[] cells, so the
                // LDS updates must be atomic (RMW without sync loses values).
                int g   = lane >> 4;
                int col = nt * 16 + g;
                if (col < ck) atomicAdd(&score[chunk + col], v0);
                if (col + 4 < ck) atomicAdd(&score[chunk + col + 4], v1);
                if (col + 8 < ck) atomicAdd(&score[chunk + col + 8], v2);
                if (col + 12 < ck) atomicAdd(&score[chunk + col + 12], v3);
            }
        }
        __syncthreads();
    }

    // Mask invalid keys and write raw (this block's slice).
    for (int t = threadIdx.x; t < ck_all; t += 256) {
        int idx = idx_lds[t];
        float v = score[t];
        if (idx < 0 || idx >= (topk_indices_global ? B * Sk : Sk)) v = (MODE == 0) ? -INFINITY : 0.0f;
        raw[(size_t)row * topk + t0 + t] = v;
    }

    // ---- Fused normalization epilogue (single launch) ---------------------
    // Softmax (MODE 0) / L1 (MODE 1) normalization needs the full row, but
    // the topk range is split across n_slices blocks (blockIdx.y).  The
    // block that observes the last atomic increment for its row normalizes
    // the whole row in place; all other blocks exit.  row_count must be
    // zeroed by the launcher before the launch.  This removes the second
    // softmax/L1 kernel launch (and its scheduling overhead) while keeping
    // the column-slice parallelism.
    __threadfence();
    __syncthreads();
    __shared__ int last_block;
    if (threadIdx.x == 0) { last_block = (atomicAdd(&row_count[row], 1) == n_slices - 1); }
    __syncthreads();
    if (last_block) {
        float* rrow = raw + (size_t)row * topk;
        __shared__ float red[256];
        if (MODE == 1) {
            float sum = 0.0f;
            for (int t = threadIdx.x; t < topk; t += 256) sum += rrow[t];
            red[threadIdx.x] = sum;
            __syncthreads();
            for (int st = 1; st < 256; st <<= 1) {
                int idx = threadIdx.x * st * 2;
                if (idx + st < 256) red[idx] += red[idx + st];
                __syncthreads();
            }
            sum = red[0];
            {
                // Match the reference's denominator.clamp(min=1e-12), including tiny rows.
                const float inv = 1.0f / (isnan(sum) ? sum : fmaxf(sum, 1e-12f));
                for (int t = threadIdx.x; t < topk; t += 256) rrow[t] *= inv;
            }
        } else {
            float maxv = -INFINITY;
            for (int t = threadIdx.x; t < topk; t += 256) {
                const float v = rrow[t];
                if (v > maxv) maxv = v;
            }
            red[threadIdx.x] = maxv;
            __syncthreads();
            for (int st = 1; st < 256; st <<= 1) {
                int idx = threadIdx.x * st * 2;
                if (idx + st < 256 && red[idx + st] > red[idx]) red[idx] = red[idx + st];
                __syncthreads();
            }
            maxv = red[0];
            // Every wave must consume the maximum before the buffer is reused for the sum.
            __syncthreads();
            float sum = 0.0f;
            for (int t = threadIdx.x; t < topk; t += 256) {
                const float e  = (maxv == -INFINITY ? 0.0f : expf(rrow[t] - maxv));
                rrow[t]        = e;
                sum           += e;
            }
            red[threadIdx.x] = sum;
            __syncthreads();
            for (int st = 1; st < 256; st <<= 1) {
                int idx = threadIdx.x * st * 2;
                if (idx + st < 256) red[idx] += red[idx + st];
                __syncthreads();
            }
            sum                 = red[0] + 1e-30f;
            const float inv_sum = 1.0f / sum;
            for (int t = threadIdx.x; t < topk; t += 256) rrow[t] *= inv_sum;
        }
    }
}

template<int MODE, int DK>
static __global__ void __launch_bounds__(256)
    sparse_gather_mmac_kernel_g(const bf16* __restrict__ q,                 // (B, Sq, Hq, D)
                                const bf16* __restrict__ k,                 // (B, Sk, D)
                                const bf16* __restrict__ w,                 // (B, Sq, Hq)  [MODE 0]
                                const float* __restrict__ lse,              // (B, Sq, Hq)  [MODE 1]
                                const int* __restrict__ idxs,               // (B*Sq, topk)
                                float* __restrict__ raw,                    // (B*Sq, topk)
                                int* __restrict__ row_count, int n_slices,  // fused-norm last-block sync
                                int B, int Sq, int Sk, int Hq, int D, int topk, float scale, bool topk_indices_global) {
    const int row    = blockIdx.x;
    const int slice  = blockIdx.y;
    const int b      = row / Sq;
    const int lane   = threadIdx.x & 63;
    const int warp   = threadIdx.x >> 6;
    const int Hq_pad = (Hq + 15) & ~15;
    const int t0     = slice * SPARSE_CSLICE;  // first topk column of this block
    const int ck_all = min(SPARSE_CSLICE, topk - t0);
    const int KLD    = D + 8;

    extern __shared__ char smem_raw[];
    hbf16* k_lds = (hbf16*)smem_raw;
    int* idx_lds = (int*)(k_lds + SPARSE_CHUNK * KLD);
    float* score = (float*)(idx_lds + SPARSE_CSLICE);

    // Stage this slice's topk indices in LDS so the gather address chain is
    // LDS-fast.  Each block owns a disjoint column range, so the epilogue
    // needs no cross-block atomics and the grid is n_rows * ceil(topk/slice).
    for (int t = threadIdx.x; t < ck_all; t += 256) idx_lds[t] = idxs[(size_t)row * topk + t0 + t];
    for (int t = threadIdx.x; t < ck_all; t += 256) score[t] = 0.0f;
    // Count work from actual indices; holes are allowed and masked again
    // in the epilogue. The fixed output slice is still initialized in full.
    __shared__ int valid_end[4];
    int last_valid = 0;
    for (int i = threadIdx.x; i < ck_all; i += 256)
        if (idx_lds[i] >= 0 && idx_lds[i] < (topk_indices_global ? B * Sk : Sk)) last_valid = i + 1;
#    pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) last_valid = max(last_valid, __shfl_xor(last_valid, offset));
    if (lane == 0) valid_end[warp] = last_valid;
    __syncthreads();
    const int active_end = max(max(valid_end[0], valid_end[1]), max(valid_end[2], valid_end[3]));

    const int n_hpass = (Hq_pad + 63) / 64;  // 4 warps x 16 heads per pass
    // gfx938 can route an aligned 16-byte global load straight into LDS.
    // One wave owns one gathered K row at a time, so the LDS destination is
    // wave-uniform while lanes cover the row's 16-byte chunks.  This removes
    // the VGPR payload plus the separate ds_write from the hot gather loop.
    // D=576 has 72 chunks and retains the cooperative vector-copy fallback.
    typedef int v4i __attribute__((ext_vector_type(4)));
    const bf16* k_batch   = k + (size_t)(topk_indices_global ? 0 : b) * Sk * D;
    const uint64_t k_addr = (uint64_t)k_batch;
    v4i k_desc;
    k_desc[0] = (int)(uint32_t)(k_addr & 0xffffffffu);
    k_desc[1] = (int)((uint32_t)(k_addr >> 32) | 0x40000000u);
    k_desc[2] = (int)0x80000000u;
    k_desc[3] = (int)0x00020000u;
    // Per-64-head-pass A-fragment state.  When Hq > 64 the two head passes
    // share each gathered K chunk below, so a K chunk is staged into LDS
    // from global once per chunk instead of once per head pass.  Runtime
    // guards keep the Hq <= 64 (single-pass) behavior identical.
    constexpr int SGM_MAXP = 2;
    DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> af[SGM_MAXP][DK];
    float head_a[SGM_MAXP];
    bool r0ok[SGM_MAXP];
#    pragma unroll
    for (int p = 0; p < SGM_MAXP; p++) {
        const int h0p = p * 64;
        const int r0p = h0p + warp * 16;
        const int hp_ = h0p + warp * 16 + (lane & 15);
        r0ok[p]       = (r0p < Hq);
        if (MODE == 1) {
            head_a[p] = (hp_ < Hq) ? lse[(size_t)row * Hq + hp_] : INFINITY;
        } else {
            head_a[p] = (hp_ < Hq) ? bf16_to_float_sm(w[(size_t)row * Hq + hp_]) : 0.0f;
        }
        // Hoist the full-D A fragments (this warp's 16 heads) from global.
        // Rows past Hq (partial head block) are clamped to row Hq-1 and
        // zeroed per-lane: a plain du_load_matrix_sync of a 16-row fragment
        // would read past the head block (out-of-bounds at the last query
        // row) and stale/NaN data would poison the MODE-1 exp() epilogue.
        if (p < n_hpass && r0ok[p]) {
            const int qrow_h  = (hp_ < Hq) ? hp_ : (Hq - 1);
            const hbf16* qrow = (const hbf16*)(q + (size_t)row * Hq * D + (size_t)qrow_h * D);
#    pragma unroll
            for (int kk = 0; kk < DK; kk++) {
                *(uint64_t*)&af[p][kk].x[0] = *(const uint64_t*)(qrow + kk * 16 + ((lane >> 4) << 2));
                if (hp_ >= Hq) af[p][kk].x[0] = af[p][kk].x[1] = af[p][kk].x[2] = af[p][kk].x[3] = hbf16(0);
            }
        }
    }

    for (int chunk = 0; chunk < active_end; chunk += SPARSE_CHUNK) {
        int ck = ck_all - chunk;
        if (ck > SPARSE_CHUNK) ck = SPARSE_CHUNK;

        // Gather K chunk into LDS: k_lds[n * KLD + d] = K[b][idx[chunk+n]][d].
        // Only the tail chunk needs zero padding (rows >= ck are never
        // consumed by the masked epilogue).
        if (ck < SPARSE_CHUNK) {
            for (int i = threadIdx.x; i < (SPARSE_CHUNK - ck) * KLD; i += 256) k_lds[(size_t)ck * KLD + i] = hbf16(0);
        }
        constexpr int K_CHUNKS = DK * 2;
        if constexpr (K_CHUNKS <= 64) {
            for (int n = warp; n < ck; n += SPARSE_NWARPS) {
                int idx = idx_lds[chunk + n];
                if (idx < 0 || idx >= (topk_indices_global ? B * Sk : Sk)) idx = 0;
                if (lane < K_CHUNKS) {
                    __builtin_amdgcn_raw_buffer_load_lds(
                        k_desc, (__attribute__((address_space(3))) int*)(k_lds + (size_t)n * KLD), 16,
                        idx * (D * 2) + lane * 16, 0, 0, 0);
                }
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        } else {
            for (int u = threadIdx.x; u < ck * K_CHUNKS; u += 256) {
                int n   = u / K_CHUNKS;
                int d8  = u % K_CHUNKS;
                int idx = idx_lds[chunk + n];
                if (idx < 0 || idx >= (topk_indices_global ? B * Sk : Sk)) idx = 0;
                const bf16* src   = k_batch + (size_t)idx * D + d8 * 8;
                bf16* dst         = (bf16*)(k_lds + (size_t)n * KLD + d8 * 8);
                ((float2*)dst)[0] = ((const float2*)src)[0];
                ((float2*)dst)[1] = ((const float2*)src)[1];
            }
        }
        __syncthreads();

        int ntiles = (ck + 15) >> 4;
#    pragma unroll
        for (int p = 0; p < SGM_MAXP; p++) {
            if (p >= n_hpass) continue;
            DUFragment<accumulator, 16, 16, 16, float> acc[2];
#    pragma unroll
            for (int nt = 0; nt < 2; nt++)
                if (nt < ntiles) du_fill_fragment(acc[nt], 0.0f);

// K-loop over the D dimension (16 per MMAC step), fully unrolled
// so the hoisted A fragments stay in registers.
#    pragma unroll
            for (int kk = 0; kk < DK; kk++) {
#    pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    if (nt < ntiles) {
                        DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr;
                        du_load_matrix_sync(bfr, k_lds + (size_t)(nt * 16) * KLD + kk * 16, KLD);
                        if (r0ok[p]) { du_mma_sync(acc[nt], af[p][kk], bfr, acc[nt]); }
                    }
                }
            }

// Epilogue: per-element f(), reduce over the 16 rows, add to score[].
#    pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                if (nt >= ntiles) continue;
                float v0, v1, v2, v3;
                if (MODE == 1) {
                    v0 = expf(acc[nt].x[0] * scale - head_a[p]);
                    v1 = expf(acc[nt].x[1] * scale - head_a[p]);
                    v2 = expf(acc[nt].x[2] * scale - head_a[p]);
                    v3 = expf(acc[nt].x[3] * scale - head_a[p]);
                } else {
                    float d0 = acc[nt].x[0] * scale;
                    float d1 = acc[nt].x[1] * scale;
                    float d2 = acc[nt].x[2] * scale;
                    float d3 = acc[nt].x[3] * scale;
                    v0       = (d0 > 0.0f ? d0 : 0.0f) * head_a[p];
                    v1       = (d1 > 0.0f ? d1 : 0.0f) * head_a[p];
                    v2       = (d2 > 0.0f ? d2 : 0.0f) * head_a[p];
                    v3       = (d3 > 0.0f ? d3 : 0.0f) * head_a[p];
                }
#    pragma unroll
                for (int off = 8; off > 0; off >>= 1) {
                    v0 += __shfl_xor(v0, off);
                    v1 += __shfl_xor(v1, off);
                    v2 += __shfl_xor(v2, off);
                    v3 += __shfl_xor(v3, off);
                }
                if ((lane & 15) == 0) {
                    int g   = lane >> 4;
                    int col = nt * 16 + g;
                    if (col < ck) atomicAdd(&score[chunk + col], v0);
                    if (col + 4 < ck) atomicAdd(&score[chunk + col + 4], v1);
                    if (col + 8 < ck) atomicAdd(&score[chunk + col + 8], v2);
                    if (col + 12 < ck) atomicAdd(&score[chunk + col + 12], v3);
                }
            }
        }
        __syncthreads();
    }

    // Mask invalid keys and write raw (this block's slice).
    for (int t = threadIdx.x; t < ck_all; t += 256) {
        int idx = idx_lds[t];
        float v = score[t];
        if (idx < 0 || idx >= (topk_indices_global ? B * Sk : Sk)) v = (MODE == 0) ? -INFINITY : 0.0f;
        raw[(size_t)row * topk + t0 + t] = v;
    }

    // ---- Fused normalization epilogue (single launch) ---------------------
    // Identical to sparse_gather_mmac_kernel: the last block for the row
    // normalizes the whole row in place (see the LDS-Q kernel for details).
    __threadfence();
    __syncthreads();
    __shared__ int last_block;
    if (threadIdx.x == 0) { last_block = (atomicAdd(&row_count[row], 1) == n_slices - 1); }
    __syncthreads();
    if (last_block) {
        float* rrow = raw + (size_t)row * topk;
        __shared__ float red[256];
        if (MODE == 1) {
            float sum = 0.0f;
            for (int t = threadIdx.x; t < topk; t += 256) sum += rrow[t];
            red[threadIdx.x] = sum;
            __syncthreads();
            for (int st = 1; st < 256; st <<= 1) {
                int idx = threadIdx.x * st * 2;
                if (idx + st < 256) red[idx] += red[idx + st];
                __syncthreads();
            }
            sum = red[0];
            {
                // Match the reference's denominator.clamp(min=1e-12), including tiny rows.
                const float inv = 1.0f / (isnan(sum) ? sum : fmaxf(sum, 1e-12f));
                for (int t = threadIdx.x; t < topk; t += 256) rrow[t] *= inv;
            }
        } else {
            float maxv = -INFINITY;
            for (int t = threadIdx.x; t < topk; t += 256) {
                const float v = rrow[t];
                if (v > maxv) maxv = v;
            }
            red[threadIdx.x] = maxv;
            __syncthreads();
            for (int st = 1; st < 256; st <<= 1) {
                int idx = threadIdx.x * st * 2;
                if (idx + st < 256 && red[idx + st] > red[idx]) red[idx] = red[idx + st];
                __syncthreads();
            }
            maxv = red[0];
            // Every wave must consume the maximum before the buffer is reused for the sum.
            __syncthreads();
            float sum = 0.0f;
            for (int t = threadIdx.x; t < topk; t += 256) {
                const float e  = (maxv == -INFINITY ? 0.0f : expf(rrow[t] - maxv));
                rrow[t]        = e;
                sum           += e;
            }
            red[threadIdx.x] = sum;
            __syncthreads();
            for (int st = 1; st < 256; st <<= 1) {
                int idx = threadIdx.x * st * 2;
                if (idx + st < 256) red[idx] += red[idx + st];
                __syncthreads();
            }
            sum                 = red[0] + 1e-30f;
            const float inv_sum = 1.0f / sum;
            for (int t = threadIdx.x; t < topk; t += 256) rrow[t] *= inv_sum;
        }
    }
}

#else  // gfx906 / gfx926: no MMAC intrinsics — never launched at runtime.

template<int MODE>
static __global__ void __launch_bounds__(256)
    sparse_gather_mmac_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ w,
                              const float* __restrict__ lse, const int* __restrict__ idxs, float* __restrict__ raw,
                              int* __restrict__ row_count, int n_slices, int B, int Sq, int Sk, int Hq, int D, int topk,
                              float scale, bool topk_indices_global) {}

template<int MODE, int DK>
static __global__ void __launch_bounds__(256)
    sparse_gather_mmac_kernel_g(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ w,
                                const float* __restrict__ lse, const int* __restrict__ idxs, float* __restrict__ raw,
                                int* __restrict__ row_count, int n_slices, int B, int Sq, int Sk, int Hq, int D,
                                int topk, float scale, bool topk_indices_global) {}

#endif  // supported gfx archs

template<int MODE>
static bool launch_sparse_gather_mmac(hipStream_t stream, const void* q, const void* k, const void* w, const float* lse,
                                      const int* topk_idxs, float* raw, int* row_count, int B, int Sq, int Sk, int Hq,
                                      int D, int topk, float scale, bool topk_indices_global);
static bool launch_sparse_attn_target_mmac(hipStream_t stream, const void* q, const void* k, const float* lse,
                                           const int* topk_idxs, float* raw, int* row_count, int B, int Sq, int Sk,
                                           int Hq, int D, int topk, float softmax_scale, bool topk_indices_global);
static bool launch_sparse_attn_target(hipStream_t stream, const void* q, const void* k, const float* lse,
                                      const int* topk_idxs, float* target, int* row_count, int B, int Sq, int Sk,
                                      int Hq, int D, int topk, float softmax_scale, bool topk_indices_global);

template<int MODE>
static bool launch_sparse_gather_mmac(hipStream_t stream, const void* q, const void* k, const void* w, const float* lse,
                                      const int* topk_idxs, float* raw, int* row_count, int B, int Sq, int Sk, int Hq,
                                      int D, int topk, float scale, bool topk_indices_global) {
    int n_rows   = B * Sq;
    int n_slices = (topk + SPARSE_CSLICE - 1) / SPARSE_CSLICE;

    int Hq_pad        = (Hq + 15) & ~15;
    const int max_lds = sparse_mmac_max_lds();

    // Prefer the register-Q kernel (sparse_gather_mmac_kernel_g) whenever it
    // fits the LDS budget: it avoids the per-chunk Q fragment loads and the
    // Q LDS tile, so LDS pressure is far lower (higher occupancy).  The
    // LDS-Q kernel remains the fallback for shapes the _g kernel cannot
    // express (DK not in the supported set, or LDS overflow).
    const int dk = D / 16;
    size_t shm_g = (size_t)SPARSE_CHUNK * (D + 8) * 2 + (size_t)SPARSE_CSLICE * 8;
    if (Hq <= 128 && shm_g + 2048 <= (size_t)max_lds &&
        (dk == 4 || dk == 8 || dk == 16 || dk == 28 || dk == 32 || dk == 36)) {
#define SPARSE_G_CASE(ND)                                                                                            \
    case ND:                                                                                                         \
        hipLaunchKernelGGL((sparse_gather_mmac_kernel_g<MODE, ND>), dim3(n_rows, n_slices), 256, shm_g, stream,      \
                           (const bf16*)q, (const bf16*)k, (const bf16*)w, lse, topk_idxs, raw, row_count, n_slices, \
                           B, Sq, Sk, Hq, D, topk, scale, topk_indices_global);                                      \
        break
        switch (dk) {
            SPARSE_G_CASE(4);
            SPARSE_G_CASE(8);
            SPARSE_G_CASE(16);
            SPARSE_G_CASE(28);
            SPARSE_G_CASE(32);
            SPARSE_G_CASE(36);
            default: return false;
        }
#undef SPARSE_G_CASE
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_target HIP call failed: %s",
                                        hipGetErrorString(status));
            }
        }
        return true;
    }
    size_t shm_lds = (size_t)Hq_pad * (D + 8) * 2 + SPARSE_CHUNK * (D + 8) * 2 + (size_t)SPARSE_CSLICE * 8;
    if (Hq <= 64 && shm_lds + 2048 <= (size_t)max_lds) {
        hipLaunchKernelGGL((sparse_gather_mmac_kernel<MODE>), dim3(n_rows, n_slices), 256, shm_lds, stream,
                           (const bf16*)q, (const bf16*)k, (const bf16*)w, lse, topk_idxs, raw, row_count, n_slices, B,
                           Sq, Sk, Hq, D, topk, scale, topk_indices_global);
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_target HIP call failed: %s",
                                        hipGetErrorString(status));
            }
        }
        return true;
    }
    return false;
}

static bool launch_sparse_attn_target_mmac(hipStream_t stream, const void* q, const void* k, const float* lse,
                                           const int* topk_idxs, float* raw, int* row_count, int B, int Sq, int Sk,
                                           int Hq, int D, int topk, float softmax_scale, bool topk_indices_global) {
    // The binding owns a fresh, zeroed counter for this in-flight call.
    if (!row_count || B <= 0 || Sq <= 0 || Sk <= 0 || Hq <= 0 || Hq > 128) return false;
    return launch_sparse_gather_mmac<1>(stream, q, k, nullptr, lse, topk_idxs, raw, row_count, B, Sq, Sk, Hq, D, topk,
                                        softmax_scale, topk_indices_global);
}

// sparse_attn_target: BSHD and packed THD entry points (where supported).

// Recompute normalized teacher scores without retaining a 512-channel output.
// Each CTA owns one query, shares each gathered 64-key tile across all heads,
// then normalizes its row. The zeroed per-call counter also guards head padding.
namespace hip_dsa_target64 {
using namespace hip_dsa_mla64;
#if defined(__gfx936__) || defined(__gfx938__)

using saf_f2 = float __attribute__((ext_vector_type(2)));
__device__ __forceinline__ static saf_f2 saf_pk_mul(saf_f2 a, saf_f2 b) {
    saf_f2 c;
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
    return c;
}
__device__ __forceinline__ static saf_f2 saf_pk_add(saf_f2 a, saf_f2 b) {
    saf_f2 c;
    asm volatile("v_pk_add_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
    return c;
}
__device__ __forceinline__ static saf_f2 saf_pk_fma(saf_f2 a, saf_f2 b, saf_f2 c) {
    saf_f2 d;
    asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}
__device__ __forceinline__ static saf_f2 saf_exp_pair(saf_f2 x) {
#    pragma clang fp contract(off)
    const float hi = __uint_as_float(0x3fb8aa3bu), lo = __uint_as_float(0x32a5705fu);
    const saf_f2 y = saf_pk_mul(x, (saf_f2){hi, hi});
    const saf_f2 n = {__builtin_rintf(y[0]), __builtin_rintf(y[1])};
    // First compensation retains scalar negate operands; packed negation would
    // cost two extra instructions. Pair the independent second FMA and finaladd.
    saf_f2 r       = {__builtin_fmaf(x[0], hi, -y[0]), __builtin_fmaf(x[1], hi, -y[1])};
    r              = saf_pk_fma(x, (saf_f2){lo, lo}, r);
    const saf_f2 f = saf_pk_add((saf_f2){y[0] - n[0], y[1] - n[1]}, r);
    saf_f2 z;
#    pragma unroll
    for (int i = 0; i < 2; i++) {
        int exponent;
        asm("v_cvt_i32_f32 %0, %1" : "=v"(exponent) : "v"(n[i]));
        float v = __builtin_amdgcn_ldexpf(__builtin_amdgcn_exp2f(f[i]), exponent);
        if (x[i] < __uint_as_float(0xc2ce8ed0u)) v = 0.f;
        if (x[i] > __uint_as_float(0x42b17218u)) v = INFINITY;
        z[i] = v;
    }
    return z;
}
// AITER Hygon hip_reduce.h uses row_ror:8 (0x128); warp_sort.h
// uses quad_perm 0xb1/0x4e and masked row_shl/shr:4 for exact XOR swaps.
// Compiler intrinsics retain dependency/hazard tracking. Reduction order
// stays XOR 8,4,2,1, and every floating-point addition is unchanged.
template<int MODE>
__device__ __forceinline__ static float target_xor(float x, int off) {
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

template<int H>
__global__ __launch_bounds__(256, 1) static void target64(const bf* q, const bf* kv, const float* ls, const int* idx,
                                                          float* out, int T, int S, int K, float scale) {
    constexpr int D = 512;
    // H64 permits five resident CTAs with a 12 KiB ring; H128 is VGPR-limited.
    constexpr int STAGES = H == 64 ? 3 : 4;
    const int t = blockIdx.x, lane = threadIdx.x & 63, w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6),
              r = lane & 15, g = lane >> 4, h = w * 16 + r, valid = K;
    __shared__ __align__(16) bf sm[STAGES * 2048];
    // The existing post-MMAC barrier retires all K readers before head sums.
    // The end-of-tile barrier retires sums readers before K staging resumes.
    float* sums                 = reinterpret_cast<float*>(sm);
    const unsigned long long ka = (unsigned long long)kv;
    const Desc kd               = {(int)ka, (int)((ka >> 32) | 0x40000000u), (int)0x80000000u, 0x00020000};
    AF qf[H / 64][32];
#    pragma unroll
    for (int hp = 0; hp < H / 64; hp++) {
        // Same aligned NN32 fragment pairing as attention forward.
        using QVec = unsigned int __attribute__((ext_vector_type(4)));
#    pragma unroll
        for (int j = 0; j < 16; j++) {
            QVec v = *(const QVec*)(q + ((size_t)t * H + h + hp * 64) * D + j * 32 + g * 8);
            asm volatile("" : "+v"(v)::"memory");
            *(unsigned long long*)&qf[hp][j * 2].x[0]     = ((unsigned long long*)&v)[0];
            *(unsigned long long*)&qf[hp][j * 2 + 1].x[0] = ((unsigned long long*)&v)[1];
        }
    }
    float lse[H / 64];
#    pragma unroll
    for (int hp = 0; hp < H / 64; hp++) lse[hp] = ls[(size_t)t * H + h + hp * 64];
    float total = 0;
    for (int base = 0; base < K; base += 64) {
        int pos = base + lane, ix = pos < K ? idx[(size_t)t * K + pos] : -1;
        // Exact validity of this query tile; retain the separate K-gather index.
        const unsigned long long valid_mask = __ballot(ix >= 0 && ix < S);
        if (!valid_mask) {
            if (w == 0 && pos < K) out[(size_t)t * K + pos] = 0;
            continue;
        }
        const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3), row = w * 16 + unsw / 32, col = unsw % 32;
        // The tile's original lane-distributed indices also serve K gathering
        // and score masks, with no assumptions about other query rows.
        int key     = __shfl(ix, row);
        auto fetchk = [&](int j) {
            int off = (key >= 0 && key < S) ? key * (D * 2) + (j * 32 + col) * 2 : -1;
            ld(kd, off, sm + (j % STAGES) * 2048 + w * 512);
        };
        // Lane-only swizzled offsets for the four 16-dim fragment groups.  They do
        // not depend on the K stage or the key, so compute them once and keep them
        // opaque: without that the backend rematerialises the whole swizzle chain
        // (~10 VALU ops) for every one of the 64 LDS reads per tile, and VALU is
        // this kernel's binding pipe (measured 91-94 % utilisation).
        const int lane_r = lane & 15, lane_g = lane >> 4;
        int frag_a[4];
#    pragma unroll
        for (int k = 0; k < 4; k++) {
            int a      = (k * 16 + lane_r) * 32 + lane_g * 8;
            a         ^= (a >> 6 & 7) << 3;
            frag_a[k]  = a;
        }
        asm volatile("" ::"v"(frag_a[0]), "v"(frag_a[1]), "v"(frag_a[2]), "v"(frag_a[3]) : "memory");
        CF sc[H / 64][4];
#    pragma unroll
        for (int k = 0; k < 4; k++) {
#    pragma unroll
            for (int hp = 0; hp < H / 64; hp++) du_fill_fragment(sc[hp][k], 0.f);
        }
#    pragma unroll
        for (int j = 0; j < STAGES - 1; j++) fetchk(j);
#    pragma unroll
        for (int j = 0; j < 16; j++) {
            waitn(min(STAGES - 2, 16 - j - 1));
            if (j + STAGES - 1 < 16) fetchk(j + STAGES - 1);  // Slot j+STAGES-1 is free after the preceding barrier.
#    pragma unroll
            for (int k = 0; k < 4; k++) {
                using U4 = unsigned int __attribute__((ext_vector_type(4)));
                U4 v     = *(const U4*)(sm + (j % STAGES) * 2048 + frag_a[k]);
                BC b0, b1;
                *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
#    pragma unroll
                for (int hp = 0; hp < H / 64; hp++) {
                    du_mma_sync(sc[hp][k], qf[hp][j * 2], b0, sc[hp][k]);
                    du_mma_sync(sc[hp][k], qf[hp][j * 2 + 1], b1, sc[hp][k]);
                }
            }
        }

        bar();
#    pragma unroll
        for (int k = 0; k < 4; k++) {
#    pragma unroll
            for (int z = 0; z < 4; z += 2) {
                const unsigned valid16 = (unsigned)(valid_mask >> (k * 16));
                const bool ok0 = (valid16 >> (g + 4 * z)) & 1u, ok1 = (valid16 >> (g + 4 * (z + 1))) & 1u;
                saf_f2 v = {0.f, 0.f};
#    pragma unroll
                for (int hp = 0; hp < H / 64; hp++) {
                    const saf_f2 scores  = {fmaf(sc[hp][k].x[z], scale, -lse[hp]),
                                            fmaf(sc[hp][k].x[z + 1], scale, -lse[hp])};
                    const saf_f2 ex      = saf_exp_pair(scores);
                    v[0]                += ok0 ? ex[0] : 0.f;
                    v[1]                += ok1 ? ex[1] : 0.f;
                }
#    pragma unroll
                for (int off = 8; off; off >>= 1) {
                    v = saf_pk_add(v, (saf_f2){target_xor<2>(v[0], off), target_xor<2>(v[1], off)});
                }
                if (r == 0) {
                    sums[w * 64 + k * 16 + g + 4 * z]       = v[0];
                    sums[w * 64 + k * 16 + g + 4 * (z + 1)] = v[1];
                }
            }
        }
        bar();
        if (w == 0 && pos < K) {
            float v                   = sums[lane] + sums[64 + lane] + sums[128 + lane] + sums[192 + lane];
            out[(size_t)t * K + pos]  = v;
            total                    += v;
        }
        bar();
    }
    if (w == 0) {
#    pragma unroll
        for (int off = 32; off; off >>= 1) total += __shfl_xor(total, off);
        if (lane == 0) sums[0] = total;
    }
    __threadfence_block();
    bar();
    float sum = sums[0];
    {
        // Preserve the reference's clamp for zero and tiny target rows.
        float inv = 1.f / (isnan(sum) ? sum : fmaxf(sum, 1e-12f));
        for (int k = threadIdx.x; k < K; k += 256) out[(size_t)t * K + k] *= inv;
    }
}
template<int H, int PH = H, int NT = 256, int GUARD = 0>
__global__ __launch_bounds__(NT, 1) static void target576(const bf* q, const bf* kv, const float* ls, const int* idx,
                                                          float* out, int T, int S, int K, float scale,
                                                          const int* padding_guard) {
    if constexpr (GUARD == 1) {
        if (padding_guard[0] == 0) return;
    }
    if constexpr (GUARD == 2) {
        if (padding_guard[0] != 0) return;
    }

    constexpr int D = 576;
    // H64 permits five resident CTAs with a 12 KiB ring; H128 is VGPR-limited.
    constexpr int STAGES = (H == 64 || NT == 128) ? 3 : 4;
    const int t = blockIdx.x, lane = threadIdx.x & 63, w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6),
              r = lane & 15, g = lane >> 4, h = w * 16 + r, valid = K;
    __shared__ __align__(16) bf sm[STAGES * 2048];
    // The existing post-MMAC barrier retires all K readers before head sums.
    // The end-of-tile barrier retires sums readers before K staging resumes.
    float* sums                 = reinterpret_cast<float*>(sm);
    const unsigned long long ka = (unsigned long long)kv;
    const Desc kd               = {(int)ka, (int)((ka >> 32) | 0x40000000u), (int)0x80000000u, 0x00020000};
    AF qf[((H + 63) / 64)][D / 16];
#    pragma unroll
    for (int hp = 0; hp < ((H + 63) / 64); hp++) {
        // Same aligned NN32 fragment pairing as attention forward.
        using QVec = unsigned int __attribute__((ext_vector_type(4)));
#    pragma unroll
        for (int j = 0; j < D / 32; j++) {
            QVec v = (H % 64 == 0 || h + hp * 64 < H)
                         ? *(const QVec*)(q + ((size_t)t * PH + h + hp * 64) * D + j * 32 + g * 8)
                         : QVec{0, 0, 0, 0};
            asm volatile("" : "+v"(v)::"memory");
            *(unsigned long long*)&qf[hp][j * 2].x[0]     = ((unsigned long long*)&v)[0];
            *(unsigned long long*)&qf[hp][j * 2 + 1].x[0] = ((unsigned long long*)&v)[1];
        }
    }
    float lse[((H + 63) / 64)];
#    pragma unroll
    for (int hp = 0; hp < ((H + 63) / 64); hp++)
        lse[hp] = (H % 64 == 0 || h + hp * 64 < H) ? ls[(size_t)t * PH + h + hp * 64] : INFINITY;
    float total = 0;
    for (int base = 0; base < K; base += 64) {
        int pos = base + lane, ix = pos < K ? idx[(size_t)t * K + pos] : -1;
        // Exact validity of this query tile; retain the separate K-gather index.
        const unsigned long long valid_mask = __ballot(ix >= 0 && ix < S);
        if (!valid_mask) {
            if (w == 0 && pos < K) out[(size_t)t * K + pos] = 0;
            continue;
        }
        const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3), row = w * 16 + unsw / 32, col = unsw % 32;
        // The tile's original lane-distributed indices also serve K gathering
        // and score masks, with no assumptions about other query rows.
        int key     = __shfl(ix, row);
        int key2    = __shfl(ix, row + 32);
        auto fetchk = [&](int j) {
            int off = (key >= 0 && key < S) ? key * (D * 2) + (j * 32 + col) * 2 : -1;
            ld(kd, off, sm + (j % STAGES) * 2048 + w * 512);
            if constexpr (NT == 128) {
                int off2 = (key2 >= 0 && key2 < S) ? key2 * (D * 2) + (j * 32 + col) * 2 : -1;
                ld(kd, off2, sm + (j % STAGES) * 2048 + (w + 2) * 512);
            }
        };
        // Lane-only swizzled offsets for the four 16-dim fragment groups.  They do
        // not depend on the K stage or the key, so compute them once and keep them
        // opaque: without that the backend rematerialises the whole swizzle chain
        // (~10 VALU ops) for every one of the 64 LDS reads per tile, and VALU is
        // this kernel's binding pipe (measured 91-94 % utilisation).
        const int lane_r = lane & 15, lane_g = lane >> 4;
        int frag_a[4];
#    pragma unroll
        for (int k = 0; k < 4; k++) {
            int a      = (k * 16 + lane_r) * 32 + lane_g * 8;
            a         ^= (a >> 6 & 7) << 3;
            frag_a[k]  = a;
        }
        asm volatile("" ::"v"(frag_a[0]), "v"(frag_a[1]), "v"(frag_a[2]), "v"(frag_a[3]) : "memory");
        CF sc[((H + 63) / 64)][4];
#    pragma unroll
        for (int k = 0; k < 4; k++) {
#    pragma unroll
            for (int hp = 0; hp < ((H + 63) / 64); hp++) du_fill_fragment(sc[hp][k], 0.f);
        }
#    pragma unroll
        for (int j = 0; j < STAGES - 1; j++) fetchk(j);
#    pragma unroll
        for (int j = 0; j < D / 32; j++) {
            waitn((256 / NT) * min(STAGES - 2, D / 32 - j - 1));
            if (j + STAGES - 1 < D / 32)
                fetchk(j + STAGES - 1);  // Slot j+STAGES-1 is free after the preceding barrier.
#    pragma unroll
            for (int k = 0; k < 4; k++) {
                using U4 = unsigned int __attribute__((ext_vector_type(4)));
                U4 v     = *(const U4*)(sm + (j % STAGES) * 2048 + frag_a[k]);
                BC b0, b1;
                *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
#    pragma unroll
                for (int hp = 0; hp < ((H + 63) / 64); hp++) {
                    du_mma_sync(sc[hp][k], qf[hp][j * 2], b0, sc[hp][k]);
                    du_mma_sync(sc[hp][k], qf[hp][j * 2 + 1], b1, sc[hp][k]);
                }
            }
        }

        bar();
#    pragma unroll
        for (int k = 0; k < 4; k++) {
#    pragma unroll
            for (int z = 0; z < 4; z += 2) {
                const unsigned valid16 = (unsigned)(valid_mask >> (k * 16));
                const bool ok0 = (valid16 >> (g + 4 * z)) & 1u, ok1 = (valid16 >> (g + 4 * (z + 1))) & 1u;
                saf_f2 v = {0.f, 0.f};
#    pragma unroll
                for (int hp = 0; hp < ((H + 63) / 64); hp++) {
                    const saf_f2 scores  = {fmaf(sc[hp][k].x[z], scale, -lse[hp]),
                                            fmaf(sc[hp][k].x[z + 1], scale, -lse[hp])};
                    const saf_f2 ex      = saf_exp_pair(scores);
                    v[0]                += (ok0 && (H % 64 == 0 || h + hp * 64 < H)) ? ex[0] : 0.f;
                    v[1]                += (ok1 && (H % 64 == 0 || h + hp * 64 < H)) ? ex[1] : 0.f;
                }
#    pragma unroll
                for (int off = 8; off; off >>= 1) {
                    v = saf_pk_add(v, (saf_f2){target_xor<2>(v[0], off), target_xor<2>(v[1], off)});
                }
                if (r == 0) {
                    sums[w * 64 + k * 16 + g + 4 * z]       = v[0];
                    sums[w * 64 + k * 16 + g + 4 * (z + 1)] = v[1];
                }
            }
        }
        bar();
        if (w == 0 && pos < K) {
            float v;
            if constexpr (NT == 256)
                v = sums[lane] + sums[64 + lane] + sums[128 + lane] + sums[192 + lane];
            else
                v = sums[lane] + sums[64 + lane];
            out[(size_t)t * K + pos]  = v;
            total                    += v;
        }
        bar();
    }
    if (w == 0) {
#    pragma unroll
        for (int off = 32; off; off >>= 1) total += __shfl_xor(total, off);
        if (lane == 0) sums[0] = total;
    }
    __threadfence_block();
    bar();
    float sum = sums[0];
    {
        // Preserve the reference's clamp for zero and tiny target rows.
        float inv = 1.f / (isnan(sum) ? sum : fmaxf(sum, 1e-12f));
        for (int k = threadIdx.x; k < K; k += NT) out[(size_t)t * K + k] *= inv;
    }
}
#else
template<int H>
__global__ static void target64(const bf*, const bf*, const float*, const int*, float*, int, int, int, float) {}
template<int H, int PH = H, int NT = 256, int GUARD = 0>
__global__ static void target576(const bf*, const bf*, const float*, const int*, float*, int, int, int, float,
                                 const int*) {}

#endif
}  // namespace hip_dsa_target64

// Per-call GPU guards are refreshed by the binding's zeroed row counters.
// Require all padded Q bits to be zero, LSE=+inf, and finite KV/scale.
// In particular LSE=+inf alone must not hide NaNs in the omitted QK work.
__global__ static void target_check_padding(const unsigned short* q, const float* ls, int* flag, int T, float scale) {
    const int t  = blockIdx.x;
    unsigned bad = !isfinite(scale);
    if (threadIdx.x < 32) bad |= __float_as_uint(ls[(size_t)t * 64 + 32 + threadIdx.x]) != 0x7f800000u;
    // The block-wide test is uniform across barriers and makes dense inputs cheap.
    if (__syncthreads_or(bad)) {
        if (threadIdx.x == 0) atomicOr(flag, 1);
        return;
    }
    for (int i = threadIdx.x; i < 32 * 576 / 8; i += blockDim.x) {
        const uint4 v  = ((const uint4*)q)[((size_t)t * 64 + 32) * 576 / 8 + i];
        bad           |= (v.x | v.y | v.z | v.w) & 0x7fff7fff;
    }
    if (__any(bad) && (threadIdx.x & 63) == 0) atomicOr(flag, 1);
}
__global__ static void target_check_finite_kv(const unsigned short* kv, int* flag, int n) {
    // Other CTAs can set the rejection flag with atomicOr during this scan.
    if (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) return;
    unsigned bad = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        bad |= (kv[i] & 0x7f80) == 0x7f80;
    if (__any(bad) && (threadIdx.x & 63) == 0) atomicOr(flag, 1);
}
static bool launch_sparse_attn_target(hipStream_t stream, const void* q, const void* k, const float* lse,
                                      const int* topk_idxs, float* target, int* row_count, int B, int Sq, int Sk,
                                      int Hq, int D, int topk, float softmax_scale, bool topk_indices_global) {
    // Fused MMAC gather-GEMM path (tensor core) only; no non-fused fallback.
    // Hq may exceed 64 and D may be 512 (DSv4-flash/pro): the dispatcher picks
    // the LDS-Q kernel or the global-A kernel variant as the shape requires.
    if (!row_count || B <= 0 || Sq <= 0 || Sk <= 0 || Hq > 128) return false;
    if (!(sparse_mmac_supported() && Hq >= 1 && (D % 16) == 0 && topk >= 1 && sparse_mmac_shm_fits(Hq, D, topk))) {
        return false;
    }
    if (B == 1 && Sq >= 1024 && D == 512 && (Hq == 64 || Hq == 128) && (topk == 512 || topk == 1024 || topk == 1152) &&
        (int64_t)Sk * D * 2 <= INT32_MAX) {
        int dev = 0;
        hipDeviceProp_t_v2 prop{};
        {
            const hipError_t status = hipGetDevice(&dev);
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipGetDevice failed: %s",
                                        hipGetErrorString(status));
            }
        }
        {
            const hipError_t status = hipGetDeviceProperties_v2(&prop, dev);
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipGetDeviceProperties_v2 failed: %s",
                                        hipGetErrorString(status));
            }
        }
        if (strncmp(prop.gcnArchName, "gfx936", 6) == 0 || strncmp(prop.gcnArchName, "gfx938", 6) == 0) {
#define TARGET64_RUN(HEADS)                                                                                        \
    hipLaunchKernelGGL((hip_dsa_target64::target64<HEADS>), dim3(Sq), 256, 0, stream, (const hip_dsa_mla64::bf*)q, \
                       (const hip_dsa_mla64::bf*)k, lse, topk_idxs, target, Sq, Sk, topk, softmax_scale)
            if (Hq == 64) {
                TARGET64_RUN(64);
            } else {
                TARGET64_RUN(128);
            }
#undef TARGET64_RUN
            {
                const hipError_t status = hipGetLastError();
                if (status != hipSuccess) {
                    throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_target HIP call failed: %s",
                                            hipGetErrorString(status));
                }
            }
            return true;
        }
    }
    if (B == 1 && Sq == 4096 && Sk == 4096 && D == 576 && (Hq == 32 || Hq == 64) && topk == 2048) {
        int dev = 0;
        hipDeviceProp_t_v2 prop{};
        {
            const hipError_t status = hipGetDevice(&dev);
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipGetDevice failed: %s",
                                        hipGetErrorString(status));
            }
        }
        {
            const hipError_t status = hipGetDeviceProperties_v2(&prop, dev);
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipGetDeviceProperties_v2 failed: %s",
                                        hipGetErrorString(status));
            }
        }
        if (strncmp(prop.gcnArchName, "gfx936", 6) == 0 || strncmp(prop.gcnArchName, "gfx938", 6) == 0) {
#define GLM_TARGET(HEADS, PH, NT, G)                                                                                   \
    hipLaunchKernelGGL((hip_dsa_target64::target576<HEADS, PH, NT, G>), dim3(Sq), NT, 0, stream,                       \
                       (const hip_dsa_mla64::bf*)q, (const hip_dsa_mla64::bf*)k, lse, topk_idxs, target, Sq, Sk, topk, \
                       softmax_scale, row_count)
            if (Hq == 32) {
                GLM_TARGET(32, 32, 128, 0);
            } else {
                hipLaunchKernelGGL(target_check_padding, dim3(Sq), 256, 0, stream, (const unsigned short*)q, lse,
                                   row_count, Sq, softmax_scale);
                hipLaunchKernelGGL(target_check_finite_kv, dim3(128), 256, 0, stream, (const unsigned short*)k,
                                   row_count, Sk * D);
                GLM_TARGET(64, 64, 256, 1);
                GLM_TARGET(32, 64, 128, 2);
            }
#undef GLM_TARGET
            {
                const hipError_t status = hipGetLastError();
                if (status != hipSuccess) {
                    throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_target HIP call failed: %s",
                                            hipGetErrorString(status));
                }
            }
            return true;
        }
    }
    if (!launch_sparse_attn_target_mmac(stream, q, k, lse, topk_idxs, target, row_count, B, Sq, Sk, Hq, D, topk,
                                        softmax_scale, topk_indices_global)) {
        return false;
    }
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_target HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}

namespace ftrain {

bool launchBf16SparseAttnTarget(FTrainStream stream, const void* q, const void* k, const float* lse,
                                const int* topk_indices, float* target, int* row_count, int batch, int query_length,
                                int key_length, int query_heads, int head_dim, int selection_width, float softmax_scale,
                                bool topk_indices_global) {
    return launch_sparse_attn_target(stream, q, k, lse, topk_indices, target, row_count, batch, query_length,
                                     key_length, query_heads, head_dim, selection_width, softmax_scale,
                                     topk_indices_global);
}

bool bf16SparseAttnTargetSparseMmacSupported() { return sparse_mmac_supported(); }

bool bf16SparseAttnTargetSparseMmacShmFits(int query_heads, int head_dim, int selection_width) {
    return sparse_mmac_shm_fits(query_heads, head_dim, selection_width);
}

void bf16SparseAttnTargetZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    const hipError_t status = hipMemsetAsync(memory, 0, static_cast<std::size_t>(bytes), stream);
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Workspace zero-fill failed: %s", hipGetErrorString(status));
    }
}

}  // namespace ftrain
