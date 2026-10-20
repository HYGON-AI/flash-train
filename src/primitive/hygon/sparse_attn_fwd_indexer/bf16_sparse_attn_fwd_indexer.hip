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

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_attn_fwd_indexer/bf16_sparse_attn_fwd_indexer.hpp"

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

namespace hip_dsa_mla64 {
using bf = __hip_bfloat16;
#if defined(__gfx936__) || defined(__gfx938__)
using namespace du::dumma;
using AF   = DUFragment<matrix_a, 16, 16, 16, bf, row_major>;
using BC   = DUFragment<matrix_b, 16, 16, 16, bf, col_major>;
using BR   = DUFragment<matrix_b, 16, 16, 16, bf, row_major>;
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

// Adapted from fused_dsa/hip_dsa/kernels/sparse_attn_fwd.hip.

typedef uint16_t bf16;

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;
typedef __hip_bfloat16 hbf16;
// Large-D QK dots otherwise form one dependency chain of 32+ fp32 MMAC
// updates.  Splitting that chain and merging the partials in double mirrors
// the large-D dense-attention fix: it keeps the score handed to softmax within
// one fp32 rounding even when cancellation makes the serial sum drift.
template<int NACC>
__device__ __forceinline__ static float saf_scaled_score(const DUFragment<accumulator, 16, 16, 16, float>* acc,
                                                         int element, float scale) {
    if constexpr (NACC == 1) {
        return acc[0].x[element] * scale;
    } else {
        double sum = 0.0;
#    pragma unroll
        for (int i = 0; i < NACC; ++i) sum += (double)acc[i].x[element];
        return (float)(sum * (double)scale);
    }
}

// TOPK > 0: topk is a compile-time constant (the inner loops, LDS strides and
// MMAC chains are then fully unrolled / strength-reduced; runtime shape params
// cost ~5x here).  TOPK == 0: runtime topk fallback.  D is always ND*16.

#endif  // supported gfx archs

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
template<int ND, int NT, int TOPK = 0>
__global__ static void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_fwd_fused_kernel(const bf16* __restrict__ q,           // (T, H, D)
                                 const bf16* __restrict__ kv,          // (T_kv, D)
                                 const float* __restrict__ attn_sink,  // (H,)
                                 const int* __restrict__ idxs,         // (T, topk)
                                 const int* __restrict__ topk_length,  // (T,) or NULL
                                 bf16* __restrict__ out,               // (T, H, D_v)
                                 float* __restrict__ lse,              // (T, H)
                                 int T, int H, int D_rt, int D_v_rt, int T_kv, int topk_rt, float softmax_scale) {
    // The launcher guarantees V=KV[..., :512]. Keep the accumulator width
    // static so runtime bounds do not force the register array into scratch.
    constexpr int D_v         = 512;
    constexpr bool TILE_INDEX = (ND == 32 || ND == 36) && NT == 1 && TOPK >= 1024;
    (void)D_v_rt;
    constexpr int D = ND * 16;                      // compile-time head dim
    const int topk  = (TOPK > 0) ? TOPK : topk_rt;  // folds to a constant for TOPK > 0
    (void)D_rt;
    const int t        = blockIdx.x;
    const int hb       = blockIdx.y * SAF_HBLK;
    const int lane     = threadIdx.x & 63;
    const int warp     = ((ND == 32 || ND == 36) && NT == 1 && !(ND == 32 && TOPK == 512))
                             ? __builtin_amdgcn_readfirstlane(threadIdx.x >> 6)
                             : threadIdx.x >> 6;  // four wave64 groups
    const int r0       = warp * 16;
    const int h        = hb + r0 + (lane & 15);
    const bool lane_ok = (h < H);
    const int QLD = D + 8, KLD = D + 8;
    constexpr int nd = ND;
    const int g      = lane >> 4;

    extern __shared__ char smem_raw[];
    hbf16* k_lds = (hbf16*)smem_raw;  // (16*NT) x KLD K/V stage
    int* idx_lds = (int*)(k_lds + 16 * NT * KLD);
    hbf16* p_lds = (hbf16*)(idx_lds + (TILE_INDEX ? 32 : topk));  // 4 warps x 256 bf16

    int n_valid_t = topk;
    if (topk_length != NULL) {
        int tl = topk_length[t];
        if (tl < n_valid_t) n_valid_t = tl > 0 ? tl : 0;
    }

    bool keys_all_valid;
    if constexpr (TILE_INDEX) {
        bool bad_key = n_valid_t != topk;
        for (int i = threadIdx.x; i < topk; i += SAF_THREADS) {
            const int key = idxs[(size_t)t * topk + i];
            if (i < 16) idx_lds[i] = key;
            bad_key |= key < 0 || key >= T_kv;
        }
        keys_all_valid = __syncthreads_count(bad_key) == 0;
    } else {
        __shared__ int all_valid_lds;
        if (threadIdx.x == 0) all_valid_lds = (n_valid_t == topk);
        __syncthreads();
        for (int i = threadIdx.x; i < topk; i += SAF_THREADS) {
            const int idx = idxs[(size_t)t * topk + i];
            idx_lds[i]    = idx;
            if (idx < 0 || idx >= T_kv) atomicExch(&all_valid_lds, 0);
        }
        // idx_lds is read by the first K-gather below; without a barrier the
        // gather can race ahead and see uninitialized LDS (idx 0 for every slot).
        __syncthreads();

        keys_all_valid = all_valid_lds;
    }

    DUFragment<accumulator, 16, 16, 16, float> outacc[ND];
#    pragma unroll
    for (int dc = 0; dc < ND; dc++) du_fill_fragment(outacc[dc], 0.0f);

    float sink_val = -INFINITY;
    float m        = -INFINITY;
    float l        = 0.0f;
    if (lane_ok) { sink_val = attn_sink[h]; }

    const int hq = hb + r0;  // first global Q row of this warp
    // Cache this warp's Q rows (16 heads x D) in registers once per block.
    // The S-tile MMAC loop re-reads the same Q fragment every 16-key round;
    // loading it from global each round makes Q the dominant VMEM read
    // (~89% of VMEM_RD instructions in the dsv4_pro profile, TCC_MISS 22x
    // FlashMLA).  With the fragment cached, Q touches global memory once.
    hbf16 qreg[ND][4];
    if (hq < H) {
        const int qrow_h   = hq + (lane & 15);
        const int qrow     = qrow_h < H ? qrow_h : (H - 1);
        const hbf16* qbase = (const hbf16*)(q + ((size_t)t * H + qrow) * D);
#    pragma unroll
        for (int kk = 0; kk < ND; kk++) {
            *(uint64_t*)&qreg[kk][0] = *(const uint64_t*)(qbase + kk * 16 + ((lane >> 4) << 2));
            if (qrow_h >= H) qreg[kk][0] = qreg[kk][1] = qreg[kk][2] = qreg[kk][3] = hbf16(0);
        }
    } else {
#    pragma unroll
        for (int kk = 0; kk < ND; kk++) qreg[kk][0] = qreg[kk][1] = qreg[kk][2] = qreg[kk][3] = hbf16(0);
    }
    // Direct global->LDS K/V gather descriptor for gfx936/gfx938.
    // buffer_load_dwordx4 .. lds writes 16B per active lane to LDS at
    // (m0 + vaddr); the same vaddr is the byte offset inside the global
    // K/V row, so the lane's 16B chunk lands at the matching LDS chunk.
    typedef int v4i __attribute__((ext_vector_type(4)));
    const uint64_t kv_addr = (uint64_t)kv;
    v4i kv_desc;
    kv_desc[0] = (int)(uint32_t)(kv_addr & 0xffffffffu);
    kv_desc[1] = (int)((uint32_t)(kv_addr >> 32) | 0x40000000u);
    kv_desc[2] = (int)0x80000000u;
    kv_desc[3] = (int)0x00020000u;
    // Compact short rows otherwise execute full MMAC tiles for their padded
    // suffix. Large rows already skip empty tiles and benefit from retaining
    // the compile-time traversal bound. Keep physical index/LDS strides.
    const int traversal_topk = ((ND == 32 || ND == 36) && topk < 1024) ? n_valid_t : topk;
    const int ntiles         = (traversal_topk + 15) >> 4;

    auto gather_tile = [&](int kt0, const int ck[NT], int ck_tot, hbf16* region) {
        // Gather the tile's K/V rows into LDS via direct global->LDS loads.
        // The intrinsic writes 16B per active lane at (dst + lane*16), so
        // one 64-lane warp stages one row per iteration: lane l reads the
        // l-th 16B chunk of the global row and lands it at the same chunk
        // offset of LDS row n.  n must be warp-uniform (m0 is scalar), which
        // fixes the row stride across all head dims (D/8 <= 64 lanes/row).
        // D512/K512 retains the original schedule: batched gathering is
        // neutral for full rows but regresses short causal rows by about 2%.
        if constexpr (NT == 1 && (ND == 32 || ND == 36) && !(ND == 32 && TOPK == 512)) {
            // Load every index before issuing global-to-LDS transfers. A
            // later index load would otherwise wait for an earlier K transfer.
            int keys[4];
#    pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = warp + j * 4;
                keys[j]     = n < ck_tot ? idx_lds[TILE_INDEX ? ((kt0 & 1) * 16 + n) : kt0 * 16 + n] : -1;
            }
            asm volatile("" ::"v"(keys[0]), "v"(keys[1]), "v"(keys[2]), "v"(keys[3]) : "memory");
#    pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = warp + j * 4;
                if (n < ck_tot) {
                    const int idx    = keys[j];
                    const bool valid = idx >= 0 && idx < T_kv && kt0 * 16 + n < n_valid_t;
#    pragma unroll
                    for (int dc = 0; dc < D; dc += 512) {
                        if (dc + lane * 8 < D) {
                            if (valid) {
                                __builtin_amdgcn_raw_buffer_load_lds(
                                    kv_desc, (__attribute__((address_space(3))) int*)(region + n * KLD + dc), 16,
                                    idx * (D * 2) + dc * 2 + lane * 16, 0, 0, 0);
                            } else {
                                uint64_t* dst = (uint64_t*)(region + n * KLD + dc + lane * 8);
                                dst[0] = dst[1] = 0;
                            }
                        }
                    }
                }
            }
        } else {
            constexpr int CHUNKS = D / 8;
            if constexpr (CHUNKS <= 64) {
                for (int n = warp; n < ck_tot; n += 4) {
                    int ti = 0, rem = n;
#    pragma unroll
                    for (int i = 0; i < NT; i++) {
                        if (rem < ck[i]) break;  // n lives in tile i at offset rem
                        rem -= ck[i];
                        ti++;
                    }
                    int gidx = (kt0 + ti) * 16 + rem;
                    int idx  = idx_lds[TILE_INDEX ? ((kt0 & 1) * 16 + rem) : gidx];
                    if (lane < CHUNKS) {
                        if (keys_all_valid) {
                            __builtin_amdgcn_raw_buffer_load_lds(
                                kv_desc, (__attribute__((address_space(3))) int*)(region + (size_t)n * KLD), 16,
                                idx * (D * 2) + lane * 16, 0, 0, 0);
                        } else {
                            hbf16* dst       = region + (size_t)n * KLD + lane * 8;
                            const bool valid = gidx < n_valid_t && idx >= 0 && idx < T_kv;
                            if (valid) {
                                __builtin_amdgcn_raw_buffer_load_lds(
                                    kv_desc, (__attribute__((address_space(3))) int*)(region + (size_t)n * KLD), 16,
                                    idx * (D * 2) + lane * 16, 0, 0, 0);
                            } else {
                                ((uint64_t*)dst)[0] = 0;
                                ((uint64_t*)dst)[1] = 0;
                            }
                        }
                    }
                }
            } else {
                for (int n = warp; n < ck_tot; n += 4) {
                    int ti = 0, rem = n;
#    pragma unroll
                    for (int i = 0; i < NT; i++) {
                        if (rem < ck[i]) break;
                        rem -= ck[i];
                        ti++;
                    }
                    const int gidx   = (kt0 + ti) * 16 + rem;
                    const int idx    = idx_lds[TILE_INDEX ? ((kt0 & 1) * 16 + rem) : gidx];
                    const bool valid = gidx < n_valid_t && idx >= 0 && idx < T_kv;
#    pragma unroll
                    for (int dc = 0; dc < D; dc += 512) {
                        if (dc + lane * 8 < D) {
                            if (valid) {
                                __builtin_amdgcn_raw_buffer_load_lds(
                                    kv_desc, (__attribute__((address_space(3))) int*)(region + n * KLD + dc), 16,
                                    idx * (D * 2) + dc * 2 + lane * 16, 0, 0, 0);
                            } else {
                                uint64_t* dst = (uint64_t*)(region + n * KLD + dc + lane * 8);
                                dst[0] = dst[1] = 0;
                            }
                        }
                    }
                }
            }
        }
        // Zero the region's rows that the partial tail tile did not stage.
        // The P@V MMA multiplies a zero P weight by every V row of the tile,
        // so a stale V row (e.g. an INF/NaN bit pattern left in LDS by a
        // previous kernel) would turn 0 * stale into NaN and poison the
        // valid heads' accumulator.
        if (ck_tot < 16 * NT) {
            for (int u = threadIdx.x; u < (16 * NT) * KLD; u += SAF_THREADS) {
                if (u / KLD >= ck_tot) region[u] = hbf16(0);
            }
        }
    };

    // A skipped tile must also publish the next index tile before the
    // CTA barrier, preserving traversal through holes and topk_length.
    auto queue_index = [&](int kt) {
        if constexpr (TILE_INDEX) {
            if (threadIdx.x < 16 && (kt + 1) * 16 + threadIdx.x < topk)
                idx_lds[((kt + 1) & 1) * 16 + threadIdx.x] = idxs[(size_t)t * topk + (kt + 1) * 16 + threadIdx.x];
        }
    };
    const bool test_empty_tiles = !keys_all_valid;
    for (int kt = 0; kt < ntiles; kt += NT) {
        if (NT == 1 && (TOPK == 0 || TOPK >= 1024) && topk >= 1024 && test_empty_tiles) {
            const int ki  = kt * 16 + (lane & 15);
            const int key = ki < n_valid_t ? idx_lds[TILE_INDEX ? ((kt & 1) * 16 + (lane & 15)) : ki] : -1;
            if (!__builtin_amdgcn_readfirstlane(__any(key >= 0 && key < T_kv))) {
                queue_index(kt);
                __syncthreads();
                continue;
            }
        }

        int ck[NT];
        int ck_tot = 0;
#    pragma unroll
        for (int i = 0; i < NT; i++) {
            int c   = topk - (kt + i) * 16;
            ck[i]   = (c > 16) ? 16 : (c < 0 ? 0 : c);
            ck_tot += ck[i];
        }
        gather_tile(kt, ck, ck_tot, k_lds);
        queue_index(kt);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        const hbf16* kbase = k_lds;

        // S tiles (16 heads x 16 keys) via MMAC over D; NT chains interleave.
        constexpr int SCORE_NACC = (D >= 512) ? 2 : 1;
        DUFragment<accumulator, 16, 16, 16, float> acc[NT * SCORE_NACC];
#    pragma unroll
        for (int i = 0; i < NT * SCORE_NACC; i++) du_fill_fragment(acc[i], 0.0f);
        if (hq < H) {
// Issue two independent LDS matrix reads before consuming either
// fragment.  On gfx936/gfx938 this lets the second ds_read_m* sit
// behind the first MMAC instead of rebuilding a load/MMAC wait
// chain for every 16 columns.
#    pragma unroll
            for (int kk = 0; kk < ND; kk += 2) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> a0, a1;
                *(uint64_t*)&a0.x[0] = *(const uint64_t*)&qreg[kk][0];
                *(uint64_t*)&a1.x[0] = *(const uint64_t*)&qreg[kk + 1][0];
#    pragma unroll
                for (int i = 0; i < NT; i++) {
                    if (ck[i] > 0) {
                        DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> b0, b1;
                        int koff = 0;
#    pragma unroll
                        for (int j = 0; j < i; j++) koff += ck[j];
                        du_load_matrix_sync(b0, kbase + (size_t)koff * KLD + kk * 16, KLD);
                        du_load_matrix_sync(b1, kbase + (size_t)koff * KLD + (kk + 1) * 16, KLD);
                        auto& acc0 = acc[i * SCORE_NACC + (kk % SCORE_NACC)];
                        auto& acc1 = acc[i * SCORE_NACC + ((kk + 1) % SCORE_NACC)];
                        du_mma_sync(acc0, a0, b0, acc0);
                        du_mma_sync(acc1, a1, b1, acc1);
                    }
                }
            }
        }

// Per-tile online-softmax epilogues: P tiles staged in LDS; when a
// later tile in this round raises the max, the earlier staged P tiles
// are rescaled in LDS so the P@V pass uses exp(S - m_new).
#    pragma unroll
        for (int i = 0; i < NT; i++) {
            if (ck[i] > 0) {
                const int c0 = (kt + i) * 16 + g;
                const int c1 = c0 + 4, c2 = c0 + 8, c3 = c0 + 12;
                const bool ok0 = (c0 < n_valid_t) && (idx_lds[TILE_INDEX ? ((kt & 1) * 16 + g + 0) : c0] >= 0) &&
                                 (idx_lds[TILE_INDEX ? ((kt & 1) * 16 + g + 0) : c0] < T_kv);
                const bool ok1 = (c1 < n_valid_t) && (idx_lds[TILE_INDEX ? ((kt & 1) * 16 + g + 4) : c1] >= 0) &&
                                 (idx_lds[TILE_INDEX ? ((kt & 1) * 16 + g + 4) : c1] < T_kv);
                const bool ok2 = (c2 < n_valid_t) && (idx_lds[TILE_INDEX ? ((kt & 1) * 16 + g + 8) : c2] >= 0) &&
                                 (idx_lds[TILE_INDEX ? ((kt & 1) * 16 + g + 8) : c2] < T_kv);
                const bool ok3 = (c3 < n_valid_t) && (idx_lds[TILE_INDEX ? ((kt & 1) * 16 + g + 12) : c3] >= 0) &&
                                 (idx_lds[TILE_INDEX ? ((kt & 1) * 16 + g + 12) : c3] < T_kv);
                // Phantom heads (h >= H): -inf scores so they never win the
                // max and their e = 0 (no pollution of the group reduction).
                const auto* score_acc = &acc[i * SCORE_NACC];
                const float s0        = lane_ok ? saf_scaled_score<SCORE_NACC>(score_acc, 0, softmax_scale) : -INFINITY;
                const float s1        = lane_ok ? saf_scaled_score<SCORE_NACC>(score_acc, 1, softmax_scale) : -INFINITY;
                const float s2        = lane_ok ? saf_scaled_score<SCORE_NACC>(score_acc, 2, softmax_scale) : -INFINITY;
                const float s3        = lane_ok ? saf_scaled_score<SCORE_NACC>(score_acc, 3, softmax_scale) : -INFINITY;
                float tmax            = -INFINITY;
                if (ok0) tmax = fmaxf(tmax, s0);
                if (ok1) tmax = fmaxf(tmax, s1);
                if (ok2) tmax = fmaxf(tmax, s2);
                if (ok3) tmax = fmaxf(tmax, s3);
                tmax = fmaxf(tmax, __shfl_xor(tmax, 16));
                tmax = fmaxf(tmax, __shfl_xor(tmax, 32));
                if (tmax > m) {
                    const float corr  = expf(m - tmax);
                    l                *= corr;
                    for (int dc = 0; dc < ND; dc++) {
                        if (dc * 16 >= D_v) continue;
                        outacc[dc].x[0] *= corr;
                        outacc[dc].x[1] *= corr;
                        outacc[dc].x[2] *= corr;
                        outacc[dc].x[3] *= corr;
                    }
                    m = tmax;
                    // Rescale this round's already-staged P tiles (j < i).
                    hbf16* pw0 = p_lds + (size_t)warp * (NT * 256);
#    pragma unroll
                    for (int j = 0; j < i; j++) {
                        const int pr           = (lane & 15) * 16 + g;
                        pw0[j * 256 + pr]      = __float2bfloat16(__bfloat162float(pw0[j * 256 + pr]) * corr);
                        pw0[j * 256 + pr + 4]  = __float2bfloat16(__bfloat162float(pw0[j * 256 + pr + 4]) * corr);
                        pw0[j * 256 + pr + 8]  = __float2bfloat16(__bfloat162float(pw0[j * 256 + pr + 8]) * corr);
                        pw0[j * 256 + pr + 12] = __float2bfloat16(__bfloat162float(pw0[j * 256 + pr + 12]) * corr);
                    }
                }
                const float e0                 = (ok0 && lane_ok) ? expf(s0 - m) : 0.0f;
                const float e1                 = (ok1 && lane_ok) ? expf(s1 - m) : 0.0f;
                const float e2                 = (ok2 && lane_ok) ? expf(s2 - m) : 0.0f;
                const float e3                 = (ok3 && lane_ok) ? expf(s3 - m) : 0.0f;
                float srow                     = e0 + e1 + e2 + e3;
                srow                          += __shfl_xor(srow, 16);
                srow                          += __shfl_xor(srow, 32);
                l                             += srow;
                hbf16* pw                      = p_lds + (size_t)warp * (NT * 256) + i * 256;
                pw[(lane & 15) * 16 + g]       = __float2bfloat16(e0);
                pw[(lane & 15) * 16 + g + 4]   = __float2bfloat16(e1);
                pw[(lane & 15) * 16 + g + 8]   = __float2bfloat16(e2);
                pw[(lane & 15) * 16 + g + 12]  = __float2bfloat16(e3);
            }
        }
        // No barrier here: each warp stages P in its own p_lds region and
        // only that warp reads it back, so the writes are warp-local.

        // QK uses all D channels; PV uses only V = K[..., :D_v].
        if (hq < H) {
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[NT];
#    pragma unroll
            for (int i = 0; i < NT; i++) {
                if (ck[i] > 0) du_load_matrix_sync(pa[i], p_lds + (size_t)warp * (NT * 256) + i * 256, 16);
            }
            // Reuse the indexer forward's verified normal matrix-read layout.
            // Each 128-bit LDS read supplies two contiguous-column fragments.
            if (NT == 1 && (ND == 32 || ND == 36) && D_v == 512) {
                typedef unsigned int mat4 __attribute__((ext_vector_type(4)));
#    pragma unroll
                for (int dc = 0; dc < D_v / 16; dc += 8) {
                    mat4 packed[4];
#    pragma unroll
                    for (int z = 0; z < 4; z++) {
                        const unsigned addr =
                            (unsigned)(size_t)(kbase + (dc + 2 * z) * 16 + (lane / 4) * KLD + (lane % 4) * 8);
                        asm volatile("ds_read_m32x16_b16 %0, %1" : "=v"(packed[z]) : "v"(addr) : "memory");
                    }
                    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#    pragma unroll
                    for (int z = 0; z < 4; z++) {
                        DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                        ((uint32_t*)&b0.x[0])[0] = packed[z][0];
                        ((uint32_t*)&b0.x[0])[1] = packed[z][1];
                        ((uint32_t*)&b1.x[0])[0] = packed[z][2];
                        ((uint32_t*)&b1.x[0])[1] = packed[z][3];
                        du_mma_sync(outacc[dc + 2 * z], pa[0], b0, outacc[dc + 2 * z]);
                        du_mma_sync(outacc[dc + 2 * z + 1], pa[0], b1, outacc[dc + 2 * z + 1]);
                    }
                }
            } else {
#    pragma unroll
                for (int dc = 0; dc < ND; dc += 2) {
                    if (dc * 16 >= D_v) continue;
#    pragma unroll
                    for (int i = 0; i < NT; i++) {
                        if (ck[i] > 0) {
                            DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                            int koff = 0;
#    pragma unroll
                            for (int j = 0; j < i; j++) koff += ck[j];
                            du_load_matrix_sync(b0, kbase + (size_t)koff * KLD + dc * 16, KLD);
                            du_mma_sync(outacc[dc], pa[i], b0, outacc[dc]);
                            if ((dc + 1) * 16 < D_v) {
                                du_load_matrix_sync(b1, kbase + (size_t)koff * KLD + (dc + 1) * 16, KLD);
                                du_mma_sync(outacc[dc + 1], pa[i], b1, outacc[dc + 1]);
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
    }

    // ---- Finalize KV-only LSE, then merge the sink only into output. ----
    // Keeping the online (m,l) state KV-only is important: deriving the
    // contract's KV-only LSE by subtracting exp(sink-m) from a sink-merged l
    // loses every KV contribution once the sink dominates by ~16 logits.
    if (lane_ok) {
        float out_scale = 0.0f;
        if (sink_val != sink_val || l != l) {
            out_scale = NAN;
        } else if (isinf(sink_val) && sink_val > 0.0f) {
            out_scale = 0.0f;
        } else if (l > 0.0f) {
            const float merged_m  = fmaxf(m, sink_val);
            const float kv_corr   = expf(m - merged_m);
            const float sink_mass = (sink_val > -INFINITY) ? expf(sink_val - merged_m) : 0.0f;
            out_scale             = kv_corr / (l * kv_corr + sink_mass + 1e-30f);
        }
        bf16* orow = out + ((size_t)t * H + h) * D_v;
        for (int dc = 0; dc < ND; dc++) {
            if (dc * 16 >= D_v) continue;
            const int col  = dc * 16 + g;
            const float p0 = outacc[dc].x[0] * out_scale;
            const float p1 = outacc[dc].x[1] * out_scale;
            const float p2 = outacc[dc].x[2] * out_scale;
            const float p3 = outacc[dc].x[3] * out_scale;
            // gfx938 quirk: after an MMAC the result register aliases the
            // bit-twiddling conversion's source, so saf(NaN) can read a
            // stale +0.0 bit pattern and flush NaN to zero.  A float
            // self-compare reads the live value; propagate a quiet NaN.
            if (col < D_v) orow[col] = (p0 != p0) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(p0);
            if (col + 4 < D_v) orow[col + 4] = (p1 != p1) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(p1);
            if (col + 8 < D_v) orow[col + 8] = (p2 != p2) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(p2);
            if (col + 12 < D_v) orow[col + 12] = (p3 != p3) ? (uint16_t)0x7FC0 : hip_dsa_numeric::bf16_rne(p3);
        }
        if ((lane >> 4) == 0) {
            if (l > 0.0f) {
                lse[(size_t)t * H + h] = m + logf(l);
            } else if (l != l) {
                lse[(size_t)t * H + h] = NAN;  // degenerate NaN input: keep NaN
            } else {
                lse[(size_t)t * H + h] = -INFINITY;
            }
        }
    }
}
// Native dtype forward; the online-softmax scalar path remains fp32 exactly
// as in the optimized bf16 kernel above.

#else
template<int ND, int NT, int TOPK = 0>
__global__ static void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_fwd_fused_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kv,
                                 const float* __restrict__ attn_sink, const int* __restrict__ idxs,
                                 const int* __restrict__ topk_length, bf16* __restrict__ out, float* __restrict__ lse,
                                 int T, int H, int D_rt, int D_v, int T_kv, int topk_rt, float softmax_scale) {}

#endif

// 64 heads x 64 keys. Each wave64 owns 16 heads and all 512 value channels.
// FlashMLA's NN32 layout pairs two MMACs with one 128-bit Q/K fragment.
// Four 4-KiB LDS slots carry three outstanding global-to-LDS transfers;
// the barrier before each consume also releases the previous slot.
// QK and PV reuse the same 16-KiB allocation. FP32 accumulation, BF16 RNE,
// KV-only LSE and sink-merged indexer LSE follow the hip_dsa contract.
// The host dispatch guarantees complete head groups and a 64-aligned prefix.

namespace hip_dsa_mla64 {
// Exact HIP round-to-nearest-even, including Inf and NaN payload handling.
__device__ __forceinline__ static uint16_t bf16_rne_bits(float f) { return hip_dsa_numeric::bf16_rne(f); }
__device__ __forceinline__ static __hip_bfloat16 bf16_rne(float f) { return __ushort_as_bfloat16(bf16_rne_bits(f)); }

#if defined(__gfx936__) || defined(__gfx938__)
using mla_f2 = float __attribute__((ext_vector_type(2)));
__device__ __forceinline__ static mla_f2 mla_pk_mul(mla_f2 a, mla_f2 b) {
    mla_f2 c;
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
    return c;
}
__device__ __forceinline__ static mla_f2 mla_pk_add(mla_f2 a, mla_f2 b) {
    mla_f2 c;
    asm volatile("v_pk_add_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
    return c;
}
__device__ __forceinline__ static mla_f2 mla_pk_fma(mla_f2 a, mla_f2 b, mla_f2 c) {
    mla_f2 d;
    asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}
__device__ __forceinline__ static mla_f2 mla_exp_pair(mla_f2 x) {
#    pragma clang fp contract(off)
    const float hi = __uint_as_float(0x3fb8aa3bu), lo = __uint_as_float(0x32a5705fu);
    const mla_f2 y = mla_pk_mul(x, (mla_f2){hi, hi});
    const mla_f2 n = {__builtin_rintf(y[0]), __builtin_rintf(y[1])};
    // First compensation retains scalar negate operands; packed negation would
    // cost two extra instructions. Pair the independent second FMA and finaladd.
    mla_f2 r       = {__builtin_fmaf(x[0], hi, -y[0]), __builtin_fmaf(x[1], hi, -y[1])};
    r              = mla_pk_fma(x, (mla_f2){lo, lo}, r);
    const mla_f2 f = mla_pk_add((mla_f2){y[0] - n[0], y[1] - n[1]}, r);
    mla_f2 z;
#    pragma unroll
    for (int i = 0; i < 2; i++) {
        int exponent;
        asm("v_cvt_i32_f32 %0, %1" : "=v"(exponent) : "v"(n[i]));
        float v = __builtin_amdgcn_ldexpf(__builtin_amdgcn_exp2f(f[i]), exponent);
        if (x[i] < __uint_as_float(0xc2ce8ed0u)) v = 0.f;
        // Online max ensures score-nm <= 0 or NaN; positive overflow is impossible.
        z[i] = v;
    }
    return z;
}
// Short sparse rows retain logits and probabilities in registers between
// QK and PV. Q fragments die before output accumulators become live.
template<int TOPK, int STRIDE = TOPK>
__global__ __launch_bounds__(256, 1) static void h64_short_two_pass(const bf* q, const bf* kv, const float* sink,
                                                                    const int* idx, const int* lens, bf* out, float* mx,
                                                                    float* lse, int H, int S, float scale) {
    constexpr int D = 512, NT = (TOPK + 63) / 64, NF = TOPK / 16;
    const int t = blockIdx.y, hb = blockIdx.x * 64, lane = threadIdx.x & 63;
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int r = lane & 15, g = lane >> 4, h = hb + w * 16 + r;
    const int valid = lens ? max(0, min(TOPK, lens[t])) : TOPK;
    __shared__ bf sm[8192];
    const unsigned long long ka = (unsigned long long)kv;
    const Desc kd               = {(int)ka, (int)((ka >> 32) | 0x40000000u), (int)0x80000000u, 0x00020000};
    // The specialized prefix is legal only when every physical suffix key
    // is invalid. Each wave reads the same suffix; the early exit is uniform.
    if constexpr (STRIDE > TOPK) {
        bool tail_active = false;
#    pragma unroll
        for (int pos = TOPK + lane; pos < STRIDE; pos += 64) {
            const int ix  = idx[(size_t)t * STRIDE + pos];
            tail_active  |= ix >= 0 && ix < S;
        }
        if (__ballot(tail_active)) return;
    }
    // Exact lane-XOR16 mapping from FlashMLA gfx93 fwd/utils.h:100.
    // This permutes raw32 bits and preserves every arithmetic operation.
    auto swap16 = [&](auto value) {
        if constexpr (TOPK == 128) {
            int bits = __builtin_bit_cast(int, value);
            bits     = __builtin_amdgcn_ds_swizzle(bits, 0x401F);
            return __builtin_bit_cast(decltype(value), bits);
        } else
            return __shfl_xor(value, 16);
    };
    CF score[NF];
    int rowkeys[NT];
    {
        AF qf[32];
        using U4 = unsigned int __attribute__((ext_vector_type(4)));
        // Same strided Q descriptor as FlashMLA gfx93 phase1.cuh:137-149.
        const unsigned long long qa = (unsigned long long)(q + ((size_t)t * H + hb) * D);
        const U4 qdesc = {(unsigned)qa, (unsigned)(qa >> 32) | ((D * 2) << 16) | 0x40000000u, 64u, 0x00020000u};
#    pragma unroll
        for (int j = 0; j < 16; j++) {
            U4 v;
            if constexpr (TOPK == 128) {
                using B8 = __bf16 __attribute__((ext_vector_type(8)));
                union {
                    B8 b;
                    U4 u;
                } loaded;
                loaded.b = __builtin_amdgcn_buffer_load_dwordx4(qdesc, w * 16 + r, (j * 32 + g * 8) * 2, false, false);
                v        = loaded.u;
            } else
                v = *(const U4*)(q + ((size_t)t * H + h) * D + j * 32 + g * 8);
            *(unsigned long long*)&qf[j * 2].x[0]     = ((unsigned long long*)&v)[0];
            *(unsigned long long*)&qf[j * 2 + 1].x[0] = ((unsigned long long*)&v)[1];
        }
#    pragma unroll
        for (int tile = 0; tile < NT; tile++) {
            const int BK       = tile * 64 + 64 <= TOPK ? 64 : 32;
            const int pos      = tile * 64 + lane;
            const int tile_key = pos < valid ? idx[(size_t)t * STRIDE + pos] : -1;
            rowkeys[tile]      = tile_key;
#    pragma unroll
            for (int k = 0; k < BK / 16; k++) du_fill_fragment(score[tile * 4 + k], 0.f);
            const bool active = __builtin_amdgcn_readfirstlane(__any(tile_key >= 0 && tile_key < S));
            if (active) {
                const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3);
                const int row = w * 16 + unsw / 32, col = unsw % 32;
                const int key = __shfl(tile_key, row);
                auto fetch    = [&](int j) {
                    const int off = key >= 0 && key < S ? key * (D * 2) + (j * 32 + col) * 2 : -1;
                    if (w < BK / 16) ld(kd, off, sm + (j % 4) * (BK * 32) + w * 512);
                };
#    pragma unroll
                for (int j = 0; j < 3; j++) fetch(j);
#    pragma unroll
                for (int j = 0; j < 16; j++) {
                    waitn(min(2, 15 - j));
                    if (j + 3 < 16) fetch(j + 3);
                    const int raw = (lane & 15) * 32 + (lane >> 4) * 8;
                    const int kb  = raw ^ ((raw >> 6) << 3);
                    if constexpr (TOPK == 128) {
                        U4 packed_k[4];
#    pragma unroll
                        for (int k = 0; k < 4; k++) {
                            const unsigned address = (unsigned)(size_t)(sm + (j % 4) * (BK * 32) + kb + 512 * k);
                            asm volatile("ds_read_b128 %0, %1" : "=v"(packed_k[k]) : "v"(address) : "memory");
                        }
#    pragma unroll
                        for (int k = 0; k < 4; k++) {
                            if (k == 0) asm volatile("s_waitcnt lgkmcnt(3)" ::: "memory");
                            if (k == 1) asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
                            if (k == 2) asm volatile("s_waitcnt lgkmcnt(1)" ::: "memory");
                            if (k == 3) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                            U4 v = packed_k[k];
                            BC b0, b1;
                            *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                            *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                            du_mma_sync(score[tile * 4 + k], qf[j * 2], b0, score[tile * 4 + k]);
                            du_mma_sync(score[tile * 4 + k], qf[j * 2 + 1], b1, score[tile * 4 + k]);
                        }
                    } else {
#    pragma unroll
                        for (int k = 0; k < BK / 16; k++) {
                            U4 v = *(const U4*)(sm + (j % 4) * (BK * 32) + kb + 512 * k);
                            BC b0, b1;
                            *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                            *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                            du_mma_sync(score[tile * 4 + k], qf[j * 2], b0, score[tile * 4 + k]);
                            du_mma_sync(score[tile * 4 + k], qf[j * 2 + 1], b1, score[tile * 4 + k]);
                        }
                    }
                }
                bar();
            }
#    pragma unroll
            for (int k = 0; k < BK / 16; k++) {
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    const int ix             = __shfl(tile_key, k * 16 + g + 4 * z);
                    score[tile * 4 + k].x[z] = ix >= 0 && ix < S ? score[tile * 4 + k].x[z] * scale : -INFINITY;
                }
            }
        }
    }
    float m = -INFINITY;
#    pragma unroll
    for (int k = 0; k < NF; k++) {
#    pragma unroll
        for (int z = 0; z < 4; z++) m = fmaxf(m, score[k].x[z]);
    }
    m       = fmaxf(m, swap16(m));
    m       = fmaxf(m, __shfl_xor(m, 32));
    float l = 0.f;
    AF pa[NF];
#    pragma unroll
    for (int k = 0; k < NF; k++) {
        unsigned short eb[4];
#    pragma unroll
        for (int z = 0; z < 4; z += 2) {
            const mla_f2 ex  = mla_exp_pair((mla_f2){score[k].x[z] - m, score[k].x[z + 1] - m});
            const float e0   = score[k].x[z] == -INFINITY ? 0.f : ex[0];
            const float e1   = score[k].x[z + 1] == -INFINITY ? 0.f : ex[1];
            l               += e0;
            l               += e1;
            eb[z]            = bf16_rne_bits(e0);
            eb[z + 1]        = bf16_rne_bits(e1);
        }
        unsigned a  = unsigned(eb[0]) | (unsigned(eb[1]) << 16);
        unsigned b  = unsigned(eb[2]) | (unsigned(eb[3]) << 16);
        unsigned ap = swap16(a), bp = swap16(b);
        unsigned lo    = (g & 1) ? (ap >> 16) | (a & 0xffff0000u) : (a & 0xffffu) | (ap << 16);
        unsigned hi    = (g & 1) ? (bp >> 16) | (b & 0xffff0000u) : (b & 0xffffu) | (bp << 16);
        unsigned other = __shfl_xor((g & 2) ? lo : hi, 32);
        if (g & 2)
            lo = other;
        else
            hi = other;
        ((unsigned*)&pa[k].x[0])[0] = lo;
        ((unsigned*)&pa[k].x[0])[1] = hi;
    }
    l              += swap16(l);
    l              += __shfl_xor(l, 32);
    const float sk  = sink ? sink[h] : -INFINITY;
    float factor    = 0.f;
    if (isnan(sk) || isnan(l))
        factor = NAN;
    else if (!(isinf(sk) && sk > 0) && l > 0) {
        float mm = fmaxf(m, sk), c = hip_dsa_numeric::softmax_exp(m - mm);
        factor = c / (l * c + ((sk > -INFINITY) ? hip_dsa_numeric::softmax_exp(sk - mm) : 0.f) + 1e-30f);
    }
    if (g == 0) {
        if (mx) mx[(size_t)t * H + h] = m;
        lse[(size_t)t * H + h] = isnan(l) ? NAN : (l > 0 ? m + logf(l) : -INFINITY);
    }
    CF acc[32];
#    pragma unroll
    for (int j = 0; j < 32; j++) du_fill_fragment(acc[j], 0.f);
#    pragma unroll
    for (int tile = 0; tile < NT; tile++) {
        const int BK       = tile * 64 + 64 <= TOPK ? 64 : 32;
        const int tile_key = rowkeys[tile];
        if (!__builtin_amdgcn_readfirstlane(__any(tile_key >= 0 && tile_key < S))) continue;
        int vk[4];
#    pragma unroll
        for (int k = 0; k < BK / 16; k++) vk[k] = __shfl(tile_key, k * 16 + lane / 4);
#    pragma unroll
        for (int k = 0; k < BK / 16; k++) asm volatile("" : "+v"(vk[k])::"memory");
        auto fetch = [&](int j) {
            int k = j % (BK / 16), c = j / (BK / 16), ix = vk[k];
            int off = ix >= 0 && ix < S ? ix * (D * 2) + (c * 128 + w * 32 + lane % 4 * 8) * 2 : -1;
            ld(kd, off, sm + (j % 4) * 2048 + w * 512);
        };
#    pragma unroll
        for (int j = 0; j < 3; j++) fetch(j);
#    pragma unroll
        for (int j = 0; j < BK / 4; j++) {
            waitn(min(2, BK / 4 - 1 - j));
            if (j + 3 < BK / 4) fetch(j + 3);
            int vl = lane;
            asm volatile("" : "+v"(vl));
            if constexpr (TOPK == 128) {
                using Read4 = unsigned __attribute__((ext_vector_type(4)));
                Read4 packed_v[4];
#    pragma unroll
                for (int dc = 0; dc < 4; dc++) {
                    const unsigned address = (unsigned)(size_t)(sm + (j % 4) * 2048 + dc * 512 + vl * 8);
                    asm volatile("ds_read_m32x16_b16_alt %0, %1" : "=v"(packed_v[dc]) : "v"(address) : "memory");
                }
#    pragma unroll
                for (int dc = 0; dc < 4; dc++) {
                    if (dc == 0) asm volatile("s_waitcnt lgkmcnt(3)" ::: "memory");
                    if (dc == 1) asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
                    if (dc == 2) asm volatile("s_waitcnt lgkmcnt(1)" ::: "memory");
                    if (dc == 3) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                    auto v = packed_v[dc];
                    BR b0, b1;
                    *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                    *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                    du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2], pa[tile * 4 + j % (BK / 16)], b0,
                                acc[(j / (BK / 16)) * 8 + dc * 2]);
                    du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2 + 1], pa[tile * 4 + j % (BK / 16)], b1,
                                acc[(j / (BK / 16)) * 8 + dc * 2 + 1]);
                }
            } else {
#    pragma unroll
                for (int dc = 0; dc < 4; dc++) {
                    auto vp = (__attribute__((address_space(3))) __fp16*)(sm + (j % 4) * 2048 + dc * 512 + vl * 8);
                    auto v  = __builtin_amdgcn_ds_read_m32x16f16_alt(vp, 0);
                    BR b0, b1;
                    *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                    *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                    du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2], pa[tile * 4 + j % (BK / 16)], b0,
                                acc[(j / (BK / 16)) * 8 + dc * 2]);
                    du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2 + 1], pa[tile * 4 + j % (BK / 16)], b1,
                                acc[(j / (BK / 16)) * 8 + dc * 2 + 1]);
                }
            }
        }
        bar();
    }
#    pragma unroll
    for (int j = 0; j < 16; j++) {
        unsigned v[4], x[4], y[4];
#    pragma unroll
        for (int z = 0; z < 4; z++)
            v[z] = hip_dsa_numeric::bf16_pair(acc[j * 2].x[z] * factor, acc[j * 2 + 1].x[z] * factor);
#    pragma unroll
        for (int z = 0; z < 4; z++) {
            const unsigned peer = swap16(v[z ^ 1]);
            x[z]                = ((g ^ z) & 1) ? peer : v[z];
        }
#    pragma unroll
        for (int z = 0; z < 4; z++) {
            const unsigned peer = __shfl_xor(x[z ^ 2], 32);
            y[z]                = ((g ^ z) & 2) ? peer : x[z];
        }
        using Store = unsigned __attribute__((ext_vector_type(4)));
        __builtin_nontemporal_store((Store){y[0], y[1], y[2], y[3]},
                                    (Store*)(out + ((size_t)t * H + h) * 512 + j * 32 + g * 8));
    }
}

template<int D, bool MASK16 = false, bool PARTIAL_HEADS = false, int NT = 256>
__global__ __launch_bounds__(NT, 1) static void h64(const bf* q, const bf* kv, const float* sink, const int* idx,
                                                    const int* lens, bf* out, float* mx, float* lse, float* li, int T,
                                                    int H, int S, int K, int IK, float scale) {
    const int t = blockIdx.y, hb = blockIdx.x * 64, lane = threadIdx.x & 63,
              w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), r = lane & 15, g = lane >> 4, h = hb + w * 16 + r;
    const int valid = lens ? max(0, min(K, lens[t])) : K;
    __shared__ bf sm[8192];  // 4 slots: three loads ahead, one barrier per consumer
    const unsigned long long ka = (unsigned long long)kv;
    const Desc kd               = {(int)ka, (int)((ka >> 32) | 0x40000000u), (int)0x80000000u, 0x00020000};
    auto swap16                 = [&](auto value) {
        if constexpr (D == 576) {
            int bits = __builtin_bit_cast(int, value);
            bits     = __builtin_amdgcn_ds_swizzle(bits, 0x401F);
            return __builtin_bit_cast(decltype(value), bits);
        } else
            return __shfl_xor(value, 16);
    };
    AF qf[D / 16];
    // Adjacent NN32 Q fragments occupy one aligned 16-byte vector.
    // Load the pair together; preserve every BF16 bit and the MMAC order.
    using QVec = unsigned int __attribute__((ext_vector_type(4)));
#    pragma unroll
    for (int j = 0; j < D / 32; j++) {
        QVec v;
        if constexpr (D == 576) {
            const unsigned long long qa = (unsigned long long)(q + ((size_t)t * H + hb) * D);
            const QVec qdesc            = {(unsigned)qa, (unsigned)(qa >> 32) | ((D * 2) << 16) | 0x40000000u,
                                           (unsigned)(PARTIAL_HEADS ? H - hb : 64), 0x00020000u};
            using B8                    = __bf16 __attribute__((ext_vector_type(8)));
            union {
                B8 b;
                QVec u;
            } loaded;
            loaded.b = __builtin_amdgcn_buffer_load_dwordx4(qdesc, w * 16 + r, (j * 32 + g * 8) * 2, false, false);
            v        = loaded.u;
        } else {
            v = (!PARTIAL_HEADS || h < H) ? *(const QVec*)(q + ((size_t)t * H + h) * D + j * 32 + g * 8)
                                          : QVec{0, 0, 0, 0};
            asm volatile("" : "+v"(v)::"memory");
        }
        *(unsigned long long*)&qf[j * 2].x[0]     = ((unsigned long long*)&v)[0];
        *(unsigned long long*)&qf[j * 2 + 1].x[0] = ((unsigned long long*)&v)[1];
    }
    CF acc[32];
#    pragma unroll
    for (int j = 0; j < 32; j++) du_fill_fragment(acc[j], 0.f);
    float m = -INFINITY, l = 0, mi = -INFINITY, il = 0;
    for (int base = 0; base < valid; base += 64) {
        // Every wave examines the same 64 keys, so skipping an empty tile is
        // CTA-uniform.
        const int tile_pos                  = base + lane;
        const int tile_key                  = tile_pos < valid ? idx[(size_t)t * K + tile_pos] : -1;
        const unsigned long long valid_mask = MASK16 ? __ballot(tile_key >= 0 && tile_key < S) : 0;
        if constexpr (MASK16) {
            if (!valid_mask) continue;
        } else if (!__builtin_amdgcn_readfirstlane(__any(tile_key >= 0 && tile_key < S)))
            continue;
        const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3), row = w * 16 + unsw / 32, col = unsw % 32;
        // Every lane already holds one of this query's 64 tile indices.
        // Reuse it for K, score masking and V; invalid and tail slots remain -1.
        int key     = __shfl(tile_key, row);
        int key2    = __shfl(tile_key, row + 32);
        auto fetchk = [&](int j) {
            int off = (key >= 0 && key < S) ? key * (D * 2) + (j * 32 + col) * 2 : -1;
            ld(kd, off, sm + (j % 4) * 2048 + w * 512);
            if constexpr (NT == 128) {
                const int off2 = (key2 >= 0 && key2 < S) ? key2 * (D * 2) + (j * 32 + col) * 2 : -1;
                ld(kd, off2, sm + (j % 4) * 2048 + (w + 2) * 512);
            }
        };
        CF sc[4];
#    pragma unroll
        for (int k = 0; k < 4; k++) du_fill_fragment(sc[k], 0.f);
#    pragma unroll
        for (int j = 0; j < 3; j++) fetchk(j);
#    pragma unroll
        for (int j = 0; j < D / 32; j++) {
            // Three gathers are initially in flight. Complete the oldest before
            // each consume; the CTA barrier also releases the previous ring slot.
            // The fourth slot accepts the next gather without overwriting a slot
            // another wave can still be reading. Drain the remaining transfers
            // at the end of the tile.
            const int rem = D / 32 - 1 - j;
            if constexpr (NT == 128) {
                if (rem >= 2) {
                    asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
                    bar();
                } else
                    waitn(2 * rem);
            } else
                waitn(min(2, rem));
            if (j + 3 < D / 32) fetchk(j + 3);  // Slot j+3 is free after the preceding barrier.
            // The LDS swizzle only uses bits 6..8 of the element index, and every
            // k step adds 512 elements (a multiple of 64), so the swizzle result is
            // identical for all k. Hoist one per-lane base address and let the k
            // loop use immediate offsets instead of recomputing the whole address
            // for every read (VALU was this kernel's top pipe).
            const int kb_raw = (lane & 15) * 32 + (lane >> 4) * 8;
            const int kbase  = kb_raw ^ ((kb_raw >> 6) << 3);
#    pragma unroll
            for (int k = 0; k < 4; k++) {
                using U4 = unsigned int __attribute__((ext_vector_type(4)));
                U4 v     = *(const U4*)(sm + (j % 4) * 2048 + kbase + 512 * k);
                BC b0, b1;
                *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                du_mma_sync(sc[k], qf[j * 2], b0, sc[k]);
                du_mma_sync(sc[k], qf[j * 2 + 1], b1, sc[k]);
            }
        }

        bar();
        float nm = m;
#    pragma unroll
        for (int k = 0; k < 4; k++) {
            const unsigned valid16 = (unsigned)(valid_mask >> (k * 16));
#    pragma unroll
            for (int z = 0; z < 4; z++) {
                bool slot_valid;
                if constexpr (MASK16)
                    slot_valid = (valid16 >> (g + 4 * z)) & 1u;
                else {
                    const int ix = __shfl(tile_key, k * 16 + g + 4 * z);
                    slot_valid   = ix >= 0 && ix < S;
                }
                sc[k].x[z] = slot_valid ? sc[k].x[z] * scale : -INFINITY;
                nm         = fmaxf(nm, sc[k].x[z]);
            }
        }
        nm         = fmaxf(nm, swap16(nm));
        nm         = fmaxf(nm, __shfl_xor(nm, 32));
        float corr = l > 0 ? hip_dsa_numeric::softmax_exp(m - nm) : 0, sum = 0;
        // Transpose the 4x4 probability fragment within each head's four lanes.
        // This removes the shared-memory round trip and its two CTA barriers.
        AF pa[4];
#    pragma unroll
        for (int k = 0; k < 4; k++) {
            unsigned short eb[4];
            if constexpr (D == 512 || (D == 576)) {
// nm includes every score and the prior running maximum. Thus each
// difference is nonpositive or NaN, including infinite-score inputs.
// Pair independent FP32 exp arithmetic; keep the original sum order.
#    pragma unroll
                for (int z = 0; z < 4; z += 2) {
                    const mla_f2 ex  = mla_exp_pair((mla_f2){sc[k].x[z] - nm, sc[k].x[z + 1] - nm});
                    const float e0   = sc[k].x[z] == -INFINITY ? 0.f : ex[0];
                    const float e1   = sc[k].x[z + 1] == -INFINITY ? 0.f : ex[1];
                    sum             += e0;
                    sum             += e1;
                    eb[z]            = bf16_rne_bits(e0);
                    eb[z + 1]        = bf16_rne_bits(e1);
                }
            } else {
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    float e  = sc[k].x[z] == -INFINITY ? 0 : hip_dsa_numeric::softmax_exp(sc[k].x[z] - nm);
                    sum     += e;
                    eb[z]    = bf16_rne_bits(e);
                }
            }
            unsigned a  = unsigned(eb[0]) | (unsigned(eb[1]) << 16);
            unsigned b  = unsigned(eb[2]) | (unsigned(eb[3]) << 16);
            unsigned ap = swap16(a), bp = swap16(b);
            unsigned lo    = (g & 1) ? (ap >> 16) | (a & 0xffff0000u) : (a & 0xffffu) | (ap << 16);
            unsigned hi    = (g & 1) ? (bp >> 16) | (b & 0xffff0000u) : (b & 0xffffu) | (bp << 16);
            unsigned other = __shfl_xor((g & 2) ? lo : hi, 32);
            if (g & 2)
                lo = other;
            else
                hi = other;
            ((unsigned*)&pa[k].x[0])[0] = lo;
            ((unsigned*)&pa[k].x[0])[1] = hi;
        }
        sum += swap16(sum);
        sum += __shfl_xor(sum, 32);
        l    = l * corr + sum;
        m    = nm;
        // The host dispatch requires an indexer prefix aligned to 64 keys.
        if (base < IK) {
            mi = m;
            il = l;
        }
#    pragma unroll
        // Packed FP32 multiplies preserve independent per-element rounding.
        // Pair adjacent elements within each accumulator fragment; no reduction
        // order changes, extra BF16 conversion, or cross-head/query operations.
        for (int j = 0; j < 32; j++) {
#    pragma unroll
            for (int z = 0; z < 4; z += 2) {
                float2 x = {acc[j].x[z], acc[j].x[z + 1]};
                float2 c = {corr, corr}, result;
                asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(result) : "v"(x), "v"(c));
                acc[j].x[z]     = result.x;
                acc[j].x[z + 1] = result.y;
            }
        }
        int vk[4];
#    pragma unroll
        for (int k = 0; k < 4; k++) { vk[k] = __shfl(tile_key, k * 16 + lane / 4); }
#    pragma unroll
        for (int k = 0; k < 4; k++) asm volatile("" : "+v"(vk[k])::"memory");
        auto fetchv = [&](int j) {
            int k = j % 4, c = j / 4, ix = vk[k];
            int off = (ix >= 0 && ix < S) ? ix * (D * 2) + (c * 128 + w * 32 + lane % 4 * 8) * 2 : -1;
            ld(kd, off, sm + (j % 4) * 2048 + w * 512);
            if constexpr (NT == 128) {
                const int off2 = (ix >= 0 && ix < S) ? ix * (D * 2) + (c * 128 + (w + 2) * 32 + lane % 4 * 8) * 2 : -1;
                ld(kd, off2, sm + (j % 4) * 2048 + (w + 2) * 512);
            }
        };
#    pragma unroll
        for (int j = 0; j < 3; j++) fetchv(j);
#    pragma unroll
        for (int j = 0; j < 16; j++) {
            const int rem = 15 - j;
            if constexpr (NT == 128) {
                if (rem >= 2) {
                    asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
                    bar();
                } else
                    waitn(2 * rem);
            } else
                waitn(min(2, rem));
            if (j + 3 < 16) fetchv(j + 3);
            // Keep V address lifetimes within this pipeline stage.
            int vl = lane;
            asm volatile("" : "+v"(vl));
#    pragma unroll
            for (int dc = 0; dc < 4; dc++) {
                auto vp = (__attribute__((address_space(3))) __fp16*)(sm + (j % 4) * 2048 + dc * 512 + vl * 8);
                auto v  = __builtin_amdgcn_ds_read_m32x16f16_alt(vp, 0);
                BR b0, b1;
                *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                du_mma_sync(acc[(j / 4) * 8 + dc * 2], pa[j % 4], b0, acc[(j / 4) * 8 + dc * 2]);
                du_mma_sync(acc[(j / 4) * 8 + dc * 2 + 1], pa[j % 4], b1, acc[(j / 4) * 8 + dc * 2 + 1]);
            }
        }
        bar();
    }
    if constexpr (PARTIAL_HEADS) {
        if (h >= H) return;
    }
    {
        float sk = sink ? sink[h] : -INFINITY, factor = 0;
        if (isnan(sk) || isnan(l))
            factor = NAN;
        else if (!(isinf(sk) && sk > 0) && l > 0) {
            float mm = fmaxf(m, sk), c = hip_dsa_numeric::softmax_exp(m - mm);
            factor = c / (l * c + ((sk > -INFINITY) ? hip_dsa_numeric::softmax_exp(sk - mm) : 0.f) + 1e-30f);
        }
        // Exchange already-rounded BF16 pairs across the four lanes of each head.
        // This gives each lane eight adjacent outputs for one aligned 128-bit store.
#    pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned v[4], x[4], y[4];
#    pragma unroll
            for (int z = 0; z < 4; z++)
                v[z] = hip_dsa_numeric::bf16_pair(acc[j * 2].x[z] * factor, acc[j * 2 + 1].x[z] * factor);
#    pragma unroll
            for (int z = 0; z < 4; z++) {
                const unsigned peer = __shfl_xor(v[z ^ 1], 16);
                x[z]                = ((g ^ z) & 1) ? peer : v[z];
            }
#    pragma unroll
            for (int z = 0; z < 4; z++) {
                const unsigned peer = __shfl_xor(x[z ^ 2], 32);
                y[z]                = ((g ^ z) & 2) ? peer : x[z];
            }
            using Store = unsigned __attribute__((ext_vector_type(4)));
            __builtin_nontemporal_store((Store){y[0], y[1], y[2], y[3]},
                                        (Store*)(out + ((size_t)t * H + h) * 512 + j * 32 + g * 8));
        }
        if (g == 0) {
            if (mx) mx[(size_t)t * H + h] = m;
            lse[(size_t)t * H + h] = isnan(l) ? NAN : (l > 0 ? m + logf(l) : -INFINITY);
            if (li) {
                float a = il > 0 ? mi + logf(il) : -INFINITY, b = fmaxf(a, sk);
                li[(size_t)t * H + h] =
                    isnan(sk) ? NAN
                              : (b == -INFINITY ? -INFINITY
                                                : (isinf(b) ? b
                                                            : b + logf(hip_dsa_numeric::softmax_exp(a - b) +
                                                                       hip_dsa_numeric::softmax_exp(sk - b))));
            }
        }
    }
}

// Physical K=160 ends in 32 slots. Keep the existing 64-key kernel above
// untouched for other shapes. Peeling the final tile avoids loop-merge
// register spills, and skips only physical padding, preserving softmax groups.
__global__ __launch_bounds__(256, 1) static void h64_tail160(const bf* q, const bf* kv, const float* sink,
                                                             const int* idx, const int* lens, bf* out, float* mx,
                                                             float* lse, float* li, int T, int H, int S, int K, int IK,
                                                             float scale) {
    constexpr int D       = 512;
    constexpr bool MASK16 = false, TAIL32 = true;
    const int t = blockIdx.y, hb = blockIdx.x * 64, lane = threadIdx.x & 63,
              w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), r = lane & 15, g = lane >> 4, h = hb + w * 16 + r;
    const int valid = lens ? max(0, min(K, lens[t])) : K;
    __shared__ bf sm[8192];  // 4 slots: three loads ahead, one barrier per consumer
    const unsigned long long ka = (unsigned long long)kv;
    const Desc kd               = {(int)ka, (int)((ka >> 32) | 0x40000000u), (int)0x80000000u, 0x00020000};
    AF qf[D / 16];
    // Adjacent NN32 Q fragments occupy one aligned 16-byte vector.
    // Load the pair together; preserve every BF16 bit and the MMAC order.
    using QVec = unsigned int __attribute__((ext_vector_type(4)));
#    pragma unroll
    for (int j = 0; j < D / 32; j++) {
        QVec v = *(const QVec*)(q + ((size_t)t * H + h) * D + j * 32 + g * 8);
        asm volatile("" : "+v"(v)::"memory");
        *(unsigned long long*)&qf[j * 2].x[0]     = ((unsigned long long*)&v)[0];
        *(unsigned long long*)&qf[j * 2 + 1].x[0] = ((unsigned long long*)&v)[1];
    }
    CF acc[32];
#    pragma unroll
    for (int j = 0; j < 32; j++) du_fill_fragment(acc[j], 0.f);
    float m = -INFINITY, l = 0, mi = -INFINITY, il = 0;
    for (int base = 0; base < valid && (!TAIL32 || base < 128); base += 64) {
        constexpr int BK = 64;
        // Every wave examines the same 64 keys, so skipping an empty tile is
        // CTA-uniform.
        const int tile_pos                  = base + lane;
        const int tile_key                  = lane < BK && tile_pos < valid ? idx[(size_t)t * K + tile_pos] : -1;
        const unsigned long long valid_mask = MASK16 ? __ballot(tile_key >= 0 && tile_key < S) : 0;
        if constexpr (MASK16) {
            if (!valid_mask) continue;
        } else if (!__builtin_amdgcn_readfirstlane(__any(tile_key >= 0 && tile_key < S)))
            continue;
        const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3), row = w * 16 + unsw / 32, col = unsw % 32;
        // Every lane already holds one of this query's 64 tile indices.
        // Reuse it for K, score masking and V; invalid and tail slots remain -1.
        int key     = __shfl(tile_key, row);
        auto fetchk = [&](int j) {
            int off = (key >= 0 && key < S) ? key * (D * 2) + (j * 32 + col) * 2 : -1;
            if (w < BK / 16) ld(kd, off, sm + (j % 4) * (BK * 32) + w * 512);
        };
        CF sc[BK / 16];
#    pragma unroll
        for (int k = 0; k < BK / 16; k++) du_fill_fragment(sc[k], 0.f);
#    pragma unroll
        for (int j = 0; j < 3; j++) fetchk(j);
#    pragma unroll
        for (int j = 0; j < D / 32; j++) {
            const int rem = D / 32 - 1 - j;
            waitn(min(2, rem));
            if (j + 3 < D / 32) fetchk(j + 3);  // The preceding barrier releases the previous ring slot.
            // The LDS swizzle only uses bits 6..8 of the element index, and every
            // k step adds 512 elements (a multiple of 64), so the swizzle result is
            // identical for all k. Hoist one per-lane base address and let the k
            // loop use immediate offsets instead of recomputing the whole address
            // for every read (VALU was this kernel's top pipe).
            const int kb_raw = (lane & 15) * 32 + (lane >> 4) * 8;
            const int kbase  = kb_raw ^ ((kb_raw >> 6) << 3);
#    pragma unroll
            for (int k = 0; k < BK / 16; k++) {
                using U4 = unsigned int __attribute__((ext_vector_type(4)));
                U4 v     = *(const U4*)(sm + (j % 4) * (BK * 32) + kbase + 512 * k);
                BC b0, b1;
                *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                du_mma_sync(sc[k], qf[j * 2], b0, sc[k]);
                du_mma_sync(sc[k], qf[j * 2 + 1], b1, sc[k]);
            }
        }

        bar();
        float nm = m;
#    pragma unroll
        for (int k = 0; k < BK / 16; k++) {
            const unsigned valid16 = (unsigned)(valid_mask >> (k * 16));
#    pragma unroll
            for (int z = 0; z < 4; z++) {
                bool slot_valid;
                if constexpr (MASK16)
                    slot_valid = (valid16 >> (g + 4 * z)) & 1u;
                else {
                    const int ix = __shfl(tile_key, k * 16 + g + 4 * z);
                    slot_valid   = ix >= 0 && ix < S;
                }
                sc[k].x[z] = slot_valid ? sc[k].x[z] * scale : -INFINITY;
                nm         = fmaxf(nm, sc[k].x[z]);
            }
        }
        nm         = fmaxf(nm, __shfl_xor(nm, 16));
        nm         = fmaxf(nm, __shfl_xor(nm, 32));
        float corr = l > 0 ? hip_dsa_numeric::softmax_exp(m - nm) : 0, sum = 0;
        // Transpose the 4x4 probability fragment within each head's four lanes.
        // This removes the shared-memory round trip and its two CTA barriers.
        AF pa[BK / 16];
#    pragma unroll
        for (int k = 0; k < BK / 16; k++) {
            unsigned short eb[4];
#    pragma unroll
            for (int z = 0; z < 4; z += 2) {
                const mla_f2 ex  = mla_exp_pair((mla_f2){sc[k].x[z] - nm, sc[k].x[z + 1] - nm});
                const float e0   = sc[k].x[z] == -INFINITY ? 0.f : ex[0];
                const float e1   = sc[k].x[z + 1] == -INFINITY ? 0.f : ex[1];
                sum             += e0;
                sum             += e1;
                eb[z]            = bf16_rne_bits(e0);
                eb[z + 1]        = bf16_rne_bits(e1);
            }
            unsigned a  = unsigned(eb[0]) | (unsigned(eb[1]) << 16);
            unsigned b  = unsigned(eb[2]) | (unsigned(eb[3]) << 16);
            unsigned ap = __shfl_xor(a, 16), bp = __shfl_xor(b, 16);
            unsigned lo    = (g & 1) ? (ap >> 16) | (a & 0xffff0000u) : (a & 0xffffu) | (ap << 16);
            unsigned hi    = (g & 1) ? (bp >> 16) | (b & 0xffff0000u) : (b & 0xffffu) | (bp << 16);
            unsigned other = __shfl_xor((g & 2) ? lo : hi, 32);
            if (g & 2)
                lo = other;
            else
                hi = other;
            ((unsigned*)&pa[k].x[0])[0] = lo;
            ((unsigned*)&pa[k].x[0])[1] = hi;
        }
        sum += __shfl_xor(sum, 16);
        sum += __shfl_xor(sum, 32);
        l    = l * corr + sum;
        m    = nm;
        // The host dispatch requires an indexer prefix aligned to 64 keys.
        if (base < IK) {
            mi = m;
            il = l;
        }
#    pragma unroll
        // Packed FP32 multiplies preserve independent per-element rounding.
        // Pair adjacent elements within each accumulator fragment; no reduction
        // order changes, extra BF16 conversion, or cross-head/query operations.
        for (int j = 0; j < 32; j++) {
#    pragma unroll
            for (int z = 0; z < 4; z += 2) {
                float2 x = {acc[j].x[z], acc[j].x[z + 1]};
                float2 c = {corr, corr}, result;
                asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(result) : "v"(x), "v"(c));
                acc[j].x[z]     = result.x;
                acc[j].x[z + 1] = result.y;
            }
        }
        int vk[BK / 16];
#    pragma unroll
        for (int k = 0; k < BK / 16; k++) { vk[k] = __shfl(tile_key, k * 16 + lane / 4); }
#    pragma unroll
        for (int k = 0; k < BK / 16; k++) asm volatile("" : "+v"(vk[k])::"memory");
        auto fetchv = [&](int j) {
            int k = j % (BK / 16), c = j / (BK / 16), ix = vk[k];
            int off = (ix >= 0 && ix < S) ? ix * (D * 2) + (c * 128 + w * 32 + lane % 4 * 8) * 2 : -1;
            ld(kd, off, sm + (j % 4) * 2048 + w * 512);
        };
#    pragma unroll
        for (int j = 0; j < 3; j++) fetchv(j);
#    pragma unroll
        for (int j = 0; j < BK / 4; j++) {
            const int rem = BK / 4 - 1 - j;
            waitn(min(2, rem));
            if (j + 3 < BK / 4) fetchv(j + 3);
            // Keep V address lifetimes within this pipeline stage.
            int vl = lane;
            asm volatile("" : "+v"(vl));
#    pragma unroll
            for (int dc = 0; dc < 4; dc++) {
                auto vp = (__attribute__((address_space(3))) __fp16*)(sm + (j % 4) * 2048 + dc * 512 + vl * 8);
                auto v  = __builtin_amdgcn_ds_read_m32x16f16_alt(vp, 0);
                BR b0, b1;
                *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2], pa[j % (BK / 16)], b0,
                            acc[(j / (BK / 16)) * 8 + dc * 2]);
                du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2 + 1], pa[j % (BK / 16)], b1,
                            acc[(j / (BK / 16)) * 8 + dc * 2 + 1]);
            }
        }
        bar();
    }
    if constexpr (TAIL32) {
        if (valid > 128) {
            const int base = 128;
            do {
                constexpr int BK = 32;
                // Every wave examines the same 64 keys, so skipping an empty tile is
                // CTA-uniform.
                const int tile_pos = base + lane;
                const int tile_key = lane < BK && tile_pos < valid ? idx[(size_t)t * K + tile_pos] : -1;
                const unsigned long long valid_mask = MASK16 ? __ballot(tile_key >= 0 && tile_key < S) : 0;
                if constexpr (MASK16) {
                    if (!valid_mask) break;
                } else if (!__builtin_amdgcn_readfirstlane(__any(tile_key >= 0 && tile_key < S)))
                    break;
                const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3), row = w * 16 + unsw / 32,
                          col = unsw % 32;
                // Every lane already holds one of this query's 64 tile indices.
                // Reuse it for K, score masking and V; invalid and tail slots remain -1.
                int key     = __shfl(tile_key, row);
                auto fetchk = [&](int j) {
                    int off = (key >= 0 && key < S) ? key * (D * 2) + (j * 32 + col) * 2 : -1;
                    if (w < BK / 16) ld(kd, off, sm + (j % 4) * (BK * 32) + w * 512);
                };
                CF sc[BK / 16];
#    pragma unroll
                for (int k = 0; k < BK / 16; k++) du_fill_fragment(sc[k], 0.f);
#    pragma unroll
                for (int j = 0; j < 3; j++) fetchk(j);
#    pragma unroll
                for (int j = 0; j < D / 32; j++) {
                    const int rem = D / 32 - 1 - j;
                    waitn(min(2, rem));
                    if (j + 3 < D / 32) fetchk(j + 3);  // The preceding barrier releases the previous ring slot.
                    // The LDS swizzle only uses bits 6..8 of the element index, and every
                    // k step adds 512 elements (a multiple of 64), so the swizzle result is
                    // identical for all k. Hoist one per-lane base address and let the k
                    // loop use immediate offsets instead of recomputing the whole address
                    // for every read (VALU was this kernel's top pipe).
                    const int kb_raw = (lane & 15) * 32 + (lane >> 4) * 8;
                    const int kbase  = kb_raw ^ ((kb_raw >> 6) << 3);
#    pragma unroll
                    for (int k = 0; k < BK / 16; k++) {
                        using U4 = unsigned int __attribute__((ext_vector_type(4)));
                        U4 v     = *(const U4*)(sm + (j % 4) * (BK * 32) + kbase + 512 * k);
                        BC b0, b1;
                        *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                        *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                        du_mma_sync(sc[k], qf[j * 2], b0, sc[k]);
                        du_mma_sync(sc[k], qf[j * 2 + 1], b1, sc[k]);
                    }
                }

                bar();
                float nm = m;
#    pragma unroll
                for (int k = 0; k < BK / 16; k++) {
                    const unsigned valid16 = (unsigned)(valid_mask >> (k * 16));
#    pragma unroll
                    for (int z = 0; z < 4; z++) {
                        bool slot_valid;
                        if constexpr (MASK16)
                            slot_valid = (valid16 >> (g + 4 * z)) & 1u;
                        else {
                            const int ix = __shfl(tile_key, k * 16 + g + 4 * z);
                            slot_valid   = ix >= 0 && ix < S;
                        }
                        sc[k].x[z] = slot_valid ? sc[k].x[z] * scale : -INFINITY;
                        nm         = fmaxf(nm, sc[k].x[z]);
                    }
                }
                nm         = fmaxf(nm, __shfl_xor(nm, 16));
                nm         = fmaxf(nm, __shfl_xor(nm, 32));
                float corr = l > 0 ? hip_dsa_numeric::softmax_exp(m - nm) : 0, sum = 0;
                // Transpose the 4x4 probability fragment within each head's four lanes.
                // This removes the shared-memory round trip and its two CTA barriers.
                AF pa[BK / 16];
#    pragma unroll
                for (int k = 0; k < BK / 16; k++) {
                    unsigned short eb[4];
#    pragma unroll
                    for (int z = 0; z < 4; z += 2) {
                        const mla_f2 ex  = mla_exp_pair((mla_f2){sc[k].x[z] - nm, sc[k].x[z + 1] - nm});
                        const float e0   = sc[k].x[z] == -INFINITY ? 0.f : ex[0];
                        const float e1   = sc[k].x[z + 1] == -INFINITY ? 0.f : ex[1];
                        sum             += e0;
                        sum             += e1;
                        eb[z]            = bf16_rne_bits(e0);
                        eb[z + 1]        = bf16_rne_bits(e1);
                    }
                    unsigned a  = unsigned(eb[0]) | (unsigned(eb[1]) << 16);
                    unsigned b  = unsigned(eb[2]) | (unsigned(eb[3]) << 16);
                    unsigned ap = __shfl_xor(a, 16), bp = __shfl_xor(b, 16);
                    unsigned lo    = (g & 1) ? (ap >> 16) | (a & 0xffff0000u) : (a & 0xffffu) | (ap << 16);
                    unsigned hi    = (g & 1) ? (bp >> 16) | (b & 0xffff0000u) : (b & 0xffffu) | (bp << 16);
                    unsigned other = __shfl_xor((g & 2) ? lo : hi, 32);
                    if (g & 2)
                        lo = other;
                    else
                        hi = other;
                    ((unsigned*)&pa[k].x[0])[0] = lo;
                    ((unsigned*)&pa[k].x[0])[1] = hi;
                }
                sum += __shfl_xor(sum, 16);
                sum += __shfl_xor(sum, 32);
                l    = l * corr + sum;
                m    = nm;
                // The host dispatch requires an indexer prefix aligned to 64 keys.
                if (base < IK) {
                    mi = m;
                    il = l;
                }
#    pragma unroll
                // Packed FP32 multiplies preserve independent per-element rounding.
                // Pair adjacent elements within each accumulator fragment; no reduction
                // order changes, extra BF16 conversion, or cross-head/query operations.
                for (int j = 0; j < 32; j++) {
#    pragma unroll
                    for (int z = 0; z < 4; z += 2) {
                        float2 x = {acc[j].x[z], acc[j].x[z + 1]};
                        float2 c = {corr, corr}, result;
                        asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(result) : "v"(x), "v"(c));
                        acc[j].x[z]     = result.x;
                        acc[j].x[z + 1] = result.y;
                    }
                }
                int vk[BK / 16];
#    pragma unroll
                for (int k = 0; k < BK / 16; k++) { vk[k] = __shfl(tile_key, k * 16 + lane / 4); }
#    pragma unroll
                for (int k = 0; k < BK / 16; k++) asm volatile("" : "+v"(vk[k])::"memory");
                auto fetchv = [&](int j) {
                    int k = j % (BK / 16), c = j / (BK / 16), ix = vk[k];
                    int off = (ix >= 0 && ix < S) ? ix * (D * 2) + (c * 128 + w * 32 + lane % 4 * 8) * 2 : -1;
                    ld(kd, off, sm + (j % 4) * 2048 + w * 512);
                };
#    pragma unroll
                for (int j = 0; j < 3; j++) fetchv(j);
#    pragma unroll
                for (int j = 0; j < BK / 4; j++) {
                    const int rem = BK / 4 - 1 - j;
                    waitn(min(2, rem));
                    if (j + 3 < BK / 4) fetchv(j + 3);
                    // Keep V address lifetimes within this pipeline stage.
                    int vl = lane;
                    asm volatile("" : "+v"(vl));
#    pragma unroll
                    for (int dc = 0; dc < 4; dc++) {
                        auto vp = (__attribute__((address_space(3))) __fp16*)(sm + (j % 4) * 2048 + dc * 512 + vl * 8);
                        auto v  = __builtin_amdgcn_ds_read_m32x16f16_alt(vp, 0);
                        BR b0, b1;
                        *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                        *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                        du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2], pa[j % (BK / 16)], b0,
                                    acc[(j / (BK / 16)) * 8 + dc * 2]);
                        du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2 + 1], pa[j % (BK / 16)], b1,
                                    acc[(j / (BK / 16)) * 8 + dc * 2 + 1]);
                    }
                }
                bar();
            } while (false);
        }
    }
    {
        float sk = sink ? sink[h] : -INFINITY, factor = 0;
        if (isnan(sk) || isnan(l))
            factor = NAN;
        else if (!(isinf(sk) && sk > 0) && l > 0) {
            float mm = fmaxf(m, sk), c = hip_dsa_numeric::softmax_exp(m - mm);
            factor = c / (l * c + ((sk > -INFINITY) ? hip_dsa_numeric::softmax_exp(sk - mm) : 0.f) + 1e-30f);
        }
        // Exchange already-rounded BF16 pairs across the four lanes of each head.
        // This gives each lane eight adjacent outputs for one aligned 128-bit store.
#    pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned v[4], x[4], y[4];
#    pragma unroll
            for (int z = 0; z < 4; z++)
                v[z] = hip_dsa_numeric::bf16_pair(acc[j * 2].x[z] * factor, acc[j * 2 + 1].x[z] * factor);
#    pragma unroll
            for (int z = 0; z < 4; z++) {
                const unsigned peer = __shfl_xor(v[z ^ 1], 16);
                x[z]                = ((g ^ z) & 1) ? peer : v[z];
            }
#    pragma unroll
            for (int z = 0; z < 4; z++) {
                const unsigned peer = __shfl_xor(x[z ^ 2], 32);
                y[z]                = ((g ^ z) & 2) ? peer : x[z];
            }
            using Store = unsigned __attribute__((ext_vector_type(4)));
            __builtin_nontemporal_store((Store){y[0], y[1], y[2], y[3]},
                                        (Store*)(out + ((size_t)t * H + h) * 512 + j * 32 + g * 8));
        }
        if (g == 0) {
            if (mx) mx[(size_t)t * H + h] = m;
            lse[(size_t)t * H + h] = isnan(l) ? NAN : (l > 0 ? m + logf(l) : -INFINITY);
            if (li) {
                float a = il > 0 ? mi + logf(il) : -INFINITY, b = fmaxf(a, sk);
                li[(size_t)t * H + h] =
                    isnan(sk) ? NAN
                              : (b == -INFINITY ? -INFINITY
                                                : (isinf(b) ? b
                                                            : b + logf(hip_dsa_numeric::softmax_exp(a - b) +
                                                                       hip_dsa_numeric::softmax_exp(sk - b))));
            }
        }
    }
}

// Each wave checks the same immutable row suffix, so the result is CTA-uniform.
// Padding may contain arbitrary integers: both negative and >=S mean invalid.
__device__ __forceinline__ static bool tail256_is_short(const int* idx, const int* lens, int K, int S) {
    const int t = blockIdx.y, lane = threadIdx.x & 63;
    const int n = lens ? max(0, min(K, lens[t])) : K;
    bool bad    = false;
#    pragma unroll
    for (int pos = 160 + lane; pos < 256; pos += 64) {
        if (pos < n) {
            const int key  = idx[(size_t)t * K + pos];
            bad           |= (key >= 0 && key < S);
        }
    }
    return __ballot(bad) == 0;
}
__global__ __launch_bounds__(256, 1) static void h64_tail256(const bf* q, const bf* kv, const float* sink,
                                                             const int* idx, const int* lens, bf* out, float* mx,
                                                             float* lse, float* li, int T, int H, int S, int K, int IK,
                                                             float scale) {
    __shared__ bf sm[8192];  // 4 slots: three loads ahead, one barrier per consumer
    if (tail256_is_short(idx, lens, K, S)) {
        constexpr int D       = 512;
        constexpr bool MASK16 = false;
        constexpr bool TAIL32 = true;

        const int t = blockIdx.y, hb = blockIdx.x * 64, lane = threadIdx.x & 63,
                  w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), r = lane & 15, g = lane >> 4,
                  h     = hb + w * 16 + r;
        const int valid = lens ? max(0, min(K, lens[t])) : K;

        const unsigned long long ka = (unsigned long long)kv;
        const Desc kd               = {(int)ka, (int)((ka >> 32) | 0x40000000u), (int)0x80000000u, 0x00020000};
        AF qf[D / 16];
        // Adjacent NN32 Q fragments occupy one aligned 16-byte vector.
        // Load the pair together; preserve every BF16 bit and the MMAC order.
        using QVec = unsigned int __attribute__((ext_vector_type(4)));
#    pragma unroll
        for (int j = 0; j < D / 32; j++) {
            QVec v = *(const QVec*)(q + ((size_t)t * H + h) * D + j * 32 + g * 8);
            asm volatile("" : "+v"(v)::"memory");
            *(unsigned long long*)&qf[j * 2].x[0]     = ((unsigned long long*)&v)[0];
            *(unsigned long long*)&qf[j * 2 + 1].x[0] = ((unsigned long long*)&v)[1];
        }
        CF acc[32];
#    pragma unroll
        for (int j = 0; j < 32; j++) du_fill_fragment(acc[j], 0.f);
        float m = -INFINITY, l = 0, mi = -INFINITY, il = 0;
        for (int base = 0; base < valid && (!TAIL32 || base < 128); base += 64) {
            constexpr int BK = 64;
            // Every wave examines the same 64 keys, so skipping an empty tile is
            // CTA-uniform.
            const int tile_pos                  = base + lane;
            const int tile_key                  = lane < BK && tile_pos < valid ? idx[(size_t)t * K + tile_pos] : -1;
            const unsigned long long valid_mask = MASK16 ? __ballot(tile_key >= 0 && tile_key < S) : 0;
            if constexpr (MASK16) {
                if (!valid_mask) continue;
            } else if (!__builtin_amdgcn_readfirstlane(__any(tile_key >= 0 && tile_key < S)))
                continue;
            const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3), row = w * 16 + unsw / 32, col = unsw % 32;
            // Every lane already holds one of this query's 64 tile indices.
            // Reuse it for K, score masking and V; invalid and tail slots remain -1.
            int key     = __shfl(tile_key, row);
            auto fetchk = [&](int j) {
                int off = (key >= 0 && key < S) ? key * (D * 2) + (j * 32 + col) * 2 : -1;
                if (w < BK / 16) ld(kd, off, sm + (j % 4) * (BK * 32) + w * 512);
            };
            CF sc[BK / 16];
#    pragma unroll
            for (int k = 0; k < BK / 16; k++) du_fill_fragment(sc[k], 0.f);
#    pragma unroll
            for (int j = 0; j < 3; j++) fetchk(j);
#    pragma unroll
            for (int j = 0; j < D / 32; j++) {
                const int rem = D / 32 - 1 - j;
                waitn(min(2, rem));
                if (j + 3 < D / 32) fetchk(j + 3);  // The preceding barrier releases the previous ring slot.
                // The LDS swizzle only uses bits 6..8 of the element index, and every
                // k step adds 512 elements (a multiple of 64), so the swizzle result is
                // identical for all k. Hoist one per-lane base address and let the k
                // loop use immediate offsets instead of recomputing the whole address
                // for every read (VALU was this kernel's top pipe).
                const int kb_raw = (lane & 15) * 32 + (lane >> 4) * 8;
                const int kbase  = kb_raw ^ ((kb_raw >> 6) << 3);
#    pragma unroll
                for (int k = 0; k < BK / 16; k++) {
                    using U4 = unsigned int __attribute__((ext_vector_type(4)));
                    U4 v     = *(const U4*)(sm + (j % 4) * (BK * 32) + kbase + 512 * k);
                    BC b0, b1;
                    *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                    *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                    du_mma_sync(sc[k], qf[j * 2], b0, sc[k]);
                    du_mma_sync(sc[k], qf[j * 2 + 1], b1, sc[k]);
                }
            }

            bar();
            float nm = m;
#    pragma unroll
            for (int k = 0; k < BK / 16; k++) {
                const unsigned valid16 = (unsigned)(valid_mask >> (k * 16));
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    bool slot_valid;
                    if constexpr (MASK16)
                        slot_valid = (valid16 >> (g + 4 * z)) & 1u;
                    else {
                        const int ix = __shfl(tile_key, k * 16 + g + 4 * z);
                        slot_valid   = ix >= 0 && ix < S;
                    }
                    sc[k].x[z] = slot_valid ? sc[k].x[z] * scale : -INFINITY;
                    nm         = fmaxf(nm, sc[k].x[z]);
                }
            }
            nm         = fmaxf(nm, __shfl_xor(nm, 16));
            nm         = fmaxf(nm, __shfl_xor(nm, 32));
            float corr = l > 0 ? hip_dsa_numeric::softmax_exp(m - nm) : 0, sum = 0;
            // Transpose the 4x4 probability fragment within each head's four lanes.
            // This removes the shared-memory round trip and its two CTA barriers.
            AF pa[BK / 16];
#    pragma unroll
            for (int k = 0; k < BK / 16; k++) {
                unsigned short eb[4];
#    pragma unroll
                for (int z = 0; z < 4; z += 2) {
                    const mla_f2 ex  = mla_exp_pair((mla_f2){sc[k].x[z] - nm, sc[k].x[z + 1] - nm});
                    const float e0   = sc[k].x[z] == -INFINITY ? 0.f : ex[0];
                    const float e1   = sc[k].x[z + 1] == -INFINITY ? 0.f : ex[1];
                    sum             += e0;
                    sum             += e1;
                    eb[z]            = bf16_rne_bits(e0);
                    eb[z + 1]        = bf16_rne_bits(e1);
                }
                unsigned a  = unsigned(eb[0]) | (unsigned(eb[1]) << 16);
                unsigned b  = unsigned(eb[2]) | (unsigned(eb[3]) << 16);
                unsigned ap = __shfl_xor(a, 16), bp = __shfl_xor(b, 16);
                unsigned lo    = (g & 1) ? (ap >> 16) | (a & 0xffff0000u) : (a & 0xffffu) | (ap << 16);
                unsigned hi    = (g & 1) ? (bp >> 16) | (b & 0xffff0000u) : (b & 0xffffu) | (bp << 16);
                unsigned other = __shfl_xor((g & 2) ? lo : hi, 32);
                if (g & 2)
                    lo = other;
                else
                    hi = other;
                ((unsigned*)&pa[k].x[0])[0] = lo;
                ((unsigned*)&pa[k].x[0])[1] = hi;
            }
            sum += __shfl_xor(sum, 16);
            sum += __shfl_xor(sum, 32);
            l    = l * corr + sum;
            m    = nm;
            // The host dispatch requires an indexer prefix aligned to 64 keys.
            if (base < IK) {
                mi = m;
                il = l;
            }
#    pragma unroll
            // Packed FP32 multiplies preserve independent per-element rounding.
            // Pair adjacent elements within each accumulator fragment; no reduction
            // order changes, extra BF16 conversion, or cross-head/query operations.
            for (int j = 0; j < 32; j++) {
#    pragma unroll
                for (int z = 0; z < 4; z += 2) {
                    float2 x = {acc[j].x[z], acc[j].x[z + 1]};
                    float2 c = {corr, corr}, result;
                    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(result) : "v"(x), "v"(c));
                    acc[j].x[z]     = result.x;
                    acc[j].x[z + 1] = result.y;
                }
            }
            int vk[BK / 16];
#    pragma unroll
            for (int k = 0; k < BK / 16; k++) { vk[k] = __shfl(tile_key, k * 16 + lane / 4); }
#    pragma unroll
            for (int k = 0; k < BK / 16; k++) asm volatile("" : "+v"(vk[k])::"memory");
            auto fetchv = [&](int j) {
                int k = j % (BK / 16), c = j / (BK / 16), ix = vk[k];
                int off = (ix >= 0 && ix < S) ? ix * (D * 2) + (c * 128 + w * 32 + lane % 4 * 8) * 2 : -1;
                ld(kd, off, sm + (j % 4) * 2048 + w * 512);
            };
#    pragma unroll
            for (int j = 0; j < 3; j++) fetchv(j);
#    pragma unroll
            for (int j = 0; j < BK / 4; j++) {
                const int rem = BK / 4 - 1 - j;
                waitn(min(2, rem));
                if (j + 3 < BK / 4) fetchv(j + 3);
                // Keep V address lifetimes within this pipeline stage.
                int vl = lane;
                asm volatile("" : "+v"(vl));
#    pragma unroll
                for (int dc = 0; dc < 4; dc++) {
                    auto vp = (__attribute__((address_space(3))) __fp16*)(sm + (j % 4) * 2048 + dc * 512 + vl * 8);
                    auto v  = __builtin_amdgcn_ds_read_m32x16f16_alt(vp, 0);
                    BR b0, b1;
                    *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                    *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                    du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2], pa[j % (BK / 16)], b0,
                                acc[(j / (BK / 16)) * 8 + dc * 2]);
                    du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2 + 1], pa[j % (BK / 16)], b1,
                                acc[(j / (BK / 16)) * 8 + dc * 2 + 1]);
                }
            }
            bar();
        }
        if constexpr (TAIL32) {
            if (valid > 128) {
                const int base = 128;
                do {
                    constexpr int BK = 32;
                    // Every wave examines the same 64 keys, so skipping an empty tile is
                    // CTA-uniform.
                    const int tile_pos = base + lane;
                    const int tile_key = lane < BK && tile_pos < valid ? idx[(size_t)t * K + tile_pos] : -1;
                    const unsigned long long valid_mask = MASK16 ? __ballot(tile_key >= 0 && tile_key < S) : 0;
                    if constexpr (MASK16) {
                        if (!valid_mask) break;
                    } else if (!__builtin_amdgcn_readfirstlane(__any(tile_key >= 0 && tile_key < S)))
                        break;
                    const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3), row = w * 16 + unsw / 32,
                              col = unsw % 32;
                    // Every lane already holds one of this query's 64 tile indices.
                    // Reuse it for K, score masking and V; invalid and tail slots remain -1.
                    int key     = __shfl(tile_key, row);
                    auto fetchk = [&](int j) {
                        int off = (key >= 0 && key < S) ? key * (D * 2) + (j * 32 + col) * 2 : -1;
                        if (w < BK / 16) ld(kd, off, sm + (j % 4) * (BK * 32) + w * 512);
                    };
                    CF sc[BK / 16];
#    pragma unroll
                    for (int k = 0; k < BK / 16; k++) du_fill_fragment(sc[k], 0.f);
#    pragma unroll
                    for (int j = 0; j < 3; j++) fetchk(j);
#    pragma unroll
                    for (int j = 0; j < D / 32; j++) {
                        const int rem = D / 32 - 1 - j;
                        waitn(min(2, rem));
                        if (j + 3 < D / 32) fetchk(j + 3);  // The preceding barrier releases the previous ring slot.
                        // The LDS swizzle only uses bits 6..8 of the element index, and every
                        // k step adds 512 elements (a multiple of 64), so the swizzle result is
                        // identical for all k. Hoist one per-lane base address and let the k
                        // loop use immediate offsets instead of recomputing the whole address
                        // for every read (VALU was this kernel's top pipe).
                        const int kb_raw = (lane & 15) * 32 + (lane >> 4) * 8;
                        const int kbase  = kb_raw ^ ((kb_raw >> 6) << 3);
#    pragma unroll
                        for (int k = 0; k < BK / 16; k++) {
                            using U4 = unsigned int __attribute__((ext_vector_type(4)));
                            U4 v     = *(const U4*)(sm + (j % 4) * (BK * 32) + kbase + 512 * k);
                            BC b0, b1;
                            *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                            *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                            du_mma_sync(sc[k], qf[j * 2], b0, sc[k]);
                            du_mma_sync(sc[k], qf[j * 2 + 1], b1, sc[k]);
                        }
                    }

                    bar();
                    float nm = m;
#    pragma unroll
                    for (int k = 0; k < BK / 16; k++) {
                        const unsigned valid16 = (unsigned)(valid_mask >> (k * 16));
#    pragma unroll
                        for (int z = 0; z < 4; z++) {
                            bool slot_valid;
                            if constexpr (MASK16)
                                slot_valid = (valid16 >> (g + 4 * z)) & 1u;
                            else {
                                const int ix = __shfl(tile_key, k * 16 + g + 4 * z);
                                slot_valid   = ix >= 0 && ix < S;
                            }
                            sc[k].x[z] = slot_valid ? sc[k].x[z] * scale : -INFINITY;
                            nm         = fmaxf(nm, sc[k].x[z]);
                        }
                    }
                    nm         = fmaxf(nm, __shfl_xor(nm, 16));
                    nm         = fmaxf(nm, __shfl_xor(nm, 32));
                    float corr = l > 0 ? hip_dsa_numeric::softmax_exp(m - nm) : 0, sum = 0;
                    // Transpose the 4x4 probability fragment within each head's four lanes.
                    // This removes the shared-memory round trip and its two CTA barriers.
                    AF pa[BK / 16];
#    pragma unroll
                    for (int k = 0; k < BK / 16; k++) {
                        unsigned short eb[4];
#    pragma unroll
                        for (int z = 0; z < 4; z += 2) {
                            const mla_f2 ex  = mla_exp_pair((mla_f2){sc[k].x[z] - nm, sc[k].x[z + 1] - nm});
                            const float e0   = sc[k].x[z] == -INFINITY ? 0.f : ex[0];
                            const float e1   = sc[k].x[z + 1] == -INFINITY ? 0.f : ex[1];
                            sum             += e0;
                            sum             += e1;
                            eb[z]            = bf16_rne_bits(e0);
                            eb[z + 1]        = bf16_rne_bits(e1);
                        }
                        unsigned a  = unsigned(eb[0]) | (unsigned(eb[1]) << 16);
                        unsigned b  = unsigned(eb[2]) | (unsigned(eb[3]) << 16);
                        unsigned ap = __shfl_xor(a, 16), bp = __shfl_xor(b, 16);
                        unsigned lo    = (g & 1) ? (ap >> 16) | (a & 0xffff0000u) : (a & 0xffffu) | (ap << 16);
                        unsigned hi    = (g & 1) ? (bp >> 16) | (b & 0xffff0000u) : (b & 0xffffu) | (bp << 16);
                        unsigned other = __shfl_xor((g & 2) ? lo : hi, 32);
                        if (g & 2)
                            lo = other;
                        else
                            hi = other;
                        ((unsigned*)&pa[k].x[0])[0] = lo;
                        ((unsigned*)&pa[k].x[0])[1] = hi;
                    }
                    sum += __shfl_xor(sum, 16);
                    sum += __shfl_xor(sum, 32);
                    l    = l * corr + sum;
                    m    = nm;
                    // The host dispatch requires an indexer prefix aligned to 64 keys.
                    if (base < IK) {
                        mi = m;
                        il = l;
                    }
#    pragma unroll
                    // Packed FP32 multiplies preserve independent per-element rounding.
                    // Pair adjacent elements within each accumulator fragment; no reduction
                    // order changes, extra BF16 conversion, or cross-head/query operations.
                    for (int j = 0; j < 32; j++) {
#    pragma unroll
                        for (int z = 0; z < 4; z += 2) {
                            float2 x = {acc[j].x[z], acc[j].x[z + 1]};
                            float2 c = {corr, corr}, result;
                            asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(result) : "v"(x), "v"(c));
                            acc[j].x[z]     = result.x;
                            acc[j].x[z + 1] = result.y;
                        }
                    }
                    int vk[BK / 16];
#    pragma unroll
                    for (int k = 0; k < BK / 16; k++) { vk[k] = __shfl(tile_key, k * 16 + lane / 4); }
#    pragma unroll
                    for (int k = 0; k < BK / 16; k++) asm volatile("" : "+v"(vk[k])::"memory");
                    auto fetchv = [&](int j) {
                        int k = j % (BK / 16), c = j / (BK / 16), ix = vk[k];
                        int off = (ix >= 0 && ix < S) ? ix * (D * 2) + (c * 128 + w * 32 + lane % 4 * 8) * 2 : -1;
                        ld(kd, off, sm + (j % 4) * 2048 + w * 512);
                    };
#    pragma unroll
                    for (int j = 0; j < 3; j++) fetchv(j);
#    pragma unroll
                    for (int j = 0; j < BK / 4; j++) {
                        const int rem = BK / 4 - 1 - j;
                        waitn(min(2, rem));
                        if (j + 3 < BK / 4) fetchv(j + 3);
                        // Keep V address lifetimes within this pipeline stage.
                        int vl = lane;
                        asm volatile("" : "+v"(vl));
#    pragma unroll
                        for (int dc = 0; dc < 4; dc++) {
                            auto vp =
                                (__attribute__((address_space(3))) __fp16*)(sm + (j % 4) * 2048 + dc * 512 + vl * 8);
                            auto v = __builtin_amdgcn_ds_read_m32x16f16_alt(vp, 0);
                            BR b0, b1;
                            *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                            *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                            du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2], pa[j % (BK / 16)], b0,
                                        acc[(j / (BK / 16)) * 8 + dc * 2]);
                            du_mma_sync(acc[(j / (BK / 16)) * 8 + dc * 2 + 1], pa[j % (BK / 16)], b1,
                                        acc[(j / (BK / 16)) * 8 + dc * 2 + 1]);
                        }
                    }
                    bar();
                } while (false);
            }
        }
        {
            float sk = sink ? sink[h] : -INFINITY, factor = 0;
            if (isnan(sk) || isnan(l))
                factor = NAN;
            else if (!(isinf(sk) && sk > 0) && l > 0) {
                float mm = fmaxf(m, sk), c = hip_dsa_numeric::softmax_exp(m - mm);
                factor = c / (l * c + ((sk > -INFINITY) ? hip_dsa_numeric::softmax_exp(sk - mm) : 0.f) + 1e-30f);
            }
            // Exchange already-rounded BF16 pairs across the four lanes of each head.
            // This gives each lane eight adjacent outputs for one aligned 128-bit store.
#    pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned v[4], x[4], y[4];
#    pragma unroll
                for (int z = 0; z < 4; z++)
                    v[z] = hip_dsa_numeric::bf16_pair(acc[j * 2].x[z] * factor, acc[j * 2 + 1].x[z] * factor);
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    const unsigned peer = __shfl_xor(v[z ^ 1], 16);
                    x[z]                = ((g ^ z) & 1) ? peer : v[z];
                }
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    const unsigned peer = __shfl_xor(x[z ^ 2], 32);
                    y[z]                = ((g ^ z) & 2) ? peer : x[z];
                }
                using Store = unsigned __attribute__((ext_vector_type(4)));
                __builtin_nontemporal_store((Store){y[0], y[1], y[2], y[3]},
                                            (Store*)(out + ((size_t)t * H + h) * 512 + j * 32 + g * 8));
            }
            if (g == 0) {
                if (mx) mx[(size_t)t * H + h] = m;
                lse[(size_t)t * H + h] = isnan(l) ? NAN : (l > 0 ? m + logf(l) : -INFINITY);
                if (li) {
                    float a = il > 0 ? mi + logf(il) : -INFINITY, b = fmaxf(a, sk);
                    li[(size_t)t * H + h] =
                        isnan(sk) ? NAN
                                  : (b == -INFINITY ? -INFINITY
                                                    : (isinf(b) ? b
                                                                : b + logf(hip_dsa_numeric::softmax_exp(a - b) +
                                                                           hip_dsa_numeric::softmax_exp(sk - b))));
                }
            }
        }

        return;
    }
    {
        constexpr int D       = 512;
        constexpr bool MASK16 = false;

        const int t = blockIdx.y, hb = blockIdx.x * 64, lane = threadIdx.x & 63,
                  w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), r = lane & 15, g = lane >> 4,
                  h     = hb + w * 16 + r;
        const int valid = lens ? max(0, min(K, lens[t])) : K;

        const unsigned long long ka = (unsigned long long)kv;
        const Desc kd               = {(int)ka, (int)((ka >> 32) | 0x40000000u), (int)0x80000000u, 0x00020000};
        AF qf[D / 16];
        // Adjacent NN32 Q fragments occupy one aligned 16-byte vector.
        // Load the pair together; preserve every BF16 bit and the MMAC order.
        using QVec = unsigned int __attribute__((ext_vector_type(4)));
#    pragma unroll
        for (int j = 0; j < D / 32; j++) {
            QVec v = *(const QVec*)(q + ((size_t)t * H + h) * D + j * 32 + g * 8);
            asm volatile("" : "+v"(v)::"memory");
            *(unsigned long long*)&qf[j * 2].x[0]     = ((unsigned long long*)&v)[0];
            *(unsigned long long*)&qf[j * 2 + 1].x[0] = ((unsigned long long*)&v)[1];
        }
        CF acc[32];
#    pragma unroll
        for (int j = 0; j < 32; j++) du_fill_fragment(acc[j], 0.f);
        float m = -INFINITY, l = 0, mi = -INFINITY, il = 0;
        for (int base = 0; base < valid; base += 64) {
            // Every wave examines the same 64 keys, so skipping an empty tile is
            // CTA-uniform.
            const int tile_pos                  = base + lane;
            const int tile_key                  = tile_pos < valid ? idx[(size_t)t * K + tile_pos] : -1;
            const unsigned long long valid_mask = MASK16 ? __ballot(tile_key >= 0 && tile_key < S) : 0;
            if constexpr (MASK16) {
                if (!valid_mask) continue;
            } else if (!__builtin_amdgcn_readfirstlane(__any(tile_key >= 0 && tile_key < S)))
                continue;
            const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3), row = w * 16 + unsw / 32, col = unsw % 32;
            // Every lane already holds one of this query's 64 tile indices.
            // Reuse it for K, score masking and V; invalid and tail slots remain -1.
            int key     = __shfl(tile_key, row);
            auto fetchk = [&](int j) {
                int off = (key >= 0 && key < S) ? key * (D * 2) + (j * 32 + col) * 2 : -1;
                ld(kd, off, sm + (j % 4) * 2048 + w * 512);
            };
            CF sc[4];
#    pragma unroll
            for (int k = 0; k < 4; k++) du_fill_fragment(sc[k], 0.f);
#    pragma unroll
            for (int j = 0; j < 3; j++) fetchk(j);
#    pragma unroll
            for (int j = 0; j < D / 32; j++) {
                const int rem = D / 32 - 1 - j;
                waitn(min(2, rem));
                if (j + 3 < D / 32) fetchk(j + 3);  // The preceding barrier releases the previous ring slot.
                // The LDS swizzle only uses bits 6..8 of the element index, and every
                // k step adds 512 elements (a multiple of 64), so the swizzle result is
                // identical for all k. Hoist one per-lane base address and let the k
                // loop use immediate offsets instead of recomputing the whole address
                // for every read (VALU was this kernel's top pipe).
                const int kb_raw = (lane & 15) * 32 + (lane >> 4) * 8;
                const int kbase  = kb_raw ^ ((kb_raw >> 6) << 3);
#    pragma unroll
                for (int k = 0; k < 4; k++) {
                    using U4 = unsigned int __attribute__((ext_vector_type(4)));
                    U4 v     = *(const U4*)(sm + (j % 4) * 2048 + kbase + 512 * k);
                    BC b0, b1;
                    *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                    *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                    du_mma_sync(sc[k], qf[j * 2], b0, sc[k]);
                    du_mma_sync(sc[k], qf[j * 2 + 1], b1, sc[k]);
                }
            }

            bar();
            float nm = m;
#    pragma unroll
            for (int k = 0; k < 4; k++) {
                const unsigned valid16 = (unsigned)(valid_mask >> (k * 16));
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    bool slot_valid;
                    if constexpr (MASK16)
                        slot_valid = (valid16 >> (g + 4 * z)) & 1u;
                    else {
                        const int ix = __shfl(tile_key, k * 16 + g + 4 * z);
                        slot_valid   = ix >= 0 && ix < S;
                    }
                    sc[k].x[z] = slot_valid ? sc[k].x[z] * scale : -INFINITY;
                    nm         = fmaxf(nm, sc[k].x[z]);
                }
            }
            nm         = fmaxf(nm, __shfl_xor(nm, 16));
            nm         = fmaxf(nm, __shfl_xor(nm, 32));
            float corr = l > 0 ? hip_dsa_numeric::softmax_exp(m - nm) : 0, sum = 0;
            // Transpose the 4x4 probability fragment within each head's four lanes.
            // This removes the shared-memory round trip and its two CTA barriers.
            AF pa[4];
#    pragma unroll
            for (int k = 0; k < 4; k++) {
                unsigned short eb[4];
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    float e  = sc[k].x[z] == -INFINITY ? 0 : hip_dsa_numeric::softmax_exp(sc[k].x[z] - nm);
                    sum     += e;
                    eb[z]    = bf16_rne_bits(e);
                }
                unsigned a  = unsigned(eb[0]) | (unsigned(eb[1]) << 16);
                unsigned b  = unsigned(eb[2]) | (unsigned(eb[3]) << 16);
                unsigned ap = __shfl_xor(a, 16), bp = __shfl_xor(b, 16);
                unsigned lo    = (g & 1) ? (ap >> 16) | (a & 0xffff0000u) : (a & 0xffffu) | (ap << 16);
                unsigned hi    = (g & 1) ? (bp >> 16) | (b & 0xffff0000u) : (b & 0xffffu) | (bp << 16);
                unsigned other = __shfl_xor((g & 2) ? lo : hi, 32);
                if (g & 2)
                    lo = other;
                else
                    hi = other;
                ((unsigned*)&pa[k].x[0])[0] = lo;
                ((unsigned*)&pa[k].x[0])[1] = hi;
            }
            sum += __shfl_xor(sum, 16);
            sum += __shfl_xor(sum, 32);
            l    = l * corr + sum;
            m    = nm;
            // The host dispatch requires an indexer prefix aligned to 64 keys.
            if (base < IK) {
                mi = m;
                il = l;
            }
#    pragma unroll
            // Packed FP32 multiplies preserve independent per-element rounding.
            // Pair adjacent elements within each accumulator fragment; no reduction
            // order changes, extra BF16 conversion, or cross-head/query operations.
            for (int j = 0; j < 32; j++) {
#    pragma unroll
                for (int z = 0; z < 4; z += 2) {
                    float2 x = {acc[j].x[z], acc[j].x[z + 1]};
                    float2 c = {corr, corr}, result;
                    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(result) : "v"(x), "v"(c));
                    acc[j].x[z]     = result.x;
                    acc[j].x[z + 1] = result.y;
                }
            }
            int vk[4];
#    pragma unroll
            for (int k = 0; k < 4; k++) { vk[k] = __shfl(tile_key, k * 16 + lane / 4); }
#    pragma unroll
            for (int k = 0; k < 4; k++) asm volatile("" : "+v"(vk[k])::"memory");
            auto fetchv = [&](int j) {
                int k = j % 4, c = j / 4, ix = vk[k];
                int off = (ix >= 0 && ix < S) ? ix * (D * 2) + (c * 128 + w * 32 + lane % 4 * 8) * 2 : -1;
                ld(kd, off, sm + (j % 4) * 2048 + w * 512);
            };
#    pragma unroll
            for (int j = 0; j < 3; j++) fetchv(j);
#    pragma unroll
            for (int j = 0; j < 16; j++) {
                const int rem = 15 - j;
                waitn(min(2, rem));
                if (j + 3 < 16) fetchv(j + 3);
                // Keep V address lifetimes within this pipeline stage.
                int vl = lane;
                asm volatile("" : "+v"(vl));
#    pragma unroll
                for (int dc = 0; dc < 4; dc++) {
                    auto vp = (__attribute__((address_space(3))) __fp16*)(sm + (j % 4) * 2048 + dc * 512 + vl * 8);
                    auto v  = __builtin_amdgcn_ds_read_m32x16f16_alt(vp, 0);
                    BR b0, b1;
                    *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                    *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                    du_mma_sync(acc[(j / 4) * 8 + dc * 2], pa[j % 4], b0, acc[(j / 4) * 8 + dc * 2]);
                    du_mma_sync(acc[(j / 4) * 8 + dc * 2 + 1], pa[j % 4], b1, acc[(j / 4) * 8 + dc * 2 + 1]);
                }
            }
            bar();
        }
        {
            float sk = sink ? sink[h] : -INFINITY, factor = 0;
            if (isnan(sk) || isnan(l))
                factor = NAN;
            else if (!(isinf(sk) && sk > 0) && l > 0) {
                float mm = fmaxf(m, sk), c = hip_dsa_numeric::softmax_exp(m - mm);
                factor = c / (l * c + ((sk > -INFINITY) ? hip_dsa_numeric::softmax_exp(sk - mm) : 0.f) + 1e-30f);
            }
            // Exchange already-rounded BF16 pairs across the four lanes of each head.
            // This gives each lane eight adjacent outputs for one aligned 128-bit store.
#    pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned v[4], x[4], y[4];
#    pragma unroll
                for (int z = 0; z < 4; z++)
                    v[z] = hip_dsa_numeric::bf16_pair(acc[j * 2].x[z] * factor, acc[j * 2 + 1].x[z] * factor);
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    const unsigned peer = __shfl_xor(v[z ^ 1], 16);
                    x[z]                = ((g ^ z) & 1) ? peer : v[z];
                }
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    const unsigned peer = __shfl_xor(x[z ^ 2], 32);
                    y[z]                = ((g ^ z) & 2) ? peer : x[z];
                }
                using Store = unsigned __attribute__((ext_vector_type(4)));
                __builtin_nontemporal_store((Store){y[0], y[1], y[2], y[3]},
                                            (Store*)(out + ((size_t)t * H + h) * 512 + j * 32 + g * 8));
            }
            if (g == 0) {
                if (mx) mx[(size_t)t * H + h] = m;
                lse[(size_t)t * H + h] = isnan(l) ? NAN : (l > 0 ? m + logf(l) : -INFINITY);
                if (li) {
                    float a = il > 0 ? mi + logf(il) : -INFINITY, b = fmaxf(a, sk);
                    li[(size_t)t * H + h] =
                        isnan(sk) ? NAN
                                  : (b == -INFINITY ? -INFINITY
                                                    : (isinf(b) ? b
                                                                : b + logf(hip_dsa_numeric::softmax_exp(a - b) +
                                                                           hip_dsa_numeric::softmax_exp(sk - b))));
                }
            }
        }
    }
}
__global__ __launch_bounds__(256, 1) static void h64_tail256_long_rows(const bf* q, const bf* kv, const float* sink,
                                                                       const int* idx, const int* lens, bf* out,
                                                                       float* mx, float* lse, float* li, int T, int H,
                                                                       int S, int K, int IK, float scale) {
    __shared__ bf sm[8192];  // 4 slots: three loads ahead, one barrier per consumer
    if (tail256_is_short(idx, lens, K, S)) return;
    {
        constexpr int D       = 512;
        constexpr bool MASK16 = false;

        const int t = blockIdx.y, hb = blockIdx.x * 64, lane = threadIdx.x & 63,
                  w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), r = lane & 15, g = lane >> 4,
                  h     = hb + w * 16 + r;
        const int valid = lens ? max(0, min(K, lens[t])) : K;

        const unsigned long long ka = (unsigned long long)kv;
        const Desc kd               = {(int)ka, (int)((ka >> 32) | 0x40000000u), (int)0x80000000u, 0x00020000};
        AF qf[D / 16];
        // Adjacent NN32 Q fragments occupy one aligned 16-byte vector.
        // Load the pair together; preserve every BF16 bit and the MMAC order.
        using QVec = unsigned int __attribute__((ext_vector_type(4)));
#    pragma unroll
        for (int j = 0; j < D / 32; j++) {
            QVec v = *(const QVec*)(q + ((size_t)t * H + h) * D + j * 32 + g * 8);
            asm volatile("" : "+v"(v)::"memory");
            *(unsigned long long*)&qf[j * 2].x[0]     = ((unsigned long long*)&v)[0];
            *(unsigned long long*)&qf[j * 2 + 1].x[0] = ((unsigned long long*)&v)[1];
        }
        CF acc[32];
#    pragma unroll
        for (int j = 0; j < 32; j++) du_fill_fragment(acc[j], 0.f);
        float m = -INFINITY, l = 0, mi = -INFINITY, il = 0;
        for (int base = 0; base < valid; base += 64) {
            // Every wave examines the same 64 keys, so skipping an empty tile is
            // CTA-uniform.
            const int tile_pos                  = base + lane;
            const int tile_key                  = tile_pos < valid ? idx[(size_t)t * K + tile_pos] : -1;
            const unsigned long long valid_mask = MASK16 ? __ballot(tile_key >= 0 && tile_key < S) : 0;
            if constexpr (MASK16) {
                if (!valid_mask) continue;
            } else if (!__builtin_amdgcn_readfirstlane(__any(tile_key >= 0 && tile_key < S)))
                continue;
            const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3), row = w * 16 + unsw / 32, col = unsw % 32;
            // Every lane already holds one of this query's 64 tile indices.
            // Reuse it for K, score masking and V; invalid and tail slots remain -1.
            int key     = __shfl(tile_key, row);
            auto fetchk = [&](int j) {
                int off = (key >= 0 && key < S) ? key * (D * 2) + (j * 32 + col) * 2 : -1;
                ld(kd, off, sm + (j % 4) * 2048 + w * 512);
            };
            CF sc[4];
#    pragma unroll
            for (int k = 0; k < 4; k++) du_fill_fragment(sc[k], 0.f);
#    pragma unroll
            for (int j = 0; j < 3; j++) fetchk(j);
#    pragma unroll
            for (int j = 0; j < D / 32; j++) {
                const int rem = D / 32 - 1 - j;
                waitn(min(2, rem));
                if (j + 3 < D / 32) fetchk(j + 3);  // The preceding barrier releases the previous ring slot.
                // The LDS swizzle only uses bits 6..8 of the element index, and every
                // k step adds 512 elements (a multiple of 64), so the swizzle result is
                // identical for all k. Hoist one per-lane base address and let the k
                // loop use immediate offsets instead of recomputing the whole address
                // for every read (VALU was this kernel's top pipe).
                const int kb_raw = (lane & 15) * 32 + (lane >> 4) * 8;
                const int kbase  = kb_raw ^ ((kb_raw >> 6) << 3);
#    pragma unroll
                for (int k = 0; k < 4; k++) {
                    using U4 = unsigned int __attribute__((ext_vector_type(4)));
                    U4 v     = *(const U4*)(sm + (j % 4) * 2048 + kbase + 512 * k);
                    BC b0, b1;
                    *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                    *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                    du_mma_sync(sc[k], qf[j * 2], b0, sc[k]);
                    du_mma_sync(sc[k], qf[j * 2 + 1], b1, sc[k]);
                }
            }

            bar();
            float nm = m;
#    pragma unroll
            for (int k = 0; k < 4; k++) {
                const unsigned valid16 = (unsigned)(valid_mask >> (k * 16));
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    bool slot_valid;
                    if constexpr (MASK16)
                        slot_valid = (valid16 >> (g + 4 * z)) & 1u;
                    else {
                        const int ix = __shfl(tile_key, k * 16 + g + 4 * z);
                        slot_valid   = ix >= 0 && ix < S;
                    }
                    sc[k].x[z] = slot_valid ? sc[k].x[z] * scale : -INFINITY;
                    nm         = fmaxf(nm, sc[k].x[z]);
                }
            }
            nm         = fmaxf(nm, __shfl_xor(nm, 16));
            nm         = fmaxf(nm, __shfl_xor(nm, 32));
            float corr = l > 0 ? hip_dsa_numeric::softmax_exp(m - nm) : 0, sum = 0;
            // Transpose the 4x4 probability fragment within each head's four lanes.
            // This removes the shared-memory round trip and its two CTA barriers.
            AF pa[4];
#    pragma unroll
            for (int k = 0; k < 4; k++) {
                unsigned short eb[4];
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    float e  = sc[k].x[z] == -INFINITY ? 0 : hip_dsa_numeric::softmax_exp(sc[k].x[z] - nm);
                    sum     += e;
                    eb[z]    = bf16_rne_bits(e);
                }
                unsigned a  = unsigned(eb[0]) | (unsigned(eb[1]) << 16);
                unsigned b  = unsigned(eb[2]) | (unsigned(eb[3]) << 16);
                unsigned ap = __shfl_xor(a, 16), bp = __shfl_xor(b, 16);
                unsigned lo    = (g & 1) ? (ap >> 16) | (a & 0xffff0000u) : (a & 0xffffu) | (ap << 16);
                unsigned hi    = (g & 1) ? (bp >> 16) | (b & 0xffff0000u) : (b & 0xffffu) | (bp << 16);
                unsigned other = __shfl_xor((g & 2) ? lo : hi, 32);
                if (g & 2)
                    lo = other;
                else
                    hi = other;
                ((unsigned*)&pa[k].x[0])[0] = lo;
                ((unsigned*)&pa[k].x[0])[1] = hi;
            }
            sum += __shfl_xor(sum, 16);
            sum += __shfl_xor(sum, 32);
            l    = l * corr + sum;
            m    = nm;
            // The host dispatch requires an indexer prefix aligned to 64 keys.
            if (base < IK) {
                mi = m;
                il = l;
            }
#    pragma unroll
            // Packed FP32 multiplies preserve independent per-element rounding.
            // Pair adjacent elements within each accumulator fragment; no reduction
            // order changes, extra BF16 conversion, or cross-head/query operations.
            for (int j = 0; j < 32; j++) {
#    pragma unroll
                for (int z = 0; z < 4; z += 2) {
                    float2 x = {acc[j].x[z], acc[j].x[z + 1]};
                    float2 c = {corr, corr}, result;
                    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(result) : "v"(x), "v"(c));
                    acc[j].x[z]     = result.x;
                    acc[j].x[z + 1] = result.y;
                }
            }
            int vk[4];
#    pragma unroll
            for (int k = 0; k < 4; k++) { vk[k] = __shfl(tile_key, k * 16 + lane / 4); }
#    pragma unroll
            for (int k = 0; k < 4; k++) asm volatile("" : "+v"(vk[k])::"memory");
            auto fetchv = [&](int j) {
                int k = j % 4, c = j / 4, ix = vk[k];
                int off = (ix >= 0 && ix < S) ? ix * (D * 2) + (c * 128 + w * 32 + lane % 4 * 8) * 2 : -1;
                ld(kd, off, sm + (j % 4) * 2048 + w * 512);
            };
#    pragma unroll
            for (int j = 0; j < 3; j++) fetchv(j);
#    pragma unroll
            for (int j = 0; j < 16; j++) {
                const int rem = 15 - j;
                waitn(min(2, rem));
                if (j + 3 < 16) fetchv(j + 3);
                // Keep V address lifetimes within this pipeline stage.
                int vl = lane;
                asm volatile("" : "+v"(vl));
#    pragma unroll
                for (int dc = 0; dc < 4; dc++) {
                    auto vp = (__attribute__((address_space(3))) __fp16*)(sm + (j % 4) * 2048 + dc * 512 + vl * 8);
                    auto v  = __builtin_amdgcn_ds_read_m32x16f16_alt(vp, 0);
                    BR b0, b1;
                    *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                    *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
                    du_mma_sync(acc[(j / 4) * 8 + dc * 2], pa[j % 4], b0, acc[(j / 4) * 8 + dc * 2]);
                    du_mma_sync(acc[(j / 4) * 8 + dc * 2 + 1], pa[j % 4], b1, acc[(j / 4) * 8 + dc * 2 + 1]);
                }
            }
            bar();
        }
        {
            float sk = sink ? sink[h] : -INFINITY, factor = 0;
            if (isnan(sk) || isnan(l))
                factor = NAN;
            else if (!(isinf(sk) && sk > 0) && l > 0) {
                float mm = fmaxf(m, sk), c = hip_dsa_numeric::softmax_exp(m - mm);
                factor = c / (l * c + ((sk > -INFINITY) ? hip_dsa_numeric::softmax_exp(sk - mm) : 0.f) + 1e-30f);
            }
            // Exchange already-rounded BF16 pairs across the four lanes of each head.
            // This gives each lane eight adjacent outputs for one aligned 128-bit store.
#    pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned v[4], x[4], y[4];
#    pragma unroll
                for (int z = 0; z < 4; z++)
                    v[z] = hip_dsa_numeric::bf16_pair(acc[j * 2].x[z] * factor, acc[j * 2 + 1].x[z] * factor);
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    const unsigned peer = __shfl_xor(v[z ^ 1], 16);
                    x[z]                = ((g ^ z) & 1) ? peer : v[z];
                }
#    pragma unroll
                for (int z = 0; z < 4; z++) {
                    const unsigned peer = __shfl_xor(x[z ^ 2], 32);
                    y[z]                = ((g ^ z) & 2) ? peer : x[z];
                }
                using Store = unsigned __attribute__((ext_vector_type(4)));
                __builtin_nontemporal_store((Store){y[0], y[1], y[2], y[3]},
                                            (Store*)(out + ((size_t)t * H + h) * 512 + j * 32 + g * 8));
            }
            if (g == 0) {
                if (mx) mx[(size_t)t * H + h] = m;
                lse[(size_t)t * H + h] = isnan(l) ? NAN : (l > 0 ? m + logf(l) : -INFINITY);
                if (li) {
                    float a = il > 0 ? mi + logf(il) : -INFINITY, b = fmaxf(a, sk);
                    li[(size_t)t * H + h] =
                        isnan(sk) ? NAN
                                  : (b == -INFINITY ? -INFINITY
                                                    : (isinf(b) ? b
                                                                : b + logf(hip_dsa_numeric::softmax_exp(a - b) +
                                                                           hip_dsa_numeric::softmax_exp(sk - b))));
                }
            }
        }
    }
}
#else
template<int TOPK, int STRIDE = TOPK>
__global__ static void h64_short_two_pass(const bf*, const bf*, const float*, const int*, const int*, bf*, float*,
                                          float*, int, int, float) {}

template<int D, bool MASK16 = false, bool PARTIAL_HEADS = false, int NT = 256>
__global__ static void h64(const bf*, const bf*, const float*, const int*, const int*, bf*, float*, float*, float*, int,
                           int, int, int, int, float) {}
__global__ static void h64_tail160(const bf*, const bf*, const float*, const int*, const int*, bf*, float*, float*,
                                   float*, int, int, int, int, int, float) {}
__global__ static void h64_tail256(const bf*, const bf*, const float*, const int*, const int*, bf*, float*, float*,
                                   float*, int, int, int, int, int, float) {}
__global__ static void h64_tail256_long_rows(const bf*, const bf*, const float*, const int*, const int*, bf*, float*,
                                             float*, float*, int, int, int, int, int, float) {}
#endif
}  // namespace hip_dsa_mla64

// Forward launchers. Shared device templates are instantiated only when used.

static bool sparse_attn_fused_supported() { return sparse_mmac_supported(); }

/**
 * Sparse attention forward with indexer statistics
 *
 * Single-launch fused sparse-attention forward with the DSv4 indexer
 * variant (functionally equivalent to FlashMLA's ``flash_mla_sparse_fwd``
 * with ``indexer_topk > 0``, but implemented purely with the hip_dsa MMAC
 * kernels so the CSA path no longer needs the FlashMLA indexer ABI):
 *
 *   S = Q@K^T (fp32), P = softmax([S, sink]), out = P@V
 *   max_logits : max over the valid KV scores (no sink)
 *   lse        : KV-only logsumexp over ALL topk entries
 *   lse_indexer: logsumexp over the FIRST ``indexer_topk`` entries of
 *                topk_idxs (the indexer/compress portion), merged with the
 *                attention sink: log(exp(lse_idx) + exp(sink)).
 *
 * The kernel is a copy of ``sparse_attn_fwd_fused_kernel``
 * (above in this file) with the indexer-region online stats and the two
 * extra outputs added; the grid / LDS / MMAC structure is identical.
 */

typedef uint16_t bf16;

__device__ __forceinline__ static uint16_t safi_float_to_bf16(float f) { return hip_dsa_numeric::bf16_rne(f); }

#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;
typedef __hip_bfloat16 hbf16;

// TOPK > 0: topk is a compile-time constant (the inner loops, LDS strides
// and MMAC chains are then fully unrolled / strength-reduced; runtime shape
// params cost ~5x here).  TOPK == 0: runtime topk fallback.  D is always
// ND*16.  indexer_topk is a runtime parameter: the extra per-tile indexer
// stats only cost a few flops, so keeping it runtime avoids doubling the
// template-instantiation count.
template<int ND, int NT, int TOPK = 0, int INDEXER_TOPK = 0, bool TILE_INDEX = false>
__global__ static void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_fwd_indexer_fused_kernel(const bf16* __restrict__ q,           // (T, H, D)
                                         const bf16* __restrict__ kv,          // (T_kv, D)
                                         const float* __restrict__ attn_sink,  // (H,)
                                         const int* __restrict__ idxs,         // (T, topk)
                                         const int* __restrict__ topk_length,  // (T,) or NULL
                                         bf16* __restrict__ out,               // (T, H, D_v)
                                         float* __restrict__ max_logits,       // (T, H) KV-only max logit (no sink)
                                         float* __restrict__ lse,              // (T, H)
                                         float* __restrict__ lse_indexer,  // (T, H) indexer-region LSE (sink-merged)
                                         int indexer_topk, int T, int H, int D_rt, int D_v_rt, int T_kv, int topk_rt,
                                         float softmax_scale) {
    // All public launchers guarantee 512 value channels, including D=576 QK.
    // A static bound keeps the PV accumulator in registers.
    constexpr int D_v = 512;
    (void)D_v_rt;
    constexpr int D              = ND * 16;                      // compile-time head dim
    const int topk               = (TOPK > 0) ? TOPK : topk_rt;  // folds to a constant for TOPK > 0
    const int indexer_topk_bound = (INDEXER_TOPK > 0) ? INDEXER_TOPK : indexer_topk;
    (void)D_rt;
    const int t    = blockIdx.x;
    const int hb   = blockIdx.y * SAF_HBLK;
    const int lane = threadIdx.x & 63;
    // D576 benefits from scalar wave ownership; D512 retains its measured mapping.
    const int warp     = (ND == 36 && NT == 1) ? __builtin_amdgcn_readfirstlane(threadIdx.x >> 6)
                                               : threadIdx.x >> 6;  // 0..7 (SAF_HBLK/16)
    const int r0       = warp * 16;
    const int h        = hb + r0 + (lane & 15);
    const bool lane_ok = (h < H);
    const int QLD = D + 8, KLD = D + 8;
    constexpr int nd = ND;
    const int g      = lane >> 4;

    extern __shared__ char smem_raw[];
    hbf16* k_lds = (hbf16*)smem_raw;  // (16*NT) keys x KLD (NT 16-key tiles)
    int* idx_lds = (int*)(k_lds + 16 * NT * KLD);
    // Production D=512 shapes use two tiny ping-pong index tiles instead of
    // keeping the complete 640/1152-entry row in LDS.  The compact hot set
    // permits a direct global->LDS transfer for the next tile and avoids an
    // intermediate VGPR payload plus LDS write instruction.
    const int idx_slots = TILE_INDEX ? 2 * 16 * NT : topk;
    hbf16* p_lds        = (hbf16*)(idx_lds + idx_slots);  // 4 warps x 256 bf16

    int n_valid_t = topk;
    if (topk_length != NULL) {
        int tl = topk_length[t];
        if (tl < n_valid_t) n_valid_t = tl > 0 ? tl : 0;
    }

    if constexpr (TILE_INDEX) {
        for (int i = threadIdx.x; i < 16 * NT; i += SAF_THREADS)
            idx_lds[i] = (i < topk) ? idxs[(size_t)t * topk + i] : -1;
    } else {
        for (int i = threadIdx.x; i < topk; i += SAF_THREADS) idx_lds[i] = idxs[(size_t)t * topk + i];
    }
    // idx_lds is read by the first K-gather below; without a barrier the
    // gather can race ahead and see uninitialized LDS (idx 0 for every slot).
    __syncthreads();

    DUFragment<accumulator, 16, 16, 16, float> outacc[ND];
#    pragma unroll
    for (int dc = 0; dc < ND; dc++) du_fill_fragment(outacc[dc], 0.0f);

    float sink_val = -INFINITY;
    float m        = -INFINITY;
    float l        = 0.0f;
    // KV-only max logit (max_logits) + online stats over the indexer
    // region (the first ``indexer_topk`` entries of idxs, lse_indexer).
    float m_kv  = -INFINITY;
    float m_idx = -INFINITY;
    float l_idx = 0.0f;
    if (lane_ok) { sink_val = attn_sink[h]; }

    const int hq = hb + r0;  // first global Q row of this warp
    // Cache this warp's Q rows (16 heads x D) in registers once per block
    // instead of re-loading the same fragment from global every 16-key
    // round (Q was ~89% of VMEM_RD instructions in the dsv4_pro profile).
    hbf16 qreg[ND][4];
    if (hq < H) {
        const int qrow_h   = hq + (lane & 15);
        const int qrow     = qrow_h < H ? qrow_h : (H - 1);
        const hbf16* qbase = (const hbf16*)(q + ((size_t)t * H + qrow) * D);
#    pragma unroll
        for (int kk = 0; kk < ND; kk++) {
            *(uint64_t*)&qreg[kk][0] = *(const uint64_t*)(qbase + kk * 16 + ((lane >> 4) << 2));
            if (qrow_h >= H) qreg[kk][0] = qreg[kk][1] = qreg[kk][2] = qreg[kk][3] = hbf16(0);
        }
    } else {
#    pragma unroll
        for (int kk = 0; kk < ND; kk++) qreg[kk][0] = qreg[kk][1] = qreg[kk][2] = qreg[kk][3] = hbf16(0);
    }
    // Use the same gfx938 direct global->LDS gather as sparse_attn_fwd.
    // Each active lane transfers one aligned 16-byte KV chunk directly into
    // LDS, avoiding the intermediate VGPR payload and the following ds_write.
    // D=576 needs 72 chunks per row and therefore keeps the cooperative
    // vector-load fallback because one wave only has 64 lanes.
    typedef int v4i __attribute__((ext_vector_type(4)));
    const uint64_t kv_addr = (uint64_t)kv;
    v4i kv_desc;
    kv_desc[0]              = (int)(uint32_t)(kv_addr & 0xffffffffu);
    kv_desc[1]              = (int)((uint32_t)(kv_addr >> 32) | 0x40000000u);
    kv_desc[2]              = (int)0x80000000u;
    kv_desc[3]              = (int)0x00020000u;
    const uint64_t idx_addr = (uint64_t)idxs;
    v4i idx_desc;
    idx_desc[0]      = (int)(uint32_t)(idx_addr & 0xffffffffu);
    idx_desc[1]      = (int)((uint32_t)(idx_addr >> 32) | 0x40000000u);
    idx_desc[2]      = (int)0x80000000u;
    idx_desc[3]      = (int)0x00020000u;
    const int ntiles = (topk + 15) >> 4;

    auto gather_tile = [&](int kt0, const int ck[NT], int ck_tot, int tile_idx_base, hbf16* region) {
        if constexpr (NT == 1 && (ND == 32 || ND == 36)) {
            // Batch all four LDS index reads before any global-to-LDS copy,
            // preventing each next index read from draining the prior gather.
            int keys[4];
#    pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = warp + j * 4;
                keys[j]     = n < ck_tot ? idx_lds[TILE_INDEX ? (tile_idx_base + n) : kt0 * 16 + n] : -1;
            }
            asm volatile("" ::"v"(keys[0]), "v"(keys[1]), "v"(keys[2]), "v"(keys[3]) : "memory");
#    pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = warp + j * 4;
                if (n < ck_tot) {
                    const int idx    = keys[j];
                    const bool valid = idx >= 0 && idx < T_kv && kt0 * 16 + n < n_valid_t;
#    pragma unroll
                    for (int dc = 0; dc < D; dc += 512) {
                        if (dc + lane * 8 < D) {
                            if (valid) {
                                __builtin_amdgcn_raw_buffer_load_lds(
                                    kv_desc, (__attribute__((address_space(3))) int*)(region + n * KLD + dc), 16,
                                    idx * (D * 2) + dc * 2 + lane * 16, 0, 0, 0);
                            } else {
                                uint64_t* dst = (uint64_t*)(region + n * KLD + dc + lane * 8);
                                dst[0] = dst[1] = 0;
                            }
                        }
                    }
                }
            }
        } else {
            constexpr int CHUNKS = D / 8;
            if constexpr (CHUNKS <= 64) {
                // m0/LDS destination is wave-uniform, so one wave stages one KV
                // row at a time while its lanes cover the row's 16-byte chunks.
                for (int n = warp; n < ck_tot; n += 4) {
                    int ti = 0, rem = n;
#    pragma unroll
                    for (int i = 0; i < NT; i++) {
                        if (rem < ck[i]) break;
                        rem -= ck[i];
                        ti++;
                    }
                    const int gidx     = (kt0 + ti) * 16 + rem;
                    const int idx_slot = TILE_INDEX ? tile_idx_base + ti * 16 + rem : gidx;
                    int idx            = idx_lds[idx_slot];
                    const bool valid   = gidx < n_valid_t && idx >= 0 && idx < T_kv;
                    if (lane < CHUNKS) {
                        if (valid) {
                            __builtin_amdgcn_raw_buffer_load_lds(
                                kv_desc, (__attribute__((address_space(3))) int*)(region + (size_t)n * KLD), 16,
                                idx * (D * 2) + lane * 16, 0, 0, 0);
                        } else {
                            uint64_t* dst = (uint64_t*)(region + (size_t)n * KLD + lane * 8);
                            dst[0] = dst[1] = 0;
                        }
                    }
                }
            } else {
                for (int n = warp; n < ck_tot; n += 4) {
                    int ti = 0, rem = n;
#    pragma unroll
                    for (int i = 0; i < NT; i++) {
                        if (rem < ck[i]) break;
                        rem -= ck[i];
                        ti++;
                    }
                    const int gidx     = (kt0 + ti) * 16 + rem;
                    const int idx_slot = TILE_INDEX ? tile_idx_base + ti * 16 + rem : gidx;
                    int idx            = idx_lds[idx_slot];
                    const bool valid   = gidx < n_valid_t && idx >= 0 && idx < T_kv;
#    pragma unroll
                    for (int dc = 0; dc < D; dc += 512) {
                        if (dc + lane * 8 < D) {
                            if (valid) {
                                __builtin_amdgcn_raw_buffer_load_lds(
                                    kv_desc, (__attribute__((address_space(3))) int*)(region + n * KLD + dc), 16,
                                    idx * (D * 2) + dc * 2 + lane * 16, 0, 0, 0);
                            } else {
                                uint64_t* dst = (uint64_t*)(region + n * KLD + dc + lane * 8);
                                dst[0] = dst[1] = 0;
                            }
                        }
                    }
                }
            }
        }
        if (ck_tot < 16 * NT) {
            for (int u = threadIdx.x; u < (16 * NT) * KLD; u += SAF_THREADS) {
                if (u / KLD >= ck_tot) region[u] = hbf16(0);
            }
        }
    };

    for (int kt = 0; kt < ntiles; kt += NT) {
        int ck[NT];
        int ck_tot = 0;
#    pragma unroll
        for (int i = 0; i < NT; i++) {
            int c   = topk - (kt + i) * 16;
            ck[i]   = (c > 16) ? 16 : (c < 0 ? 0 : c);
            ck_tot += ck[i];
        }

        const int tile_idx_base = TILE_INDEX ? ((kt / NT) & 1) * (16 * NT) : 0;
        // Keep the online-softmax loop and index pipeline unchanged. An
        // entirely masked tile needs neither a K gather nor QK/PV MMAC. A loop-level continue inhibits accumulator scalarization
        // on DTK and spills hundreds of registers on the D=512 path.
        // Compact index tiles keep the large-K path below the LDS and
        // register-pressure threshold. Reuse a ballot for D512 softmax;
        // D576 retains its original predicates to avoid register spills.
        constexpr bool REUSE_MASK = ND == 32 && NT == 1 && TILE_INDEX;
        unsigned key_mask         = 0;
        bool matrix_active        = true;
        if constexpr (REUSE_MASK || (ND == 36 && NT == 1 && TILE_INDEX)) {
            const int col = kt * 16 + (lane & 15);
            const int key = col < n_valid_t ? idx_lds[tile_idx_base + (lane & 15)] : -1;
            if constexpr (REUSE_MASK) {
                key_mask      = (unsigned)__ballot(key >= 0 && key < T_kv) & 0xffffu;
                matrix_active = key_mask != 0;
            } else {
                matrix_active = __builtin_amdgcn_readfirstlane(__any(key >= 0 && key < T_kv));
            }
        }
        if (matrix_active) gather_tile(kt, ck, ck_tot, tile_idx_base, k_lds);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        const hbf16* kbase = k_lds;

        // Queue the next compact index tile directly into the alternate LDS
        // slot.  gfx938 conservatively resolves this LDS transfer before the
        // following LDS reads, but it is still cheaper than a VGPR round trip.
        if constexpr (TILE_INDEX) {
            const int next_col = (kt + NT) * 16;
            if (warp == 0 && next_col < topk && lane < 16 * NT) {
                const int next_base = tile_idx_base ^ (16 * NT);
                __builtin_amdgcn_raw_buffer_load_lds(idx_desc,
                                                     (__attribute__((address_space(3))) int*)(idx_lds + next_base), 4,
                                                     (int)(((size_t)t * topk + next_col) * 4) + lane * 4, 0, 0, 0);
            }
        }

        // S tiles (16 heads x 16 keys) via MMAC over D; NT chains interleave.
        DUFragment<accumulator, 16, 16, 16, float> acc[NT];
#    pragma unroll
        for (int i = 0; i < NT; i++) du_fill_fragment(acc[i], 0.0f);
        if (hq < H && matrix_active) {
            for (int kk = 0; kk < ND; kk++) {
                DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> a;
                *(uint64_t*)&a.x[0] = *(const uint64_t*)&qreg[kk][0];
#    pragma unroll
                for (int i = 0; i < NT; i++) {
                    if (ck[i] > 0) {
                        DUFragment<matrix_b, 16, 16, 16, hbf16, col_major> bfr;
                        int koff = 0;
#    pragma unroll
                        for (int j = 0; j < i; j++) koff += ck[j];
                        du_load_matrix_sync(bfr, kbase + (size_t)koff * KLD + kk * 16, KLD);
                        du_mma_sync(acc[i], a, bfr, acc[i]);
                    }
                }
            }
        }

// Per-tile online-softmax epilogues: P tiles staged in LDS; when a
// later tile in this round raises the max, the earlier staged P tiles
// are rescaled in LDS so the P@V pass uses exp(S - m_new).
#    pragma unroll
        for (int i = 0; i < NT; i++) {
            if (ck[i] > 0) {
                const int c0 = (kt + i) * 16 + g;
                const int c1 = c0 + 4, c2 = c0 + 8, c3 = c0 + 12;
                const int epi_idx_base = TILE_INDEX ? tile_idx_base + i * 16 : (kt + i) * 16;
                const bool ok0         = REUSE_MASK ? ((key_mask >> (g + 0)) & 1u)
                                                    : ((c0 < n_valid_t) && (idx_lds[epi_idx_base + g] >= 0) &&
                                                       (idx_lds[epi_idx_base + g] < T_kv));
                const bool ok1         = REUSE_MASK ? ((key_mask >> (g + 4)) & 1u)
                                                    : ((c1 < n_valid_t) && (idx_lds[epi_idx_base + g + 4] >= 0) &&
                                                       (idx_lds[epi_idx_base + g + 4] < T_kv));
                const bool ok2         = REUSE_MASK ? ((key_mask >> (g + 8)) & 1u)
                                                    : ((c2 < n_valid_t) && (idx_lds[epi_idx_base + g + 8] >= 0) &&
                                                       (idx_lds[epi_idx_base + g + 8] < T_kv));
                const bool ok3         = REUSE_MASK ? ((key_mask >> (g + 12)) & 1u)
                                                    : ((c3 < n_valid_t) && (idx_lds[epi_idx_base + g + 12] >= 0) &&
                                                       (idx_lds[epi_idx_base + g + 12] < T_kv));
                // Phantom heads (h >= H): -inf scores so they never win the
                // max and their e = 0 (no pollution of the group reduction).
                const float s0 = lane_ok ? acc[i].x[0] * softmax_scale : -INFINITY;
                const float s1 = lane_ok ? acc[i].x[1] * softmax_scale : -INFINITY;
                const float s2 = lane_ok ? acc[i].x[2] * softmax_scale : -INFINITY;
                const float s3 = lane_ok ? acc[i].x[3] * softmax_scale : -INFINITY;
                float tmax     = -INFINITY;
                if (ok0) tmax = fmaxf(tmax, s0);
                if (ok1) tmax = fmaxf(tmax, s1);
                if (ok2) tmax = fmaxf(tmax, s2);
                if (ok3) tmax = fmaxf(tmax, s3);
                tmax = fmaxf(tmax, __shfl_xor(tmax, 16));
                tmax = fmaxf(tmax, __shfl_xor(tmax, 32));
                // KV-only max logit: max over the valid scores of this
                // tile, independent of the attention sink.
                m_kv = fmaxf(m_kv, tmax);
                // Indexer-region online softmax stats: only the entries
                // with key column < indexer_topk participate.
                // Before the indexer boundary the indexer prefix and the
                // KV prefix are identical. Reuse (m,l) after its update;
                // only a tile straddling the boundary needs separate stats.
                const bool full_indexer_tile = (kt + i + 1) * 16 <= indexer_topk_bound;
                if (!full_indexer_tile && (kt + i) * 16 < indexer_topk_bound) {
                    const bool ok_idx0 = ok0 && (c0 < indexer_topk_bound);
                    const bool ok_idx1 = ok1 && (c1 < indexer_topk_bound);
                    const bool ok_idx2 = ok2 && (c2 < indexer_topk_bound);
                    const bool ok_idx3 = ok3 && (c3 < indexer_topk_bound);
                    float tmax_idx     = -INFINITY;
                    if (ok_idx0) tmax_idx = fmaxf(tmax_idx, s0);
                    if (ok_idx1) tmax_idx = fmaxf(tmax_idx, s1);
                    if (ok_idx2) tmax_idx = fmaxf(tmax_idx, s2);
                    if (ok_idx3) tmax_idx = fmaxf(tmax_idx, s3);
                    tmax_idx = fmaxf(tmax_idx, __shfl_xor(tmax_idx, 16));
                    tmax_idx = fmaxf(tmax_idx, __shfl_xor(tmax_idx, 32));
                    if (tmax_idx > m_idx) {
                        l_idx *= expf(m_idx - tmax_idx);
                        m_idx  = tmax_idx;
                    }
                    const float e_idx0  = ok_idx0 ? expf(s0 - m_idx) : 0.0f;
                    const float e_idx1  = ok_idx1 ? expf(s1 - m_idx) : 0.0f;
                    const float e_idx2  = ok_idx2 ? expf(s2 - m_idx) : 0.0f;
                    const float e_idx3  = ok_idx3 ? expf(s3 - m_idx) : 0.0f;
                    float srow_idx      = e_idx0 + e_idx1 + e_idx2 + e_idx3;
                    srow_idx           += __shfl_xor(srow_idx, 16);
                    srow_idx           += __shfl_xor(srow_idx, 32);
                    l_idx              += srow_idx;
                }
                if (tmax > m) {
                    const float corr  = expf(m - tmax);
                    l                *= corr;
                    for (int dc = 0; dc < ND; dc++) {
                        if (dc * 16 >= D_v) continue;
                        outacc[dc].x[0] *= corr;
                        outacc[dc].x[1] *= corr;
                        outacc[dc].x[2] *= corr;
                        outacc[dc].x[3] *= corr;
                    }
                    m = tmax;
                    // Rescale this round's already-staged P tiles (j < i).
                    hbf16* pw0 = p_lds + (size_t)warp * (NT * 256);
#    pragma unroll
                    for (int j = 0; j < i; j++) {
                        const int pr           = (lane & 15) * 16 + g;
                        pw0[j * 256 + pr]      = __float2bfloat16(__bfloat162float(pw0[j * 256 + pr]) * corr);
                        pw0[j * 256 + pr + 4]  = __float2bfloat16(__bfloat162float(pw0[j * 256 + pr + 4]) * corr);
                        pw0[j * 256 + pr + 8]  = __float2bfloat16(__bfloat162float(pw0[j * 256 + pr + 8]) * corr);
                        pw0[j * 256 + pr + 12] = __float2bfloat16(__bfloat162float(pw0[j * 256 + pr + 12]) * corr);
                    }
                }
                const float e0  = (ok0 && lane_ok) ? expf(s0 - m) : 0.0f;
                const float e1  = (ok1 && lane_ok) ? expf(s1 - m) : 0.0f;
                const float e2  = (ok2 && lane_ok) ? expf(s2 - m) : 0.0f;
                const float e3  = (ok3 && lane_ok) ? expf(s3 - m) : 0.0f;
                float srow      = e0 + e1 + e2 + e3;
                srow           += __shfl_xor(srow, 16);
                srow           += __shfl_xor(srow, 32);
                l              += srow;
                if (full_indexer_tile) {
                    m_idx = m;
                    l_idx = l;
                }
                hbf16* pw                     = p_lds + (size_t)warp * (NT * 256) + i * 256;
                pw[(lane & 15) * 16 + g]      = __float2bfloat16(e0);
                pw[(lane & 15) * 16 + g + 4]  = __float2bfloat16(e1);
                pw[(lane & 15) * 16 + g + 8]  = __float2bfloat16(e2);
                pw[(lane & 15) * 16 + g + 12] = __float2bfloat16(e3);
            }
        }
        // No barrier here: each warp stages P in its own p_lds region and
        // only that warp reads it back, so the writes are warp-local.

        // QK uses all D channels; PV uses only V = K[..., :D_v].
        if (hq < H && matrix_active) {
            DUFragment<matrix_a, 16, 16, 16, hbf16, row_major> pa[NT];
#    pragma unroll
            for (int i = 0; i < NT; i++) {
                if (ck[i] > 0) du_load_matrix_sync(pa[i], p_lds + (size_t)warp * (NT * 256) + i * 256, 16);
            }
            // The normal DS matrix read preserves the public column order.
            // Four pending reads feed eight MMACs from the existing padded K
            // tile; global gather and online softmax stay unchanged.
            if (NT == 1 && (ND == 32 || ND == 36) && D_v == 512) {
                typedef unsigned int mat4 __attribute__((ext_vector_type(4)));
#    pragma unroll
                for (int dc = 0; dc < D_v / 16; dc += 8) {
                    mat4 packed[4];
#    pragma unroll
                    for (int z = 0; z < 4; z++) {
                        const unsigned addr =
                            (unsigned)(size_t)(kbase + (dc + 2 * z) * 16 + (lane / 4) * KLD + (lane % 4) * 8);
                        asm volatile("ds_read_m32x16_b16 %0, %1" : "=v"(packed[z]) : "v"(addr) : "memory");
                    }
                    // Flash's 640-key specialization benefits from consuming
                    // each completed LDS read while the remaining reads run.
                    // Every read is retired before its two MMAC consumers;
                    // the last wait also drains LDS before buffer reuse.
                    if constexpr (TOPK != 640 || INDEXER_TOPK != 512) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#    pragma unroll
                    for (int z = 0; z < 4; z++) {
                        if constexpr (TOPK == 640 && INDEXER_TOPK == 512) {
                            if (z == 0) asm volatile("s_waitcnt lgkmcnt(3)" ::: "memory");
                            if (z == 1) asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
                            if (z == 2) asm volatile("s_waitcnt lgkmcnt(1)" ::: "memory");
                            if (z == 3) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                        }
                        DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> b0, b1;
                        ((uint32_t*)&b0.x[0])[0] = packed[z][0];
                        ((uint32_t*)&b0.x[0])[1] = packed[z][1];
                        ((uint32_t*)&b1.x[0])[0] = packed[z][2];
                        ((uint32_t*)&b1.x[0])[1] = packed[z][3];
                        du_mma_sync(outacc[dc + 2 * z], pa[0], b0, outacc[dc + 2 * z]);
                        du_mma_sync(outacc[dc + 2 * z + 1], pa[0], b1, outacc[dc + 2 * z + 1]);
                    }
                }
            } else {
                for (int dc = 0; dc < ND; dc++) {
                    if (dc * 16 >= D_v) continue;
#    pragma unroll
                    for (int i = 0; i < NT; i++) {
                        if (ck[i] > 0) {
                            DUFragment<matrix_b, 16, 16, 16, hbf16, row_major> bfr;
                            int koff = 0;
#    pragma unroll
                            for (int j = 0; j < i; j++) koff += ck[j];
                            du_load_matrix_sync(bfr, kbase + (size_t)koff * KLD + dc * 16, KLD);
                            du_mma_sync(outacc[dc], pa[i], bfr, outacc[dc]);
                        }
                    }
                }
            }
        }
        if constexpr (TILE_INDEX) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();  // protect k_lds and publish next round's indices
    }

    // ---- Finalize KV-only LSE, then merge the sink only into output. ----
    if (lane_ok) {
        float out_scale = 0.0f;
        if (sink_val != sink_val || l != l) {
            out_scale = NAN;
        } else if (isinf(sink_val) && sink_val > 0.0f) {
            out_scale = 0.0f;
        } else if (l > 0.0f) {
            const float merged_m  = fmaxf(m, sink_val);
            const float kv_corr   = expf(m - merged_m);
            const float sink_mass = (sink_val > -INFINITY) ? expf(sink_val - merged_m) : 0.0f;
            out_scale             = kv_corr / (l * kv_corr + sink_mass + 1e-30f);
        }
        bf16* orow = out + ((size_t)t * H + h) * D_v;
        for (int dc = 0; dc < ND; dc++) {
            if (dc * 16 >= D_v) continue;
            const int col  = dc * 16 + g;
            const float p0 = outacc[dc].x[0] * out_scale;
            const float p1 = outacc[dc].x[1] * out_scale;
            const float p2 = outacc[dc].x[2] * out_scale;
            const float p3 = outacc[dc].x[3] * out_scale;
            if (col < D_v) orow[col] = safi_float_to_bf16(p0);
            if (col + 4 < D_v) orow[col + 4] = safi_float_to_bf16(p1);
            if (col + 8 < D_v) orow[col + 8] = safi_float_to_bf16(p2);
            if (col + 12 < D_v) orow[col + 12] = safi_float_to_bf16(p3);
        }
        if ((lane >> 4) == 0) {
            if (l > 0.0f) {
                lse[(size_t)t * H + h] = m + logf(l);
            } else if (l != l) {
                lse[(size_t)t * H + h] = NAN;  // degenerate NaN input: keep NaN
            } else {
                lse[(size_t)t * H + h] = -INFINITY;
            }
            // max_logits: max over the valid KV scores (no sink);
            // -INFINITY when every KV entry of the row is invalid.
            max_logits[(size_t)t * H + h] = m_kv;
            // lse_indexer: logsumexp over the first ``indexer_topk``
            // entries, merged with the attention sink
            // (log(exp(lse_idx) + exp(sink)) — matches the
            // ``_dsa_fwd_hip_dsa_indexer`` replacement and the native
            // sink-softmax KL target).
            float lse_idx = -INFINITY;
            if (l_idx > 0.0f) {
                lse_idx = m_idx + logf(l_idx);
            } else if (l_idx != l_idx) {
                lse_idx = NAN;  // degenerate NaN input: keep NaN
            }
            if (sink_val > -INFINITY) {
                if (lse_idx != lse_idx) {
                    lse_idx = NAN;
                } else if (lse_idx > -INFINITY) {
                    lse_idx = fmaxf(lse_idx, sink_val) + log1pf(expf(-fabsf(lse_idx - sink_val)));
                } else {
                    lse_idx = sink_val;
                }
            }
            lse_indexer[(size_t)t * H + h] = lse_idx;
        }
    }
}

#else  // gfx906 / gfx926 / others: no MMAC intrinsics - never launched.

template<int ND, int NT, int TOPK = 0, int INDEXER_TOPK = 0, bool TILE_INDEX = false>
__global__ static void __launch_bounds__(SAF_THREADS, 1)
    sparse_attn_fwd_indexer_fused_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kv,
                                         const float* __restrict__ attn_sink, const int* __restrict__ idxs,
                                         const int* __restrict__ topk_length, bf16* __restrict__ out,
                                         float* __restrict__ max_logits, float* __restrict__ lse,
                                         float* __restrict__ lse_indexer, int indexer_topk, int T, int H, int D_rt,
                                         int D_v, int T_kv, int topk_rt, float softmax_scale) {}

#endif  // supported gfx archs

static bool launch_sparse_attn_mla64(hipStream_t stream, const void* q, const void* kv, const float* attn_sink,
                                     const int* topk_idxs, const int* topk_length, void* out, float* max_logits,
                                     float* lse, float* lse_indexer, int indexer_topk, int T, int H, int D, int T_kv,
                                     int topk, float softmax_scale) {
    // Buffer offsets are signed 32-bit byte offsets.
    if (T_kv <= 0 || (int64_t)T_kv * D * 2 > INT32_MAX) return false;
    // Measured large-query cases; use the existing path for small grids,
    // other partial head groups and an indexer boundary inside a 64-key tile.
    // Plain forward has no prefix output and accepts non-aligned topk, e.g. 160.
    if (T >= 1024 && ((H == 64 || H == 128) || (D == 576 && (H == 8 || H == 16 || H == 32))) &&
        (!lse_indexer || indexer_topk % 64 == 0) &&
        ((D == 512 && (topk == 128 || topk == 160 || topk == 256 || topk == 512 || topk == 640 || topk == 1024 ||
                       topk == 1152 || topk == 2048)) ||
         (D == 576 && topk == 2048))) {
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
        if (T > prop.maxGridSize[1]) return false;
        if (strncmp(prop.gcnArchName, "gfx936", 6) == 0 || strncmp(prop.gcnArchName, "gfx938", 6) == 0) {
#define MLA64_RUN(DIM, MASK)                                                                                          \
    hipLaunchKernelGGL((hip_dsa_mla64::h64<DIM, MASK>), dim3(H / 64, T), 256, 0, stream, (const hip_dsa_mla64::bf*)q, \
                       (const hip_dsa_mla64::bf*)kv, attn_sink, topk_idxs, topk_length, (hip_dsa_mla64::bf*)out,      \
                       max_logits, lse, lse_indexer, T, H, T_kv, topk, indexer_topk, softmax_scale)
#define MLA64_TAIL_RUN()                                                                                         \
    hipLaunchKernelGGL(hip_dsa_mla64::h64_tail160, dim3(H / 64, T), 256, 0, stream, (const hip_dsa_mla64::bf*)q, \
                       (const hip_dsa_mla64::bf*)kv, attn_sink, topk_idxs, topk_length, (hip_dsa_mla64::bf*)out, \
                       max_logits, lse, lse_indexer, T, H, T_kv, topk, indexer_topk, softmax_scale)
#define MLA64_TAIL256_RUN()                                                                                      \
    hipLaunchKernelGGL(hip_dsa_mla64::h64_tail256, dim3(H / 64, T), 256, 0, stream, (const hip_dsa_mla64::bf*)q, \
                       (const hip_dsa_mla64::bf*)kv, attn_sink, topk_idxs, topk_length, (hip_dsa_mla64::bf*)out, \
                       max_logits, lse, lse_indexer, T, H, T_kv, topk, indexer_topk, softmax_scale)
            // Exact mask reuse removes score-only index shuffles. Restrict to
            // measured long D512 cases; D576 retains its spill-free schedule.
            if (D == 512 && !lse_indexer && (topk == 128 || topk == 160)) {
#define SHORT_TWO_PASS(K)                                                                                            \
    hipLaunchKernelGGL((hip_dsa_mla64::h64_short_two_pass<K>), dim3(H / 64, T), 256, 0, stream,                      \
                       (const hip_dsa_mla64::bf*)q, (const hip_dsa_mla64::bf*)kv, attn_sink, topk_idxs, topk_length, \
                       (hip_dsa_mla64::bf*)out, max_logits, lse, H, T_kv, softmax_scale)
                if (topk == 128) {
                    SHORT_TWO_PASS(128);
                } else {
                    SHORT_TWO_PASS(160);
                }
#undef SHORT_TWO_PASS
            } else if (D == 512 && !lse_indexer && !topk_length && topk == 256 && T % 128 == 0 && T_kv == T + T / 128) {
                // The external CSA adapter may pad 160 entries to 256.
                // Two disjoint row guards preserve arbitrary valid tails.
                hipLaunchKernelGGL((hip_dsa_mla64::h64_short_two_pass<160, 256>), dim3(H / 64, T), 256, 0, stream,
                                   (const hip_dsa_mla64::bf*)q, (const hip_dsa_mla64::bf*)kv, attn_sink, topk_idxs,
                                   topk_length, (hip_dsa_mla64::bf*)out, max_logits, lse, H, T_kv, softmax_scale);
                hipLaunchKernelGGL(hip_dsa_mla64::h64_tail256_long_rows, dim3(H / 64, T), 256, 0, stream,
                                   (const hip_dsa_mla64::bf*)q, (const hip_dsa_mla64::bf*)kv, attn_sink, topk_idxs,
                                   topk_length, (hip_dsa_mla64::bf*)out, max_logits, lse, lse_indexer, T, H, T_kv, topk,
                                   indexer_topk, softmax_scale);
            } else if (H <= 32) {
                // Two waves cover the live heads; each also loads half of the
                // otherwise missing K/V tile, without doing padding MMAC.
                hipLaunchKernelGGL((hip_dsa_mla64::h64<576, false, true, 128>), dim3(1, T), 128, 0, stream,
                                   (const hip_dsa_mla64::bf*)q, (const hip_dsa_mla64::bf*)kv, attn_sink, topk_idxs,
                                   topk_length, (hip_dsa_mla64::bf*)out, max_logits, lse, lse_indexer, T, H, T_kv, topk,
                                   indexer_topk, softmax_scale);
            } else if (H < 64) {
                // Cooperative K/V loads still use four waves; mask only the
                // final head tile instead of allocating external padding.
                hipLaunchKernelGGL((hip_dsa_mla64::h64<576, false, true>), dim3(1, T), 256, 0, stream,
                                   (const hip_dsa_mla64::bf*)q, (const hip_dsa_mla64::bf*)kv, attn_sink, topk_idxs,
                                   topk_length, (hip_dsa_mla64::bf*)out, max_logits, lse, lse_indexer, T, H, T_kv, topk,
                                   indexer_topk, softmax_scale);
            } else if (D == 512) {
                if (topk == 160) {
                    MLA64_TAIL_RUN();
                } else if (topk == 256 && topk_length == nullptr) {
                    MLA64_TAIL256_RUN();
                } else if (topk >= 1024) {
                    MLA64_RUN(512, true);
                } else {
                    MLA64_RUN(512, false);
                }
            } else {
                MLA64_RUN(576, false);
            }
#undef MLA64_RUN
#undef MLA64_TAIL_RUN
#undef MLA64_TAIL256_RUN
            return hipGetLastError() == hipSuccess;
        }
    }

    return false;
}

static bool launch_sparse_attn_mla64(hipStream_t stream, const void* q, const void* kv, const float* attn_sink,
                                     const int* topk_idxs, const int* topk_length, void* out, float* max_logits,
                                     float* lse, float* lse_indexer, int indexer_topk, int T, int H, int D, int T_kv,
                                     int topk, float softmax_scale);
static bool launch_sparse_attn_fwd_indexer_fused(hipStream_t stream, const void* q, const void* kv,
                                                 const float* attn_sink, const int* topk_idxs, void* out,
                                                 float* max_logits, float* lse, float* lse_indexer,
                                                 const int* topk_length, int indexer_topk, int T, int H, int D, int D_v,
                                                 int T_kv, int topk, float softmax_scale) {
    if (!sparse_attn_fused_supported()) return false;
    if (D % 16 != 0 || D > SAF_MAX_D || (D_v != 512 || D < 512)) return false;
    if (T <= 0 || H <= 0 || topk <= 0) return false;
    if (indexer_topk <= 0 || indexer_topk > topk) return false;

    if (launch_sparse_attn_mla64(stream, q, kv, attn_sink, topk_idxs, topk_length, out, max_logits, lse, lse_indexer,
                                 indexer_topk, T, H, D, T_kv, topk, softmax_scale))
        return true;

    const int max_lds     = dsa_shared_memory_limit();
    const int KLD         = D + 8;
    const int SAF_NT      = 1;
    const bool tile_index = (D == 512 && (topk == 1024 || topk == 1152)) || (D == 576 && topk == 2048);
    const int idx_slots   = tile_index ? 2 * 16 * SAF_NT : topk;
    size_t shm            = (size_t)(16 * SAF_NT) * KLD * 2 + (size_t)idx_slots * 4 + 4 * SAF_NT * 16 * 16 * 2;
    if (shm > (size_t)max_lds) return false;

    dim3 grid(T, (H + SAF_HBLK - 1) / SAF_HBLK);
// TOPK > 0 instantiations are much faster (compile-time topk/D); fall
// back to the runtime-topk kernel for other topk values.
#define SAF_LAUNCH(ND, NT)                                                                                             \
    hipLaunchKernelGGL((sparse_attn_fwd_indexer_fused_kernel<ND, NT>), grid, SAF_THREADS, shm, stream, (const bf16*)q, \
                       (const bf16*)kv, attn_sink, topk_idxs, topk_length, (bf16*)out, max_logits, lse, lse_indexer,   \
                       indexer_topk, T, H, D, D_v, T_kv, topk, softmax_scale)
#define SAF_LAUNCH_T(ND, NT, TOPK)                                                                                 \
    hipLaunchKernelGGL((sparse_attn_fwd_indexer_fused_kernel<ND, NT, TOPK>), grid, SAF_THREADS, shm, stream,       \
                       (const bf16*)q, (const bf16*)kv, attn_sink, topk_idxs, topk_length, (bf16*)out, max_logits, \
                       lse, lse_indexer, indexer_topk, T, H, D, D_v, T_kv, topk, softmax_scale)
#define SAF_LAUNCH_TI(ND, NT, TOPK, INDEXER_TOPK)                                                                  \
    hipLaunchKernelGGL((sparse_attn_fwd_indexer_fused_kernel<ND, NT, TOPK, INDEXER_TOPK>), grid, SAF_THREADS, shm, \
                       stream, (const bf16*)q, (const bf16*)kv, attn_sink, topk_idxs, topk_length, (bf16*)out,     \
                       max_logits, lse, lse_indexer, indexer_topk, T, H, D, D_v, T_kv, topk, softmax_scale)
#define SAF_LAUNCH_TIL(ND, NT, TOPK, INDEXER_TOPK)                                                                  \
    hipLaunchKernelGGL((sparse_attn_fwd_indexer_fused_kernel<ND, NT, TOPK, INDEXER_TOPK, true>), grid, SAF_THREADS, \
                       shm, stream, (const bf16*)q, (const bf16*)kv, attn_sink, topk_idxs, topk_length, (bf16*)out, \
                       max_logits, lse, lse_indexer, indexer_topk, T, H, D, D_v, T_kv, topk, softmax_scale)
    const int nd_ = D / 16;
    // FusedIndexerSparseAttnFunc appends its 128-key local window to the
    // indexer selection, producing these two production-only D=512 shapes.
    // Isolate the extra instantiations here rather than multiplying them
    // across every supported head dimension in SAF_CASE.
    if (nd_ == 32 && topk == 640 && indexer_topk == 512) {
        {
            const hipError_t status =
                hipFuncSetAttribute((const void*)sparse_attn_fwd_indexer_fused_kernel<32, SAF_NT, 640, 512>,
                                    hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipFuncSetAttribute failed: %s",
                                        hipGetErrorString(status));
            }
        }
        SAF_LAUNCH_TI(32, SAF_NT, 640, 512);
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                        "bf16_sparse_attn_fwd_indexer HIP call failed: %s", hipGetErrorString(status));
            }
        }
        return true;
    }
    if (nd_ == 32 && topk == 1024) {
        SAF_LAUNCH_TIL(32, SAF_NT, 1024, 0);
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                        "bf16_sparse_attn_fwd_indexer HIP call failed: %s", hipGetErrorString(status));
            }
        }
        return true;
    }
    if (nd_ == 36 && topk == 2048) {
        SAF_LAUNCH_TIL(36, SAF_NT, 2048, 0);
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                        "bf16_sparse_attn_fwd_indexer HIP call failed: %s", hipGetErrorString(status));
            }
        }
        return true;
    }
    if (nd_ == 32 && topk == 1152) {
        // Keeping indexer_topk runtime is 0.33% faster for Pro even though
        // total topk remains specialized.
        {
            const hipError_t status =
                hipFuncSetAttribute((const void*)sparse_attn_fwd_indexer_fused_kernel<32, SAF_NT, 1152, 0, true>,
                                    hipFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipFuncSetAttribute failed: %s",
                                        hipGetErrorString(status));
            }
        }
        SAF_LAUNCH_TIL(32, SAF_NT, 1152, 0);
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                        "bf16_sparse_attn_fwd_indexer HIP call failed: %s", hipGetErrorString(status));
            }
        }
        return true;
    }
#define SAF_CASE(NDV)                                          \
    case NDV:                                                  \
        switch (topk) {                                        \
            case 64: SAF_LAUNCH_T(NDV, SAF_NT, 64); break;     \
            case 128: SAF_LAUNCH_T(NDV, SAF_NT, 128); break;   \
            case 256: SAF_LAUNCH_T(NDV, SAF_NT, 256); break;   \
            case 512: SAF_LAUNCH_T(NDV, SAF_NT, 512); break;   \
            case 1024: SAF_LAUNCH_T(NDV, SAF_NT, 1024); break; \
            case 2048: SAF_LAUNCH_T(NDV, SAF_NT, 2048); break; \
            default: SAF_LAUNCH(NDV, SAF_NT); break;           \
        }                                                      \
        break
    switch (nd_) {
        SAF_CASE(32);
        SAF_CASE(36);
        default: return false;
    }
#undef SAF_LAUNCH
#undef SAF_LAUNCH_T
#undef SAF_LAUNCH_TI
#undef SAF_LAUNCH_TIL
#undef SAF_CASE
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_sparse_attn_fwd_indexer HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}

namespace ftrain {

bool bf16SparseAttnFwdIndexerLaunchFits(int width, int columns) {
    if (width != 512 && width != 576) return false;
    const int slots =
        ((width == 512 && (columns == 1024 || columns == 1152)) || (width == 576 && columns == 2048)) ? 32 : columns;
    return dsa_shared_memory_fits(16ULL * (width + 8) * 2 + static_cast<std::size_t>(slots) * 4 + 4 * 256 * 2);
}

bool launchBf16SparseAttnFwdIndexer(FTrainStream stream, const void* q, const void* kv, const float* attn_sink,
                                    const int* topk_indices, void* out, float* max_logits, float* lse,
                                    float* lse_indexer, const int* topk_length, int indexer_topk, int tokens, int heads,
                                    int head_dim, int value_dim, int kv_tokens, int selection_width,
                                    float softmax_scale) {
    return launch_sparse_attn_fwd_indexer_fused(stream, q, kv, attn_sink, topk_indices, out, max_logits, lse,
                                                lse_indexer, topk_length, indexer_topk, tokens, heads, head_dim,
                                                value_dim, kv_tokens, selection_width, softmax_scale);
}

bool bf16SparseAttnFwdIndexerSparseAttnFusedSupported() { return sparse_attn_fused_supported(); }

}  // namespace ftrain
