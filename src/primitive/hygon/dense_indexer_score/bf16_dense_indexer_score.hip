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
#include "flash_train/primitive/hygon/dense_indexer_score/bf16_dense_indexer_score.hpp"

// Multi-query index-score implementation.
// Four query waves share one 64-key BF16 tile; each wave retains its own Q.
#if (defined(__gfx936__) || defined(__gfx938__))
#    include <du_mma.h>
#endif
namespace hip_dsa_mla64 {
using bf = __hip_bfloat16;
#if (defined(__gfx936__) || defined(__gfx938__))
using namespace du::dumma;
using AF   = DUFragment<matrix_a, 16, 16, 16, bf, row_major>;
using BC   = DUFragment<matrix_b, 16, 16, 16, bf, col_major>;
using BR   = DUFragment<matrix_b, 16, 16, 16, bf, row_major>;
using CF   = DUFragment<accumulator, 16, 16, 16, float>;
using Desc = int __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void bar() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
__device__ __forceinline__ void waitn(int n) {
    if (n == 3) asm volatile("s_waitcnt vmcnt(3)" ::: "memory");
    if (n == 2) asm volatile("s_waitcnt vmcnt(2)" ::: "memory");
    if (n == 1) asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
    if (n == 0) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    bar();
}
__device__ __forceinline__ void ld(Desc desc, int off, bf* dst) {
    asm volatile(
        "s_mov_b32 m0, %1\n\ts_nop 0\n\tbuffer_load_dwordx4 %0, %2, 0 " "offen lds" ::"v"(off),
        "s"((unsigned)(size_t)dst), "s"(desc)
        : "m0", "memory");
}
#endif
}  // namespace hip_dsa_mla64
namespace hip_dsa_index_multi {
using namespace hip_dsa_mla64;
#if (defined(__gfx936__) || defined(__gfx938__))
using saf_f2 = float __attribute__((ext_vector_type(2)));
__device__ __forceinline__ saf_f2 saf_pk_mul(saf_f2 a, saf_f2 b) {
    saf_f2 c;
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
    return c;
}
__device__ __forceinline__ saf_f2 saf_pk_add(saf_f2 a, saf_f2 b) {
    saf_f2 c;
    asm volatile("v_pk_add_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
    return c;
}
__device__ __forceinline__ saf_f2 saf_pk_fma(saf_f2 a, saf_f2 b, saf_f2 c) {
    saf_f2 d;
    asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}
__device__ __forceinline__ saf_f2 saf_exp_pair(saf_f2 x) {
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
__device__ __forceinline__ float target_xor(float x, int off) {
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

template<int H, int MODE, int NT = 256, bool THD = false>
__global__ __launch_bounds__(NT, 1) void index_score(const bf* q, const bf* kv, const bf* weights, const int* idx,
                                                     float* out, float* denom, int T, int S, int K, int ratio,
                                                     float scale, const int* cuq = nullptr, const int* cuk = nullptr,
                                                     const int* offsets = nullptr, int nseq = 0) {
    // One 64x128 BF16 K tile occupies16KiB; H64 is primarily VGPR-limited.
    constexpr int D      = 128;
    constexpr int STAGES = 4;
    constexpr int QGROUP = 4;
    const int lane = threadIdx.x & 63, w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), r = lane & 15,
              g = lane >> 4;
    int group = blockIdx.x, qbase = blockIdx.y * T, kbase = blockIdx.y * S, offset = 0;
    if constexpr (THD) {
        int lo = 0, hi = nseq;
        while (lo + 1 < hi) {
            int mid = (lo + hi) / 2;
            if (cuq[mid] / 4 + mid <= group)
                lo = mid;
            else
                hi = mid;
        }
        group  -= cuq[lo] / 4 + lo;
        qbase   = cuq[lo];
        kbase   = cuk[lo];
        T       = cuq[lo + 1] - qbase;
        S       = cuk[lo + 1] - kbase;
        offset  = offsets ? offsets[lo] : 0;
        if (group * 4 >= T) return;
    }
    const int qgroup = w, qt = group * QGROUP + qgroup, t = qbase + qt, h = r;
    const int valid      = qt < T ? int(max(0LL, min(((long long)qt + offset + 1) / ratio, (long long)S))) : 0;
    const int loop_limit = int(max(0LL, min(((long long)min((group + 1) * QGROUP, T) + offset) / ratio, (long long)S)));
    constexpr int PH     = H;
    kv += (size_t)kbase * D;
    __shared__ __align__(16) bf sm[STAGES * 2048];
    static_assert(MODE == 0 || MODE == 1, "Only score and score/LSE modes are validated");
    // The post-MMAC barrier retires all K readers before head sums.
    // The end-of-tile barrier retires sums readers before K staging resumes.
    float* sums                 = reinterpret_cast<float*>(sm);
    const unsigned long long ka = (unsigned long long)kv;
    const Desc kd               = {(int)ka, (int)((ka >> 32) | 0x40000000u), (int)0x80000000u, 0x00020000};
    AF qf[(H / 16)][D / 16];
#    pragma unroll
    for (int hp = 0; hp < (H / 16); hp++) {
        // Same aligned NN32 fragment pairing as attention forward.
        using QVec = unsigned int __attribute__((ext_vector_type(4)));
#    pragma unroll
        for (int j = 0; j < D / 32; j++) {
            QVec v = (qt < T && h + hp * 16 < H)
                         ? *(const QVec*)(q + ((size_t)t * PH + h + hp * 16) * D + j * 32 + g * 8)
                         : QVec{0, 0, 0, 0};
            asm volatile("" : "+v"(v)::"memory");
            *(unsigned long long*)&qf[hp][j * 2].x[0]     = ((unsigned long long*)&v)[0];
            *(unsigned long long*)&qf[hp][j * 2 + 1].x[0] = ((unsigned long long*)&v)[1];
        }
    }
    float lse[(H / 16)];
#    pragma unroll
    for (int hp = 0; hp < (H / 16); hp++)
        lse[hp] = (qt < T && h + hp * 16 < H) ? (float)weights[(size_t)t * PH + h + hp * 16] : 0.f;
    float row_mx = -INFINITY;
    for (int base = 0; base < loop_limit; base += 64) {
        int pos = base + lane, ix = pos < loop_limit ? pos : -1;
        // Exact validity of this query tile; retain the separate K-gather index.
        const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3), row = w * 16 + unsw / 32, col = unsw % 32;
        // The tile's original lane-distributed indices also serve K gathering
        // and score masks, with no assumptions about other query rows.
        int key     = base + row;
        int key2    = base + row + 32;
        auto fetchk = [&](int j) {
            int off = (key < loop_limit) ? key * (D * 2) + (j * 32 + col) * 2 : -1;
            ld(kd, off, sm + (j % STAGES) * 2048 + w * 512);
            if constexpr (NT == 128) {
                int off2 = (key2 < loop_limit) ? key2 * (D * 2) + (j * 32 + col) * 2 : -1;
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
        CF sc[(H / 16)][4];
#    pragma unroll
        for (int k = 0; k < 4; k++) {
#    pragma unroll
            for (int hp = 0; hp < (H / 16); hp++) du_fill_fragment(sc[hp][k], 0.f);
        }
#    pragma unroll
        for (int j = 0; j < STAGES; j++) fetchk(j);
#    pragma unroll
        for (int j = 0; j < D / 32; j++) {
            if (j == 0) waitn(0);
#    pragma unroll
            for (int k = 0; k < 4; k++) {
                using U4 = unsigned int __attribute__((ext_vector_type(4)));
                U4 v     = *(const U4*)(sm + (j % STAGES) * 2048 + frag_a[k]);
                BC b0, b1;
                *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
#    pragma unroll
                for (int hp = 0; hp < (H / 16); hp++) {
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
                const bool ok0 = base + k * 16 + g + 4 * z < loop_limit,
                           ok1 = base + k * 16 + g + 4 * (z + 1) < loop_limit;
                saf_f2 v       = {0.f, 0.f};
#    pragma unroll
                for (int hp = 0; hp < (H / 16); hp++) {
                    const float x = sc[hp][k].x[z] * scale, y = sc[hp][k].x[z + 1] * scale;
                    const float vx = (x > 0 || isnan(x)) ? x : 0.f, vy = (y > 0 || isnan(y)) ? y : 0.f;
                    v[0] += (ok0 && h + hp * 16 < H) ? vx * lse[hp] : 0.f;
                    v[1] += (ok1 && h + hp * 16 < H) ? vy * lse[hp] : 0.f;
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
        if (qt < T && pos < K) {
            float v;
            v = sums[w * 64 + lane];

            out[(size_t)t * K + pos] = (pos < valid) ? v : -INFINITY;
            if (pos < valid) row_mx = fmaxf(row_mx, v);
        }
        bar();
    }
    if (qt < T)
        for (int k = loop_limit + lane; k < K; k += 64) out[(size_t)t * K + k] = -INFINITY;
    if constexpr (MODE != 0) {
        bar();
        if (qt < T) {
            float mx = row_mx;
            for (int off = 32; off; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off));
            float sum = 0.f;
            for (int j = lane; j < valid; j += 64) sum += expf(out[(size_t)t * K + j] - mx);
            for (int off = 32; off; off >>= 1) sum += __shfl_xor(sum, off);
            if constexpr (MODE == 1) {
                bool any_nan = false;
                if (!isfinite(mx)) {
                    for (int j = lane; j < valid; j += 64) any_nan |= isnan(out[(size_t)t * K + j]);
                    any_nan = __ballot(any_nan) != 0;
                }
                if (lane == 0) denom[t] = valid ? (isfinite(mx) ? mx + logf(sum) : (any_nan ? NAN : mx)) : -INFINITY;
            }
            if constexpr (MODE == 3) {
                for (int j = lane; j < K; j += 64)
                    out[(size_t)t * K + j] = mx == -INFINITY ? 0.f : expf(out[(size_t)t * K + j] - mx) / sum;
            }
        }
    }
}

#else
template<int H, int MODE, int NT = 256, bool THD = false>
__global__ void index_score(const bf*, const bf*, const bf*, const int*, float*, float*, int, int, int, int, float,
                            const int* = nullptr, const int* = nullptr, const int* = nullptr, int = 0) {}
#endif
}  // namespace hip_dsa_index_multi
// Transposed index-score implementation.
// Adapted from fused_dsa/hip_dsa/kernels/softmax_exp.h.

namespace hip_dsa_numeric {
// Exhaustive binary32 comparison against DTK 26.04 expf on gfx936 found
// zero bit differences, including all NaN payloads and exponent boundaries.
// gfx936/gfx938 share this implementation; no reduced-precision mode.
__device__ __forceinline__ float softmax_exp(float x) {
#pragma clang fp contract(off)
#if (defined(__gfx936__) || defined(__gfx938__))
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

// Four query waves share one 64-key BF16 tile; each wave retains its own Q.
namespace hip_dsa_index_transposed {
using namespace hip_dsa_mla64;
#if (defined(__gfx936__) || defined(__gfx938__))
using saf_f2 = float __attribute__((ext_vector_type(2)));
__device__ __forceinline__ saf_f2 saf_pk_mul(saf_f2 a, saf_f2 b) {
    saf_f2 c;
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
    return c;
}
__device__ __forceinline__ saf_f2 saf_pk_add(saf_f2 a, saf_f2 b) {
    saf_f2 c;
    asm volatile("v_pk_add_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));
    return c;
}
__device__ __forceinline__ saf_f2 saf_pk_fma(saf_f2 a, saf_f2 b, saf_f2 c) {
    saf_f2 d;
    asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}
__device__ __forceinline__ saf_f2 saf_exp_pair(saf_f2 x) {
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
__device__ __forceinline__ float target_xor(float x, int off) {
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

// The common launcher selects exact unit-scale/LSE work where supported.
// Defaults retain the original arithmetic for every other route.
template<bool EXACT_EXP>
__device__ __forceinline__ float index_lse_exp(float x) {
    if constexpr (EXACT_EXP) return hip_dsa_numeric::softmax_exp(x);
    return expf(x);
}

template<int H, int MODE, int NT = 256, bool THD = false, bool FAST_UNIT_SCALE = false>
__global__ __launch_bounds__(NT, 1) void index_score(const bf* q, const bf* kv, const bf* weights, const int* idx,
                                                     float* out, float* denom, int T, int S, int K, int ratio,
                                                     float scale, const int* cuq = nullptr, const int* cuk = nullptr,
                                                     const int* offsets = nullptr, int nseq = 0) {
    // One 64x128 BF16 K tile occupies16KiB; H64 is primarily VGPR-limited.
    constexpr int D      = 128;
    constexpr int STAGES = 4;
    constexpr int QGROUP = 4;
    const int lane = threadIdx.x & 63, w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), r = lane & 15,
              g = lane >> 4;
    int group = blockIdx.x, qbase = blockIdx.y * T, kbase = blockIdx.y * S, offset = 0;
    if constexpr (THD) {
        int lo = 0, hi = nseq;
        while (lo + 1 < hi) {
            int mid = (lo + hi) / 2;
            if (cuq[mid] / 4 + mid <= group)
                lo = mid;
            else
                hi = mid;
        }
        group  -= cuq[lo] / 4 + lo;
        qbase   = cuq[lo];
        kbase   = cuk[lo];
        T       = cuq[lo + 1] - qbase;
        S       = cuk[lo + 1] - kbase;
        offset  = offsets ? offsets[lo] : 0;
        if (group * 4 >= T) return;
    }
    const int qgroup = w, qt = group * QGROUP + qgroup, t = qbase + qt, h = r;
    const int valid      = qt < T ? int(max(0LL, min(((long long)qt + offset + 1) / ratio, (long long)S))) : 0;
    const int loop_limit = int(max(0LL, min(((long long)min((group + 1) * QGROUP, T) + offset) / ratio, (long long)S)));
    constexpr int PH     = H;
    kv += (size_t)kbase * D;
    __shared__ __align__(16) bf sm[STAGES * 2048];
    static_assert(MODE == 0 || MODE == 1, "Only score and score/LSE modes are validated");
    // The post-MMAC barrier retires all K readers before head sums.
    // The end-of-tile barrier retires sums readers before K staging resumes.
    float* sums                 = reinterpret_cast<float*>(sm);
    const unsigned long long ka = (unsigned long long)kv;
    const Desc kd               = {(int)ka, (int)((ka >> 32) | 0x40000000u), (int)0x80000000u, 0x00020000};
    BC qf[(H / 16)][D / 16];
    if constexpr (MODE == 1 && NT == 256 && (H == 32 || H == 64)) {
        if constexpr (H == 64) {
            // Same eight QVec values spanning two complete head fragments.
            // Compiler-visible vector outputs keep all reads live before consumption.
            using QVec = unsigned int __attribute__((ext_vector_type(4)));
#    pragma unroll
            for (int hpbase = 0; hpbase < H / 16; hpbase += 2) {
                QVec v[2][4];
#    pragma unroll
                for (int u = 0; u < 2; u++) {
                    const bool ok = qt < T;
                    const QVec* ptr =
                        ok ? reinterpret_cast<const QVec*>(q + ((size_t)t * PH + h + (hpbase + u) * 16) * D + g * 8)
                           : nullptr;
#    pragma unroll
                    for (int j = 0; j < 4; j++) v[u][j] = ok ? ptr[j * 4] : QVec{0, 0, 0, 0};
                }
                asm volatile(""
                             : "+v"(v[0][0]), "+v"(v[0][1]), "+v"(v[0][2]), "+v"(v[0][3]), "+v"(v[1][0]), "+v"(v[1][1]),
                               "+v"(v[1][2]), "+v"(v[1][3])::"memory");
#    pragma unroll
                for (int u = 0; u < 2; u++) {
#    pragma unroll
                    for (int j = 0; j < 4; j++) {
                        *(unsigned long long*)&qf[hpbase + u][j * 2].x[0]     = ((unsigned long long*)&v[u][j])[0];
                        *(unsigned long long*)&qf[hpbase + u][j * 2 + 1].x[0] = ((unsigned long long*)&v[u][j])[1];
                    }
                }
            }
        } else {
            // Four independent 16-byte reads per head fragment. One opaque vector
            // group keeps all four live before fragments are consumed; same addresses
            // and bit packing as the original loader, without a wait after each read.
            using QVec = unsigned int __attribute__((ext_vector_type(4)));
#    pragma unroll
            for (int hp = 0; hp < H / 16; hp++) {
                const bool ok = qt < T && h + hp * 16 < H;
                const QVec* ptr =
                    ok ? reinterpret_cast<const QVec*>(q + ((size_t)t * PH + h + hp * 16) * D + g * 8) : nullptr;
                QVec v0 = ok ? ptr[0] : QVec{0, 0, 0, 0};
                QVec v1 = ok ? ptr[4] : QVec{0, 0, 0, 0};
                QVec v2 = ok ? ptr[8] : QVec{0, 0, 0, 0};
                QVec v3 = ok ? ptr[12] : QVec{0, 0, 0, 0};
                asm volatile("" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3)::"memory");
                QVec vectors[4] = {v0, v1, v2, v3};
#    pragma unroll
                for (int j = 0; j < 4; j++) {
                    *(unsigned long long*)&qf[hp][j * 2].x[0]     = ((unsigned long long*)&vectors[j])[0];
                    *(unsigned long long*)&qf[hp][j * 2 + 1].x[0] = ((unsigned long long*)&vectors[j])[1];
                }
            }
        }
    } else {
#    pragma unroll
        for (int hp = 0; hp < (H / 16); hp++) {
            // Same aligned NN32 fragment pairing as attention forward.
            using QVec = unsigned int __attribute__((ext_vector_type(4)));
#    pragma unroll
            for (int j = 0; j < D / 32; j++) {
                QVec v = (qt < T && h + hp * 16 < H)
                             ? *(const QVec*)(q + ((size_t)t * PH + h + hp * 16) * D + j * 32 + g * 8)
                             : QVec{0, 0, 0, 0};
                asm volatile("" : "+v"(v)::"memory");
                *(unsigned long long*)&qf[hp][j * 2].x[0]     = ((unsigned long long*)&v)[0];
                *(unsigned long long*)&qf[hp][j * 2 + 1].x[0] = ((unsigned long long*)&v)[1];
            }
        }
    }
    float lse[(H / 16)];
#    pragma unroll
    for (int hp = 0; hp < (H / 16); hp++)
        lse[hp] = (qt < T && h + hp * 16 < H) ? (float)weights[(size_t)t * PH + h + hp * 16] : 0.f;
    float weighted_heads[H / 16][4];
#    pragma unroll
    for (int hp = 0; hp < H / 16; hp++) {
#    pragma unroll
        for (int z = 0; z < 4; z++) weighted_heads[hp][z] = __shfl(lse[hp], g + 4 * z, 64);
    }
    float row_mx = -INFINITY, row_sum = 0.f;
    bool row_nan = false;
    for (int base = 0; base < loop_limit; base += 64) {
        int pos = base + lane, ix = pos < loop_limit ? pos : -1;
        // Exact validity of this query tile; retain the separate K-gather index.
        const int flat = lane * 8, unsw = flat ^ ((flat >> 6 & 7) << 3), row = w * 16 + unsw / 32, col = unsw % 32;
        // The tile's original lane-distributed indices also serve K gathering
        // and score masks, with no assumptions about other query rows.
        int key     = base + row;
        int key2    = base + row + 32;
        auto fetchk = [&](int j) {
            int off = (key < loop_limit) ? key * (D * 2) + (j * 32 + col) * 2 : -1;
            ld(kd, off, sm + (j % STAGES) * 2048 + w * 512);
            if constexpr (NT == 128) {
                int off2 = (key2 < loop_limit) ? key2 * (D * 2) + (j * 32 + col) * 2 : -1;
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
        CF sc[(H / 16)][4];
#    pragma unroll
        for (int k = 0; k < 4; k++) {
#    pragma unroll
            for (int hp = 0; hp < (H / 16); hp++) du_fill_fragment(sc[hp][k], 0.f);
        }
#    pragma unroll
        for (int j = 0; j < STAGES; j++) fetchk(j);
#    pragma unroll
        for (int j = 0; j < D / 32; j++) {
            if (j == 0) waitn(0);
            if constexpr (MODE == 1 && NT == 256 && (H == 32 || H == 64)) {
                // Same source-backed two-key grouping used by the validated predictor.
                // Each accumulator retains ascending j/half contractions exactly.
#    pragma unroll
                for (int keybase = 0; keybase < 4; keybase += 2) {
                    AF kcache[2][2];
#    pragma unroll
                    for (int key = 0; key < 2; key++) {
                        using U4   = unsigned int __attribute__((ext_vector_type(4)));
                        const U4 v = *(const U4*)(sm + (j % STAGES) * 2048 + frag_a[keybase + key]);
                        *(unsigned long long*)&kcache[key][0].x[0] = ((const unsigned long long*)&v)[0];
                        *(unsigned long long*)&kcache[key][1].x[0] = ((const unsigned long long*)&v)[1];
                    }
#    pragma unroll
                    for (int half = 0; half < 2; half++) {
#    pragma unroll
                        for (int key = 0; key < 2; key++) {
#    pragma unroll
                            for (int hp = 0; hp < H / 16; hp++)
                                du_mma_sync(sc[hp][keybase + key], kcache[key][half], qf[hp][j * 2 + half],
                                            sc[hp][keybase + key]);
                        }
                    }
                }
            } else {
#    pragma unroll
                for (int k = 0; k < 4; k++) {
                    using U4 = unsigned int __attribute__((ext_vector_type(4)));
                    U4 v     = *(const U4*)(sm + (j % STAGES) * 2048 + frag_a[k]);
                    AF b0, b1;
                    *(unsigned long long*)&b0.x[0] = ((unsigned long long*)&v)[0];
                    *(unsigned long long*)&b1.x[0] = ((unsigned long long*)&v)[1];
#    pragma unroll
                    for (int hp = 0; hp < (H / 16); hp++) {
                        du_mma_sync(sc[hp][k], b0, qf[hp][j * 2], sc[hp][k]);
                        du_mma_sync(sc[hp][k], b1, qf[hp][j * 2 + 1], sc[hp][k]);
                    }
                }
            }
        }

        bar();
#    pragma unroll
        for (int k = 0; k < 4; k++) {
            // C now has keys as rows and heads as columns. Keep the old head
            // summation tree: combine hp first, then head XOR8/4 locally, and
            // head XOR2/1 across the four lane groups. Only g0 writes each key.
            // Out/online-LSE use stricter pos<valid; unused key partials cannot contribute.
            const bool ok = FAST_UNIT_SCALE || base + k * 16 + r < loop_limit;
            saf_f2 v01 = {0.f, 0.f}, v23 = {0.f, 0.f};
#    pragma unroll
            for (int hp = 0; hp < H / 16; hp++) {
                const saf_f2 scale2 = {scale, scale};
                const saf_f2 x01    = FAST_UNIT_SCALE ? (saf_f2){sc[hp][k].x[0], sc[hp][k].x[1]}
                                                      : saf_pk_mul((saf_f2){sc[hp][k].x[0], sc[hp][k].x[1]}, scale2);
                const saf_f2 x23    = FAST_UNIT_SCALE ? (saf_f2){sc[hp][k].x[2], sc[hp][k].x[3]}
                                                      : saf_pk_mul((saf_f2){sc[hp][k].x[2], sc[hp][k].x[3]}, scale2);
                saf_f2 a01, a23;
#    pragma unroll
                for (int z = 0; z < 2; z++) {
                    a01[z] = ok && (x01[z] > 0 || isnan(x01[z])) ? x01[z] : 0.f;
                    a23[z] = ok && (x23[z] > 0 || isnan(x23[z])) ? x23[z] : 0.f;
                }
                v01 = saf_pk_fma(a01, (saf_f2){weighted_heads[hp][0], weighted_heads[hp][1]}, v01);
                v23 = saf_pk_fma(a23, (saf_f2){weighted_heads[hp][2], weighted_heads[hp][3]}, v23);
            }
            const saf_f2 sum    = saf_pk_add(v01, v23);
            const float partial = sum[0] + sum[1];
            // 32-bank LDS: xor bit4 with the head-group parity. Both producer
            // and consumer waves then have the natural two accesses per bank.
            sums[w * 256 + g * 64 + ((k * 16 + r) ^ ((g & 1) << 4))] = partial;
        }
        bar();
        if (qt < T && pos < K) {
            float v;
            const float p0 = sums[w * 256 + lane];
            const float p1 = sums[w * 256 + 64 + (lane ^ 16)];
            const float p2 = sums[w * 256 + 128 + lane];
            const float p3 = sums[w * 256 + 192 + (lane ^ 16)];
            v              = (p0 + p2) + (p1 + p3);

            out[(size_t)t * K + pos] = (pos < valid) ? v : -INFINITY;
            if (pos < valid) {
                if constexpr (MODE == 1) {
                    row_nan             |= isnan(v);
                    const float a        = index_lse_exp<FAST_UNIT_SCALE>(-fabsf(v - row_mx));
                    const float old_sum  = row_mx == -INFINITY ? 0.f : row_sum;
                    const float rise     = old_sum * a + 1.f;
                    const float stay     = old_sum + (row_mx == -INFINITY ? 0.f : a);
                    row_sum              = v > row_mx ? rise : stay;
                }
                row_mx = fmaxf(row_mx, v);
            }
        }
        bar();
    }
    if (qt < T)
        for (int k = loop_limit + lane; k < K; k += 64) out[(size_t)t * K + k] = -INFINITY;
    if constexpr (MODE != 0) {
        bar();
        if (qt < T) {
            float mx = row_mx;
            for (int off = 32; off; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off));
            if constexpr (MODE == 1) {
                float sum = row_mx == -INFINITY ? 0.f : row_sum * index_lse_exp<FAST_UNIT_SCALE>(row_mx - mx);
                for (int off = 32; off; off >>= 1) sum += __shfl_xor(sum, off);
                const bool any_nan = __ballot(row_nan) != 0;
                if (lane == 0) denom[t] = valid ? (any_nan ? NAN : (isfinite(mx) ? mx + logf(sum) : mx)) : -INFINITY;
            } else {
                float sum = 0.f;
                for (int j = lane; j < valid; j += 64)
                    sum += index_lse_exp<FAST_UNIT_SCALE>(out[(size_t)t * K + j] - mx);
                for (int off = 32; off; off >>= 1) sum += __shfl_xor(sum, off);
                if constexpr (MODE == 1) {
                    bool any_nan = false;
                    if (!isfinite(mx)) {
                        for (int j = lane; j < valid; j += 64) any_nan |= isnan(out[(size_t)t * K + j]);
                        any_nan = __ballot(any_nan) != 0;
                    }
                    if (lane == 0)
                        denom[t] = valid ? (isfinite(mx) ? mx + logf(sum) : (any_nan ? NAN : mx)) : -INFINITY;
                }
            }
            if constexpr (MODE == 3) {
                float sum = 0.f;
                for (int j = lane; j < valid; j += 64)
                    sum += index_lse_exp<FAST_UNIT_SCALE>(out[(size_t)t * K + j] - mx);
                for (int off = 32; off; off >>= 1) sum += __shfl_xor(sum, off);
                for (int j = lane; j < K; j += 64)
                    out[(size_t)t * K + j] =
                        mx == -INFINITY ? 0.f : index_lse_exp<FAST_UNIT_SCALE>(out[(size_t)t * K + j] - mx) / sum;
            }
        }
    }
}

#else
template<int H, int MODE, int NT = 256, bool THD = false, bool FAST_UNIT_SCALE = false>
__global__ void index_score(const bf*, const bf*, const bf*, const int*, float*, float*, int, int, int, int, float,
                            const int* = nullptr, const int* = nullptr, const int* = nullptr, int = 0) {}
#endif
}  // namespace hip_dsa_index_transposed

#define DENSE_CHUNK 32
#define DENSE_THREADS 256
#define DENSE_DCHUNK 128
#define DENSE_MAX_TILES 36  // D <= 576 (DSv4 absorbed MLA)

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

static inline size_t dense_mmac_shm_v2(int D, int Sk) {
    return (size_t)DENSE_CHUNK * (D + 8) * 2 + (size_t)Sk * 4 + DENSE_THREADS * 4;
}

// Adapted from fused_dsa/hip_dsa/kernels/dense_mmac.hip.

typedef uint16_t bf16;

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

#if defined(__gfx936__) || defined(__gfx938__)

using namespace du::dumma;
typedef __hip_bfloat16 hbf16;

__device__ __forceinline__ static void dm_zero_bf16(hbf16* dst, int n) {
    char* p = reinterpret_cast<char*>(dst);
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

template<int MODE, int DK>
static __global__ void __launch_bounds__(DENSE_THREADS)
    dense_score_mmac_kernel_v2(const bf16*, const bf16*, const bf16*, const float*, float*, float*, int, int, int, int,
                               int, int, float, int) {}

#endif  // supported gfx archs

namespace ftrain {
namespace {

template<class Implementation, int MODE, int WIDTH, int HEADS>
static inline void launch_dense_score_mmac_dispatch(hipStream_t stream, const void* q, const void* k, const void* w,
                                                    const float* lse, float* out, float* denom, int B, int Sq, int Sk,
                                                    int Hq, int D, int ratio, float scale, int score_stride = 0);
template<class Implementation, int WIDTH, int HEADS>
static void launch_dense_indexer_score_mmac(hipStream_t stream, const void* q, const void* k, const void* w, float* out,
                                            float* denom, int B, int Sq, int Sk, int Hq, int D, int ratio,
                                            float sm_scale, int score_stride = 0);

template<class Implementation, int MODE, int WIDTH, int HEADS>
static inline void launch_dense_score_mmac_dispatch(hipStream_t stream, const void* q, const void* k, const void* w,
                                                    const float* lse, float* out, float* denom, int B, int Sq, int Sk,
                                                    int Hq, int D, int ratio, float scale, int score_stride) {
    if (score_stride <= 0) score_stride = Sk;

    // Hq > 64 (DSv4-pro: 128 heads) runs the large-D kernel, which processes
    // heads in 64-head passes; the v2/v1 kernels cover Hq <= 64 only.
    if constexpr ((Implementation::kLargeD)) {
        hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, WIDTH>), B * Sq, DENSE_THREADS,
                           dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k, (const bf16*)w,
                           lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_dense_indexer_score HIP call failed: %s",
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
    if constexpr ((Implementation::kHoisted)) {
        size_t shm = dense_mmac_shm_v2(D, Sk);

        hipLaunchKernelGGL((dense_score_mmac_kernel_v2<MODE, WIDTH / 16>), B * Sq, DENSE_THREADS, shm, stream,
                           (const bf16*)q, (const bf16*)k, (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D, ratio,
                           scale, score_stride);
        {
            const hipError_t status = hipGetLastError();
            if (status != hipSuccess) {
                throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_dense_indexer_score HIP call failed: %s",
                                        hipGetErrorString(status));
            }
        }
        return;
    }
    if constexpr (WIDTH > DENSE_DCHUNK) {
        hipLaunchKernelGGL((dense_score_mmac_largeD_kernel<MODE, WIDTH>), B * Sq, DENSE_THREADS,
                           dense_mmac_shm_largeD_tpl(Hq, D, Sk), stream, (const bf16*)q, (const bf16*)k, (const bf16*)w,
                           lse, out, denom, B, Sq, Sk, Hq, D, ratio, scale, score_stride);
    } else {
        hipLaunchKernelGGL((dense_score_mmac_kernel<MODE>), B * Sq, DENSE_THREADS, dense_mmac_shm(Hq, D, Sk, false),
                           stream, (const bf16*)q, (const bf16*)k, (const bf16*)w, lse, out, denom, B, Sq, Sk, Hq, D,
                           ratio, scale, score_stride);
    }
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_dense_indexer_score HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
}

template<class Implementation, int WIDTH, int HEADS>
static void launch_dense_indexer_score_mmac(hipStream_t stream, const void* q, const void* k, const void* w, float* out,
                                            float* denom, int B, int Sq, int Sk, int Hq, int D, int ratio,
                                            float sm_scale, int score_stride) {
    launch_dense_score_mmac_dispatch<Implementation, 1, WIDTH, HEADS>(stream, q, k, w, nullptr, out, denom, B, Sq, Sk,
                                                                      Hq, D, ratio, sm_scale, score_stride);
}

template<class Implementation, int WIDTH, int HEADS>
static bool launch_dense_indexer_score(hipStream_t stream, const void* q, const void* k, const void* w, float* out,
                                       float* denom, int B, int Sq, int Sk, int Hq, int, int D, int ratio,
                                       float sm_scale) {
    // Fused single-kernel path (MMAC tensor core) only; no non-fused
    // fallback and no hipblas GEMM anywhere in this operator.
    launch_dense_indexer_score_mmac<Implementation, WIDTH, HEADS>(stream, q, k, w, out, denom, B, Sq, Sk, Hq, D, ratio,
                                                                  sm_scale);
    {
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "bf16_dense_indexer_score HIP call failed: %s",
                                    hipGetErrorString(status));
        }
    }
    return true;
}
}  // namespace

}  // namespace ftrain

namespace ftrain {

template<class Implementation, int WIDTH, int HEADS>
bool launchBf16DenseIndexerScore(FTrainStream stream, const void* q, const void* k, const void* w, float* out,
                                 float* denom, int batch, int query_length, int key_length, int query_heads,
                                 int kv_heads, int head_dim, int ratio, float sm_scale) {
    if constexpr ((Implementation::kMultiQuery) || (Implementation::kTransposed)) {
        static_assert(WIDTH == 128, "index fast path requires D=128");
        using hip_dsa_mla64::bf;
        if constexpr ((Implementation::kMultiQuery)) {
            static_assert(HEADS == 16 || HEADS == 32 || HEADS == 64, "unsupported index fast-path HEADS");
            hipLaunchKernelGGL((hip_dsa_index_multi::index_score<HEADS, 1, 256, false>),
                               dim3((query_length + 3) / 4, batch), 256, 0, stream, (const bf*)q, (const bf*)k,
                               (const bf*)w, nullptr, out, denom, query_length, key_length, key_length, ratio,
                               sm_scale);
        }
        if constexpr ((Implementation::kTransposed)) {
            static_assert(HEADS == 32 || HEADS == 64, "unsupported index fast-path HEADS");
            hipLaunchKernelGGL((hip_dsa_index_transposed::index_score<HEADS, 1, 256, false>),
                               dim3((query_length + 3) / 4, batch), 256, 0, stream, (const bf*)q, (const bf*)k,
                               (const bf*)w, nullptr, out, denom, query_length, key_length, key_length, ratio,
                               sm_scale);
        }
        const auto status = hipGetLastError();
        if (status != hipSuccess) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "dense fast launch failed: %s",
                                    hipGetErrorString(status));
        }
        return true;
    }

    return launch_dense_indexer_score<Implementation, WIDTH, HEADS>(
        stream, q, k, w, out, denom, batch, query_length, key_length, query_heads, kv_heads, head_dim, ratio, sm_scale);
}

// Explicit fixed launch instances used by the C++ configuration code.
#define FTRAIN_INSTANTIATE_LAUNCH(Implementation, WIDTH, HEADS)                                                \
    template bool launchBf16DenseIndexerScore<Implementation, WIDTH, HEADS>(                                   \
        FTrainStream stream, const void* q, const void* k, const void* w, float* out, float* denom, int batch, \
        int query_length, int key_length, int query_heads, int kv_heads, int head_dim, int ratio, float sm_scale);
#define FTRAIN_INSTANTIATE_LAUNCH_SET_1(Implementation, HEADS) \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, 128, HEADS);     \
    FTRAIN_INSTANTIATE_LAUNCH(Implementation, 64, HEADS);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScore, 0, 0);
FTRAIN_INSTANTIATE_LAUNCH_SET_1(Bf16DenseIndexerScoreHoisted, 0);
FTRAIN_INSTANTIATE_LAUNCH_SET_1(Bf16DenseIndexerScoreHoisted, 128);
FTRAIN_INSTANTIATE_LAUNCH_SET_1(Bf16DenseIndexerScoreHoisted, 16);
FTRAIN_INSTANTIATE_LAUNCH_SET_1(Bf16DenseIndexerScoreHoisted, 32);
FTRAIN_INSTANTIATE_LAUNCH_SET_1(Bf16DenseIndexerScoreHoisted, 64);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 112, 0);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 128, 0);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 16, 0);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 256, 0);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 32, 0);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 448, 0);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 48, 0);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 512, 0);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 576, 0);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 64, 0);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 80, 0);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreLargeD, 96, 0);

FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreMultiQuery, 128, 16);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreMultiQuery, 128, 32);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreMultiQuery, 128, 64);

FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreTransposed, 128, 32);
FTRAIN_INSTANTIATE_LAUNCH(Bf16DenseIndexerScoreTransposed, 128, 64);
#undef FTRAIN_INSTANTIATE_LAUNCH_SET_1
#undef FTRAIN_INSTANTIATE_LAUNCH

}  // namespace ftrain
