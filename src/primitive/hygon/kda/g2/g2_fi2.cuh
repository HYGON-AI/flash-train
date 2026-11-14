// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Handwritten gfx936 G2 kernels.
// clang-format off
// Shared device helpers of the forward intra stage (namespace fi2): the fp32 16x16x8 MMAC
// wrapper, the LDS tile strides, the C-pi fragment layout (the B operand's key rows are
// pi16-permuted, so every C fragment holds 4 contiguous columns) and the deterministic
// cross-wave reduction.  g2_fwd_prep.cuh builds the single-pass prep kernel on top of them.
#include "g2_common.h"

namespace g2 {

#ifndef G2_F32_MMAC32_DEFINED
#define G2_F32_MMAC32_DEFINED
typedef float f2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ f4 mmac32(f2 a, f2 b, f4 c) {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__)
    return __builtin_amdgcn_mmac_f32_16x16x8f32(a, b, c);
#else
    __builtin_trap();  // gfx936-only matrix instruction: the KDA Primitives reject other archs
    return c;
#endif
}
#endif

namespace fi2 {

constexpr int kTS = 20;             // merge tile row stride (floats, 16B-aligned rows)
constexpr int kTile = 16 * kTS;
constexpr int kT2 = 68;             // T2 [dim][token] row stride (bf16)

// Wave-scope LDS ordering.  LDS ops of one wave execute in order; this only has
// to stop the compiler from moving LDS accesses across it and drain lgkmcnt.
// (NO s_barrier: it is called from wave-divergent code.)
__device__ __forceinline__ void wave_lds_order() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}
__device__ __forceinline__ v4bh pack4(f4 x) {
    v4bh r;
    r[0] = bf_rne(x[0]); r[1] = bf_rne(x[1]); r[2] = bf_rne(x[2]); r[3] = bf_rne(x[3]);
    return r;
}
__device__ __forceinline__ constexpr int pidx(int rb, int cb) { return rb * (rb - 1) / 2 + cb; }
__device__ __forceinline__ float lo_bf(uint32_t w) { return bf2f((short)(w & 0xffffu)); }
__device__ __forceinline__ float hi_bf(uint32_t w) { return bf2f((short)(w >> 16)); }

// C-pi (A) x LDS tile (B, row-major stride kTS) -> C-pi.  k-slot bijection:
// tile kt, lane q4, element i  <->  physical index 4*q4 + 2*kt + i (A col / B row).
__device__ __forceinline__ f4 mmB(f4 a, const float* Bt, int q4, int pm) {
    f4 c = f4{0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        const f2 a2 = f2{a[2 * kt], a[2 * kt + 1]};
        const f2 b2 = f2{Bt[(4 * q4 + 2 * kt) * kTS + pm], Bt[(4 * q4 + 2 * kt + 1) * kTS + pm]};
        c = mmac32(a2, b2, c);
    }
    return c;
}

}  // namespace fi2

}  // namespace g2
