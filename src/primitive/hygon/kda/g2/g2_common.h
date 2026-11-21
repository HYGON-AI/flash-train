// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// G2 common device helpers -- extracted verbatim from the verified prototypes.
//   g2_engine.hip sha256(ca9f6fa3...)  : bf16/v4bh/f4/mmac/bf_rne/lds_barrier/ld8
//   p3_wy.hip     sha256(260f2315...)  : ld16f/bfr/bf2f/pi16/xsum16/msum
// The two prototypes name the same helpers differently (bf_rne vs bfr); both spellings
// are kept so the extracted kernel bodies stay byte-identical to their sources.
#ifndef G2_COMMON_H
#define G2_COMMON_H

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "flash_train/env.h"

namespace g2 {

typedef short v4bh __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef __hip_bfloat16 bf16;

__device__ __forceinline__ int c_row(int l) { return l & 15; }
// C-fragment column mapping of the 16x16x16 MMAC on gfx936 (validated at runtime by
// the prototype selftest): element e of lane l sits at column (l >> 4) + 4 * e.
__device__ __forceinline__ int c_col(int l, int e) { return (l >> 4) + 4 * e; }

__device__ __forceinline__ f4 mmac(v4bh a, v4bh b, f4 c) {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__)
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
#elif defined(__gfx938__) && __has_builtin(__builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts)
    // gfx938's native MMAC builtin with lit = lts = false: the same operation as gfx936's
    // v_mmac (bit-identical); the generic amdgcn builtin lowers with lit / lts set on gfx938.
    return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, c, false, false);
#else
    __builtin_trap();  // gfx936-only matrix instruction: the KDA Primitives reject other archs
    return c;
#endif
}
__device__ __forceinline__ short bfbits(float x) {
    return (short)__bfloat16_as_ushort(__float2bfloat16(x));
}
// CTA barrier that orders LDS only.  HIP's __syncthreads() also drains vmcnt, which
// would force every in-flight next-chunk prefetch to land before the barrier.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}
__device__ __forceinline__ v4bh ld8(const bf16* p) {  // 4 consecutive bf16 (8-byte aligned)
    return *reinterpret_cast<const v4bh*>(p);
}
__device__ __forceinline__ f4 ld16f(const float* p) { return *reinterpret_cast<const f4*>(p); }

#if defined(__gfx938__)
#define G2_SIGFAST 1
// kra P-938sg: the sigmoid's expf and divide without their range handling, valid where proven exhaustively on gfx938
// (rcp_exh.hip / exp_exh.hip): expf_core == expf for every fp32 |x| <= 87 and every NaN (the ocml sequence of the
// prep_a ISA minus the overflow / underflow selects); rcp1 == 1.f / d for every fp32 d in [1, 2^126) (the divide
// sequence with numerator 1, no scale / fixup; NaN payloads differ, so NaN takes the reference path).
__device__ __forceinline__ float expf_core(float x) {
    #pragma clang fp contract(off)
    const float ph = x * __uint_as_float(0x3fb8aa3bu);
    float pl = __builtin_fmaf(x, __uint_as_float(0x3fb8aa3bu), -ph);
    const float e = __builtin_rintf(ph);
    pl = __builtin_fmaf(x, __uint_as_float(0x32a5705fu), pl);
    const float t = (ph - e) + pl;
    return __builtin_amdgcn_ldexpf(__builtin_amdgcn_exp2f(t), (int)e);
}
__device__ __forceinline__ float rcp1(float d) {
    #pragma clang fp contract(off)
    const float r0 = __builtin_amdgcn_rcpf(d);
    const float r1 = __builtin_fmaf(__builtin_fmaf(-d, r0, 1.f), r0, r0);
    const float q1 = __builtin_fmaf(__builtin_fmaf(-d, r1, 1.f), r1, r1);
    return __builtin_fmaf(__builtin_fmaf(-d, q1, 1.f), r1, q1);
}
// wave-uniform selector: every argument |x| <= 87 and none NaN (a NaN anywhere makes the running sum NaN)
__device__ __forceinline__ void g2_sig_acc(float x, float& mx, float& sx) { mx = fmaxf(mx, fabsf(x)); sx += x; }
__device__ __forceinline__ bool g2_sig_fast(float mx, float sx) { return __all(mx <= 87.f && sx == sx); }
// kra P-938b1 (B class, numerics change -- gated against the FP64 oracle): the safe-gate sigmoid as bare v_exp_f32 +
// v_rcp_f32 (~1e-6 relative), applied consistently in prep_a, the backward gate kernel and bwd_tail
#ifndef KDA_B1_SIG
#define KDA_B1_SIG 1
#endif
// kra P-938t2 (B class, numerics change -- gated against the FP64 oracle): bwd_tail's per-CTA dA_log partial (512
// fp32 terms) as 64 lane runs of 8 + a fixed shfl-xor tree on wave 0, instead of one lane's 512-add chain
#ifndef KDA_B2_TREE
#define KDA_B2_TREE 1
#endif
__device__ __forceinline__ float g2_sig_b1(float x) {
    return __builtin_amdgcn_rcpf(1.f + __builtin_amdgcn_exp2f(-x * 1.4426950408889634f));
}
#endif

// kra P-938t: gfx938 MLS as a tile copier: one wave instruction moves 32 rows x 64 B (row stride stride_b16 2-byte
// elements) from memory into 2 KB of contiguous LDS, no VGPRs, 0.29x the TA work per byte of per-lane loads
// (ta_forms.hip).  Tracked by vmcnt.
#if defined(__gfx938__) && __has_builtin(__builtin_hcu_matrix_load_32x32_b16)
#define G2_MLS938 1
__device__ __forceinline__ void g2_mls_tile(const void* g, int stride_b16, void* lds) {
    typedef int v4i_t __attribute__((ext_vector_type(4)));
    const uint64_t a = reinterpret_cast<uint64_t>(g) & 0xffffffffffffull;
    v4i_t r;
    r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a);
    r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32));
    r[2] = __builtin_amdgcn_readfirstlane(stride_b16);
    r[3] = 0;
    const int so = __builtin_amdgcn_readfirstlane((int)(size_t)lds);
    __builtin_hcu_matrix_load_32x32_b16(r, (__attribute__((address_space(3))) short*)(so), 0, 0, 0, 0, 0);
}
#else
#define G2_MLS938 0
#endif
// kra L16 paired native layout (Wn Wb Un Kn KgA Qn QgA AqkA): inside every 512-element group
// the two 256-element sub-blocks are interleaved per lane (o = 256s + 4l + e -> 8l + 4s + e),
// so a lane's fragments of sub-blocks 2j and 2j+1 are ONE aligned 16 B load (a dwordx2 costs
// ~16 issue cycles per CU, a dwordx4 ~18-23: measured, kra vmem_issue).  Bijective per 1 KB
// group; commutes with any 512-aligned base.  Values unchanged -> bit-identical.
__device__ __forceinline__ size_t nat16(size_t o) {
    return (o & ~(size_t)511) | (((o >> 2) & 63) << 3) | (((o >> 8) & 1) << 2) | (o & 3);
}
typedef short v8bh_l16 __attribute__((ext_vector_type(8)));
// store both halves of a pair at once (p = base(512-aligned) + 512*j + 8*lane)
__device__ __forceinline__ void st16x2(void* p, v4bh a, v4bh b) {
    *reinterpret_cast<v8bh_l16*>(p) = __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7);
}
// fragments of sub-blocks 2j, 2j+1: p = base(512-aligned) + 512*j + 8*lane
__device__ __forceinline__ void ld16x2(const bf16* p, v4bh& a, v4bh& b) {
    const v8bh_l16 x = *reinterpret_cast<const v8bh_l16*>(p);
    a = __builtin_shufflevector(x, x, 0, 1, 2, 3);
    b = __builtin_shufflevector(x, x, 4, 5, 6, 7);
}
// RNE bf16 conversion for finite inputs, identical to __float2bfloat16.
__device__ __forceinline__ short bf_rne(float x) {
#if defined(__gfx938__) && __has_builtin(__builtin_hcu_cvt_pk_bf16_f32)
    // gfx938's v_cvt_pk_bf16_f32 rounds to nearest even and equals the integer sequence below for
    // every non-NaN fp32 input (all 2^32 inputs checked on BW1102; NaNs come out quieted): one VALU
    // instead of three.
    return (short)(uint32_t)__builtin_hcu_cvt_pk_bf16_f32(0, x, 0, x, false);
#else
    uint32_t u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (short)(u >> 16);
#endif
}
// The integer RNE sequence on every arch.  Kept where gfx938's v_cvt_pk_bf16_f32 does not pay: the
// recurrence kernels (fwd_h / dhu: latency-chain bound, their schedule moved +-25% with it) and the
// l2norm backward in bwd_tail (the compiler contracts the surrounding fp differently).
__device__ __forceinline__ short bf_rne_int(float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (short)(u >> 16);
}
__device__ __forceinline__ short bfr(float x) { return bf_rne(x); }
__device__ __forceinline__ float bf2f(short s) { return __uint_as_float(((uint32_t)(unsigned short)s) << 16); }
// P3's output-column permutation: lane index n reads source column 4*(n&3) + (n>>2).
__device__ __forceinline__ int pi16(int n) { return 4 * (n & 3) + (n >> 2); }
// kra P-52: opaque register copy (ends the source's live range at this point; value unchanged)
__device__ __forceinline__ float g2_vcopy(float x) {
    float y;
    asm volatile("v_mov_b32 %0, %1" : "=v"(y) : "v"(x));
    return y;
}
__device__ __forceinline__ float xsum16(float x) {     // sum over the 4 lanes sharing (l&15)
    x += __shfl_xor(x, 16, 64); x += __shfl_xor(x, 32, 64); return x;
}
__device__ __forceinline__ float msum(float x) {        // sum over the 16 lanes sharing (l>>4)
    x += __shfl_xor(x, 1, 64); x += __shfl_xor(x, 2, 64); x += __shfl_xor(x, 4, 64); x += __shfl_xor(x, 8, 64);
    return x;
}

// ---------------------------------------------------------------------------
// The ONE native-layout addressing authority (REVIEW_S3-12 §1).  Three rounds
// produced the same bug class -- chunk base double-added (wu, dAv) and the
// row stride missing its H factor (layout2) -- all invisible at single-chunk,
// H=1 debug shapes.  Every new kernel addresses global memory through these;
// hand-written base arithmetic is banned in review.
//   rm_tok:   [B,T,H,D] element offset for token `tok` (global, NOT chunk-local)
//   nat_blk:  native-tensor block base; p3=false -> (hh*NT + i), p3=true ->
//             (i*H + hh); nslot = elements per (chunk, head) block.
__device__ __forceinline__ size_t g2_rm_tok(long tok, int hh, int H, int D) {
    return ((size_t)tok * H + hh) * D;
}
// kra P-24: L2-aware concurrency cap.  Returns the dynamic-LDS bytes that lift a kernel's static
// LDS to `target_total` per CTA (so that floor(64 KB / target_total) CTAs fit per CU); 0 if the
// kernel is already larger.  The static size comes from the code object (cached per kernel).
template <typename K>
inline size_t g2_lds_topup(K kernel, size_t target_total) {
    hipFuncAttributes a{};
    if (hipFuncGetAttributes(&a, reinterpret_cast<const void*>(kernel)) != hipSuccess) return 0;
    return a.sharedSizeBytes < target_total ? target_total - a.sharedSizeBytes : 0;
}
// flash-train: the G2 tuning knobs (KDA_G2_*, G2_*_STOP) are hip_kda research switches; they are
// read only under FTRAIN_KDA_DEV (flash_train/env.h), otherwise every knob takes its default, so a
// training environment set up for hip_kda cannot steer flash-train's kernels.
inline bool g2_dev() {
    static const bool on = [] {
        const char* e = getenv(FTRAIN_KDA_DEV);
        return e != nullptr && *e != '\0' && strcmp(e, "0") != 0;
    }();
    return on;
}
inline const char* g2_getenv(const char* name) { return g2_dev() ? getenv(name) : nullptr; }
inline long g2_env_long(const char* name, long dflt) {
    const char* e = g2_getenv(name);
    return e ? atol(e) : dflt;
}
// kra P-938z: is the current device a gfx938 (BW1102)?  Cached per device (host side, launch decisions only).
inline bool g2_cur_gfx938() {
    static int known[64] = {};   // 0 unknown, 1 gfx938, 2 other
    int d = 0;
    if (hipGetDevice(&d) != hipSuccess || d < 0 || d >= 64) return false;
    if (known[d] == 0) {
        hipDeviceProp_t prop;
        known[d] = (hipGetDeviceProperties(&prop, d) == hipSuccess && strncmp(prop.gcnArchName, "gfx938", 6) == 0) ? 1 : 2;
    }
    return known[d] == 1;
}
// kra P-61e / P-938z: the lane-permuted Kn / natural-k Gn layout (kp) for H <= 20; on gfx938 not at H <= 8, where the
// 16-column fwd_h slabs (8 CTAs per head, P-938z) need the plain layout.  One rule for every producer / consumer.
inline bool g2_kp_layout(long H) {
    if (const char* e = g2_getenv("KDA_G2_KPERM_MAX_H")) return H <= atol(e);
    if (H <= 8 && g2_cur_gfx938()) return false;
    return H <= 20;
}


// kra P-35: stamp a pointer global (generic -> addrspace(1) -> generic); InferAddressSpaces then
// emits global_* for every access through it, even after the value went through memory (lambdas).
template <typename T>
__device__ __forceinline__ T* g2_glb(T* x) {
    return (T*)(__attribute__((address_space(1))) T*)x;
}

// kra P-938o2: CTA visiting order on a (NT, H) grid (wu's P-95 orders, shared).  Bitwise-neutral: every CTA is
// independent.  0 = chunk-fastest (blockIdx.x = chunk); 1 = head-fastest; 2 = diagonal (i = (L/H + hh) % NT);
// 3 = M adjacent heads per chunk (M | H).
__device__ __forceinline__ void g2_cta_order(int order, int m, int H, int NT, int& i, int& hh) {
    const int L = blockIdx.y * gridDim.x + blockIdx.x;
    if (order == 0) { i = blockIdx.x; hh = blockIdx.y; }
    else if (order == 3) { const int r = L / m; hh = (r / NT) * m + L % m; i = r % NT; }
    else { hh = L % H; i = order == 1 ? L / H : (L / H + hh) % NT; }
}

__device__ __forceinline__ size_t g2_nat_blk(int i, int hh, int H, int NT,
                                             bool p3, int nslot) {
    return (size_t)(p3 ? (size_t)i * H + hh : (size_t)hh * NT + i) * nslot;
}

// Varlen phase 2 (DESIGN_VARLEN_PHASE2.md): row-major [T,H,D] tensors are PACKED (real
// tokens only); native tensors stay chunk-padded (NT chunks, every sequence starts on a
// chunk).  Chunk i covers tokens ctok[i] .. ctok[i] + clen[i] - 1 (1 <= clen <= 64).
// Kernels are templated on VL (0 = dense: identity map, the original code; 1 = varlen) and
// a VL kernel branches ONCE per CTA (uniform) into a body of mode M:
//   M = 0  dense                 (VL = 0)
//   M = 1  mapped, FULL chunk     base = ctok[i], no masks -> the dense instruction stream
//   M = 2  mapped, PARTIAL chunk  rows r >= clen read as 0 (as the zero padding would) and
//          are never written; g rows are CLAMPED to row clen-1 (the padding decays by 0:
//          the chunk-local cumsum is flat there, the safe-gate references stay in range
//          and the chunk-end decay is the true last token's).
// Bodies are generic lambdas over g2_mode<M> so __shared__ storage is declared once.
template <int M> using g2_mode = std::integral_constant<int, M>;
template <int VL>
__device__ __forceinline__ long g2_row_tok(const int* ctok, int i) {
    return VL ? (long)ctok[i] : (long)i * 64;
}
template <int VL>
__device__ __forceinline__ int g2_row_len(const int* clen, int i) {
    return VL ? clen[i] : 64;
}
template <int M>
__device__ __forceinline__ bool g2_row_ok(int r, int len) {
    return M != 2 || r < len;
}
template <int M>
__device__ __forceinline__ int g2_row_g(int r, int len) {      // clamped row for g loads
    return (M != 2 || r < len) ? r : len - 1;
}
// Run body(g2_mode<M>{}) with M = 0 (dense), else 1 / 2 by the chunk's fill (CTA-uniform).
template <int VL, typename F>
__device__ __forceinline__ void g2_vl_dispatch(int len, F&& body) {
    if constexpr (VL == 0) {
        body(g2_mode<0>{});
    } else {
        if (len == 64) body(g2_mode<1>{});
        else body(g2_mode<2>{});
    }
}

// Wave-level LDS ordering (REVIEW_S3-15 rule 2).  Plain __syncwarp() carries NO
// LDS ordering -- the compiler hoisted LDS reads across it and the S3-14 merge
// raced.  LDS ops of one wave execute in order, so wave scope only needs the
// compiler fence + lgkmcnt drain.  NO s_barrier here (S3-16 fix): s_barrier is a
// CTA-wide rendezvous and deadlocks/mismatches when called from wave-divergent
// code.  Use lds_barrier() for CTA scope.
__device__ __forceinline__ void wave_lds_sync() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
}

constexpr int kK = 128, kV = 128, kBT = 64, kBV = 32, kPadK = kK + 4, kPadT = kBT + 4;
// P3 spells the same two axes differently (kD/kPT) in its source; both spellings are kept
// so the extracted kernel bodies stay byte-identical to p3_wy.hip.
constexpr int kD = 128, kPT = kBT + 4;

}  // namespace g2

#endif  // G2_COMMON_H
