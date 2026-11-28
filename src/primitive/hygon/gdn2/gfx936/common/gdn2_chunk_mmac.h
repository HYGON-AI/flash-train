// mmac / GEMM device helpers for the GDN2 chunked forward kernels.
// Copied from the GDN project (gdn_chunk_mmac.h), namespace renamed to gdn2_mmac.
#ifndef FLASH_TRAIN_GDN2_CHUNK_MMAC_H
#define FLASH_TRAIN_GDN2_CHUNK_MMAC_H

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gdn2_mmac {

__device__ __forceinline__ float bf2f(unsigned short b) {
    const unsigned int u = static_cast<unsigned int>(b) << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

__device__ __forceinline__ unsigned short f2bf(float x) {
    unsigned int u;
    __builtin_memcpy(&u, &x, 4);
    return static_cast<unsigned short>(
        ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16) & 0xffffu);
}

template <typename T>
__device__ __forceinline__ float load_elem(const T* p, std::size_t i);
template <>
__device__ __forceinline__ float load_elem<float>(const float* p, std::size_t i) {
    return p[i];
}
template <>
__device__ __forceinline__ float load_elem<uint16_t>(const uint16_t* p, std::size_t i) {
    return bf2f(p[i]);
}

template <typename T>
__device__ __forceinline__ void store_elem(T* p, std::size_t i, float x);
template <>
__device__ __forceinline__ void store_elem<float>(float* p, std::size_t i, float x) {
    p[i] = x;
}
template <>
__device__ __forceinline__ void store_elem<uint16_t>(uint16_t* p, std::size_t i, float x) {
    p[i] = f2bf(x);
}

// ---------------------------------------------------------------------------
// GEMM helpers.
//
// The chunk GEMMs all have the shape D = A B^T with both operands k-contiguous,
// which is exactly the mmac fragment layout.  For v_mmac_f32_16x16x16_bf16 the
// mapping (verified bit-exact against a scalar reference) is
//     A frag (lane,i): m = lane%16, k = 4*(lane/16)+i   -> A[m][k]
//     B frag (lane,i): n = lane%16, k = 4*(lane/16)+i   -> B[n][k]
//     D frag (lane,i): m = lane%16, n = (lane/16)+4*i   -> D[m][n]
// A and B use the identical pattern, so two tiles staged the same way multiply
// as A B^T with no transpose anywhere.
//
// Row padding is load bearing.  A fragment read is ds_read_b64 at element
// offset (lane%16)*STR + 4*(lane/16); with STR = KT the row term vanishes mod
// 32 banks and all 16 lanes of a quarter-wave collide.  STR = KT+4 (bf16) puts
// lane l on bank pair ((l%16 + l/16) * 2) mod 32, i.e. each 16-lane group
// covers all 32 banks exactly once -- the b64 minimum.  KT+2 does the same for
// fp32.  Measured worth of getting this right: 1.85x.
// ---------------------------------------------------------------------------

typedef short v4s __attribute__((ext_vector_type(4)));
typedef unsigned short u16x4 __attribute__((ext_vector_type(4)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef float v4f __attribute__((ext_vector_type(4)));

// D[MM][NN] = sum_k A[MM][ASTR] * B[NN][BSTR], bf16 operands, fp32 accumulate.
// Wave w takes output tiles w, w+NWAVES, ... of the (MM/16 x NN/16) tile grid.
// The caller supplies the epilogue: ep(m, n, value) is invoked once per output
// element with its position in the MM x NN tile.
//
// Every phase here used to follow its GEMM with a separate elementwise pass
// over the result -- masking M, masking tril(P), the Y transform, the O store.
// Each of those cost a barrier and a full round trip through LDS for a value
// the producing lane already held in a register.  With 64 KiB of LDS only one
// block is resident per CU, so there is no second block to cover a barrier and
// every one is fully exposed; at ~27 barriers per chunk they were more than
// half the chunk time, against 2.4% for mmac issue.  Folding the pass into the
// epilogue removes the barrier and the round trip together.
template <int MM, int NN, int KC, int ASTR, int BSTR, int THREADS, typename Ep>
__device__ void gemm_nt_mmac_ep(const unsigned short* __restrict__ A,
                                const unsigned short* __restrict__ B, Ep ep) {
    static_assert(MM % 16 == 0 && NN % 16 == 0 && KC % 16 == 0, "mmac tile mismatch");
    static_assert(ASTR % 4 == 0 && BSTR % 4 == 0, "fragment reads need 8B alignment");
    constexpr int NTN = NN / 16, NTILES = (MM / 16) * NTN, NWAVES = THREADS / 64;
    const int lane = threadIdx.x & 63;
    const int lo = lane & 15, hi = (lane >> 4) * 4;
    for (int t = threadIdx.x >> 6; t < NTILES; t += NWAVES) {
        const int tm = t / NTN, tn = t % NTN;
        const unsigned short* ap = A + (tm * 16 + lo) * ASTR + hi;
        const unsigned short* bp = B + (tn * 16 + lo) * BSTR + hi;
        v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k0 = 0; k0 < KC; k0 += 16) {
            acc = __builtin_amdgcn_mmac_f32_16x16x16bf16(
                *reinterpret_cast<const v4s*>(ap + k0),
                *reinterpret_cast<const v4s*>(bp + k0), acc);
        }
        // D scatter: lane l writes banks (2*(l%16) + l/16) mod 32, all distinct.
        const int m = tm * 16 + lo, n0 = tn * 16 + (lane >> 4);
#pragma unroll
        for (int i = 0; i < 4; ++i) ep(m, n0 + 4 * i, acc[i]);
    }
}

// Plain store into D[MM][DSTR], for the products nothing is fused onto.
template <int MM, int NN, int KC, int ASTR, int BSTR, int DSTR, int THREADS>
__device__ void gemm_nt_mmac(const unsigned short* __restrict__ A,
                             const unsigned short* __restrict__ B,
                             float* __restrict__ D) {
    gemm_nt_mmac_ep<MM, NN, KC, ASTR, BSTR, THREADS>(
        A, B, [D](int m, int n, float x) { D[m * DSTR + n] = x; });
}


// Square C x C x C product with fp32 operands, one 16x16 output tile per wave,
// result left in registers so the caller can accumulate in place.  The solve's
// matrices are fp32, so this uses the tf32 mmac (221.8 TFLOPS measured, 10-bit
// mantissa) rather than converting them to bf16.  The tf32 fragment is v2f at
// k = 2*(lane/16)+i; with STR even each 16-lane group covers 16 distinct bank
// pairs, again the ds_read_b64 minimum.  D keeps the same layout as the bf16
// form -- it is fixed by the 16x16 output tile, not by the k depth.
template <int C, int STR, int THREADS>
__device__ __forceinline__ v4f mmac_sq_tf32(const float* __restrict__ A,
                                            const float* __restrict__ B) {
    constexpr int NT = C / 16;
    static_assert(NT * NT <= THREADS / 64, "solve wants at least one wave per tile");
    static_assert(STR % 2 == 0 && C % 8 == 0, "tf32 fragment reads need 8B alignment");
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const float* ap = A + ((wave / NT) * 16 + (lane & 15)) * STR + (lane >> 4) * 2;
    const float* bp = B + ((wave % NT) * 16 + (lane & 15)) * STR + (lane >> 4) * 2;
    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k0 = 0; k0 < C; k0 += 8) {
        acc = __builtin_amdgcn_mmac_f32_16x16x8tf32(
            *reinterpret_cast<const v2f*>(ap + k0),
            *reinterpret_cast<const v2f*>(bp + k0), acc);
    }
    return acc;
}

// Address in D of the four values mmac_sq_tf32 leaves in this lane.
template <int C, int STR, int THREADS>
__device__ __forceinline__ float* mmac_sq_dst(float* D) {
    constexpr int NT = C / 16;
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    return D + ((wave / NT) * 16 + (lane & 15)) * STR + (wave % NT) * 16 + (lane >> 4);
}


// D[MM][NN] (= or +=) sum_k A[MM][ASTR] * B[NN][BSTR] with fp32 operands fed
// through the tf32 matrix core.  Used for the two C x VT x C products, whose
// operands (T, P, Y^T, U^T) are fp32 accumulators: rounding them to bf16 would
// cost more precision than tf32's 10-bit mantissa for no gain, since either
// instruction is one issue per 16x16 tile.
//
// These replace a vector gemm_nn_tril that exploited the triangular structure
// to halve the MACs.  mmac cannot skip the zero half, but it does 4096 MACs per
// instruction, so giving up the factor of two still wins by an order of
// magnitude.  The reason they could not use mmac before is the B operand: mmac
// needs B stored [n][k], and T @ Y wants Y^T.  Carrying Y, U and O transposed
// for their whole lifetime makes every operand land in its natural layout --
//     U^T[v][j] = sum_i Y^T[v][i] T[j][i]      = mmac(Y^T, T)
//     O^T[v][i] += sum_j U^T[v][j] Ptil[i][j]  = mmac(U^T, Ptil)
// -- with no transpose pass anywhere.  The V-side mmac products feeding them
// transpose for free by swapping their operands: mmac(St, k) is Z^T.
template <int MM, int NN, int KC, int ASTR, int BSTR, int THREADS, typename Ep>
__device__ void gemm_nt_tf32_ep(const float* __restrict__ A, const float* __restrict__ B,
                                Ep ep) {
    static_assert(MM % 16 == 0 && NN % 16 == 0 && KC % 8 == 0, "tf32 tile mismatch");
    static_assert(ASTR % 2 == 0 && BSTR % 2 == 0, "fragment reads need 8B alignment");
    constexpr int NTN = NN / 16, NTILES = (MM / 16) * NTN, NWAVES = THREADS / 64;
    const int lane = threadIdx.x & 63;
    const int lo = lane & 15, hi = (lane >> 4) * 2;
    for (int t = threadIdx.x >> 6; t < NTILES; t += NWAVES) {
        const int tm = t / NTN, tn = t % NTN;
        const float* ap = A + (tm * 16 + lo) * ASTR + hi;
        const float* bp = B + (tn * 16 + lo) * BSTR + hi;
        v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k0 = 0; k0 < KC; k0 += 8) {
            acc = __builtin_amdgcn_mmac_f32_16x16x8tf32(
                *reinterpret_cast<const v2f*>(ap + k0),
                *reinterpret_cast<const v2f*>(bp + k0), acc);
        }
        const int m = tm * 16 + lo, n0 = tn * 16 + (lane >> 4);
#pragma unroll
        for (int i = 0; i < 4; ++i) ep(m, n0 + 4 * i, acc[i]);
    }
}

template <int MM, int NN, int KC, int ASTR, int BSTR, int DSTR, int THREADS, bool ACCUM>
__device__ void gemm_nt_tf32(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ D) {
    gemm_nt_tf32_ep<MM, NN, KC, ASTR, BSTR, THREADS>(
        A, B, [D](int m, int n, float x) {
            float* dp = D + m * DSTR + n;
            *dp = ACCUM ? *dp + x : x;
        });
}


// Four consecutive elements of a q/k row as one load, converted to bf16.  The
// bf16 entry point gets a plain 8 B global_load_dwordx2, fp32 a 16 B dwordx4
// plus four cvt.  The scalar version this replaces issued one 2 B load per
// element, so a wave64 pulled 128 B per instruction where this pulls 512 B.
template <typename T>
__device__ __forceinline__ u16x4 load4_bf(const T* p, std::size_t i);
template <>
__device__ __forceinline__ u16x4 load4_bf<uint16_t>(const uint16_t* p, std::size_t i) {
    return *reinterpret_cast<const u16x4*>(p + i);
}
template <>
__device__ __forceinline__ u16x4 load4_bf<float>(const float* p, std::size_t i) {
    const v4f f = *reinterpret_cast<const v4f*>(p + i);
    u16x4 o;
#pragma unroll
    for (int r = 0; r < 4; ++r) o[r] = f2bf(f[r]);
    return o;
}


}  // namespace gdn2_mmac

#endif  // FLASH_TRAIN_GDN2_CHUNK_MMAC_H
