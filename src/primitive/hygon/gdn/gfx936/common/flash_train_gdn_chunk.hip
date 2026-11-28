// Chunked / WY-representation GDN forward for gfx936 (HCU BW1000).
//
// The existing kernels run the recurrence one timestep at a time, which makes
// the large shape latency-bound (1.76 us per step on a 4096-long dependency
// chain) and caps occupancy at B*HV*V*KS threads.  This file reformulates the
// same recurrence over chunks of C timesteps so the sequential depth drops to
// S/C and the work becomes dense GEMM, which is what the matrix cores want.
//
// Per chunk (local index i, St = state slice at chunk start held as [v][k],
// c_i = cumsum(g)_i within the chunk):
//   G[i][j]   = k_i . k_j            M[i][j] = beta_i e^{c_i-c_j} G[i][j], j<i
//   T         = (I + strict_tril(M))^{-1}
//   Z[j][v]   = sum_kk k[j][kk] St[v][kk]
//   Y[j][v]   = beta_j ( v[j][v] - e^{c_j} Z[j][v] )
//   U         = T @ Y
//   P[i][j]   = q_i . k_j
//   O[i][v]   = scale e^{c_i} ( sum_kk q[i][kk] St[v][kk]
//                               + sum_{j<=i} e^{-c_j} P[i][j] U[j][v] )
//   St'[v][kk] = e^{c_C-1} St[v][kk] + sum_i e^{c_C-1 - c_i} U[i][v] k[i][kk]
// Verified against the recurrent reference to 1e-15 in float64
// (bench/chunked_proto.py).
//
// Two algebraic simplifications versus the textbook WY form:
//   * khat/kcheck/qhat are row-scalings of raw k/q, so only raw k and q are
//     staged in LDS and every gate factor folds into a GEMM result row/column.
//   * W_mat is never formed: U = T(beta*V - Khat St^T) collapses the C*K*C and
//     C*V*K pair into the single C*V*K product Z = k St^T.  Half the FLOPs and
//     16 KiB less LDS than the FLA-shaped version.
//
// v4 runs the four chunk-scale GEMMs on the matrix cores.  gfx936 has no MFMA;
// its matrix unit is the mmac family, and v_mmac_f32_16x16x16_bf16 measured
// 424.5 TFLOPS register-resident (88% of the 480 spec) against 26.9 TFLOPS for
// the vector FP32 path.  The v3 vector GEMMs were LDS-issue-bound at one
// ds_read per FMA; a 16x16x16 tile is one instruction per 4096 MACs with two
// ds_read_b64 of operand, which is ~32x fewer LDS instructions.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "flash_train_gdn_chunk.h"

namespace {

constexpr int kChunk = 32;      // C; 1/A stays <= exp(0.05*32) = 4.95
constexpr int kThreads = 256;  // narrow-tile default; see BlockThreads

// Phase mask for the cost-attribution probe.  Runtime (not compile-time) so the
// optimiser cannot fold a disabled phase away at the wrong granularity; -1 is
// every phase on, which is what the production entry points always pass.
enum {
    PH_STAGE = 1,    // global q/k loads + gate prefix scan
    PH_GEMM_M = 2,   // G = k k^T and the M mask
    PH_SOLVE = 4,    // T = (I + strict_tril M)^-1
    PH_ZY = 8,       // Z = k St^T, Y = beta (v - e^c Z)
    PH_U = 16,       // U = T Y
    PH_QGEMM = 32,   // q St^T and P = q k^T
    PH_OUT = 64,     // tril(P) U accumulate + output store
    PH_STATE = 128,  // St' = decay St + U^T k
};
int g_phase_mask = -1;

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

// ---------------------------------------------------------------------------
// Forward kernel.  One block owns (b, hv, v-slice) and walks its chunks in
// order.  LDS pool at C=32, KT=128, VT=64 is 58.8 KiB:
//   sK   C*KSTR   bf16   8.3K  raw k tile, live for the whole chunk
//   sQ   C*KSTR   bf16   8.3K  raw q tile
//   sStB VT*KSTR  bf16  16.5K  bf16 mirror of the state, mmac operand only
//   sM   C*CSTR   fp32   4.3K  G -> M (kept separate from sT: the solve reads
//                              M[i][j] while writing T[i][c])
//   sT   C*CSTR   fp32   4.3K  T, then reused for P~
//   sY   VT*CSTR  fp32   8.5K  Z -> Y, then reused for the output accumulator
//   sU   VT*CSTR  fp32   8.5K  U
//
// The 17.0K pool behind sM|sT|sY has three tenants, none of which overlap in
// time: those three, sKT (the transposed key tile the state mmac needs), and
// four of the solve's six ping-pong slabs -- the other two live in sU.  All
// three wants are exactly 4352 floats at VT = 64, so nothing has to grow.
//
// The fp32 state master is NOT in LDS.  It is the one value carried across
// every chunk, and the only three things that touch it -- the initial load,
// the state mmac, and the write-out -- all address it through the state mmac's
// output-fragment map, which is a compile-time function of (wave, lane).  So
// it lives in registers, VT*KT/THREADS = 32 VGPRs per thread at VT = 64, where
// a wave has the whole 512-VGPR SIMD budget to itself anyway.  That is what
// buys VT = 64: keeping it in LDS costs 32.5 KiB and puts the block at
// 93,440 B, a third over the 65,536 B limit.
//
// mmac needs both operands in the same precision, so the state also carries a
// bf16 mirror, refreshed in place from the same registers the mmac just wrote
// (no extra barrier, no reload).  The fp32 copy stays authoritative, so only
// the two O-side products round to bf16 -- the same precision class the q/k
// staging already imposes.
// The row padding is load bearing, not cosmetic: see gemm_nt_mmac.
// ---------------------------------------------------------------------------
template <typename T, int C, int KT, int VT, int THREADS>
__global__ __launch_bounds__(THREADS) void gdn_chunk_forward_kernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ initial_state, T* __restrict__ output,
    float* __restrict__ final_state, float* __restrict__ chunk_states,
    int B, int S, int H, int HV, int K, int V, float scale, int phases) {
    constexpr int KSTR = KT + 4;  // bf16 operand rows; +4 elems = +8 B
    constexpr int CSTR = C + 2;   // fp32 rows of length C
    // Y, U and O live transposed (VT rows of C) so that every C x VT x C
    // product is an mmac in natural layout; see gemm_nt_tf32.  sU also holds
    // two of the solve's ping-pong slabs when that is enabled, but only when
    // VT >= 2C already makes it that wide, so it never costs LDS.
    constexpr bool kPingPong = VT >= 2 * C;
    constexpr int UROWS = VT > C ? VT : C;
    constexpr int kLog2Chunk = C == 16 ? 4 : (C == 32 ? 5 : (C == 64 ? 6 : 7));
    static_assert((1 << kLog2Chunk) == C, "chunk size must be a power of two");
    static_assert(VT % 16 == 0, "value tile must be an mmac multiple");
    // M, T and Y are all dead by the time the state update runs, so the
    // transposed key tile that update's mmac needs is carved out of the same
    // pool instead of costing new LDS.  Its stride is CSTR too: the transpose
    // has C columns, exactly like the matrices it overlays.  The solve's four
    // pool-side slabs are the third claim on it, and are taken only when
    // VT >= 2C already sizes the pool to (2C + VT) * CSTR >= 4C * CSTR.
    constexpr int kPoolNat = (C + C + VT) * CSTR;   // sM | sT | sY
    constexpr int kPoolKT = KT * CSTR;              // sKT
    constexpr int kPool = kPoolNat > kPoolKT ? kPoolNat : kPoolKT;
    static_assert(!kPingPong || kPoolNat >= 4 * C * CSTR, "ping-pong must be free");
    __shared__ __align__(16) unsigned short sK[C * KSTR];
    __shared__ __align__(16) unsigned short sQ[C * KSTR];
    __shared__ __align__(16) unsigned short sStB[VT * KSTR];
    __shared__ __align__(16) float sPool[kPool];
    __shared__ __align__(16) float sU[UROWS * CSTR];
    __shared__ float sGate[C];
    __shared__ float sBeta[C];
    float* const sM = sPool;
    float* const sT = sPool + C * CSTR;
    float* const sY = sPool + 2 * C * CSTR;
    float* const sKT = sPool;

    // The solve carries S, P and P^T.  When VT >= 2C the pool and sU are wide
    // enough for two slabs of each, so the iterates ping-pong: nothing a step
    // writes is something a step reads, the "everyone is done reading" barrier
    // disappears, and eleven barriers become six for no extra LDS.  Below that
    // the second set would cost real LDS, and on the narrow-v-tile shapes a
    // second resident block is worth far more than five barriers -- growing the
    // pool there cost 18-39%.  Those keep the in-place form, which falls out of
    // aliasing both slabs of each iterate onto one buffer.
    // The a/b naming is by step parity, with the starting parity chosen so the
    // last step leaves S in sv1 == sT, where the rest of the chunk expects T.
    constexpr int kPar0 = (kLog2Chunk & 1) ? 0 : 1;
    float* const sSa = kPingPong ? (kPar0 ? sT : sPool) : sT;
    float* const sSb = kPingPong ? (kPar0 ? sPool : sT) : sT;
    float* const sPa = kPingPong ? (kPar0 ? sPool + 3 * C * CSTR : sY) : sPool;
    float* const sPb = kPingPong ? (kPar0 ? sY : sPool + 3 * C * CSTR) : sPool;
    float* const sQa = kPingPong ? (kPar0 ? sU + C * CSTR : sU) : sU;
    float* const sQb = kPingPong ? (kPar0 ? sU : sU + C * CSTR) : sU;

    // Tiling of the state update's mmac, hoisted here because the register-
    // resident fp32 state is addressed through exactly this map.  Wave w owns
    // output tiles w, w+NWAVES, ...; within tile t = (tm, tn) lane l holds the
    // four values at (m = 16*tm + l%16, n = 16*tn + l/16 + 4r).  Every slot is
    // a compile-time constant, so stAcc stays in VGPRs and never spills.
    constexpr int NWAVES = THREADS / 64;
    constexpr int NTN_S = KT / 16;
    constexpr int NTILES_S = (VT / 16) * NTN_S;
    constexpr int NSLOT = (NTILES_S + NWAVES - 1) / NWAVES;
    static_assert(KT % 16 == 0 && C % 8 == 0, "state mmac tile mismatch");

    const int slices = V / VT;
    const int bhv = blockIdx.x / slices;
    const int slice = blockIdx.x % slices;
    const int b = bhv / HV;
    const int hv = bhv % HV;
    const int h = hv / (HV / H);
    const int v0 = slice * VT;
    const int tid = threadIdx.x;
    const int nchunks = S / C;

    // Lane's fixed coordinates within any state tile it owns.
    const int wave0 = tid >> 6, lane0 = tid & 63;
    const int lo0 = lane0 & 15, hcol = lane0 >> 4;
    v4f stAcc[NSLOT];
#pragma unroll
    for (int s = 0; s < NSLOT; ++s) {
        const int t = wave0 + s * NWAVES;
        v4f a = {0.0f, 0.0f, 0.0f, 0.0f};
        if (t < NTILES_S) {
            const int m = (t / NTN_S) * 16 + lo0, n0 = (t % NTN_S) * 16 + hcol;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int n = n0 + 4 * r;
                float init = 0.0f;
                if (initial_state != nullptr && n < K) {
                    init = initial_state[((static_cast<std::size_t>(b) * HV + hv) * K + n) *
                                             static_cast<std::size_t>(V) + (v0 + m)];
                }
                a[r] = init;
                sStB[m * KSTR + n] = f2bf(init);
            }
        }
        stAcc[s] = a;
    }
    __syncthreads();

    // Registers holding the NEXT chunk's q/k tiles.  At C=32, KT=128 this is
    // four u16x4 of each = 16 VGPRs; with 59.6 KiB of LDS only one block fits
    // per CU, so a wave has the whole 512-VGPR SIMD budget to itself and the
    // pressure is irrelevant.
    constexpr int SG = 4;                       // elements staged per load
    constexpr int NROW = KT / SG;               // load groups per row
    constexpr int NGRP = C * NROW / THREADS;    // groups per thread
    static_assert(NGRP * THREADS * SG == C * KT, "staging must divide evenly");
    u16x4 pq[NGRP], pk[NGRP];

    // Issue the loads for the chunk starting at tt0 without waiting on them.
#define GDN_PREFETCH_QK(tt0)                                                        \
    _Pragma("unroll") for (int r = 0; r < NGRP; ++r) {                              \
        const int grp_ = tid + r * THREADS;                                         \
        const int i_ = grp_ / NROW, kk_ = (grp_ % NROW) * SG;                       \
        u16x4 kb_ = {0, 0, 0, 0}, qb_ = kb_;                                        \
        const std::size_t qi_ =                                                     \
            ((static_cast<std::size_t>(b) * S + ((tt0) + i_)) * H + h) *             \
                static_cast<std::size_t>(K) + kk_;                                   \
        if (kk_ + SG <= K && (K % SG) == 0) {                                        \
            kb_ = load4_bf<T>(k, qi_);                                               \
            qb_ = load4_bf<T>(q, qi_);                                               \
        } else {                                                                     \
            _Pragma("unroll") for (int e = 0; e < SG; ++e) {                          \
                if (kk_ + e < K) {                                                    \
                    kb_[e] = f2bf(load_elem<T>(k, qi_ + e));                           \
                    qb_[e] = f2bf(load_elem<T>(q, qi_ + e));                           \
                }                                                                     \
            }                                                                         \
        }                                                                             \
        pk[r] = kb_;                                                                  \
        pq[r] = qb_;                                                                  \
    }

    if (phases & PH_STAGE) { GDN_PREFETCH_QK(0) }

    for (int n = 0; n < nchunks; ++n) {
        const int t0 = n * C;

        // ---- publish the staged q/k tiles and the gate prefix sums ---------
        if (phases & PH_STAGE) {
#pragma unroll
        for (int r = 0; r < NGRP; ++r) {
            const int grp = tid + r * THREADS;
            const int i = grp / NROW, kk = (grp % NROW) * SG;
            *reinterpret_cast<u16x4*>(sK + i * KSTR + kk) = pk[r];
            *reinterpret_cast<u16x4*>(sQ + i * KSTR + kk) = pq[r];
        }
        if (tid < 64) {
            // Gate load + inclusive prefix scan, entirely within wave 0.  The
            // LDS Hillis-Steele version this replaces needed two block-wide
            // barriers per step to order its cross-lane reads; a shuffle needs
            // none, and lanes >= C contribute zero because shfl_up only ever
            // pulls from lower lanes.
            float gv = 0.0f, bv = 0.0f;
            if (tid < C) {
                const std::size_t gi =
                    (static_cast<std::size_t>(b) * S + (t0 + tid)) * HV + hv;
                gv = g[gi];
                bv = beta[gi];
            }
#pragma unroll
            for (int off = 1; off < C; off <<= 1) {
                const float up = __shfl_up(gv, off, 64);
                if (tid >= off) gv += up;
            }
            if (tid < C) {
                sGate[tid] = gv;
                sBeta[tid] = bv;
            }
        }
        __syncthreads();
        // Start the next chunk's fetch here.  Nothing waits on pq/pk until the
        // top of the next iteration, so the s_waitcnt lands after ~1.4 ms of
        // compute instead of immediately: with one block per CU there is no
        // second block to hide the HBM latency, so the block hides it itself.
        if (n + 1 < nchunks) { GDN_PREFETCH_QK(t0 + C) }
        }
        const float c_last = sGate[C - 1];

        // ---- M = beta_i e^{c_i-c_j} (k_i . k_j), strictly lower ------------
        if (phases & PH_GEMM_M) {
        gemm_nt_mmac_ep<C, C, KT, KSTR, KSTR, THREADS>(
            sK, sK, [&](int i, int j, float x) {
                sM[i * CSTR + j] =
                    (j < i) ? sBeta[i] * __expf(sGate[i] - sGate[j]) * x : 0.0f;
            });
        __syncthreads();
        }

        // ---- T = (I + strict_tril(M))^{-1} by Neumann doubling -------------
        // N = strict_tril(M) is nilpotent with N^C = 0, so the series
        //     T = sum_{m<C} R^m,   R = -N
        // terminates and this is exact, not an approximation.  Five doublings
        //     S_0 = I, P_0 = R;  S_{j+1} = S_j + S_j P_j,  P_{j+1} = P_j^2
        // give S_5 = sum_{m<32} R^m = T.
        //
        // Why replace forward substitution: it is C serial steps on C of the
        // 256 threads, i.e. one wave on one of four SIMDs, where a wave64
        // instruction occupies its SIMD for 4 cycles -- ~8 cycles per FMA is
        // the floor no matter how well it unrolls, and the measured 17 says
        // the waitcnt stalls on top are not hidden either (nothing else is
        // resident: 59.6 KiB of LDS means one block per CU).  Doubling is
        // log2(C) steps of C x C x C GEMM: every wave busy, 2 mmac each.
        //
        // mmac computes A B^T, so P is carried alongside its transpose and no
        // explicit transpose pass is needed:
        //     P' = P P = mmac(P, Pt)      Pt' = (P P)^T = Pt Pt = mmac(Pt, P)
        //     S P     = mmac(S, Pt)
        // The strict upper of every iterate stays exactly zero (each entry is
        // an empty sum), so T is still exactly unit lower triangular.
        //
        // Each iterate is ping-ponged between two slabs rather than updated in
        // place.  In place, a step needs two barriers -- one to prove every
        // wave is done reading S, P and Pt before any wave overwrites them, and
        // one to publish the new values.  Writing to the other set removes the
        // first: nothing a step writes is something a step reads, so the read
        // and the write can overlap and only the publish barrier remains.  Eleven
        // barriers become six.  That costs three more C x C slabs, which is why
        // it was not worth doing until the state moved to registers -- and at
        // VT = 64 the pool was already this size for sKT, so it now costs
        // nothing at all.
        if (phases & PH_SOLVE) {
        // The solve's output is C x C = (C/16)^2 tiles, so it can only keep
        // that many waves busy; any beyond that sit out the arithmetic but must
        // still reach every barrier, which is why the guard wraps the work and
        // not the __syncthreads().
        constexpr int kSolveWaves = (C / 16) * (C / 16);
        const bool solver = (tid >> 6) < kSolveWaves;
        for (int idx = tid; idx < C * C; idx += THREADS) {
            const int i = idx / C, j = idx % C;
            const float r = -sM[i * CSTR + j];   // already strictly lower
            sPa[i * CSTR + j] = r;               // P_0 = -N  (in place when !kPingPong)
            sQa[j * CSTR + i] = r;               // Pt_0
            sSa[i * CSTR + j] = (i == j) ? 1.0f : 0.0f;  // S_0 = I
        }
        __syncthreads();
#pragma unroll
        for (int step = 0; step < kLog2Chunk; ++step) {
            const bool odd = (step & 1) != 0;
            float* const Sc = odd ? sSb : sSa;
            float* const Sn = odd ? sSa : sSb;
            float* const Pc = odd ? sPb : sPa;
            float* const Pn = odd ? sPa : sPb;
            float* const Qc = odd ? sQb : sQa;
            float* const Qn = odd ? sQa : sQb;
            v4f xs = {0.0f, 0.0f, 0.0f, 0.0f}, xp = xs, xpt = xs;
            if (solver) {
                xs = mmac_sq_tf32<C, CSTR, THREADS>(Sc, Qc);
                if (step + 1 < kLog2Chunk) {
                    xp = mmac_sq_tf32<C, CSTR, THREADS>(Pc, Qc);
                    xpt = mmac_sq_tf32<C, CSTR, THREADS>(Qc, Pc);
                }
            }
            if (!kPingPong) __syncthreads();  // in place: every operand read is done
            if (solver) {
                const float* const sc = mmac_sq_dst<C, CSTR, THREADS>(Sc);
                float* const sn = mmac_sq_dst<C, CSTR, THREADS>(Sn);
#pragma unroll
                for (int i = 0; i < 4; ++i) sn[4 * i] = sc[4 * i] + xs[i];
                if (step + 1 < kLog2Chunk) {
                    float* const pn = mmac_sq_dst<C, CSTR, THREADS>(Pn);
                    float* const qn = mmac_sq_dst<C, CSTR, THREADS>(Qn);
#pragma unroll
                    for (int i = 0; i < 4; ++i) { pn[4 * i] = xp[i]; qn[4 * i] = xpt[i]; }
                }
            }
            __syncthreads();
        }
        }

        // ---- Z^T = St k^T, then Y^T = beta_j ( v_j - e^{c_j} Z_j ) ---------
        if (phases & PH_ZY) {
        // The v read moves into the epilogue: the producing lane owns
        // Z^T[vv][i] with vv = lane%16 running over 16 consecutive values at a
        // fixed i, so a wave64 still pulls four contiguous 32 B segments of v
        // -- nearly the coalescing of the old separate loop, and it costs
        // neither the barrier nor the LDS round trip that loop needed.
        gemm_nt_mmac_ep<VT, C, KT, KSTR, KSTR, THREADS>(
            sStB, sK, [&](int vv, int i, float z) {
                const float vval = load_elem<T>(
                    v, ((static_cast<std::size_t>(b) * S + (t0 + i)) * HV + hv) *
                               static_cast<std::size_t>(V) + (v0 + vv));
                sY[vv * CSTR + i] = sBeta[i] * (vval - __expf(sGate[i]) * z);
            });
        __syncthreads();
        }

        // ---- U^T = Y^T T^T -------------------------------------------------
        if (phases & PH_U) {
        gemm_nt_tf32<VT, C, C, CSTR, CSTR, CSTR, THREADS, false>(sY, sT, sU);
        __syncthreads();
        }

        // ---- O = scale e^{c_i} ( q St^T + tril(q k^T) diag(e^{-c_j}) U ) ---
        if (phases & PH_QGEMM) {
        gemm_nt_mmac<VT, C, KT, KSTR, KSTR, CSTR, THREADS>(sStB, sQ, sY);
        gemm_nt_mmac_ep<C, C, KT, KSTR, KSTR, THREADS>(
            sQ, sK, [&](int i, int j, float x) {
                sT[i * CSTR + j] = (j <= i) ? x * __expf(-sGate[j]) : 0.0f;
            });
        __syncthreads();
        }
        if (phases & PH_OUT) {
        // The accumulate onto q St^T and the scaled store both fold in here,
        // so O never makes a second trip through LDS.
        gemm_nt_tf32_ep<VT, C, C, CSTR, CSTR, THREADS>(
            sU, sT, [&](int vv, int i, float x) {
                store_elem<T>(output,
                              ((static_cast<std::size_t>(b) * S + (t0 + i)) * HV + hv) *
                                      static_cast<std::size_t>(V) + (v0 + vv),
                              scale * __expf(sGate[i]) * (sY[vv * CSTR + i] + x));
            });
        }

        // ---- record the chunk-entry state for the backward pass ------------
        if (chunk_states != nullptr) {
            const std::size_t base =
                ((static_cast<std::size_t>(bhv) * slices + slice) *
                     static_cast<std::size_t>(nchunks) + n) *
                (static_cast<std::size_t>(VT) * KT);
#pragma unroll
            for (int s = 0; s < NSLOT; ++s) {
                const int t = wave0 + s * NWAVES;
                if (t >= NTILES_S) continue;
                const int m = (t / NTN_S) * 16 + lo0, n0 = (t % NTN_S) * 16 + hcol;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    chunk_states[base + static_cast<std::size_t>(m) * KT + (n0 + 4 * r)] =
                        stAcc[s][r];
                }
            }
        }

        // ---- St' = e^{c_last} St + sum_i e^{c_last-c_i} U[i][v] k[i][kk] ---
        __syncthreads();
        if (phases & PH_STATE) {
        // Transpose the raw key tile into the dead M|T|Y pool as fp32, scaled
        // by e^{c_last-c_i} on the way.  The scale belongs to either operand --
        // sum_i U[i][v] (Gamma/A_i) k[i][kk] is the same product grouped either
        // way -- and putting it on k folds a whole pass over U, and its
        // barrier, into a transpose the state mmac needs regardless.
        //
        // That transpose is why this product was the last one still on the
        // vector path: mmac contracts over the last index of both operands
        // while the contraction here is over i, which is k's first index.
        // 8 KiB of LDS shuffling buys a 12x cut in LDS traffic, because the
        // vector form re-read a k row for every (vv, kk) group where the mmac
        // form reads each fragment once per 16x16 output tile.  fp32 + tf32
        // rather than bf16 because the state is the one value carried across
        // every chunk, so it is where mantissa bits are worth the most; the
        // operands must match precision and U is an fp32 accumulator anyway.
        // THREADS is a multiple of C, so idx % C is the same for every iteration
        // a thread runs: the exp -- the only transcendental in the loop -- is a
        // loop invariant and is hoisted by hand rather than left to alias
        // analysis across two shared arrays.
        static_assert(THREADS % C == 0, "transpose assumes one i per thread");
        {
            const int ti = tid % C;
            const float esc = __expf(c_last - sGate[ti]);
            const unsigned short* kr = sK + ti * KSTR;
            for (int idx = tid; idx < C * KT; idx += THREADS) {
                const int kk = idx / C;
                sKT[kk * CSTR + ti] = esc * bf2f(kr[kk]);
            }
        }
        __syncthreads();
        // St'[v][kk] = e^{c_last} St[v][kk] + sum_i U^T[v][i] k^T[kk][i].
        // St lives in stAcc, in the layout this very mmac produces, so the
        // accumulate is a register fmaf and the bf16 mirror is refreshed from
        // the same registers: no LDS round trip and no reload for the one
        // value that crosses every chunk boundary.
        const float decay = __expf(c_last);
        const int hi = hcol * 2;
#pragma unroll
        for (int s = 0; s < NSLOT; ++s) {
            const int t = wave0 + s * NWAVES;
            if (t >= NTILES_S) continue;
            const int tm = t / NTN_S, tn = t % NTN_S;
            const float* ap = sU + (tm * 16 + lo0) * CSTR + hi;
            const float* bp = sKT + (tn * 16 + lo0) * CSTR + hi;
            v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int c0 = 0; c0 < C; c0 += 8) {
                acc = __builtin_amdgcn_mmac_f32_16x16x8tf32(
                    *reinterpret_cast<const v2f*>(ap + c0),
                    *reinterpret_cast<const v2f*>(bp + c0), acc);
            }
            unsigned short* bpm = sStB + (tm * 16 + lo0) * KSTR + tn * 16 + hcol;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float nv = fmaf(decay, stAcc[s][r], acc[r]);
                stAcc[s][r] = nv;
                bpm[4 * r] = f2bf(nv);
            }
        }
        __syncthreads();
        }
    }

    if (final_state != nullptr) {
#pragma unroll
        for (int s = 0; s < NSLOT; ++s) {
            const int t = wave0 + s * NWAVES;
            if (t >= NTILES_S) continue;
            const int m = (t / NTN_S) * 16 + lo0, n0 = (t % NTN_S) * 16 + hcol;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int col = n0 + 4 * r;
                if (col < K) {
                    final_state[((static_cast<std::size_t>(b) * HV + hv) * K + col) *
                                    static_cast<std::size_t>(V) + (v0 + m)] = stAcc[s][r];
                }
            }
        }
    }
}
#undef GDN_PREFETCH_QK

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

int cu_count() {
    static const int n = [] {
        int dev = 0, cus = 80;
        if (hipGetDevice(&dev) == hipSuccess) {
            hipDeviceProp_t p;
            if (hipGetDeviceProperties(&p, dev) == hipSuccess) cus = p.multiProcessorCount;
        }
        return cus;
    }();
    return n;
}

// Splitting V across blocks buys parallelism and costs duplicated work: every
// slice re-stages q and k and recomputes G, the solve and P, which do not
// depend on V at all.  The phase profile at (1,1024,32,32,128,128) split one
// chunk into 0.413 ms of that VT-independent work against 0.363 ms that scales
// with VT, i.e. 0.53 : 0.47 per VT = 32 unit.  So the cost of a choice is
//     ceil(blocks / CUs) * (0.53 + 0.47 * VT/32)
// -- rounds of the grid times the work per round.  A wider tile only pays when
// it removes a whole round.  At baseline VT = 64 halves 128 blocks to 64 and
// turns two rounds (the second only 60% full) into one: 2.00 -> 1.47.  At
// medium2 the block count is already 64, so widening would just make the single
// round longer and VT = 32 stays.  Ties go to the wider tile, which duplicates
// less.
int value_tile(int V, int B, int HV) {
    if (V <= 0 || B <= 0 || HV <= 0) return 0;
    const int cus = cu_count();
    const int kCandidates[3] = {64, 32, 16};
    int best = 0;
    double best_cost = 0.0;
    for (int c = 0; c < 3; ++c) {
        const int VT = kCandidates[c];
        if (V % VT != 0) continue;
        const long long blocks = static_cast<long long>(B) * HV * (V / VT);
        const double rounds = static_cast<double>((blocks + cus - 1) / cus);
        const double cost = rounds * (0.53 + 0.47 * (VT / 32.0));
        if (best == 0 || cost < best_cost - 1e-9) {
            best = VT;
            best_cost = cost;
        }
    }
    return best;
}

// K is padded up to the instantiated tile, and chunk_states is written in the
// padded layout, so the size query has to agree with the kernel.
int key_tile(int K) { return K <= 32 ? 32 : (K <= 64 ? 64 : 128); }

int selected_value_tile(FlashTrainGDNChunkShape s) {
    return value_tile(s.V, s.B, s.HV);
}

bool shape_fields_ok(FlashTrainGDNChunkShape s) {
    if (s.B <= 0 || s.S <= 0 || s.H <= 0 || s.HV <= 0 || s.K <= 0 || s.V <= 0) return false;
    if (s.HV % s.H != 0) return false;
    if (s.K > 128) return false;
    if (s.S % kChunk != 0) return false;
    if (s.chunk_size != 0 && s.chunk_size != kChunk) return false;
    return true;
}

// Waves per block.  LDS pins the kernel to one block per CU, so the only way to
// get a second wave onto each SIMD -- and the solve is latency-bound with one,
// spending ~4800 cycles on ~800 cycles of issue -- is to make the block itself
// twice as wide.  Nothing in the LDS budget depends on THREADS, so this is free
// occupancy.  The solve and the two C x C products cannot use more than
// (C/16)^2 = 4 waves and simply idle the rest; every other phase has at least
// eight tiles and scales.  Only taken where the staging loop still divides
// evenly (C * KT / SG must be a multiple of THREADS), which is KT >= 64.
constexpr int block_threads_for(int KT, int VT) {
    return (VT >= 64 && KT >= 64) ? 512 : 256;
}

template <int KT, int VT>
struct BlockThreads {
    static constexpr int value = block_threads_for(KT, VT);
};

struct ForwardConfig {
    int route;
    int chunk_size;
    int key_tile;
    int value_tile;
    int threads;
    int grid_blocks;
};

bool forward_config_fused(FlashTrainGDNChunkShape s, ForwardConfig* config) {
    if (config == nullptr || !shape_fields_ok(s)) return false;
    const int VT = selected_value_tile(s);
    if (VT == 0) return false;
    const int KT = key_tile(s.K);
    config->route = FLASH_TRAIN_GDN_CHUNK_ROUTE_FUSED;
    config->chunk_size = kChunk;
    config->key_tile = KT;
    config->value_tile = VT;
    config->threads = block_threads_for(KT, VT);
    config->grid_blocks = s.B * s.HV * (s.V / VT);
    return true;
}

bool forward_config(FlashTrainGDNChunkShape s, ForwardConfig* config) {
    return forward_config_fused(s, config);
}

bool shape_ok(FlashTrainGDNChunkShape s) {
    ForwardConfig config{};
    return forward_config(s, &config);
}

template <typename T, int KT, int VT>
int launch_one(const T* q, const T* k, const T* v, const float* g, const float* beta,
               const float* initial_state, T* output, float* final_state,
               float* chunk_states, FlashTrainGDNChunkShape s, float scale,
               hipStream_t stream) {
    constexpr int NTHR = BlockThreads<KT, VT>::value;
    const int slices = s.V / VT;
    hipLaunchKernelGGL((gdn_chunk_forward_kernel<T, kChunk, KT, VT, NTHR>),
                       dim3(s.B * s.HV * slices), dim3(NTHR), 0, stream, q, k, v, g,
                       beta, initial_state, output, final_state, chunk_states, s.B, s.S,
                       s.H, s.HV, s.K, s.V, scale, g_phase_mask);
    return hipGetLastError() == hipSuccess ? FLASH_TRAIN_GDN_CHUNK_SUCCESS
                                           : FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
}

template <typename T>
int dispatch(const T* q, const T* k, const T* v, const float* g, const float* beta,
             const float* initial_state, T* output, float* final_state,
             float* chunk_states, FlashTrainGDNChunkShape s, float scale, void* stream) {
    if (q == nullptr || k == nullptr || v == nullptr || g == nullptr || beta == nullptr ||
        output == nullptr) {
        return FLASH_TRAIN_GDN_CHUNK_INVALID_POINTER;
    }
    ForwardConfig config{};
    if (!forward_config(s, &config)) return FLASH_TRAIN_GDN_CHUNK_UNSUPPORTED_CONTRACT;
    auto st = static_cast<hipStream_t>(stream);
#define FT_GDN_CHUNK_CASE(KT_, VT_)                                                     \
    return launch_one<T, KT_, VT_>(q, k, v, g, beta, initial_state, output, final_state, \
                                   chunk_states, s, scale, st)
    if (config.value_tile == 64) {
        if (config.key_tile == 32) FT_GDN_CHUNK_CASE(32, 64);
        if (config.key_tile == 64) FT_GDN_CHUNK_CASE(64, 64);
        FT_GDN_CHUNK_CASE(128, 64);
    }
    if (config.value_tile == 32) {
        if (config.key_tile == 32) FT_GDN_CHUNK_CASE(32, 32);
        if (config.key_tile == 64) FT_GDN_CHUNK_CASE(64, 32);
        FT_GDN_CHUNK_CASE(128, 32);
    }
    if (config.key_tile == 32) FT_GDN_CHUNK_CASE(32, 16);
    if (config.key_tile == 64) FT_GDN_CHUNK_CASE(64, 16);
    FT_GDN_CHUNK_CASE(128, 16);
#undef FT_GDN_CHUNK_CASE
}

}  // namespace

extern "C" int flash_train_gdn_chunk_supported(FlashTrainGDNChunkShape shape) {
    return shape_ok(shape) ? 1 : 0;
}

extern "C" int flash_train_gdn_chunk_forward_config(
    FlashTrainGDNChunkShape shape, FlashTrainGDNChunkForwardConfig* config) {
    if (config == nullptr) return FLASH_TRAIN_GDN_CHUNK_INVALID_POINTER;
    ForwardConfig selected{};
    if (!forward_config(shape, &selected)) {
        *config = FlashTrainGDNChunkForwardConfig{
            FLASH_TRAIN_GDN_CHUNK_ROUTE_NONE, 0, 0, 0, 0, 0};
        return FLASH_TRAIN_GDN_CHUNK_UNSUPPORTED_CONTRACT;
    }
    *config = FlashTrainGDNChunkForwardConfig{
        selected.route, selected.chunk_size, selected.key_tile,
        selected.value_tile, selected.threads, selected.grid_blocks};
    return FLASH_TRAIN_GDN_CHUNK_SUCCESS;
}

extern "C" const char* flash_train_gdn_chunk_forward_route_string(int route) {
    switch (route) {
        case FLASH_TRAIN_GDN_CHUNK_ROUTE_FUSED: return "fused_chunk";
        default: return "unsupported";
    }
}

extern "C" int flash_train_gdn_chunk_value_tile(FlashTrainGDNChunkShape shape) {
    FlashTrainGDNChunkForwardConfig config{};
    return flash_train_gdn_chunk_forward_config(shape, &config) == FLASH_TRAIN_GDN_CHUNK_SUCCESS
               ? config.value_tile
               : 0;
}

// Diagnostic only: restrict the kernel to a subset of its phases so the cost of
// each can be attributed.  -1 (the default) runs everything.
extern "C" void flash_train_gdn_chunk_set_phase_mask(int mask) { g_phase_mask = mask; }

extern "C" size_t flash_train_gdn_chunk_states_bytes(FlashTrainGDNChunkShape shape) {
    if (!shape_ok(shape)) return 0;
    return static_cast<size_t>(shape.B) * shape.HV * (shape.S / kChunk) * shape.V *
           key_tile(shape.K) * sizeof(float);
}

extern "C" int flash_train_gdn_chunk_fp32_forward_launch(
    const float* q, const float* k, const float* v, const float* g, const float* beta,
    const float* initial_state, float* output, float* final_state, float* chunk_states,
    FlashTrainGDNChunkShape shape, float scale, void* stream) {
    return dispatch<float>(q, k, v, g, beta, initial_state, output, final_state,
                           chunk_states, shape, scale, stream);
}

extern "C" int flash_train_gdn_chunk_bf16_forward_launch(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* g,
    const float* beta, const float* initial_state, uint16_t* output, float* final_state,
    float* chunk_states, FlashTrainGDNChunkShape shape, float scale, void* stream) {
    return dispatch<uint16_t>(q, k, v, g, beta, initial_state, output, final_state,
                              chunk_states, shape, scale, stream);
}
