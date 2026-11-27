// Option A: the chunked GDN forward split into three kernels so the chunk
// recurrence stops serialising everything.
//
// The fused kernel (flash_train_gdn_chunk.hip) runs the whole chunk body
// inside one serial loop, so its grid is B*HV*(V/VT) -- 64 blocks at llm128k,
// on 80 CUs -- and every one of the ~27 barriers per chunk sits on the
// critical path S/C times.  Measured: 13.2 TFLOPS, 3.1% of the 424 TFLOPS bf16
// peak.  That is a latency/barrier problem, not a FLOP problem.
//
// FLA's reordering makes most of the chunk body state-independent.  With
//     Yv^T[v][j] = beta_j v[j][v]          (no state)
//     Khat[j][kk] = beta_j e^{c_j} k[j][kk]
//     W = T Khat                            (no state)
//     Uv^T = Yv^T T^T                       (no state)
// the U of the fused kernel is exactly
//     U^T = Y^T T^T = (Yv^T - St Khat^T) T^T = Uv^T - St W^T
// because (T Khat)^T = Khat^T T^T.  So:
//
//   P1  grid B*HV*(S/C)          gates, M, the Neumann solve, Uv and W.
//                                Fully parallel over chunks.
//   S1  grid B*HV*(V/VT)         u = Uv - St W^T  and  St' = e^c St + Ktilde^T u.
//                                Still serial, but 2 GEMMs and ~5 barriers per
//                                chunk instead of ~27, and no solve.
//   P2  grid B*HV*(V/VT)*G       O = q St^T + tril(q k^T) u, replaying the state
//                                inside its own segment from the boundary state
//                                S1 stored.  Parallel over G segments.
//
// Cost of the reordering, stated up front (this is not free):
//   * +58% total MACs -- W is formed explicitly, which is exactly the product
//     the fused kernel's header comment says it deliberately collapsed away.
//     That collapse is only valid while everything is in one kernel.
//   * Uv/W must be materialised to HBM: (V+K) floats per token per head, read
//     back once.  12.8 GB of extra traffic at llm128k = ~9.5 ms at 1354 GB/s.
//     `u` is written back over `Uv` in place (same index, read-then-write), which
//     is what keeps this from being ~17 GB.  Storing the three in bf16 would
//     halve it again; not done here because u drives the state recurrence.
//   * All three kernels still need >32 KiB of LDS, so all three still run one
//     block per CU.  A does NOT fix occupancy.  Its gain is that the number of
//     barriers on the critical path drops from (S/C)*27 to roughly
//     (blocks/80)*barriers_per_block summed over the three kernels.
//
// Not produced by this path: chunk_states for the backward pass.  This is a
// forward-performance experiment; the fused kernel remains the backward's
// producer.

#include "gfx938_device_symbols.h"
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "gdn_chunk_mmac.h"

namespace {

using namespace gdn_mmac;

constexpr int kChunk = 32;

template <int KT, int VT>
struct SplitThreads {
    static constexpr int value = (VT >= 64 && KT >= 64) ? 512 : 256;
};

// ---------------------------------------------------------------------------
// Shared: gate prefix scan for one chunk, wave 0 only (same shuffle form as the
// fused kernel -- an LDS Hillis-Steele needs two block-wide barriers per step,
// a shuffle needs none).
// ---------------------------------------------------------------------------
template <int C>
__device__ __forceinline__ void gate_scan(const float* __restrict__ g,
                                          const float* __restrict__ beta,
                                          float* sGate, float* sBeta, int tid,
                                          std::size_t base, int HV) {
    if (tid < 64) {
        float gv = 0.0f, bv = 0.0f;
        if (tid < C) {
            const std::size_t gi = base + static_cast<std::size_t>(tid) * HV;
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
}

// ---------------------------------------------------------------------------
// P1: state-independent half of the chunk body.
//
// grid  = B * HV * (S/C), one block per chunk.
// VT1   = V (the whole value dimension): splitting v here would recompute the
//         C x C solve once per slice, and the solve is 492k of the 885k MACs.
//
// LDS at C=32, KT=128, V=128 -- 60,160 B, one block per CU:
//   sK    C*KSTR   bf16   8.3K
//   pool  (V+2C)*CSTR fp32 25.5K   laid out [sYv][sM][sT] so that sKhatT, which
//                                  is built only after Uv has been written out
//                                  and the solve is done, can alias [sYv|sM]
//                                  (KT*CSTR <= (V+C)*CSTR).
//   sU    2C*CSTR   fp32   8.5K    solve ping-pong only
//   sW    C*KSTRF   fp32  16.3K    W staged before the store: the mmac output
//                                  fragment gives a quarter-wave one kk across
//                                  16 rows, and w rows are HV*K apart, so a
//                                  direct store would be 4096 separate lines.
// ---------------------------------------------------------------------------
template <typename T, int C, int KT, int VT, int THREADS>
__global__ __launch_bounds__(THREADS) void gdn_split_p1_kernel(
    const T* __restrict__ k, const T* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, float* __restrict__ uv_out,
    float* __restrict__ w_out, int B, int S, int H, int HV, int K, int V) {
    constexpr int KSTR = KT + 4;
    constexpr int CSTR = C + 2;
    constexpr int KSTRF = KT + 2;
    constexpr int kLog2Chunk = C == 16 ? 4 : (C == 32 ? 5 : (C == 64 ? 6 : 7));
    static_assert((1 << kLog2Chunk) == C, "chunk size must be a power of two");
    static_assert(VT % 16 == 0, "value tile must be an mmac multiple");

    // sKhatT aliases the [sYv|sM] prefix, so that prefix has to be at least KT
    // rows wide.  At V=64,KT=128 it is not (VT+C = 96 < 128), so pad it out --
    // 4,352 B of slack, which is cheaper than a second KT*CSTR array.
    constexpr int kFront = (VT + C) > KT ? (VT + C) : KT;
    constexpr int kPool = (kFront + C) * CSTR;
    __shared__ __align__(16) unsigned short sK[C * KSTR];
    __shared__ __align__(16) float sPool[kPool];
    __shared__ __align__(16) float sU[2 * C * CSTR];
    __shared__ __align__(16) float sW[C * KSTRF];
    __shared__ float sGate[C];
    __shared__ float sBeta[C];

    float* const sYv = sPool;
    float* const sM = sPool + VT * CSTR;
    float* const sT = sPool + kFront * CSTR;
    float* const sKhatT = sPool;   // aliases [sYv | sM], both dead by then

    // Ping-pong the solve iterates when there is room for three C x C slabs in
    // the sYv region (dead during the solve: Yv is loaded afterwards).  Below
    // that, fall back to the in-place form, which needs one extra barrier per
    // doubling step.
    constexpr bool kPingPong = VT >= 3 * C;
    constexpr int kPar0 = (kLog2Chunk & 1) ? 0 : 1;
    float* const sSa = kPingPong ? (kPar0 ? sT : sYv) : sT;
    float* const sSb = kPingPong ? (kPar0 ? sYv : sT) : sT;
    float* const sPa = kPingPong ? (kPar0 ? sYv + C * CSTR : sYv + 2 * C * CSTR) : sM;
    float* const sPb = kPingPong ? (kPar0 ? sYv + 2 * C * CSTR : sYv + C * CSTR) : sM;
    float* const sQa = kPingPong ? (kPar0 ? sU + C * CSTR : sU) : sU;
    float* const sQb = kPingPong ? (kPar0 ? sU : sU + C * CSTR) : sU;

    const int nchunks = S / C;
    const int n = blockIdx.x % nchunks;
    const int bhv = blockIdx.x / nchunks;
    const int b = bhv / HV;
    const int hv = bhv % HV;
    const int h = hv / (HV / H);
    const int tid = threadIdx.x;
    const int t0 = n * C;

    // ---- stage the raw key tile ----------------------------------------
    constexpr int SG = 4;
    constexpr int NROW = KT / SG;
    constexpr int NGRP = C * NROW / THREADS;
    static_assert(NGRP * THREADS * SG == C * KT, "staging must divide evenly");
#pragma unroll
    for (int r = 0; r < NGRP; ++r) {
        const int grp = tid + r * THREADS;
        const int i = grp / NROW, kk = (grp % NROW) * SG;
        u16x4 kb = {0, 0, 0, 0};
        const std::size_t ki =
            ((static_cast<std::size_t>(b) * S + (t0 + i)) * H + h) *
                static_cast<std::size_t>(K) + kk;
        if (kk + SG <= K && (K % SG) == 0) {
            kb = load4_bf<T>(k, ki);
        } else {
#pragma unroll
            for (int e = 0; e < SG; ++e) {
                if (kk + e < K) kb[e] = f2bf(load_elem<T>(k, ki + e));
            }
        }
        *reinterpret_cast<u16x4*>(sK + i * KSTR + kk) = kb;
    }
    gate_scan<C>(g, beta, sGate, sBeta,
                 tid, (static_cast<std::size_t>(b) * S + t0) * HV + hv, HV);
    __syncthreads();

    // ---- M = beta_i e^{c_i-c_j} (k_i . k_j), strictly lower -------------
    gemm_nt_mmac_ep<C, C, KT, KSTR, KSTR, THREADS>(
        sK, sK, [&](int i, int j, float x) {
            sM[i * CSTR + j] =
                (j < i) ? sBeta[i] * __expf(sGate[i] - sGate[j]) * x : 0.0f;
        });
    __syncthreads();

    // ---- T = (I + strict_tril(M))^{-1} by Neumann doubling ---------------
    {
        constexpr int kSolveWaves = (C / 16) * (C / 16);
        const bool solver = (tid >> 6) < kSolveWaves;
        for (int idx = tid; idx < C * C; idx += THREADS) {
            const int i = idx / C, j = idx % C;
            const float r = -sM[i * CSTR + j];
            sPa[i * CSTR + j] = r;
            sQa[j * CSTR + i] = r;
            sSa[i * CSTR + j] = (i == j) ? 1.0f : 0.0f;
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
            if (!kPingPong) __syncthreads();
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
    // T now lives in sT for both parities (kPar0 is chosen for that).

    // ---- Yv^T[v][j] = beta_j v[j][v] ------------------------------------
    for (int idx = tid; idx < VT * C; idx += THREADS) {
        const int j = idx / VT, vv = idx % VT;   // vv contiguous -> coalesced v read
        const float vval = load_elem<T>(
            v, ((static_cast<std::size_t>(b) * S + (t0 + j)) * HV + hv) *
                   static_cast<std::size_t>(V) + vv);
        sYv[vv * CSTR + j] = sBeta[j] * vval;
    }
    __syncthreads();

    // ---- Uv^T = Yv^T T^T, straight out to HBM ----------------------------
    gemm_nt_tf32_ep<VT, C, C, CSTR, CSTR, THREADS>(
        sYv, sT, [&](int vv, int i, float x) {
            uv_out[((static_cast<std::size_t>(b) * S + (t0 + i)) * HV + hv) *
                       static_cast<std::size_t>(V) + vv] = x;
        });
    __syncthreads();   // Uv is out; sYv (and sM) may now be overwritten

    // ---- KhatT[kk][j] = beta_j e^{c_j} k[j][kk] --------------------------
    for (int idx = tid; idx < C * KT; idx += THREADS) {
        const int j = idx / KT, kk = idx % KT;
        sKhatT[kk * CSTR + j] = sBeta[j] * __expf(sGate[j]) * bf2f(sK[j * KSTR + kk]);
    }
    __syncthreads();

    // ---- W = T Khat ------------------------------------------------------
    gemm_nt_tf32_ep<C, KT, C, CSTR, CSTR, THREADS>(
        sT, sKhatT, [&](int i, int kk, float x) { sW[i * KSTRF + kk] = x; });
    __syncthreads();
    for (int idx = tid; idx < C * KT; idx += THREADS) {
        const int i = idx / KT, kk = idx % KT;   // kk contiguous -> coalesced store
        if (kk < K) {
            w_out[((static_cast<std::size_t>(b) * S + (t0 + i)) * HV + hv) *
                      static_cast<std::size_t>(K) + kk] = sW[i * KSTRF + kk];
        }
    }
}

// ---------------------------------------------------------------------------
// S1: the part that is irreducibly serial.
//
// grid = B * HV * (V/VT).  Per chunk: two GEMMs (St W^T, and the state update),
// no solve, no transcendental except the gate scan and the decay.
//
// LDS at C=32, KT=128, VT=64 -- 59,904 B:
//   sK   C*KSTR  bf16   8.3K
//   sWb  C*KSTR  bf16   8.3K
//   sStB VT*KSTR bf16  16.5K   bf16 mirror of the register-resident fp32 state
//   sU   VT*CSTR fp32   8.5K   u^T
//   sKT  KT*CSTR fp32  17.0K   transposed, gate-scaled key tile
// ---------------------------------------------------------------------------
template <typename T, int C, int KT, int VT, int THREADS>
__global__ __launch_bounds__(THREADS) void gdn_split_s1_kernel(
    const T* __restrict__ k, const float* __restrict__ g,
    const float* __restrict__ beta, const float* __restrict__ initial_state,
    const float* __restrict__ w_in, float* __restrict__ u_io,
    float* __restrict__ seg_states, float* __restrict__ final_state,
    int B, int S, int H, int HV, int K, int V, int nseg, int chunks_per_seg) {
    constexpr int KSTR = KT + 4;
    constexpr int CSTR = C + 2;
    __shared__ __align__(16) unsigned short sK[C * KSTR];
    __shared__ __align__(16) unsigned short sWb[C * KSTR];
    __shared__ __align__(16) unsigned short sStB[VT * KSTR];
    __shared__ __align__(16) float sU[VT * CSTR];
    __shared__ __align__(16) float sKT[KT * CSTR];
    __shared__ float sGate[C];
    __shared__ float sBeta[C];

    constexpr int NWAVES = THREADS / 64;
    constexpr int NTN_S = KT / 16;
    constexpr int NTILES_S = (VT / 16) * NTN_S;
    constexpr int NSLOT = (NTILES_S + NWAVES - 1) / NWAVES;
    static_assert(KT % 16 == 0 && C % 8 == 0, "state mmac tile mismatch");
    static_assert(THREADS % C == 0, "transpose assumes one i per thread");

    const int slices = V / VT;
    const int bhv = blockIdx.x / slices;
    const int slice = blockIdx.x % slices;
    const int b = bhv / HV;
    const int hv = bhv % HV;
    const int h = hv / (HV / H);
    const int v0 = slice * VT;
    const int tid = threadIdx.x;
    const int nchunks = S / C;

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
                const int nn = n0 + 4 * r;
                float init = 0.0f;
                if (initial_state != nullptr && nn < K) {
                    init = initial_state[((static_cast<std::size_t>(b) * HV + hv) * K + nn) *
                                             static_cast<std::size_t>(V) + (v0 + m)];
                }
                a[r] = init;
                sStB[m * KSTR + nn] = f2bf(init);
            }
        }
        stAcc[s] = a;
    }
    __syncthreads();

    constexpr int SG = 4;
    constexpr int NROW = KT / SG;
    constexpr int NGRP = C * NROW / THREADS;
    static_assert(NGRP * THREADS * SG == C * KT, "staging must divide evenly");

    for (int n = 0; n < nchunks; ++n) {
        const int t0 = n * C;

        // ---- checkpoint the chunk-entry state at segment boundaries ------
        if ((n % chunks_per_seg) == 0) {
            const int seg = n / chunks_per_seg;
            if (seg < nseg) {
                const std::size_t base =
                    ((static_cast<std::size_t>(bhv) * nseg + seg) *
                         static_cast<std::size_t>(V)) * K;
#pragma unroll
                for (int s = 0; s < NSLOT; ++s) {
                    const int t = wave0 + s * NWAVES;
                    if (t >= NTILES_S) continue;
                    const int m = (t / NTN_S) * 16 + lo0, n0 = (t % NTN_S) * 16 + hcol;
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        const int col = n0 + 4 * r;
                        if (col < K) {
                            seg_states[base +
                                       static_cast<std::size_t>(v0 + m) * K + col] =
                                stAcc[s][r];
                        }
                    }
                }
            }
        }

        // ---- stage k and W -----------------------------------------------
#pragma unroll
        for (int r = 0; r < NGRP; ++r) {
            const int grp = tid + r * THREADS;
            const int i = grp / NROW, kk = (grp % NROW) * SG;
            u16x4 kb = {0, 0, 0, 0};
            const std::size_t ki =
                ((static_cast<std::size_t>(b) * S + (t0 + i)) * H + h) *
                    static_cast<std::size_t>(K) + kk;
            if (kk + SG <= K && (K % SG) == 0) {
                kb = load4_bf<T>(k, ki);
            } else {
#pragma unroll
                for (int e = 0; e < SG; ++e) {
                    if (kk + e < K) kb[e] = f2bf(load_elem<T>(k, ki + e));
                }
            }
            *reinterpret_cast<u16x4*>(sK + i * KSTR + kk) = kb;
        }
        for (int idx = tid; idx < C * KT; idx += THREADS) {
            const int i = idx / KT, kk = idx % KT;   // kk contiguous
            float wv = 0.0f;
            if (kk < K) {
                wv = w_in[((static_cast<std::size_t>(b) * S + (t0 + i)) * HV + hv) *
                              static_cast<std::size_t>(K) + kk];
            }
            sWb[i * KSTR + kk] = f2bf(wv);
        }
        gate_scan<C>(g, beta, sGate, sBeta,
                     tid, (static_cast<std::size_t>(b) * S + t0) * HV + hv, HV);
        __syncthreads();
        const float c_last = sGate[C - 1];

        // ---- u^T = Uv^T - St W^T, in place over Uv in HBM -----------------
        // The Uv read rides in the epilogue: the producing lane owns
        // (vv = lane%16) across 16 consecutive values at a fixed i, so a wave64
        // still pulls four contiguous 64 B segments -- the same trick the fused
        // kernel uses for its v read.
        gemm_nt_mmac_ep<VT, C, KT, KSTR, KSTR, THREADS>(
            sStB, sWb, [&](int vv, int i, float x) {
                const std::size_t oi =
                    ((static_cast<std::size_t>(b) * S + (t0 + i)) * HV + hv) *
                        static_cast<std::size_t>(V) + (v0 + vv);
                const float uval = u_io[oi] - x;
                u_io[oi] = uval;
                sU[vv * CSTR + i] = uval;
            });
        __syncthreads();

        // ---- St' = e^{c_last} St + sum_i e^{c_last-c_i} u^T[v][i] k[i][kk] --
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

// ---------------------------------------------------------------------------
// P2: the output, parallel over G segments.
//
// grid = B * HV * (V/VT) * G.  Each block reloads the chunk-entry state S1
// checkpointed for its segment and replays the state accumulation inside the
// segment -- u is already known, so the replay is one GEMM per chunk, not the
// whole chunk body.  Choosing G so that B*HV*(V/VT)*G is a multiple of 80 is
// what turns the fused kernel's permanent 0.80 occupancy into 1.00.
//
// LDS at C=32, KT=128, VT=64 -- 55,552 B:
//   sK   C*KSTR  bf16   8.3K
//   sStB VT*KSTR bf16  16.5K
//   sU   VT*CSTR fp32   8.5K   u^T
//   pool 21,504 B              [sQ bf16][sY fp32][sT fp32], all three dead by
//                              the state advance, so sKT (KT*CSTR fp32, 17.0K)
//                              aliases the whole pool.
// ---------------------------------------------------------------------------
template <typename T, int C, int KT, int VT, int THREADS>
__global__ __launch_bounds__(THREADS) void gdn_split_p2_kernel(
    const T* __restrict__ q, const T* __restrict__ k, const float* __restrict__ g,
    const float* __restrict__ beta, const float* __restrict__ seg_states,
    const float* __restrict__ u_in, T* __restrict__ output,
    int B, int S, int H, int HV, int K, int V, float scale,
    int nseg, int chunks_per_seg) {
    constexpr int KSTR = KT + 4;
    constexpr int CSTR = C + 2;
    constexpr int kQFloats = (C * KSTR * 2 + 3) / 4;
    constexpr int kPoolNat = kQFloats + VT * CSTR + C * CSTR;
    constexpr int kPool = kPoolNat > KT * CSTR ? kPoolNat : KT * CSTR;
    __shared__ __align__(16) unsigned short sK[C * KSTR];
    __shared__ __align__(16) unsigned short sStB[VT * KSTR];
    __shared__ __align__(16) float sU[VT * CSTR];
    __shared__ __align__(16) float sPool[kPool];
    __shared__ float sGate[C];
    __shared__ float sBeta[C];

    unsigned short* const sQ = reinterpret_cast<unsigned short*>(sPool);
    float* const sY = sPool + kQFloats;
    float* const sT = sPool + kQFloats + VT * CSTR;
    float* const sKT = sPool;

    constexpr int NWAVES = THREADS / 64;
    constexpr int NTN_S = KT / 16;
    constexpr int NTILES_S = (VT / 16) * NTN_S;
    constexpr int NSLOT = (NTILES_S + NWAVES - 1) / NWAVES;
    static_assert(THREADS % C == 0, "transpose assumes one i per thread");

    const int slices = V / VT;
    const int seg = blockIdx.x % nseg;
    const int rest = blockIdx.x / nseg;
    const int bhv = rest / slices;
    const int slice = rest % slices;
    const int b = bhv / HV;
    const int hv = bhv % HV;
    const int h = hv / (HV / H);
    const int v0 = slice * VT;
    const int tid = threadIdx.x;
    const int nchunks = S / C;
    const int nbeg = seg * chunks_per_seg;
    int nend = nbeg + chunks_per_seg;
    if (nend > nchunks) nend = nchunks;

    const int wave0 = tid >> 6, lane0 = tid & 63;
    const int lo0 = lane0 & 15, hcol = lane0 >> 4;
    const std::size_t sbase =
        ((static_cast<std::size_t>(bhv) * nseg + seg) * static_cast<std::size_t>(V)) * K;
    v4f stAcc[NSLOT];
#pragma unroll
    for (int s = 0; s < NSLOT; ++s) {
        const int t = wave0 + s * NWAVES;
        v4f a = {0.0f, 0.0f, 0.0f, 0.0f};
        if (t < NTILES_S) {
            const int m = (t / NTN_S) * 16 + lo0, n0 = (t % NTN_S) * 16 + hcol;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int nn = n0 + 4 * r;
                float init = 0.0f;
                if (nn < K) {
                    init = seg_states[sbase +
                                      static_cast<std::size_t>(v0 + m) * K + nn];
                }
                a[r] = init;
                sStB[m * KSTR + nn] = f2bf(init);
            }
        }
        stAcc[s] = a;
    }
    __syncthreads();

    constexpr int SG = 4;
    constexpr int NROW = KT / SG;
    constexpr int NGRP = C * NROW / THREADS;
    static_assert(NGRP * THREADS * SG == C * KT, "staging must divide evenly");

    for (int n = nbeg; n < nend; ++n) {
        const int t0 = n * C;

        // ---- stage q, k, u ------------------------------------------------
#pragma unroll
        for (int r = 0; r < NGRP; ++r) {
            const int grp = tid + r * THREADS;
            const int i = grp / NROW, kk = (grp % NROW) * SG;
            u16x4 kb = {0, 0, 0, 0}, qb = kb;
            const std::size_t qi =
                ((static_cast<std::size_t>(b) * S + (t0 + i)) * H + h) *
                    static_cast<std::size_t>(K) + kk;
            if (kk + SG <= K && (K % SG) == 0) {
                kb = load4_bf<T>(k, qi);
                qb = load4_bf<T>(q, qi);
            } else {
#pragma unroll
                for (int e = 0; e < SG; ++e) {
                    if (kk + e < K) {
                        kb[e] = f2bf(load_elem<T>(k, qi + e));
                        qb[e] = f2bf(load_elem<T>(q, qi + e));
                    }
                }
            }
            *reinterpret_cast<u16x4*>(sK + i * KSTR + kk) = kb;
            *reinterpret_cast<u16x4*>(sQ + i * KSTR + kk) = qb;
        }
        for (int idx = tid; idx < VT * C; idx += THREADS) {
            const int i = idx / VT, vv = idx % VT;   // vv contiguous
            sU[vv * CSTR + i] =
                u_in[((static_cast<std::size_t>(b) * S + (t0 + i)) * HV + hv) *
                         static_cast<std::size_t>(V) + (v0 + vv)];
        }
        gate_scan<C>(g, beta, sGate, sBeta,
                     tid, (static_cast<std::size_t>(b) * S + t0) * HV + hv, HV);
        __syncthreads();
        const float c_last = sGate[C - 1];

        // ---- O = scale e^{c_i} ( q St^T + tril(q k^T) diag(e^{-c_j}) u ) ---
        gemm_nt_mmac<VT, C, KT, KSTR, KSTR, CSTR, THREADS>(sStB, sQ, sY);
        gemm_nt_mmac_ep<C, C, KT, KSTR, KSTR, THREADS>(
            sQ, sK, [&](int i, int j, float x) {
                sT[i * CSTR + j] = (j <= i) ? x * __expf(-sGate[j]) : 0.0f;
            });
        __syncthreads();
        gemm_nt_tf32_ep<VT, C, C, CSTR, CSTR, THREADS>(
            sU, sT, [&](int vv, int i, float x) {
                store_elem<T>(output,
                              ((static_cast<std::size_t>(b) * S + (t0 + i)) * HV + hv) *
                                      static_cast<std::size_t>(V) + (v0 + vv),
                              scale * __expf(sGate[i]) * (sY[vv * CSTR + i] + x));
            });
        __syncthreads();   // sQ, sY, sT are dead; sKT may take the pool

        // ---- replay the state, but only inside this segment ---------------
        if (n + 1 < nend) {
            const int ti = tid % C;
            const float esc = __expf(c_last - sGate[ti]);
            const unsigned short* kr = sK + ti * KSTR;
            for (int idx = tid; idx < C * KT; idx += THREADS) {
                const int kk = idx / C;
                sKT[kk * CSTR + ti] = esc * bf2f(kr[kk]);
            }
            __syncthreads();
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
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

int split_cu_count() {
    static const int n = [] {
        int dev = 0, cus = 80;
        if (hipGetDevice(&dev) == hipSuccess) {
            hipDeviceProp_t p{};
            if (hipGetDeviceProperties(&p, dev) == hipSuccess) cus = p.multiProcessorCount;
        }
        return cus;
    }();
    return n;
}

int split_value_tile(int V) { return (V % 64 == 0) ? 64 : 32; }

// Smallest G that makes B*HV*(V/VT)*G a multiple of the CU count, capped at the
// chunk count.  Larger multiples of it would give the same total barrier count
// with shorter replay chains, so there is nothing to gain from going higher.
int split_seg_count(int blocks0, int nchunks) {
    const int cus = split_cu_count();
    int a = blocks0 % cus, b = cus;
    while (a != 0) { const int t = b % a; b = a; a = t; }
    int g = cus / b;
    if (g < 1) g = 1;
    if (g > nchunks) g = nchunks;
    return g;
}

struct SplitShape {
    int B, S, H, HV, K, V;
};

bool split_shape_ok(const SplitShape& s) {
    if (s.B <= 0 || s.S <= 0 || s.H <= 0 || s.HV <= 0 || s.K <= 0 || s.V <= 0) return false;
    if (s.HV % s.H) return false;
    if (s.S % kChunk) return false;
    // First cut targets the long-sequence shapes the fused kernel loses on.
    // K/V outside this set fall back to the fused route, which already wins there.
    if (s.K != 64 && s.K != 128) return false;
    if (s.V != 64 && s.V != 128) return false;
    return true;
}

template <typename T, int KT, int V1, int VT>
void split_launch_inst(const T* q, const T* k, const T* v, const float* g,
                       const float* beta, const float* initial_state, T* output,
                       float* final_state, float* uv, float* w, float* seg_states,
                       const SplitShape& s, float scale, int nseg, int cps,
                       hipStream_t stream) {
    constexpr int TP1 = SplitThreads<KT, V1>::value;
    constexpr int TS1 = SplitThreads<KT, VT>::value;
    const int nchunks = s.S / kChunk;
    const int slices = s.V / VT;

    hipLaunchKernelGGL((gdn_split_p1_kernel<T, kChunk, KT, V1, TP1>),
                       dim3(s.B * s.HV * nchunks), dim3(TP1), 0, stream,
                       k, v, g, beta, uv, w, s.B, s.S, s.H, s.HV, s.K, s.V);
    hipLaunchKernelGGL((gdn_split_s1_kernel<T, kChunk, KT, VT, TS1>),
                       dim3(s.B * s.HV * slices), dim3(TS1), 0, stream,
                       k, g, beta, initial_state, w, uv, seg_states, final_state,
                       s.B, s.S, s.H, s.HV, s.K, s.V, nseg, cps);
    hipLaunchKernelGGL((gdn_split_p2_kernel<T, kChunk, KT, VT, TS1>),
                       dim3(s.B * s.HV * slices * nseg), dim3(TS1), 0, stream,
                       q, k, g, beta, seg_states, uv, output,
                       s.B, s.S, s.H, s.HV, s.K, s.V, scale, nseg, cps);
}

template <typename T>
void split_launch(const T* q, const T* k, const T* v, const float* g,
                  const float* beta, const float* initial_state, T* output,
                  float* final_state, float* workspace, const SplitShape& s,
                  float scale, hipStream_t stream) {
    const int VT = split_value_tile(s.V);
    const int nchunks = s.S / kChunk;
    const int blocks0 = s.B * s.HV * (s.V / VT);
    const int nseg = split_seg_count(blocks0, nchunks);
    const int cps = (nchunks + nseg - 1) / nseg;

    const std::size_t tok = static_cast<std::size_t>(s.B) * s.S * s.HV;
    float* const uv = workspace;
    float* const w = uv + tok * s.V;
    float* const seg_states = w + tok * s.K;

#define SPLIT_CASE(KTv, V1v, VTv)                                                   \
    if (s.K == (KTv) && s.V == (V1v) && VT == (VTv)) {                              \
        split_launch_inst<T, KTv, V1v, VTv>(q, k, v, g, beta, initial_state, output,\
                                            final_state, uv, w, seg_states, s,      \
                                            scale, nseg, cps, stream);              \
        return;                                                                     \
    }
    SPLIT_CASE(64, 64, 32)
    SPLIT_CASE(64, 64, 64)
    SPLIT_CASE(64, 128, 64)
    SPLIT_CASE(128, 64, 32)
    SPLIT_CASE(128, 64, 64)
    SPLIT_CASE(128, 128, 64)
#undef SPLIT_CASE
}

}  // namespace

extern "C" {

int flash_train_gdn_split_supported(int B, int S, int H, int HV, int K, int V) {
    return split_shape_ok(SplitShape{B, S, H, HV, K, V}) ? 1 : 0;
}

// Bytes of scratch the three-kernel path needs: Uv (overwritten in place by u),
// W, and the per-segment chunk-entry states.
std::size_t flash_train_gdn_split_workspace(int B, int S, int H, int HV, int K, int V) {
    (void)H;
    const SplitShape s{B, S, H, HV, K, V};
    if (!split_shape_ok(s)) return 0;
    const int VT = split_value_tile(V);
    const int nchunks = S / kChunk;
    const int nseg = split_seg_count(B * HV * (V / VT), nchunks);
    const std::size_t tok = static_cast<std::size_t>(B) * S * HV;
    const std::size_t nfloat = tok * (static_cast<std::size_t>(V) + K) +
                               static_cast<std::size_t>(B) * HV * nseg * V * K;
    return nfloat * sizeof(float);
}

void flash_train_gdn_split_bf16_forward_launch(
    const void* q, const void* k, const void* v, const float* g, const float* beta,
    const float* initial_state, void* output, float* final_state, float* workspace,
    int B, int S, int H, int HV, int K, int V, float scale, hipStream_t stream) {
    split_launch<uint16_t>(static_cast<const uint16_t*>(q), static_cast<const uint16_t*>(k),
                           static_cast<const uint16_t*>(v), g, beta, initial_state,
                           static_cast<uint16_t*>(output), final_state, workspace,
                           SplitShape{B, S, H, HV, K, V}, scale, stream);
}

void flash_train_gdn_split_fp32_forward_launch(
    const float* q, const float* k, const float* v, const float* g, const float* beta,
    const float* initial_state, float* output, float* final_state, float* workspace,
    int B, int S, int H, int HV, int K, int V, float scale, hipStream_t stream) {
    split_launch<float>(q, k, v, g, beta, initial_state, output, final_state, workspace,
                        SplitShape{B, S, H, HV, K, V}, scale, stream);
}

}  // extern "C"
