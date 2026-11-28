// Chunk-parallel stages of the large-shape GDN forward route:
//   intra   (stages 1+2): gc = chunk-local cumsum(g)/ln2, A = (I + M)^-1 with
//           M = strict_tril(beta_i 2^(gc_i-gc_j) k_i.k_j), w = A (k*beta*2^gc),
//           u = A (v*beta).   One block per (chunk, b*hv); A stays in LDS.
//   chunk_o (stage 4):  o = scale (2^gc_t q_t h + sum_{s<=t} 2^(gc_t-gc_s) q_t.k_s vn_s)
// I/O layouts match FLA (chunk = 64, gc in log2 units, h [B,NT,HV,K,V]) so each
// stage can be checked against FLA's own intermediate.  Full chunks only
// (S % 64 == 0).  All global traffic is 8- or 16-byte vector accesses.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "flash_train_gdn_chunk_parallel.h"
#include "gdn_chunk_mmac.h"

namespace {

using gdn_mmac::bf2f;
using gdn_mmac::f2bf;
using gdn_mmac::u16x4;
using gdn_mmac::v4f;
using gdn_mmac::v4s;
typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));

constexpr int kC = 64;
constexpr int kThreads = 256;
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ v4f mmac(v4s a, v4s b, v4f c) {
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
}

// ============================================================================ intra
constexpr int kIT = 1024;  // 16 waves; LDS is ~25 KB so two blocks share a CU
constexpr int kIW = kIT / 64;

// Transposed operand build: dst[n][s] = bf16(src[s][n] * scale_s) for a 64 x N tile
// read from global (rows `rstride` elements apart).  One thread handles a 4(s) x 4(n)
// block: four 8-byte global loads (issued early, held in registers), an in-register
// transpose, four 8-byte LDS stores.  The lane -> block mapping keeps each 16-lane
// LDS store group on distinct banks.  16*N/4 <= 512 units, so one block pass covers it.
struct TileRegs {
    u16x4 x[4];
};

template <int N>
__device__ __forceinline__ void load_tile(TileRegs& t, const uint16_t* src, std::size_t rstride, int u) {
    if (u < 16 * (N / 4)) {
        const int sq = (u & 3) | (((u >> 5) & 3) << 2);
        const int nq = ((u >> 2) & 7) | ((u >> 7) << 3);
#pragma unroll
        for (int r = 0; r < 4; ++r)
            t.x[r] = *reinterpret_cast<const u16x4*>(src + static_cast<std::size_t>(sq * 4 + r) * rstride + nq * 4);
    }
}

template <int N, int BS, bool TWO_SCALES>
__device__ __forceinline__ void store_tile(unsigned short* dst, const TileRegs& t, const float* sc0,
                                           const float* sc1, int u) {
    if (u < 16 * (N / 4)) {
        const int sq = (u & 3) | (((u >> 5) & 3) << 2);
        const int nq = ((u >> 2) & 7) | ((u >> 7) << 3);
        const int s0 = sq * 4, n0 = nq * 4;
        float sc[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) sc[r] = sc0[s0 + r];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            u16x4 o;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                float f = bf2f(t.x[r][j]) * sc[r];
                if (TWO_SCALES) f *= sc1[s0 + r];
                o[r] = f2bf(f);
            }
            *reinterpret_cast<u16x4*>(dst + (n0 + j) * BS + s0) = o;
        }
    }
}

template <int K, int V, bool WRITE_A>
__global__ __launch_bounds__(kIT) void gdn_intra_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    uint16_t* __restrict__ w, uint16_t* __restrict__ u, float* __restrict__ gc_out,
    uint16_t* __restrict__ A_out, int S, int H, int HV) {
    constexpr int KMAX = K > V ? K : V;
    constexpr int KS = K + 4;        // sK row stride (bf16)
    constexpr int OS = KMAX + 8;     // output staging stride (bf16, 16B rows)
    constexpr int XS = kC + 4;       // fp32 64x64 stride
    constexpr int BS = kC + 4;       // bf16 [n][s] operand stride
    constexpr int RA = kC * XS * 4;  // one 17 KB region, reused by every phase
    static_assert(kC * KS * 2 <= RA && KMAX * BS * 2 <= RA && kC * OS * 2 <= RA, "region RA too small");
    // RA holds, in turn: k tile -> X (M strictly-lower tiles + Inv^T upper tiles, one
    // fp32 64x64) -> kb^T / vb^T (bf16 [n][s]) -> w/u output staging.
    __shared__ __align__(16) unsigned char sRA[RA];
    __shared__ __align__(16) float sRB[7 * 256];  // sX (3 x 16x16) + sD (4 x 16x16) solver scratch
    __shared__ float sGc[kC], sB[kC], sE[kC];
    unsigned short* sK = reinterpret_cast<unsigned short*>(sRA);
    float* sXm = reinterpret_cast<float*>(sRA);
    unsigned short* sT = reinterpret_cast<unsigned short*>(sRA);
    unsigned short* sKO = reinterpret_cast<unsigned short*>(sRA);
    float* sX = sRB;
    float* sD = sRB + 3 * 256;

    const int tid = threadIdx.x, lane = tid & 63;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int lo = lane & 15, hq = lane >> 4, hi4 = hq * 4;
    // hv is the fastest block index: blocks resident together then touch neighbouring heads of
    // one chunk.  With chunk fastest they were a power-of-two stride (HV*K*2 B per chunk) apart
    // and every global stream of the kernel ran ~2.5% (A off) / ~12% (A on) slower.
    const int lin = blockIdx.y * gridDim.x + blockIdx.x;
    const int hv = lin % HV, n = (lin / HV) % gridDim.x, b = lin / HV / gridDim.x;
    const int hk = hv / (HV / H);
    const int t0 = n * kC;
    const std::size_t row0 = static_cast<std::size_t>(b) * S + t0;

    // kb^T / vb^T source tiles: issue the global loads now, consume them after the solve.
    TileRegs kreg, vreg;
    load_tile<K>(kreg, k + (row0 * H + hk) * K, static_cast<std::size_t>(H) * K, tid);
    load_tile<V>(vreg, v + (row0 * HV + hv) * V, static_cast<std::size_t>(HV) * V, tid);

    // ---- stage k, gates ----
    {
        const uint16_t* kp = k + (row0 * H + hk) * K;
        constexpr int PER = K / 8;
        for (int e = tid; e < kC * PER; e += kIT) {
            const int t = e / PER, c = (e % PER) * 8;
            const u16x8 x = *reinterpret_cast<const u16x8*>(kp + static_cast<std::size_t>(t) * H * K + c);
            *reinterpret_cast<u16x4*>(sK + t * KS + c) = u16x4{x[0], x[1], x[2], x[3]};
            *reinterpret_cast<u16x4*>(sK + t * KS + c + 4) = u16x4{x[4], x[5], x[6], x[7]};
        }
        if (wave == 0) {
            float x = g[(row0 + lane) * HV + hv] * kLog2e;
#pragma unroll
            for (int o = 1; o < 64; o <<= 1) {
                const float y = __shfl_up(x, o, 64);
                if (lane >= o) x += y;
            }
            const float bt = beta[(row0 + lane) * HV + hv];
            sGc[lane] = x;
            sB[lane] = bt;
            sE[lane] = exp2f(x);
            gc_out[(row0 + lane) * HV + hv] = x;
        }
    }
    __syncthreads();

    // ---- M = strict_tril(beta_i 2^(gc_i-gc_j) k_i.k_j): one 16x16 tile per wave ----
    {
        const int tm = wave >> 2, tn = wave & 3;
        v4f acc = {0.f, 0.f, 0.f, 0.f};
        if (tn <= tm) {
            const unsigned short* ap = sK + (tm * 16 + lo) * KS + hi4;
            const unsigned short* bp = sK + (tn * 16 + lo) * KS + hi4;
#pragma unroll
            for (int k0 = 0; k0 < K; k0 += 16)
                acc = mmac(*reinterpret_cast<const v4s*>(ap + k0), *reinterpret_cast<const v4s*>(bp + k0), acc);
        }
        __syncthreads();  // sK is overwritten by X below
        if (tn <= tm) {
            const int r = tm * 16 + lo;
            const float br = sB[r], gr = sGc[r];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int c = tn * 16 + hq + 4 * q;
                if (r > c) sXm[r * XS + c] = br * exp2f(gr - sGc[c]) * acc[q];
            }
        }
    }
    __syncthreads();

    // X holds Inv^T ([col][row]) in its diagonal + upper tiles, so it is directly the
    // [n][k] B operand of Inv_mj; sD keeps the diagonal blocks in normal layout for Inv_ii.
    // ---- diagonal 16x16 blocks: wave 0, lane = blk*16 + column j ----
    if (wave == 0) {
        const int blk = lane >> 4, j = lane & 15, base = blk * 16;
        float col[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float acc = 0.0f;
#pragma unroll
            for (int m = 0; m < i; ++m) acc += sXm[(base + i) * XS + base + m] * col[m];
            col[i] = (i > j) ? -acc : (i == j ? 1.0f : 0.0f);
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            sXm[(base + j) * XS + base + i] = col[i];
            sD[blk * 256 + i * 16 + j] = col[i];
        }
    }
    __syncthreads();

    // ---- off-diagonal block rows (tf32 mmac): Inv_ij = -Inv_ii sum_{m=j}^{i-1} M_im Inv_mj ----
    // One wave per block column bj.  X^T = (sum_m M_im Inv_mj)^T is staged in sX.
    if (wave < 3) {
        const int bj = wave;
        for (int bi = bj + 1; bi < 4; ++bi) {
            v4f acc = {0.f, 0.f, 0.f, 0.f};
            for (int bm = bj; bm < bi; ++bm) {
                const float* ap = sXm + (bi * 16 + lo) * XS + bm * 16 + hq * 2;
                const float* bp = sXm + (bj * 16 + lo) * XS + bm * 16 + hq * 2;
#pragma unroll
                for (int k0 = 0; k0 < 16; k0 += 8)
                    acc = __builtin_amdgcn_mmac_f32_16x16x8tf32(
                        *reinterpret_cast<const gdn_mmac::v2f*>(ap + k0),
                        *reinterpret_cast<const gdn_mmac::v2f*>(bp + k0), acc);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) sX[bj * 256 + (hq + 4 * q) * 16 + lo] = acc[q];
            asm volatile("" ::: "memory");
            __builtin_amdgcn_wave_barrier();
            v4f acc2 = {0.f, 0.f, 0.f, 0.f};
            const float* ap = sX + bj * 256 + lo * 16 + hq * 2;
            const float* bp = sD + bi * 256 + lo * 16 + hq * 2;
#pragma unroll
            for (int k0 = 0; k0 < 16; k0 += 8)
                acc2 = __builtin_amdgcn_mmac_f32_16x16x8tf32(
                    *reinterpret_cast<const gdn_mmac::v2f*>(ap + k0),
                    *reinterpret_cast<const gdn_mmac::v2f*>(bp + k0), acc2);
#pragma unroll
            for (int q = 0; q < 4; ++q) sXm[(bj * 16 + lo) * XS + bi * 16 + hq + 4 * q] = -acc2[q];
            asm volatile("" ::: "memory");
            __builtin_amdgcn_wave_barrier();
        }
    }
    __syncthreads();

    // ---- Inv rows of this wave's output tile -> bf16 A fragments, kept in registers ----
    // wave -> (tm = wave & 3, tn = wave >> 2 + 4 r); Inv[t][s] = X[s][t], zero for s-block > t-block.
    const int atm = wave & 3;
    v4s afr[4];
#pragma unroll
    for (int bs = 0; bs < 4; ++bs) {
        afr[bs] = v4s{0, 0, 0, 0};
        if (bs <= atm) {
#pragma unroll
            for (int e = 0; e < 4; ++e) afr[bs][e] = f2bf(sXm[(bs * 16 + hi4 + e) * XS + atm * 16 + lo]);
        }
    }
    // A (optional output): (I+M)^-1 is lower triangular, so afr of row tile atm has the blocks
    // bs <= atm non-zero.  Those are packed into the solver scratch sRB (dead since the
    // off-diagonal phase; 6 KB, row stride 16(atm+1)+8 elements), which needs no extra barrier
    // before the existing one below; each thread then writes one 16-byte piece of a row and the
    // blocks above the diagonal are written as zeros.  8-byte fragment stores straight from the
    // registers left only 32 B contiguous per row (+0.46 ms at llm32k).
    unsigned short* sAp = reinterpret_cast<unsigned short*>(sRB);
    static_assert((128 * 3 * (3 + 2) + 16 * (16 * 4 + 8)) * 2 <= 7 * 256 * 4, "A staging fits sRB");
    if constexpr (WRITE_A) {
        if (wave < 4) {
            unsigned short* rp = sAp + 128 * atm * (atm + 2) + lo * (16 * (atm + 1) + 8) + hi4;
#pragma unroll
            for (int bs = 0; bs < 4; ++bs)
                if (bs <= atm) *reinterpret_cast<v4s*>(rp + bs * 16) = afr[bs];
        }
    }
    __syncthreads();  // X is dead; its region becomes kb^T
    if constexpr (WRITE_A) {
        if (tid < kC * (kC / 8)) {
            const int t = tid / 8, j = tid % 8, ta = t >> 4, bs = j >> 1;
            u16x8 x = {0, 0, 0, 0, 0, 0, 0, 0};
            if (bs <= ta)
                x = *reinterpret_cast<const u16x8*>(sAp + 128 * ta * (ta + 2) + (t & 15) * (16 * (ta + 1) + 8) + j * 8);
            *reinterpret_cast<u16x8*>(A_out + ((row0 + t) * HV + hv) * kC + j * 8) = x;
        }
    }

    // ---- out[t][n] = sum_s Inv[t][s] T[n][s]; acc held in registers, then staged into sKO ----
    auto gemm_out = [&](auto nn) {
        constexpr int NN = decltype(nn)::value;
        constexpr int R = NN / 64;
        v4f acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] = v4f{0.f, 0.f, 0.f, 0.f};
            const unsigned short* bp = sT + (((wave >> 2) + 4 * r) * 16 + lo) * BS + hi4;
#pragma unroll
            for (int bs = 0; bs < 4; ++bs)
                if (bs <= atm) acc[r] = mmac(afr[bs], *reinterpret_cast<const v4s*>(bp + bs * 16), acc[r]);
        }
        __syncthreads();  // kb^T / vb^T is overwritten by the staging tile
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                sKO[(atm * 16 + lo) * OS + ((wave >> 2) + 4 * r) * 16 + hq + 4 * q] = f2bf(acc[r][q]);
        __syncthreads();
    };
    auto store_rows = [&](uint16_t* dst, int NN) {
        const int per = NN / 8;
        for (int e = tid; e < kC * per; e += kIT) {
            const int t = e / per, c = (e % per) * 8;
            *reinterpret_cast<u16x8*>(dst + static_cast<std::size_t>(t) * HV * NN + c) =
                *reinterpret_cast<const u16x8*>(sKO + t * OS + c);
        }
    };
    // kb^T[kk][s] = k[s][kk] beta_s 2^gc_s  (k tile prefetched in registers: its LDS copy was reused)
    store_tile<K, BS, true>(sT, kreg, sB, sE, tid);
    __syncthreads();
    gemm_out(std::integral_constant<int, K>{});
    store_rows(w + (row0 * HV + hv) * K, K);
    __syncthreads();
    // vb^T[v][s] = v[s][v] beta_s
    store_tile<V, BS, false>(sT, vreg, sB, sB, tid);
    __syncthreads();
    gemm_out(std::integral_constant<int, V>{});
    store_rows(u + (row0 * HV + hv) * V, V);
}

// ============================================================================ chunk_o
// o = scale (2^gc_t q_t h + sum_{s<=t} 2^(gc_t-gc_s) q_t.k_s vn_s) for one 64-row chunk.
//
// Wave w owns rows t = 16w..16w+15.  P = q k^T is gated in registers and feeds the
// P.vn mmac directly as the A operand: the D layout hands a lane the columns
// s = 16ts + hq + 4i, so the P.vn contraction runs over that permuted s order and
// vn^T is stored in LDS with the same permutation (no P round trip through LDS).
//
// A block covers BV = VB*NP columns of V (the whole of V for the llm shapes), so q, k
// and P are read / computed once per chunk-head.  The V range is walked in NP passes of
// VB columns and only one pass of h and vn^T is resident in LDS (the 17 KB block also
// holds the k tile, which the four waves share through LDS instead of re-reading 8-byte
// fragments from global), so three blocks fit a CU.  All global loads (k, every pass of h
// and vn, q, gc) are issued before the first wait.
//
// The B-operand row of lane `lo` is the actual v column 4*(lo&3)+(lo>>2) of its tile, so
// each lane ends up holding four consecutive v of one row.  OG consecutive passes are
// staged in LDS (aliasing the h buffer) and written as 16-byte stores, OG*VB columns per
// row at a time.
//
// Block order matters a lot: hv is the fastest grid index, so the blocks resident at the
// same time read neighbouring memory (q / k / h of one chunk across heads).  With the
// natural (v, chunk, bhv) order they sit a power-of-two stride apart and the kernel was
// 1.3x slower.
constexpr int kCoVB = 32;  // columns of V per pass
constexpr int kCoNP = 4;   // passes per block (BV = kCoVB * kCoNP = 128)
constexpr int kCoOG = 2;   // passes whose output is staged together

template <int K, int VB, int NP, int OG>
__global__ __launch_bounds__(kThreads) void gdn_chunk_o_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ vn, const uint16_t* __restrict__ h,
    const float* __restrict__ gc, uint16_t* __restrict__ o,
    int S, int H, int HV, int V, float scale) {
    constexpr int KT16 = K / 16, VT16 = VB / 16, BV = VB * NP;
    constexpr int VS = kC + 4, HS = K + 4, OS = OG * VB + 4;
    constexpr int HPER = K / 8, NH = (VB * HPER + kThreads - 1) / kThreads;
    constexpr int VPER = VB / 8, NV = (kC * VPER + kThreads - 1) / kThreads;
    constexpr int OPER = OG * VB / 8;
    constexpr int HSZ = (VB * HS > kC * OS) ? VB * HS : kC * OS;
    static_assert(NP % OG == 0, "output groups tile the passes");
    constexpr int KS = K + 4;  // sK[s][kk]
    constexpr int BUF = (HSZ + VB * VS > kC * KS) ? HSZ + VB * VS : kC * KS;
    __shared__ __align__(16) unsigned short sBuf[BUF];
    __shared__ float sGc[kC];
    unsigned short* const sH = sBuf;  // h pass, later the output staging
    unsigned short* const sVT = sBuf + HSZ;
    unsigned short* const sO = sH;
    unsigned short* const sK = sBuf;  // k tile, only alive during P = q k^T

    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int lo = lane & 15, hq = lane >> 4, hi4 = hq * 4;
    const int NT = S / kC, nvb = V / BV;
    int id = blockIdx.x;
    const int hv = id % HV;
    id /= HV;
    const int v0 = (id % nvb) * BV;
    id /= nvb;
    const int n = id % NT, b = id / NT;
    const int hk = hv / (HV / H);
    const std::size_t row0 = static_cast<std::size_t>(b) * S + n * kC;
    const std::size_t qkRow = static_cast<std::size_t>(H) * K;
    const int tm = wave, tr = tm * 16 + lo;
    const int fl = 4 * (lo & 3) + (lo >> 2);
    const uint16_t* qB = q + row0 * qkRow + static_cast<std::size_t>(hk) * K + hi4;
    const uint16_t* hB = h + ((static_cast<std::size_t>(b) * NT + n) * HV + hv) * V * K +
                         static_cast<std::size_t>(v0) * K;
    const uint16_t* vB = vn + (row0 * HV + hv) * V + v0;

    // ---- every global load of the prologue is issued before the first wait ----
    u16x8 xh[NP][NH], xv[NP][NV];
#pragma unroll
    for (int j = 0; j < NP; ++j) {
#pragma unroll
        for (int i = 0; i < NH; ++i) {
            const int e = tid + i * kThreads;
            if (NH * kThreads == VB * HPER || e < VB * HPER)
                xh[j][i] = *reinterpret_cast<const u16x8*>(
                    hB + static_cast<std::size_t>(j * VB + e / HPER) * K + (e % HPER) * 8);
        }
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int e = tid + i * kThreads;
            if (NV * kThreads == kC * VPER || e < kC * VPER)
                xv[j][i] = *reinterpret_cast<const u16x8*>(
                    vB + static_cast<std::size_t>(e / VPER) * HV * V + j * VB + (e % VPER) * 8);
        }
    }
    constexpr int KPER = K / 8, NK = kC * KPER / kThreads;
    static_assert(kC * KPER % kThreads == 0, "k tile / thread mapping");
    u16x8 xk[NK];
#pragma unroll
    for (int i = 0; i < NK; ++i) {
        const int e = tid + i * kThreads;
        xk[i] = *reinterpret_cast<const u16x8*>(k + (row0 + e / KPER) * qkRow + static_cast<std::size_t>(hk) * K + (e % KPER) * 8);
    }
    v4s qa[KT16];
#pragma unroll
    for (int tk = 0; tk < KT16; ++tk)
        qa[tk] = *reinterpret_cast<const v4s*>(qB + static_cast<std::size_t>(tr) * qkRow + 16 * tk);
    const float gcv = tid < kC ? gc[(row0 + tid) * HV + hv] : 0.f;

    // ---- P = q k^T over the s-tiles <= this wave's row tile (runs under the loads) ----
#pragma unroll
    for (int i = 0; i < NK; ++i) {
        const int e = tid + i * kThreads, s = e / KPER, c = (e % KPER) * 8;
        *reinterpret_cast<u16x4*>(sK + s * KS + c) = u16x4{xk[i][0], xk[i][1], xk[i][2], xk[i][3]};
        *reinterpret_cast<u16x4*>(sK + s * KS + c + 4) = u16x4{xk[i][4], xk[i][5], xk[i][6], xk[i][7]};
    }
    __syncthreads();
    v4f pacc[4];
#pragma unroll
    for (int ts = 0; ts < 4; ++ts) {
        pacc[ts] = v4f{0.f, 0.f, 0.f, 0.f};
        if (ts <= tm) {
            const unsigned short* kp = sK + (ts * 16 + lo) * KS + hi4;
#pragma unroll
            for (int tk = 0; tk < KT16; ++tk)
                pacc[ts] = mmac(qa[tk], *reinterpret_cast<const v4s*>(kp + 16 * tk), pacc[ts]);
        }
    }
    __syncthreads();  // sK is overwritten by the first h / vn pass

    auto store_pass = [&](int j) {
#pragma unroll
        for (int i = 0; i < NH; ++i) {
            const int e = tid + i * kThreads;
            if (NH * kThreads == VB * HPER || e < VB * HPER) {
                const int vr = e / HPER, c = (e % HPER) * 8;
                *reinterpret_cast<u16x4*>(sH + vr * HS + c) = u16x4{xh[j][i][0], xh[j][i][1], xh[j][i][2], xh[j][i][3]};
                *reinterpret_cast<u16x4*>(sH + vr * HS + c + 4) = u16x4{xh[j][i][4], xh[j][i][5], xh[j][i][6], xh[j][i][7]};
            }
        }
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int e = tid + i * kThreads;
            if (NV * kThreads == kC * VPER || e < kC * VPER) {
                const int s = e / VPER, c = (e % VPER) * 8;
                const int ps = (s & ~15) | ((s & 3) << 2) | ((s >> 2) & 3);
#pragma unroll
                for (int r = 0; r < 8; ++r) sVT[(c + r) * VS + ps] = xv[j][i][r];
            }
        }
    };
    if (tid < kC) sGc[tid] = gcv;
    store_pass(0);
    __syncthreads();

    // ---- gate + causal mask in registers: P becomes the A operand of P.vn ----
    const float gt = sGc[tr];
    const float et = exp2f(gt) * scale;
    v4s af[4];
#pragma unroll
    for (int ts = 0; ts < 4; ++ts) {
        af[ts] = v4s{0, 0, 0, 0};
        if (ts <= tm) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int c = ts * 16 + hq + 4 * i;
                const float x = (c <= tr) ? pacc[ts][i] * exp2f(gt - sGc[c]) * scale : 0.0f;
                af[ts][i] = static_cast<short>(f2bf(x));
            }
        }
    }

    uint16_t* oB = o + (row0 * HV + hv) * V + v0;
    v4f og[OG][VT16];
#pragma unroll
    for (int j = 0; j < NP; ++j) {
        if (j > 0) {
            store_pass(j);
            __syncthreads();
        }
#pragma unroll
        for (int tv = 0; tv < VT16; ++tv) {
            const unsigned short* hp = sH + (tv * 16 + fl) * HS + hi4;
            v4f acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int tk = 0; tk < KT16; ++tk)
                acc = mmac(qa[tk], *reinterpret_cast<const v4s*>(hp + 16 * tk), acc);
            og[j % OG][tv] = acc * et;
        }
#pragma unroll
        for (int tv = 0; tv < VT16; ++tv) {
            const unsigned short* bp = sVT + (tv * 16 + fl) * VS + hi4;
#pragma unroll
            for (int ts = 0; ts < 4; ++ts)
                if (ts <= tm)
                    og[j % OG][tv] = mmac(af[ts], *reinterpret_cast<const v4s*>(bp + ts * 16), og[j % OG][tv]);
        }
        __syncthreads();  // nobody reads this pass's sH / sVT any more
        if (j % OG == OG - 1) {
#pragma unroll
            for (int g = 0; g < OG; ++g)
#pragma unroll
                for (int tv = 0; tv < VT16; ++tv)
                    *reinterpret_cast<u16x4*>(sO + tr * OS + g * VB + tv * 16 + hi4) =
                        u16x4{f2bf(og[g][tv][0]), f2bf(og[g][tv][1]), f2bf(og[g][tv][2]), f2bf(og[g][tv][3])};
            __builtin_amdgcn_wave_barrier();
#pragma unroll
            for (int e = lane; e < 16 * OPER; e += 64) {
                const int rr = tm * 16 + e / OPER, c = (e % OPER) * 8;
                const u16x4 x0 = *reinterpret_cast<const u16x4*>(sO + rr * OS + c);
                const u16x4 x1 = *reinterpret_cast<const u16x4*>(sO + rr * OS + c + 4);
                *reinterpret_cast<u16x8*>(oB + static_cast<std::size_t>(rr) * HV * V + (j + 1 - OG) * VB + c) =
                    u16x8{x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]};
            }
            __builtin_amdgcn_wave_barrier();
            if (j + 1 < NP) __syncthreads();  // sH is about to be refilled
        }
    }
}

}  // namespace

extern "C" int flash_train_gdn_intra_bf16(const FlashTrainGDNIntraArgs* a, void* stream) {
    if (!a || !a->k || !a->v || !a->g || !a->beta || !a->w || !a->u || !a->gc) return 1;
    if (a->S % kC || a->HV % a->H) return 2;
    hipStream_t st = static_cast<hipStream_t>(stream);
    dim3 grid(a->S / kC, a->B * a->HV);
#define L(KK, VV)                                                                                     \
    if (a->K == KK && a->V == VV) {                                                                   \
        if (a->A) {                                                                                   \
            hipLaunchKernelGGL((gdn_intra_kernel<KK, VV, true>), grid, dim3(kIT), 0, st, a->k, a->v,  \
                               a->g, a->beta, a->w, a->u, a->gc, a->A, a->S, a->H, a->HV);            \
        } else {                                                                                      \
            hipLaunchKernelGGL((gdn_intra_kernel<KK, VV, false>), grid, dim3(kIT), 0, st, a->k, a->v, \
                               a->g, a->beta, a->w, a->u, a->gc, nullptr, a->S, a->H, a->HV);         \
        }                                                                                             \
        return hipGetLastError() == hipSuccess ? 0 : 5;                                               \
    }
    L(128, 128) L(64, 128) L(128, 64) L(64, 64)
#undef L
    return 2;
}

extern "C" int flash_train_gdn_chunk_o_bf16(const FlashTrainGDNChunkOArgs* a, void* stream) {
    if (!a || !a->q || !a->k || !a->v_new || !a->h || !a->gc || !a->o) return 1;
    if (a->S % kC || a->V % (kCoVB * 2) || a->HV % a->H) return 2;
    hipStream_t st = static_cast<hipStream_t>(stream);
    // Whole-V blocks when V allows it (q / k / P done once per chunk-head), else half-V.
    const bool wide = a->V % (kCoVB * kCoNP) == 0;
    const int bv = wide ? kCoVB * kCoNP : kCoVB * 2;
    const dim3 grid(static_cast<unsigned>((a->V / bv) * (a->S / kC) * a->B * a->HV));
#define CO_LAUNCH(KK, NPP)                                                                         \
    hipLaunchKernelGGL((gdn_chunk_o_kernel<KK, kCoVB, NPP, kCoOG>), grid, dim3(kThreads), 0, st, \
                       a->q, a->k, a->v_new, a->h, a->gc, a->o, a->S, a->H, a->HV, a->V, a->scale)
    if (a->K == 128) {
        if (wide) CO_LAUNCH(128, kCoNP); else CO_LAUNCH(128, 2);
    } else if (a->K == 64) {
        if (wide) CO_LAUNCH(64, kCoNP); else CO_LAUNCH(64, 2);
    } else {
        return 2;
    }
#undef CO_LAUNCH
    return hipGetLastError() == hipSuccess ? 0 : 5;
}
