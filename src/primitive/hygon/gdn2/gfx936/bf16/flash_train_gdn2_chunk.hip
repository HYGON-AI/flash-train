// Chunked (chunk = 64) three-kernel forward for GDN-2 (per-channel decay).
//
//   intra   : gc = chunk-local cumsum(g)/ln2 (per channel), Akk, Akk^-1, w, u,
//             plus the operands the later stages need: kg = k*2^(gc_last-gc),
//             qg = q*2^gc, Aqk (bf16, scale and j<=i mask applied), gc_last.
//   state_h : v_new = u - w h ; h <- diag(2^gc_last) h + kg^T v_new       (serial over chunks)
//   chunk_o : o = scale qg h + Aqk v_new
//
// Per-channel decay as GEMM.  2^(gc_i[k]-gc_j[k]) is split around the first row r
// of the 16-row sub-block tn that holds j:
//     2^(gc_i-gc_r) * 2^(gc_r-gc_j)
// gc is non-increasing in t, so for i >= r the left factor is <= 1, and for j in
// the same sub-block as r the right factor is >= 1 and bounded by 15 steps of
// decay.  Both are safe in fp32/bf16 range for realistic gates (right factor is
// clamped at 2^kExpClamp so a pathological gate cannot make Inf*0 = NaN).  Each
// pair (tm >= tn) of sub-blocks is then a plain 16x16xK mmac GEMM with
//     A[i][k] = (k_i b_i | q_i) * 2^(gc_i-gc_r(tn)),  B[j][k] = k_j * 2^(gc_r(tn)-gc_j).
// The Gram runs on the tf32 matrix core with fp32 operands built in registers: bf16
// operands lose the gate factor's precision and made Aqk/o measurably worse than FLA
// (whose tl.dot also takes fp32 -> tf32).  The w/u products keep bf16 operands.
//
// Layouts follow FLA (chunk = 64; h V-first [B,NT,HV,V,K]).  Full chunks only.
// All global traffic is 8- or 16-byte vector accesses.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "../common/flash_train_gdn2_chunk.h"
#include "../common/gdn2_chunk_mmac.h"

namespace {

using gdn2_mmac::bf2f;
using gdn2_mmac::f2bf;
using gdn2_mmac::u16x4;
using gdn2_mmac::v4f;
using gdn2_mmac::v4s;
typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));

constexpr int kC = 64;
constexpr int kThreads = 256;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kExpClamp = 120.0f;  // max exponent of the right (>= 1) factor

__device__ __forceinline__ v4f mmac(v4s a, v4s b, v4f c) {
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
}

// ============================================================================ intra
constexpr int kIT = 1024;  // 16 waves
constexpr int kKS = 32;    // channels per slice of the Gram loop

struct SliceRegs {
    v4f g, b;
    u16x4 k, q;
};

struct VRegs {
    u16x4 p[4];  // bf16(v * wg) for four consecutive rows
};

template <int K, int V, bool WRITE_AKK>
__global__ __launch_bounds__(kIT) void gdn2_intra_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ bg, const float* __restrict__ wg,
    uint16_t* __restrict__ w, uint16_t* __restrict__ u, uint16_t* __restrict__ kg,
    uint16_t* __restrict__ qg, uint16_t* __restrict__ aqk, float* __restrict__ gclast,
    float* __restrict__ gc_out, uint16_t* __restrict__ akk_inv, int S, int H, int HV, float scale) {
    constexpr int KMAX = K > V ? K : V;
    constexpr int NS = K / kKS;
    constexpr int GS = kKS + 4;   // fp32 slice stride (sG, sKB)
    constexpr int QS = kKS + 4;   // bf16 slice stride (sQ, sBall)
    constexpr int XS = kC + 4;    // fp32 64x64 stride
    constexpr int BS = kC + 4;    // bf16 [n][s] operand stride
    constexpr int OS = KMAX + 8;  // output staging stride
    constexpr int AS = kC + 8;    // Aqk staging stride
    constexpr int R1B = K * BS * 2;
    constexpr int R2B = 28672;
    static_assert(kKS * 2 == 64 && K % kKS == 0, "slice layout");
    static_assert(kC * GS * 4 * 2 + kC * QS * 2 * 2 + 8 * kKS * 4 <= R2B, "slice buffers");
    static_assert(QS % 4 == 0, "u16x4 fragment reads");
    static_assert(kC * XS * 4 + kC * AS * 2 <= R2B, "X + Aqk staging");
    static_assert(V * BS * 2 <= R2B && kC * OS * 2 <= R2B, "vb^T / output staging");

    __shared__ __align__(16) unsigned char sR1[R1B];  // kb2^T [K][BS] bf16, alive from the Gram loop to the w product
    __shared__ __align__(16) unsigned char sR2[R2B];  // slice buffers | X + Aqk staging | vb^T | output staging
    __shared__ __align__(16) float sRB[7 * 256];      // solver scratch

    unsigned short* sKb2T = reinterpret_cast<unsigned short*>(sR1);
    float* sG = reinterpret_cast<float*>(sR2);
    float* sKB = sG + kC * GS;
    unsigned short* sQ = reinterpret_cast<unsigned short*>(sR2 + 2 * kC * GS * 4);
    unsigned short* sBall = sQ + kC * QS;
    float* sTot = reinterpret_cast<float*>(sR2 + 2 * kC * GS * 4 + 2 * kC * QS * 2);
    float* sXm = reinterpret_cast<float*>(sR2);
    unsigned short* sAqS = reinterpret_cast<unsigned short*>(sR2 + kC * XS * 4);
    unsigned short* sT = reinterpret_cast<unsigned short*>(sR2);
    unsigned short* sKO = reinterpret_cast<unsigned short*>(sR2);
    float* sX = sRB;
    float* sD = sRB + 3 * 256;

    const int tid = threadIdx.x, lane = tid & 63;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int lo = lane & 15, hq = lane >> 4, hi4 = hq * 4;
    int id = blockIdx.x;
    const int hv = id % HV;
    id /= HV;
    const int NT = S / kC;
    const int n = id % NT, b = id / NT;
    const int hk = hv / (HV / H);
    const int t0 = n * kC;
    const std::size_t row0 = static_cast<std::size_t>(b) * S + t0;

    // ---- slice loads: thread (row, c4) of the first 512 threads owns 4 channels of one row ----
    // Lane-dependent indices are re-derived from tid inside the slice loop: the compiler
    // otherwise hoists ~26 loop invariants into VGPRs and spills them (64-VGPR cap at 1024 threads).
    auto hid = [](int x) { asm volatile("" : "+v"(x)); return x; };
    auto load_slice = [&](int ks, SliceRegs& r) {
        if (tid < 512) {
            const int tl = hid(tid), lr = tl >> 3, cc = tl & 7;
            const unsigned oGK = static_cast<unsigned>(((row0 + lr) * HV + hv) * K + cc * 4);
            const unsigned oQK = static_cast<unsigned>(((row0 + lr) * H + hk) * K + cc * 4);
            r.g = *reinterpret_cast<const v4f*>(g + oGK + ks * kKS);
            r.b = *reinterpret_cast<const v4f*>(bg + oGK + ks * kKS);
            r.k = *reinterpret_cast<const u16x4*>(k + oQK + ks * kKS);
            r.q = *reinterpret_cast<const u16x4*>(q + oQK + ks * kKS);
        }
    };

    // pair (tm >= tn) of this wave (waves 0..9 only)
    const int pm = (wave >= 1) + (wave >= 3) + (wave >= 6);
    const int pn = wave - pm * (pm + 1) / 2;
    v4f acck = {0.f, 0.f, 0.f, 0.f}, accq = {0.f, 0.f, 0.f, 0.f};

    SliceRegs cur;
    load_slice(0, cur);
#pragma unroll 1
    for (int ks = 0; ks < NS; ++ks) {
        const int tl = hid(tid), lrow = tl >> 3, c4 = tl & 7;
        const int lo = tl & 15, hq = (tl >> 4) & 3;
        const unsigned offGK = static_cast<unsigned>(((row0 + lrow) * HV + hv) * K + c4 * 4);
        // ---- stage g (log2 units) ----
        if (tid < 512) *reinterpret_cast<v4f*>(sG + lrow * GS + c4 * 4) = cur.g * kLog2e;
        __syncthreads();
        // ---- cumsum over the 64 rows, per channel: 8 segments of 8 rows ----
        const int sch = tl & 31, sseg = tl >> 5;
        if (tid < 256) {
            float run = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                run += sG[(sseg * 8 + i) * GS + sch];
                sG[(sseg * 8 + i) * GS + sch] = run;
            }
            sTot[sseg * 32 + sch] = run;
        }
        __syncthreads();
        if (tid < 256) {
            float off = 0.f;
            for (int s = 0; s < sseg; ++s) off += sTot[s * 32 + sch];
#pragma unroll
            for (int i = 0; i < 8; ++i) sG[(sseg * 8 + i) * GS + sch] += off;
        }
        __syncthreads();
        // ---- build operands from gc ----
        if (tid < 512) {
            const v4f gv = *reinterpret_cast<const v4f*>(sG + lrow * GS + c4 * 4);
            const v4f gl = *reinterpret_cast<const v4f*>(sG + (kC - 1) * GS + c4 * 4);
            u16x4 qgv, kgv;
            v4f kbv;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float kk = bf2f(cur.k[i]), qq = bf2f(cur.q[i]), e = gv[i];
                qgv[i] = f2bf(qq * exp2f(e));
                kgv[i] = f2bf(kk * exp2f(gl[i] - e));
                const float kbf = kk * cur.b[i];
                kbv[i] = kbf;
                sKb2T[(ks * kKS + c4 * 4 + i) * BS + lrow] = f2bf(kbf * exp2f(e));
            }
            *reinterpret_cast<v4f*>(sKB + lrow * GS + c4 * 4) = kbv;
            *reinterpret_cast<u16x4*>(sQ + lrow * QS + c4 * 4) = cur.q;
            *reinterpret_cast<u16x4*>(sBall + lrow * QS + c4 * 4) = cur.k;
            *reinterpret_cast<u16x4*>(qg + offGK + ks * kKS) = qgv;
            *reinterpret_cast<u16x4*>(kg + offGK + ks * kKS) = kgv;
            if (gc_out != nullptr) *reinterpret_cast<v4f*>(gc_out + offGK + ks * kKS) = gv;
            if (lrow == kC - 1)
                *reinterpret_cast<v4f*>(gclast + ((static_cast<std::size_t>(b) * NT + n) * HV + hv) * K + ks * kKS + c4 * 4) = gl;
        }
        if (ks + 1 < NS) load_slice(ks + 1, cur);  // cur is dead; the loads overlap the Gram below
        __syncthreads();
        // ---- Gram tiles of this slice (tf32 mmac, k = 2*hq + i within each step of 8 channels) ----
        if (wave < 10) {
            const int irow = pm * 16 + lo;
#pragma unroll 1
            for (int c = 0; c < kKS / 8; ++c) {
                const int ch0 = 8 * c + 2 * hq;
                const gdn2_mmac::v2f gi = *reinterpret_cast<const gdn2_mmac::v2f*>(sG + irow * GS + ch0);
                const gdn2_mmac::v2f gr = *reinterpret_cast<const gdn2_mmac::v2f*>(sG + (pn * 16) * GS + ch0);
                const gdn2_mmac::v2f gj = *reinterpret_cast<const gdn2_mmac::v2f*>(sG + (pn * 16 + lo) * GS + ch0);
                const gdn2_mmac::v2f kb = *reinterpret_cast<const gdn2_mmac::v2f*>(sKB + irow * GS + ch0);
                const u16x4 q2 = *reinterpret_cast<const u16x4*>(sQ + irow * QS + (ch0 & ~3));
                const u16x4 k2 = *reinterpret_cast<const u16x4*>(sBall + (pn * 16 + lo) * QS + (ch0 & ~3));
                const int sub = ch0 & 2;
                gdn2_mmac::v2f ak, aq, bk;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const float fa = exp2f(gi[i] - gr[i]);
                    const float fb = exp2f(fminf(gr[i] - gj[i], kExpClamp));
                    ak[i] = kb[i] * fa;
                    aq[i] = bf2f(q2[sub + i]) * fa;
                    bk[i] = bf2f(k2[sub + i]) * fb;
                }
                acck = __builtin_amdgcn_mmac_f32_16x16x8tf32(ak, bk, acck);
                accq = __builtin_amdgcn_mmac_f32_16x16x8tf32(aq, bk, accq);
            }
        }
        __syncthreads();
    }

    // ---- M = strict_tril(Akk) into X, scaled Aqk (j <= i) into its staging tile ----
    if (wave < 10) {
        const int i = pm * 16 + lo;
#pragma unroll
        for (int qd = 0; qd < 4; ++qd) {
            const int j = pn * 16 + hq + 4 * qd;
            if (i > j) sXm[i * XS + j] = acck[qd];
            sAqS[i * AS + j] = f2bf(i >= j ? accq[qd] * scale : 0.0f);
        }
    }
    __syncthreads();
    if (tid < 512) {
        const int row = tid >> 3, c8 = (tid & 7) * 8;
        *reinterpret_cast<u16x8*>(aqk + ((row0 + row) * HV + hv) * kC + c8) =
            *reinterpret_cast<const u16x8*>(sAqS + row * AS + c8);
    }

    // ---- diagonal 16x16 blocks of (I + M)^-1: wave 0, lane = blk*16 + column j ----
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
                        *reinterpret_cast<const gdn2_mmac::v2f*>(ap + k0),
                        *reinterpret_cast<const gdn2_mmac::v2f*>(bp + k0), acc);
            }
#pragma unroll
            for (int qd = 0; qd < 4; ++qd) sX[bj * 256 + (hq + 4 * qd) * 16 + lo] = acc[qd];
            asm volatile("" ::: "memory");
            __builtin_amdgcn_wave_barrier();
            v4f acc2 = {0.f, 0.f, 0.f, 0.f};
            const float* ap = sX + bj * 256 + lo * 16 + hq * 2;
            const float* bp = sD + bi * 256 + lo * 16 + hq * 2;
#pragma unroll
            for (int k0 = 0; k0 < 16; k0 += 8)
                acc2 = __builtin_amdgcn_mmac_f32_16x16x8tf32(
                    *reinterpret_cast<const gdn2_mmac::v2f*>(ap + k0),
                    *reinterpret_cast<const gdn2_mmac::v2f*>(bp + k0), acc2);
#pragma unroll
            for (int qd = 0; qd < 4; ++qd) sXm[(bj * 16 + lo) * XS + bi * 16 + hq + 4 * qd] = -acc2[qd];
            asm volatile("" ::: "memory");
            __builtin_amdgcn_wave_barrier();
        }
    }
    __syncthreads();

    // ---- Inv rows of this wave's output tile -> bf16 A fragments, kept in registers ----
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
    // ---- optional Akk_inv output (FLA Akk layout [B,S,HV,64], lower triangle incl. diagonal, upper 0).
    // sXm holds Inv^T: sXm[s][t] = Inv[t][s].  Rows are staged in the (dead) Aqk staging tile and
    // written as 16 B row segments after the barrier below; that tile is not touched again.
    if constexpr (WRITE_AKK) {
        if (tid < 512) {
            const int t = tid & 63, c8 = (tid >> 6) * 8;
            u16x8 pk;
#pragma unroll
            for (int r = 0; r < 8; ++r) pk[r] = (c8 + r <= t) ? f2bf(sXm[(c8 + r) * XS + t]) : static_cast<unsigned short>(0);
            *reinterpret_cast<u16x8*>(sAqS + t * AS + c8) = pk;
        }
    }
    __syncthreads();  // X is dead; its region becomes the staging / vb^T tiles
    if constexpr (WRITE_AKK) {
        if (tid < 512) {
            const int row = tid >> 3, c8 = (tid & 7) * 8;
            *reinterpret_cast<u16x8*>(akk_inv + ((row0 + row) * HV + hv) * kC + c8) =
                *reinterpret_cast<const u16x8*>(sAqS + row * AS + c8);
        }
    }

    // out[t][nn] = sum_s Inv[t][s] T[nn][s]; acc in registers, then staged into sKO
    auto gemm_out = [&](auto nn, const unsigned short* Tp) {
        constexpr int NN = decltype(nn)::value;
        constexpr int R = NN / 64;
        v4f acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] = v4f{0.f, 0.f, 0.f, 0.f};
            const unsigned short* bp = Tp + (((wave >> 2) + 4 * r) * 16 + lo) * BS + hi4;
#pragma unroll
            for (int bs = 0; bs < 4; ++bs)
                if (bs <= atm) acc[r] = mmac(afr[bs], *reinterpret_cast<const v4s*>(bp + bs * 16), acc[r]);
        }
        __syncthreads();  // T may alias the staging tile
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int qd = 0; qd < 4; ++qd)
                sKO[(atm * 16 + lo) * OS + ((wave >> 2) + 4 * r) * 16 + hq + 4 * qd] = f2bf(acc[r][qd]);
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
    // ---- value-side tiles: issued before the w product, consumed after it ----
    gemm_out(std::integral_constant<int, K>{}, sKb2T);
    store_rows(w + (row0 * HV + hv) * K, K);
    __syncthreads();
    // vb^T[nn][s] = bf16(v[s][nn] * wg[s][nn])
    VRegs vr;
    if (tid < 16 * (V / 4)) {
        const uint16_t* vsrc = v + (row0 * HV + hv) * V;
        const float* wsrc = wg + (row0 * HV + hv) * V;
        const std::size_t rs = static_cast<std::size_t>(HV) * V;
        const int sq = (tid & 3) | (((tid >> 5) & 3) << 2);
        const int nq = ((tid >> 2) & 7) | ((tid >> 7) << 3);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const u16x4 vv = *reinterpret_cast<const u16x4*>(vsrc + (sq * 4 + r) * rs + nq * 4);
            const v4f ww = *reinterpret_cast<const v4f*>(wsrc + (sq * 4 + r) * rs + nq * 4);
            u16x4 pp;
#pragma unroll
            for (int j = 0; j < 4; ++j) pp[j] = f2bf(bf2f(vv[j]) * ww[j]);
            vr.p[r] = pp;
        }
    }
    if (tid < 16 * (V / 4)) {
        const int sq = (tid & 3) | (((tid >> 5) & 3) << 2);
        const int nq = ((tid >> 2) & 7) | ((tid >> 7) << 3);
        const int s0 = sq * 4, n0 = nq * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            u16x4 o;
#pragma unroll
            for (int r = 0; r < 4; ++r) o[r] = vr.p[r][j];
            *reinterpret_cast<u16x4*>(sT + (n0 + j) * BS + s0) = o;
        }
    }
    __syncthreads();
    gemm_out(std::integral_constant<int, V>{}, sT);
    store_rows(u + (row0 * HV + hv) * V, V);
}

// ============================================================================ state_h
// Inter-chunk recursion, V-first h.  Differences from the scalar-decay kernel:
// the gate is already folded into kg (so v_new is not rescaled), and the state
// decay is a per-channel vector 2^gc_last[kk] applied to the fp32 registers.
__device__ __forceinline__ void lds_barrier() {
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_waitcnt(0xC07F);
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
}

template <int K, int BV, int NW>
__global__ __launch_bounds__(64 * NW) void gdn2_state_h_kernel(
    const uint16_t* __restrict__ kgp, const uint16_t* __restrict__ w, const uint16_t* __restrict__ u,
    const float* __restrict__ gl, uint16_t* __restrict__ h, uint16_t* __restrict__ v_new,
    int S, int HV, int V) {
    constexpr int THREADS = 64 * NW;
    constexpr int KT16 = K / 16, VT16 = BV / 16;
    constexpr int WG = NW / 4;
    constexpr int VPW = VT16 / WG;
    constexpr int TPW = VT16 * KT16 / NW;
    constexpr int NDT = KT16 / 4;  // distinct k-tiles per wave: tk = wave + 4 j
    constexpr int KTSTR = kC + 4;
    constexpr int HSTR = K + 8;
    constexpr int VTSTR = kC + 4;
    constexpr int USTR = BV + 8;
    constexpr int NK8 = kC * K / 8 / THREADS;
    constexpr int NU8 = kC * BV / 8 / THREADS;
    constexpr int NH8 = K * BV / 8 / THREADS;
    static_assert(NW == 4 && KT16 % 4 == 0 && VT16 % WG == 0 && (VT16 * KT16) % NW == 0, "wave split");
    static_assert(NK8 == 1 || NK8 == 2 || NK8 == 4, "k packing");
    static_assert((kC * K / 8) % THREADS == 0 && (kC * BV / 8) % THREADS == 0 &&
                  (K * BV / 8) % THREADS == 0, "vector staging must divide evenly");

    __shared__ __align__(16) unsigned short sKT[K * KTSTR];
    __shared__ __align__(16) unsigned short sHb[BV * HSTR];
    __shared__ __align__(16) unsigned short sVT[BV * VTSTR];
    __shared__ __align__(16) unsigned short sU[2 * kC * USTR];
    __shared__ __align__(16) float sGl[K];

    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int lo = lane & 15, hq = lane >> 4, hi4 = hq * 4;
    const int v0 = blockIdx.x * BV;
    const int bhv = blockIdx.y;
    const int b = bhv / HV, hv = bhv % HV;
    const int NT = S / kC;
    const int pl = 4 * (lo & 3) + (lo >> 2);
    const int tw = (wave % 4) * 16 + lo;
    const int vg = wave / 4;

    v4f hreg[TPW];
#pragma unroll
    for (int i = 0; i < TPW; ++i) hreg[i] = v4f{0.f, 0.f, 0.f, 0.f};

    const std::size_t rowHVK = static_cast<std::size_t>(HV) * K;
    const std::size_t rowHVV = static_cast<std::size_t>(HV) * V;
    const uint16_t* wB = w + static_cast<std::size_t>(b) * S * rowHVK + static_cast<std::size_t>(hv) * K;
    const uint16_t* uB = u + static_cast<std::size_t>(b) * S * rowHVV + static_cast<std::size_t>(hv) * V + v0;
    uint16_t* vnB = v_new + static_cast<std::size_t>(b) * S * rowHVV + static_cast<std::size_t>(hv) * V + v0;
    const uint16_t* kB = kgp + static_cast<std::size_t>(b) * S * rowHVK + static_cast<std::size_t>(hv) * K;
    const float* glB = gl + (static_cast<std::size_t>(b) * NT * HV + hv) * K;
    const unsigned offW = static_cast<unsigned>(tw * rowHVK + 8 * hq);
    constexpr int KTG = kC / NK8;
    const int kt0 = (tid % KTG) * NK8, kk8 = (tid / KTG) * 8;
    const unsigned offK = static_cast<unsigned>(kt0 * rowHVK + kk8);
    constexpr int UPR = BV / 8;
    const int ut = tid / UPR, uv8 = (tid % UPR) * 8;
    constexpr int UTSTEP = THREADS / UPR;

    u16x8 pa[KT16 / 2];
    u16x8 pk[NK8];
    u16x8 pu[NU8];
    float pg = 0.0f;
    auto prefetch = [&](int n) {
        const int t0 = n * kC;
        const uint16_t* wc = wB + t0 * rowHVK;
#pragma unroll
        for (int j = 0; j < KT16 / 2; ++j) pa[j] = *reinterpret_cast<const u16x8*>(wc + offW + 32 * j);
        const uint16_t* kc = kB + t0 * rowHVK;
#pragma unroll
        for (int r = 0; r < NK8; ++r) pk[r] = *reinterpret_cast<const u16x8*>(kc + offK + static_cast<unsigned>(r * rowHVK));
        const uint16_t* uc = uB + t0 * rowHVV;
#pragma unroll
        for (int r = 0; r < NU8; ++r)
            pu[r] = *reinterpret_cast<const u16x8*>(uc + static_cast<unsigned>((ut + r * UTSTEP) * rowHVV) + uv8);
        if (tid < K) pg = glB[static_cast<std::size_t>(n) * HV * K + tid];
    };
    prefetch(0);

    for (int n = 0; n < NT; ++n) {
        const int t0 = n * kC;
        unsigned short* sUc = sU + (n & 1) * (kC * USTR);

#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            const int t = wave + i * NW, tv = t / KT16, tk = t % KT16;
            *reinterpret_cast<u16x4*>(sHb + (tv * 16 + lo) * HSTR + tk * 16 + 4 * hq) =
                u16x4{f2bf(hreg[i][0]), f2bf(hreg[i][1]), f2bf(hreg[i][2]), f2bf(hreg[i][3])};
        }
#pragma unroll
        for (int r = 0; r < NU8; ++r)
            *reinterpret_cast<u16x8*>(sUc + (ut + r * UTSTEP) * USTR + uv8) = pu[r];
        if (tid < K) sGl[tid] = pg;
        u16x8 ca[KT16 / 2];
#pragma unroll
        for (int j = 0; j < KT16 / 2; ++j) ca[j] = pa[j];
        lds_barrier();
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            unsigned short* kp = sKT + (kk8 + q) * KTSTR + kt0;
            if constexpr (NK8 == 4) {
                *reinterpret_cast<u16x4*>(kp) = u16x4{pk[0][q], pk[1][q], pk[2][q], pk[3][q]};
            } else if constexpr (NK8 == 2) {
                *reinterpret_cast<unsigned*>(kp) = static_cast<unsigned>(pk[0][q]) | (static_cast<unsigned>(pk[1][q]) << 16);
            } else {
#pragma unroll
                for (int r = 0; r < NK8; ++r) kp[r] = pk[r][q];
            }
        }
        if (n + 1 < NT) prefetch(n + 1);
        // per-channel state decay of this lane's k-tiles, read before the second barrier
        v4f dv[NDT];
#pragma unroll
        for (int j = 0; j < NDT; ++j) {
            const v4f e = *reinterpret_cast<const v4f*>(sGl + (wave + 4 * j) * 16 + 4 * hq);
            dv[j] = v4f{exp2f(e[0]), exp2f(e[1]), exp2f(e[2]), exp2f(e[3])};
        }

        // ---- GEMM1: v_new[t][v] = u - sum_kk w[t][kk] h^T[v][kk] ----
        {
#pragma unroll
            for (int j = 0; j < VPW; ++j) {
                const int tn2 = vg + j * WG;
                const unsigned short* bp = sHb + (tn2 * 16 + pl) * HSTR + 8 * hq;
                v4f acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int jj = 0; jj < KT16 / 2; ++jj) {
                    const u16x8 bb = *reinterpret_cast<const u16x8*>(bp + 32 * jj);
                    acc = __builtin_amdgcn_mmac_f32_16x16x16bf16(
                        __builtin_bit_cast(v4s, __builtin_shufflevector(ca[jj], ca[jj], 0, 1, 2, 3)),
                        __builtin_bit_cast(v4s, __builtin_shufflevector(bb, bb, 0, 1, 2, 3)), acc);
                    acc = __builtin_amdgcn_mmac_f32_16x16x16bf16(
                        __builtin_bit_cast(v4s, __builtin_shufflevector(ca[jj], ca[jj], 4, 5, 6, 7)),
                        __builtin_bit_cast(v4s, __builtin_shufflevector(bb, bb, 4, 5, 6, 7)), acc);
                }
                unsigned short* up = sUc + tw * USTR + tn2 * 16 + 4 * hq;
                const u16x4 u4 = *reinterpret_cast<const u16x4*>(up);
                u16x4 o4;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const unsigned short xb = f2bf(bf2f(u4[r]) - acc[r]);
                    o4[r] = xb;
                    sVT[(tn2 * 16 + 4 * hq + r) * VTSTR + tw] = xb;  // v_new as stored (bf16): the gate lives in kg
                }
                *reinterpret_cast<u16x4*>(up) = o4;
            }
        }
        // ---- h (chunk-start state), V-first rows out of sHb ----
        {
            uint16_t* hc = h + ((static_cast<std::size_t>(b) * NT + n) * HV + hv) * V * K +
                           static_cast<std::size_t>(v0) * K;
#pragma unroll
            for (int r = 0; r < NH8; ++r) {
                const int e = tid + r * THREADS, vr = e / (K / 8), c = (e % (K / 8)) * 8;
                *reinterpret_cast<u16x8*>(hc + vr * K + c) =
                    *reinterpret_cast<const u16x8*>(sHb + vr * HSTR + c);
            }
        }

        lds_barrier();

        {
            uint16_t* vc = vnB + t0 * rowHVV;
#pragma unroll
            for (int r = 0; r < NU8; ++r) {
                const int t = ut + r * UTSTEP;
                *reinterpret_cast<u16x8*>(vc + static_cast<unsigned>(t * rowHVV) + uv8) =
                    *reinterpret_cast<const u16x8*>(sUc + t * USTR + uv8);
            }
        }

        // ---- GEMM2: h^T[v][kk] = 2^gc_last[kk] h^T + sum_t vT[v][t] kgT[kk][t] ----
        {
#pragma unroll
            for (int i = 0; i < TPW; ++i) {
                const int t = wave + i * NW, tv = t / KT16, tk = t % KT16;
                const unsigned short* ap = sVT + (tv * 16 + lo) * VTSTR + hi4;
                const unsigned short* bp = sKT + (tk * 16 + pl) * KTSTR + hi4;
                v4f acc = hreg[i] * dv[(i % NDT)];
#pragma unroll
                for (int k0 = 0; k0 < kC; k0 += 16)
                    acc = __builtin_amdgcn_mmac_f32_16x16x16bf16(
                        *reinterpret_cast<const v4s*>(ap + k0),
                        *reinterpret_cast<const v4s*>(bp + k0), acc);
                hreg[i] = acc;
            }
        }
    }
}

// ============================================================================ chunk_o
// o = scale qg h + Aqk v_new for one 64-row chunk.  Wave w owns rows 16w..16w+15.
// Aqk (already masked and scaled) is the A operand of Aqk.v_new straight from
// global; qg is the A operand of qg.h.  See the scalar-decay kernel for the
// block-order / staging rationale: hv fastest, whole V per block, h via LDS.
constexpr int kCoVB = 32;
constexpr int kCoNP = 4;
constexpr int kCoOG = 2;

template <int K, int VB, int NP, int OG>
__global__ __launch_bounds__(kThreads) void gdn2_chunk_o_kernel(
    const uint16_t* __restrict__ qgp, const uint16_t* __restrict__ aqk,
    const uint16_t* __restrict__ vn, const uint16_t* __restrict__ h,
    uint16_t* __restrict__ o, int S, int HV, int V, float scale) {
    constexpr int KT16 = K / 16, VT16 = VB / 16, BV = VB * NP;
    constexpr int VS = kC + 4, HS = K + 4, OS = OG * VB + 4;
    constexpr int HPER = K / 8, NH = (VB * HPER + kThreads - 1) / kThreads;
    constexpr int VPER = VB / 8, NV = (kC * VPER + kThreads - 1) / kThreads;
    constexpr int OPER = OG * VB / 8;
    constexpr int HSZ = (VB * HS > kC * OS) ? VB * HS : kC * OS;
    static_assert(NP % OG == 0, "output groups tile the passes");
    constexpr int BUF = HSZ + VB * VS;
    __shared__ __align__(16) unsigned short sBuf[BUF];
    unsigned short* const sH = sBuf;
    unsigned short* const sVT = sBuf + HSZ;
    unsigned short* const sO = sH;

    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int lo = lane & 15, hq = lane >> 4, hi4 = hq * 4;
    const int NT = S / kC, nvb = V / BV;
    int id = blockIdx.x;
    const int hv = id % HV;
    id /= HV;
    const int v0 = (id % nvb) * BV;
    id /= nvb;
    const int n = id % NT, b = id / NT;
    const std::size_t row0 = static_cast<std::size_t>(b) * S + n * kC;
    const int tm = wave, tr = tm * 16 + lo;
    const int fl = 4 * (lo & 3) + (lo >> 2);
    const uint16_t* qB = qgp + (row0 * HV + hv) * K + hi4;
    const uint16_t* aB = aqk + (row0 * HV + hv) * kC + hi4;
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
    v4s qa[KT16];
#pragma unroll
    for (int tk = 0; tk < KT16; ++tk)
        qa[tk] = *reinterpret_cast<const v4s*>(qB + static_cast<std::size_t>(tr) * HV * K + 16 * tk);
    v4s af[4];
#pragma unroll
    for (int ts = 0; ts < 4; ++ts) {
        af[ts] = v4s{0, 0, 0, 0};
        if (ts <= tm) af[ts] = *reinterpret_cast<const v4s*>(aB + static_cast<std::size_t>(tr) * HV * kC + 16 * ts);
    }

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
#pragma unroll
                for (int r = 0; r < 8; ++r) sVT[(c + r) * VS + s] = xv[j][i][r];
            }
        }
    };
    store_pass(0);
    __syncthreads();

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
            og[j % OG][tv] = acc * scale;
        }
#pragma unroll
        for (int tv = 0; tv < VT16; ++tv) {
            const unsigned short* bp = sVT + (tv * 16 + fl) * VS + hi4;
#pragma unroll
            for (int ts = 0; ts < 4; ++ts)
                if (ts <= tm)
                    og[j % OG][tv] = mmac(af[ts], *reinterpret_cast<const v4s*>(bp + ts * 16), og[j % OG][tv]);
        }
        __syncthreads();
        if (j % OG == OG - 1) {
#pragma unroll
            for (int gq = 0; gq < OG; ++gq)
#pragma unroll
                for (int tv = 0; tv < VT16; ++tv)
                    *reinterpret_cast<u16x4*>(sO + tr * OS + gq * VB + tv * 16 + hi4) =
                        u16x4{f2bf(og[gq][tv][0]), f2bf(og[gq][tv][1]), f2bf(og[gq][tv][2]), f2bf(og[gq][tv][3])};
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
            if (j + 1 < NP) __syncthreads();
        }
    }
}

constexpr int kMinS = 512;

std::size_t align256(std::size_t x) { return (x + 255) / 256 * 256; }

struct Ws {
    std::size_t w, u, kg, qg, aqk, h, vn, gl, total;
};

Ws layout(int B, int S, int HV, int K, int V) {
    const std::size_t rows = static_cast<std::size_t>(B) * S * HV;
    const std::size_t NT = S / kC;
    Ws l{};
    std::size_t off = 0;
    auto take = [&](std::size_t bytes) { std::size_t o = off; off += align256(bytes); return o; };
    l.w = take(rows * K * 2);
    l.u = take(rows * V * 2);
    l.kg = take(rows * K * 2);
    l.qg = take(rows * K * 2);
    l.aqk = take(rows * kC * 2);
    l.h = take(static_cast<std::size_t>(B) * NT * HV * V * K * 2);
    l.vn = take(rows * V * 2);
    l.gl = take(static_cast<std::size_t>(B) * NT * HV * K * 4);
    l.total = off;
    return l;
}

bool kv_ok(int K, int V) { return (K == 64 || K == 128) && (V == 64 || V == 128); }

}  // namespace

extern "C" int flash_train_gdn2_chunk_shape_supported(int S, int H, int HV, int K, int V) {
    return S >= kMinS && S % kC == 0 && H > 0 && HV > 0 && HV % H == 0 && kv_ok(K, V) ? 1 : 0;
}

extern "C" size_t flash_train_gdn2_chunk_workspace_bytes(int B, int S, int HV, int K, int V) {
    if (B <= 0 || S <= 0 || S % kC || HV <= 0 || !kv_ok(K, V)) return 0;
    return layout(B, S, HV, K, V).total;
}

extern "C" int flash_train_gdn2_intra_bf16(const FlashTrainGDN2IntraArgs* a, void* stream) {
    if (!a || !a->q || !a->k || !a->v || !a->g || !a->b || !a->wg || !a->w || !a->u || !a->kg ||
        !a->qg || !a->aqk || !a->gclast)
        return 1;
    if (a->B <= 0 || a->S % kC || a->S <= 0 || a->HV % a->H || !kv_ok(a->K, a->V)) return 2;
    hipStream_t st = static_cast<hipStream_t>(stream);
    const dim3 grid(static_cast<unsigned>((a->S / kC) * a->B * a->HV));
#define L(KK, VV)                                                                                 \
    if (a->K == KK && a->V == VV) {                                                               \
        if (a->akk_inv)                                                                           \
            hipLaunchKernelGGL((gdn2_intra_kernel<KK, VV, true>), grid, dim3(kIT), 0, st, a->q,   \
                               a->k, a->v, a->g, a->b, a->wg, a->w, a->u, a->kg, a->qg, a->aqk,   \
                               a->gclast, a->gc, a->akk_inv, a->S, a->H, a->HV, a->scale);        \
        else                                                                                      \
            hipLaunchKernelGGL((gdn2_intra_kernel<KK, VV, false>), grid, dim3(kIT), 0, st, a->q,  \
                               a->k, a->v, a->g, a->b, a->wg, a->w, a->u, a->kg, a->qg, a->aqk,   \
                               a->gclast, a->gc, nullptr, a->S, a->H, a->HV, a->scale);           \
        return hipGetLastError() == hipSuccess ? 0 : 5;                                           \
    }
    L(128, 128) L(64, 128) L(128, 64) L(64, 64)
#undef L
    return 2;
}

extern "C" int flash_train_gdn2_state_h_bf16(const FlashTrainGDN2StateHArgs* a, void* stream) {
    if (!a || !a->kg || !a->w || !a->u || !a->gclast || !a->h || !a->v_new) return 1;
    if (a->B <= 0 || a->S <= 0 || a->S % kC || a->V % 64 || !kv_ok(a->K, a->V)) return 2;
    hipStream_t st = static_cast<hipStream_t>(stream);
    const dim3 grid(a->V / 64, a->B * a->HV);
#define L(KK)                                                                                 \
    if (a->K == KK) {                                                                         \
        hipLaunchKernelGGL((gdn2_state_h_kernel<KK, 64, 4>), grid, dim3(256), 0, st, a->kg,   \
                           a->w, a->u, a->gclast, a->h, a->v_new, a->S, a->HV, a->V);         \
        return hipGetLastError() == hipSuccess ? 0 : 5;                                       \
    }
    L(128) L(64)
#undef L
    return 2;
}

extern "C" int flash_train_gdn2_chunk_o_bf16(const FlashTrainGDN2ChunkOArgs* a, void* stream) {
    if (!a || !a->qg || !a->aqk || !a->v_new || !a->h || !a->o) return 1;
    if (a->B <= 0 || a->S <= 0 || a->S % kC || a->V % (kCoVB * 2) || !kv_ok(a->K, a->V)) return 2;
    hipStream_t st = static_cast<hipStream_t>(stream);
    const bool wide = a->V % (kCoVB * kCoNP) == 0;
    const int bv = wide ? kCoVB * kCoNP : kCoVB * 2;
    const dim3 grid(static_cast<unsigned>((a->V / bv) * (a->S / kC) * a->B * a->HV));
#define CO_LAUNCH(KK, NPP)                                                                          \
    hipLaunchKernelGGL((gdn2_chunk_o_kernel<KK, kCoVB, NPP, kCoOG>), grid, dim3(kThreads), 0, st, \
                       a->qg, a->aqk, a->v_new, a->h, a->o, a->S, a->HV, a->V, a->scale)
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

extern "C" int flash_train_gdn2_chunk_forward_bf16(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* g, const float* b,
    const float* wg, uint16_t* out, int B, int S, int H, int HV, int K, int V, float scale,
    void* workspace, size_t workspace_bytes, void* stream) {
    if (!q || !k || !v || !g || !b || !wg || !out || !workspace) return 1;
    if (B <= 0 || !flash_train_gdn2_chunk_shape_supported(S, H, HV, K, V)) return 2;
    const Ws l = layout(B, S, HV, K, V);
    if (workspace_bytes < l.total) return 4;
    char* ws = static_cast<char*>(workspace);
    FlashTrainGDN2IntraArgs ia{};
    ia.q = q; ia.k = k; ia.v = v; ia.g = g; ia.b = b; ia.wg = wg;
    ia.w = reinterpret_cast<uint16_t*>(ws + l.w);
    ia.u = reinterpret_cast<uint16_t*>(ws + l.u);
    ia.kg = reinterpret_cast<uint16_t*>(ws + l.kg);
    ia.qg = reinterpret_cast<uint16_t*>(ws + l.qg);
    ia.aqk = reinterpret_cast<uint16_t*>(ws + l.aqk);
    ia.gclast = reinterpret_cast<float*>(ws + l.gl);
    ia.gc = nullptr;
    ia.B = B; ia.S = S; ia.H = H; ia.HV = HV; ia.K = K; ia.V = V; ia.scale = scale;
    int rc = flash_train_gdn2_intra_bf16(&ia, stream);
    if (rc) return rc;
    FlashTrainGDN2StateHArgs sa{};
    sa.kg = ia.kg; sa.w = ia.w; sa.u = ia.u; sa.gclast = ia.gclast;
    sa.h = reinterpret_cast<uint16_t*>(ws + l.h);
    sa.v_new = reinterpret_cast<uint16_t*>(ws + l.vn);
    sa.B = B; sa.S = S; sa.H = H; sa.HV = HV; sa.K = K; sa.V = V;
    rc = flash_train_gdn2_state_h_bf16(&sa, stream);
    if (rc) return rc;
    FlashTrainGDN2ChunkOArgs oa{};
    oa.qg = ia.qg; oa.aqk = ia.aqk; oa.v_new = sa.v_new; oa.h = sa.h; oa.o = out;
    oa.B = B; oa.S = S; oa.H = H; oa.HV = HV; oa.K = K; oa.V = V; oa.scale = scale;
    return flash_train_gdn2_chunk_o_bf16(&oa, stream);
}
