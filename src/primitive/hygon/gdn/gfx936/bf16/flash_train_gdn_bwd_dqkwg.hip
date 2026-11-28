// B3 dqkwg of the large-shape GDN backward (chunk = 64, FLA layouts), phase P2.
//
// Per (chunk n, head hv), with h / dh the state at the START / state gradient at the END of
// the chunk (V-first [V][K]), g = gc (log2 units, chunk-local cumsum), vn = v_new:
//   dq_raw = do h        dk_raw = vn dh        dw = - dv2 h        ds_raw = do vn^T      (contract V)
//   dg_last = 2^g_63 sum(h * dh) + sum(dk_g * k),   dk_g = dk_raw 2^(g_63 - g_t)
//   dq  = scale 2^g_t dq_raw + ds k            ds[t][s] = scale 2^(g_t-g_s) ds_raw[t][s]   (t >= s)
//   dk  = dk_g + ds^T q                        (ds rounded to bf16 for both products, as FLA)
//   dg_t = dq_t . q_t - dk_t . k_t   (+ dg_last on the last row)                      (contract K)
// dq / dk / dw are written per v-head ([B,S,HV,K], FLA writes the same before its GQA group
// sum).  dg is fp32 [K/KKH][B,S,HV]: one plane per KKH-column K slice (FLA also emits NK partial
// planes and sums them in a separate kernel); the consumer sums the planes before the reverse
// cumsum that follows wy_bwd.
//
// One 4-wave block = (chunk, hv, KKH-column K slice), k slice fastest in the 1-D grid (the slices of
// a (chunk, hv) read the same do / dv2 / vn rows), then hv.  Wave w owns rows t = 16w..16w+15
// of the slice.  ds depends on V only, so every slice recomputes it; in exchange the dq / dk / dw
// accumulators are KKH/16 tiles each, small enough for 2-3 blocks per CU.
//
// V is walked in passes of KVB.  A pass is written to LDS before it is used: h^T / dh^T as
// [k][v] (contraction index contiguous, stored transposed), do / dv2 / vn natural [t][v]; the
// next pass's global loads (16-byte cooperative row loads) are issued right after.  The B-row
// order of the K tiles is chosen so a lane ends with 8 consecutive k columns over a tile pair.
// The final phase contracts over t: k^T (permuted s order, so the gated ds stays in registers as
// the A operand) and q^T / ds^T staged in LDS.  Outputs go through an LDS tile and leave as
// full-slice rows.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../common/flash_train_gdn_bwd.h"
#include "../common/gdn_chunk_mmac.h"

#ifndef KKH
#define KKH 64   // k columns per block
#endif
#ifndef DBUF
#define DBUF 0   // 1 = double-buffered pass tiles (measured slower: 207 VGPR, 11.6 ms)
#endif
#ifndef KVB
#define KVB 32   // V columns per pass
#endif
#ifndef WPE
#define WPE 0    // amdgpu_waves_per_eu minimum (0 = compiler's choice)
#endif

namespace {

using gdn_mmac::bf2f;
using gdn_mmac::f2bf;
using gdn_mmac::u16x4;
using gdn_mmac::v4f;
using gdn_mmac::v4s;
typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));

constexpr int kC = 64;
constexpr int kThreads = 256;
constexpr int kVB = KVB;
constexpr int kKH = KKH;
constexpr int kKT = kKH / 16;   // 16-column tiles per block
constexpr int kCPS = kKH / 8;   // 16-byte chunks per slice row
static_assert(kKH == 32 || kKH == 64, "slice width");

__device__ __forceinline__ v4f mmac(v4s a, v4s b, v4f c) {
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
}
__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
    for (int m = 1; m < 64; m <<= 1) x += __shfl_xor(x, m, 64);
    return x;
}

template <int K, int V>
__global__
#if WPE > 0
    __attribute__((amdgpu_waves_per_eu(WPE)))
#endif
    __launch_bounds__(kThreads) void gdn_bwd_dqkwg_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k, const uint16_t* __restrict__ vn,
    const uint16_t* __restrict__ dout, const uint16_t* __restrict__ dv2, const uint16_t* __restrict__ h,
    const uint16_t* __restrict__ dh, const float* __restrict__ gc, uint16_t* __restrict__ dq,
    uint16_t* __restrict__ dk, uint16_t* __restrict__ dw, float* __restrict__ dg, int S, int H, int HV,
    int B, float scale) {
    constexpr int KS = K / kKH, NP = V / kVB;
    constexpr int HT = kVB + 4;  // sHT / sDT / sDo / sDv / sVn row stride (bf16): [k][v] and [t][v]
    constexpr int KTS = kC + 4;  // sKT / sQT / sDSs row stride
    constexpr int OS = kKH + 8;  // output tile row stride (16B aligned)
    constexpr int NQ = kC * kCPS / kThreads;  // q / k items per thread
    constexpr int AIT = kC * (kVB / 8);                             // do / dv2 / vn items per pass
    constexpr int NA = (AIT + kThreads - 1) / kThreads;
    constexpr int HIT = kVB * kCPS;                                 // h / dh items per pass
    constexpr int NHI = (HIT + kThreads - 1) / kThreads;
    static_assert(kC * kCPS % kThreads == 0, "q/k tile mapping");
    constexpr int PASS = 2 * kKH * HT + 3 * kC * HT;         // one pass of tiles (elements)
    constexpr int RING = (DBUF ? 2 : 1) * PASS;
    constexpr int R = kC * OS > kC * KTS ? kC * OS : kC * KTS;  // one phase-2 region
    constexpr int BUF = RING > 3 * R ? RING : 3 * R;
    __shared__ __align__(16) unsigned short sBuf[BUF];
    __shared__ float sGc[kC];
    __shared__ float sRed[4];
    // phase 2 (aliases the pass tiles): R0 = k^T then dq tile; R1 = ds^T then dk tile; R2 = dw tile then q^T
    unsigned short* const sKT = sBuf;
    unsigned short* const sOq = sBuf;
    unsigned short* const sDSs = sBuf + R;
    unsigned short* const sOk = sBuf + R;
    unsigned short* const sOw = sBuf + 2 * R;
    unsigned short* const sQT = sOw;

    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int lo = lane & 15, hq = lane >> 4;
    const int tr = wave * 16 + lo;
    const int NT = S / kC;
    int id = blockIdx.x;
    const int ks = id % KS;
    id /= KS;
    const int hv = id % HV;
    id /= HV;
    const int n = id % NT, b = id / NT;
    const int hk = hv / (HV / H);
    const std::size_t row0 = static_cast<std::size_t>(b) * S + n * kC;
    const std::size_t qkRow = static_cast<std::size_t>(H) * K;
    const std::size_t dRow = static_cast<std::size_t>(HV) * V;

    const std::size_t stOff = (static_cast<std::size_t>(b) * NT + n) * HV + hv;
    const uint16_t* const hB = h + stOff * V * K + ks * kKH;
    const uint16_t* const dhB = dh + stOff * V * K + ks * kKH;
    const std::size_t aBase = row0 * dRow + static_cast<std::size_t>(hv) * V;

    u16x8 xdo[NA], xdv[NA], xvn[NA], xh[NHI], xdh[NHI], xk[NQ], xq[NQ];
    auto load_pass = [&](int sp) {
#pragma unroll
        for (int i = 0; i < NA; ++i) {
            const int e = tid + i * kThreads;
            if (AIT % kThreads == 0 || e < AIT) {
                const std::size_t o = aBase + static_cast<std::size_t>(e / (kVB / 8)) * dRow + sp * kVB + (e % (kVB / 8)) * 8;
                xdo[i] = *reinterpret_cast<const u16x8*>(dout + o);
                xdv[i] = *reinterpret_cast<const u16x8*>(dv2 + o);
                xvn[i] = *reinterpret_cast<const u16x8*>(vn + o);
            }
        }
#pragma unroll
        for (int i = 0; i < NHI; ++i) {
            const int e = tid + i * kThreads;
            if (HIT % kThreads == 0 || e < HIT) {
                const std::size_t o = static_cast<std::size_t>(sp * kVB + e / kCPS) * K + (e % kCPS) * 8;
                xh[i] = *reinterpret_cast<const u16x8*>(hB + o);
                xdh[i] = *reinterpret_cast<const u16x8*>(dhB + o);
            }
        }
    };
    load_pass(0);
    const float gcv = tid < kC ? gc[(row0 + tid) * HV + hv] : 0.f;
    if (tid < kC) sGc[tid] = gcv;

    float s1 = 0.f;  // partial sum(h * dh) over the slice
    v4f dsacc[4], dqacc[kKT], dkacc[kKT], dwacc[kKT];
#pragma unroll
    for (int i = 0; i < 4; ++i) dsacc[i] = v4f{0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < kKT; ++i) {
        dqacc[i] = v4f{0.f, 0.f, 0.f, 0.f};
        dkacc[i] = v4f{0.f, 0.f, 0.f, 0.f};
        dwacc[i] = v4f{0.f, 0.f, 0.f, 0.f};
    }

    auto stage_pass = [&](int bufi) {
        unsigned short* const sHT = sBuf + bufi * PASS;
        unsigned short* const sDT = sHT + kKH * HT;
        unsigned short* const sDo = sDT + kKH * HT;
        unsigned short* const sDv = sDo + kC * HT;
        unsigned short* const sVn = sDv + kC * HT;
#pragma unroll
        for (int i = 0; i < NHI; ++i) {
            const int e = tid + i * kThreads;
            if (HIT % kThreads == 0 || e < HIT) {
                const int vr = e / kCPS, cm = e % kCPS, p = cm >> 2, mm = cm & 3;
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    const int row = (2 * p + (jj >> 2)) * 16 + mm + 4 * (jj & 3);
                    sHT[row * HT + vr] = xh[i][jj];
                    sDT[row * HT + vr] = xdh[i][jj];
                    s1 += bf2f(xh[i][jj]) * bf2f(xdh[i][jj]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < NA; ++i) {
            const int e = tid + i * kThreads;
            if (AIT % kThreads == 0 || e < AIT) {
                const int ar = e / (kVB / 8), ac = (e % (kVB / 8)) * 8;
                unsigned short* const d0 = sDo + ar * HT + ac;
                unsigned short* const d1 = sDv + ar * HT + ac;
                unsigned short* const d2 = sVn + ar * HT + ac;
                *reinterpret_cast<u16x4*>(d0) = u16x4{xdo[i][0], xdo[i][1], xdo[i][2], xdo[i][3]};
                *reinterpret_cast<u16x4*>(d0 + 4) = u16x4{xdo[i][4], xdo[i][5], xdo[i][6], xdo[i][7]};
                *reinterpret_cast<u16x4*>(d1) = u16x4{xdv[i][0], xdv[i][1], xdv[i][2], xdv[i][3]};
                *reinterpret_cast<u16x4*>(d1 + 4) = u16x4{xdv[i][4], xdv[i][5], xdv[i][6], xdv[i][7]};
                *reinterpret_cast<u16x4*>(d2) = u16x4{xvn[i][0], xvn[i][1], xvn[i][2], xvn[i][3]};
                *reinterpret_cast<u16x4*>(d2 + 4) = u16x4{xvn[i][4], xvn[i][5], xvn[i][6], xvn[i][7]};
            }
        }
    };
    auto compute_pass = [&](int bufi) {
        const unsigned short* const sHT = sBuf + bufi * PASS;
        const unsigned short* const sDT = sHT + kKH * HT;
        const unsigned short* const sDo = sDT + kKH * HT;
        const unsigned short* const sDv = sDo + kC * HT;
        const unsigned short* const sVn = sDv + kC * HT;
#pragma unroll
        for (int r = 0; r < kVB / 32; ++r) {
            const int vo = r * 32 + hq * 8;  // per lane 8 consecutive v = two k-steps
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                const v4s ado = *reinterpret_cast<const v4s*>(sDo + tr * HT + vo + st * 4);
                const v4s adv = *reinterpret_cast<const v4s*>(sDv + tr * HT + vo + st * 4);
                const v4s avn = *reinterpret_cast<const v4s*>(sVn + tr * HT + vo + st * 4);
#pragma unroll
                for (int ts = 0; ts < 4; ++ts)
                    if (ts <= wave)
                        dsacc[ts] = mmac(ado, *reinterpret_cast<const v4s*>(sVn + (ts * 16 + lo) * HT + vo + st * 4), dsacc[ts]);
#pragma unroll
                for (int tk = 0; tk < kKT; ++tk) {
                    const int rb = (tk * 16 + lo) * HT + vo + st * 4;
                    const v4s bh = *reinterpret_cast<const v4s*>(sHT + rb);
                    dqacc[tk] = mmac(ado, bh, dqacc[tk]);
                    dwacc[tk] = mmac(adv, bh, dwacc[tk]);
                    dkacc[tk] = mmac(avn, *reinterpret_cast<const v4s*>(sDT + rb), dkacc[tk]);
                }
            }
        }
    };
    auto issue_qk = [&]() {  // q / k rows of the slice: requested before the last pass, used in the epilogue
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            const int e = tid + i * kThreads;
            const std::size_t o = (row0 + e / kCPS) * qkRow + static_cast<std::size_t>(hk) * K + ks * kKH + (e % kCPS) * 8;
            xk[i] = *reinterpret_cast<const u16x8*>(k + o);
            xq[i] = *reinterpret_cast<const u16x8*>(q + o);
        }
    };
#if DBUF
    stage_pass(0);
    if (NP > 1) load_pass(1);
    __syncthreads();
#pragma unroll
    for (int sp = 0; sp < NP; ++sp) {
        if (sp == NP - 1) issue_qk();
        compute_pass(sp & 1);
        if (sp + 1 < NP) stage_pass((sp + 1) & 1);
        if (sp + 2 < NP) load_pass(sp + 2);
        __syncthreads();
    }
#else
#pragma unroll 1
    for (int sp = 0; sp < NP - 1; ++sp) {
        if (sp > 0) __syncthreads();  // the previous pass is no longer read
        stage_pass(0);
        load_pass(sp + 1);
        __syncthreads();
        compute_pass(0);
    }
    if (NP > 1) __syncthreads();
    stage_pass(0);
    issue_qk();
    __syncthreads();
    compute_pass(0);
    __syncthreads();  // B1: the pass tiles are dead
#endif

    // output tile [64][OS]: this wave's 16 rows x kKH columns, then full-slice rows out
    auto put = [&](unsigned short* tile, const v4f* acc, float mul) {
#pragma unroll
        for (int p = 0; p < kKT / 2; ++p) {
            const v4f a = acc[2 * p] * mul, c = acc[2 * p + 1] * mul;
            *reinterpret_cast<u16x8*>(tile + tr * OS + (hq + 4 * p) * 8) =
                u16x8{f2bf(a[0]), f2bf(a[1]), f2bf(a[2]), f2bf(a[3]), f2bf(c[0]), f2bf(c[1]), f2bf(c[2]), f2bf(c[3])};
        }
    };
    auto get = [&](const unsigned short* tile, uint16_t* dst) {
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            const int e = tid + i * kThreads;
            const int r = e / kCPS, cc = (e % kCPS) * 8;
            *reinterpret_cast<u16x8*>(dst + ((row0 + r) * HV + hv) * K + ks * kKH + cc) =
                *reinterpret_cast<const u16x8*>(tile + r * OS + cc);
        }
    };
    // k^T / q^T [k][t] from the cooperatively loaded rows; permuted = s order of the gated-ds A operand
    auto stage_t = [&](unsigned short* dstT, const u16x8* x, bool permuted) {
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            const int e = tid + i * kThreads;
            const int s = e / kCPS, cm = e % kCPS, p = cm >> 2, mm = cm & 3;
            const int col = permuted ? ((s & ~15) | ((s & 3) << 2) | ((s >> 2) & 3)) : s;
#pragma unroll
            for (int jj = 0; jj < 8; ++jj)
                dstT[((2 * p + (jj >> 2)) * 16 + mm + 4 * (jj & 3)) * KTS + col] = x[i][jj];
        }
    };

    put(sOw, dwacc, -1.0f);
    stage_t(sKT, xk, true);
    const float gt = sGc[tr], gl = sGc[kC - 1];
    const float et = exp2f(gt) * scale, ek = exp2f(gl - gt);
    v4s af[4];
#pragma unroll
    for (int ts = 0; ts < 4; ++ts) {
        af[ts] = v4s{0, 0, 0, 0};
        if (ts <= wave) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int sc = ts * 16 + hq + 4 * i;
                const float x = (sc <= tr) ? dsacc[ts][i] * exp2f(gt - sGc[sc]) * scale : 0.0f;
                const unsigned short xb = f2bf(x);
                af[ts][i] = static_cast<short>(xb);
                sDSs[sc * KTS + tr] = xb;
            }
        }
    }
    __syncthreads();  // B2
    get(sOw, dw);

    // dg_last partial: sum(h dh) 2^g_63 + sum(dk_g k), k read back at the accumulator positions
    const int pst = (tr & ~15) | ((tr & 3) << 2) | ((tr >> 2) & 3);
    float dgl = s1 * exp2f(gl);
#pragma unroll
    for (int tk = 0; tk < kKT; ++tk) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            dkacc[tk][i] *= ek;
            dgl += dkacc[tk][i] * bf2f(sKT[(tk * 16 + hq + 4 * i) * KTS + pst]);
        }
        dqacc[tk] *= et;
    }
    dgl = wave_sum(dgl);
    if (lane == 0) sRed[wave] = dgl;

    // ---- dq += ds k   (A = gated ds from registers, B = k^T in the permuted s order) ----
#pragma unroll
    for (int ts = 0; ts < 4; ++ts)
        if (ts <= wave)
#pragma unroll
            for (int tk = 0; tk < kKT; ++tk)
                dqacc[tk] = mmac(af[ts], *reinterpret_cast<const v4s*>(sKT + (tk * 16 + lo) * KTS + ts * 16 + hq * 4), dqacc[tk]);
    __syncthreads();  // B3: dw tile reads are done

    stage_t(sQT, xq, false);
    __syncthreads();  // B4

    // ---- dk += ds^T q ----
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
        if (tt >= wave) {
            const v4s a = *reinterpret_cast<const v4s*>(sDSs + (wave * 16 + lo) * KTS + tt * 16 + hq * 4);
#pragma unroll
            for (int tk = 0; tk < kKT; ++tk)
                dkacc[tk] = mmac(a, *reinterpret_cast<const v4s*>(sQT + (tk * 16 + lo) * KTS + tt * 16 + hq * 4), dkacc[tk]);
        }
    }

    // ---- dg_t = dq_t.q_t - dk_t.k_t ----
    float dgr = 0.f;
#pragma unroll
    for (int tk = 0; tk < kKT; ++tk)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int rr = (tk * 16 + hq + 4 * i) * KTS;
            dgr += dqacc[tk][i] * bf2f(sQT[rr + tr]) - dkacc[tk][i] * bf2f(sKT[rr + pst]);
        }
    dgr += __shfl_xor(dgr, 16, 64);
    dgr += __shfl_xor(dgr, 32, 64);
    if (hq == 0) {
        if (tr == kC - 1) dgr += (sRed[0] + sRed[1]) + (sRed[2] + sRed[3]);
        dg[(static_cast<std::size_t>(ks) * B * S + row0 + tr) * HV + hv] = dgr;
    }
    __syncthreads();  // B5: k^T / q^T / ds reads are done; the output tiles reuse them
    put(sOq, dqacc, 1.0f);
    put(sOk, dkacc, 1.0f);
    __syncthreads();  // B6
    get(sOq, dq);
    get(sOk, dk);
}

}  // namespace

// Number of dg planes the kernel writes for a given K (the caller sums them): one per KKH-column slice.
extern "C" int flash_train_gdn_bwd_dqkwg_dg_planes(int K) { return K / kKH; }

extern "C" int flash_train_gdn_bwd_dqkwg_bf16(const FlashTrainGDNBwdDqkwgArgs* a, void* stream) {
    if (!a || !a->q || !a->k || !a->v_new || !a->do_ || !a->dv2 || !a->h || !a->dh || !a->gc || !a->dq ||
        !a->dk || !a->dw || !a->dg)
        return 1;
    if (a->S % kC || a->HV % a->H || a->B <= 0 || a->S <= 0) return 2;
    hipStream_t st = static_cast<hipStream_t>(stream);
    const dim3 grid(static_cast<unsigned>((a->K / kKH) * (a->S / kC) * a->B * a->HV));
#define L(KK, VV)                                                                                          \
    if (a->K == KK && a->V == VV) {                                                                        \
        hipLaunchKernelGGL((gdn_bwd_dqkwg_kernel<KK, VV>), grid, dim3(kThreads), 0, st, a->q, a->k,        \
                           a->v_new, a->do_, a->dv2, a->h, a->dh, a->gc, a->dq, a->dk, a->dw, a->dg,       \
                           a->S, a->H, a->HV, a->B, a->scale);                                             \
        return hipGetLastError() == hipSuccess ? 0 : 5;                                                    \
    }
    L(128, 128) L(64, 128) L(128, 64) L(64, 64)
#undef L
    return 2;
}
