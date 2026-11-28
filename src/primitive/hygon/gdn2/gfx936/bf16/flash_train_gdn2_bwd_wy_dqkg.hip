// B3 wy_dqkg: fused WY backward of GDN-2 (FLA: chunk_gdn2_bwd_kernel_wy_dqkg_fused).  See flash_train_gdn2_bwd.h for the
// equations and the partial-output contract.
//
// One block (4 waves) per (chunk, hv), hv fastest.  Wave w owns token rows 16w..16w+15 (lane: row 16w + lo).  Every mmac
// product is D = A B^T with both operands contraction-contiguous; the B rows are permuted (row = 8 (lo&3) + (lo>>2) (+4)
// inside a 32-wide pair of 16-tiles) so that the D layout hands a lane 8 consecutive columns of its row: all global
// accesses on the [t][channel] side are 16 B and the results of one product are directly the A (or B) fragment of the
// next one (dwfn -> dA, Y^T -> Z).
//
//  * V side (before the channel loop), per 64-wide V slice u:  sVWG[t][v] = bf16(v wg) and the transposed sDV2T[v][t'] go
//    to LDS;  dA += dv2 . sVWG^T  (dv2 kept in registers as A fragments),  dvb = X . dv2  -> dv, dw.
//  * K side, per 64-channel slice s:  the V loop streams h, dh in 32-wide V passes (global -> transposed LDS tiles
//    sHT[kk][v]), accumulating dq_raw = do.h, dk_raw = v_new.dh, dwf = dv2.h in registers;  then the elementwise part 1
//    (gates, dwfn, kgb -> LDS), dkgb = X dwfn and dA += dwfn kgb^T (mmac), and the elementwise part 2 (stores).
//  * Last: dA (bf16, strictly lower) -> Y^T = Akk dA^T -> Z = X Y -> dAkk = -tril(Z).
//
// The in-chunk cumulative gate gc is computed here from g (row scan with DPP, wave totals through LDS).
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../common/flash_train_gdn2_bwd.h"
#include "../common/gdn2_chunk_mmac.h"

namespace {

using gdn2_mmac::bf2f;
using gdn2_mmac::f2bf;
using gdn2_mmac::u16x4;
using gdn2_mmac::v4f;
using gdn2_mmac::v4s;
typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));

constexpr int kC = 64;
constexpr int kNW = 4;
constexpr int kThreads = 64 * kNW;
constexpr float kLog2e = 1.4426950408889634f;
constexpr int XS = 72;  // stride of every 64-wide bf16 matrix in LDS (16 B aligned rows)
constexpr int TS = 40;  // stride of the h / dh tiles [64 kk][32 v]
constexpr int kTileEl = 64 * TS;
constexpr int kUEl = 4 * kTileEl;  // union region: two buffers of (h, dh) tiles; also two 64 x 72 matrices
static_assert(kUEl >= 2 * 64 * XS, "union region");

__device__ __forceinline__ v4s half_lo(u16x8 x) { return __builtin_bit_cast(v4s, __builtin_shufflevector(x, x, 0, 1, 2, 3)); }
__device__ __forceinline__ v4s half_hi(u16x8 x) { return __builtin_bit_cast(v4s, __builtin_shufflevector(x, x, 4, 5, 6, 7)); }
__device__ __forceinline__ v4f mmac(v4s a, v4s b, v4f c) { return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c); }
__device__ __forceinline__ float ex2(float x) { return __builtin_amdgcn_exp2f(x); }

// Block barrier waiting for LDS only (see the P1 note: the builtin makes LLVM drain vmcnt before every barrier on gfx936).
__device__ __forceinline__ void lds_barrier() {
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_waitcnt(0xC07F);
    asm volatile("s_barrier" ::: "memory");
    asm volatile("" ::: "memory");
}

template <int CTRL>
__device__ __forceinline__ float dpp_f(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x), CTRL, 0xF, 0xF, false));
}
// Inclusive prefix sum over the 16 lanes of a DPP row (lane index lo).
__device__ __forceinline__ float row_scan(float x) {
    x += dpp_f<0x111>(x);
    x += dpp_f<0x112>(x);
    x += dpp_f<0x114>(x);
    x += dpp_f<0x118>(x);
    return x;
}
__device__ __forceinline__ float quad_sum(float x) {
    x += dpp_f<0xB1>(x);
    x += dpp_f<0x4E>(x);
    return x;
}

template <int K, int V>
__global__ __launch_bounds__(kThreads) void gdn2_bwd_wy_dqkg_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const uint16_t* __restrict__ vn, const uint16_t* __restrict__ dout, const uint16_t* __restrict__ dv2in,
    const uint16_t* __restrict__ hst, const uint16_t* __restrict__ dhst, const uint16_t* __restrict__ akk,
    const float* __restrict__ g, const float* __restrict__ bg, const float* __restrict__ wg,
    float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ db, float* __restrict__ dg,
    uint16_t* __restrict__ dvo, uint16_t* __restrict__ dwo, float* __restrict__ dakk, int S, int H, int HV,
    float scale, int lane_order) {
    constexpr int KS = K / 64, VS = V / 64, NP = V / 32;
    __shared__ __align__(16) unsigned short sX[64 * XS];
    __shared__ __align__(16) unsigned short sU[kUEl];
    __shared__ __align__(16) float sGT[kNW * 64];
    __shared__ __align__(16) float sKd[kNW * 64];
    __shared__ __align__(16) float sDgk[64];
    unsigned short* const sDWT = sU;
    unsigned short* const sKGB = sU + 64 * XS;
    unsigned short* const sVWG = sU;
    unsigned short* const sDV2T = sU + 64 * XS;
    unsigned short* const sDA = sU;

    const int tid = threadIdx.x, lane = tid & 63;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int lo = lane & 15, hq = lane >> 4;
    const int fl = 4 * (lo & 3) + (lo >> 2);   // row permutation inside a 16-tile
    const int flp = 8 * (lo & 3) + (lo >> 2);  // inside a 32-wide pair of tiles
    int id = blockIdx.x;
    const int hv = id % HV;
    id /= HV;
    const int NT = S / kC;
    const int n = id % NT, bb = id / NT;
    const int hk = hv / (HV / H);
    const std::size_t row0 = static_cast<std::size_t>(bb) * S + static_cast<std::size_t>(n) * kC;
    const int sK = H * K, sG = HV * K, sV = HV * V, sA = HV * 64;
    const int t = 16 * wave + lo;  // chunk-local row of this lane

    const uint16_t* qB = q + (row0 * H + hk) * K;
    const uint16_t* kB = k + (row0 * H + hk) * K;
    const float* gB = g + (row0 * HV + hv) * K;
    const float* bB = bg + (row0 * HV + hv) * K;
    float* dqB = dq + (row0 * HV + hv) * K;
    float* dkB = dk + (row0 * HV + hv) * K;
    float* dbB = db + (row0 * HV + hv) * K;
    float* dgB = dg + (row0 * HV + hv) * K;
    const uint16_t* vB = v + (row0 * HV + hv) * V;
    const uint16_t* vnB = vn + (row0 * HV + hv) * V;
    const uint16_t* doB = dout + (row0 * HV + hv) * V;
    const uint16_t* d2B = dv2in + (row0 * HV + hv) * V;
    const float* wB = wg + (row0 * HV + hv) * V;
    uint16_t* dvB = dvo + (row0 * HV + hv) * V;
    uint16_t* dwB = dwo + (row0 * HV + hv) * V;
    const uint16_t* akB = akk + (row0 * HV + hv) * 64;
    float* dakB = dakk + (row0 * HV + hv) * 64;
    const std::size_t blk = (static_cast<std::size_t>(bb) * NT + n) * HV + hv;
    const uint16_t* hB = hst + blk * V * K;
    const uint16_t* dhB = dhst + blk * V * K;

    // ---- A fragments of do, v_new, dv2 for the whole block (row t, v = 32 j + 8 hq .. +7) ----
    u16x8 fdo[NP], fvn[NP], fd2[NP];
#pragma unroll
    for (int j = 0; j < NP; ++j) {
        fdo[j] = *reinterpret_cast<const u16x8*>(doB + static_cast<std::size_t>(t) * sV + 32 * j + 8 * hq);
        fvn[j] = *reinterpret_cast<const u16x8*>(vnB + static_cast<std::size_t>(t) * sV + 32 * j + 8 * hq);
        fd2[j] = *reinterpret_cast<const u16x8*>(d2B + static_cast<std::size_t>(t) * sV + 32 * j + 8 * hq);
    }

    // ---- V side operand loads, one slice ahead: fill set (v, wg, two dv2 rows) and epilogue set (v, wg at row t) ----
    u16x8 vx[2], vy0, vy1, ex[2];
    v4f vw0[2], vw1[2], ew[2][2];
    auto vload = [&](int u) {
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int r = 16 * wave + (lane >> 3) + 8 * it, pc = lane & 7;
            const std::size_t off = static_cast<std::size_t>(r) * sV + 64 * u + 8 * pc;
            vx[it] = *reinterpret_cast<const u16x8*>(vB + off);
            vw0[it] = *reinterpret_cast<const v4f*>(wB + off);
            vw1[it] = *reinterpret_cast<const v4f*>(wB + off + 4);
        }
        const int rp = tid >> 3, pc = tid & 7;
        vy0 = *reinterpret_cast<const u16x8*>(d2B + static_cast<std::size_t>(2 * rp) * sV + 64 * u + 8 * pc);
        vy1 = *reinterpret_cast<const u16x8*>(d2B + static_cast<std::size_t>(2 * rp + 1) * sV + 64 * u + 8 * pc);
    };
    auto eload = [&](int u) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const std::size_t off = static_cast<std::size_t>(t) * sV + 64 * u + 32 * p + 8 * hq;
            ex[p] = *reinterpret_cast<const u16x8*>(vB + off);
            ew[p][0] = *reinterpret_cast<const v4f*>(wB + off);
            ew[p][1] = *reinterpret_cast<const v4f*>(wB + off + 4);
        }
    };
    // g of the K slice s_, loaded one stage ahead (slice 0: during the last V iteration)
    v4f gr[2][2];
    auto gload = [&](int s_) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const std::size_t off = static_cast<std::size_t>(t) * sG + 64 * s_ + 32 * p + 8 * hq;
            gr[p][0] = *reinterpret_cast<const v4f*>(gB + off);
            gr[p][1] = *reinterpret_cast<const v4f*>(gB + off + 4);
        }
    };
    vload(0);

    // ---- sX[r][c] = X[r][c] = Akk[c][r] (upper triangular): two Akk rows per thread, 4 B transposed writes ----
    {
        const int rp = tid >> 3, pc = tid & 7;
        const u16x8 r0 = *reinterpret_cast<const u16x8*>(akB + static_cast<std::size_t>(2 * rp) * sA + 8 * pc);
        const u16x8 r1 = *reinterpret_cast<const u16x8*>(akB + static_cast<std::size_t>(2 * rp + 1) * sA + 8 * pc);
#pragma unroll
        for (int qq = 0; qq < 8; ++qq)
            *reinterpret_cast<unsigned*>(sX + (8 * pc + qq) * XS + 2 * rp) =
                static_cast<unsigned>(r0[qq]) | (static_cast<unsigned>(r1[qq]) << 16);
    }

    v4f dA[4];  // dA[i = t][j]: D layout, columns j = 16 jt + 4 hq + e
#pragma unroll
    for (int jt = 0; jt < 4; ++jt) dA[jt] = v4f{0.f, 0.f, 0.f, 0.f};

    // =============================== V side ===============================
#pragma unroll
    for (int u = 0; u < VS; ++u) {
        eload(u);
        // fill the LDS operands of slice u from the registers loaded one iteration earlier
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int r = 16 * wave + (lane >> 3) + 8 * it, pc = lane & 7;
            u16x8 o;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                o[e] = f2bf(bf2f(vx[it][e]) * vw0[it][e]);
                o[4 + e] = f2bf(bf2f(vx[it][4 + e]) * vw1[it][e]);
            }
            *reinterpret_cast<u16x8*>(sVWG + r * XS + 8 * pc) = o;
        }
        {
            const int rp = tid >> 3, pc = tid & 7;
#pragma unroll
            for (int qq = 0; qq < 8; ++qq)
                *reinterpret_cast<unsigned*>(sDV2T + (8 * pc + qq) * XS + 2 * rp) =
                    static_cast<unsigned>(vy0[qq]) | (static_cast<unsigned>(vy1[qq]) << 16);
        }
        if (u + 1 < VS)
            vload(u + 1);
        else
            gload(0);
        lds_barrier();

        // dA += dv2 . vwg^T
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const u16x8 a = fd2[2 * u + c];
#pragma unroll
            for (int jt = 0; jt < 4; ++jt) {
                const u16x8 bf = *reinterpret_cast<const u16x8*>(sVWG + (16 * jt + fl) * XS + 32 * c + 8 * hq);
                dA[jt] = mmac(half_lo(a), half_lo(bf), dA[jt]);
                dA[jt] = mmac(half_hi(a), half_hi(bf), dA[jt]);
            }
        }
        // dvb = X . dv2   (4 tiles of 16 v)
        v4f dvb[4];
#pragma unroll
        for (int T = 0; T < 4; ++T) dvb[T] = v4f{0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            if (c == 1 || wave < 2) {  // X[t][t'] = 0 for t' < t
                const u16x8 a = *reinterpret_cast<const u16x8*>(sX + t * XS + 32 * c + 8 * hq);
#pragma unroll
                for (int T = 0; T < 4; ++T) {
                    const u16x8 bf = *reinterpret_cast<const u16x8*>(
                        sDV2T + (32 * (T >> 1) + flp + 4 * (T & 1)) * XS + 32 * c + 8 * hq);
                    dvb[T] = mmac(half_lo(a), half_lo(bf), dvb[T]);
                    dvb[T] = mmac(half_hi(a), half_hi(bf), dvb[T]);
                }
            }
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const std::size_t off = static_cast<std::size_t>(t) * sV + 64 * u + 32 * p + 8 * hq;
            u16x8 od, ow;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                od[e] = f2bf(dvb[2 * p][e] * ew[p][0][e]);
                ow[e] = f2bf(dvb[2 * p][e] * bf2f(ex[p][e]));
                od[4 + e] = f2bf(dvb[2 * p + 1][e] * ew[p][1][e]);
                ow[4 + e] = f2bf(dvb[2 * p + 1][e] * bf2f(ex[p][4 + e]));
            }
            *reinterpret_cast<u16x8*>(dvB + off) = od;
            *reinterpret_cast<u16x8*>(dwB + off) = ow;
        }
        lds_barrier();
    }

    // =============================== K side ===============================
#pragma unroll 1
    for (int s = 0; s < KS; ++s) {
        v4f aq[4], ak[4], aw[4];
#pragma unroll
        for (int T = 0; T < 4; ++T) aq[T] = ak[T] = aw[T] = v4f{0.f, 0.f, 0.f, 0.f};
        float dgacc = 0.f;

        // tile loader of pass j: mat 0 = h (threads 0..127), 1 = dh
        const int mat = tid >> 7, rp = (tid >> 3) & 15, pc = tid & 7;
        const uint16_t* tsrc = (mat ? dhB : hB) + static_cast<std::size_t>(2 * rp) * K + 64 * s + 8 * pc;
        u16x8 pt0, pt1;
        auto load_tiles = [&](int j) {
            pt0 = *reinterpret_cast<const u16x8*>(tsrc + static_cast<std::size_t>(32 * j) * K);
            pt1 = *reinterpret_cast<const u16x8*>(tsrc + static_cast<std::size_t>(32 * j + 1) * K);
        };
        auto write_tiles = [&](int j) {
            unsigned short* dst = sU + (j & 1) * 2 * kTileEl + mat * kTileEl + (8 * pc) * TS + 2 * rp;
#pragma unroll
            for (int qq = 0; qq < 8; ++qq)
                *reinterpret_cast<unsigned*>(dst + qq * TS) = static_cast<unsigned>(pt0[qq]) | (static_cast<unsigned>(pt1[qq]) << 16);
        };
        load_tiles(0);

        // ---- gc scan: g (natural) * log2e, inclusive over the wave's 16 rows; row totals to sGT ----
        float sc[2][8];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                sc[p][e] = row_scan(gr[p][0][e] * kLog2e);
                sc[p][4 + e] = row_scan(gr[p][1][e] * kLog2e);
            }
            if (lo == 15) {
                *reinterpret_cast<v4f*>(sGT + wave * 64 + 32 * p + 8 * hq) = v4f{sc[p][0], sc[p][1], sc[p][2], sc[p][3]};
                *reinterpret_cast<v4f*>(sGT + wave * 64 + 32 * p + 8 * hq + 4) = v4f{sc[p][4], sc[p][5], sc[p][6], sc[p][7]};
            }
        }

        // ---- V loop over the h / dh tiles: double-buffered LDS, one barrier per pass, loads two passes ahead ----
        const int dkl = tid >> 2, dpart = tid & 3;
        write_tiles(0);
        if (NP > 1) load_tiles(1);
        lds_barrier();
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            if (j + 1 < NP) {
                write_tiles(j + 1);
                if (j + 2 < NP) load_tiles(j + 2);
            }
            const unsigned short* sHT = sU + (j & 1) * 2 * kTileEl;
            const unsigned short* sDhT = sHT + kTileEl;
            // dgk partial: sum_v h dh for channel dkl, 8 values of v
            {
                const u16x8 hh = *reinterpret_cast<const u16x8*>(sHT + dkl * TS + 8 * dpart);
                const u16x8 dd = *reinterpret_cast<const u16x8*>(sDhT + dkl * TS + 8 * dpart);
#pragma unroll
                for (int e = 0; e < 8; ++e) dgacc += bf2f(hh[e]) * bf2f(dd[e]);
            }
            const u16x8 a_do = fdo[j], a_vn = fvn[j], a_d2 = fd2[j];
#pragma unroll
            for (int T = 0; T < 4; ++T) {
                const int row = 32 * (T >> 1) + flp + 4 * (T & 1);
                const u16x8 bh = *reinterpret_cast<const u16x8*>(sHT + row * TS + 8 * hq);
                const u16x8 bd = *reinterpret_cast<const u16x8*>(sDhT + row * TS + 8 * hq);
                aq[T] = mmac(half_lo(a_do), half_lo(bh), aq[T]);
                aq[T] = mmac(half_hi(a_do), half_hi(bh), aq[T]);
                aw[T] = mmac(half_lo(a_d2), half_lo(bh), aw[T]);
                aw[T] = mmac(half_hi(a_d2), half_hi(bh), aw[T]);
                ak[T] = mmac(half_lo(a_vn), half_lo(bd), ak[T]);
                ak[T] = mmac(half_hi(a_vn), half_hi(bd), ak[T]);
            }
            lds_barrier();
        }
        dgacc = quad_sum(dgacc);
        if (dpart == 0) sDgk[dkl] = dgacc;

        // ---- channel pairs: p = 0, 1 (32 channels each) are independent from here on ----
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            // gate carries: gc = scan + sum of the earlier waves' totals, gn = sum of all four
            float gcv[8], gnv[8];
            {
                float carry[8], tot[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) carry[e] = tot[e] = 0.f;
#pragma unroll
                for (int w2 = 0; w2 < kNW; ++w2) {
                    const v4f t0 = *reinterpret_cast<const v4f*>(sGT + w2 * 64 + 32 * p + 8 * hq);
                    const v4f t1 = *reinterpret_cast<const v4f*>(sGT + w2 * 64 + 32 * p + 8 * hq + 4);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        if (w2 < wave) {
                            carry[e] += t0[e];
                            carry[4 + e] += t1[e];
                        }
                        tot[e] += t0[e];
                        tot[4 + e] += t1[e];
                    }
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    gcv[e] = sc[p][e] + carry[e];
                    gnv[e] = tot[e];
                }
            }

            // ---- part 1: gates, dq (final for this stage), dwfn, kgb, partial dg ----
            const std::size_t offk = static_cast<std::size_t>(t) * sK + 64 * s + 32 * p + 8 * hq;
            const std::size_t offg = static_cast<std::size_t>(t) * sG + 64 * s + 32 * p + 8 * hq;
            const u16x8 kv8 = *reinterpret_cast<const u16x8*>(kB + offk);
            const u16x8 q8 = *reinterpret_cast<const u16x8*>(qB + offk);
            const v4f b0 = *reinterpret_cast<const v4f*>(bB + offg);
            const v4f b1 = *reinterpret_cast<const v4f*>(bB + offg + 4);
            float gke[8], dkp[8], dg1[8], bvv[8];
            u16x8 kgb8, dwfn8;
            v4f o_dq[2];
            float kdk[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int T = 2 * p + (e >> 2), c = e & 3;
                const float gk = ex2(gcv[e]);
                gke[e] = gk;
                bvv[e] = e < 4 ? b0[e] : b1[e - 4];
                const float dqo = aq[T][c] * gk * scale;
                o_dq[e >> 2][c] = dqo;
                dkp[e] = ak[T][c] * ex2(gnv[e] - gcv[e]);
                dwfn8[e] = f2bf(-aw[T][c]);
                const float kf = bf2f(kv8[e]);
                kgb8[e] = f2bf(kf * gk * bvv[e]);
                kdk[e] = kf * dkp[e];
                dg1[e] = bf2f(q8[e]) * dqo - kdk[e];
            }
            // lane-order partial layout (see flash_train_gdn2_bwd.h): this store instruction writes 1 KB contiguous
            const std::size_t ptile = (static_cast<std::size_t>(bb) * NT + n) * HV + hv;
            auto pidx = [&](int h2) { return ((((((ptile * KS + s) * 2 + p) * 2 + h2) * 4 + wave) * 64) + lane) * 4; };
            if (lane_order) {
                *reinterpret_cast<v4f*>(dq + pidx(0)) = o_dq[0];
                *reinterpret_cast<v4f*>(dq + pidx(1)) = o_dq[1];
            } else {
                *reinterpret_cast<v4f*>(dqB + offg) = o_dq[0];
                *reinterpret_cast<v4f*>(dqB + offg + 4) = o_dq[1];
            }
            // column sums of kdk over the wave's 16 rows -> sKd
            float cs[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) cs[e] = row_scan(kdk[e]);
            if (lo == 15) {
                *reinterpret_cast<v4f*>(sKd + wave * 64 + 32 * p + 8 * hq) = v4f{cs[0], cs[1], cs[2], cs[3]};
                *reinterpret_cast<v4f*>(sKd + wave * 64 + 32 * p + 8 * hq + 4) = v4f{cs[4], cs[5], cs[6], cs[7]};
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) sDWT[(32 * p + 8 * hq + e) * XS + t] = dwfn8[e];
            *reinterpret_cast<u16x8*>(sKGB + t * XS + 32 * p + 8 * hq) = kgb8;
            lds_barrier();

            // ---- part 2: dkgb = X dwfn (2 tiles) ; dA += dwfn kgb^T ----
            v4f dkb[2];
            dkb[0] = dkb[1] = v4f{0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                if (c == 1 || wave < 2) {
                    const u16x8 a = *reinterpret_cast<const u16x8*>(sX + t * XS + 32 * c + 8 * hq);
#pragma unroll
                    for (int sub = 0; sub < 2; ++sub) {
                        const u16x8 bf = *reinterpret_cast<const u16x8*>(
                            sDWT + (32 * p + flp + 4 * sub) * XS + 32 * c + 8 * hq);
                        dkb[sub] = mmac(half_lo(a), half_lo(bf), dkb[sub]);
                        dkb[sub] = mmac(half_hi(a), half_hi(bf), dkb[sub]);
                    }
                }
            }
#pragma unroll
            for (int jt = 0; jt < 4; ++jt) {
                const u16x8 bf = *reinterpret_cast<const u16x8*>(sKGB + (16 * jt + fl) * XS + 32 * p + 8 * hq);
                dA[jt] = mmac(half_lo(dwfn8), half_lo(bf), dA[jt]);
                dA[jt] = mmac(half_hi(dwfn8), half_hi(bf), dA[jt]);
            }

            // ---- part 3: dk, db, dg stores ----
            float dgt[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) dgt[e] = 0.f;
            if (wave == kNW - 1 && lo == 15) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    float cs2 = 0.f;
#pragma unroll
                    for (int w2 = 0; w2 < kNW; ++w2) cs2 += sKd[w2 * 64 + 32 * p + 8 * hq + e];
                    dgt[e] = sDgk[32 * p + 8 * hq + e] * ex2(gnv[e]) + cs2;
                }
            }
            v4f o_dk[2], o_db[2], o_dg[2];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float kg = bf2f(kv8[e]) * gke[e];
                const float dkg = dkb[e >> 2][e & 3];
                o_db[e >> 2][e & 3] = dkg * kg;
                o_dg[e >> 2][e & 3] = dg1[e] + kg * dkg * bvv[e] + dgt[e];
                o_dk[e >> 2][e & 3] = dkp[e] + dkg * gke[e] * bvv[e];
            }
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                if (lane_order) {
                    *reinterpret_cast<v4f*>(dk + pidx(h2)) = o_dk[h2];
                    *reinterpret_cast<v4f*>(db + pidx(h2)) = o_db[h2];
                    *reinterpret_cast<v4f*>(dg + pidx(h2)) = o_dg[h2];
                } else {
                    *reinterpret_cast<v4f*>(dkB + offg + 4 * h2) = o_dk[h2];
                    *reinterpret_cast<v4f*>(dbB + offg + 4 * h2) = o_db[h2];
                    *reinterpret_cast<v4f*>(dgB + offg + 4 * h2) = o_dg[h2];
                }
            }
        }
        if (s + 1 < KS) gload(s + 1);
        lds_barrier();
    }

    // =============================== dAkk ===============================
    v4s ak_a[4];
#pragma unroll
    for (int ts = 0; ts < 4; ++ts) {
        ak_a[ts] = v4s{0, 0, 0, 0};
        if (ts <= wave) {
            const unsigned short* xp = sX + (16 * ts + 4 * hq) * XS + 16 * wave + fl;  // Akk[16 wave + fl][16 ts + 4 hq + i]
            ak_a[ts] = v4s{static_cast<short>(xp[0]), static_cast<short>(xp[XS]), static_cast<short>(xp[2 * XS]), static_cast<short>(xp[3 * XS])};
        }
    }
#pragma unroll
    for (int jt = 0; jt < 4; ++jt) {
        u16x4 o;
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = (t > 16 * jt + 4 * hq + e) ? f2bf(dA[jt][e]) : static_cast<unsigned short>(0);
        *reinterpret_cast<u16x4*>(sDA + t * XS + 16 * jt + 4 * hq) = o;
    }
    lds_barrier();
    v4s yb[4];
#pragma unroll
    for (int ti = 0; ti < 4; ++ti) {
        v4f acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ts = 0; ts < 4; ++ts) {
            if (ts <= ti && ts <= wave) {
                const v4s bf = *reinterpret_cast<const v4s*>(sDA + (16 * ti + fl) * XS + 16 * ts + 4 * hq);
                acc = mmac(ak_a[ts], bf, acc);
            }
        }
        yb[ti] = v4s{static_cast<short>(f2bf(acc[0])), static_cast<short>(f2bf(acc[1])), static_cast<short>(f2bf(acc[2])),
                     static_cast<short>(f2bf(acc[3]))};
    }
#pragma unroll
    for (int ti2 = 0; ti2 < 4; ++ti2) {
        v4f acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ti = ti2; ti < 4; ++ti) {
            const v4s a = *reinterpret_cast<const v4s*>(sX + (16 * ti2 + lo) * XS + 16 * ti + 4 * hq);
            acc = mmac(a, yb[ti], acc);
        }
        v4f o;
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = (16 * ti2 + lo > 16 * wave + 4 * hq + e) ? -acc[e] : 0.f;
        *reinterpret_cast<v4f*>(dakB + static_cast<std::size_t>(16 * ti2 + lo) * sA + 16 * wave + 4 * hq) = o;
    }
}

template <int K, int V>
int launch(const FlashTrainGDN2BwdWyDqkgArgs& a, hipStream_t st) {
    const dim3 grid(static_cast<unsigned>((a.S / kC) * a.B * a.HV));
    hipLaunchKernelGGL((gdn2_bwd_wy_dqkg_kernel<K, V>), grid, dim3(kThreads), 0, st, a.q, a.k, a.v, a.v_new, a.do_, a.dv2,
                       a.h, a.dh, a.akk_inv, a.g, a.b, a.wg, a.dq, a.dk, a.db, a.dg, a.dv, a.dw, a.dAkk, a.S, a.H, a.HV,
                       a.scale, a.partial_lane_order);
    return hipGetLastError() == hipSuccess ? 0 : 5;
}

}  // namespace

extern "C" int flash_train_gdn2_bwd_wy_dqkg_bf16(const FlashTrainGDN2BwdWyDqkgArgs* a, void* stream) {
    if (a == nullptr || !a->q || !a->k || !a->v || !a->v_new || !a->do_ || !a->dv2 || !a->h || !a->dh || !a->akk_inv ||
        !a->g || !a->b || !a->wg || !a->dq || !a->dk || !a->db || !a->dg || !a->dv || !a->dw || !a->dAkk)
        return 1;
    if (a->B <= 0 || a->S <= 0 || a->S % kC != 0 || a->H <= 0 || a->HV <= 0 || a->HV % a->H != 0) return 2;
    hipStream_t st = static_cast<hipStream_t>(stream);
    if (a->K == 128 && a->V == 128) return launch<128, 128>(*a, st);
    if (a->K == 64 && a->V == 128) return launch<64, 128>(*a, st);
    if (a->K == 128 && a->V == 64) return launch<128, 64>(*a, st);
    if (a->K == 64 && a->V == 64) return launch<64, 64>(*a, st);
    return 2;
}
