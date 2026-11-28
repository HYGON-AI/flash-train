// GDN backward, stage B4 (phase P3): wy_bwd + epilogue.
//
// Per (chunk n, b, hv) -- 1-D grid, hv fastest, 4 waves.  With A the chunk's (I+L)^-1 (rows t,
// columns a, lower triangular, bf16 from the intra kernel), e_t = 2^gc_t, dw/dv2 the gradients of
// w = A (beta e k) and u = A (beta v) (dw from dqkwg carries the sign), the block computes
//   Dk = A^T dw, Dv = A^T dv2                        (A^T[a][t] = A[t][a])
//   dk2_a = beta_a e_a Dk_a ;  dv_a = beta_a Dv_a
//   dA[i][j] = beta_j e_j (dw_i . k_j) + beta_j (dv2_i . v_j)           (strict lower i > j, bf16)
//   Y = dA A^T (bf16),  Z = A^T Y,  dA'[i][j] = -Z[i][j] 2^(gc_i - gc_j)  (strict lower, bf16)
//   G = k k^T (Gram)
//   dk_j = dk(dqkwg)_j + dk2_j + beta_j sum_x dA'[j][x] k_x + sum_i beta_i dA'[i][j] k_i
//   dbeta_a = e_a (Dk_a . k_a) + Dv_a . v_a + sum_x dA'[a][x] G[a][x]
//   dg2_a = beta_a e_a (Dk_a . k_a) + beta_a sum_x dA'[a][x] G[a][x] - sum_i dA'[i][a] beta_i G[i][a]
//   dg = reverse in-chunk cumsum( planes(dqkwg) + dg2 )
// There is no ln2 anywhere: gc = g / ln2 and d 2^gc / dg = 2^gc, exactly as in FLA's kernels.
// The second sum in dk_j and dbeta use dA' k = dA' @ k and its Gram form, which removes the dkb
// product of FLA's kernel.
//
// Data flow: K and V are consumed in 64-column slices.  dw/du and k/v are read once in the
// "wave row" layout (lane = row 16*wave + lo, 16B chunks hq + 4c) and used as A operands of
// dA = dw k^T straight from registers; k (K slices) is also kept for the final dk product.
// A^T lives in LDS for the whole block; A^T's A-fragments live in registers.
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "../common/flash_train_gdn_bwd.h"
#include "../common/gdn_chunk_mmac.h"

namespace {
using namespace gdn_mmac;

constexpr int ST = 68;  // LDS tile row stride (elements); 64 rows
constexpr int TILE = 64 * ST;

struct Sl {
    uint4 a[2];  // k or v, wave-row layout
    uint4 b[2];  // dw or dv2
};

__device__ __forceinline__ v4s pk4(unsigned a, unsigned b) {
    const unsigned long long x = (static_cast<unsigned long long>(b) << 32) | a;
    v4s r;
    __builtin_memcpy(&r, &x, 8);
    return r;
}
__device__ __forceinline__ v4s frag(uint4 q, int h2) {
    return h2 == 0 ? pk4(q.x, q.y) : pk4(q.z, q.w);
}
__device__ __forceinline__ unsigned elem(uint4 q, int e) {  // bf16 bits of element e
    const unsigned w = (e >> 1) == 0 ? q.x : (e >> 1) == 1 ? q.y : (e >> 1) == 2 ? q.z : q.w;
    return (w >> (16 * (e & 1))) & 0xffffu;
}
__device__ __forceinline__ unsigned pack2(unsigned lo, unsigned hi) { return lo | (hi << 16); }
__device__ __forceinline__ v4f mma(v4s a, v4s b, v4f c) {
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
}
// 16B -> two 8B LDS stores (row stride 68 is only 8B aligned)
__device__ __forceinline__ void put8(uint16_t* dst, uint4 q) {
    *reinterpret_cast<uint2*>(dst) = make_uint2(q.x, q.y);
    *reinterpret_cast<uint2*>(dst + 4) = make_uint2(q.z, q.w);
}
// transposed store: base[(col0 + e) * ST + row] = element e
__device__ __forceinline__ void put_T(uint16_t* base, int col0, int row, uint4 q) {
#pragma unroll
    for (int e = 0; e < 8; ++e) base[(col0 + e) * ST + row] = static_cast<uint16_t>(elem(q, e));
}
// B-operand row for output tile q (pair m = q>>1, half rr = q&1) so that lane (lo, hq) element i
// of D holds column 32m + 8hq + 4rr + i: eight consecutive columns per lane over a tile pair.
__device__ __forceinline__ int bcol(int q, int lo) {
    return 32 * (q >> 1) + 8 * (lo & 3) + 4 * (q & 1) + (lo >> 2);
}
// one reduction over the 4 lanes of a lane%16 group that share a row
__device__ __forceinline__ float rsum4(float x) {
    x += __shfl_xor(x, 16);
    x += __shfl_xor(x, 32);
    return x;
}

template <int K, int V>
__global__ void __launch_bounds__(256)
gdn_bwd_wy_kernel(const uint16_t* __restrict__ gk, const uint16_t* __restrict__ gv,
                  const uint16_t* __restrict__ gA, const uint16_t* __restrict__ gdw,
                  const uint16_t* __restrict__ gdu, const uint16_t* gdkin,
                  const float* __restrict__ gbeta, const float* __restrict__ ggc,
                  const float* __restrict__ gplanes, uint16_t* gdk, uint16_t* __restrict__ gdv,
                  float* __restrict__ gdg, float* __restrict__ gdbeta, int B, int S, int H, int HV,
                  int nplanes) {
    constexpr int NKS = K / 64, NVS = V / 64, NS = NKS + NVS;
    __shared__ __align__(16) uint16_t sAT[TILE];
    __shared__ __align__(16) uint16_t sR1[TILE];
    __shared__ __align__(16) uint16_t sR2[TILE];
    __shared__ __align__(16) float sGc[64];
    __shared__ __align__(16) float sBeta[64];
    __shared__ __align__(16) float sBE[64];
    __shared__ float sCol[256];
    __shared__ float sDg[64];

    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63, lo = lane & 15, hq = lane >> 4;
    const int r = 16 * wave + lo;
    const int NT = S / 64, G = HV / H;
    const int hv = blockIdx.x % HV, cb = blockIdx.x / HV;
    const int n = cb % NT, b = cb / NT, hk = hv / G;
    const size_t tokbase = static_cast<size_t>(b) * S + static_cast<size_t>(n) * 64;
    const size_t tok = tokbase + r;

    const uint16_t* kq = gk + (tok * H + hk) * K;
    const uint16_t* vq = gv + (tok * HV + hv) * V;
    const uint16_t* Aq = gA + (tok * HV + hv) * 64;
    const uint16_t* dwq = gdw + (tok * HV + hv) * K;
    const uint16_t* duq = gdu + (tok * HV + hv) * V;
    const uint16_t* dkq = gdkin + (tok * HV + hv) * K;
    uint16_t* dkout = gdk + (tok * HV + hv) * K;
    uint16_t* dvout = gdv + (tok * HV + hv) * V;

    auto ld = [](const uint16_t* p, int col) { return *reinterpret_cast<const uint4*>(p + col); };
    auto load_slice = [&](int s, Sl& x) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int col = 8 * (hq + 4 * c);
            if (s < NKS) {
                x.a[c] = ld(kq, s * 64 + col);
                x.b[c] = ld(dwq, s * 64 + col);
            } else {
                x.a[c] = ld(vq, (s - NKS) * 64 + col);
                x.b[c] = ld(duq, (s - NKS) * 64 + col);
            }
        }
    };

    // ---- prologue loads: A, per-token scalars, first slice ----
    uint4 aq[2];
#pragma unroll
    for (int c = 0; c < 2; ++c) aq[c] = ld(Aq, 8 * (hq + 4 * c));
    Sl cur, nxt;
    load_slice(0, cur);
    float plane = 0.0f;
    if (tid < 64) {
        const size_t tt = tokbase + tid;
        const float gcv = ggc[tt * HV + hv], bv = gbeta[tt * HV + hv];
        sGc[tid] = gcv;
        sBeta[tid] = bv;
        sBE[tid] = bv * exp2f(gcv);
        for (int p = 0; p < nplanes; ++p)
            plane += gplanes[static_cast<size_t>(p) * B * S * HV + tt * HV + hv];
    }
#pragma unroll
    for (int c = 0; c < 2; ++c) put_T(sAT, 8 * (hq + 4 * c), r, aq[c]);
    if (tid < 64) sDg[tid] = plane;

    v4f dAk[4], dAv[4], Gm[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dAk[i] = v4f{0, 0, 0, 0};
        dAv[i] = v4f{0, 0, 0, 0};
        Gm[i] = v4f{0, 0, 0, 0};
    }
    float dkacc[NKS][2][8];  // dk(dqkwg) is added later; this holds beta e Dk
    uint4 kreg[NKS][2];      // raw k, kept for the dk chain product
    float sK = 0.0f, sV = 0.0f;
    v4s afr[4];
    float betar = 0.0f, bbe = 0.0f;

#pragma unroll
    for (int s = 0; s < NS; ++s) {
        // stage the current slice
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            put8(sR1 + r * ST + 8 * (hq + 4 * c), cur.a[c]);
            put_T(sR2, 8 * (hq + 4 * c), r, cur.b[c]);
        }
        if (s + 1 < NS) load_slice(s + 1, nxt);
        __syncthreads();
        if (s == 0) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
                afr[ks] = *reinterpret_cast<const v4s*>(sAT + r * ST + 16 * ks + 4 * hq);
            betar = sBeta[r];
            bbe = sBE[r];
        }
        // dA += (dw|dv2) (k|v)^T, Gram += k k^T
#pragma unroll
        for (int it = 0; it < 4; ++it) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                    const v4s bf = *reinterpret_cast<const v4s*>(
                        sR1 + (16 * it + lo) * ST + 8 * (hq + 4 * c) + 4 * h2);
                    if (s < NKS) {
                        dAk[it] = mma(frag(cur.b[c], h2), bf, dAk[it]);
                        Gm[it] = mma(frag(cur.a[c], h2), bf, Gm[it]);
                    } else {
                        dAv[it] = mma(frag(cur.b[c], h2), bf, dAv[it]);
                    }
                }
            }
        }
        // D = A^T (dw|dv2) slice, eight consecutive columns per lane
        v4f D[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) D[q] = v4f{0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const v4s bq = *reinterpret_cast<const v4s*>(sR2 + bcol(q, lo) * ST + 16 * ks + 4 * hq);
                D[q] = mma(afr[ks], bq, D[q]);
            }
        }
        if (s < NKS) {
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                kreg[s][m] = cur.a[m];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float val = D[2 * m + (e >> 2)][e & 3];
                    sK += val * bf2f(static_cast<uint16_t>(elem(cur.a[m], e)));
                    dkacc[s][m][e] = val * bbe;
                }
            }
        } else {
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                unsigned o[4];
#pragma unroll
                for (int e2 = 0; e2 < 4; ++e2) {
                    unsigned h2v[2];
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int e = 2 * e2 + j;
                        const float val = D[2 * m + (e >> 2)][e & 3];
                        sV += val * bf2f(static_cast<uint16_t>(elem(cur.a[m], e)));
                        h2v[j] = f2bf(val * betar);
                    }
                    o[e2] = pack2(h2v[0], h2v[1]);
                }
                *reinterpret_cast<uint4*>(dvout + (s - NKS) * 64 + 8 * (hq + 4 * m)) =
                    make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
        __syncthreads();
        if (s + 1 < NS) cur = nxt;
    }
    sK = rsum4(sK);
    sV = rsum4(sV);
    // dk(dqkwg) rows: issue now, used after the chain
    uint4 dki[NKS][2];
#pragma unroll
    for (int s = 0; s < NKS; ++s)
#pragma unroll
        for (int m = 0; m < 2; ++m) dki[s][m] = ld(dkq, s * 64 + 8 * (hq + 4 * m));

    // ---- X = strict_lower(dA) as bf16, straight from the accumulator layout ----
    // lane (lo, hq), tile it, element e <-> row i = r, column j = 16 it + hq + 4 e.
    v4s Xfr[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        unsigned x[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int j = 16 * it + hq + 4 * e;
            const float val = dAk[it][e] * sBE[j] + dAv[it][e] * sBeta[j];
            x[e] = (r > j) ? f2bf(val) : 0u;
        }
        Xfr[it] = pk4(pack2(x[0], x[1]), pack2(x[2], x[3]));
    }
    // Y = X A^T : contraction index x = 16 it + hq + 4 s on both operands; B[c][x] = A[c][x] = AT[x][c]
    v4f Y[4];
#pragma unroll
    for (int cn = 0; cn < 4; ++cn) {
        Y[cn] = v4f{0, 0, 0, 0};
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            unsigned g[4];
#pragma unroll
            for (int s = 0; s < 4; ++s) g[s] = sAT[(16 * it + hq + 4 * s) * ST + 16 * cn + lo];
            Y[cn] = mma(Xfr[it], pk4(pack2(g[0], g[1]), pack2(g[2], g[3])), Y[cn]);
        }
    }
#pragma unroll
    for (int cn = 0; cn < 4; ++cn)
#pragma unroll
        for (int e = 0; e < 4; ++e) sR2[(16 * cn + hq + 4 * e) * ST + r] = static_cast<uint16_t>(f2bf(Y[cn][e]));
    __syncthreads();
    // Z = A^T Y
    v4f Z[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        Z[it] = v4f{0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const v4s bq = *reinterpret_cast<const v4s*>(sR2 + (16 * it + lo) * ST + 16 * ks + 4 * hq);
            Z[it] = mma(afr[ks], bq, Z[it]);
        }
    }
    // dA' = -Z 2^(gc_i - gc_j), strict lower, bf16
    const float gcr = sGc[r];
    float rb = 0.0f;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int c = 16 * it + hq + 4 * e;
            const unsigned bits = (r > c) ? f2bf(-Z[it][e] * exp2f(gcr - sGc[c])) : 0u;
            sR1[r * ST + c] = static_cast<uint16_t>(bits);
            const float dap = bf2f(static_cast<uint16_t>(bits));
            const float gg = Gm[it][e];
            rb += dap * gg;
            float cw = dap * betar * gg;  // column partial: sum over rows (lo) and waves
            cw += __shfl_xor(cw, 1);
            cw += __shfl_xor(cw, 2);
            cw += __shfl_xor(cw, 4);
            cw += __shfl_xor(cw, 8);
            if (lo == 0) sCol[wave * 64 + c] = cw;
        }
    }
    rb = rsum4(rb);
    __syncthreads();
    // Mtot[j][x] = beta_j dA'[j][x] + beta_x dA'[x][j] as A fragments, contraction x = 16 ks + 4 hq + s
    v4s Mfr[4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int x0 = 16 * ks + 4 * hq;
        const uint2 t1 = *reinterpret_cast<const uint2*>(sR1 + r * ST + x0);
        const float4 bx = *reinterpret_cast<const float4*>(sBeta + x0);
        const float bxv[4] = {bx.x, bx.y, bx.z, bx.w};
        unsigned m4[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const unsigned w1 = (s >> 1) == 0 ? t1.x : t1.y;
            const float d1 = bf2f(static_cast<uint16_t>((w1 >> (16 * (s & 1))) & 0xffffu));
            const float d2 = bf2f(sR1[(x0 + s) * ST + r]);
            m4[s] = f2bf(betar * d1 + bxv[s] * d2);
        }
        Mfr[ks] = pk4(pack2(m4[0], m4[1]), pack2(m4[2], m4[3]));
    }
    // per-row scalars (valid in the hq == 0 lanes, used there)
    if (hq == 0) {
        const float csum = sCol[r] + sCol[64 + r] + sCol[128 + r] + sCol[192 + r];
        const float e2 = exp2f(gcr);
        sDg[r] = sDg[r] + betar * e2 * sK + betar * rb - csum;
        gdbeta[(tokbase + r) * HV + hv] = e2 * sK + sV + rb;
    }
    // ---- dk = dk(dqkwg) + beta e Dk + Mtot k ----
#pragma unroll
    for (int s = 0; s < NKS; ++s) {
#pragma unroll
        for (int m = 0; m < 2; ++m) put_T(sR2, 8 * (hq + 4 * m), r, kreg[s][m]);
        __syncthreads();
        v4f ch[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) ch[q] = v4f{0, 0, 0, 0};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const v4s bq = *reinterpret_cast<const v4s*>(sR2 + bcol(q, lo) * ST + 16 * ks + 4 * hq);
                ch[q] = mma(Mfr[ks], bq, ch[q]);
            }
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            unsigned o[4];
#pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) {
                unsigned h2v[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int e = 2 * e2 + j;
                    const float tot = bf2f(static_cast<uint16_t>(elem(dki[s][m], e))) + dkacc[s][m][e] +
                                      ch[2 * m + (e >> 2)][e & 3];
                    h2v[j] = f2bf(tot);
                }
                o[e2] = pack2(h2v[0], h2v[1]);
            }
            *reinterpret_cast<uint4*>(dkout + s * 64 + 8 * (hq + 4 * m)) = make_uint4(o[0], o[1], o[2], o[3]);
        }
        if (s + 1 < NKS) __syncthreads();
    }
    __syncthreads();
    // reverse in-chunk cumsum of dg (one wave, lane = token)
    if (wave == 0) {
        float v = sDg[lane];
#pragma unroll
        for (int off = 1; off < 64; off <<= 1) {
            const float t = __shfl_down(v, off);
            if (lane + off < 64) v += t;
        }
        gdg[(tokbase + lane) * HV + hv] = v;
    }
}

// Group sum of per-v-head dq / dk to the H key heads: out[b,s,hk] = sum_g in[b,s,hk*G+g].  Fixed
// summation order (g ascending, fp32), no atomics.  grid.y selects dq or dk.
__global__ void __launch_bounds__(256)
gdn_bwd_groupsum_kernel(const uint16_t* __restrict__ in0, const uint16_t* __restrict__ in1,
                        uint16_t* __restrict__ out0, uint16_t* __restrict__ out1, size_t nchunks,
                        int H, int HV, int K) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (idx >= nchunks) return;
    const uint16_t* in = blockIdx.y == 0 ? in0 : in1;
    uint16_t* out = blockIdx.y == 0 ? out0 : out1;
    const int cpr = K / 8;  // 16B chunks per row
    const int c = static_cast<int>(idx % cpr);
    const size_t rest = idx / cpr;
    const int hk = static_cast<int>(rest % H);
    const size_t t = rest / H;
    const int G = HV / H;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < G; ++g) {
        const uint4 q = *reinterpret_cast<const uint4*>(in + (t * HV + hk * G + g) * K + 8 * c);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] += bf2f(static_cast<uint16_t>(elem(q, e)));
    }
    unsigned o[4];
#pragma unroll
    for (int e2 = 0; e2 < 4; ++e2) o[e2] = pack2(f2bf(acc[2 * e2]), f2bf(acc[2 * e2 + 1]));
    *reinterpret_cast<uint4*>(out + (t * H + hk) * K + 8 * c) = make_uint4(o[0], o[1], o[2], o[3]);
}

bool shape_ok(int B, int S, int H, int HV, int K, int V) {
    return B > 0 && S > 0 && S % 64 == 0 && H > 0 && HV > 0 && HV % H == 0 && (K == 64 || K == 128) &&
           (V == 64 || V == 128);
}
}  // namespace

extern "C" int flash_train_gdn_bwd_wy_bf16(const FlashTrainGDNBwdWyArgs* a, void* stream) {
    if (a == nullptr || a->k == nullptr || a->v == nullptr || a->A == nullptr || a->dw == nullptr ||
        a->dv2 == nullptr || a->dk_in == nullptr || a->beta == nullptr || a->gc == nullptr ||
        a->dg_planes == nullptr || a->dk == nullptr || a->dv == nullptr || a->dg == nullptr ||
        a->dbeta == nullptr)
        return 1;
    if (!shape_ok(a->B, a->S, a->H, a->HV, a->K, a->V)) return 2;
    const long long blocks = static_cast<long long>(a->B) * (a->S / 64) * a->HV;
    if (blocks > 0x7fffffffLL) return 2;
    const int planes = flash_train_gdn_bwd_dqkwg_dg_planes(a->K);
    hipStream_t st = static_cast<hipStream_t>(stream);
#define FT_WY(K_, V_)                                                                              \
    hipLaunchKernelGGL((gdn_bwd_wy_kernel<K_, V_>), dim3(static_cast<unsigned>(blocks)), dim3(256), \
                       0, st, a->k, a->v, a->A, a->dw, a->dv2, a->dk_in, a->beta, a->gc,           \
                       a->dg_planes, a->dk, a->dv, a->dg, a->dbeta, a->B, a->S, a->H, a->HV, planes)
    if (a->K == 128 && a->V == 128) FT_WY(128, 128);
    else if (a->K == 128) FT_WY(128, 64);
    else if (a->V == 128) FT_WY(64, 128);
    else FT_WY(64, 64);
#undef FT_WY
    return hipGetLastError() == hipSuccess ? 0 : 5;
}

extern "C" int flash_train_gdn_bwd_groupsum_bf16(const FlashTrainGDNBwdGroupSumArgs* a, void* stream) {
    if (a == nullptr || a->dq_in == nullptr || a->dk_in == nullptr || a->dq == nullptr || a->dk == nullptr)
        return 1;
    if (a->B <= 0 || a->S <= 0 || a->H <= 0 || a->HV % a->H != 0 || a->K % 8 != 0) return 2;
    const size_t nchunks = static_cast<size_t>(a->B) * a->S * a->H * (a->K / 8);
    const unsigned grid = static_cast<unsigned>((nchunks + 255) / 256);
    hipLaunchKernelGGL(gdn_bwd_groupsum_kernel, dim3(grid, 2), dim3(256), 0,
                       static_cast<hipStream_t>(stream), a->dq_in, a->dk_in, a->dq, a->dk, nchunks, a->H,
                       a->HV, a->K);
    return hipGetLastError() == hipSuccess ? 0 : 5;
}
