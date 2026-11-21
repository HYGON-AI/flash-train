// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// F1 (PLAN_FLASHTRAIN.md): the head of the G2 pipeline in HIP, replacing the FLA Triton
// scaffolds on the default path.  Same formulas as FLA; reduction / scan ORDER differs, so
// the outputs are not bitwise FLA (FP64-adjudicated, like every G2 stage).
//   g2_l2n2_kernel     l2norm_fwd for q AND k (grid.y): rstd = 1/sqrt(sum(x^2) + 1e-6),
//                      y = bf16(x * rstd); 16 lanes x 16 B per token row, fixed shfl tree.
//   g2_gate_kernel     chunk-local cumsum (x scale after the scan) of
//                        GATE=1: lower_bound * sigmoid(exp(A_log) * (g + dt_bias))   (kda_gate_chunk_cumsum)
//                        GATE=0: g as given                                           (chunk_local_cumsum)
//                      per (chunk, head): lane c owns columns 2c, 2c+1; four quarter-CTAs scan
//                      16 rows each serially and add the lower quarters' totals (fixed order).
//                      VL: packed rows, rows r >= clen neither read nor written.
//   g2_beta_sig_kernel fused_beta_sigmoid: y(fp32) = scale * sigmoid(x).
#include "g2_common.h"

namespace g2 {

// 16 lanes per row, 8 bf16 (16 B) per lane, 16 rows per CTA; fixed 4-step shfl tree.
__global__ void __launch_bounds__(256) g2_l2n2_kernel(
        const short* __restrict__ q, const short* __restrict__ k, short* __restrict__ qn,
        short* __restrict__ kn, float* __restrict__ q_rstd, float* __restrict__ k_rstd,
        long nrows) {
    const long row = (long)blockIdx.x * 16 + (threadIdx.x >> 4);
    const int j = threadIdx.x & 15;                               // 8 columns per lane
    if (row >= nrows) return;                                     // whole 16-lane groups
    const bool isk = blockIdx.y != 0;
    const short* x = isk ? k : q;
    short* y = isk ? kn : qn;
    float* rstd = isk ? k_rstd : q_rstd;
    const uint4 raw = *reinterpret_cast<const uint4*>(x + row * 128 + j * 8);
    const short* e = reinterpret_cast<const short*>(&raw);
    float f[8], ss = 0.f;
    #pragma unroll
    for (int u = 0; u < 8; ++u) { f[u] = bf2f(e[u]); ss += f[u] * f[u]; }
    #pragma unroll
    for (int off = 1; off < 16; off <<= 1) ss += __shfl_xor(ss, off, 64);   // within the group
    const float rs = 1.f / sqrtf(ss + 1e-6f);
    if (j == 0) rstd[row] = rs;
    uint4 out;
    short* o = reinterpret_cast<short*>(&out);
    #pragma unroll
    for (int u = 0; u < 8; ++u) o[u] = bf_rne(f[u] * rs);
    *reinterpret_cast<uint4*>(y + row * 128 + j * 8) = out;
}

struct GateParams {
    const void* g_raw;            // GATE=1: bf16, GATE=0: fp32
    const float* A_log; const float* dt_bias;
    float* g_out;
    float lower_bound, scale;
    int H;
    const int* ctok; const int* clen;   // varlen phase 2 (null = dense)
    int NT = 0, order = 0, order_m = 4; // kra P-938o2: CTA visiting order (g2_cta_order)
};

// Lane c (0..63) owns columns 2c, 2c+1; quarter w (0..3) scans rows [16w, 16w+16)
// serially; quarter totals meet in LDS and each quarter adds the lower quarters' totals
// in fixed order (q0 + q1 + ...).
template <int GATE, bool VL>
__global__ void __launch_bounds__(256) g2_gate_kernel(GateParams p) {
    // Z6: the VL and dense instantiations compiled to different FMA contraction choices
    // (identical source, different guards -> different scheduling), giving ~1-2 bf16 ulp
    // between the two instances on full chunks.  Pin contraction off for the whole body
    // so both instantiations emit identical arithmetic.
    #pragma clang fp contract(off)
    __shared__ float2 sTot[4][64];
    int i, hh;
    g2_cta_order(p.order, p.order_m, p.H, p.NT, i, hh);   // kra P-938o2
    const int c = threadIdx.x & 63, w = threadIdx.x >> 6;
    const long t0 = g2_row_tok<VL>(p.ctok, i);
    const int cl = g2_row_len<VL>(p.clen, i);
    const size_t step = (size_t)p.H * 128;
    const size_t base = ((size_t)t0 * p.H + hh) * 128 + 2 * c;
    float A = 0.f, dt0 = 0.f, dt1 = 0.f;
    if constexpr (GATE == 1) {
        A = expf(p.A_log[hh]);
        const float2 d2 = *reinterpret_cast<const float2*>(p.dt_bias + hh * 128 + 2 * c);
        dt0 = d2.x; dt1 = d2.y;
    }
    const int r0 = 16 * w;
    float2 v[16];
    #pragma unroll
    for (int u = 0; u < 16; ++u) {                      // independent loads first (ILP)
        const int r = r0 + u;
        float2 x = float2{0.f, 0.f};
        if (!VL || r < cl) {
            if constexpr (GATE == 1) {
                const short2 s2 = *reinterpret_cast<const short2*>(
                    reinterpret_cast<const short*>(p.g_raw) + base + (size_t)r * step);
                x = float2{bf2f(s2.x) + dt0, bf2f(s2.y) + dt1};
            } else {
                x = *reinterpret_cast<const float2*>(
                    reinterpret_cast<const float*>(p.g_raw) + base + (size_t)r * step);
            }
        }
        v[u] = x;
    }
    // Compensated (Kahan) scan: the prefix error stays O(1) ulp instead of growing with
    // the row index (FLA's tl.cumsum tree is O(log n)); the compensation rides along into
    // the carry so every prefix is the correctly-rounded-ish sum of its gates.
    float2 run = float2{0.f, 0.f}, cmp = float2{0.f, 0.f};
#if G2_SIGFAST
    bool fast = false;                                  // kra P-938sg (gfx938): range-free sigmoid when proven
    if constexpr (GATE == 1 && !KDA_B1_SIG) {           // kra P-938sd: B1 never reads the range test
        float mx = 0.f, sx = 0.f;
        #pragma unroll
        for (int u = 0; u < 16; ++u) { g2_sig_acc(A * v[u].x, mx, sx); g2_sig_acc(A * v[u].y, mx, sx); }
        fast = g2_sig_fast(mx, sx);
    }
#endif
    #pragma unroll
    for (int u = 0; u < 16; ++u) {
        float2 gt = v[u];
        if constexpr (GATE == 1) {
#if G2_SIGFAST
#if KDA_B1_SIG
            if (true) {
                gt.x = p.lower_bound * g2_sig_b1(A * gt.x);
                gt.y = p.lower_bound * g2_sig_b1(A * gt.y);
            } else
#endif
            if (fast) {
                gt.x = p.lower_bound * rcp1(1.f + expf_core(-(A * gt.x)));
                gt.y = p.lower_bound * rcp1(1.f + expf_core(-(A * gt.y)));
            } else
#endif
            {
            gt.x = p.lower_bound * (1.f / (1.f + expf(-(A * gt.x))));
            gt.y = p.lower_bound * (1.f / (1.f + expf(-(A * gt.y))));
            }
        }
        if (VL && r0 + u >= cl) gt = float2{0.f, 0.f};  // rows past the end add nothing
        {
            const float yx = gt.x - cmp.x, tx = run.x + yx;
            cmp.x = (tx - run.x) - yx; run.x = tx;
            const float yy = gt.y - cmp.y, ty = run.y + yy;
            cmp.y = (ty - run.y) - yy; run.y = ty;
        }
        v[u] = float2{run.x, run.y};
    }
    sTot[w][c] = float2{run.x - cmp.x, run.y - cmp.y};   // quarter total, compensation folded
    __syncthreads();
    float2 carry = float2{0.f, 0.f}, cc = float2{0.f, 0.f};
    for (int b = 0; b < w; ++b) {                        // fixed order, compensated
        const float yx = sTot[b][c].x - cc.x, tx = carry.x + yx;
        cc.x = (tx - carry.x) - yx; carry.x = tx;
        const float yy = sTot[b][c].y - cc.y, ty = carry.y + yy;
        cc.y = (ty - carry.y) - yy; carry.y = ty;
    }
    carry = float2{carry.x - cc.x, carry.y - cc.y};
    #pragma unroll
    for (int u = 0; u < 16; ++u) {
        const int r = r0 + u;
        if (!VL || r < cl)
            *reinterpret_cast<float2*>(p.g_out + base + (size_t)r * step) =
                float2{(v[u].x + carry.x) * p.scale, (v[u].y + carry.y) * p.scale};
    }
}

template <typename T>
__global__ void __launch_bounds__(256) g2_beta_sig_kernel(const T* __restrict__ x,
                                                         float* __restrict__ y, float scale,
                                                         long n) {
    const long t = (long)blockIdx.x * 256 + threadIdx.x;
    if (t >= n) return;
    float v;
    if constexpr (sizeof(T) == 2) v = bf2f(reinterpret_cast<const short*>(x)[t]);
    else v = reinterpret_cast<const float*>(x)[t];
    y[t] = scale * (1.f / (1.f + expf(-v)));
}

}  // namespace g2

extern "C" int g2_head_l2n2(const void* q, const void* k, void* qn, void* kn, void* q_rstd,
                            void* k_rstd, long rows, hipStream_t s) {
    if (rows <= 0) return 0;
    hipLaunchKernelGGL(g2::g2_l2n2_kernel, dim3((unsigned)((rows + 15) / 16), 2), dim3(256), 0, s,
                       (const short*)q, (const short*)k, (short*)qn, (short*)kn, (float*)q_rstd,
                       (float*)k_rstd, rows);
    return (int)hipGetLastError();
}

// NT chunks x H heads; dense (ctok == null): chunk i = rows [64i, 64i+64).
extern "C" int g2_head_gate(const void* g_raw, const void* A_log, const void* dt_bias,
                            void* g_out, double lower_bound, double scale, int gate, int NT,
                            int H, const int* ctok, const int* clen, hipStream_t s) {
    if (NT <= 0) return 0;
    if ((ctok == nullptr) != (clen == nullptr)) return -1;
    g2::GateParams p{g_raw, (const float*)A_log, (const float*)dt_bias, (float*)g_out,
                     (float)lower_bound, (float)scale, H, ctok, clen};
    p.NT = NT;
    // kra P-938o2 (gfx938): row-major in and out -> head-fastest at H % 32 == 0 (1.4-2.1x at H = 32 / 64)
    p.order = (int)g2::g2_env_long("KDA_G2_GATE_ORDER", g2::g2_cur_gfx938() && H % 32 == 0 ? 1 : 0);
    p.order_m = (int)g2::g2_env_long("KDA_G2_GATE_ORDER_M", 4);
    if (p.order == 3 && (p.order_m < 1 || H % p.order_m)) p.order = 0;
    const dim3 grid(NT, H), blk(256);
    if (gate) {
        if (ctok) hipLaunchKernelGGL((g2::g2_gate_kernel<1, true>), grid, blk, 0, s, p);
        else hipLaunchKernelGGL((g2::g2_gate_kernel<1, false>), grid, blk, 0, s, p);
    } else {
        if (ctok) hipLaunchKernelGGL((g2::g2_gate_kernel<0, true>), grid, blk, 0, s, p);
        else hipLaunchKernelGGL((g2::g2_gate_kernel<0, false>), grid, blk, 0, s, p);
    }
    return (int)hipGetLastError();
}

extern "C" int g2_head_beta_sigmoid(const void* x, int x_bf16, void* y, double scale, long n,
                                    hipStream_t s) {
    if (n <= 0) return 0;
    const unsigned g = (unsigned)((n + 255) / 256);
    if (x_bf16)
        hipLaunchKernelGGL(g2::g2_beta_sig_kernel<short>, dim3(g), dim3(256), 0, s,
                           (const short*)x, (float*)y, (float)scale, n);
    else
        hipLaunchKernelGGL(g2::g2_beta_sig_kernel<float>, dim3(g), dim3(256), 0, s,
                           (const float*)x, (float*)y, (float)scale, n);
    return (int)hipGetLastError();
}
