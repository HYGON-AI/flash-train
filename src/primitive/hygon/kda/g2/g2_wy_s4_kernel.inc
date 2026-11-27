// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// Vendored from the hip_kda G2 kernels (csrc/g2); kept byte-aligned with upstream.
// clang-format off
// p3_wy_s4 kernel body, included twice by g2_wy.cuh (production step G2_P4_STEP and the
// step-0 reference).  It must stay textually inside the __global__ function: moving the
// same body into an inlined __device__ function template cost the step-0/1 builds 7/32
// VGPR spills (P4', measured) -- hence the include instead of a function template.
//   G2_S4_KNAME : kernel name     G2_S4_KST : P4' step     G2_S4_KWMIN : min waves/EU
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(G2_S4_KWMIN, 3))) G2_S4_KNAME(WyS4Params p) {
    constexpr int ST = G2_S4_KST;
    __shared__ __align__(16) S4Smem sm;
    bf16* const S = sm.S;
    float* const dgk1 = sm.dgk1;
    float* const dgk2 = sm.dgk2;
    const int tid = threadIdx.x, r = tid >> 6, l = tid & 63, m = l & 15, q = l >> 4, pm = pi16(m);
    const int i = blockIdx.x, hh = blockIdx.y, H = p.H;
    const int tr = 16 * r + m;
    const size_t rs = (size_t)H * kD, rsA = (size_t)H * kBT;
    const size_t cb = (size_t)i * kBT * rs + (size_t)hh * kD;
    const size_t cbA = (size_t)i * kBT * rsA + (size_t)hh * kBT;
    const int R = H * kD;
    const int RA = H * kBT;
    const bf16 *k_ = p.k + cb, *v_ = p.v_row + cb;
    const float* g_ = p.g + cb;
    const bf16* A_ = p.A + cbA;
    const float* beta_ = p.beta + (size_t)i * kBT * H + hh;
    float *dq_ = p.dq + cb, *dk_ = p.dk + cb, *dg_ = p.dg + cb;
    bf16* dv2_ = p.dv2 + cb;
    const size_t cA = ((size_t)i * H + hh) * 4 * 8 * 256, cB = ((size_t)i * H + hh) * 8 * 8 * 256;
    const bf16* hb = p.h + cB;
    const bf16* dhb = p.dh + cB;
    const bf16* vnN = p.v_new + cA;
    const bf16* dvN = p.dv + cA;
    const bf16* doN = p.do_ + cA;
    const bf16* vBN = p.v_vb + ((size_t)i * H + hh) * 4 * 8 * 256;
    const bf16* kgB = p.q_kgb + ((size_t)i * H + hh) * 4 * 8 * 256;
    auto aN = [&](const bf16* base, int vt) { return ld8(base + ((r * 8 + vt) * 64 + l) * 4); };
    const int ro = tr * R + 4 * q;
    auto ex2 = [](float x) { return __builtin_amdgcn_exp2f(x); };
    // G2_S4_STOP: dump the live accumulators (x[4] + db + aa) and end the kernel
#define G2_S4_STOP_AT(n_, x_)                                                             \
    if (G2_S4_STOP == (n_)) {                                                             \
        float* o_ = p.dA + cbA + tr * RA + 4 * q;                                         \
        _Pragma("unroll") for (int st_ = 0; st_ < 4; ++st_)                               \
            *reinterpret_cast<f4*>(o_ + 16 * st_) = (x_)[st_] + bf2f(aa[st_][0]) + db;     \
        return;                                                                           \
    }

    if (tid < kD) { dgk1[tid] = 0.f; dgk2[tid] = 0.f; }
    const float beta_t = beta_[tr * H];
    v4bh aa[4];
    #pragma unroll
    for (int st = 0; st < 4; ++st)
        #pragma unroll
        for (int e = 0; e < 4; ++e) aa[st][e] = reinterpret_cast<const short*>(A_)[(16 * st + 4 * q + e) * RA + tr];
    v4bh dva[8];
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt) dva[vt] = aN(dvN, vt);
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) reinterpret_cast<short*>(S)[(16 * vt + 4 * q + e) * kPT + tr] = dva[vt][e];
    lds_barrier();                                                      // dvT
    float db = 0.f;
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt) {
        f4 acc = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st) acc = mmac(aa[st], ld8(S + (16 * vt + pm) * kPT + 16 * st + 4 * q), acc);
        const v4bh vv = ld8(v_ + ro + 16 * vt);
        v4bh o;
        #pragma unroll
        for (int e = 0; e < 4; ++e) { o[e] = bfr(acc[e] * beta_t); db += acc[e] * bf2f(vv[e]); }
        *reinterpret_cast<v4bh*>(dv2_ + ro + 16 * vt) = o;
    }
    f4 dA[4];
    #pragma unroll
    for (int st = 0; st < 4; ++st) dA[st] = f4{0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int vt = 0; vt < 8; ++vt)
        #pragma unroll
        for (int st = 0; st < 4; ++st)
            dA[st] = mmac(dva[vt], ld8(vBN + ((st * 8 + vt) * 64 + l) * 4), dA[st]);
    G2_S4_STOP_AT(1, dA)
    lds_barrier();                                                      // dvT dead -> dwT

    v4bh af[8], bf0[8], bf1[8];
    // pass Q: dq_P3 only (P4' step >= 1: computed inside the tail loop instead)
    if constexpr (ST < 1) {
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) af[vt] = aN(doN, vt);
        auto loadB = [&](int kt, v4bh* bf) {
            const bf16* pb = hb + (kt * 8 * 64 + l) * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) bf[vt] = ld8(pb + vt * 256);
        };
        auto body = [&](int kt, v4bh* bf) {
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(af[vt], bf[vt], a4[vt & 3]);
            const f4 acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            const int kc = 16 * kt;
            const f4 gv = ld16f(g_ + ro + kc);
            f4 dqf;
            #pragma unroll
            for (int e = 0; e < 4; ++e) dqf[e] = acc[e] * ex2(gv[e]) * p.scale;
            // bit 4: timing ablation -- the store is skipped at run time (scale > 0) but
            // stays in the code, so the compute feeding it cannot be dead-code eliminated
            if (!(G2_P4_EXP & 16) || p.scale < 0.f) *reinterpret_cast<f4*>(dq_ + ro + kc) = dqf;
        };
        loadB(0, bf0);
        #pragma unroll 1
        for (int kt = 0; kt < 8; kt += 2) {
            loadB(kt + 1, bf1);
            body(kt, bf0);
            loadB(kt < 6 ? kt + 2 : kt + 1, bf0);
            body(kt + 1, bf1);
        }
    }
    G2_S4_STOP_AT(2, dA)
    // pass K: dk_K = dk_pre, dg_K = -k*dk_pre; column sums; dgk1 partial over v-tiles {2r, 2r+1}
    // (P4' step >= 2: computed inside the tail loop instead)
    if constexpr (ST < 2) {
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) af[vt] = aN(vnN, vt);
        auto loadB = [&](int kt, v4bh* bf) {
            const bf16* pb = dhb + (kt * 8 * 64 + l) * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) bf[vt] = ld8(pb + vt * 256);
        };
        auto body = [&](int kt, v4bh* bf) {
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(af[vt], bf[vt], a4[vt & 3]);
            const f4 acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            const int kc = 16 * kt;
            const f4 gv = ld16f(g_ + ro + kc), gn = ld16f(g_ + (kBT - 1) * R + 4 * q + kc);
            const v4bh kk = ld8(k_ + ro + kc);
            f4 dkp, dg;
            float kd[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                dkp[e] = acc[e] * ex2(gn[e] - gv[e]);
                kd[e] = bf2f(kk[e]) * dkp[e];
                dg[e] = -kd[e];
            }
            if (!(G2_P4_EXP & 16) || p.scale < 0.f) {
                *reinterpret_cast<f4*>(dk_ + ro + kc) = dkp;
                *reinterpret_cast<f4*>(dg_ + ro + kc) = dg;
            }
#if !(G2_P4_EXP & 1)    // P4 timing experiment only: bit 0 drops the dgk2 column sums
            #pragma unroll
            for (int e = 0; e < 4; ++e) { const float c = msum(kd[e]); if (m == 0) atomicAdd(&dgk2[kc + 4 * q + e], c); }
#endif
#if !(G2_P4_EXP & 2)    // bit 1 drops the h.dh (dgk1) term
            float hd = 0.f;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int off = ((kt * 8 + 2 * r + j) * 64 + l) * 4;
                const v4bh h0 = ld8(hb + off), d0 = ld8(dhb + off);
                #pragma unroll
                for (int e = 0; e < 4; ++e) hd += bf2f(h0[e]) * bf2f(d0[e]);
            }
            hd = xsum16(hd);
            if (q == 0) atomicAdd(&dgk1[16 * kt + pm], hd);
#endif
        };
        loadB(0, bf0);
        #pragma unroll 1
        for (int kt = 0; kt < 8; kt += 2) {
            loadB(kt + 1, bf1);
            body(kt, bf0);
            loadB(kt < 6 ? kt + 2 : kt + 1, bf0);
            body(kt + 1, bf1);
        }
    }
    G2_S4_STOP_AT(3, dA)
    // pass W: -dw -> dwT, dA += (-dw) kg^T
    {
        #pragma unroll
        for (int vt = 0; vt < 8; ++vt) af[vt] = aN(dvN, vt);
        auto loadB = [&](int kt, v4bh* bf) {
            const bf16* pb = hb + (kt * 8 * 64 + l) * 4;
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) bf[vt] = ld8(pb + vt * 256);
        };
        auto body = [&](int kt, v4bh* bf) {
            f4 a4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a4[j] = f4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int vt = 0; vt < 8; ++vt) a4[vt & 3] = mmac(af[vt], bf[vt], a4[vt & 3]);
            const f4 acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            const int kc = 16 * kt;
            v4bh dwn;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                dwn[e] = bfr(-acc[e]);
                reinterpret_cast<short*>(S)[(kc + 4 * q + e) * kPT + tr] = dwn[e];
            }
            #pragma unroll
            for (int st = 0; st < 4; ++st) {
                const v4bh kgb = ld8(kgB + ((st * 8 + kt) * 64 + l) * 4);
                dA[st] = mmac(dwn, kgb, dA[st]);
            }
        };
        loadB(0, bf0);
        #pragma unroll 1
        for (int kt = 0; kt < 8; kt += 2) {
            loadB(kt + 1, bf1);
            body(kt, bf0);
            loadB(kt < 6 ? kt + 2 : kt + 1, bf0);
            body(kt + 1, bf1);
        }
    }
    G2_S4_STOP_AT(4, dA)
    // dA stage (moved ahead of the epilogue; XT in its own region, S keeps dwT)
    v4bh da[4];
    #pragma unroll
    for (int st = 0; st < 4; ++st)
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int s = 16 * st + 4 * q + e;
            da[st][e] = bfr(tr > s ? dA[st][e] * beta_[s * H] : 0.f);
        }
    f4 xc[4];
    #pragma unroll
    for (int ut = 0; ut < 4; ++ut) {
        xc[ut] = f4{0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st) xc[ut] = mmac(da[st], ld8(A_ + (16 * ut + pm) * RA + 16 * st + 4 * q), xc[ut]);
    }
    #pragma unroll
    for (int ut = 0; ut < 4; ++ut)
        #pragma unroll
        for (int e = 0; e < 4; ++e) reinterpret_cast<short*>(sm.u.a.XT)[(16 * ut + 4 * q + e) * kPT + tr] = bfr(xc[ut][e]);
    lds_barrier();                                                      // XT, dwT, dgk1/2 complete
    f4 dak[4];
    #pragma unroll
    for (int ut = 0; ut < 4; ++ut) {
        f4 acc = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int st = 0; st < 4; ++st) acc = mmac(aa[st], ld8(sm.u.a.XT + (16 * ut + pm) * kPT + 16 * st + 4 * q), acc);
        #pragma unroll
        for (int e = 0; e < 4; ++e) { const int u = 16 * ut + 4 * q + e; dak[ut][e] = tr > u ? -acc[e] : 0.f; }
        #pragma unroll
        for (int e = 0; e < 4; ++e) sm.u.a.dAT[(16 * ut + 4 * q + e) * ipi::TS + tr] = dak[ut][e];
        if (p.dA) *reinterpret_cast<f4*>(p.dA + cbA + tr * RA + 16 * ut + 4 * q) = dak[ut];
    }
    G2_S4_STOP_AT(5, dak)
    lds_barrier();                                                      // dAT complete
#undef G2_S4_STOP_AT
    switch (__builtin_amdgcn_readfirstlane(r)) {
        case 0: s4_tail<0, ST>(p, sm, aa, dak, db, beta_t); break;
        case 1: s4_tail<1, ST>(p, sm, aa, dak, db, beta_t); break;
        case 2: s4_tail<2, ST>(p, sm, aa, dak, db, beta_t); break;
        default: s4_tail<3, ST>(p, sm, aa, dak, db, beta_t); break;
    }
}

