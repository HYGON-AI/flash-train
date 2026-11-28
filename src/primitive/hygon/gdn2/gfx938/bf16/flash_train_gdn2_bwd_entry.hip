// Top-level GDN-2 chunked backward (chunk = 64): R1 intra recompute -> R2 state_h recompute -> B1 dAv -> B2 dhu ->
// B3 wy_dqkg -> B4 intra_bwd (+ GQA group sum).  Everything is recomputed from the forward inputs (no forward ctx),
// so the entry takes the forward's inputs and the output gradient.  See flash_train_gdn2_bwd.h for the contract.
//
// Shapes: S % 64 == 0 with K in {64,128}, V in {64,128,256} run the six stages directly; any other shape (S >= 1, K <= 128, V <= 256) is
// zero-padded to that form first and the gradients are cropped afterwards (one extra launch each).
//
// Workspace (caller provided, same convention as flash_train_gdn2_chunk_forward_bf16):
//   forward part   w, u, kg, qg, aqk, h, v_new, gclast          (identical to the forward route)
//   backward part  akk_inv, dAqk, dAkk, dv_local (aliases u), dh, dv2, partial dq,dk,db,dg (fp32 per v-head, lane order),
//                  and for GQA the fp32 per-v-head dq,dk
//   pad route      padded copies of the inputs / outputs
// The B3 partials are stored in lane order (1 KB contiguous stores, read back by B4 with 256 B segments); dv, dw are
// written straight to the outputs.
#include "../common/gfx938_device_symbols.h"
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../common/flash_train_gdn2_bwd.h"
#include "../common/flash_train_gdn2_chunk.h"

namespace {

constexpr int kC = 64;

std::size_t align256(std::size_t x) { return (x + 255) / 256 * 256; }

// Backward route: any S >= 1, K <= 128, V <= 256, HV % H == 0.  Shapes with S % 64 != 0 or K,V not in {64,128} are zero-padded
// (tokens at the end, channels at the end; Kp in {64,128}, Vp in {64,128,256}) before the six stages and the gradients are cropped afterwards.  Padded
// tokens carry q=k=v=do=0, g=0, b=0, wg=0: they are exact no-ops of the recurrence in both directions (the state and
// the state gradient pass through unchanged, dht flows through them), padded channels do not mix into real ones.
struct Plan {
    int Sp, Kp, Vp;
    bool pad;
};

Plan make_plan(int S, int K, int V) {
    Plan p;
    p.Sp = (S + kC - 1) / kC * kC;
    p.Kp = K <= 64 ? 64 : 128;
    p.Vp = V <= 64 ? 64 : (V <= 128 ? 128 : 256);
    p.pad = p.Sp != S || p.Kp != K || p.Vp != V;
    return p;
}

bool dims_ok(int B, int S, int H, int HV, int K, int V) {
    return B > 0 && S > 0 && H > 0 && HV > 0 && HV % H == 0 && K > 0 && K <= 128 && V > 0 && V <= 256;
}

struct Ws {
    std::size_t w, u, kg, qg, aqk, h, vn, gl, akk, dAqk, dAkk, dh, dv2, pdq, pdk, pdb, pdg, gq, gk;
    // pad route only: padded copies of the inputs and of the outputs
    std::size_t qp, kp, vp, dop, gp, bp, wp, dhtp, dqp, dkp, dvp, dwp, dbp, dgp, dh0p;
    std::size_t total;
};

Ws layout(int B, int S, int H, int HV, int K, int V) {
    const Plan pl = make_plan(S, K, V);
    const int Sp = pl.Sp, Kp = pl.Kp, Vp = pl.Vp;
    const std::size_t rows = static_cast<std::size_t>(B) * Sp * HV;
    const std::size_t NT = Sp / kC;
    Ws l{};
    std::size_t off = 0;
    auto take = [&](std::size_t bytes) { std::size_t o = off; off += align256(bytes); return o; };
    l.w = take(rows * Kp * 2);
    l.u = take(rows * Vp * 2);
    l.kg = take(rows * Kp * 2);
    l.qg = take(rows * Kp * 2);
    l.aqk = take(rows * kC * 2);
    l.h = take(static_cast<std::size_t>(B) * NT * HV * Vp * Kp * 2);
    l.vn = take(rows * Vp * 2);
    l.gl = take(static_cast<std::size_t>(B) * NT * HV * Kp * 4);
    l.akk = take(rows * kC * 2);
    l.dAqk = take(rows * kC * 4);
    l.dAkk = take(rows * kC * 4);
    l.dh = take(static_cast<std::size_t>(B) * NT * HV * Vp * Kp * 2);
    l.dv2 = take(rows * Vp * 2);
    l.pdq = take(rows * Kp * 4);
    l.pdk = take(rows * Kp * 4);
    l.pdb = take(rows * Kp * 4);
    l.pdg = take(rows * Kp * 4);
    if (HV != H) {  // fp32 per-v-head dq,dk of the GQA group sum
        l.gq = take(rows * Kp * 4);
        l.gk = take(rows * Kp * 4);
    }
    if (pl.pad) {
        const std::size_t qrows = static_cast<std::size_t>(B) * Sp * H;
        const std::size_t st = static_cast<std::size_t>(B) * HV * Vp * Kp * 4;
        l.qp = take(qrows * Kp * 2);
        l.kp = take(qrows * Kp * 2);
        l.vp = take(rows * Vp * 2);
        l.dop = take(rows * Vp * 2);
        l.gp = take(rows * Kp * 4);
        l.bp = take(rows * Kp * 4);
        l.wp = take(rows * Vp * 4);
        l.dhtp = take(st);
        l.dqp = take(qrows * Kp * 2);
        l.dkp = take(qrows * Kp * 2);
        l.dvp = take(rows * Vp * 2);
        l.dwp = take(rows * Vp * 2);
        l.dbp = take(rows * Kp * 2);
        l.dgp = take(rows * Kp * 4);
        l.dh0p = take(st);
    }
    l.total = off;
    return l;
}

// ---------------------------------------------------------------------------- pad / crop
// One launch copies up to kMaxCopy tensors (tensor id = blockIdx.y).  Every tensor is viewed as [B][n1][n2][n3] with a
// source extent (s1,s2,s3) and a destination extent (d1,d2,d3): dst[b][i1][i2][i3] = src[..] when i is inside the
// source extent, else 0.  Pad: destination extent >= source extent; crop: destination <= source, nothing is zeroed.
constexpr int kMaxCopy = 8;
struct CopyArgs {
    const void* src[kMaxCopy];
    void* dst[kMaxCopy];
    int s1[kMaxCopy], s2[kMaxCopy], s3[kMaxCopy], d1[kMaxCopy], d2[kMaxCopy], d3[kMaxCopy], esz[kMaxCopy];
    int B;
};

__global__ __launch_bounds__(256) void gdn2_bwd_copy_kernel(CopyArgs a) {
    const int t = blockIdx.y;
    const int d1 = a.d1[t], d2 = a.d2[t], d3 = a.d3[t], s1 = a.s1[t], s2 = a.s2[t], s3 = a.s3[t];
    const std::size_t total = static_cast<std::size_t>(a.B) * d1 * d2 * d3;
    const bool b16 = a.esz[t] == 2;
    for (std::size_t e = static_cast<std::size_t>(blockIdx.x) * 256 + threadIdx.x; e < total;
         e += static_cast<std::size_t>(gridDim.x) * 256) {
        const int i3 = static_cast<int>(e % d3);
        std::size_t r = e / d3;
        const int i2 = static_cast<int>(r % d2);
        r /= d2;
        const int i1 = static_cast<int>(r % d1);
        const std::size_t b = r / d1;
        const bool in = i1 < s1 && i2 < s2 && i3 < s3;
        const std::size_t so = ((b * s1 + i1) * s2 + i2) * s3 + i3;
        if (b16)
            static_cast<uint16_t*>(a.dst[t])[e] = in ? static_cast<const uint16_t*>(a.src[t])[so] : static_cast<uint16_t>(0);
        else
            static_cast<float*>(a.dst[t])[e] = in ? static_cast<const float*>(a.src[t])[so] : 0.0f;
    }
}

struct CopyBuilder {
    CopyArgs a{};
    int n = 0;
    std::size_t mx = 0;
    void add(const void* src, void* dst, int B, int s1, int s2, int s3, int d1, int d2, int d3, int esz) {
        a.src[n] = src; a.dst[n] = dst;
        a.s1[n] = s1; a.s2[n] = s2; a.s3[n] = s3; a.d1[n] = d1; a.d2[n] = d2; a.d3[n] = d3; a.esz[n] = esz;
        a.B = B;
        const std::size_t tot = static_cast<std::size_t>(B) * d1 * d2 * d3;
        mx = tot > mx ? tot : mx;
        ++n;
    }
    int launch(hipStream_t st) const {
        std::size_t blocks = (mx + 255) / 256;
        if (blocks > 4096) blocks = 4096;
        hipLaunchKernelGGL(gdn2_bwd_copy_kernel, dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(n)), dim3(256), 0,
                           st, a);
        return hipGetLastError() == hipSuccess ? 0 : 5;
    }
};

// The six stages on a shape with S % 64 == 0, K in {64,128}, V in {64,128,256}.
int run_stages(const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* g, const float* b, const float* wg,
               const uint16_t* do_, const float* dht, uint16_t* dq, uint16_t* dk, uint16_t* dv, float* dg, uint16_t* db,
               uint16_t* dw, float* dh0, int B, int S, int H, int HV, int K, int V, float scale, char* ws, const Ws& l,
               void* stream) {
    auto U16 = [&](std::size_t o) { return reinterpret_cast<uint16_t*>(ws + o); };
    auto F32 = [&](std::size_t o) { return reinterpret_cast<float*>(ws + o); };

    // R1: forward intra with the optional Akk_inv output
    FlashTrainGDN2IntraArgs ia{};
    ia.q = q; ia.k = k; ia.v = v; ia.g = g; ia.b = b; ia.wg = wg;
    ia.w = U16(l.w); ia.u = U16(l.u); ia.kg = U16(l.kg); ia.qg = U16(l.qg); ia.aqk = U16(l.aqk);
    ia.gclast = F32(l.gl); ia.gc = nullptr; ia.akk_inv = U16(l.akk);
    ia.B = B; ia.S = S; ia.H = H; ia.HV = HV; ia.K = K; ia.V = V; ia.scale = scale;
    int rc = flash_train_gdn2_intra_bf16(&ia, stream);
    if (rc) return rc;
    // R2: state_h recompute
    FlashTrainGDN2StateHArgs sa{};
    sa.kg = ia.kg; sa.w = ia.w; sa.u = ia.u; sa.gclast = ia.gclast; sa.h = U16(l.h); sa.v_new = U16(l.vn);
    sa.B = B; sa.S = S; sa.H = H; sa.HV = HV; sa.K = K; sa.V = V;
    rc = flash_train_gdn2_state_h_bf16(&sa, stream);
    if (rc) return rc;
    // B1: dAqk, dv_local (u is dead after R2, dv_local takes its place)
    uint16_t* dv_local = U16(l.u);
    FlashTrainGDN2BwdDAvArgs va{};
    va.v_new = sa.v_new; va.do_ = do_; va.aqk = ia.aqk; va.dAqk = F32(l.dAqk); va.dv_local = dv_local;
    va.B = B; va.S = S; va.HV = HV; va.V = V; va.scale = scale;
    rc = flash_train_gdn2_bwd_dav_bf16(&va, stream);
    if (rc) return rc;
    // B2: dh (serial over chunks), dv2, dh0
    FlashTrainGDN2BwdDhuArgs da{};
    da.qg = ia.qg; da.kg = ia.kg; da.w = ia.w; da.do_ = do_; da.dv_local = dv_local; da.gclast = ia.gclast;
    da.dht = dht; da.dh = U16(l.dh); da.dv2 = U16(l.dv2); da.dh0 = dh0;
    da.B = B; da.S = S; da.H = H; da.HV = HV; da.K = K; da.V = V; da.scale = scale; da.dh_vk = 1;
    rc = flash_train_gdn2_bwd_dhu_bf16(&da, stream);
    if (rc) return rc;
    // B3: partial dq,dk,db,dg (dg straight into the output buffer), final dv, dw, dAkk
    FlashTrainGDN2BwdWyDqkgArgs wa{};
    wa.q = q; wa.k = k; wa.v = v; wa.v_new = sa.v_new; wa.do_ = do_; wa.dv2 = da.dv2; wa.h = sa.h; wa.dh = da.dh;
    wa.akk_inv = ia.akk_inv; wa.g = g; wa.b = b; wa.wg = wg;
    wa.dq = F32(l.pdq); wa.dk = F32(l.pdk); wa.db = F32(l.pdb); wa.dg = F32(l.pdg); wa.dv = dv; wa.dw = dw; wa.dAkk = F32(l.dAkk);
    wa.partial_lane_order = 1;
    wa.B = B; wa.S = S; wa.H = H; wa.HV = HV; wa.K = K; wa.V = V; wa.scale = scale;
    rc = flash_train_gdn2_bwd_wy_dqkg_bf16(&wa, stream);
    if (rc) return rc;
    // B4: intra backward + reverse cumsum + GQA group sum + casts
    FlashTrainGDN2BwdIntraArgs ba{};
    ba.q = q; ba.k = k; ba.g = g; ba.b = b; ba.dAqk = va.dAqk; ba.dAkk = wa.dAkk;
    ba.dq_in = wa.dq; ba.dk_in = wa.dk; ba.db_in = wa.db; ba.dg_in = wa.dg;
    ba.partials_lane_order = 1;
    if (HV != H) { ba.dq_f32 = F32(l.gq); ba.dk_f32 = F32(l.gk); }
    ba.dq = dq; ba.dk = dk; ba.db = db; ba.dg = dg;
    ba.B = B; ba.S = S; ba.H = H; ba.HV = HV; ba.K = K;
    return flash_train_gdn2_bwd_intra_bf16(&ba, stream);
}

}  // namespace

extern "C" int flash_train_gdn2_chunk_backward_shape_supported(int S, int H, int HV, int K, int V) {
    return dims_ok(1, S, H, HV, K, V) ? 1 : 0;
}

extern "C" int flash_train_gdn2_chunk_backward_needs_pad(int S, int K, int V) {
    return make_plan(S, K, V).pad ? 1 : 0;
}

extern "C" size_t flash_train_gdn2_chunk_backward_workspace_bytes(int B, int S, int H, int HV, int K, int V) {
    if (!dims_ok(B, S, H, HV, K, V)) return 0;
    return layout(B, S, H, HV, K, V).total;
}

extern "C" int flash_train_gdn2_chunk_backward_bf16(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* g, const float* b, const float* wg,
    const uint16_t* do_, const float* dht, uint16_t* dq, uint16_t* dk, uint16_t* dv, float* dg, uint16_t* db,
    uint16_t* dw, float* dh0, int B, int S, int H, int HV, int K, int V, float scale, void* workspace,
    size_t workspace_bytes, void* stream) {
    if (!q || !k || !v || !g || !b || !wg || !do_ || !dq || !dk || !dv || !dg || !db || !dw || !workspace) return 1;
    if (!dims_ok(B, S, H, HV, K, V)) return 2;
    const Plan pl = make_plan(S, K, V);
    const Ws l = layout(B, S, H, HV, K, V);
    if (workspace_bytes < l.total) return 4;
    char* ws = static_cast<char*>(workspace);
    if (!pl.pad)
        return run_stages(q, k, v, g, b, wg, do_, dht, dq, dk, dv, dg, db, dw, dh0, B, S, H, HV, K, V, scale, ws, l, stream);

    hipStream_t st = static_cast<hipStream_t>(stream);
    const int Sp = pl.Sp, Kp = pl.Kp, Vp = pl.Vp;
    auto U16 = [&](std::size_t o) { return reinterpret_cast<uint16_t*>(ws + o); };
    auto F32 = [&](std::size_t o) { return reinterpret_cast<float*>(ws + o); };
    CopyBuilder pad;
    pad.add(q, U16(l.qp), B, S, H, K, Sp, H, Kp, 2);
    pad.add(k, U16(l.kp), B, S, H, K, Sp, H, Kp, 2);
    pad.add(v, U16(l.vp), B, S, HV, V, Sp, HV, Vp, 2);
    pad.add(do_, U16(l.dop), B, S, HV, V, Sp, HV, Vp, 2);
    pad.add(g, F32(l.gp), B, S, HV, K, Sp, HV, Kp, 4);
    pad.add(b, F32(l.bp), B, S, HV, K, Sp, HV, Kp, 4);
    pad.add(wg, F32(l.wp), B, S, HV, V, Sp, HV, Vp, 4);
    if (dht) pad.add(dht, F32(l.dhtp), B, HV, V, K, HV, Vp, Kp, 4);  // [B,HV,V,K] state gradient
    int rc = pad.launch(st);
    if (rc) return rc;
    rc = run_stages(U16(l.qp), U16(l.kp), U16(l.vp), F32(l.gp), F32(l.bp), F32(l.wp), U16(l.dop), dht ? F32(l.dhtp) : nullptr,
                    U16(l.dqp), U16(l.dkp), U16(l.dvp), F32(l.dgp), U16(l.dbp), U16(l.dwp), dh0 ? F32(l.dh0p) : nullptr, B, Sp,
                    H, HV, Kp, Vp, scale, ws, l, stream);
    if (rc) return rc;
    CopyBuilder crop;
    crop.add(U16(l.dqp), dq, B, Sp, H, Kp, S, H, K, 2);
    crop.add(U16(l.dkp), dk, B, Sp, H, Kp, S, H, K, 2);
    crop.add(U16(l.dvp), dv, B, Sp, HV, Vp, S, HV, V, 2);
    crop.add(U16(l.dwp), dw, B, Sp, HV, Vp, S, HV, V, 2);
    crop.add(U16(l.dbp), db, B, Sp, HV, Kp, S, HV, K, 2);
    crop.add(F32(l.dgp), dg, B, Sp, HV, Kp, S, HV, K, 4);
    if (dh0) crop.add(F32(l.dh0p), dh0, B, HV, Vp, Kp, HV, V, K, 4);
    return crop.launch(st);
}
