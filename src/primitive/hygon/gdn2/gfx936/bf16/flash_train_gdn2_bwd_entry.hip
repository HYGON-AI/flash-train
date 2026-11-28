// Top-level GDN-2 chunked backward (chunk = 64): R1 intra recompute -> R2 state_h recompute -> B1 dAv -> B2 dhu ->
// B3 wy_dqkg -> B4 intra_bwd (+ GQA group sum).  Everything is recomputed from the forward inputs (no forward ctx),
// so the entry takes the forward's inputs and the output gradient.  See flash_train_gdn2_bwd.h for the contract.
//
// Workspace (caller provided, same convention as flash_train_gdn2_chunk_forward_bf16):
//   forward part   w, u, kg, qg, aqk, h, v_new, gclast          (identical to the forward route)
//   backward part  akk_inv, dAqk, dAkk, dv_local (aliases u), dh, dv2, partial dq,dk,db,dg (fp32 per v-head, lane order),
//                  and for GQA the fp32 per-v-head dq,dk
// The B3 partials are stored in lane order (1 KB contiguous stores, read back by B4 with 256 B segments); dv, dw are
// written straight to the outputs.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../common/flash_train_gdn2_bwd.h"
#include "../common/flash_train_gdn2_chunk.h"

namespace {

constexpr int kC = 64;
constexpr int kMinS = 512;

std::size_t align256(std::size_t x) { return (x + 255) / 256 * 256; }

struct Ws {
    std::size_t w, u, kg, qg, aqk, h, vn, gl, akk, dAqk, dAkk, dh, dv2, pdq, pdk, pdb, pdg, gq, gk, total;
};

Ws layout(int B, int S, int H, int HV, int K, int V) {
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
    l.akk = take(rows * kC * 2);
    l.dAqk = take(rows * kC * 4);
    l.dAkk = take(rows * kC * 4);
    l.dh = take(static_cast<std::size_t>(B) * NT * HV * V * K * 2);
    l.dv2 = take(rows * V * 2);
    l.pdq = take(rows * K * 4);
    l.pdk = take(rows * K * 4);
    l.pdb = take(rows * K * 4);
    l.pdg = take(rows * K * 4);
    if (HV != H) {  // fp32 per-v-head dq,dk of the GQA group sum
        l.gq = take(rows * K * 4);
        l.gk = take(rows * K * 4);
    }
    l.total = off;
    return l;
}

bool kv_ok(int K, int V) { return (K == 64 || K == 128) && (V == 64 || V == 128); }

}  // namespace

extern "C" int flash_train_gdn2_chunk_backward_shape_supported(int S, int H, int HV, int K, int V) {
    return flash_train_gdn2_chunk_shape_supported(S, H, HV, K, V);
}

extern "C" size_t flash_train_gdn2_chunk_backward_workspace_bytes(int B, int S, int H, int HV, int K, int V) {
    if (B <= 0 || !flash_train_gdn2_chunk_backward_shape_supported(S, H, HV, K, V)) return 0;
    return layout(B, S, H, HV, K, V).total;
}

extern "C" int flash_train_gdn2_chunk_backward_bf16(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* g, const float* b, const float* wg,
    const uint16_t* do_, const float* dht, uint16_t* dq, uint16_t* dk, uint16_t* dv, float* dg, uint16_t* db,
    uint16_t* dw, float* dh0, int B, int S, int H, int HV, int K, int V, float scale, void* workspace,
    size_t workspace_bytes, void* stream) {
    if (!q || !k || !v || !g || !b || !wg || !do_ || !dq || !dk || !dv || !dg || !db || !dw || !workspace) return 1;
    if (B <= 0 || !flash_train_gdn2_chunk_backward_shape_supported(S, H, HV, K, V)) return 2;
    const Ws l = layout(B, S, H, HV, K, V);
    if (workspace_bytes < l.total) return 4;
    char* ws = static_cast<char*>(workspace);
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
