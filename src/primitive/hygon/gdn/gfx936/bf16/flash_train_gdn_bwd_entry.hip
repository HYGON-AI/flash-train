// GDN backward entry: orchestrates the stage kernels (see flash_train_gdn_bwd.h and
// docs/GDN_backward_design_20261027.md).  Lives in gdn_bwd, which links gdn_chunk, so the forward
// library does not depend on any backward code and the applicability guard is the forward
// route query itself.
#include <hip/hip_runtime.h>

#include "../common/flash_train_gdn_bwd.h"
#include "../common/flash_train_gdn_bwd_entry.h"
#include "../common/flash_train_gdn_chunk_parallel.h"
#include "../common/flash_train_gdn_state_h.h"

namespace {

constexpr size_t al(size_t x) { return (x + 255) & ~size_t(255); }

struct Layout {
    size_t gc, w, u, vn, A, h, dh, dv2, pl, dqhv, dkhv, total;
};

Layout layout(const FlashTrainGDNChunkShape& s) {
    const size_t tok = static_cast<size_t>(s.B) * s.S * s.HV;
    const size_t nt = static_cast<size_t>(s.B) * (s.S / 64) * s.HV;
    const size_t planes = flash_train_gdn_bwd_dqkwg_dg_planes(s.K);
    const bool gqa = s.HV != s.H;
    Layout l{};
    size_t off = 0;
    auto put = [&](size_t& dst, size_t bytes) { dst = off; off += al(bytes); };
    put(l.gc, tok * 4);
    put(l.w, tok * s.K * 2);      // later reused as dw
    put(l.u, tok * s.V * 2);      // later reused as dv_local
    put(l.vn, tok * s.V * 2);
    put(l.A, tok * 64 * 2);
    put(l.h, nt * s.K * s.V * 2);
    put(l.dh, nt * s.K * s.V * 2);
    put(l.dv2, tok * s.V * 2);
    put(l.pl, planes * tok * 4);
    put(l.dqhv, gqa ? tok * s.K * 2 : 0);
    put(l.dkhv, gqa ? tok * s.K * 2 : 0);
    l.total = off;
    return l;
}

bool backward_ok(const FlashTrainGDNChunkShape& s) {
    return s.B > 0 && s.H > 0 && s.HV > 0 && s.HV % s.H == 0 &&
           s.S >= 512 && s.S % 64 == 0 && (s.K == 64 || s.K == 128) &&
           (s.V == 64 || s.V == 128) && (s.chunk_size == 0 || s.chunk_size == 32);
}

}  // namespace

extern "C" int flash_train_gdn_chunk_backward_supported(FlashTrainGDNChunkShape s) {
    return backward_ok(s) ? 1 : 0;
}

extern "C" size_t flash_train_gdn_chunk_backward_workspace_bytes(FlashTrainGDNChunkShape s) {
    return backward_ok(s) ? layout(s).total : 0;
}

extern "C" int flash_train_gdn_chunk_bf16_backward_launch(const FlashTrainGDNChunkBackwardArgs* a,
                                                          void* stream) {
    if (a == nullptr || a->q == nullptr || a->k == nullptr || a->v == nullptr || a->g == nullptr ||
        a->beta == nullptr || a->do_ == nullptr || a->dq == nullptr || a->dk == nullptr ||
        a->dv == nullptr || a->dg == nullptr || a->dbeta == nullptr) {
        return FLASH_TRAIN_GDN_CHUNK_INVALID_POINTER;
    }
    const FlashTrainGDNChunkShape s = a->shape;
    if (!backward_ok(s)) return FLASH_TRAIN_GDN_CHUNK_UNSUPPORTED_CONTRACT;
    const Layout l = layout(s);
    if (a->workspace == nullptr) return FLASH_TRAIN_GDN_CHUNK_INVALID_POINTER;
    if (a->workspace_bytes < l.total) return FLASH_TRAIN_GDN_CHUNK_INSUFFICIENT_WORKSPACE;
    char* p = static_cast<char*>(a->workspace);
    float* gc = reinterpret_cast<float*>(p + l.gc);
    uint16_t* w = reinterpret_cast<uint16_t*>(p + l.w);
    uint16_t* u = reinterpret_cast<uint16_t*>(p + l.u);
    uint16_t* vn = reinterpret_cast<uint16_t*>(p + l.vn);
    uint16_t* A = reinterpret_cast<uint16_t*>(p + l.A);
    uint16_t* h = reinterpret_cast<uint16_t*>(p + l.h);
    uint16_t* dh = reinterpret_cast<uint16_t*>(p + l.dh);
    uint16_t* dv2 = reinterpret_cast<uint16_t*>(p + l.dv2);
    float* pl = reinterpret_cast<float*>(p + l.pl);
    const bool gqa = s.HV != s.H;
    uint16_t* dq_hv = gqa ? reinterpret_cast<uint16_t*>(p + l.dqhv) : a->dq;
    uint16_t* dk_hv = gqa ? reinterpret_cast<uint16_t*>(p + l.dkhv) : a->dk;
    uint16_t* dw = w;   // w is dead once B2 has run
    uint16_t* dvl = u;  // u is dead once R2 has run
    void* st = stream;
    int rc;

    FlashTrainGDNIntraArgs ia{a->k, a->v, a->g, a->beta, w, u, gc, s.B, s.S, s.H, s.HV, s.K, s.V, A};
    if ((rc = flash_train_gdn_intra_bf16(&ia, st)) != 0) return rc == 1 ? 1 : FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    FlashTrainGDNStateHArgs ha{a->k, w, u, gc, a->h0, h, vn, nullptr, s.B, s.S, s.H, s.HV, s.K, s.V, 0, 1};
    if (flash_train_gdn_state_h_bf16(&ha, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    FlashTrainGDNBwdDvLocalArgs va{a->q, a->k, a->do_, gc, dvl, s.B, s.S, s.H, s.HV, s.K, s.V, a->scale};
    if (flash_train_gdn_bwd_dv_local_bf16(&va, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    FlashTrainGDNBwdDhuArgs da{a->q, a->k, w, a->do_, dvl, gc, a->dht, dh, dv2, a->dh0,
                               s.B, s.S, s.H, s.HV, s.K, s.V, a->scale, 1};
    if (flash_train_gdn_bwd_dhu_bf16(&da, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    FlashTrainGDNBwdDqkwgArgs qa{a->q, a->k, vn, a->do_, dv2, h, dh, gc, dq_hv, dk_hv, dw, pl,
                                 s.B, s.S, s.H, s.HV, s.K, s.V, a->scale};
    if (flash_train_gdn_bwd_dqkwg_bf16(&qa, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    FlashTrainGDNBwdWyArgs wa{a->k, a->v, A, dw, dv2, dk_hv, a->beta, gc, pl, dk_hv, a->dv, a->dg,
                              a->dbeta, s.B, s.S, s.H, s.HV, s.K, s.V};
    if (flash_train_gdn_bwd_wy_bf16(&wa, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    if (gqa) {
        FlashTrainGDNBwdGroupSumArgs ga{dq_hv, dk_hv, a->dq, a->dk, s.B, s.S, s.H, s.HV, s.K};
        if (flash_train_gdn_bwd_groupsum_bf16(&ga, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    }
    return FLASH_TRAIN_GDN_CHUNK_SUCCESS;
}
