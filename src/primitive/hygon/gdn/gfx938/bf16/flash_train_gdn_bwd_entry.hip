// GDN backward entry: orchestrates the stage kernels (see flash_train_gdn_bwd.h and
// docs/GDN_backward_design_20261027.md).  Lives in gdn_bwd, which links gdn_chunk; the backward
// recomputes every intermediate itself, so it is independent of the forward route and has its own
// applicability rule (backward_ok below).
//
// Two paths, both on the caller stream and without any host synchronisation:
//   native: S % 64 == 0, K in {64,128}, V in {64,128,256}: the stage kernels run directly on the caller
//           tensors.
//   padded: anything else with 1 <= K,V <= 128 (V = 256 is native only): inputs are zero-padded into
//           workspace buffers up to Sp = ceil(S/64)*64, Kp/Vp in {64,128}, the stage kernels run on the padded shape and the
//           gradients are cropped back to the caller tensors.  Padded tokens have q=k=v=do=0, g=0,
//           beta=0, so S_t = S_{t-1} there: they do not influence real tokens, the final state passes
//           through unchanged and dht flows back unchanged.  Padded K/V columns are zero everywhere
//           (including h0/dht), and scale is the caller value.
// h0 is given as [B,HV,K,V] but the state_h stage (h_vk = 1) reads it V-first, so the entry always
// transposes it into a workspace buffer (inside the pad copy on the padded path, one extra tiny copy
// kernel on the native path) -- only when h0 is present.
//
// Stream placement (dqkwg): that stage streams v_new, do, dv2 (and q, k to a lesser degree) at once, and on
// this hardware two streams whose device addresses are closer than ~2 MiB modulo 16 MiB collide (extra
// latency of up to ~2.2 ms at distance 0, shrinking linearly to zero at ~2 MiB).  Only measured, no
// hardware documentation behind the 16 MiB period or the 2 MiB window.  q, k, do come from the caller (or
// from the workspace on the padded path), so the entry pads the workspace in front of v_new and dv2 so that
// the five residues mod 16 MiB are as far apart as possible (choose_place).  Pure host arithmetic.
#include "../common/gfx938_device_symbols.h"
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../common/flash_train_gdn_bwd.h"
#include "../common/flash_train_gdn_bwd_entry.h"
#include "../common/flash_train_gdn_bwd_pad.h"
#include "../common/flash_train_gdn_chunk_parallel.h"
#include "../common/flash_train_gdn_state_h.h"

namespace {

constexpr size_t al(size_t x) { return (x + 255) & ~size_t(255); }

// Native kernel contract: full 64-token chunks, K in {64,128}, V in {64,128,256}.
bool native_ok(const FlashTrainGDNChunkShape& s) {
    return s.S % 64 == 0 && (s.K == 64 || s.K == 128) && (s.V == 64 || s.V == 128 || s.V == 256);
}

// Backward applicability: any S >= 1, 1 <= K,V <= 128, HV a multiple of H; V = 256 only on the native
// path (the padded path pads V to 64 or 128, never to 256, so V = 192 / 512 / V = 256 with S % 64 != 0
// or K not in {64,128} are unsupported).  chunk_size is the legacy field of the forward shape (0 or 32
// were accepted before; 64 is this backward chunk).
bool backward_ok(const FlashTrainGDNChunkShape& s) {
    const bool common = s.B > 0 && s.S > 0 && s.H > 0 && s.HV > 0 && s.HV % s.H == 0 && s.K >= 1 &&
                        s.K <= 128 && (s.chunk_size == 0 || s.chunk_size == 32 || s.chunk_size == 64);
    return common && ((s.V >= 1 && s.V <= 128) || (s.V == 256 && native_ok(s)));
}

// The shape the stage kernels actually run on.
FlashTrainGDNChunkShape padded_shape(const FlashTrainGDNChunkShape& s) {
    if (native_ok(s)) return s;
    FlashTrainGDNChunkShape p = s;
    p.S = (s.S + 63) / 64 * 64;
    p.K = s.K <= 64 ? 64 : 128;
    p.V = s.V <= 64 ? 64 : 128;
    return p;
}

struct Layout {
    // stage buffers (padded shape)
    size_t gc, w, u, vn, A, h, dh, dv2, pl, dqhv, dkhv;
    // pad path only: padded copies of the inputs and outputs
    size_t iq, ik, iv, ido, ig, ibeta, ih0, idht, oq, ok, ov, og, obeta, odh0;
    size_t total;
};

// Address period and step of the dqkwg stream placement (see the header comment).
constexpr uint64_t kPlacePeriod = uint64_t(16) << 20;
constexpr uint64_t kPlaceStep = uint64_t(1) << 20;

// Placement padding in front of the dqkwg streams, bytes, multiples of kPlaceStep, each <= 15 MiB.
// Native path: only vn and dv2 (q, k, do are the caller's).  Padded path: all five, so the workspace
// copies of q, k, do (an exact multiple of 16 MiB apart without padding) can be spread as well; the
// first stream (vn) keeps pad 0, so that path needs four paddings.
struct PlacePads {
    size_t vn = 0, dv2 = 0, iq = 0, ik = 0, ido = 0;
};

// Workspace slack the paddings may consume: native 2 x 16 MiB, padded 4 x 16 MiB.
constexpr size_t place_slack(bool pad) { return (pad ? 4 : 2) * static_cast<size_t>(kPlacePeriod); }

// pads = {} is the unpadded layout choose_place starts from.
Layout layout(const FlashTrainGDNChunkShape& s, const PlacePads& pads = PlacePads{}){
    const FlashTrainGDNChunkShape p = padded_shape(s);
    const bool pad = !native_ok(s);
    const size_t tok = static_cast<size_t>(p.B) * p.S * p.HV;
    const size_t nt = static_cast<size_t>(p.B) * (p.S / 64) * p.HV;
    const size_t planes = flash_train_gdn_bwd_dqkwg_dg_planes(p.K);
    const bool gqa = p.HV != p.H;
    const size_t tokh = static_cast<size_t>(p.B) * p.S * p.H;  // q/k rows
    const size_t state = static_cast<size_t>(p.B) * p.HV * p.K * p.V;
    Layout l{};
    size_t off = 0;
    auto put = [&](size_t& dst, size_t bytes) { dst = off; off += al(bytes); };
    put(l.gc, tok * 4);
    put(l.w, tok * p.K * 2);      // later reused as dw
    put(l.u, tok * p.V * 2);      // later reused as dv_local
    off += pads.vn;
    put(l.vn, tok * p.V * 2);
    put(l.A, tok * 64 * 2);
    put(l.h, nt * p.K * p.V * 2);
    put(l.dh, nt * p.K * p.V * 2);
    off += pads.dv2;
    put(l.dv2, tok * p.V * 2);
    put(l.pl, planes * tok * 4);
    put(l.dqhv, gqa ? tok * p.K * 2 : 0);
    put(l.dkhv, gqa ? tok * p.K * 2 : 0);
    // h0 in the V-first layout the state_h stage reads (the caller gives [B,HV,K,V]); both paths.
    // Native layout: this is the LAST buffer, so its offset is the size without it (plus the padding).
    put(l.ih0, state * 4);
    if (pad) {
        off += pads.iq;
        put(l.iq, tokh * p.K * 2);
        off += pads.ik;
        put(l.ik, tokh * p.K * 2);
        put(l.iv, tok * p.V * 2);
        off += pads.ido;
        put(l.ido, tok * p.V * 2);
        put(l.ig, tok * 4);
        put(l.ibeta, tok * 4);
        put(l.idht, state * 4);
        put(l.oq, tokh * p.K * 2);
        put(l.ok, tokh * p.K * 2);
        put(l.ov, tok * p.V * 2);
        put(l.og, tok * 4);
        put(l.obeta, tok * 4);
        put(l.odh0, state * 4);
    }
    l.total = off;
    return l;
}

uint64_t residue(const void* x) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(x)) % kPlacePeriod; }

// Circular distance of two residues mod 16 MiB.
uint64_t ring_dist(uint64_t a, uint64_t b) {
    const uint64_t d = a >= b ? a - b : b - a;
    return d < kPlacePeriod - d ? d : kPlacePeriod - d;
}

// Smallest pairwise ring distance of the five dqkwg streams q, k, do, v_new, dv2.
uint64_t min_ring_dist(const uint64_t (&r)[5]) {
    uint64_t m = kPlacePeriod;
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j) {
            const uint64_t d = ring_dist(r[i], r[j]);
            if (d < m) m = d;
        }
    return m;
}

struct Place {
    PlacePads pads;
    uint64_t mind;  // resulting smallest ring distance of the five streams
};

// Picks the paddings that spread the five dqkwg streams q, k, do, v_new, dv2 around the 16 MiB ring.
// l0 is the unpadded layout, ws the workspace base; the residues of the workspace tensors are taken from
// the real pointer (no alignment assumed).  Every buffer behind a padding moves with it; that is
// accepted, no other stage is tuned for its residue yet.
//   native: q, k, do are the caller tensors (residues rq, rk, rdo).  Exhaustive over the paddings in
//     front of v_new and dv2 (16 x 16), maximising the smallest ring distance (ties: smallest total
//     padding, then first found).
//   padded: all five streams are workspace buffers.  Greedy in layout order (v_new, dv2, q, k, do): each
//     stream takes the smallest padding that maximises its distance to the streams placed before it.  On
//     a 16 MiB ring with 1 MiB steps that gives at least 2 MiB between any two (the i-th stream finds a
//     gap of at least 16/(i-1) MiB), which is the width of the collision window.
Place choose_place(uintptr_t ws, const Layout& l0, bool pad, uint64_t rq, uint64_t rk, uint64_t rdo) {
    Place best{};
    if (!pad) {
        bool have = false;
        for (uint64_t sv = 0; sv < 16; ++sv) {
            for (uint64_t sd = 0; sd < 16; ++sd) {
                const uint64_t r[5] = {rq, rk, rdo, (ws + l0.vn + sv * kPlaceStep) % kPlacePeriod,
                                       (ws + l0.dv2 + (sv + sd) * kPlaceStep) % kPlacePeriod};
                const uint64_t m = min_ring_dist(r);
                if (!have || m > best.mind || (m == best.mind && sv + sd < (best.pads.vn + best.pads.dv2) / kPlaceStep)) {
                    best.pads.vn = static_cast<size_t>(sv * kPlaceStep);
                    best.pads.dv2 = static_cast<size_t>(sd * kPlaceStep);
                    best.mind = m;
                    have = true;
                }
            }
        }
        return best;
    }
    const uint64_t nat[5] = {l0.vn, l0.dv2, l0.iq, l0.ik, l0.ido};  // layout order
    size_t* const slot[5] = {&best.pads.vn, &best.pads.dv2, &best.pads.iq, &best.pads.ik, &best.pads.ido};
    uint64_t r[5] = {};
    uint64_t cum = 0;  // total padding in front of the current stream
    for (int i = 0; i < 5; ++i) {
        uint64_t bp = 0, bm = 0;
        for (uint64_t pp = 0; pp < (i == 0 ? 1u : 16u); ++pp) {
            const uint64_t x = (ws + nat[i] + cum + pp * kPlaceStep) % kPlacePeriod;
            uint64_t m = kPlacePeriod;
            for (int j = 0; j < i; ++j) {
                const uint64_t d = ring_dist(x, r[j]);
                if (d < m) m = d;
            }
            if (pp == 0 || m > bm) {
                bm = m;
                bp = pp;
            }
        }
        cum += bp * kPlaceStep;
        r[i] = (ws + nat[i] + cum) % kPlacePeriod;
        *slot[i] = static_cast<size_t>(bp * kPlaceStep);
    }
    best.mind = min_ring_dist(r);  // layout order differs from min_ring_dist's, the set is the same
    return best;
}

using Desc = FlashTrainGDNBwdPadDesc;

Desc desc(const void* src, void* dst, int elem, int a0, int a1, int a2, int a3, int d0, int d1, int d2, int d3,
          int swap23 = 0) {
    Desc x{};
    x.src = src;
    x.dst = dst;
    x.elem = elem;
    x.swap23 = swap23;
    x.a[0] = a0; x.a[1] = a1; x.a[2] = a2; x.a[3] = a3;
    x.d[0] = d0; x.d[1] = d1; x.d[2] = d2; x.d[3] = d3;
    return x;
}

}  // namespace

extern "C" int flash_train_gdn_chunk_backward_supported(FlashTrainGDNChunkShape s) {
    return backward_ok(s) ? 1 : 0;
}

extern "C" size_t flash_train_gdn_chunk_backward_workspace_bytes(FlashTrainGDNChunkShape s) {
    return backward_ok(s) ? layout(s).total + place_slack(!native_ok(s)) : 0;
}

extern "C" int flash_train_gdn_chunk_bf16_backward_launch(const FlashTrainGDNChunkBackwardArgs* a,
                                                          void* stream) {
    if (a == nullptr || a->q == nullptr || a->k == nullptr || a->v == nullptr || a->g == nullptr ||
        a->beta == nullptr || a->do_ == nullptr || a->dq == nullptr || a->dk == nullptr ||
        a->dv == nullptr || a->dg == nullptr || a->dbeta == nullptr) {
        return FLASH_TRAIN_GDN_CHUNK_INVALID_POINTER;
    }
    const FlashTrainGDNChunkShape o = a->shape;  // the caller shape
    if (!backward_ok(o)) return FLASH_TRAIN_GDN_CHUNK_UNSUPPORTED_CONTRACT;
    const bool pad = !native_ok(o);
    const FlashTrainGDNChunkShape s = padded_shape(o);  // the shape the stage kernels run on
    const Layout l0 = layout(o);
    // The native path without h0 does not touch the h0 transpose buffer (the last one in its layout), so it
    // does not allocate it.  The placement paddings (choose_place) need place_slack() on top of either size.
    // The allocation size itself no longer matters for dqkwg: the placement is computed from the real
    // pointers (earlier versions had to keep the size fixed because dqkwg ran up to 7% slower in ~40% of
    // the processes for any other size).
    const size_t need = ((pad || a->h0 != nullptr) ? l0.total : l0.ih0) + place_slack(pad);
    static void* ws = nullptr;
    static size_t cap = 0;
    if (cap < need) {
        if (ws != nullptr && hipFree(ws) != hipSuccess) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
        ws = nullptr;
        cap = 0;
        if (hipMalloc(&ws, need) != hipSuccess) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
        cap = need;
    }
    const Place place = choose_place(reinterpret_cast<uintptr_t>(ws), l0, pad, residue(a->q), residue(a->k),
                                     residue(a->do_));
    const Layout l = layout(o, place.pads);
    char* p = static_cast<char*>(ws);
    float* gc = reinterpret_cast<float*>(p + l.gc);
    uint16_t* w = reinterpret_cast<uint16_t*>(p + l.w);
    uint16_t* u = reinterpret_cast<uint16_t*>(p + l.u);
    uint16_t* vn = reinterpret_cast<uint16_t*>(p + l.vn);
    uint16_t* A = reinterpret_cast<uint16_t*>(p + l.A);
    uint16_t* h = reinterpret_cast<uint16_t*>(p + l.h);
    uint16_t* dh = reinterpret_cast<uint16_t*>(p + l.dh);
    uint16_t* dv2 = reinterpret_cast<uint16_t*>(p + l.dv2);
    float* pl = reinterpret_cast<float*>(p + l.pl);
    void* st = stream;
    int rc;

    // Tensors the stage kernels see: the caller tensors (native) or the padded workspace copies.
    const uint16_t* q = a->q;
    const uint16_t* k = a->k;
    const uint16_t* v = a->v;
    const uint16_t* dout = a->do_;
    const float* g = a->g;
    const float* beta = a->beta;
    const float* h0 = a->h0;
    const float* dht = a->dht;
    uint16_t* dq = a->dq;
    uint16_t* dk = a->dk;
    uint16_t* dv = a->dv;
    float* dg = a->dg;
    float* dbeta = a->dbeta;
    float* dh0 = a->dh0;
    if (pad) {
        const int B = o.B, S = o.S, H = o.H, HV = o.HV, K = o.K, V = o.V;
        const int Sp = s.S, Kp = s.K, Vp = s.V;
        q = reinterpret_cast<const uint16_t*>(p + l.iq);
        k = reinterpret_cast<const uint16_t*>(p + l.ik);
        v = reinterpret_cast<const uint16_t*>(p + l.iv);
        dout = reinterpret_cast<const uint16_t*>(p + l.ido);
        g = reinterpret_cast<const float*>(p + l.ig);
        beta = reinterpret_cast<const float*>(p + l.ibeta);
        Desc in[FLASH_TRAIN_GDN_BWD_PAD_MAX_DESCS];
        int n = 0;
        in[n++] = desc(a->q, const_cast<uint16_t*>(q), 2, B, S, H, K, B, Sp, H, Kp);
        in[n++] = desc(a->k, const_cast<uint16_t*>(k), 2, B, S, H, K, B, Sp, H, Kp);
        in[n++] = desc(a->v, const_cast<uint16_t*>(v), 2, B, S, HV, V, B, Sp, HV, Vp);
        in[n++] = desc(a->do_, const_cast<uint16_t*>(dout), 2, B, S, HV, V, B, Sp, HV, Vp);
        in[n++] = desc(a->g, const_cast<float*>(g), 4, B, S, HV, 1, B, Sp, HV, 1);
        in[n++] = desc(a->beta, const_cast<float*>(beta), 4, B, S, HV, 1, B, Sp, HV, 1);
        if (a->h0 != nullptr) {  // [B,HV,K,V] -> zero-padded V-first [B,HV,Vp,Kp]
            h0 = reinterpret_cast<const float*>(p + l.ih0);
            in[n++] = desc(a->h0, const_cast<float*>(h0), 4, B, HV, K, V, B, HV, Vp, Kp, 1);
        }
        if (a->dht != nullptr) {
            dht = reinterpret_cast<const float*>(p + l.idht);
            in[n++] = desc(a->dht, const_cast<float*>(dht), 4, B, HV, V, K, B, HV, Vp, Kp);
        }
        if ((rc = flash_train_gdn_bwd_pad_copy(in, n, st)) != 0) return rc == 1 ? 1 : FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
        dq = reinterpret_cast<uint16_t*>(p + l.oq);
        dk = reinterpret_cast<uint16_t*>(p + l.ok);
        dv = reinterpret_cast<uint16_t*>(p + l.ov);
        dg = reinterpret_cast<float*>(p + l.og);
        dbeta = reinterpret_cast<float*>(p + l.obeta);
        if (a->dh0 != nullptr) dh0 = reinterpret_cast<float*>(p + l.odh0);
    } else if (a->h0 != nullptr) {
        // state_h (h_vk = 1) reads h0 V-first [B,HV,V,K]; the caller gives [B,HV,K,V].  One small
        // transposing copy, only when h0 is present.
        float* h0t = reinterpret_cast<float*>(p + l.ih0);
        const Desc x = desc(a->h0, h0t, 4, o.B, o.HV, o.K, o.V, o.B, o.HV, o.V, o.K, 1);
        if ((rc = flash_train_gdn_bwd_pad_copy(&x, 1, st)) != 0) return rc == 1 ? 1 : FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
        h0 = h0t;
    }
    const bool gqa = s.HV != s.H;
    uint16_t* dq_hv = gqa ? reinterpret_cast<uint16_t*>(p + l.dqhv) : dq;
    uint16_t* dk_hv = gqa ? reinterpret_cast<uint16_t*>(p + l.dkhv) : dk;
    uint16_t* dw = w;   // w is dead once B2 has run
    uint16_t* dvl = u;  // u is dead once R2 has run

    FlashTrainGDNIntraArgs ia{k, v, g, beta, w, u, gc, s.B, s.S, s.H, s.HV, s.K, s.V, A};
    if ((rc = flash_train_gdn_intra_bf16(&ia, st)) != 0) return rc == 1 ? 1 : FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    FlashTrainGDNStateHArgs ha{k, w, u, gc, h0, h, vn, nullptr, s.B, s.S, s.H, s.HV, s.K, s.V, 0, 1};
    if (flash_train_gdn_state_h_bf16(&ha, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    FlashTrainGDNBwdDvLocalArgs va{q, k, dout, gc, dvl, s.B, s.S, s.H, s.HV, s.K, s.V, a->scale};
    if (flash_train_gdn_bwd_dv_local_bf16(&va, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    FlashTrainGDNBwdDhuArgs da{q, k, w, dout, dvl, gc, dht, dh, dv2, dh0,
                               s.B, s.S, s.H, s.HV, s.K, s.V, a->scale, 1};
    if (flash_train_gdn_bwd_dhu_bf16(&da, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    FlashTrainGDNBwdDqkwgArgs qa{q, k, vn, dout, dv2, h, dh, gc, dq_hv, dk_hv, dw, pl,
                                 s.B, s.S, s.H, s.HV, s.K, s.V, a->scale};
    // GDN_PLACE_PRINT=1: residues mod 16 MiB (MiB) of the stage tensors and the chosen placement, stderr.
    // Read once; host arithmetic only (no device sync), off by default.
    static const bool place_print = [] {
        const char* e = std::getenv("GDN_PLACE_PRINT");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    if (place_print) {
        auto mib = [](const void* x) { return static_cast<double>(residue(x)) / double(kPlaceStep); };
        const uint64_t r[5] = {residue(q), residue(k), residue(dout), residue(vn), residue(dv2)};
        uint64_t r0[5] = {residue(a->q), residue(a->k), residue(a->do_), residue(p + l0.vn), residue(p + l0.dv2)};
        if (pad) {  // the unpadded layout, for comparison
            r0[0] = residue(p + l0.iq);
            r0[1] = residue(p + l0.ik);
            r0[2] = residue(p + l0.ido);
        }
        std::fprintf(stderr,
                     "PLACE path=%s pads(vn,dv2,q,k,do)=%zu,%zu,%zu,%zu,%zu MiB mindist=%.2fMiB (unpadded %.2fMiB) "
                     "q=%.3f k=%.3f do=%.3f vn=%.3f dv2=%.3f h=%.3f dh=%.3f gc=%.3f dq=%.3f dk=%.3f "
                     "dw=%.3f pl=%.3f ws=%.3f\n",
                     pad ? "pad" : "native", static_cast<size_t>(place.pads.vn / kPlaceStep),
                     static_cast<size_t>(place.pads.dv2 / kPlaceStep), static_cast<size_t>(place.pads.iq / kPlaceStep),
                     static_cast<size_t>(place.pads.ik / kPlaceStep), static_cast<size_t>(place.pads.ido / kPlaceStep),
                     static_cast<double>(min_ring_dist(r)) / double(kPlaceStep),
                     static_cast<double>(min_ring_dist(r0)) / double(kPlaceStep), mib(q), mib(k), mib(dout), mib(vn),
                     mib(dv2), mib(h), mib(dh), mib(gc), mib(dq_hv), mib(dk_hv), mib(dw), mib(pl), mib(ws));
    }
    if (flash_train_gdn_bwd_dqkwg_bf16(&qa, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    FlashTrainGDNBwdWyArgs wa{k, v, A, dw, dv2, dk_hv, beta, gc, pl, dk_hv, dv, dg,
                              dbeta, s.B, s.S, s.H, s.HV, s.K, s.V};
    if (flash_train_gdn_bwd_wy_bf16(&wa, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    if (gqa) {
        FlashTrainGDNBwdGroupSumArgs ga{dq_hv, dk_hv, dq, dk, s.B, s.S, s.H, s.HV, s.K};
        if (flash_train_gdn_bwd_groupsum_bf16(&ga, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    }
    if (pad) {
        const int B = o.B, S = o.S, H = o.H, HV = o.HV, K = o.K, V = o.V;
        const int Sp = s.S, Kp = s.K, Vp = s.V;
        Desc out[FLASH_TRAIN_GDN_BWD_PAD_MAX_DESCS];
        int n = 0;
        out[n++] = desc(dq, a->dq, 2, B, Sp, H, Kp, B, S, H, K);
        out[n++] = desc(dk, a->dk, 2, B, Sp, H, Kp, B, S, H, K);
        out[n++] = desc(dv, a->dv, 2, B, Sp, HV, Vp, B, S, HV, V);
        out[n++] = desc(dg, a->dg, 4, B, Sp, HV, 1, B, S, HV, 1);
        out[n++] = desc(dbeta, a->dbeta, 4, B, Sp, HV, 1, B, S, HV, 1);
        if (a->dh0 != nullptr) out[n++] = desc(dh0, a->dh0, 4, B, HV, Vp, Kp, B, HV, V, K);
        if (flash_train_gdn_bwd_pad_copy(out, n, st) != 0) return FLASH_TRAIN_GDN_CHUNK_HIP_ERROR;
    }
    return FLASH_TRAIN_GDN_CHUNK_SUCCESS;
}
