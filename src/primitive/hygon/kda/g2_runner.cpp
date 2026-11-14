// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime.h>

#include "flash_train/env.h"
#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/kda/g2_runner.hpp"

// Launchers defined in g2_kernels.hip (and the g2/ headers it includes).
extern "C" {
int g2_kda_tables(const int* cu, int N, int T, int NT, int* ctok, int* clen, int* seg, hipStream_t s);
int g2_head_gate(const void* g_raw, const void* A_log, const void* dt_bias, void* g_out, double lower_bound,
                 double scale, int gate, int NT, int H, const int* ctok, const int* clen, hipStream_t s);
int g2_l2n_apply(const void* q, const void* k, const void* q_rstd, const void* k_rstd, void* qn, void* kn,
                 long total_tokens, hipStream_t s);
int g2_fwd_prep_l2n(const void* q, const void* k, const void* v, const void* g_raw, const void* A_log,
                    const void* dt_bias, const void* beta, void* Wn, void* Kn, void* Un, void* QgA, void* AqkA,
                    void* Aqk, void* Akk, void* q_rstd, void* k_rstd, void* Gn, void* R, void* Kb, double lower_bound,
                    double gscale, int gate, double scale, int H, int NT, const int* ctok, const int* clen,
                    void* qn_out, void* kn_out, int l2n, hipStream_t s);  // kra P-94: l2n = 0 -> no l2norm
int g2_fwd_h_v8o(const void* Wn, const void* Kn, const void* Un, const void* Gn, const void* QgA, const void* AqkA,
                 void* O, void* Ob, void* Vn, void* HbN, void* ANv, float scale, int B, int T, int H, const int* seg,
                 int nseg, const int* ctok, const int* clen, const void* GnRaw, hipStream_t s);
int g2_kda_gn_all(const float* g, const int* ctok, const int* clen, int H, int NT, float* raw, float* gn, float* gnat,
                  int kp, hipStream_t s);  // kra P-99: to_gn (+ to_gn + exp2 | + gn_kp) in one pass
int g2_wu(const void* k, const void* v, const void* q, const void* beta, const void* A, const void* g, void* Wn,
          void* Wb, void* Un, void* KgA, void* Kn, void* Qn, void* kgB, void* vB, int B, int T, int H, const int* ctok,
          const int* clen, const void* hbn, void* vn_out, hipStream_t s);  // kra P-82: + hbn / vn_out
int g2_dav(const void* do_, const void* Vn, const void* A, void* dA, void* dVn, void* dOT, void* aNdo, double scale,
           int B, int T, int H, const int* ctok, const int* clen, hipStream_t s);
int g2_dhu_v8b(const void* KgA, const void* Qg, const void* Wb, const void* dOT, const void* dv, const void* Gn,
               void* dv2, void* HbN, void* ANv2, float scale, int B, int T, int H, const int* seg, int nseg,
               hipStream_t s);
// kra P-20/P-22a: + do_row (row-major dO for s4b), dv2c (C-form dv2, unused here), vnc (C-form v_new)
int g2_wy_s4_split_r(const void* q_kgb, const void* k, const void* v_vb, const void* v_new, const void* g,
                     const void* beta, const void* A, const void* h, const void* do_, const void* dh, const void* dv,
                     const void* dAqk, void* dq, void* dk, void* dv2, void* dg, void* db, void* dA, void* dwT,
                     void* db_part, float scale, int T, int H, int NT, const void* q_row, const void* v_row,
                     const int* ctok, const int* clen, const void* do_row, const void* dv2c, const void* vnc,
                     hipStream_t s);
int g2_bwd_tail(const void* dq, const void* dk, const void* dg, const void* db, const void* qn, const void* kn,
                const void* q_rstd, const void* k_rstd, const void* g_raw, const void* A_log, const void* dt_bias,
                const void* beta_raw, void* dqo, void* dko, void* dg_raw, void* dA_log, void* ddt_bias, void* db_out,
                void* dA_part, void* ddt_part, double lower_bound, double beta_scale, int use_l2, int use_beta, int B,
                int T, int H, const int* ctok, const int* clen, int NTv, int gate_none, int beta_f32, hipStream_t s);
}

namespace ftrain {
namespace kda_g2 {
namespace {

// The log2-domain scale of the chunk-local gate cumsum (the FLA literal; fp32 1/ln 2).
constexpr double kRcpLn2       = 1.4426950216;
constexpr std::uint64_t kAlign = 256;

// Bump allocator over the caller's workspace; with a null base it only measures.
class Arena {
  public:
    explicit Arena(void* base) noexcept : base_(static_cast<char*>(base)) {}

    void* take(std::uint64_t bytes) noexcept {
        offset_  = (offset_ + kAlign - 1) / kAlign * kAlign;
        void* p  = base_ == nullptr ? nullptr : base_ + offset_;
        offset_ += bytes;
        return p;
    }
    std::uint64_t used() const noexcept { return (offset_ + kAlign - 1) / kAlign * kAlign; }

  private:
    char* base_;
    std::uint64_t offset_ = 0;
};

void check(int rc, const char* what) {
    if (rc != 0) { throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "KDA G2 launch %s failed: %d", what, rc); }
}

// Per-chunk-head native block sizes (elements) and row-tensor sizes (elements).
struct Sizes {
    std::uint64_t T, H, N, NT, CH;
    explicit Sizes(const Shape& s) noexcept
        : T(static_cast<std::uint64_t>(s.tokens)), H(static_cast<std::uint64_t>(s.heads)),
          N(static_cast<std::uint64_t>(s.has_cu_seqlens ? s.seqs : 1)), NT(static_cast<std::uint64_t>(s.chunks())),
          CH(NT * static_cast<std::uint64_t>(s.heads)) {}
    std::uint64_t rows128() const noexcept { return T * H * 128; }  // [T, H, 128]
};

// Sub-buffers of one region that a kernel streams through together must not sit a multiple of
// the HBM channel-hash period apart: the native / row sizes are multiples of large powers of two,
// so back to back they put every stream on the same channels (measured on gfx936: s4b, which
// writes the fp32 dq | dk | dg rows, 1.2-1.3x slower; TCP->TA data stalls 1.9x). spread(spacing,
// n) is the smallest pad (64 KiB granular) after which n streams `spacing + pad` apart start at
// least kHashGap from each other modulo the period.
constexpr std::uint64_t kHashPeriod = std::uint64_t{32} << 20;
constexpr std::uint64_t kHashGap    = std::uint64_t{4} << 20;
constexpr std::uint64_t kHashStep   = std::uint64_t{64} << 10;

std::uint64_t spread(std::uint64_t spacing, std::uint64_t n) noexcept {
    const auto apart = [n](std::uint64_t r) {
        for (std::uint64_t i = 1; i < n; ++i) {
            const std::uint64_t d = i * r % kHashPeriod;
            if (d < kHashGap || kHashPeriod - d < kHashGap) { return false; }
        }
        return true;
    };
    for (std::uint64_t pad = 0; pad < kHashPeriod; pad += kHashStep) {
        if (apart((spacing + pad) % kHashPeriod)) { return pad; }
    }
    return 0;
}

// kra P-61e: the kp layouts (Kn lane-permuted, natural-k recompute decays) -- same rule the
// kernels' launchers apply (g2_common.h kPermMaxH)
inline bool g2KpLayout(long H) { return H <= 20; }

struct Tables {
    int* ctok = nullptr;
    int* clen = nullptr;
    int* seg  = nullptr;
};

Tables takeTables(Arena& arena, const Shape& shape, const Sizes& z) {
    Tables t;
    if (shape.isDense()) { return t; }
    t.ctok = static_cast<int*>(arena.take(z.NT * 4));
    t.clen = static_cast<int*>(arena.take(z.NT * 4));
    t.seg  = static_cast<int*>(arena.take((z.N + 1) * 4));
    return t;
}

// The single-pass prep reads the raw q/k/g once, so neither qn/kn nor the fp32 gate cumsum is
// materialized -- only its R (fp32 A partials) and Kb (bf16 k*beta*2^g) hand-offs.
// Vn (v_new) is a backward-only tensor: the forward does not write it. Kb lives only between the
// two prep kernels, so it is staged in the output o (same bf16 [T, H, 128]; the recurrence writes o last).
struct ForwardLayout {
    Tables tab;
    void *Wn, *Kn, *Un, *QgA, *AqkA, *R, *Gn;
};

ForwardLayout layoutForward(const Shape& shape, void* base, std::uint64_t* used) {
    const Sizes z(shape);
    Arena a(base);
    ForwardLayout L{};
    L.tab = takeTables(a, shape, z);
    // Wn | Kn | Un | QgA: the recurrence streams them together (spread over the channels, see spread())
    const std::uint64_t blk = z.CH * 8192 * 2, pw = blk + spread(blk, 4);
    char* w = static_cast<char*>(a.take(3 * pw + blk));
    L.Wn    = w;
    L.Kn    = w == nullptr ? nullptr : w + pw;
    L.Un    = w == nullptr ? nullptr : w + 2 * pw;
    L.QgA   = w == nullptr ? nullptr : w + 3 * pw;
    L.AqkA  = a.take(z.CH * 4096 * 2);
    L.R     = a.take(z.CH * 5120 * 4);
    L.Gn    = a.take(z.CH * 128 * 4);
    if (used != nullptr) { *used = a.used(); }
    return L;
}

// Backward lifetimes let three regions be reused:
//   B1 = Wn | Kn | Un (live through the bN recompute)   -> later bN(dh) | aN(dv2)
//   B2 = KgA | Qn | Wb | dVn | dOT (through dhu)       -> later fp32 dq | dk | dg (wy -> tail)
// Both reuses fit: 3 x 8192 = 16384 + 8192 elements per chunk-head, and B2 is sized for the
// larger of its two uses (3 x T x H x 128 fp32 <= 6 x 8192 bf16 per chunk-head).
//
// Staged reuse of the caller's output memory: dq, dk, dg are only written by the final tail
// kernel, so until then they host three row-sized intermediates (qn in dq, kn in dk, fp32 dA
// [T, H, 64] in dg -- equal or smaller byte sizes, any packed layout). The tail reads each
// qn / kn element before writing the dq / dk element at the same position (same lane), and dA
// is dead after wy. This keeps the outputs out of the peak (they used to sit idle beside the
// wy-stage working set): the workspace no longer holds qn, kn, dA.
struct BackwardLayout {
    Tables tab;
    void *g, *Gn, *GnRaw, *GnNat;
    void *kgB, *HbN;
    void *Wn, *Kn, *Un, *HbNdh, *ANv2;     // B1
    void *KgA, *Qn, *Wb, *Vn, *dVn, *dOT;  // B2
    void *dq, *dk, *dg;                    // B2 reuse (fp32 rows)
    void *dAout, *dwT, *dbPart, *dAPart, *ddtPart;
};

BackwardLayout layoutBackward(const Shape& shape, void* base, std::uint64_t* used) {
    const Sizes z(shape);
    Arena a(base);
    BackwardLayout L{};
    L.tab   = takeTables(a, shape, z);
    L.g     = shape.g_in_dqdk ? nullptr : a.take(z.rows128() * 4);  // else staged in dq | dk
    L.Gn    = a.take(z.CH * 128 * 4);
    L.GnRaw = a.take(z.CH * 128 * 4);                                                 // kra P-52
    L.GnNat = g2KpLayout(static_cast<long>(z.H)) ? a.take(z.CH * 128 * 4) : nullptr;  // kra P-61e
    L.kgB   = a.take(z.CH * 8192 * 2);
    // kra P-17/P-20/P-22a: no vB / aN_do / aN(v_new) copies (s4a reads v row-major, s4b reads dO
    // row-major and rebuilds aN(v_new) from Vn).  Vn lives OUTSIDE b2: s4b reads it while wy
    // writes dq/dk/dg over b2.  Keep-state: Vn and HbN come from the forward; saved-h (kra P-82):
    // HbN comes from the forward and wu writes Vn.
    L.Vn  = shape.has_state ? nullptr : a.take(z.CH * 8192 * 2);
    L.HbN = (shape.has_state || shape.has_saved_h) ? nullptr : a.take(z.CH * 16384 * 2);
    // The streams of each region are spread over the HBM channels (see spread()): B1's Wn | Kn |
    // Un (the recompute reads them together; bN(dh) | aN(dv2) reuse it), B2's KgA | Qn | Wb | dVn
    // | dOT (dhu) and its fp32 dq | dk | dg rows (wy).
    const std::uint64_t blk  = z.CH * 8192 * 2;                         // one 8192-element bf16 native, bytes
    const std::uint64_t rows = z.rows128() * 4;                         // fp32 [T, H, 128] bytes, <= 2 blk
    const bool recompute     = !shape.has_state && !shape.has_saved_h;  // Wn | Kn | Un are used
    const std::uint64_t p1   = blk + (recompute ? spread(blk, 3) : 0);
    const std::uint64_t pa   = 2 * p1 + spread(2 * p1, 2);  // bN(dh) [2 blk] | aN(dv2)
    const std::uint64_t p2   = blk + spread(blk, 5);
    const std::uint64_t pr   = rows + spread(rows, 3);
    char* b1                 = static_cast<char*>(a.take(std::max(2 * p1, pa) + blk));
    char* b2                 = static_cast<char*>(a.take(std::max(4 * p2 + blk, 2 * pr + rows)));
    if (b1 != nullptr) {
        L.Wn    = b1;
        L.Kn    = b1 + p1;
        L.Un    = b1 + 2 * p1;
        L.HbNdh = b1;
        L.ANv2  = b1 + pa;
    }
    if (b2 != nullptr) {
        L.KgA = b2;
        L.Qn  = b2 + p2;
        L.Wb  = b2 + 2 * p2;
        L.dVn = b2 + 3 * p2;
        L.dOT = b2 + 4 * p2;
        L.dq  = b2;
        L.dk  = b2 + pr;
        L.dg  = b2 + 2 * pr;
    }
    L.dAout   = a.take(z.NT * 64 * z.H * 64 * 4);
    L.dwT     = a.take(z.CH * 128 * 64 * 2);
    L.dbPart  = a.take(z.CH * 256 * 4);
    L.dAPart  = a.take(z.CH * 4);
    L.ddtPart = a.take(z.CH * 128 * 4);
    if (used != nullptr) { *used = a.used(); }
    return L;
}

}  // namespace

std::uint64_t forwardWorkspaceBytes(const Shape& shape) noexcept {
    std::uint64_t used = 0;
    layoutForward(shape, nullptr, &used);
    return used;
}

std::uint64_t backwardWorkspaceBytes(const Shape& shape) noexcept {
    std::uint64_t used = 0;
    layoutBackward(shape, nullptr, &used);
    return used;
}

void forward(const Shape& shape, const ForwardArgs& p, void* workspace, FTrainStream stream) {
    const Sizes z(shape);
    if (z.T == 0 || z.H == 0) { return; }
    const ForwardLayout L = layoutForward(shape, workspace, nullptr);
    const int T = static_cast<int>(z.T), H = static_cast<int>(z.H), NT = static_cast<int>(z.NT);
    const int nseg = shape.isDense() ? 1 : static_cast<int>(z.N);
    if (!shape.isDense()) {
        check(g2_kda_tables(p.cu_seqlens, static_cast<int>(z.N), T, NT, L.tab.ctok, L.tab.clen, L.tab.seg, stream),
              "tables");
    }
    // [l2norm +] gate + intra + to_gn in two kernels
    // kra P-77b: with p.qn / p.kn set, prep_a also writes the l2-normalized q / k for the backward.
    // kra P-94: without in-kernel l2norm the prep reads the (already normalized) q / k as qn / kn.
    const bool l2 = shape.l2norm_qk;
    check(g2_fwd_prep_l2n(p.q, p.k, p.v, p.g, shape.gate_in_kernel ? p.a_log : nullptr,
                          shape.gate_in_kernel ? p.dt_bias : nullptr, p.beta, L.Wn, L.Kn, L.Un, L.QgA, L.AqkA, p.aqk,
                          p.akk, l2 ? p.q_rstd : nullptr, l2 ? p.k_rstd : nullptr, L.Gn, L.R, p.o,
                          shape.gate_in_kernel ? p.lower_bound : 0.0, kRcpLn2, shape.gate_in_kernel ? 1 : 0, p.scale, H,
                          NT, L.tab.ctok, L.tab.clen, l2 ? p.qn : nullptr, l2 ? p.kn : nullptr, l2 ? 1 : 0, stream),
          "fwd_prep");
    // R5 keep-state: with p.state_* set, the recurrence also emits the backward's B2 state
    // (bN(h), v_new native and A-natural) -- bit-identical to the backward's own recompute.
    // kra P-82 saved-h: with p.saved_h set it emits bN(h) only (MODE 5 = o + h).
    void* hbn = p.state_hbn != nullptr ? p.state_hbn : p.saved_h;
    check(g2_fwd_h_v8o(L.Wn, L.Kn, L.Un, L.Gn, L.QgA, L.AqkA, nullptr, p.o, p.state_vn, hbn, p.state_anv, p.scale, 1,
                       NT * 64, H, L.tab.seg, nseg, L.tab.ctok, L.tab.clen, nullptr, stream),
          "fwd_h_o");
}

void backward(const Shape& shape, const BackwardArgs& p, void* workspace, FTrainStream stream) {
    const Sizes z(shape);
    if (z.T == 0 || z.H == 0) { return; }
    BackwardLayout L = layoutBackward(shape, workspace, nullptr);
    // fp32 g [T, H, 128] == the bytes of bf16 dq | dk: written by the gate, last read by wy, all
    // before the tail writes dq / dk (whose qn / kn staging the saved qn / kn made unnecessary).
    if (shape.g_in_dqdk) { L.g = p.dq; }
    const int T = static_cast<int>(z.T), H = static_cast<int>(z.H), NT = static_cast<int>(z.NT);
    const int nseg  = shape.isDense() ? 1 : static_cast<int>(z.N);
    const int *ctok = L.tab.ctok, *clen = L.tab.clen;
    if (!shape.isDense()) {
        check(g2_kda_tables(p.cu_seqlens, static_cast<int>(z.N), T, NT, L.tab.ctok, L.tab.clen, L.tab.seg, stream),
              "tables");
    }
    void* dA       = p.dg;  // staged: fp32 [T, H, 64] fits g's-dtype [T, H, 128]; dead before the tail writes dg
    const void* qn = p.q;
    const void* kn = p.k;
    if (p.qn != nullptr) {  // kra P-77b: the forward saved qn / kn -> no l2n_apply
        qn = p.qn;
        kn = p.kn;
    } else if (shape.l2norm_qk) {  // V1: bf16(x * rstd) from the saved rstd == the forward's qn/kn bit for bit
        check(g2_l2n_apply(p.q, p.k, p.q_rstd, p.k_rstd, p.dq, p.dk, static_cast<long>(z.T * z.H), stream),
              "l2n_apply");  // staged in the dq / dk outputs (the tail overwrites them in place)
        qn = p.dq;
        kn = p.dk;
    }
    check(g2_head_gate(p.g, shape.gate_in_kernel ? p.a_log : nullptr, shape.gate_in_kernel ? p.dt_bias : nullptr, L.g,
                       p.lower_bound, kRcpLn2, shape.gate_in_kernel ? 1 : 0, NT, H, ctok, clen, stream),
          "gate");
    // R5 keep-state: the forward handed over the B2 state -> wu skips Wn/Un/Kn (fwd_h_bn's inputs
    // only) and the bN recompute does not run.  kra P-82 saved-h: the forward handed over h only ->
    // wu also skips Wn/Un/Kn and forms v_new = bf_rne(u - W h) into Vn; no recompute either.
    const bool kept  = p.state_hbn != nullptr;         // == shape.has_state (the workspace was laid out for it)
    const bool hsave = !kept && p.saved_h != nullptr;  // == shape.has_saved_h
    const bool p1    = !kept && !hsave;                // wu emits the recompute's Wn/Un/Kn
    const void* Vn   = kept ? p.state_vn : L.Vn;
    const void* HbN  = kept ? p.state_hbn : (hsave ? p.saved_h : L.HbN);
    const void* ANv  = kept ? p.state_anv : nullptr;  // kra P-22a: s4b rebuilds aN(v_new) from Vn
    check(g2_wu(kn, p.v, qn, p.beta, p.akk, L.g, p1 ? L.Wn : nullptr, L.Wb, p1 ? L.Un : nullptr, L.KgA,
                p1 ? L.Kn : nullptr, L.Qn, L.kgB, nullptr /* kra P-17: no vB */, 1, NT * 64, H, ctok, clen,
                hsave ? p.saved_h : nullptr, hsave ? L.Vn : nullptr, stream),
          "wu");
    // kra P-99: GnRaw + Gn (exp2f, P-52) [+ GnNat and natural-k GnRaw for the kp layouts, P-61e] in one pass
    const bool kp = g2KpLayout(H);
    check(g2_kda_gn_all(static_cast<const float*>(L.g), ctok, clen, H, NT, static_cast<float*>(L.GnRaw),
                        static_cast<float*>(L.Gn), kp ? static_cast<float*>(L.GnNat) : nullptr, kp ? 1 : 0, stream),
          "gn_all");
    if (p1) {
        check(g2_fwd_h_v8o(L.Wn, L.Kn, L.Un, kp ? L.GnNat : L.Gn, nullptr, nullptr, nullptr, nullptr, L.Vn, L.HbN,
                           nullptr, 1.f, 1, NT * 64, H, L.tab.seg, nseg, nullptr, nullptr, L.GnRaw, stream),
              "fwd_h_bn");
    }
    check(g2_dav(p.d_o, Vn, p.aqk, dA, L.dVn, L.dOT, nullptr /* kra P-20: no aN_do */, p.scale, 1, NT * 64, H, ctok,
                 clen, stream),
          "dav");
    check(g2_dhu_v8b(L.KgA, L.Qn, L.Wb, L.dOT, L.dVn, L.Gn, nullptr, L.HbNdh, L.ANv2, p.scale, 1, NT * 64, H, L.tab.seg,
                     nseg, stream),
          "dhu");
    check(g2_wy_s4_split_r(L.kgB, kn, nullptr, ANv, L.g, p.beta, p.akk, HbN, nullptr, L.HbNdh, L.ANv2, dA, L.dq, L.dk,
                           p.dv, L.dg, p.dbeta, L.dAout, L.dwT, L.dbPart, p.scale, NT * 64, H, NT, qn, p.v, ctok, clen,
                           p.d_o, nullptr, kept ? nullptr : Vn, stream),
          "wy");
    // dbeta: the post-activation beta's gradient is wy's fp32 db, written in place (use_beta = 0).
    check(g2_bwd_tail(L.dq, L.dk, L.dg, p.dbeta, qn, kn, p.q_rstd, p.k_rstd, p.g,
                      shape.gate_in_kernel ? p.a_log : nullptr, shape.gate_in_kernel ? p.dt_bias : nullptr, p.beta,
                      p.dq, p.dk, p.dg, shape.gate_in_kernel ? p.da_log : nullptr,
                      shape.gate_in_kernel ? p.ddt_bias : nullptr, nullptr, L.dAPart, L.ddtPart, p.lower_bound, 1.0,
                      shape.l2norm_qk ? 1 : 0, 0, 1, T, H, ctok, clen, NT, shape.gate_in_kernel ? 0 : 1, 1, stream),
          "bwd_tail");
}

bool deviceSupported(FTrainDeviceId device) noexcept {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) { return false; }
    return std::strncmp(prop.gcnArchName, "gfx936", 6) == 0;
}

}  // namespace kda_g2
}  // namespace ftrain
