#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// GDN-2 backward, per-stage C APIs (chunk = 64).  Plan: docs/GDN2_backward_design_20261028.md.
// Layouts as FLA chunk_gdn2: q,k [B,S,H,K] bf16; v,do,v_new [B,S,HV,V] bf16; g/gc [B,S,HV,K] fp32;
// b [B,S,HV,K], wg [B,S,HV,V] fp32 holding bf16 values; hk = hv / (HV/H).  gc is the chunk-local cumsum
// of g in log2 units (what the forward intra kernel computes; gclast is its chunk-end row).
// Full chunks only (S % 64 == 0), K,V in {64,128}.
// Return: 0 ok, 1 null pointer, 2 unsupported shape, 5 HIP error.

// ---- B1 dAv (implemented) ----
// Same semantics as FLA chunk_kda_bwd_dAv (its q,k arguments are unused there and not taken here):
//   dAqk[t][s] = scale * (do_t . v_new_s)   for s <= t within the chunk, 0 above the diagonal   (fp32)
//   dv_local[s] = sum_{t >= s, same chunk} Aqk[t][s] do_t                                       (bf16)
// aqk is the forward Aqk (scale already applied); only its s <= t part is read (the rest may be undefined).
typedef struct FlashTrainGDN2BwdDAvArgs {
    const uint16_t* v_new;   // [B,S,HV,V]  bf16
    const uint16_t* do_;     // [B,S,HV,V]  bf16
    const uint16_t* aqk;     // [B,S,HV,64] bf16
    float* dAqk;             // [B,S,HV,64] fp32 (output)
    uint16_t* dv_local;      // [B,S,HV,V]  bf16 (output)
    int B, S, HV, V;
    float scale;
} FlashTrainGDN2BwdDAvArgs;

int flash_train_gdn2_bwd_dav_bf16(const FlashTrainGDN2BwdDAvArgs* args, void* stream);

// ---- B2 dhu (implemented, phase P1) ----  FLA chunk_gated_delta_rule_bwd_dhu with gk (per-channel), q=qg, k=kg, w=w_wy
// Serial reverse recursion over chunks, per-channel decay.  With dh the state gradient at the END of chunk n
// (dht, or 0, for the last chunk):
//   dh_store[n] = bf16(dh)   (before chunk n is processed; V-first [B,S/64,HV,V,K] like state_h's h; dh_vk=0: [..,K,V])
//   dv2[t]      = dv_local[t] + kg_t . dh                          (kg = k 2^(gclast-gc): the token gate is already in it)
//   dh[:,kk]    = 2^gclast[kk] dh[:,kk] + scale sum_t qg_t^T do_t - sum_t w_t^T dv2_t     (gclast: log2, chunk-end row)
// After chunk 0 the remaining dh is dh0 (fp32, [B,HV,V,K] or [B,HV,K,V] by dh_vk).  dht, dh0 optional (null).
// qg, kg, w are per v-head [B,S,HV,K] (what the forward intra kernel writes); H is accepted for symmetry and unused.
typedef struct FlashTrainGDN2BwdDhuArgs {
    const uint16_t* qg;        // [B,S,HV,K] bf16
    const uint16_t* kg;        // [B,S,HV,K] bf16
    const uint16_t* w;         // [B,S,HV,K] bf16
    const uint16_t* do_;       // [B,S,HV,V] bf16
    const uint16_t* dv_local;  // [B,S,HV,V] bf16   (B1 output)
    const float* gclast;       // [B,S/64,HV,K] fp32 (log2 units)
    const float* dht;          // optional final-state gradient fp32, layout by dh_vk (null = 0)
    uint16_t* dh;              // [B,S/64,HV,V,K] bf16 (output; dh_vk=0: [B,S/64,HV,K,V])
    uint16_t* dv2;             // [B,S,HV,V] bf16 (output)
    float* dh0;                // optional fp32 output
    int B, S, H, HV, K, V;
    float scale;
    int dh_vk;                 // 1: V-first dh/dht/dh0 (production), 0: K-first (FLA state_v_first=False)
} FlashTrainGDN2BwdDhuArgs;

int flash_train_gdn2_bwd_dhu_bf16(const FlashTrainGDN2BwdDhuArgs* args, void* stream);

// ---- B3 wy_dqkg (implemented, phase P2) ----  FLA chunk_gdn2_bwd_wy_dqkg_fused
// One block per (chunk, hv).  Everything is per v-head; q,k are read at hk = hv / (HV/H), their gradients are NOT
// group-summed here (P3 does that).  X = Akk_inv^T (upper triangular) is FLA's b_A.  With gc the in-chunk cumulative
// log2 gate (the kernel cumsums g * log2e itself), gn = gc[63], per element (t, kk), v-index vv:
//   dq_raw = do.h        dk_raw = v_new.dh        dwf = dv2.h          (contraction over V; h, dh are [V][K] blocks)
//   dgk[kk] = sum_v h[v][kk] dh[v][kk]
//   dq   = dq_raw 2^gc scale                dk_p = dk_raw 2^(gn-gc)         dwfn = bf16(-dwf)
//   kg = k 2^gc    kgb = bf16(kg b)         kdk = k dk_p
//   dkgb = X dwfn  (contraction over t')    db   = dkgb kg
//   dg   = q dq - kdk + kg dkgb b + [t==63] (dgk 2^gn + sum_t kdk)     dk = dk_p + dkgb 2^gc b
//   dvb  = X dv2                            dv   = bf16(dvb wg)        dw = bf16(dvb v)
//   dA   = dwfn kgb^T + dv2 bf16(v wg)^T   (strictly lower part kept, bf16)   dAkk = -tril_strict(X bf16(dA X) )  [X . (dA . X)]
// Partial-output contract for P3 (all [B,S,HV,*]): dq,dk,db,dg are fp32 and PARTIAL (the in-chunk part of the
// gradient, i.e. what FLA hands to chunk_gdn2_bwd_intra; dg is before the reverse in-chunk cumsum, in log2 units
// exactly like FLA); dAkk fp32 [B,S,HV,64] final per FLA (row t, col s, strictly lower, zeros elsewhere); dv, dw are
// FINAL (bf16).
typedef struct FlashTrainGDN2BwdWyDqkgArgs {
    const uint16_t* q;        // [B,S,H,K]  bf16
    const uint16_t* k;        // [B,S,H,K]  bf16
    const uint16_t* v;        // [B,S,HV,V] bf16
    const uint16_t* v_new;    // [B,S,HV,V] bf16 (state_h output)
    const uint16_t* do_;      // [B,S,HV,V] bf16
    const uint16_t* dv2;      // [B,S,HV,V] bf16 (B2 dhu output)
    const uint16_t* h;        // [B,S/64,HV,V,K] bf16 (state_h, V-first)
    const uint16_t* dh;       // [B,S/64,HV,V,K] bf16 (B2 dhu output, V-first, dh_vk=1)
    const uint16_t* akk_inv;  // [B,S,HV,64] bf16 (forward intra akk_inv output)
    const float* g;           // [B,S,HV,K] fp32 natural-log decay (raw input; chunk cumsum is done in the kernel)
    const float* b;           // [B,S,HV,K] fp32 (bf16 values)
    const float* wg;          // [B,S,HV,V] fp32 (bf16 values)
    float* dq;                // [B,S,HV,K] fp32 partial
    float* dk;                // [B,S,HV,K] fp32 partial
    float* db;                // [B,S,HV,K] fp32 partial
    float* dg;                // [B,S,HV,K] fp32 partial
    uint16_t* dv;             // [B,S,HV,V] bf16 final
    uint16_t* dw;             // [B,S,HV,V] bf16 final
    float* dAkk;              // [B,S,HV,64] fp32
    int B, S, H, HV, K, V;
    float scale;
    // Appended at the END (ctypes callers built against the older layout must add this field).
    // 0: dq,dk,db,dg are row-major [B,S,HV,K] (the contract above).  1: "lane order": the partials are stored in the order
    // of B3's own store instructions so that each store writes 1 KB contiguous (the row-major store is two half-used
    // 128 B lines per row per instruction, ~2x slower); B4 reads that layout with partials_lane_order = 1.  Element (t, ch)
    // of the head (b, chunk n, hv) lives at float index
    //     ((((((tile*KS + s)*2 + p)*2 + h2)*4 + t/16)*64 + hq*16 + t%16)*4 + e,   tile = (b*NT + n)*HV + hv, KS = K/64,
    //     s = ch/64, p = (ch%64)/32, hq = (ch%32)/8, h2 = (ch%8)/4, e = ch%4.
    int partial_lane_order;
} FlashTrainGDN2BwdWyDqkgArgs;

int flash_train_gdn2_bwd_wy_dqkg_bf16(const FlashTrainGDN2BwdWyDqkgArgs* args, void* stream);

// ---- B4 intra_bwd + epilogue (implemented, phase P3) ---- FLA chunk_gdn2_bwd_intra + reverse cumsum + GQA sum + casts
// From dAqk (B1), dAkk (B3) and B3's partial dq,dk,db,dg: back-propagate the in-chunk Aqk/Akk through the per-channel
// decay (2^(gc_i-gc_r) 2^(gc_r-gc_j) split, tf32 operands, kExpClamp as the forward), add the partials, take the
// reverse in-chunk cumsum of dg and sum dq,dk over the HV/H v-heads of each key head (fixed order).  Equations:
//   dq2[t] = sum_{s<=t} dAqk[t][s] k[s] E     D2[t] = sum_{s<t} dAkk[t][s] k[s] E     E = 2^(gc_t-gc_s), per channel
//   dkt[s] = sum_{t>=s} dAqk[t][s] q[t] E + sum_{t>s} dAkk[t][s] (k b)[t] E
//   dq = dq_in + dq2    dk = dk_in + D2 b + dkt    db = db_in + D2 k    dgp = dg_in + q dq2 + (D2 b - dkt) k
//   dg[t] = sum_{t' >= t, same chunk} dgp[t']   (gradient w.r.t. the natural-log g: no unit factor, ln2*log2e = 1)
// g is the raw natural-log gate (the chunk cumsum is done in the kernel).  dg may alias dg_in (in-place).
// GQA (HV != H): one kernel writes the per-v-head fp32 dq,dk over dq_in/dk_in, a second sums the HV/H heads of each
// key head in a fixed order into the bf16 dq,dk (deterministic).
typedef struct FlashTrainGDN2BwdIntraArgs {
    const uint16_t* q;       // [B,S,H,K]  bf16
    const uint16_t* k;       // [B,S,H,K]  bf16
    const float* g;          // [B,S,HV,K] fp32 natural-log gate
    const float* b;          // [B,S,HV,K] fp32 (bf16 values)
    const float* dAqk;       // [B,S,HV,64] fp32 (B1; lower triangle incl. diagonal read)
    const float* dAkk;       // [B,S,HV,64] fp32 (B3; strictly lower triangle read)
    float* dq_in;            // [B,S,HV,K] fp32 partials (B3).  NOTE: when HV != H they are overwritten with the fp32
    float* dk_in;            //   per-v-head dq,dk (input of the group-sum kernel); untouched when HV == H
    const float* db_in;
    const float* dg_in;
    uint16_t* dq;            // [B,S,H,K]  bf16 final (group-summed)
    uint16_t* dk;            // [B,S,H,K]  bf16 final (group-summed)
    uint16_t* db;            // [B,S,HV,K] bf16 final
    float* dg;               // [B,S,HV,K] fp32 final (w.r.t. natural-log g)
    int B, S, H, HV, K;
    // Appended at the END.  partials_lane_order = 1: dq_in,dk_in,db_in,dg_in are in B3's lane-order layout (see
    // FlashTrainGDN2BwdWyDqkgArgs.partial_lane_order); dg then must NOT alias dg_in.  When HV != H the fp32 per-v-head
    // dq,dk (input of the group sum) go to the row-major buffers dq_f32,dk_f32 [B,S,HV,K] (required then; dq_in/dk_in
    // are left untouched).  With partials_lane_order = 0 and HV != H the legacy in-place behaviour is kept
    // (dq_f32 / dk_f32 may be null).
    int partials_lane_order;
    float* dq_f32;
    float* dk_f32;
} FlashTrainGDN2BwdIntraArgs;

int flash_train_gdn2_bwd_intra_bf16(const FlashTrainGDN2BwdIntraArgs* args, void* stream);

// ---- top level (implemented, phase P3) ----
// Whole chunked backward from the forward inputs (everything is recomputed): R1 intra (with Akk_inv) -> R2 state_h ->
// B1 dAv -> B2 dhu -> B3 wy_dqkg -> B4 intra_bwd.  Applicability = the forward chunk route (S >= 512, S % 64 == 0,
// K,V in {64,128}, HV % H == 0, bf16).  Return codes as the stages: 0 ok, 1 null pointer, 2 unsupported, 4 workspace
// too small, 5 HIP error.  Workspace is caller-provided like flash_train_gdn2_chunk_forward_bf16.
int flash_train_gdn2_chunk_backward_shape_supported(int S, int H, int HV, int K, int V);
// Bytes of workspace (0 if unsupported).  Includes the forward route's buffers (recomputed) plus the stage buffers.
size_t flash_train_gdn2_chunk_backward_workspace_bytes(int B, int S, int H, int HV, int K, int V);
// q,k [B,S,H,K] bf16; v,do [B,S,HV,V] bf16; g,b [B,S,HV,K] fp32 (g natural-log gate, b holds bf16 values); wg [B,S,HV,V]
// fp32 (bf16 values).  dht: optional final-state gradient fp32 [B,HV,V,K]; dh0: optional output fp32 [B,HV,V,K]
// (gradient w.r.t. a zero initial state).  Outputs with FLA's dtypes: dq,dk [B,S,H,K] bf16 (group-summed),
// dv [B,S,HV,V] bf16, dg [B,S,HV,K] fp32, db [B,S,HV,K] bf16, dw [B,S,HV,V] bf16.
int flash_train_gdn2_chunk_backward_bf16(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* g, const float* b, const float* wg,
    const uint16_t* do_, const float* dht, uint16_t* dq, uint16_t* dk, uint16_t* dv, float* dg, uint16_t* db,
    uint16_t* dw, float* dh0, int B, int S, int H, int HV, int K, int V, float scale, void* workspace,
    size_t workspace_bytes, void* stream);

#ifdef __cplusplus
}
#endif
