#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// GDN backward, per-stage C APIs (chunk = 64, layouts as FLA: q,k [B,S,H,K] bf16;
// v,do [B,S,HV,V] bf16; g,beta [B,S,HV] fp32; hk = hv / (HV/H)).  The plan is in
// docs/GDN_backward_design_20261027.md.  gc is the chunk-local cumsum of g in log2 units
// (what flash_train_gdn_intra_bf16 writes).  Full chunks only (S % 64 == 0), K,V in {64,128}.
// Return: 0 ok, 1 null pointer, 2 unsupported shape, 5 HIP error.

// ---- B1 dv_local (implemented) ----
//   dv_local[s] = sum_{t >= s, same chunk} scale * 2^(gc_t - gc_s) * (q_t . k_s) * do_t
typedef struct FlashTrainGDNBwdDvLocalArgs {
    const uint16_t* q;     // [B,S,H,K]  bf16
    const uint16_t* k;     // [B,S,H,K]  bf16
    const uint16_t* do_;   // [B,S,HV,V] bf16
    const float* gc;       // [B,S,HV]   fp32, log2 units
    uint16_t* dv_local;    // [B,S,HV,V] bf16 (output)
    int B, S, H, HV, K, V;
    float scale;
} FlashTrainGDNBwdDvLocalArgs;

int flash_train_gdn_bwd_dv_local_bf16(const FlashTrainGDNBwdDvLocalArgs* args, void* stream);

// ---- B2 dhu (implemented, phase P1) ----
// Serial reverse recursion over chunks (grid = (V/64, B*HV), 4 waves, see flash_train_gdn_bwd_dhu.hip).
// With dh the state gradient at the END of chunk n (dht for the last chunk, else 0):
//   dh[n]  = bf16(dh)                       (stored before chunk n is processed, as FLA)
//   dv2[t] = dv_local[t] + 2^(gc_last - gc_t) (k_t . dh)
//   dh     = 2^(gc_last) dh + scale sum_t (2^gc_t q_t)^T do_t - sum_t w_t^T dv2_t
// Full chunks only; K in {64,128}; V % 64 == 0.
typedef struct FlashTrainGDNBwdDhuArgs {
    const uint16_t* q;         // [B,S,H,K]  bf16
    const uint16_t* k;         // [B,S,H,K]  bf16
    const uint16_t* w;         // [B,S,HV,K] bf16 (from the intra kernel / FLA recompute_w_u)
    const uint16_t* do_;       // [B,S,HV,V] bf16
    const uint16_t* dv_local;  // [B,S,HV,V] bf16 (B1 output)
    const float* gc;           // [B,S,HV]   fp32, chunk-local cumsum, log2 units
    const float* dht;          // optional: gradient of the final state, fp32, layout as dh0 (null = 0)
    uint16_t* dh;              // state gradient at the end of each chunk, bf16, layout per dh_vk
    uint16_t* dv2;             // [B,S,HV,V] bf16 (output)
    float* dh0;                // optional output: gradient of the initial state, fp32,
                               //   [B,HV,V,K] if dh_vk else [B,HV,K,V]
    int B, S, H, HV, K, V;
    float scale;
    int dh_vk;                 // 1: dh [B,S/64,HV,V,K] (V-first, what B3 reads, as state_h h_vk=1)
                               // 0: dh [B,S/64,HV,K,V] (FLA default layout)
} FlashTrainGDNBwdDhuArgs;

int flash_train_gdn_bwd_dhu_bf16(const FlashTrainGDNBwdDhuArgs* args, void* stream);

// ---- B3 dqkwg (implemented, phase P2) ----
// Per (chunk n, hv); h = state at the START of chunk n, dh = state gradient at the END of chunk n
// (both V-first [B,S/64,HV,V,K] bf16: state_h h_vk=1, dhu dh_vk=1), vn = v_new, dv2 from dhu:
//   dq_raw = do h, dk_raw = vn dh, dw = -dv2 h, ds_raw = do vn^T   (contract V)
//   dk_g = dk_raw 2^(gc_63 - gc_t);  dg_last = 2^gc_63 sum(h*dh) + sum(dk_g*k)
//   ds[t][s] = scale 2^(gc_t-gc_s) ds_raw[t][s] (t >= s, rounded to bf16)
//   dq = scale 2^gc_t dq_raw + ds k;  dk = dk_g + ds^T q
//   dg_t = dq_t.q_t - dk_t.k_t (+ dg_last on the last row of the chunk)
// Outputs follow FLA chunk_bwd_dqkwg: dq,dk,dw are per v-head [B,S,HV,K] bf16 (FLA sums dq,dk over
// the HV/H group afterwards when HV != H; not done here), dw already carries the minus sign, dg is
// fp32 planes [P][B,S,HV] (sum over P) before the in-chunk reverse cumsum (done after wy_bwd).  w is not an input.
typedef struct FlashTrainGDNBwdDqkwgArgs {
    const uint16_t* q;      // [B,S,H,K]  bf16
    const uint16_t* k;      // [B,S,H,K]  bf16
    const uint16_t* v_new;  // [B,S,HV,V] bf16
    const uint16_t* do_;    // [B,S,HV,V] bf16
    const uint16_t* dv2;    // [B,S,HV,V] bf16 (dhu output)
    const uint16_t* h;      // [B,S/64,HV,V,K] bf16 (V-first)
    const uint16_t* dh;     // [B,S/64,HV,V,K] bf16 (V-first)
    const float* gc;        // [B,S,HV] fp32, chunk-local cumsum, log2 units
    uint16_t* dq;           // [B,S,HV,K] bf16 (output)
    uint16_t* dk;           // [B,S,HV,K] bf16 (output)
    uint16_t* dw;           // [B,S,HV,K] bf16 (output, = -dv2 h)
    float* dg;              // [P][B,S,HV] fp32 (output), P = flash_train_gdn_bwd_dqkwg_dg_planes(K)
    int B, S, H, HV, K, V;
    float scale;
} FlashTrainGDNBwdDqkwgArgs;

int flash_train_gdn_bwd_dqkwg_bf16(const FlashTrainGDNBwdDqkwgArgs* args, void* stream);
// dg is written as flash_train_gdn_bwd_dqkwg_dg_planes(K) planes [P][B,S,HV] (one per K slice of the
// kernel, FLA likewise emits NK partial planes); the consumer sums them.
int flash_train_gdn_bwd_dqkwg_dg_planes(int K);

// ---- B4 wy_bwd + epilogue (implemented, phase P3) ----
// Per (chunk, hv).  From the recomputed A = (I+L)^-1 (intra kernel, optional output A), dw (B3), dv2 (B2),
// dk (B3, per hv) and the dqkwg dg planes:
//   Dk = A^T dw, Dv = A^T dv2;  dk2 = beta e Dk;  dv = beta Dv           (e = 2^gc)
//   dA = strict_lower(beta_j e_j dw_i.k_j + beta_j dv2_i.v_j) -> bf16;  Y = dA A^T -> bf16;  Z = A^T Y
//   dA' = strict_lower(-Z 2^(gc_i - gc_j)) -> bf16;  G = k k^T
//   dk_out = dk_in + dk2 + beta_j dA' k + sum_i beta_i dA'_ij k_i
//   dbeta = e Dk.k + Dv.v + sum_x dA'_ax G_ax
//   dg = reverse in-chunk cumsum(sum(planes) + beta e Dk.k + beta sum_x dA'_ax G_ax - sum_i dA'_ia beta_i G_ia)
// No ln2 factors (FLA convention: dg is w.r.t. the natural-log g).  dk_out is per v-head [B,S,HV,K]; dk_in may
// alias dk_out.  Group sum to the H key heads is flash_train_gdn_bwd_groupsum_bf16.
typedef struct FlashTrainGDNBwdWyArgs {
    const uint16_t* k;          // [B,S,H,K]  bf16
    const uint16_t* v;          // [B,S,HV,V] bf16
    const uint16_t* A;          // [B,S,HV,64] bf16 (intra kernel, FLA layout)
    const uint16_t* dw;         // [B,S,HV,K] bf16 (B3 output)
    const uint16_t* dv2;        // [B,S,HV,V] bf16 (B2 output)
    const uint16_t* dk_in;      // [B,S,HV,K] bf16 (B3 output)
    const float* beta;          // [B,S,HV] fp32
    const float* gc;            // [B,S,HV] fp32, chunk-local cumsum, log2 units
    const float* dg_planes;     // [P][B,S,HV] fp32 (B3 output), P = flash_train_gdn_bwd_dqkwg_dg_planes(K)
    uint16_t* dk;               // [B,S,HV,K] bf16 (output, per v-head)
    uint16_t* dv;               // [B,S,HV,V] bf16 (output)
    float* dg;                  // [B,S,HV] fp32 (output, final)
    float* dbeta;               // [B,S,HV] fp32 (output)
    int B, S, H, HV, K, V;
} FlashTrainGDNBwdWyArgs;

int flash_train_gdn_bwd_wy_bf16(const FlashTrainGDNBwdWyArgs* args, void* stream);

// out[b,s,hk,:] = sum_{g<HV/H} in[b,s,hk*(HV/H)+g,:] for dq and dk in one launch (fp32 sum, bf16 out).
typedef struct FlashTrainGDNBwdGroupSumArgs {
    const uint16_t* dq_in;      // [B,S,HV,K]
    const uint16_t* dk_in;      // [B,S,HV,K]
    uint16_t* dq;               // [B,S,H,K]
    uint16_t* dk;               // [B,S,H,K]
    int B, S, H, HV, K;
} FlashTrainGDNBwdGroupSumArgs;

int flash_train_gdn_bwd_groupsum_bf16(const FlashTrainGDNBwdGroupSumArgs* args, void* stream);

#ifdef __cplusplus
}
#endif
