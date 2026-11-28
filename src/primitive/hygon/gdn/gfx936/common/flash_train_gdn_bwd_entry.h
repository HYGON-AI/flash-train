#pragma once

#include <stddef.h>
#include <stdint.h>

#include "flash_train_gdn_chunk.h"

#ifdef __cplusplus
extern "C" {
#endif

// GDN backward, single entry (chunk = 64, bf16).  Runs R1 intra recompute -> R2 state_h ->
// B1 dv_local -> B2 dhu -> B3 dqkwg -> B4 wy_bwd (+ group sum when HV > H) on one stream.
// Applicability is the long-sequence bf16 chunk64 gate.
// anything else returns FLASH_TRAIN_GDN_CHUNK_UNSUPPORTED_CONTRACT without touching the device.
//
// Layouts as the forward entry / FLA: q,k [B,S,H,K]; v,do [B,S,HV,V] bf16; g,beta [B,S,HV] fp32
// (g = per-token log decay, natural log); HV must be a multiple of H.
// Outputs: dq,dk [B,S,H,K] bf16 (summed over the HV/H group), dv [B,S,HV,V] bf16,
// dg,dbeta [B,S,HV] fp32 (the dtypes FLA returns for bf16 q,k,v and fp32 g,beta).
// Optional state terms (null = absent):
//   h0  [B,HV,K,V] fp32 initial state (as the forward state_h argument)
//   dht [B,HV,V,K] fp32 gradient of the final state (V-first)
//   dh0 [B,HV,V,K] fp32 output gradient of the initial state (V-first; FLA's is [B,HV,K,V])
typedef struct FlashTrainGDNChunkBackwardArgs {
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    const float* g;
    const float* beta;
    const uint16_t* do_;
    const float* h0;
    const float* dht;
    uint16_t* dq;
    uint16_t* dk;
    uint16_t* dv;
    float* dg;
    float* dbeta;
    float* dh0;
    FlashTrainGDNChunkShape shape;
    float scale;
    void* workspace;
    size_t workspace_bytes;
} FlashTrainGDNChunkBackwardArgs;

// 1 when the backward covers this shape, else 0.
int flash_train_gdn_chunk_backward_supported(FlashTrainGDNChunkShape shape);

// Bytes of caller workspace, excluding optional state-layout conversion; 0 for unsupported shapes.
size_t flash_train_gdn_chunk_backward_workspace_bytes(FlashTrainGDNChunkShape shape);

// Returns a FlashTrainGDNChunkStatus value.
int flash_train_gdn_chunk_bf16_backward_launch(const FlashTrainGDNChunkBackwardArgs* args, void* stream);

#ifdef __cplusplus
}
#endif
