#pragma once

#include "gfx938_device_symbols.h"
#include <stddef.h>
#include <stdint.h>

#include "flash_train_gdn_chunk.h"

#ifdef __cplusplus
extern "C" {
#endif

// GDN backward, single entry (chunk = 64, bf16).  Runs R1 intra recompute -> R2 state_h ->
// B1 dv_local -> B2 dhu -> B3 dqkwg -> B4 wy_bwd (+ group sum when HV > H) on one stream.
//
// Applicability (independent of the forward route): B,S,H,HV >= 1, HV % H == 0, 1 <= K <= 128,
// 1 <= V <= 128 (shape.chunk_size 0, 32 or 64), plus V = 256 on the native path only; anything else
// (V = 129..255, V > 256, V = 256 with S % 64 != 0 or K not in {64,128}, K > 128, HV < H, HV % H != 0)
// returns FLASH_TRAIN_GDN_CHUNK_UNSUPPORTED_CONTRACT without touching the device.
//   native path: S % 64 == 0, K in {64,128}, V in {64,128,256}: the stage kernels run on the caller
//     tensors (V = 256 is not padded: its intra / dv_local stages use dedicated instantiations).
//   padded path: every other shape with V <= 128.  Inputs are zero-padded inside the workspace to
//     Sp = ceil(S/64)*64, Kp = (K <= 64 ? 64 : 128), Vp = (V <= 64 ? 64 : 128) (padded tokens:
//     q=k=v=do=0, g=0, beta=0; padded K/V columns and the padded parts of h0/dht are 0; scale is
//     the caller value), the native path runs on the padded shape and dq,dk,dv,dg,dbeta,dh0 are
//     cropped back; nothing outside the caller tensors is written.  One extra pad kernel before and
//     one crop kernel after the six stages, all asynchronous on the given stream.  h0 ([B,HV,K,V]) is
//     transposed into the V-first layout state_h reads (one extra copy kernel when h0 is present,
//     on both paths).
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
} FlashTrainGDNChunkBackwardArgs;

// 1 when the backward covers this shape, else 0.
int flash_train_gdn_chunk_backward_supported(FlashTrainGDNChunkShape shape);

// Bytes of internal workspace the launcher keeps (cached per process, grown on demand, single
// stream use), including the padded input/output copies of the padded path and the h0 transpose
// buffer (upper bound: the native path allocates the h0 buffer only when h0 is passed), plus placement
// slack (32 MiB native, 64 MiB padded path): the launcher pads in front of v_new, dv2 (and the padded copies
// of q, k, do) so the dqkwg streams do not collide modulo 16 MiB (see the .hip header); 0 for unsupported shapes.
size_t flash_train_gdn_chunk_backward_workspace_bytes(FlashTrainGDNChunkShape shape);

// Returns a FlashTrainGDNChunkStatus value.
int flash_train_gdn_chunk_bf16_backward_launch(const FlashTrainGDNChunkBackwardArgs* args, void* stream);

#ifdef __cplusplus
}
#endif
