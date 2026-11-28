#pragma once

#include "gfx938_device_symbols.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Three-kernel chunked forward for GDN-2 (per-channel decay).  chunk = 64.
// Status codes follow FlashTrainGDN2Status: 0 ok, 1 null pointer, 2 unsupported
// shape, 4 insufficient workspace, 5 HIP error.  See flash_train_gdn2_chunk.hip.

typedef struct FlashTrainGDN2IntraArgs {
    const uint16_t* q;   // [B,S,H,K]  bf16
    const uint16_t* k;   // [B,S,H,K]  bf16
    const uint16_t* v;   // [B,S,HV,V] bf16
    const float* g;      // [B,S,HV,K] fp32, natural-log decay
    const float* b;      // [B,S,HV,K] fp32, erase gate
    const float* wg;     // [B,S,HV,V] fp32, write gate
    uint16_t* w;         // [B,S,HV,K] bf16  A_inv (k*b*2^gc)
    uint16_t* u;         // [B,S,HV,V] bf16  A_inv (v*wg)
    uint16_t* kg;        // [B,S,HV,K] bf16  k * 2^(gc_last-gc)
    uint16_t* qg;        // [B,S,HV,K] bf16  q * 2^gc
    uint16_t* aqk;       // [B,S,HV,64] bf16 scale * tril(Aqk), j<=i
    float* gclast;       // [B,S/64,HV,K] fp32  chunk-end gc (log2 units)
    float* gc;           // [B,S,HV,K] fp32  optional debug export of gc, may be null
    int B, S, H, HV, K, V;
    float scale;
    // Optional (null = not written, forward unchanged).  Appended at the END of the struct: ctypes callers
    // built against the older layout must add this field.
    uint16_t* akk_inv;   // [B,S,HV,64] bf16  (I+M)^-1 as FLA's Akk: row t, col s<=t, upper triangle 0
} FlashTrainGDN2IntraArgs;

typedef struct FlashTrainGDN2StateHArgs {
    const uint16_t* kg;      // [B,S,HV,K] bf16
    const uint16_t* w;       // [B,S,HV,K] bf16
    const uint16_t* u;       // [B,S,HV,V] bf16
    const float* gclast;     // [B,S/64,HV,K] fp32
    uint16_t* h;             // [B,S/64,HV,V,K] bf16  state at chunk start (V-first)
    uint16_t* v_new;         // [B,S,HV,V] bf16
    int B, S, H, HV, K, V;
} FlashTrainGDN2StateHArgs;

typedef struct FlashTrainGDN2ChunkOArgs {
    const uint16_t* qg;      // [B,S,HV,K] bf16
    const uint16_t* aqk;     // [B,S,HV,64] bf16 (scale already applied)
    const uint16_t* v_new;   // [B,S,HV,V] bf16
    const uint16_t* h;       // [B,S/64,HV,V,K] bf16
    uint16_t* o;             // [B,S,HV,V] bf16
    int B, S, H, HV, K, V;
    float scale;
} FlashTrainGDN2ChunkOArgs;

int flash_train_gdn2_intra_bf16(const FlashTrainGDN2IntraArgs* args, void* stream);
int flash_train_gdn2_state_h_bf16(const FlashTrainGDN2StateHArgs* args, void* stream);
int flash_train_gdn2_chunk_o_bf16(const FlashTrainGDN2ChunkOArgs* args, void* stream);

// Shape gate of the backward-compatible dims: S >= 512, S % 64 == 0, K,V in {64,128} (the backward entry uses this).
int flash_train_gdn2_chunk_shape_supported(int S, int H, int HV, int K, int V);
// Shape gate of the forward route: any S >= 1, K <= 128, V <= 256, HV % H == 0.  Shapes that are not
// (S % 64 == 0, K in {64,128}, V in {64,128,256}) are zero-padded and cropped inside the route.
int flash_train_gdn2_chunk_fwd_shape_supported(int S, int H, int HV, int K, int V);
// 1 if the forward route has to pad/crop this shape.
int flash_train_gdn2_chunk_fwd_needs_pad(int S, int K, int V);
// Bytes of workspace the full forward route needs (0 if the shape is unsupported).  The first form assumes H == HV
// (upper bound for the padded q/k copies); the second is exact.
size_t flash_train_gdn2_chunk_workspace_bytes(int B, int S, int HV, int K, int V);
size_t flash_train_gdn2_chunk_fwd_workspace_bytes(int B, int S, int H, int HV, int K, int V);

// Whole route.  q,k,v bf16; g,b,wg fp32.
int flash_train_gdn2_chunk_forward_bf16(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* g,
    const float* b, const float* wg, uint16_t* out, int B, int S, int H, int HV,
    int K, int V, float scale, void* workspace, size_t workspace_bytes, void* stream);

#ifdef __cplusplus
}
#endif
