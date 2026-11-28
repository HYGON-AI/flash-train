#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Chunk-parallel stages of the large-shape forward route (chunk = 64, layouts as
// FLA).  See flash_train_gdn_chunk_parallel.hip.  Return: 0 ok, 1 null pointer,
// 2 unsupported shape, 5 HIP error.

typedef struct FlashTrainGDNIntraArgs {
    const uint16_t* k;     // [B,S,H,K]  bf16
    const uint16_t* v;     // [B,S,HV,V] bf16
    const float* g;        // [B,S,HV]   fp32, raw log gate (natural log)
    const float* beta;     // [B,S,HV]   fp32
    uint16_t* w;           // [B,S,HV,K] bf16
    uint16_t* u;           // [B,S,HV,V] bf16
    float* gc;             // [B,S,HV]   fp32, chunk-local cumsum in log2 units
    int B, S, H, HV, K, V;
    uint16_t* A;           // optional: [B,S,HV,64] bf16, FLA layout, full (I+M)^-1 lower triangle
                           // incl. diagonal, zeros above; written only when non-null (backward
                           // recompute).  Appended last so zero-initialised callers get nullptr.
} FlashTrainGDNIntraArgs;

typedef struct FlashTrainGDNChunkOArgs {
    const uint16_t* q;     // [B,S,H,K]  bf16
    const uint16_t* k;     // [B,S,H,K]  bf16
    const uint16_t* v_new; // [B,S,HV,V] bf16
    const uint16_t* h;     // [B,S/64,HV,V,K] bf16 (V-first, state_h h_vk=1)
    const float* gc;       // [B,S,HV]   fp32, log2 units
    uint16_t* o;           // [B,S,HV,V] bf16
    int B, S, H, HV, K, V;
    float scale;
} FlashTrainGDNChunkOArgs;

int flash_train_gdn_intra_bf16(const FlashTrainGDNIntraArgs* args, void* stream);
int flash_train_gdn_chunk_o_bf16(const FlashTrainGDNChunkOArgs* args, void* stream);

#ifdef __cplusplus
}
#endif
