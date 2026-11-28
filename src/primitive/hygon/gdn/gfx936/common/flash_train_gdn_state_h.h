#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Stage 3 of the large-shape forward route: serial inter-chunk state recursion.
// Layouts follow FLA's chunk_gated_delta_rule_fwd_h (chunk = 64, g = chunk-local
// cumulative log2 gate).  See flash_train_gdn_state_h.hip.
typedef struct FlashTrainGDNStateHArgs {
    const uint16_t* k;      // [B,S,H,K]  bf16
    const uint16_t* w;      // [B,S,HV,K] bf16
    const uint16_t* u;      // [B,S,HV,V] bf16
    const float* g;         // [B,S,HV]   fp32, chunk-local cumsum, log2 units
    const float* h0;        // [B,HV,K,V] fp32 or null
    uint16_t* h;            // state at each chunk start, layout per h_vk
    uint16_t* v_new;        // [B,S,HV,V] bf16
    float* ht;              // [B,HV,K,V] fp32 or null
    int B, S, H, HV, K, V;
    int bv;                 // tiling selector, see flash_train_gdn_state_h.hip (0 = best)
    int h_vk;               // 0: h [B,NT,HV,K,V] (FLA default); 1: h [B,NT,HV,V,K]
} FlashTrainGDNStateHArgs;

// Returns 0 on success, 1 null pointer, 2 unsupported shape, 5 HIP error.
int flash_train_gdn_state_h_bf16(const FlashTrainGDNStateHArgs* args, void* stream);

#ifdef __cplusplus
}
#endif
