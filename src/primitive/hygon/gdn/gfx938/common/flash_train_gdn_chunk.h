#pragma once

#include "gfx938_device_symbols.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Chunked/WY GDN forward.  Shares the shape struct fields with
// FlashTrainGDNShape but is declared independently so the chunked engine can be
// built and tested without linking the recurrent kernels.
typedef struct FlashTrainGDNChunkShape {
    int B;
    int S;
    int H;
    int HV;
    int K;
    int V;
    int chunk_size;  // 0 selects the default (32)
} FlashTrainGDNChunkShape;

enum FlashTrainGDNChunkStatus {
    FLASH_TRAIN_GDN_CHUNK_SUCCESS = 0,
    FLASH_TRAIN_GDN_CHUNK_INVALID_POINTER = 1,
    FLASH_TRAIN_GDN_CHUNK_UNSUPPORTED_CONTRACT = 2,
    FLASH_TRAIN_GDN_CHUNK_HIP_ERROR = 5,
};

enum FlashTrainGDNChunkForwardRoute {
    FLASH_TRAIN_GDN_CHUNK_ROUTE_NONE = 0,
    FLASH_TRAIN_GDN_CHUNK_ROUTE_FUSED = 1,
    // Large shapes: intra -> state_h -> chunk_o (chunk 64, bf16, no state I/O).
    FLASH_TRAIN_GDN_CHUNK_ROUTE_FOUR_KERNEL = 3,
};

typedef struct FlashTrainGDNChunkForwardConfig {
    int route;
    int chunk_size;
    int key_tile;
    int value_tile;
    int threads;
    int grid_blocks;
} FlashTrainGDNChunkForwardConfig;

// Returns 1 when the chunked engine covers this shape, 0 otherwise.  Shapes it
// declines (K > 128, V > 256 off the native grid, HV % H != 0, ...) stay on the
// recurrent kernels, which already beat FLA there.
//
// Native shapes (S % 32 == 0, V a multiple of 16, chunk_size 0 or 32, K <= 128) run as they are.
// Every other shape with S >= 1, 1 <= K <= 128, 1 <= V <= 256, HV % H == 0 and chunk_size in
// {0, 32, 64} is also covered: the launcher zero-pads q, k, v, g, beta (and initial_state) to
//   Sp = ceil(S / 64) * 64,  Kp = (K <= 64 ? 64 : 128),  Vp = (V <= 64 ? 64 : V <= 128 ? 128 : 256),
// runs the native route on (B, Sp, H, HV, Kp, Vp) and crops output and final_state back.  Padded
// tokens carry q = k = v = 0, g = 0, beta = 0, so state and valid outputs equal those of the original
// shape; `scale` is used as given (it is not derived from Kp).  Native shapes are never padded.
// The padding path does NOT export per-chunk states: a launch with chunk_states != nullptr on a
// padded shape returns FLASH_TRAIN_GDN_CHUNK_UNSUPPORTED_CONTRACT, and
// flash_train_gdn_chunk_states_bytes() is 0 for it.
int flash_train_gdn_chunk_supported(FlashTrainGDNChunkShape shape);

// Returns the exact launch configuration used by the forward dispatcher.
// The query and the launcher share one implementation, so diagnostics cannot
// report a different VT/KT/thread route from the kernel that actually runs.
// For a shape that runs through the padding path (see above) the configuration is that of the
// PADDED launch: route / key_tile / value_tile / threads / grid_blocks / chunk_size are those the
// native route reports for (B, Sp, H, HV, Kp, Vp); the pad and crop copies are extra launches.
int flash_train_gdn_chunk_forward_config(
    FlashTrainGDNChunkShape shape, FlashTrainGDNChunkForwardConfig* config);
const char* flash_train_gdn_chunk_forward_route_string(int route);

// Compatibility helper for callers that only need the V tile.
int flash_train_gdn_chunk_value_tile(FlashTrainGDNChunkShape shape);

// Bytes of per-chunk state the forward must retain for the backward pass.
// Pass a null chunk_states pointer to the launcher to skip the writes.  Returns 0 for shapes that
// run through the padding path (they cannot export chunk states).
size_t flash_train_gdn_chunk_states_bytes(FlashTrainGDNChunkShape shape);

// Diagnostic only.  Restricts the forward kernel to a subset of its phases so
// their cost can be attributed individually; -1 (the default) runs all of them.
// Results are meaningless for any mask other than -1.
void flash_train_gdn_chunk_set_phase_mask(int mask);

int flash_train_gdn_chunk_fp32_forward_launch(
    const float* q, const float* k, const float* v, const float* g,
    const float* beta, const float* initial_state, float* output,
    float* final_state, float* chunk_states, FlashTrainGDNChunkShape shape,
    float scale, void* stream);

int flash_train_gdn_chunk_bf16_forward_launch(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* g,
    const float* beta, const float* initial_state, uint16_t* output,
    float* final_state, float* chunk_states, FlashTrainGDNChunkShape shape,
    float scale, void* stream);

#ifdef __cplusplus
}
#endif
