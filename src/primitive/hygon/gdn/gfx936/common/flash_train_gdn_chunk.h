#pragma once

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
    FLASH_TRAIN_GDN_CHUNK_INSUFFICIENT_WORKSPACE = 4,
    FLASH_TRAIN_GDN_CHUNK_HIP_ERROR = 5,
};

enum FlashTrainGDNChunkForwardRoute {
    FLASH_TRAIN_GDN_CHUNK_ROUTE_NONE = 0,
    FLASH_TRAIN_GDN_CHUNK_ROUTE_FUSED = 1,
};

typedef struct FlashTrainGDNChunkForwardConfig {
    int route;
    int chunk_size;
    int key_tile;
    int value_tile;
    int threads;
    int grid_blocks;
} FlashTrainGDNChunkForwardConfig;

// Returns 1 when the fused engine covers this shape, 0 otherwise.
int flash_train_gdn_chunk_supported(FlashTrainGDNChunkShape shape);

// Returns the exact launch configuration used by the forward dispatcher.
// The query and the launcher share one implementation, so diagnostics cannot
// report a different VT/KT/thread route from the kernel that actually runs.
int flash_train_gdn_chunk_forward_config(
    FlashTrainGDNChunkShape shape, FlashTrainGDNChunkForwardConfig* config);
const char* flash_train_gdn_chunk_forward_route_string(int route);

// Compatibility helper for callers that only need the V tile.
int flash_train_gdn_chunk_value_tile(FlashTrainGDNChunkShape shape);

// Bytes of per-chunk state the forward must retain for the backward pass.
// Pass a null chunk_states pointer to the launcher to skip the writes.
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
