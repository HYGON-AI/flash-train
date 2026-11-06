#ifndef FTRAIN_HYGON_AG_GROUP_GEMM_KERNELS_H
#define FTRAIN_HYGON_AG_GROUP_GEMM_KERNELS_H

#include "hip/hip_runtime.h"
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <stdint.h>

namespace bytedance
{
namespace hygon_vendor
{

void ag_group_gemm_update_rank_m_offsets_impl(int *device_offsets,
                                              const int *host_offsets,
                                              int count,
                                              hipStream_t stream);

void ag_group_gemm_prepare_metadata_impl(const void *topk_ids,
                                         int *sorted_token_ids,
                                         int *sorted_gather_index,
                                         int *expert_rank_counts,
                                         int *expert_rank_counters,
                                         int *expert_rank_offsets,
                                         int *global_expert_offsets,
                                         int *rank_cursors,
                                         int *rank_tile_info,
                                         int *tile_experts,
                                         int *tile_starts,
	                                         int *tile_rank_starts,
	                                         int *tile_rank_ends,
	                                         int *rank_tile_problem_info,
	                                         int *ntiles,
	                                         const int *rank_m_offsets,
	                                         int full_m,
	                                         int topk,
	                                         int num_experts,
	                                         int world_size,
	                                         int rank,
	                                         int sorted_capacity,
	                                         int rank_tile_capacity,
                                         int signal_chunk_rows,
	                                         int chunks_per_rank,
                                         bool strict_rank_tile,
                                         bool split_local_rank,
                                         bool build_problem_info,
                                         bool stage_signal_layout,
                                         bool stage_signal_zigzag,
	                                         c10::ScalarType index_dtype,
	                                         hipStream_t stream);

// MoEReduceRS supplies one independent K-shard row for every flattened
// (token, topk) entry. Keep its reconstructed flat-input metadata contract
// separate from the standalone AGGroupGemm route.
void ag_group_gemm_prepare_metadata_flat_input_impl(
    const void *topk_ids,
    int *sorted_token_ids,
    int *sorted_gather_index,
    int *expert_rank_counts,
    int *expert_rank_counters,
    int *expert_rank_offsets,
    int *global_expert_offsets,
    int *rank_cursors,
    int *rank_tile_info,
    int *tile_experts,
    int *tile_starts,
    int *tile_rank_starts,
    int *tile_rank_ends,
    int *rank_tile_problem_info,
    int *ntiles,
    int full_m,
    int topk,
    int num_experts,
    int world_size,
    int rank,
    int sorted_capacity,
    int rank_tile_capacity,
    int signal_chunk_rows,
    int chunks_per_rank,
    bool strict_rank_tile,
    bool build_problem_info,
    c10::ScalarType index_dtype,
    hipStream_t stream);

void ag_group_gemm_flat_input_forward_impl(
    const void *flat_input,
    const void *weight_nk,
    const void *weight_kn,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *ntiles,
    const int *rank_signals,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    int *v36_task_control,
    int v36_task_control_capacity,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int max_tiles,
    int signal_value,
    int chunk_size_n,
    int num_done_groups,
    c10::ScalarType dtype,
    hipStream_t stream,
    const void *router_scale = nullptr);

// Returns the concrete v36 producer selected for this shape. Auto-direct uses
// global token M; explicit WGM8/WGM1 selectors remain force-test routes.
bool ag_group_gemm_mt256_uses_v36_wgm1_direct(int full_m,
                                               int n_dim,
                                               bool native_kn_weight = false);

// Returns true when the selected source-only MT256 variant consumes the
// public AGGroupGemm weight directly as contiguous [E,K,N]. The legacy TN
// variants consume the cached [E,N,K] packed copy instead.
bool ag_group_gemm_mt256_uses_native_kn_weight(int full_m, int n_dim);

// Returns true for the independent Megatron FC1 Native-KN chunk-ready route.
bool ag_group_gemm_mt256_uses_native_kn_chunk(int full_m, int n_dim);

// Standalone AGGroupGemm bridge for source-only native-KN variants. Routed
// input packing remains an independent selector choice.
void ag_group_gemm_native_kn_tiled_forward_impl(
    const void *full_input,
    const void *weight_kn,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *tile_experts,
    const int *tile_starts,
    const int *tile_rank_starts,
    const int *tile_rank_ends,
    const int *rank_tile_problem_info,
    const int *ntiles,
    const int *rank_signals,
    int *rank_tile_ready_flags,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int world_size,
    int rank,
    int max_tiles,
    int signal_value,
    int signal_chunk_rows,
    int chunks_per_rank,
    bool ragged_m,
    c10::ScalarType dtype,
    hipStream_t stream);

// Standalone AGGroupGemm bridge for the source-only legacy-TN variants.  The
// public weight has already been cached as contiguous [E,N,K] by the caller;
// keep this bridge separate from the native-KN path so both layouts can be
// correctness/performance tested through the same reconstructed launcher.
void ag_group_gemm_legacy_tn_tiled_forward_impl(
    const void *full_input,
    const void *weight_nk,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *tile_experts,
    const int *tile_starts,
    const int *tile_rank_starts,
    const int *tile_rank_ends,
    const int *ntiles,
    const int *rank_signals,
    int *rank_tile_ready_flags,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int world_size,
    int rank,
    int max_tiles,
    int signal_value,
    int signal_chunk_rows,
    int chunks_per_rank,
    bool ragged_m,
    c10::ScalarType dtype,
    hipStream_t stream);

void ag_group_gemm_v4_flat_input_forward_impl(
    const void *flat_input,
    const void *weight_nk,
    const void *weight_kn,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *ntiles,
    const int *rank_signals,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int max_tiles,
    int signal_value,
    int chunk_size_n,
    int num_done_groups,
    c10::ScalarType dtype,
    hipStream_t stream);

void ag_group_gemm_tiled_forward_impl(const void *full_input,
                                      const void *weight,
                                      const int *sorted_token_ids,
                                      const int *sorted_gather_index,
                                      const int *rank_tile_info,
                                      const int *tile_experts,
	                                      const int *tile_starts,
	                                      const int *tile_rank_starts,
	                                      const int *tile_rank_ends,
	                                      const int *rank_tile_problem_info,
	                                      const int *ntiles,
	                                      const int *rank_signals,
                                      int *rank_tile_ready_flags,
                                      int *n_chunk_done_counts,
                                      int *n_chunk_done_flags,
                                      void *output,
                                      int full_m,
                                      int n_dim,
	                                      int k_dim,
	                                      int topk,
	                                      int num_experts,
	                                      int world_size,
	                                      int rank,
                                      int max_tiles,
                                      bool weight_nk,
                                      bool metadata_cache_hit,
	                                      bool ragged_m,
                                      int signal_value,
	                                      int signal_chunk_rows,
	                                      int chunks_per_rank,
	                                      c10::ScalarType dtype,
	                                      hipStream_t stream);

void ag_group_gemm_backward_weight_impl(
    const void *full_routed_input,
    const void *gathered_grad_output,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *global_expert_offsets,
    const int *rank_cursors,
    void *compact_routed_input,
    void *compact_grad_output,
    void *grouped_asm_args,
    void *grad_weight,
    int sorted_capacity,
    int full_m,
    int output_dim,
    int reduction_dim,
    int num_experts,
    int topk,
    int world_size,
    int signal_value,
    const int *output_chunk_ready_flags,
    int output_chunk_size,
    c10::ScalarType dtype,
    hipStream_t stream);

void ag_group_gemm_pack_weight_impl(const void *weight,
                                    void *packed_weight,
                                    int num_experts,
                                    int k_dim,
                                    int n_dim,
                                    c10::ScalarType dtype,
                                    hipStream_t stream);

void ag_group_gemm_wait_rank_signals_impl(const int *rank_signals,
                                          int world_size,
                                          int signal_value,
                                          hipStream_t stream);

// Publish a peer-visible readiness epoch from a device kernel.  This is used
// only by the experimental Native-KN overlap path: unlike a host-enqueued
// signal copy, the system-scope release is ordered after the preceding
// hipMemcpyPeerAsync on the same stream.
void ag_group_gemm_publish_signal_release_impl(int *remote_signal,
                                               uint32_t signal_value,
                                               hipStream_t stream);

// Insert a system-scope payload fence between peer DMA and the small peer
// signal copy used by the raw Native-KN overlap route.
void ag_group_gemm_system_fence_impl(hipStream_t stream);

void ag_group_gemm_reference_forward_impl(const void *full_input,
                                          const void *weight,
                                          const void *topk_ids,
                                          void *output,
                                          int full_m,
                                          int n_dim,
                                          int k_dim,
                                          int num_experts,
                                          int topk,
                                          c10::ScalarType dtype,
                                          c10::ScalarType index_dtype,
                                          hipStream_t stream);

std::vector<at::Tensor> ag_payload_dedup_map_impl(
    const at::Tensor &input,
    int topk,
    hipStream_t stream);

at::Tensor ag_payload_output_remap_impl(
    const at::Tensor &source,
    const at::Tensor &row_map,
    at::Tensor destination,
    hipStream_t stream);

at::Tensor ag_payload_scatter_topk_impl(
    const at::Tensor &source,
    const at::Tensor &destination_rows,
    at::Tensor destination,
    hipStream_t stream);

} // namespace hygon_vendor
} // namespace bytedance

#endif // FTRAIN_HYGON_AG_GROUP_GEMM_KERNELS_H
