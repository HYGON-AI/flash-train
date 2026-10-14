#ifndef FTRAIN_HYGON_MOE_REDUCE_RS_KERNELS_H
#define FTRAIN_HYGON_MOE_REDUCE_RS_KERNELS_H

#include "hip/hip_runtime.h"
#include <c10/core/ScalarType.h>
#include <stdint.h>

namespace bytedance
{
namespace hygon_vendor
{

void ag_group_gemm_recovered_gather_impl(void **input_ptrs,
                                         int world_size,
                                         int rank,
                                         int full_m,
                                         int k_dim,
                                         hipStream_t stream);

void ag_group_gemm_recovered_forward_impl(const void *full_input,
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

void ag_group_gemm_rebuilt_prepare_metadata_impl(const void *topk_ids,
                                                 int *sorted_token_ids,
                                                 int *expert_counts,
                                                 int *expert_counters,
                                                 int *expert_offsets,
                                                 int *tile_experts,
                                                 int *tile_starts,
                                                 int *ntiles,
                                                 int full_m,
                                                 int topk,
                                                 int num_experts,
                                                 int sorted_capacity,
                                                 c10::ScalarType index_dtype,
                                                 hipStream_t stream);

void ag_group_gemm_rebuilt_prepare_rank_tile_metadata_impl(
    const void *topk_ids,
    int *sorted_token_ids,
    int *expert_counts,
    int *expert_counters,
    int *expert_offsets,
    int *rank_tile_info,
    int *ntiles,
    int full_m,
    int topk,
    int num_experts,
    int sorted_capacity,
    int max_rank_tiles,
    int world_size,
    c10::ScalarType index_dtype,
    hipStream_t stream);

void ag_group_gemm_rebuilt_prepare_rank_tile_metadata_bm_impl(
    const void *topk_ids,
    int *sorted_token_ids,
    int *expert_counts,
    int *expert_counters,
    int *expert_offsets,
    int *rank_tile_info,
    int *ntiles,
    int full_m,
    int topk,
    int num_experts,
    int sorted_capacity,
    int max_rank_tiles,
    int world_size,
    int block_m,
    c10::ScalarType index_dtype,
    hipStream_t stream);

void ag_group_gemm_rebuilt_rank_tile_direct_impl(const void *full_input,
                                                 const void *weight,
                                                 const int *sorted_token_ids,
                                                 const int *expert_offsets,
                                                 const int *rank_tile_info,
                                                 const int *ntiles,
                                                 const int *rank_signals,
                                                 void *output,
                                                 int full_m,
                                                 int n_dim,
                                                 int k_dim,
                                                 int topk,
                                                 int world_size,
                                                 int max_rank_tiles,
                                                 c10::ScalarType dtype,
                                                 hipStream_t stream);

void ag_group_gemm_rebuilt_rank_tile_mmac16_impl(const void *full_input,
                                                 const void *weight,
                                                 const int *sorted_token_ids,
                                                 const int *expert_offsets,
                                                 const int *rank_tile_info,
                                                 const int *ntiles,
                                                 const int *rank_signals,
                                                 void *output,
                                                 int full_m,
                                                 int n_dim,
                                                 int k_dim,
                                                 int topk,
                                                 int world_size,
                                                 int max_rank_tiles,
                                                 c10::ScalarType dtype,
                                                 hipStream_t stream);

void ag_group_gemm_rebuilt_rank_tile_mmac16_dsread_impl(
    const void *full_input,
    const void *weight,
    const int *sorted_token_ids,
    const int *expert_offsets,
    const int *rank_tile_info,
    const int *ntiles,
    const int *rank_signals,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int world_size,
    int max_rank_tiles,
    c10::ScalarType dtype,
    hipStream_t stream);

void ag_group_gemm_rebuilt_tiled_forward_impl(const void *full_input,
                                              const void *weight,
                                              const int *sorted_token_ids,
                                              const int *tile_experts,
                                              const int *tile_starts,
                                              const int *ntiles,
                                              const int *rank_signals,
                                              void *output,
                                              int full_m,
                                              int n_dim,
                                              int k_dim,
                                              int topk,
                                              int world_size,
                                              int max_tiles,
                                              c10::ScalarType dtype,
                                              hipStream_t stream);

void ag_group_gemm_rebuilt_pack_weight_nk_impl(const void *weight_kn,
                                               void *weight_nk,
                                               int num_experts,
                                               int n_dim,
                                               int k_dim,
                                               c10::ScalarType dtype,
                                               hipStream_t stream);

void ag_group_gemm_rebuilt_pack_sorted_input_impl(const void *full_input,
                                                  const int *sorted_rows,
                                                  void *packed_input,
                                                  int total_rows,
                                                  int k_dim,
                                                  int full_m,
                                                  int topk,
                                                  int world_size,
                                                  const int *rank_signals,
                                                  c10::ScalarType dtype,
                                                  hipStream_t stream);

void ag_group_gemm_rebuilt_scatter_packed_output_impl(const void *packed_output,
                                                      const int *sorted_rows,
                                                      void *output,
                                                      int total_rows,
                                                      int n_dim,
                                                      c10::ScalarType dtype,
                                                      hipStream_t stream);

void ag_group_gemm_build_root_s_grouped_args_impl(void *args_blob,
                                                  const int *tile_experts,
                                                  const int *tile_starts,
                                                  int problem_count,
                                                  void *packed_output,
                                                  const void *weight_nk,
                                                  const void *packed_input,
                                                  int n_dim,
                                                  int k_dim,
                                                  hipStream_t stream);

// ABI recovered from the original HCU objects. The reconstructed host control
// flow uses these entry points while their HIP sources are recovered.
void moe_reduce_rs_prepare_counts(const void *chosen_experts,
                                  int *expert_counts,
                                  int ntokens,
                                  int num_experts,
                                  int topk,
                                  bool index_is_i64,
                                  hipStream_t stream);

void moe_reduce_rs_set_done_flags(int *done_flags,
                                  int group,
                                  hipStream_t stream);

void moe_reduce_rs_scale_output(void *output,
                                const void *scale,
                                int rows,
                                int n_dim,
                                c10::ScalarType dtype,
                                hipStream_t stream);

// Transformer Engine grouped GEMM returns one expert-major packed matrix.
// Restore token/top-k order and apply the router weights in one pass.
void te_moe_unpermute_topk_reduce(const void *packed_output,
                                  const int32_t *inverse_sorted_rows,
                                  const void *expert_weight,
                                  void *partial,
                                  int ntokens,
                                  int n_dim,
                                  int topk,
                                  c10::ScalarType dtype,
                                  hipStream_t stream);

// Source MT256 compatibility epilogue. A grouped GEMM may write flattened
// top-k rows; this kernel materializes the FP32 per-token top-k stage and can
// publish the same logical N-group completion flags consumed by A2A/Ring/PG.
// The NVIDIA-aligned single-node fast path fuses this reduction into RS.
void moe_reduce_rs_build_topk_stage(
    const void *gemm_output,
    const void *expert_weight,
    void *topk_stage,
    const int *gemm_done_flags,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    int ntokens,
    int n_dim,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_value,
    c10::ScalarType dtype,
    hipStream_t stream);

void moe_reduce_rs_topk_reduce_a2a(const void *gemm_output,
                                   const void *expert_weight,
                                   void **reduce_buffer_ptrs,
                                   int *gemm_done_flags,
                                   int ntokens,
                                   int n_dim,
                                   int rank,
                                   int world_size,
                                   int topk,
                                   int chunk_size_n,
                                   int num_groups,
                                   c10::ScalarType dtype,
                                   hipStream_t stream);

// Experimental two-phase scheduler used to measure whether limiting the
// period in which scatter and a persistent AG GEMM execute concurrently can
// recover more GEMM time than it exposes as a post-GEMM communication tail.
void moe_reduce_rs_topk_reduce_a2a_range(const void *gemm_output,
                                         const void *expert_weight,
                                         void **reduce_buffer_ptrs,
                                         int *gemm_done_flags,
                                         int ntokens,
                                         int n_dim,
                                         int rank,
                                         int world_size,
                                         int topk,
                                         int chunk_size_n,
                                         int first_group,
                                         int group_count,
                                         int block_budget,
                                         c10::ScalarType dtype,
                                         hipStream_t stream);

// Work-conserving paced variant of the ordinary A2A range launcher.  The
// producer's next logical N-group flag acts as admission feedback: scatter
// runs at full speed while GEMM is ahead, and inserts short wave sleeps when
// it catches the producer.  The policy depends on device progress rather than
// a shape threshold and is kept behind an experimental host selector.
void moe_reduce_rs_topk_reduce_a2a_paced_range(
    const void *gemm_output,
    const void *expert_weight,
    void **reduce_buffer_ptrs,
    int *gemm_done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int first_group,
    int group_count,
    int block_budget,
    int check_iterations,
    int sleep_steps,
    c10::ScalarType dtype,
    hipStream_t stream);

// Stream-admitted ordinary scatter.  One short grid is submitted for each
// logical N-group after a non-resident stream wait observes the AG flag.  It
// deliberately leaves the cross-rank barrier and full-N sum in the existing
// tail path so this experiment isolates producer/scatter contention.
void moe_reduce_rs_topk_reduce_a2a_admitted(
    const void *gemm_output,
    const void *expert_weight,
    void **reduce_buffer_ptrs,
    int *gemm_done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int batch_groups,
    int early_block_budget,
    int final_block_budget,
    c10::ScalarType dtype,
    hipStream_t stream);

// v36-only producer-lead admission. The completion counters live after the
// dynamic task-claim map and expose partial progress for every logical N
// group. A scatter group is admitted only after the producer has advanced
// into a later group, avoiding a resident wait grid while WGM8 is busiest.
void moe_reduce_rs_topk_reduce_a2a_adaptive_admitted(
    const void *gemm_output,
    const void *expert_weight,
    void **reduce_buffer_ptrs,
    int *gemm_done_flags,
    int *completion_counts,
    int problem_count,
    int physical_n_tiles,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int lead_groups,
    int lead_percent,
    int early_block_budget,
    int final_block_budget,
    c10::ScalarType dtype,
    hipStream_t stream);

void moe_reduce_rs_topk_reduce_a2a_chunk(const void *gemm_output,
                                         const void *expert_weight,
                                         void **reduce_buffer_ptrs,
                                         int *gemm_done_flags,
                                         int ntokens,
                                         int n_dim,
                                         int rank,
                                         int world_size,
                                         int topk,
                                         int group,
                                         int chunk_offset_n,
                                         int chunk_size_n,
                                         c10::ScalarType dtype,
                                         hipStream_t stream);

void moe_reduce_rs_topk_stage_scatter_a2a(const void *topk_stage,
                                          void **reduce_buffer_ptrs,
                                          int *gemm_done_flags,
                                          int ntokens,
                                          int n_dim,
                                          int rank,
                                          int world_size,
                                          int chunk_size_n,
                                          int num_groups,
                                          c10::ScalarType dtype,
                                          hipStream_t stream);

// Experimental NVIDIA-style fused A2A.  One resident grid consumes every N
// group, scatters to the peer IPC buffers, executes the two all-rank barriers,
// and sums the local reduce-scatter slice.  Exactly one of gemm_output and
// topk_stage must be non-null.  When hip_cooperative_launch is true the kernel
// is launched with hipLaunchCooperativeKernel and uses this_grid().sync();
// otherwise grid_barrier points to two zero-initialized device integers owned
// by this rank (arrival counter and epoch) for the software-barrier fallback.
// group_pipeline enables the experimental per-N-group scatter/IPC/sum loop;
// it is intentionally off by default because its per-group IPC and ordering
// cost can still exceed the overlap they expose on gfx936.
void moe_reduce_rs_a2a_cooperative_fused(const void *gemm_output,
                                         const void *expert_weight,
                                         const void *topk_stage,
                                         void **reduce_buffer_ptrs,
                                         int **sync_buffer_ptrs,
                                         int *grid_barrier,
                                         int *gemm_done_flags,
                                         void *output,
                                         // Null for the fixed-M layout. For a
                                         // ragged layout this is the device
                                         // prefix array of length world + 1.
                                         const int *rank_m_offsets,
                                         int local_rows,
                                         int ntokens,
                                         int n_dim,
                                         int rank,
                                         int world_size,
                                         int topk,
                                         int chunk_size_n,
                                         int num_groups,
                                         int epoch,
                                         c10::ScalarType dtype,
                                         bool hip_cooperative_launch,
                                         bool group_pipeline,
                                         hipStream_t stream);

// Device-signaled A2A pipeline.  The scatter and sum stages are launched as
// separate persistent kernels on independent streams.  The scatter kernel
// publishes one ready epoch per N group in the source rank's signal buffer;
// the sum kernel waits for all source ranks before reducing that group.  This
// removes the all-groups barrier from the NVIDIA-style fused fallback while
// allowing sum(group g) to overlap scatter(group g+1) and the next GEMM tile.
void moe_reduce_rs_a2a_signal_pipeline(const void *gemm_output,
                                       const void *expert_weight,
                                       const void *topk_stage,
                                       void **reduce_buffer_ptrs,
                                       void **signal_buffer_ptrs,
                                       const int *gemm_done_flags,
                                       void *output,
                                       int ntokens,
                                       int n_dim,
                                       int rank,
                                       int world_size,
                                       int topk,
                                       int chunk_size_n,
                                       int num_groups,
                                       int signal_stride,
                                       int epoch,
                                       int block_budget,
                                       c10::ScalarType dtype,
                                       hipStream_t scatter_stream,
                                       hipStream_t sum_stream);

// Split-budget persistent pipeline. The producer and consumer kernels keep
// the existing device-signal protocol, but use independent CTA budgets so a
// small number of waiting sum CTAs do not reserve as much compute capacity as
// the scatter producer. This is an experimental resource-partitioned route.
void moe_reduce_rs_a2a_signal_pipeline_split(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int producer_block_budget,
    int consumer_block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream);

// Stream-wait variant of the A2A signal pipeline.  The scatter side remains
// one persistent producer, but the sum side is submitted as one ordinary
// kernel per N group.  A HIP stream wait-value command precedes each group,
// so a waiting consumer does not occupy resident wavefronts while GEMM is
// still producing a group.  signal_buffer_ptrs_host is the host-side array of
// IPC-visible signal pointers; signal_buffer_ptrs remains the device array
// consumed by the producer kernel.
void moe_reduce_rs_a2a_signal_pipeline_stream_wait(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream);

// Hardware-wait group pipeline.  Unlike the persistent signal pipeline, the
// scatter side is split into one ordinary launch per N group.  Each launch is
// preceded by hipStreamWaitValue32 on the AG completion flag, so no scatter
// CTA spins while the rank-tile GEMM is still producing that group.  The sum
// side uses the same stream-ordered waits on all source-rank ready epochs.
// Producer and consumer budgets are independent so large, finely chunked
// problems can reserve enough scatter throughput without forcing the smaller
// sum stage to use the same CTA count.
void moe_reduce_rs_a2a_signal_pipeline_hardware_wait(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    int early_scatter_block_budget,
    int early_sum_block_budget,
    int full_budget_start_group,
    bool wave64_512_scatter,
    bool wave64_512_sum,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream);

// Ragged-M counterpart of the hardware-wait A2A pipeline.  rank_m_offsets
// contains the exclusive source-token partition offsets [world_size + 1].
// The producer writes a compact (producer_rank, destination_local_row)
// layout into each destination IPC buffer, and the consumer reduces only the
// local rank's variable number of rows.  This preserves the per-N-group
// stream-ordered signaling used by the equal-M pipeline without padding the
// grouped GEMM input or output.
void moe_reduce_rs_a2a_signal_pipeline_ragged_hardware_wait(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    const int *rank_m_offsets,
    void *output,
    int ntokens,
    int local_rows,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream);

// Stream-wait variant of the ragged pipeline.  The scatter side is one
// persistent producer that spins on the AG done flags in-kernel, while the sum
// side keeps the per-N-group stream waits.  Shapes where the fixed-M policy
// does not enable hardware waiting use this scheduling instead of failing.
void moe_reduce_rs_a2a_signal_pipeline_ragged_stream_wait(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    const int *rank_m_offsets,
    void *output,
    int ntokens,
    int local_rows,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream);


// Cast one N chunk of the FP32 top-k stage after the corresponding AG
// rank-tile signal is published.  The destination is compact and therefore
// suitable for a per-chunk ProcessGroup reduce-scatter input view.

void moe_reduce_rs_sum_a2a(void **reduce_buffer_ptrs,
                           void *output,
                           int ntokens_per_rank,
                           int n_dim,
                           int rank,
                           int world_size,
                           c10::ScalarType dtype,
                           hipStream_t stream);

void moe_reduce_rs_sum_a2a_chunk(void **reduce_buffer_ptrs,
                                 void *output,
                                 int ntokens_per_rank,
                                 int n_dim,
                                 int rank,
                                 int world_size,
                                 int chunk_offset_n,
                                 int chunk_size_n,
                                 c10::ScalarType dtype,
                                 hipStream_t stream);



// Experimental persistent Ring.  One launch executes every Ring step for a
// single N chunk and uses symmetric IPC signal buffers for device-side
// readiness hand-off between adjacent ranks.


// NVIDIA-style persistent Ring.  A single resident grid consumes every
// logical N group and executes every Ring step, waiting on the corresponding
// grouped-GEMM completion flag before entering each group.


void ag_group_gemm_b16_mmac_rank_tile_impl(
    const void *input,
    const void *weight,
    const void *topk_ids,
    void *gemm_output,
    int *rank_signals,
    const int *expert_offsets,
    const int *local_expert_offsets,
    const int *expert_rank_bounds,
    int *metadata_scratch0,
    int *metadata_scratch1,
    int *metadata_scratch2,
    int routed_rows,
    const int *rank_tile_info,
    int rank_tile_count,
    int full_m,
    int n_dim,
    int k_dim,
    int num_experts,
    int world_size,
    int local_world_size,
    int rank,
    bool index_is_i64,
    bool use_rank_tile,
    int block_m,
    int group_m,
    bool reserved_bool,
    int signal_value,
    hipStream_t stream,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    int chunk_size_n,
    int num_done_groups,
    const void *expert_weight,
    void *topk_stage,
    int topk,
    int *reserved_ptr,
    int reserved_int);

void moe_reduce_rs_recovered_compute_impl(const void *input,
                                          const void *weight,
                                          const void *chosen_experts,
                                          const void *expert_weight,
                                          void *partial,
                                          int ntokens,
                                          int n_dim,
                                          int k_per_rank,
                                          int num_experts,
                                          int topk,
                                          c10::ScalarType dtype,
                                          c10::ScalarType index_dtype,
                                          hipStream_t stream);

void moe_reduce_rs_recovered_sum_impl(void **partial_ptrs,
                                      void *output,
                                      int world_size,
                                      int rank,
                                      int ntokens,
                                      int n_dim,
                                      c10::ScalarType dtype,
                                      hipStream_t stream);

} // namespace hygon_vendor
} // namespace bytedance

#endif // FTRAIN_HYGON_MOE_REDUCE_RS_KERNELS_H
