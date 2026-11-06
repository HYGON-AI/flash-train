#ifndef FTRAIN_HYGON_HELPER_KERNELS_H
#define FTRAIN_HYGON_HELPER_KERNELS_H

#include "hip/hip_runtime.h"
#include <stdint.h>

namespace bytedance
{
namespace hygon_vendor
{
void cudaipc_barrier_all_on_stream_impl(hipStream_t stream,
                                        int32_t **sync_buffer_ptr,
                                        int rank,
                                        int world_size);

void ag_copy_remote_b16_impl(int grid_size,
                             int block_size,
                             int vec_size,
                             int world_size,
                             int rank,
                             int rows,
                             int cols,
                             void **pp_srcs,
                             int *barrier_ptr,
                             hipStream_t stream);

void ag_copy_remote_b16_chunk_impl(int grid_size,
                                   int block_size,
                                   int vec_size,
                                   int world_size,
                                   int rank,
                                   int rows,
                                   int cols,
                                   int chunk_start,
                                   int chunk_rows,
                                   void **pp_srcs,
                                   hipStream_t stream);

void ag_copy_remote_b16_chunks_signal_impl(int grid_size,
                                           int block_size,
                                           int vec_size,
                                           int world_size,
                                           int rank,
                                           int rows,
                                           int cols,
                                           int chunk_rows,
                                           int signal_value,
                                           int *barrier_ptr,
                                           void **pp_srcs,
                                           hipStream_t stream);

void ag_copy_remote_b16_chunks_signal_impl(int grid_size,
                                           int block_size,
                                           int vec_size,
                                           int world_size,
                                           int rank,
                                           int rows,
                                           int cols,
                                           int chunk_rows,
                                           int *barrier_ptr,
                                           void **pp_srcs,
                                           hipStream_t stream);

void ag_signal_local_chunks_impl(int block_size,
                                 int world_size,
                                 int rank,
                                 int rows,
                                 int chunk_rows,
                                 int signal_value,
                                 int *barrier_ptr,
                                 hipStream_t stream);

void ag_copy_remote_b16_chunks_signal_skip_local_impl(int grid_size,
                                                      int block_size,
                                                      int vec_size,
                                                      int world_size,
                                                      int rank,
                                                      int rows,
                                                      int cols,
                                                      int chunk_rows,
                                                      int signal_value,
                                                      int *barrier_ptr,
                                                      void **pp_srcs,
                                                      hipStream_t stream);

void ag_copy_remote_b16_chunks_signal_stages_impl(int grid_size,
                                                  int block_size,
                                                  int vec_size,
                                                  int world_size,
                                                  int rank,
                                                  int rows,
                                                  int cols,
                                                  int chunk_rows,
                                                  int signal_value,
                                                  int stage_begin,
                                                  int stage_end,
                                                  int *barrier_ptr,
                                                  void **pp_srcs,
                                                  hipStream_t stream);

void ag_copy_ring_b16_chunks_signal_impl(int grid_size,
                                         int block_size,
                                         int vec_size,
                                         int world_size,
                                         int rank,
                                         int rows,
                                         int cols,
                                         int chunk_rows,
                                         int signal_value,
                                         int *local_signal,
                                         int *remote_signal,
                                         const void *stage1_src,
                                                  void **pp_srcs,
                                                  hipStream_t stream);

void ag_copy_remote_b16_chunks_signal_stages_grouped_impl(
    int grid_size,
    int block_size,
    int vec_size,
    int world_size,
    int rank,
    int rows,
    int cols,
    int chunk_rows,
    int signal_value,
    int stage_begin,
    int stage_end,
    int *barrier_ptr,
    void **pp_srcs,
    hipStream_t stream,
    int group_chunks);

void ag_copy_remote_ragged_ring_payload_acquire_impl(
    int grid_size,
    int block_size,
    const void *payload,
    size_t bytes,
    int *internal_signal,
    int *owner_signal,
    int *completion_counter,
    uint32_t signal_value,
    int probe_stride_words,
    int probe_mode,
    hipStream_t stream);

void ag_copy_remote_ragged_ring_light_acquire_group_impl(
    int *local_signals,
    uint32_t signal_value,
    int owner_begin,
    int owner_count,
    int rank,
    int body_slot_base,
    int *ready_mask,
    hipStream_t stream);

void ag_copy_remote_s8_impl(int grid_size,
                            int block_size,
                            int vec_size,
                            int world_size,
                            int rank,
                            int rows,
                            int cols,
                            void **pp_srcs,
                            void **pp_scales,
                            int *barrier_ptr,
                            hipStream_t stream,
                            bool all_gather_scale_a);

void ag_copy_remote_fp8_impl(int grid_size,
                             int block_size,
                             int vec_size,
                             int world_size,
                             int rank,
                             int rows,
                             int cols,
                             void **pp_srcs,
                             void **pp_scales,
                             int *barrier_ptr,
                             hipStream_t stream,
                             bool all_gather_scale_a);
} // namespace hygon_vendor
} // namespace bytedance

#endif // FTRAIN_HYGON_HELPER_KERNELS_H
