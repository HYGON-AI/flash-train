#ifndef FTRAIN_MOE_FC2_GEMM_H
#define FTRAIN_MOE_FC2_GEMM_H

#include "hip/hip_runtime.h"

namespace ftrain
{
namespace moe
{

void moe_fc2_prepare_input(const void *input,
                           void *packed_input,
                           const int *sorted_gather_index,
                           const int *rank_tile_info,
                           int *contract_tile_info,
                           int sorted_capacity,
                           int input_rows,
                           int k_dim,
                           int problem_count,
                           hipStream_t stream);

void moe_fc2_gemm_forward(const void *packed_input,
                          const void *weight,
                          const int *sorted_token_ids,
                          const int *contract_tile_info,
                          void *output,
                          int *chunk_done_counts,
                          int *chunk_done_flags,
                          int problem_count,
                          int output_rows,
                          int n_dim,
                          int k_dim,
                          int chunk_size_n,
                          int num_done_groups,
                          hipStream_t stream,
                          bool weight_nk = false);

} // namespace moe
} // namespace ftrain

#endif
