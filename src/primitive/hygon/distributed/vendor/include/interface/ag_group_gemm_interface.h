#ifndef FTRAIN_HYGON_AG_GROUP_GEMM_INTERFACE_H
#define FTRAIN_HYGON_AG_GROUP_GEMM_INTERFACE_H

#include "common_info.h"
#include <ATen/ATen.h>
#include <memory>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/torch.h>
#include <string>
#include <vector>

namespace bytedance
{
namespace hygon_vendor
{

enum class NativeKnRaggedMode
{
    Disabled = 0,
    FixedRing = 1,
    FixedRingReadyMask = 2,
};

class AGGroupGemm
{
public:
    /**
     * @brief AllGather + MoE grouped GEMM for single-node expert tensor parallelism (ETP).
     *
     * Native ABI shape: input [local_m, reduction_dim],
     * weight [num_experts, reduction_dim, output_dim].
     * full_topk_ids: [full_m, topk]
     * output: [full_m * topk, output_dim]
     * rank_m_sizes: optional per-rank local_m values. Empty means uniform
     * full_m / world_size partitioning.
     */
    AGGroupGemm(c10::intrusive_ptr<c10d::ProcessGroup> etp_group,
                int32_t nnodes,
                int32_t full_m,
                int32_t n_dim,
                int32_t k_dim,
                int32_t num_experts,
                int32_t topk,
                c10::ScalarType input_dtype,
                c10::ScalarType output_dtype,
                bool allocate_output_on_init = true);

    AGGroupGemm(c10::intrusive_ptr<c10d::ProcessGroup> etp_group,
                int32_t nnodes,
                int32_t full_m,
                int32_t n_dim,
                int32_t k_dim,
                int32_t num_experts,
                int32_t topk,
                c10::ScalarType input_dtype,
                c10::ScalarType output_dtype,
                bool allocate_output_on_init,
                std::vector<int64_t> rank_m_sizes);

    AGGroupGemm(c10::intrusive_ptr<c10d::ProcessGroup> etp_group,
                int32_t nnodes,
                int32_t full_m,
                int32_t n_dim,
                int32_t k_dim,
                int32_t num_experts,
                int32_t topk,
                c10::ScalarType input_dtype,
                c10::ScalarType output_dtype,
                bool allocate_output_on_init,
                std::vector<int64_t> rank_m_sizes,
                int32_t capacity_full_m,
                const std::string& route = "auto");

    ~AGGroupGemm();

    torch::Tensor forward(torch::Tensor input,
                          torch::Tensor weight,
                          torch::Tensor full_topk_ids,
                          c10::optional<torch::Tensor> output);

    /**
     * Compute dW after forward() while reusing its gathered input, routing
     * metadata, and rank-ready signals. full_routed_input is
     * [full_m * topk, output_dim]; the result is
     * [num_experts, output_dim, reduction_dim].
     */
    torch::Tensor backward_weight(torch::Tensor full_routed_input);

    /** Queue the metadata-reusing weight gradient and return immediately. */
    torch::Tensor backward_weight_async(torch::Tensor full_routed_input);

    /**
     * Compute an FC1-style variable-M weight gradient without expert padding.
     *
     * routed_input: [full_m, reduction_dim]
     * grad_output: [full_m, output_dim]
     * full_topk_ids: [full_m, 1]
     * result: [num_experts, reduction_dim, output_dim]
     *
     * Unlike backward_weight(), this entry point rebuilds routing metadata for
     * the supplied IDs and writes the native [E,K,N] gradient directly.  It
     * therefore does not depend on a preceding forward() call or require an
     * [E,N,K] result followed by a materialized transpose.
     */
    torch::Tensor backward_weight_direct(torch::Tensor routed_input,
                                         torch::Tensor grad_output,
                                         torch::Tensor full_topk_ids);

    /**
     * Queue the variable-M FC1 dW computation on Hygon's private wgrad
     * stream and return immediately.  The returned tensor becomes usable
     * after wait_wgrad() (or after the caller's stream waits on the recorded
     * completion event).
     */
    torch::Tensor backward_weight_direct_async(torch::Tensor routed_input,
                                               torch::Tensor grad_output,
                                               torch::Tensor full_topk_ids);

    /**
     * Queue FC1 dW in hidden/output-dimension slices.  Each slice is launched
     * only after output_chunk_ready_flags[group] reaches one.  This is used to
     * pace wgrad behind MoEReduceRS dgrad N-chunk production without a
     * resident spin-wait kernel.
     */
    torch::Tensor backward_weight_direct_chunked_async(
        torch::Tensor routed_input,
        torch::Tensor grad_output,
        torch::Tensor full_topk_ids,
        torch::Tensor output_chunk_ready_flags,
        int32_t output_chunk_size);

    /** Wait for a previously queued backward_weight_direct_async call. */
    void wait_wgrad();

    /** Update the active ragged-M shape while retaining allocated capacity. */
    void set_rank_m_sizes(int32_t active_full_m,
                          std::vector<int64_t> rank_m_sizes);

    torch::Tensor gather_input();

    /** IPC-ready signal buffer used by the chunked input producer. */
    torch::Tensor ready_signal_buffer();

    /** Publish one producer chunk with a system-scope release store. */
    void publish_ready_signal(int32_t slot, int32_t signal_value);

    /**
     * Experimental BF16 payload-dedup metadata helper.
     *
     * Input is rank-local expert-major expanded activation
     * [local_expanded_rows, K].  Returns {compact_input, row_map, unique_count,
     * mismatch_count}, where row_map[expanded_row] =
     * compact_group * logical_topk + occurrence.  mismatch_count must be zero
     * before using the compact representation.
     */
    std::vector<torch::Tensor> payload_dedup_map(
        const torch::Tensor &expanded_input,
        int32_t logical_topk = 0);

    /** Gather compact token-major output rows into expanded row order. */
    torch::Tensor payload_remap_output(
        const torch::Tensor &compact_output,
        const torch::Tensor &row_map,
        torch::Tensor expanded_output);

    /** Scatter one compact row to topk expanded destination rows. */
    torch::Tensor payload_scatter_output(
        const torch::Tensor &compact_output,
        const torch::Tensor &destination_rows,
        torch::Tensor expanded_output);

    /** Read-only diagnostics for the active rank-tile schedule. */
    torch::Tensor debug_ntiles();
    torch::Tensor debug_rank_tile_info();
    torch::Tensor debug_rank_tile_problem_info();

    torch::Tensor n_chunk_done_flags();

private:
    class AGGroupGemmImpl;
    std::unique_ptr<AGGroupGemmImpl> pimpl_;
};

} // namespace hygon_vendor
} // namespace bytedance

#endif // FTRAIN_HYGON_AG_GROUP_GEMM_INTERFACE_H
