#include "moe_core/runtime.h"
#ifndef FTRAIN_AG_GROUP_GEMM_NATIVE_INTERFACE_H
#define FTRAIN_AG_GROUP_GEMM_NATIVE_INTERFACE_H

#include <memory>
#include <string>
#include <vector>

namespace ftrain
{
namespace moe
{
namespace core
{

enum class NativeKnRaggedMode
{
    Disabled = 0,
    FixedRing = 1,
};

class AGGroupGemm
{
public:
    /**
     * @brief AllGather + MoE grouped GEMM for single-node expert tensor parallelism (ETP).
     *
     * Native ABI shape: input [local_m, reduction_dim],
     * weight [num_experts, reduction_dim, output_dim].
     * ag_nt_* routes instead consume raw NK weights
     * [num_experts, output_dim, reduction_dim]. ag_nt_a2a_chunk_32k and
     * ag_nt_ring_chunk_32k
     * require gfx936, ETP8, BF16, reduction_dim=32768, output_dim%256=0,
     * and expanded expert-major input with topk=1. ag_nt_a2a_chunk
     * requires equal rank_m_sizes; chunk_32k and ring routes accept positive
     * unequal sizes and dynamic updates through set_rank_m_sizes().
     * full_topk_ids: [full_m, topk]
     * output: [full_m * topk, output_dim]
     * rank_m_sizes: optional per-rank local_m values. Empty means uniform
     * full_m / world_size partitioning.
     */
    AGGroupGemm(std::shared_ptr<Context> etp_group,
                int32_t nnodes,
                int32_t full_m,
                int32_t n_dim,
                int32_t k_dim,
                int32_t num_experts,
                int32_t topk,
                ScalarType input_dtype,
                ScalarType output_dtype,
                bool allocate_output_on_init = true);

    AGGroupGemm(std::shared_ptr<Context> etp_group,
                int32_t nnodes,
                int32_t full_m,
                int32_t n_dim,
                int32_t k_dim,
                int32_t num_experts,
                int32_t topk,
                ScalarType input_dtype,
                ScalarType output_dtype,
                bool allocate_output_on_init,
                std::vector<int64_t> rank_m_sizes);

    AGGroupGemm(std::shared_ptr<Context> etp_group,
                int32_t nnodes,
                int32_t full_m,
                int32_t n_dim,
                int32_t k_dim,
                int32_t num_experts,
                int32_t topk,
                ScalarType input_dtype,
                ScalarType output_dtype,
                bool allocate_output_on_init,
                std::vector<int64_t> rank_m_sizes,
                int32_t capacity_full_m,
                const std::string& route = "ag_nn_ring_chunk");

    ~AGGroupGemm();

    Tensor forward(Tensor input,
                          Tensor weight,
                          Tensor full_topk_ids,
                          std::optional<Tensor> output);

    /**
     * Compute dW after forward() while reusing its gathered input, routing
     * metadata, and rank-ready signals. full_routed_input is
     * [full_m * topk, output_dim]; the result is
     * [num_experts, output_dim, reduction_dim].
     */
    Tensor backward_weight(Tensor full_routed_input);

    /** Queue the metadata-reusing weight gradient and return immediately. */
    Tensor backward_weight_async(Tensor full_routed_input);

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
    Tensor backward_weight_direct(Tensor routed_input,
                                         Tensor grad_output,
                                         Tensor full_topk_ids);

    /**
     * Queue the variable-M FC1 dW computation on FTrain's private wgrad
     * stream and return immediately.  The returned tensor becomes usable
     * after wait_wgrad() (or after the caller's stream waits on the recorded
     * completion event).
     */
    Tensor backward_weight_direct_async(Tensor routed_input,
                                               Tensor grad_output,
                                               Tensor full_topk_ids);

    /**
     * Queue FC1 dW in hidden/output-dimension slices.  Each slice is launched
     * only after output_chunk_ready_flags[group] reaches one.  This is used to
     * pace wgrad behind MoEReduceRS dgrad N-chunk production without a
     * resident spin-wait kernel.
     */
    Tensor backward_weight_direct_chunked_async(
        Tensor routed_input,
        Tensor grad_output,
        Tensor full_topk_ids,
        Tensor output_chunk_ready_flags,
        int32_t output_chunk_size);

    /** Wait for a previously queued backward_weight_direct_async call. */
    void wait_wgrad();

    /** Update the active ragged-M shape while retaining allocated capacity. */
    void set_rank_m_sizes(int32_t active_full_m,
                          std::vector<int64_t> rank_m_sizes);

    Tensor gather_input();

    /** IPC-ready signal buffer used by the chunked input producer. */
    Tensor ready_signal_buffer();

    /** Publish one producer chunk with a system-scope release store. */
    void publish_ready_signal(int32_t slot, int32_t signal_value);

    /** Read-only diagnostics for the active rank-tile schedule. */
    Tensor debug_ntiles();
    Tensor debug_rank_tile_info();
    Tensor debug_rank_tile_problem_info();

    Tensor n_chunk_done_flags();

private:
    class AGGroupGemmImpl;
    std::unique_ptr<AGGroupGemmImpl> pimpl_;
};

}
} // namespace moe
} // namespace ftrain

#endif // FTRAIN_AG_GROUP_GEMM_NATIVE_INTERFACE_H
