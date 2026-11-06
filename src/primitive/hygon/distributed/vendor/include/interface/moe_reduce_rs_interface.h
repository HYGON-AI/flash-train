#ifndef FTRAIN_HYGON_MOE_REDUCE_RS_INTERFACE_H
#define FTRAIN_HYGON_MOE_REDUCE_RS_INTERFACE_H

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

class MoEReduceRS
{
public:
    /**
     * @brief MoE grouped-GEMM output reduce-scatter for single-node expert
     * tensor parallelism (ETP).
     *
     * etp_group partitions the expert GEMM K/N tensor dimension.  Expert IDs
     * remain global within this group; expert-parallel dispatch is not part of
     * this operator's ABI.
     *
     * gemm_variant is route-owned and defaults to the validated NK-direct
     * grouped-GEMM contract.
     */
    MoEReduceRS(c10::intrusive_ptr<c10d::ProcessGroup> etp_group,
                int32_t nnodes,
                int32_t max_ntokens,
                int32_t n_dim,
                int32_t k_dim,
                int32_t num_experts,
                int32_t topk,
                c10::ScalarType input_dtype,
                int32_t n_chunks = 0,
                const std::string& route = "a2a_ragged_signal",
                const std::string& gemm_variant = "nk_direct");

    ~MoEReduceRS();

    void set_rank_m_sizes(int32_t active_ntokens,
                          std::vector<int64_t> rank_m_sizes);

    torch::Tensor run_a2a(torch::Tensor input,
                          torch::Tensor weight,
                          torch::Tensor chosen_experts,
                          torch::Tensor expert_weight,
                          c10::optional<torch::Tensor> output);

    /** Queue A2A dgrad without fencing its completion onto the caller stream. */
    torch::Tensor run_a2a_async(torch::Tensor input,
                                torch::Tensor weight,
                                torch::Tensor chosen_experts,
                                torch::Tensor expert_weight,
                                c10::optional<torch::Tensor> output);

    /** Complete a pending run_a2a_async call and restore the caller contract. */
    void wait_a2a();

    torch::Tensor n_chunk_done_flags();
    int32_t last_chunk_size_n() const;
    int32_t last_num_groups() const;
    bool last_n_chunk_signal() const;

private:
    class MoEReduceRSImpl;
    std::unique_ptr<MoEReduceRSImpl> pimpl_;
};

} // namespace hygon_vendor
} // namespace bytedance

#endif // FTRAIN_HYGON_MOE_REDUCE_RS_INTERFACE_H
