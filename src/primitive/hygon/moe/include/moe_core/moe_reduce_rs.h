#include "moe_core/runtime.h"
#ifndef FTRAIN_MOE_REDUCE_RS_NATIVE_INTERFACE_H
#define FTRAIN_MOE_REDUCE_RS_NATIVE_INTERFACE_H

#include <memory>
#include <string>
#include <vector>

namespace ftrain
{
namespace moe
{
namespace core
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
     * gemm_variant selects rs_nn ([E,K_local,N], default) or rs_nt
     * ([E,N,K_local]) dedicated BF16 grouped-GEMM.
     */
    MoEReduceRS(std::shared_ptr<Context> etp_group,
                int32_t nnodes,
                int32_t max_ntokens,
                int32_t n_dim,
                int32_t k_dim,
                int32_t num_experts,
                int32_t topk,
                ScalarType input_dtype,
                int32_t n_chunks = 0,
                const std::string& route = "a2a_ragged_signal",
                const std::string& gemm_variant = "rs_nn");

    ~MoEReduceRS();

    void set_rank_m_sizes(int32_t active_ntokens,
                          std::vector<int64_t> rank_m_sizes);

    Tensor run_a2a(Tensor input,
                          Tensor weight,
                          Tensor chosen_experts,
                          Tensor expert_weight,
                          std::optional<Tensor> output);

    /** Queue A2A dgrad without fencing its completion onto the caller stream. */
    Tensor run_a2a_async(Tensor input,
                                Tensor weight,
                                Tensor chosen_experts,
                                Tensor expert_weight,
                                std::optional<Tensor> output);

    /** Complete a pending run_a2a_async call and restore the caller contract. */
    void wait_a2a();

    Tensor n_chunk_done_flags();
    int32_t last_chunk_size_n() const;
    int32_t last_num_groups() const;
    bool last_n_chunk_signal() const;

private:
    class MoEReduceRSImpl;
    std::unique_ptr<MoEReduceRSImpl> pimpl_;
};

}
} // namespace moe
} // namespace ftrain

#endif // FTRAIN_MOE_REDUCE_RS_NATIVE_INTERFACE_H
