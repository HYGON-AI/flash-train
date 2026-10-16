// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_BWD_SPARSE_FP16_INDEXER_BWD_SPARSE_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_BWD_SPARSE_FP16_INDEXER_BWD_SPARSE_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/indexer_bwd_sparse/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool fp16IndexerBwdSparseLaunchFits(int heads, int width, int columns, int queries);

// Defined in fp16_indexer_bwd_sparse.hip.
bool fp16IndexerBwdSparseDenseMmacSupported();
void fp16IndexerBwdSparseZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
bool launchFp16IndexerBwdSparse(FTrainStream stream, const void* index_q, const void* weights, const void* index_k,
                                float* attn_score, const float* index_score, const int* topk_indices, void* dq,
                                void* dw, void* dk, float* dk_float, int* done_count, int batch, int query_length,
                                int key_length, int heads, int head_dim, int selection_width, float sm_scale,
                                float loss_coeff, float grad_loss, bool topk_indices_global);

// One dtype chain of the IndexerBwdSparse family. dk accumulates in FP32 and
// is cast by the last arriving block, so the Primitive owns the accumulator
// and the completion counter and clears both on the stream.
class Fp16IndexerBwdSparse final : public Primitive<IndexerBwdSparseProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const IndexerBwdSparseProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const IndexerBwdSparseProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* index_q_             = nullptr;
    const void* weights_             = nullptr;
    const void* index_k_             = nullptr;
    const void* index_score_         = nullptr;
    const int* topk_indices_         = nullptr;
    float* score_gradient_           = nullptr;
    void* dq_                        = nullptr;
    void* dw_                        = nullptr;
    void* dk_                        = nullptr;
    int batch_                       = 0;
    int query_length_                = 0;
    int key_length_                  = 0;
    int heads_                       = 0;
    int head_dim_                    = 0;
    int selection_width_             = 0;
    float sm_scale_                  = 0.0F;
    float loss_coeff_                = 0.0F;
    float grad_loss_                 = 0.0F;
    bool topk_indices_global_        = false;
    std::uint64_t workspace_bytes_   = 0;
    std::uint64_t dk_float_offset_   = 0;
    std::uint64_t done_count_offset_ = 0;
};

}  // namespace ftrain

#endif
