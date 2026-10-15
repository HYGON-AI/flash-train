// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_FWD_INDEXER_FP32_SPARSE_ATTN_FWD_INDEXER_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_FWD_INDEXER_FP32_SPARSE_ATTN_FWD_INDEXER_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/sparse_attn_fwd_indexer/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool fp32SparseAttnFwdIndexerLaunchFits(int width, int columns);

// Defined in fp32_sparse_attn_fwd_indexer.hip.
bool fp32SparseAttnFwdIndexerSparseAttnFusedSupported();
bool launchFp32SparseAttnFwdIndexer(FTrainStream stream, const void* q, const void* kv, const float* attn_sink,
                                    const int* topk_indices, void* out, float* max_logits, float* lse,
                                    float* lse_indexer, const int* topk_length, int indexer_topk, int tokens, int heads,
                                    int head_dim, int value_dim, int kv_tokens, int selection_width,
                                    float softmax_scale);

// One dtype chain of the SparseAttnFwdIndexer family: the ported fused MMAC
// sparse attention forward with the indexer statistics, no workspace.
class Fp32SparseAttnFwdIndexer final : public Primitive<SparseAttnFwdIndexerProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseAttnFwdIndexerProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseAttnFwdIndexerProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* q_           = nullptr;
    const void* kv_          = nullptr;
    const float* attn_sink_  = nullptr;
    const int* topk_indices_ = nullptr;
    const int* topk_length_  = nullptr;
    void* out_               = nullptr;
    float* max_logits_       = nullptr;
    float* lse_              = nullptr;
    float* lse_indexer_      = nullptr;
    int indexer_topk_        = 0;
    int tokens_              = 0;
    int heads_               = 0;
    int head_dim_            = 0;
    int value_dim_           = 0;
    int kv_tokens_           = 0;
    int selection_width_     = 0;
    float softmax_scale_     = 0.0F;
};

}  // namespace ftrain

#endif
