// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_DENSE_INDEXER_SCORE_BF16_DENSE_INDEXER_SCORE_HPP_
#define FTRAIN_PRIMITIVE_HYGON_DENSE_INDEXER_SCORE_BF16_DENSE_INDEXER_SCORE_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/dense_indexer_score/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool bf16DenseIndexerScoreLaunchFits(int heads, int width, int keys);

// Defined in bf16_dense_indexer_score.hip.
bool bf16DenseIndexerScoreDenseMmacSupported();
bool launchBf16DenseIndexerScore(FTrainStream stream, const void* q, const void* k, const void* w, float* out,
                                 float* denom, int batch, int query_length, int key_length, int query_heads,
                                 int kv_heads, int head_dim, int ratio, float sm_scale);

// One dtype chain of the DenseIndexerScore family: the ported MMAC dense
// indexer-score kernel, no workspace.
class Bf16DenseIndexerScore final : public Primitive<DenseIndexerScoreProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const DenseIndexerScoreProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const DenseIndexerScoreProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* q_    = nullptr;
    const void* k_    = nullptr;
    const void* w_    = nullptr;
    float* out_       = nullptr;
    float* denom_     = nullptr;
    int batch_        = 0;
    int query_length_ = 0;
    int key_length_   = 0;
    int query_heads_  = 0;
    int head_dim_     = 0;
    int ratio_        = 0;
    float sm_scale_   = 0.0F;
};

}  // namespace ftrain

#endif
