// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_DENSE_ATTN_SCORE_FP16_DENSE_ATTN_SCORE_HPP_
#define FTRAIN_PRIMITIVE_HYGON_DENSE_ATTN_SCORE_FP16_DENSE_ATTN_SCORE_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/dense_attn_score/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool fp16DenseAttnScoreLaunchFits(int heads, int width, int keys);

// Defined in fp16_dense_attn_score.hip.
bool fp16DenseAttnScoreDenseMmacSupported();
bool launchFp16DenseAttnScore(FTrainStream stream, const void* q, const void* k, const float* lse, float* out,
                              float* denom, int batch, int query_length, int key_length, int query_heads, int kv_heads,
                              int head_dim, int ratio, float softmax_scale);

// One dtype chain of the DenseAttnScore family: the ported MMAC dense
// attention-score kernel, no workspace.
class Fp16DenseAttnScore final : public Primitive<DenseAttnScoreProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const DenseAttnScoreProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const DenseAttnScoreProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* q_       = nullptr;
    const void* k_       = nullptr;
    const float* lse_    = nullptr;
    float* out_          = nullptr;
    float* denom_        = nullptr;
    int batch_           = 0;
    int query_length_    = 0;
    int key_length_      = 0;
    int query_heads_     = 0;
    int head_dim_        = 0;
    int ratio_           = 0;
    float softmax_scale_ = 0.0F;
};

}  // namespace ftrain

#endif
