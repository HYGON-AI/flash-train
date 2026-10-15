// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_DENSE_ATTN_SCORE_THD_MIXED_FP32_DENSE_ATTN_SCORE_THD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_DENSE_ATTN_SCORE_THD_MIXED_FP32_DENSE_ATTN_SCORE_THD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/dense_attn_score_thd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool mixedFp32DenseAttnScoreThdLaunchFits(int heads, int width, int keys);

void castMixedFp32DenseAttnScoreThd(const void* source, void* destination, FTrainNumericType source_type,
                                    FTrainNumericType destination_type, std::uint64_t elements, FTrainStream stream);

// Defined in mixed_fp32_dense_attn_score_thd.hip.
bool mixedFp32DenseAttnScoreThdDenseMmacSupported();
bool launchMixedFp32DenseAttnScoreThd(FTrainStream stream, const void* q, const void* k, const float* lse, float* out,
                                      float* denom, const int* cu_seqlens_q, const int* cu_seqlens_kv,
                                      const int* q_causal_offsets, int total_q, int total_k, int max_seqlen_kv,
                                      int query_heads, int kv_heads, int head_dim, int num_sequences, int ratio,
                                      float softmax_scale);

// Mixed input storage follows upstream FP32 promotion in caller-owned workspace.
// Results retain the original port dtypes; selection checks remain in isApplicable.
class MixedFp32DenseAttnScoreThd final : public Primitive<DenseAttnScoreThdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const DenseAttnScoreThdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const DenseAttnScoreThdProblem& problem) override;

  private:
    FTrainNumericType q_type_          = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t q_elements_          = 0;
    std::uint64_t q_conversion_offset_ = 0;
    FTrainNumericType k_type_          = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t k_elements_          = 0;
    std::uint64_t k_conversion_offset_ = 0;
    std::uint64_t workspace_bytes_     = 0;

    void executeImpl(const Resources& resources) override;

    const void* q_             = nullptr;
    const void* k_             = nullptr;
    const float* lse_          = nullptr;
    const int* cu_seqlens_q_   = nullptr;
    const int* cu_seqlens_kv_  = nullptr;
    const int* causal_offsets_ = nullptr;
    float* out_                = nullptr;
    float* denom_              = nullptr;
    int total_q_               = 0;
    int total_k_               = 0;
    int max_seqlen_kv_         = 0;
    int query_heads_           = 0;
    int head_dim_              = 0;
    int num_sequences_         = 0;
    int ratio_                 = 0;
    float softmax_scale_       = 0.0F;
};

}  // namespace ftrain

#endif
