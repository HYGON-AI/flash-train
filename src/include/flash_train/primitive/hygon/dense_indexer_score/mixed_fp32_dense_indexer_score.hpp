// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_DENSE_INDEXER_SCORE_MIXED_FP32_DENSE_INDEXER_SCORE_HPP_
#define FTRAIN_PRIMITIVE_HYGON_DENSE_INDEXER_SCORE_MIXED_FP32_DENSE_INDEXER_SCORE_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/dense_indexer_score/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool mixedFp32DenseIndexerScoreLaunchFits(int heads, int width, int keys);

void castMixedFp32DenseIndexerScore(const void* source, void* destination, FTrainNumericType source_type,
                                    FTrainNumericType destination_type, std::uint64_t elements, FTrainStream stream);

// Defined in mixed_fp32_dense_indexer_score.hip.
bool mixedFp32DenseIndexerScoreDenseMmacSupported();
bool launchMixedFp32DenseIndexerScore(FTrainStream stream, const void* q, const void* k, const void* w, float* out,
                                      float* denom, int batch, int query_length, int key_length, int query_heads,
                                      int kv_heads, int head_dim, int ratio, float sm_scale);

// Mixed input storage follows upstream FP32 promotion in caller-owned workspace.
// Results retain the original port dtypes; selection checks remain in isApplicable.
class MixedFp32DenseIndexerScore final : public Primitive<DenseIndexerScoreProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const DenseIndexerScoreProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const DenseIndexerScoreProblem& problem) override;

  private:
    FTrainNumericType q_type_          = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t q_elements_          = 0;
    std::uint64_t q_conversion_offset_ = 0;
    FTrainNumericType k_type_          = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t k_elements_          = 0;
    std::uint64_t k_conversion_offset_ = 0;
    FTrainNumericType w_type_          = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t w_elements_          = 0;
    std::uint64_t w_conversion_offset_ = 0;
    std::uint64_t workspace_bytes_     = 0;

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
