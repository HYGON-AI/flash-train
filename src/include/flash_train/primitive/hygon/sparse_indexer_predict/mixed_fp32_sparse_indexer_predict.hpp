// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_INDEXER_PREDICT_MIXED_FP32_SPARSE_INDEXER_PREDICT_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_INDEXER_PREDICT_MIXED_FP32_SPARSE_INDEXER_PREDICT_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/sparse_indexer_predict/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

void castMixedFp32SparseIndexerPredict(const void* source, void* destination, FTrainNumericType source_type,
                                       FTrainNumericType destination_type, std::uint64_t elements, FTrainStream stream);

// Defined in mixed_fp32_sparse_indexer_predict.hip.
bool mixedFp32SparseIndexerPredictSparseMmacSupported();
bool mixedFp32SparseIndexerPredictSparseMmacShmFits(int query_heads, int head_dim, int selection_width);
void mixedFp32SparseIndexerPredictZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
bool launchMixedFp32SparseIndexerPredict(FTrainStream stream, const void* q, const void* k, const void* w,
                                         const int* topk_indices, float* predict, int* row_count, int batch,
                                         int query_length, int key_length, int query_heads, int head_dim,
                                         int selection_width, float sm_scale, bool topk_indices_global);

// Mixed input storage follows upstream FP32 promotion in caller-owned workspace.
// Results retain the original port dtypes; selection checks remain in isApplicable.
class MixedFp32SparseIndexerPredict final : public Primitive<SparseIndexerPredictProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseIndexerPredictProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseIndexerPredictProblem& problem) override;

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
    void executeImpl(const Resources& resources) override;

    const void* q_                  = nullptr;
    const void* k_                  = nullptr;
    const void* w_                  = nullptr;
    const int* topk_indices_        = nullptr;
    float* predict_                 = nullptr;
    int batch_                      = 0;
    int query_length_               = 0;
    int key_length_                 = 0;
    int query_heads_                = 0;
    int head_dim_                   = 0;
    int selection_width_            = 0;
    float sm_scale_                 = 0.0F;
    bool topk_indices_global_       = false;
    std::uint64_t workspace_bytes_  = 0;
    std::uint64_t row_count_offset_ = 0;
};

}  // namespace ftrain

#endif
