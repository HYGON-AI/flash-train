// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_TOPK_FP32_INDEXER_TOPK_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_TOPK_FP32_INDEXER_TOPK_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/indexer_topk/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp32_indexer_topk.hip.
bool launchFp32IndexerTopKK(FTrainStream stream, const float* input_values, const int* seq_lens, int* out_indices,
                            float* out_values, int num_rows, int num_columns, int top_k, int batch_size, int next_n,
                            bool order_values);

// The IndexerTopK Primitive: the ported resource-bounded parallel top-K
// kernel, no workspace. The family has no dtype chain, so this single record
// is the whole catalog.
class Fp32IndexerTopK final : public Primitive<IndexerTopKProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const IndexerTopKProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const IndexerTopKProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const float* input_values_ = nullptr;
    const int* seq_lens_       = nullptr;
    int* out_indices_          = nullptr;
    float* out_values_         = nullptr;
    int num_rows_              = 0;
    int num_columns_           = 0;
    int top_k_                 = 0;
    int num_sequences_         = 0;
    int next_n_                = 0;
    bool order_values_         = false;
};

}  // namespace ftrain

#endif
