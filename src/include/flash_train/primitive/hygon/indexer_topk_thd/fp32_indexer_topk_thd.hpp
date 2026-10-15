// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_TOPK_THD_FP32_INDEXER_TOPK_THD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_TOPK_THD_FP32_INDEXER_TOPK_THD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/indexer_topk_thd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp32_indexer_topk_thd.hip.
bool launchFp32IndexerTopKThd(FTrainStream stream, const float* input_values, const int* cu_seqlens_q,
                              const int* cu_seqlens_kv, const int* q_causal_offsets, int* out_indices,
                              float* out_values, int num_rows, int num_columns, int top_k, int num_sequences, int ratio,
                              bool order_values);

// The IndexerTopKThd Primitive: the ported packed-batch top-K kernel, no
// workspace and no dtype chain.
class Fp32IndexerTopKThd final : public Primitive<IndexerTopKThdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const IndexerTopKThdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const IndexerTopKThdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const float* input_values_ = nullptr;
    const int* cu_seqlens_q_   = nullptr;
    const int* cu_seqlens_kv_  = nullptr;
    const int* causal_offsets_ = nullptr;
    int* out_indices_          = nullptr;
    float* out_values_         = nullptr;
    int num_rows_              = 0;
    int num_columns_           = 0;
    int top_k_                 = 0;
    int num_sequences_         = 0;
    int ratio_                 = 0;
    bool order_values_         = false;
};

}  // namespace ftrain

#endif
