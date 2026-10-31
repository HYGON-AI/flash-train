// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_INDEXER_BWD_SPARSE_HPP_
#define FTRAIN_OPERATION_INDEXER_BWD_SPARSE_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class IndexerBwdSparseAttributes final {
  public:
    explicit IndexerBwdSparseAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kIndexerBwdSparse> {
  public:
    using Id   = FTrainIndexerBwdSparseOpId;
    using Type = IndexerBwdSparseAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(
        FTrainTensorId index_q, FTrainTensorId weights, FTrainTensorId index_k, FTrainTensorId attn_score,
        FTrainTensorId index_score, FTrainTensorId topk_indices, FTrainTensorId grad_attn_score, FTrainTensorId dq,
        FTrainTensorId dw, FTrainTensorId dk, FTrainTensorId sm_scale, FTrainTensorId loss_coeff,
        FTrainTensorId grad_loss, FTrainTensorId topk_indices_global) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(index_q),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(weights),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(index_k),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(attn_score),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(index_score),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_indices),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(sm_scale),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(loss_coeff),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_loss),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_indices_global)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_attn_score),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(dq),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(dw),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(dk)};
        return PatternOperationNode{OperationKind::kIndexerBwdSparse, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
