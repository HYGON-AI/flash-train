// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_INDEXER_BWD_DENSE_HPP_
#define FTRAIN_OPERATION_INDEXER_BWD_DENSE_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class IndexerBwdDenseAttributes final {
  public:
    explicit IndexerBwdDenseAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kIndexerBwdDense> {
  public:
    using Id   = FTrainIndexerBwdDenseOpId;
    using Type = IndexerBwdDenseAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId index_q, FTrainTensorId weights,
                                                           FTrainTensorId index_k, FTrainTensorId attn_score,
                                                           FTrainTensorId attn_l1norm, FTrainTensorId index_score,
                                                           FTrainTensorId index_lse, FTrainTensorId grad_index_score,
                                                           FTrainTensorId dq, FTrainTensorId dw, FTrainTensorId dk,
                                                           FTrainTensorId sm_scale, FTrainTensorId loss_coeff,
                                                           FTrainTensorId grad_loss, FTrainTensorId ratio) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(index_q),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(weights),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(index_k),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(attn_score),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(attn_l1norm),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(index_score),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(index_lse),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(sm_scale),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(loss_coeff),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_loss),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(ratio)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_index_score),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(dq),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(dw),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(dk)};
        return PatternOperationNode{OperationKind::kIndexerBwdDense, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
