// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_SPARSE_ATTN_TARGET_HPP_
#define FTRAIN_OPERATION_SPARSE_ATTN_TARGET_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class SparseAttnTargetAttributes final {
  public:
    explicit SparseAttnTargetAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kSparseAttnTarget> {
  public:
    using Id   = FTrainSparseAttnTargetOpId;
    using Type = SparseAttnTargetAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId k, FTrainTensorId lse,
                                                           FTrainTensorId topk_indices, FTrainTensorId target,
                                                           FTrainTensorId softmax_scale,
                                                           FTrainTensorId topk_indices_global) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(q),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(k),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(lse),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_indices),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(softmax_scale),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_indices_global)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(target)};
        return PatternOperationNode{OperationKind::kSparseAttnTarget, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
