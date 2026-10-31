// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_DENSE_ATTN_SCORE_HPP_
#define FTRAIN_OPERATION_DENSE_ATTN_SCORE_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class DenseAttnScoreAttributes final {
  public:
    explicit DenseAttnScoreAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kDenseAttnScore> {
  public:
    using Id   = FTrainDenseAttnScoreOpId;
    using Type = DenseAttnScoreAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId k, FTrainTensorId lse,
                                                           FTrainTensorId out, FTrainTensorId denom,
                                                           FTrainTensorId ratio, FTrainTensorId softmax_scale) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(q),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(k),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(lse),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(ratio),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(softmax_scale)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(out),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(denom)};
        return PatternOperationNode{OperationKind::kDenseAttnScore, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
