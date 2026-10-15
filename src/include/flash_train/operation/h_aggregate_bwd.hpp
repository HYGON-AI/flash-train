// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_H_AGGREGATE_BWD_HPP_
#define FTRAIN_OPERATION_H_AGGREGATE_BWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class HAggregateBwdAttributes final {
  public:
    explicit HAggregateBwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kHAggregateBwd> {
  public:
    using Id   = FTrainHAggregateBwdOpId;
    using Type = HAggregateBwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId grad_aggregated, FTrainTensorId x,
                                                           FTrainTensorId h_pre, FTrainTensorId grad_x,
                                                           FTrainTensorId grad_h_pre) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_aggregated),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(x),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(h_pre)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_x),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_h_pre)};
        return PatternOperationNode{OperationKind::kHAggregateBwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
