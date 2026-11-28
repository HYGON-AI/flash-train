// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_H_AGGREGATE_FWD_HPP_
#define FTRAIN_OPERATION_H_AGGREGATE_FWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class HAggregateFwdAttributes final {
  public:
    explicit HAggregateFwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kHAggregateFwd> {
  public:
    using Id   = FTrainHAggregateFwdOpId;
    using Type = HAggregateFwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId x, FTrainTensorId h_pre,
                                                           FTrainTensorId aggregated) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(x),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(h_pre)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(aggregated)};
        return PatternOperationNode{OperationKind::kHAggregateFwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
