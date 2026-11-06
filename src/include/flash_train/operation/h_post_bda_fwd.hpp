// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_H_POST_BDA_FWD_HPP_
#define FTRAIN_OPERATION_H_POST_BDA_FWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class HPostBdaFwdAttributes final {
  public:
    explicit HPostBdaFwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kHPostBdaFwd> {
  public:
    using Id   = FTrainHPostBdaFwdOpId;
    using Type = HPostBdaFwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId h_res, FTrainTensorId residual,
                                                           FTrainTensorId h_post, FTrainTensorId x, FTrainTensorId bias,
                                                           FTrainTensorId output) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(h_res),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(residual),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(h_post),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(x),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(bias)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(output)};
        return PatternOperationNode{OperationKind::kHPostBdaFwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
