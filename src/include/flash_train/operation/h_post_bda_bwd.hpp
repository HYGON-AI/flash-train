// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_H_POST_BDA_BWD_HPP_
#define FTRAIN_OPERATION_H_POST_BDA_BWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class HPostBdaBwdAttributes final {
  public:
    explicit HPostBdaBwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kHPostBdaBwd> {
  public:
    using Id   = FTrainHPostBdaBwdOpId;
    using Type = HPostBdaBwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId grad_output, FTrainTensorId residual,
                                                           FTrainTensorId x, FTrainTensorId bias, FTrainTensorId h_res,
                                                           FTrainTensorId h_post, FTrainTensorId grad_residual,
                                                           FTrainTensorId grad_x, FTrainTensorId grad_h_res,
                                                           FTrainTensorId grad_h_post, FTrainTensorId grad_bias) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_output),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(residual),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(x),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(bias),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(h_res),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(h_post)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_residual),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_x),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_h_res),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_h_post),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_bias)};
        return PatternOperationNode{OperationKind::kHPostBdaBwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
