// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_PROJ_RMS_BWD_HPP_
#define FTRAIN_OPERATION_PROJ_RMS_BWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class ProjRmsBwdAttributes final {
  public:
    explicit ProjRmsBwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kProjRmsBwd> {
  public:
    using Id   = FTrainProjRmsBwdOpId;
    using Type = ProjRmsBwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId grad_proj, FTrainTensorId grad_r,
                                                           FTrainTensorId x, FTrainTensorId weight, FTrainTensorId r,
                                                           FTrainTensorId grad_x, FTrainTensorId grad_weight,
                                                           FTrainTensorId eps) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_proj),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_r),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(x),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(weight),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(r),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(eps)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_x),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_weight)};
        return PatternOperationNode{OperationKind::kProjRmsBwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
