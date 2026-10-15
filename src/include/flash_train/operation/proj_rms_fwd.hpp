// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_PROJ_RMS_FWD_HPP_
#define FTRAIN_OPERATION_PROJ_RMS_FWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class ProjRmsFwdAttributes final {
  public:
    explicit ProjRmsFwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kProjRmsFwd> {
  public:
    using Id   = FTrainProjRmsFwdOpId;
    using Type = ProjRmsFwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId x, FTrainTensorId weight, FTrainTensorId proj,
                                                           FTrainTensorId r, FTrainTensorId eps) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(x),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(weight),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(eps)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(proj),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(r)};
        return PatternOperationNode{OperationKind::kProjRmsFwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
