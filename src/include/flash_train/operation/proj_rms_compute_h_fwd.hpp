// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_PROJ_RMS_COMPUTE_H_FWD_HPP_
#define FTRAIN_OPERATION_PROJ_RMS_COMPUTE_H_FWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class ProjRmsComputeHFwdAttributes final {
  public:
    explicit ProjRmsComputeHFwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kProjRmsComputeHFwd> {
  public:
    using Id   = FTrainProjRmsComputeHFwdOpId;
    using Type = ProjRmsComputeHFwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId x, FTrainTensorId weight,
                                                           FTrainTensorId alpha_pre, FTrainTensorId alpha_post,
                                                           FTrainTensorId alpha_res, FTrainTensorId bias,
                                                           FTrainTensorId h_pre, FTrainTensorId h_post,
                                                           FTrainTensorId h_res, FTrainTensorId r, FTrainTensorId proj,
                                                           FTrainTensorId n, FTrainTensorId eps,
                                                           FTrainTensorId compute_h_eps) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(x),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(weight),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(alpha_pre),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(alpha_post),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(alpha_res),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(bias),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(n),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(eps),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(compute_h_eps)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(h_pre),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(h_post),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(h_res),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(r),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(proj)};
        return PatternOperationNode{OperationKind::kProjRmsComputeHFwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
