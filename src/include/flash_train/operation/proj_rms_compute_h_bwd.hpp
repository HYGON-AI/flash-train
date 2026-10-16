// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_PROJ_RMS_COMPUTE_H_BWD_HPP_
#define FTRAIN_OPERATION_PROJ_RMS_COMPUTE_H_BWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class ProjRmsComputeHBwdAttributes final {
  public:
    explicit ProjRmsComputeHBwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kProjRmsComputeHBwd> {
  public:
    using Id   = FTrainProjRmsComputeHBwdOpId;
    using Type = ProjRmsComputeHBwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(
        FTrainTensorId grad_h_pre, FTrainTensorId grad_h_post, FTrainTensorId grad_h_res, FTrainTensorId grad_r,
        FTrainTensorId h_pre, FTrainTensorId h_post, FTrainTensorId proj, FTrainTensorId r_ch, FTrainTensorId x,
        FTrainTensorId weight, FTrainTensorId alpha_pre, FTrainTensorId alpha_post, FTrainTensorId alpha_res,
        FTrainTensorId grad_x, FTrainTensorId grad_weight, FTrainTensorId grad_alpha, FTrainTensorId grad_bias,
        FTrainTensorId n, FTrainTensorId eps, FTrainTensorId compute_h_eps) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_h_pre),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_h_post),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_h_res),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_r),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(h_pre),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(h_post),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(proj),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(r_ch),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(x),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(weight),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(alpha_pre),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(alpha_post),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(alpha_res),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(n),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(eps),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(compute_h_eps)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_x),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_weight),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_alpha),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_bias)};
        return PatternOperationNode{OperationKind::kProjRmsComputeHBwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
