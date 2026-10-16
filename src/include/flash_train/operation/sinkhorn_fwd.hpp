// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_SINKHORN_FWD_HPP_
#define FTRAIN_OPERATION_SINKHORN_FWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class SinkhornFwdAttributes final {
  public:
    explicit SinkhornFwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kSinkhornFwd> {
  public:
    using Id   = FTrainSinkhornFwdOpId;
    using Type = SinkhornFwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId logits, FTrainTensorId output,
                                                           FTrainTensorId m_init, FTrainTensorId num_iterations,
                                                           FTrainTensorId eps) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(logits),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(num_iterations),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(eps)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(output),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(m_init)};
        return PatternOperationNode{OperationKind::kSinkhornFwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
