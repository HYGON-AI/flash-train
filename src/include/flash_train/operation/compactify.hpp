// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_COMPACTIFY_HPP_
#define FTRAIN_OPERATION_COMPACTIFY_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class CompactifyAttributes final {
  public:
    CompactifyAttributes() noexcept = default;
};

template<>
class OperationTraits<OperationKind::kCompactify> {
  public:
    using Id   = FTrainCompactifyOpId;
    using Type = CompactifyAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId indices, FTrainTensorId output,
                                                           FTrainTensorId topk_length) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(indices)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(output),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_length)};
        return PatternOperationNode{OperationKind::kCompactify, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
