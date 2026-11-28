// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_INDEXER_FWD_HPP_
#define FTRAIN_OPERATION_INDEXER_FWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class IndexerFwdAttributes final {
  public:
    explicit IndexerFwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kIndexerFwd> {
  public:
    using Id   = FTrainIndexerFwdOpId;
    using Type = IndexerFwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId k, FTrainTensorId w,
                                                           FTrainTensorId scores, FTrainTensorId ratio,
                                                           FTrainTensorId sm_scale) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(q),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(k),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(w),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(ratio),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(sm_scale)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(scores)};
        return PatternOperationNode{OperationKind::kIndexerFwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
