// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_INDEXER_TOPK_HPP_
#define FTRAIN_OPERATION_INDEXER_TOPK_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class IndexerTopKAttributes final {
  public:
    IndexerTopKAttributes() noexcept = default;
};

template<>
class OperationTraits<OperationKind::kIndexerTopK> {
  public:
    using Id   = FTrainIndexerTopKOpId;
    using Type = IndexerTopKAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId input_values, FTrainTensorId seq_lens,
                                                           FTrainTensorId out_indices, FTrainTensorId out_values,
                                                           FTrainTensorId top_k, FTrainTensorId next_n,
                                                           FTrainTensorId order_values) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(input_values),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(seq_lens),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(top_k),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(next_n),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(order_values)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(out_indices),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(out_values)};
        return PatternOperationNode{OperationKind::kIndexerTopK, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
