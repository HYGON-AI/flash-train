// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_INDEXER_TOPK_THD_HPP_
#define FTRAIN_OPERATION_INDEXER_TOPK_THD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class IndexerTopKThdAttributes final {
  public:
    IndexerTopKThdAttributes() noexcept = default;
};

template<>
class OperationTraits<OperationKind::kIndexerTopKThd> {
  public:
    using Id   = FTrainIndexerTopKThdOpId;
    using Type = IndexerTopKThdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId input_values, FTrainTensorId cu_seqlens_q,
                                                           FTrainTensorId cu_seqlens_kv,
                                                           FTrainTensorId q_causal_offsets, FTrainTensorId out_indices,
                                                           FTrainTensorId out_values, FTrainTensorId ratio,
                                                           FTrainTensorId top_k, FTrainTensorId order_values) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(input_values),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(cu_seqlens_q),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(cu_seqlens_kv),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(q_causal_offsets),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(ratio),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(top_k),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(order_values)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(out_indices),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(out_values)};
        return PatternOperationNode{OperationKind::kIndexerTopKThd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
