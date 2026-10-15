// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_SPARSE_ATTN_FWD_INDEXER_HPP_
#define FTRAIN_OPERATION_SPARSE_ATTN_FWD_INDEXER_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class SparseAttnFwdIndexerAttributes final {
  public:
    explicit SparseAttnFwdIndexerAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kSparseAttnFwdIndexer> {
  public:
    using Id   = FTrainSparseAttnFwdIndexerOpId;
    using Type = SparseAttnFwdIndexerAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId kv,
                                                           FTrainTensorId attn_sink, FTrainTensorId topk_indices,
                                                           FTrainTensorId topk_length, FTrainTensorId out,
                                                           FTrainTensorId max_logits, FTrainTensorId lse,
                                                           FTrainTensorId lse_indexer, FTrainTensorId softmax_scale,
                                                           FTrainTensorId indexer_topk) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(q),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(kv),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(attn_sink),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_indices),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_length),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(softmax_scale),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(indexer_topk)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(out),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(max_logits),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(lse),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(lse_indexer)};
        return PatternOperationNode{OperationKind::kSparseAttnFwdIndexer, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
