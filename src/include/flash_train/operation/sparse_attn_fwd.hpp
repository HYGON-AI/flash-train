// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_SPARSE_ATTN_FWD_HPP_
#define FTRAIN_OPERATION_SPARSE_ATTN_FWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class SparseAttnFwdAttributes final {
  public:
    explicit SparseAttnFwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kSparseAttnFwd> {
  public:
    using Id   = FTrainSparseAttnFwdOpId;
    using Type = SparseAttnFwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId kv,
                                                           FTrainTensorId attn_sink, FTrainTensorId topk_indices,
                                                           FTrainTensorId topk_length, FTrainTensorId out,
                                                           FTrainTensorId lse, FTrainTensorId softmax_scale) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(q),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(kv),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(attn_sink),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_indices),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_length),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(softmax_scale)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(out),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(lse)};
        return PatternOperationNode{OperationKind::kSparseAttnFwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
