// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_SPARSE_ATTN_BWD_HPP_
#define FTRAIN_OPERATION_SPARSE_ATTN_BWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class SparseAttnBwdAttributes final {
  public:
    explicit SparseAttnBwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kSparseAttnBwd> {
  public:
    using Id   = FTrainSparseAttnBwdOpId;
    using Type = SparseAttnBwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId kv, FTrainTensorId out,
                                                           FTrainTensorId dout, FTrainTensorId lse,
                                                           FTrainTensorId attn_sink, FTrainTensorId topk_indices,
                                                           FTrainTensorId topk_length, FTrainTensorId dq,
                                                           FTrainTensorId dkv, FTrainTensorId dsink,
                                                           FTrainTensorId softmax_scale) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(q),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(kv),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(out),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(dout),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(lse),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(attn_sink),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_indices),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_length),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(softmax_scale)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(dq),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(dkv),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(dsink)};
        return PatternOperationNode{OperationKind::kSparseAttnBwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
