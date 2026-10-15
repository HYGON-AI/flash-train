// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_DENSE_INDEXER_SCORE_THD_HPP_
#define FTRAIN_OPERATION_DENSE_INDEXER_SCORE_THD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class DenseIndexerScoreThdAttributes final {
  public:
    explicit DenseIndexerScoreThdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kDenseIndexerScoreThd> {
  public:
    using Id   = FTrainDenseIndexerScoreThdOpId;
    using Type = DenseIndexerScoreThdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId k, FTrainTensorId w,
                                                           FTrainTensorId cu_seqlens_q, FTrainTensorId cu_seqlens_kv,
                                                           FTrainTensorId q_causal_offsets, FTrainTensorId out,
                                                           FTrainTensorId denom, FTrainTensorId ratio,
                                                           FTrainTensorId sm_scale, FTrainTensorId max_seqlen_kv) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(q),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(k),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(w),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(cu_seqlens_q),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(cu_seqlens_kv),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(q_causal_offsets),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(ratio),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(sm_scale),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(max_seqlen_kv)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(out),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(denom)};
        return PatternOperationNode{OperationKind::kDenseIndexerScoreThd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
