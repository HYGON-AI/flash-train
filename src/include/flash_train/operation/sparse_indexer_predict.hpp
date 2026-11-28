// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_SPARSE_INDEXER_PREDICT_HPP_
#define FTRAIN_OPERATION_SPARSE_INDEXER_PREDICT_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class SparseIndexerPredictAttributes final {
  public:
    explicit SparseIndexerPredictAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kSparseIndexerPredict> {
  public:
    using Id   = FTrainSparseIndexerPredictOpId;
    using Type = SparseIndexerPredictAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId q, FTrainTensorId k, FTrainTensorId w,
                                                           FTrainTensorId topk_indices, FTrainTensorId predict,
                                                           FTrainTensorId sm_scale,
                                                           FTrainTensorId topk_indices_global) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(q),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(k),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(w),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_indices),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(sm_scale),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_indices_global)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(predict)};
        return PatternOperationNode{OperationKind::kSparseIndexerPredict, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
