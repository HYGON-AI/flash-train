// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_CSA_COMPRESSOR_BWD_HPP_
#define FTRAIN_OPERATION_CSA_COMPRESSOR_BWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class CSACompressorBwdAttributes final {
  public:
    explicit CSACompressorBwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kCSACompressorBwd> {
  public:
    using Id   = FTrainCSACompressorBwdOpId;
    using Type = CSACompressorBwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId kv, FTrainTensorId score, FTrainTensorId ape,
                                                           FTrainTensorId cu_seqlens, FTrainTensorId cu_seqlens_comp,
                                                           FTrainTensorId grad_compressed, FTrainTensorId grad_kv,
                                                           FTrainTensorId grad_score, FTrainTensorId grad_ape,
                                                           FTrainTensorId ratio, FTrainTensorId head_dim,
                                                           FTrainTensorId coff) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(kv),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(score),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(ape),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(cu_seqlens),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(cu_seqlens_comp),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(ratio),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(head_dim),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(coff)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_compressed),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_kv),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_score),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(grad_ape)};
        return PatternOperationNode{OperationKind::kCSACompressorBwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
