// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_CSA_COMPRESSOR_FWD_HPP_
#define FTRAIN_OPERATION_CSA_COMPRESSOR_FWD_HPP_

#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class CSACompressorFwdAttributes final {
  public:
    explicit CSACompressorFwdAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}
    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kCSACompressorFwd> {
  public:
    using Id   = FTrainCSACompressorFwdOpId;
    using Type = CSACompressorFwdAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId kv, FTrainTensorId score, FTrainTensorId ape,
                                                           FTrainTensorId cu_seqlens, FTrainTensorId cu_seqlens_comp,
                                                           FTrainTensorId compressed, FTrainTensorId ratio,
                                                           FTrainTensorId head_dim, FTrainTensorId coff,
                                                           FTrainTensorId total_comp) {
        std::vector<PatternOperandId> inputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(kv),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(score),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(ape),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(cu_seqlens),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(cu_seqlens_comp),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(ratio),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(head_dim),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(coff),
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(total_comp)};
        std::vector<PatternOperandId> outputs = {
            OperandTraits<OperandKind::kTensor>::createPatternOperandId(compressed)};
        return PatternOperationNode{OperationKind::kCSACompressorFwd, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
