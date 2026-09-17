// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_OPERATION_GEMM_HPP_
#define FTRAIN_OPERATION_GEMM_HPP_

#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

// Per-operation parameters. The attribute types below store
// already-validated values and perform no validation.
class GemmAttributes final {
  public:
    explicit GemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

class GroupedABCDGemmAttributes final {
  public:
    explicit GroupedABCDGemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

class GroupedBCDGemmAttributes final {
  public:
    explicit GroupedBCDGemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

class GroupedABGemmAttributes final {
  public:
    explicit GroupedABGemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

template<>
class OperationTraits<OperationKind::kGemm> {
  public:
    using Id   = FTrainGemmOpId;
    using Type = GemmAttributes;

    // Converts one Gemm role ID into its PatternBuilder operation ID.
    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorId a, FTrainTensorId b, FTrainTensorId c,
                                                           FTrainTensorId d, FTrainTensorId alpha,
                                                           FTrainTensorId beta) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(a),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(b),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(c),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(alpha),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(beta)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(d)};
        return PatternOperationNode{OperationKind::kGemm, std::move(inputs), std::move(outputs)};
    }
};

template<>
class OperationTraits<OperationKind::kGroupedABCDGemm> {
  public:
    using Id   = FTrainGroupedABCDGemmOpId;
    using Type = GroupedABCDGemmAttributes;

    // Converts one GroupedABCDGemm role ID into its PatternBuilder operation ID.
    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainGroupedTensorId a, FTrainGroupedTensorId b,
                                                           FTrainGroupedTensorId c, FTrainGroupedTensorId d,
                                                           FTrainTensorId alpha, FTrainTensorId beta) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kGroupedTensor>::createPatternOperandId(a),
                                                 OperandTraits<OperandKind::kGroupedTensor>::createPatternOperandId(b),
                                                 OperandTraits<OperandKind::kGroupedTensor>::createPatternOperandId(c),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(alpha),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(beta)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kGroupedTensor>::createPatternOperandId(d)};
        return PatternOperationNode{OperationKind::kGroupedABCDGemm, std::move(inputs), std::move(outputs)};
    }
};

template<>
class OperationTraits<OperationKind::kGroupedBCDGemm> {
  public:
    using Id   = FTrainGroupedBCDGemmOpId;
    using Type = GroupedBCDGemmAttributes;

    // Converts one GroupedBCDGemm role ID into its PatternBuilder operation ID.
    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainTensorListId a, FTrainGroupedTensorId b,
                                                           FTrainGroupedTensorId c, FTrainGroupedTensorId d,
                                                           FTrainTensorId alpha, FTrainTensorId beta) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kTensorList>::createPatternOperandId(a),
                                                 OperandTraits<OperandKind::kGroupedTensor>::createPatternOperandId(b),
                                                 OperandTraits<OperandKind::kGroupedTensor>::createPatternOperandId(c),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(alpha),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(beta)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kGroupedTensor>::createPatternOperandId(d)};
        return PatternOperationNode{OperationKind::kGroupedBCDGemm, std::move(inputs), std::move(outputs)};
    }
};

template<>
class OperationTraits<OperationKind::kGroupedABGemm> {
  public:
    using Id   = FTrainGroupedABGemmOpId;
    using Type = GroupedABGemmAttributes;

    // Converts one GroupedABGemm role ID into its PatternBuilder operation ID.
    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    static PatternOperationNode createPatternOperationNode(FTrainGroupedTensorId a, FTrainGroupedTensorId b,
                                                           FTrainTensorListId c, FTrainTensorListId d,
                                                           FTrainTensorId alpha, FTrainTensorId beta) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kGroupedTensor>::createPatternOperandId(a),
                                                 OperandTraits<OperandKind::kGroupedTensor>::createPatternOperandId(b),
                                                 OperandTraits<OperandKind::kTensorList>::createPatternOperandId(c),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(alpha),
                                                 OperandTraits<OperandKind::kTensor>::createPatternOperandId(beta)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensorList>::createPatternOperandId(d)};
        return PatternOperationNode{OperationKind::kGroupedABGemm, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
