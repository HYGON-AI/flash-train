// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_OPERATION_GDN_HPP_
#define FTRAIN_OPERATION_GDN_HPP_

#include <cstddef>
#include <type_traits>
#include <vector>

#include "flash_train/operation/base.hpp"

namespace ftrain {

template<OperationKind Kind>
class GdnAttributes final {
  public:
    explicit GdnAttributes(float scale) noexcept : scale_(scale) {}
    float getScale() const noexcept { return scale_; }

  private:
    float scale_;
};

template<OperationKind Kind, typename Id, std::size_t Inputs>
struct GdnOperationTraits {
    using Type = GdnAttributes<Kind>;
    using OperationId = Id;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    template<typename... TensorIds>
    static PatternOperationNode createPatternOperationNode(TensorIds... ids) {
        static_assert((std::is_same_v<TensorIds, FTrainTensorId> && ...), "GDN ports must be tensors");
        static_assert(sizeof...(ids) > Inputs, "GDN needs output ports");
        const PatternOperandId operands[] = {PatternOperandId{ids.opaque}...};
        return PatternOperationNode{Kind, std::vector<PatternOperandId>(operands, operands + Inputs),
                                    std::vector<PatternOperandId>(operands + Inputs, operands + sizeof...(ids))};
    }
};

template<>
class OperationTraits<OperationKind::kGdnForward>
    : public GdnOperationTraits<OperationKind::kGdnForward, FTrainGdnForwardOpId, 6> {
  public:
    using Id = FTrainGdnForwardOpId;
};
template<>
class OperationTraits<OperationKind::kGdnBackward>
    : public GdnOperationTraits<OperationKind::kGdnBackward, FTrainGdnBackwardOpId, 8> {
  public:
    using Id = FTrainGdnBackwardOpId;
};
template<>
class OperationTraits<OperationKind::kGdn2Forward>
    : public GdnOperationTraits<OperationKind::kGdn2Forward, FTrainGdn2ForwardOpId, 6> {
  public:
    using Id = FTrainGdn2ForwardOpId;
};
template<>
class OperationTraits<OperationKind::kGdn2Backward>
    : public GdnOperationTraits<OperationKind::kGdn2Backward, FTrainGdn2BackwardOpId, 7> {
  public:
    using Id = FTrainGdn2BackwardOpId;
};

}  // namespace ftrain

#endif
