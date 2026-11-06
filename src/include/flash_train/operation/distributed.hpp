// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_OPERATION_DISTRIBUTED_HPP_
#define FTRAIN_OPERATION_DISTRIBUTED_HPP_

#include <cstdint>
#include <vector>

#include <flash_train/common.h>
#include <flash_train/operation/base.hpp>

namespace ftrain {

// Distributed operator invocations are bound to a process-wide session id.
// The session owns communicators, persistent buffers, streams and events; the
// Plan and Primitive only reference it.  The id is opaque to this core layer.
class AGGroupGemmAttributes final {
  public:
    AGGroupGemmAttributes(FTrainAGRoute route, std::uint64_t session_id) noexcept
        : route_(route), session_id_(session_id) {}

    std::uint64_t getSessionId() const noexcept { return session_id_; }
    FTrainAGRoute getRoute() const noexcept { return route_; }

  private:
    FTrainAGRoute route_;
    std::uint64_t session_id_;
};

class MoEReduceRSAttributes final {
  public:
    MoEReduceRSAttributes(FTrainMoERSRoute route, FTrainMoERSGemmVariant gemm_variant,
                          std::uint64_t session_id) noexcept
        : route_(route), gemm_variant_(gemm_variant), session_id_(session_id) {}

    std::uint64_t getSessionId() const noexcept { return session_id_; }
    FTrainMoERSRoute getRoute() const noexcept { return route_; }
    FTrainMoERSGemmVariant getGemmVariant() const noexcept { return gemm_variant_; }

  private:
    FTrainMoERSRoute route_;
    FTrainMoERSGemmVariant gemm_variant_;
    std::uint64_t session_id_;
};

template<>
class OperationTraits<OperationKind::kAGGroupGemm> {
  public:
    using Id   = FTrainAGGroupGemmOpId;
    using Type = AGGroupGemmAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    // input, weight, topk_ids; output.
    static PatternOperationNode createPatternOperationNode(FTrainTensorId input, FTrainTensorId weight,
                                                           FTrainTensorId topk_ids, FTrainTensorId output) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(input),
                                               OperandTraits<OperandKind::kTensor>::createPatternOperandId(weight),
                                               OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_ids)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(output)};
        return PatternOperationNode{OperationKind::kAGGroupGemm, std::move(inputs), std::move(outputs)};
    }
};

template<>
class OperationTraits<OperationKind::kMoEReduceRS> {
  public:
    using Id   = FTrainMoEReduceRSOpId;
    using Type = MoEReduceRSAttributes;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    // input, weight, chosen_experts, expert_weight; output.
    static PatternOperationNode createPatternOperationNode(FTrainTensorId input, FTrainTensorId weight,
                                                           FTrainTensorId chosen_experts,
                                                           FTrainTensorId expert_weight, FTrainTensorId output) {
        std::vector<PatternOperandId> inputs  = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(input),
                                               OperandTraits<OperandKind::kTensor>::createPatternOperandId(weight),
                                               OperandTraits<OperandKind::kTensor>::createPatternOperandId(chosen_experts),
                                               OperandTraits<OperandKind::kTensor>::createPatternOperandId(expert_weight)};
        std::vector<PatternOperandId> outputs = {OperandTraits<OperandKind::kTensor>::createPatternOperandId(output)};
        return PatternOperationNode{OperationKind::kMoEReduceRS, std::move(inputs), std::move(outputs)};
    }
};

}  // namespace ftrain

#endif
