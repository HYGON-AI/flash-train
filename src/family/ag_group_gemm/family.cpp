// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <flash_train/family/ag_group_gemm/family.hpp>

#include <flash_train/primitive/hygon/distributed/ag_group_gemm.hpp>
#include <flash_train/error.hpp>

namespace ftrain {
namespace {

const Tensor& operand(const Args& args, FTrainTensorId id) {
    const auto& value =
        args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(id));
    if (!value.has_value()) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm operand is unset"); }
    return std::get<Tensor>(*value);
}

}  // namespace

std::vector<std::string> AGGroupGemmRouteFinder::findCandidates(const AGGroupGemmProblem& problem,
                                                                const Constraints&) {
    switch (problem.attributes().getRoute()) {
        case FTRAIN_AG_ROUTE_NATIVE_NK_A2A:
            return {"HygonAGGroupGemmNativeNkA2a"};
        case FTRAIN_AG_ROUTE_NATIVE_KN_A2A:
            return {"HygonAGGroupGemmNativeKnA2a"};
        case FTRAIN_AG_ROUTE_NATIVE_KN_RING_FIXED:
            return {"HygonAGGroupGemmNativeKnRingFixed"};
        case FTRAIN_AG_ROUTE_NATIVE_KN_RING:
            return {"HygonAGGroupGemmNativeKnRing"};
        case FTRAIN_AG_ROUTE_NATIVE_KN_RING_READY_MASK:
            return {"HygonAGGroupGemmNativeKnRingReadyMask"};
        default:
            return {};
    }
}

const AGGroupGemmFamily::Roles& AGGroupGemmFamily::makeRoles() {
    static const Roles roles = [] {
        PatternBuilder builder;
        const auto input    = builder.addOperand<OperandKind::kTensor>();
        const auto weight   = builder.addOperand<OperandKind::kTensor>();
        const auto topk_ids = builder.addOperand<OperandKind::kTensor>();
        const auto output   = builder.addOperand<OperandKind::kTensor>();
        const auto op       = builder.addOperation<OperationKind::kAGGroupGemm>(input, weight, topk_ids, output);
        return Roles{builder.buildPattern(), input, weight, topk_ids, output, op};
    }();
    return roles;
}

AGGroupGemmProblem AGGroupGemmFamily::makeProblem(const Args& args) {
    const auto& roles = makeRoles();
    const auto& attrs = std::get<AGGroupGemmAttributes>(
        *args.getOpArgument(OperationTraits<OperationKind::kAGGroupGemm>::createPatternOperationId(roles.op)));
    return AGGroupGemmProblem(operand(args, roles.input), operand(args, roles.weight), operand(args, roles.topk_ids),
                              operand(args, roles.output), attrs);
}

std::vector<std::shared_ptr<const Primitive<AGGroupGemmProblem>>> AGGroupGemmFamily::makeRecords() {
    return {std::make_shared<HygonAGGroupGemmPrimitive>(FTRAIN_AG_ROUTE_NATIVE_NK_A2A),
            std::make_shared<HygonAGGroupGemmPrimitive>(FTRAIN_AG_ROUTE_NATIVE_KN_A2A),
            std::make_shared<HygonAGGroupGemmPrimitive>(FTRAIN_AG_ROUTE_NATIVE_KN_RING_FIXED),
            std::make_shared<HygonAGGroupGemmPrimitive>(FTRAIN_AG_ROUTE_NATIVE_KN_RING),
            std::make_shared<HygonAGGroupGemmPrimitive>(FTRAIN_AG_ROUTE_NATIVE_KN_RING_READY_MASK)};
}

}  // namespace ftrain
