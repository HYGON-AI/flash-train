// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <flash_train/family/moe_reduce_rs/family.hpp>

#include <flash_train/primitive/hygon/distributed/moe_reduce_rs.hpp>
#include <flash_train/error.hpp>

namespace ftrain {
namespace {

const Tensor& operand(const Args& args, FTrainTensorId id) {
    const auto& value =
        args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(id));
    if (!value.has_value()) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS operand is unset"); }
    return std::get<Tensor>(*value);
}

}  // namespace

std::vector<std::string> MoEReduceRSRouteFinder::findCandidates(const MoEReduceRSProblem& problem,
                                                                const Constraints&) {
    switch (problem.attributes().getRoute()) {
        case FTRAIN_MOERS_ROUTE_A2A:
            return {"HygonMoEReduceRSA2a"};
        case FTRAIN_MOERS_ROUTE_A2A_RAGGED_SIGNAL:
            return {"HygonMoEReduceRSRaggedSignal"};
        default:
            return {};
    }
}

const MoEReduceRSFamily::Roles& MoEReduceRSFamily::makeRoles() {
    static const Roles roles = [] {
        PatternBuilder builder;
        const auto input          = builder.addOperand<OperandKind::kTensor>();
        const auto weight         = builder.addOperand<OperandKind::kTensor>();
        const auto chosen_experts = builder.addOperand<OperandKind::kTensor>();
        const auto expert_weight  = builder.addOperand<OperandKind::kTensor>();
        const auto output         = builder.addOperand<OperandKind::kTensor>();
        const auto op             = builder.addOperation<OperationKind::kMoEReduceRS>(
            input, weight, chosen_experts, expert_weight, output);
        return Roles{builder.buildPattern(), input, weight, chosen_experts, expert_weight, output, op};
    }();
    return roles;
}

MoEReduceRSProblem MoEReduceRSFamily::makeProblem(const Args& args) {
    const auto& roles = makeRoles();
    const auto& attrs = std::get<MoEReduceRSAttributes>(
        *args.getOpArgument(OperationTraits<OperationKind::kMoEReduceRS>::createPatternOperationId(roles.op)));
    return MoEReduceRSProblem(operand(args, roles.input), operand(args, roles.weight),
                              operand(args, roles.chosen_experts), operand(args, roles.expert_weight),
                              operand(args, roles.output), attrs);
}

std::vector<std::shared_ptr<const Primitive<MoEReduceRSProblem>>> MoEReduceRSFamily::makeRecords() {
    return {std::make_shared<HygonMoEReduceRSPrimitive>(FTRAIN_MOERS_ROUTE_A2A),
            std::make_shared<HygonMoEReduceRSPrimitive>(FTRAIN_MOERS_ROUTE_A2A_RAGGED_SIGNAL)};
}

}  // namespace ftrain
