// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_AGGREGATE_BWD_FAMILY_HPP_
#define FTRAIN_FAMILY_H_AGGREGATE_BWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/h_aggregate_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the HAggregateBwd family contributes to its engine.
struct HAggregateBwdFamily {
    using Problem = HAggregateBwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId grad_aggregated;
        FTrainTensorId x;
        FTrainTensorId h_pre;
        FTrainTensorId grad_x;
        FTrainTensorId grad_h_pre;
        FTrainHAggregateBwdOpId h_aggregate_bwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId grad_aggregated = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId x               = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId h_pre           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_x          = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_h_pre      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainHAggregateBwdOpId h_aggregate_bwd =
                pattern.addOperation<OperationKind::kHAggregateBwd>(grad_aggregated, x, h_pre, grad_x, grad_h_pre);
            return Roles{pattern.buildPattern(), grad_aggregated, x, h_pre, grad_x, grad_h_pre, h_aggregate_bwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<HAggregateBwdProblem>>> makeRecords();

    static HAggregateBwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return HAggregateBwdProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_aggregated))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.x))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.h_pre))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_x))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_h_pre))),
            std::get<HAggregateBwdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kHAggregateBwd>::createPatternOperationId(roles.h_aggregate_bwd)))};
    }
};

}  // namespace ftrain

#endif
