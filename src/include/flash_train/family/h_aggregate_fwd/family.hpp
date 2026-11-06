// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_AGGREGATE_FWD_FAMILY_HPP_
#define FTRAIN_FAMILY_H_AGGREGATE_FWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/h_aggregate_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the HAggregateFwd family contributes to its engine.
struct HAggregateFwdFamily {
    using Problem = HAggregateFwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId x;
        FTrainTensorId h_pre;
        FTrainTensorId aggregated;
        FTrainHAggregateFwdOpId h_aggregate_fwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId x          = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId h_pre      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId aggregated = pattern.addOperand<OperandKind::kTensor>();
            const FTrainHAggregateFwdOpId h_aggregate_fwd =
                pattern.addOperation<OperationKind::kHAggregateFwd>(x, h_pre, aggregated);
            return Roles{pattern.buildPattern(), x, h_pre, aggregated, h_aggregate_fwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<HAggregateFwdProblem>>> makeRecords();

    static HAggregateFwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return HAggregateFwdProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.x))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.h_pre))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.aggregated))),
            std::get<HAggregateFwdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kHAggregateFwd>::createPatternOperationId(roles.h_aggregate_fwd)))};
    }
};

}  // namespace ftrain

#endif
