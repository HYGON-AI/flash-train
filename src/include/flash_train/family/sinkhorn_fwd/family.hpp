// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SINKHORN_FWD_FAMILY_HPP_
#define FTRAIN_FAMILY_SINKHORN_FWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/sinkhorn_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the SinkhornFwd family contributes to its engine.
struct SinkhornFwdFamily {
    using Problem = SinkhornFwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId logits;
        FTrainTensorId output;
        FTrainTensorId m_init;
        FTrainTensorId num_iterations;
        FTrainTensorId eps;
        FTrainSinkhornFwdOpId sinkhorn_fwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId logits         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId output         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId m_init         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId num_iterations = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId eps            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainSinkhornFwdOpId sinkhorn_fwd =
                pattern.addOperation<OperationKind::kSinkhornFwd>(logits, output, m_init, num_iterations, eps);
            return Roles{pattern.buildPattern(), logits, output, m_init, num_iterations, eps, sinkhorn_fwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<SinkhornFwdProblem>>> makeRecords();

    static SinkhornFwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return SinkhornFwdProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.logits))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.output))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.m_init))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.num_iterations))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.eps))),
            std::get<SinkhornFwdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kSinkhornFwd>::createPatternOperationId(roles.sinkhorn_fwd)))};
    }
};

}  // namespace ftrain

#endif
