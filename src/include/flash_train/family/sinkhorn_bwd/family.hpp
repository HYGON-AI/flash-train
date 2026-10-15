// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SINKHORN_BWD_FAMILY_HPP_
#define FTRAIN_FAMILY_SINKHORN_BWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/sinkhorn_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the SinkhornBwd family contributes to its engine.
struct SinkhornBwdFamily {
    using Problem = SinkhornBwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId grad_output;
        FTrainTensorId m_init;
        FTrainTensorId grad_logits;
        FTrainTensorId num_iterations;
        FTrainTensorId eps;
        FTrainSinkhornBwdOpId sinkhorn_bwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId grad_output         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId m_init              = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_logits         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId num_iterations      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId eps                 = pattern.addOperand<OperandKind::kTensor>();
            const FTrainSinkhornBwdOpId sinkhorn_bwd = pattern.addOperation<OperationKind::kSinkhornBwd>(
                grad_output, m_init, grad_logits, num_iterations, eps);
            return Roles{pattern.buildPattern(), grad_output, m_init, grad_logits, num_iterations, eps, sinkhorn_bwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<SinkhornBwdProblem>>> makeRecords();

    static SinkhornBwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return SinkhornBwdProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_output))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.m_init))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_logits))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.num_iterations))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.eps))),
            std::get<SinkhornBwdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kSinkhornBwd>::createPatternOperationId(roles.sinkhorn_bwd)))};
    }
};

}  // namespace ftrain

#endif
