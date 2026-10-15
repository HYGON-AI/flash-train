// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_PROJ_RMS_BWD_FAMILY_HPP_
#define FTRAIN_FAMILY_PROJ_RMS_BWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/proj_rms_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the ProjRmsBwd family contributes to its engine.
struct ProjRmsBwdFamily {
    using Problem = ProjRmsBwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId grad_proj;
        FTrainTensorId grad_r;
        FTrainTensorId x;
        FTrainTensorId weight;
        FTrainTensorId r;
        FTrainTensorId grad_x;
        FTrainTensorId grad_weight;
        FTrainTensorId eps;
        FTrainProjRmsBwdOpId proj_rms_bwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId grad_proj          = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_r             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId x                  = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId weight             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId r                  = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_x             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_weight        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId eps                = pattern.addOperand<OperandKind::kTensor>();
            const FTrainProjRmsBwdOpId proj_rms_bwd = pattern.addOperation<OperationKind::kProjRmsBwd>(
                grad_proj, grad_r, x, weight, r, grad_x, grad_weight, eps);
            return Roles{
                pattern.buildPattern(), grad_proj, grad_r, x, weight, r, grad_x, grad_weight, eps, proj_rms_bwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<ProjRmsBwdProblem>>> makeRecords();

    static ProjRmsBwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return ProjRmsBwdProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_proj))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_r))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.x))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.weight))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.r))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_x))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_weight))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.eps))),
            std::get<ProjRmsBwdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kProjRmsBwd>::createPatternOperationId(roles.proj_rms_bwd)))};
    }
};

}  // namespace ftrain

#endif
