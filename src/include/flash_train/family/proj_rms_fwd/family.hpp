// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_PROJ_RMS_FWD_FAMILY_HPP_
#define FTRAIN_FAMILY_PROJ_RMS_FWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/proj_rms_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the ProjRmsFwd family contributes to its engine.
struct ProjRmsFwdFamily {
    using Problem = ProjRmsFwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId x;
        FTrainTensorId weight;
        FTrainTensorId proj;
        FTrainTensorId r;
        FTrainTensorId eps;
        FTrainProjRmsFwdOpId proj_rms_fwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId x      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId weight = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId proj   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId r      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId eps    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainProjRmsFwdOpId proj_rms_fwd =
                pattern.addOperation<OperationKind::kProjRmsFwd>(x, weight, proj, r, eps);
            return Roles{pattern.buildPattern(), x, weight, proj, r, eps, proj_rms_fwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<ProjRmsFwdProblem>>> makeRecords();

    static ProjRmsFwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return ProjRmsFwdProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.x))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.weight))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.proj))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.r))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.eps))),
            std::get<ProjRmsFwdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kProjRmsFwd>::createPatternOperationId(roles.proj_rms_fwd)))};
    }
};

}  // namespace ftrain

#endif
