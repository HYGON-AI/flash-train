// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_FWD_FAMILY_HPP_
#define FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_FWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/proj_rms_compute_h_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the ProjRmsComputeHFwd family contributes to its engine.
struct ProjRmsComputeHFwdFamily {
    using Problem = ProjRmsComputeHFwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId x;
        FTrainTensorId weight;
        FTrainTensorId alpha_pre;
        FTrainTensorId alpha_post;
        FTrainTensorId alpha_res;
        FTrainTensorId bias;
        FTrainTensorId h_pre;
        FTrainTensorId h_post;
        FTrainTensorId h_res;
        FTrainTensorId r;
        FTrainTensorId proj;
        FTrainTensorId n;
        FTrainTensorId eps;
        FTrainTensorId compute_h_eps;
        FTrainProjRmsComputeHFwdOpId proj_rms_compute_h_fwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId x             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId weight        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId alpha_pre     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId alpha_post    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId alpha_res     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId bias          = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId h_pre         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId h_post        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId h_res         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId r             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId proj          = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId n             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId eps           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId compute_h_eps = pattern.addOperand<OperandKind::kTensor>();
            const FTrainProjRmsComputeHFwdOpId proj_rms_compute_h_fwd =
                pattern.addOperation<OperationKind::kProjRmsComputeHFwd>(x, weight, alpha_pre, alpha_post, alpha_res,
                                                                         bias, h_pre, h_post, h_res, r, proj, n, eps,
                                                                         compute_h_eps);
            return Roles{pattern.buildPattern(),
                         x,
                         weight,
                         alpha_pre,
                         alpha_post,
                         alpha_res,
                         bias,
                         h_pre,
                         h_post,
                         h_res,
                         r,
                         proj,
                         n,
                         eps,
                         compute_h_eps,
                         proj_rms_compute_h_fwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<ProjRmsComputeHFwdProblem>>> makeRecords();

    static ProjRmsComputeHFwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return ProjRmsComputeHFwdProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.x))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.weight))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.alpha_pre))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.alpha_post))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.alpha_res))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.bias))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.h_pre))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.h_post))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.h_res))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.r))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.proj))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.n))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.eps))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.compute_h_eps))),
            std::get<ProjRmsComputeHFwdAttributes>(
                *args.getOpArgument(OperationTraits<OperationKind::kProjRmsComputeHFwd>::createPatternOperationId(
                    roles.proj_rms_compute_h_fwd)))};
    }
};

}  // namespace ftrain

#endif
