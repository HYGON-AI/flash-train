// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_BWD_FAMILY_HPP_
#define FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_BWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/proj_rms_compute_h_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the ProjRmsComputeHBwd family contributes to its engine.
struct ProjRmsComputeHBwdFamily {
    using Problem = ProjRmsComputeHBwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId grad_h_pre;
        FTrainTensorId grad_h_post;
        FTrainTensorId grad_h_res;
        FTrainTensorId grad_r;
        FTrainTensorId h_pre;
        FTrainTensorId h_post;
        FTrainTensorId proj;
        FTrainTensorId r;
        FTrainTensorId x;
        FTrainTensorId weight;
        FTrainTensorId alpha_pre;
        FTrainTensorId alpha_post;
        FTrainTensorId alpha_res;
        FTrainTensorId grad_x;
        FTrainTensorId grad_weight;
        FTrainTensorId grad_alpha;
        FTrainTensorId grad_bias;
        FTrainTensorId n;
        FTrainTensorId eps;
        FTrainTensorId compute_h_eps;
        FTrainProjRmsComputeHBwdOpId proj_rms_compute_h_bwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId grad_h_pre    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_h_post   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_h_res    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_r        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId h_pre         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId h_post        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId proj          = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId r             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId x             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId weight        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId alpha_pre     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId alpha_post    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId alpha_res     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_x        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_weight   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_alpha    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_bias     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId n             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId eps           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId compute_h_eps = pattern.addOperand<OperandKind::kTensor>();
            const FTrainProjRmsComputeHBwdOpId proj_rms_compute_h_bwd =
                pattern.addOperation<OperationKind::kProjRmsComputeHBwd>(
                    grad_h_pre, grad_h_post, grad_h_res, grad_r, h_pre, h_post, proj, r, x, weight, alpha_pre,
                    alpha_post, alpha_res, grad_x, grad_weight, grad_alpha, grad_bias, n, eps, compute_h_eps);
            return Roles{pattern.buildPattern(),
                         grad_h_pre,
                         grad_h_post,
                         grad_h_res,
                         grad_r,
                         h_pre,
                         h_post,
                         proj,
                         r,
                         x,
                         weight,
                         alpha_pre,
                         alpha_post,
                         alpha_res,
                         grad_x,
                         grad_weight,
                         grad_alpha,
                         grad_bias,
                         n,
                         eps,
                         compute_h_eps,
                         proj_rms_compute_h_bwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<ProjRmsComputeHBwdProblem>>> makeRecords();

    static ProjRmsComputeHBwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return ProjRmsComputeHBwdProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_h_pre))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_h_post))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_h_res))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_r))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.h_pre))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.h_post))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.proj))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.r))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.x))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.weight))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.alpha_pre))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.alpha_post))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.alpha_res))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_x))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_weight))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_alpha))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_bias))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.n))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.eps))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.compute_h_eps))),
            std::get<ProjRmsComputeHBwdAttributes>(
                *args.getOpArgument(OperationTraits<OperationKind::kProjRmsComputeHBwd>::createPatternOperationId(
                    roles.proj_rms_compute_h_bwd)))};
    }
};

}  // namespace ftrain

#endif
