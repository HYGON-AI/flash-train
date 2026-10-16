// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_POST_BDA_BWD_FAMILY_HPP_
#define FTRAIN_FAMILY_H_POST_BDA_BWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/h_post_bda_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the HPostBdaBwd family contributes to its engine.
struct HPostBdaBwdFamily {
    using Problem = HPostBdaBwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId grad_output;
        FTrainTensorId residual;
        FTrainTensorId x;
        FTrainTensorId bias;
        FTrainTensorId h_res;
        FTrainTensorId h_post;
        FTrainTensorId grad_residual;
        FTrainTensorId grad_x;
        FTrainTensorId grad_h_res;
        FTrainTensorId grad_h_post;
        FTrainTensorId grad_bias;
        FTrainHPostBdaBwdOpId h_post_bda_bwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId grad_output           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId residual              = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId x                     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId bias                  = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId h_res                 = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId h_post                = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_residual         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_x                = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_h_res            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_h_post           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_bias             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainHPostBdaBwdOpId h_post_bda_bwd = pattern.addOperation<OperationKind::kHPostBdaBwd>(
                grad_output, residual, x, bias, h_res, h_post, grad_residual, grad_x, grad_h_res, grad_h_post,
                grad_bias);
            return Roles{pattern.buildPattern(), grad_output, residual,   x,           bias,      h_res,         h_post,
                         grad_residual,          grad_x,      grad_h_res, grad_h_post, grad_bias, h_post_bda_bwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<HPostBdaBwdProblem>>> makeRecords();

    static HPostBdaBwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return HPostBdaBwdProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_output))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.residual))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.x))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.bias))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.h_res))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.h_post))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_residual))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_x))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_h_res))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_h_post))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_bias))),
            std::get<HPostBdaBwdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kHPostBdaBwd>::createPatternOperationId(roles.h_post_bda_bwd)))};
    }
};

}  // namespace ftrain

#endif
