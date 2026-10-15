// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_POST_BDA_FWD_FAMILY_HPP_
#define FTRAIN_FAMILY_H_POST_BDA_FWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/h_post_bda_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the HPostBdaFwd family contributes to its engine.
struct HPostBdaFwdFamily {
    using Problem = HPostBdaFwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId h_res;
        FTrainTensorId residual;
        FTrainTensorId h_post;
        FTrainTensorId x;
        FTrainTensorId bias;
        FTrainTensorId output;
        FTrainHPostBdaFwdOpId h_post_bda_fwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId h_res    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId residual = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId h_post   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId x        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId bias     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId output   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainHPostBdaFwdOpId h_post_bda_fwd =
                pattern.addOperation<OperationKind::kHPostBdaFwd>(h_res, residual, h_post, x, bias, output);
            return Roles{pattern.buildPattern(), h_res, residual, h_post, x, bias, output, h_post_bda_fwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<HPostBdaFwdProblem>>> makeRecords();

    static HPostBdaFwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return HPostBdaFwdProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.h_res))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.residual))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.h_post))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.x))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.bias))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.output))),
            std::get<HPostBdaFwdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kHPostBdaFwd>::createPatternOperationId(roles.h_post_bda_fwd)))};
    }
};

}  // namespace ftrain

#endif
