// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_ATTN_SCORE_FAMILY_HPP_
#define FTRAIN_FAMILY_DENSE_ATTN_SCORE_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/dense_attn_score/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the DenseAttnScore family contributes to its engine.
struct DenseAttnScoreFamily {
    using Problem = DenseAttnScoreProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId q;
        FTrainTensorId k;
        FTrainTensorId lse;
        FTrainTensorId out;
        FTrainTensorId denom;
        FTrainTensorId ratio;
        FTrainTensorId softmax_scale;
        FTrainDenseAttnScoreOpId dense_attn_score;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId q             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId k             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId lse           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId out           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId denom         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId ratio         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId softmax_scale = pattern.addOperand<OperandKind::kTensor>();
            const FTrainDenseAttnScoreOpId dense_attn_score =
                pattern.addOperation<OperationKind::kDenseAttnScore>(q, k, lse, out, denom, ratio, softmax_scale);
            return Roles{pattern.buildPattern(), q, k, lse, out, denom, ratio, softmax_scale, dense_attn_score};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<DenseAttnScoreProblem>>> makeRecords();

    static DenseAttnScoreProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return DenseAttnScoreProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.q))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.k))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.lse))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.out))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.denom))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.ratio))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.softmax_scale))),
            std::get<DenseAttnScoreAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kDenseAttnScore>::createPatternOperationId(roles.dense_attn_score)))};
    }
};

}  // namespace ftrain

#endif
