// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_INDEXER_SCORE_FAMILY_HPP_
#define FTRAIN_FAMILY_DENSE_INDEXER_SCORE_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/dense_indexer_score/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the DenseIndexerScore family contributes to its engine.
struct DenseIndexerScoreFamily {
    using Problem = DenseIndexerScoreProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId q;
        FTrainTensorId k;
        FTrainTensorId w;
        FTrainTensorId out;
        FTrainTensorId denom;
        FTrainTensorId ratio;
        FTrainTensorId sm_scale;
        FTrainDenseIndexerScoreOpId dense_indexer_score;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId q        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId k        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId w        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId out      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId denom    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId ratio    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId sm_scale = pattern.addOperand<OperandKind::kTensor>();
            const FTrainDenseIndexerScoreOpId dense_indexer_score =
                pattern.addOperation<OperationKind::kDenseIndexerScore>(q, k, w, out, denom, ratio, sm_scale);
            return Roles{pattern.buildPattern(), q, k, w, out, denom, ratio, sm_scale, dense_indexer_score};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<DenseIndexerScoreProblem>>> makeRecords();

    static DenseIndexerScoreProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return DenseIndexerScoreProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.q))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.k))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.w))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.out))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.denom))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.ratio))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.sm_scale))),
            std::get<DenseIndexerScoreAttributes>(
                *args.getOpArgument(OperationTraits<OperationKind::kDenseIndexerScore>::createPatternOperationId(
                    roles.dense_indexer_score)))};
    }
};

}  // namespace ftrain

#endif
