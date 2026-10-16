// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_TARGET_FAMILY_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_TARGET_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/sparse_attn_target/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the SparseAttnTarget family contributes to its engine.
struct SparseAttnTargetFamily {
    using Problem = SparseAttnTargetProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId q;
        FTrainTensorId k;
        FTrainTensorId lse;
        FTrainTensorId topk_indices;
        FTrainTensorId target;
        FTrainTensorId softmax_scale;
        FTrainTensorId topk_indices_global;
        FTrainSparseAttnTargetOpId sparse_attn_target;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId q                   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId k                   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId lse                 = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_indices        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId target              = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId softmax_scale       = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_indices_global = pattern.addOperand<OperandKind::kTensor>();
            const FTrainSparseAttnTargetOpId sparse_attn_target =
                pattern.addOperation<OperationKind::kSparseAttnTarget>(q, k, lse, topk_indices, target, softmax_scale,
                                                                       topk_indices_global);
            return Roles{pattern.buildPattern(), q, k, lse, topk_indices, target, softmax_scale, topk_indices_global,
                         sparse_attn_target};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<SparseAttnTargetProblem>>> makeRecords();

    static SparseAttnTargetProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return SparseAttnTargetProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.q))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.k))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.lse))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_indices))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.target))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.softmax_scale))),
            std::get<Tensor>(*args.getOperand(
                OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_indices_global))),
            std::get<SparseAttnTargetAttributes>(
                *args.getOpArgument(OperationTraits<OperationKind::kSparseAttnTarget>::createPatternOperationId(
                    roles.sparse_attn_target)))};
    }
};

}  // namespace ftrain

#endif
