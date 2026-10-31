// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_FWD_FAMILY_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_FWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/sparse_attn_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the SparseAttnFwd family contributes to its engine.
struct SparseAttnFwdFamily {
    using Problem = SparseAttnFwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId q;
        FTrainTensorId kv;
        FTrainTensorId attn_sink;
        FTrainTensorId topk_indices;
        FTrainTensorId topk_length;
        FTrainTensorId out;
        FTrainTensorId lse;
        FTrainTensorId softmax_scale;
        FTrainSparseAttnFwdOpId sparse_attn_fwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId q                        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId kv                       = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId attn_sink                = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_indices             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_length              = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId out                      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId lse                      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId softmax_scale            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainSparseAttnFwdOpId sparse_attn_fwd = pattern.addOperation<OperationKind::kSparseAttnFwd>(
                q, kv, attn_sink, topk_indices, topk_length, out, lse, softmax_scale);
            return Roles{pattern.buildPattern(), q, kv, attn_sink, topk_indices, topk_length, out, lse, softmax_scale,
                         sparse_attn_fwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<SparseAttnFwdProblem>>> makeRecords();

    static SparseAttnFwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return SparseAttnFwdProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.q))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.kv))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.attn_sink))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_indices))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_length))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.out))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.lse))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.softmax_scale))),
            std::get<SparseAttnFwdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kSparseAttnFwd>::createPatternOperationId(roles.sparse_attn_fwd)))};
    }
};

}  // namespace ftrain

#endif
