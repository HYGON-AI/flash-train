// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_ATTN_SCORE_THD_FAMILY_HPP_
#define FTRAIN_FAMILY_DENSE_ATTN_SCORE_THD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/dense_attn_score_thd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the DenseAttnScoreThd family contributes to its engine.
struct DenseAttnScoreThdFamily {
    using Problem = DenseAttnScoreThdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId q;
        FTrainTensorId k;
        FTrainTensorId lse;
        FTrainTensorId cu_seqlens_q;
        FTrainTensorId cu_seqlens_kv;
        FTrainTensorId q_causal_offsets;
        FTrainTensorId out;
        FTrainTensorId denom;
        FTrainTensorId ratio;
        FTrainTensorId softmax_scale;
        FTrainTensorId max_seqlen_kv;
        FTrainDenseAttnScoreThdOpId dense_attn_score_thd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId q                = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId k                = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId lse              = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId cu_seqlens_q     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId cu_seqlens_kv    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId q_causal_offsets = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId out              = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId denom            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId ratio            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId softmax_scale    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId max_seqlen_kv    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainDenseAttnScoreThdOpId dense_attn_score_thd =
                pattern.addOperation<OperationKind::kDenseAttnScoreThd>(q, k, lse, cu_seqlens_q, cu_seqlens_kv,
                                                                        q_causal_offsets, out, denom, ratio,
                                                                        softmax_scale, max_seqlen_kv);
            return Roles{pattern.buildPattern(), q,   k,     lse,   cu_seqlens_q,  cu_seqlens_kv,
                         q_causal_offsets,       out, denom, ratio, softmax_scale, max_seqlen_kv,
                         dense_attn_score_thd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<DenseAttnScoreThdProblem>>> makeRecords();

    static DenseAttnScoreThdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return DenseAttnScoreThdProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.q))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.k))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.lse))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.cu_seqlens_q))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.cu_seqlens_kv))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.q_causal_offsets))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.out))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.denom))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.ratio))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.softmax_scale))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.max_seqlen_kv))),
            std::get<DenseAttnScoreThdAttributes>(
                *args.getOpArgument(OperationTraits<OperationKind::kDenseAttnScoreThd>::createPatternOperationId(
                    roles.dense_attn_score_thd)))};
    }
};

}  // namespace ftrain

#endif
