// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_TOPK_THD_FAMILY_HPP_
#define FTRAIN_FAMILY_INDEXER_TOPK_THD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/indexer_topk_thd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the IndexerTopKThd family contributes to its engine.
struct IndexerTopKThdFamily {
    using Problem = IndexerTopKThdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId input_values;
        FTrainTensorId cu_seqlens_q;
        FTrainTensorId cu_seqlens_kv;
        FTrainTensorId q_causal_offsets;
        FTrainTensorId out_indices;
        FTrainTensorId out_values;
        FTrainTensorId ratio;
        FTrainTensorId top_k;
        FTrainTensorId order_values;
        FTrainIndexerTopKThdOpId indexer_topk_thd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId input_values               = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId cu_seqlens_q               = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId cu_seqlens_kv              = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId q_causal_offsets           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId out_indices                = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId out_values                 = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId ratio                      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId top_k                      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId order_values               = pattern.addOperand<OperandKind::kTensor>();
            const FTrainIndexerTopKThdOpId indexer_topk_thd = pattern.addOperation<OperationKind::kIndexerTopKThd>(
                input_values, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, ratio, top_k,
                order_values);
            return Roles{pattern.buildPattern(),
                         input_values,
                         cu_seqlens_q,
                         cu_seqlens_kv,
                         q_causal_offsets,
                         out_indices,
                         out_values,
                         ratio,
                         top_k,
                         order_values,
                         indexer_topk_thd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<IndexerTopKThdProblem>>> makeRecords();

    static IndexerTopKThdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return IndexerTopKThdProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.input_values))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.cu_seqlens_q))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.cu_seqlens_kv))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.q_causal_offsets))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.out_indices))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.out_values))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.ratio))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.top_k))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.order_values))),
            std::get<IndexerTopKThdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kIndexerTopKThd>::createPatternOperationId(roles.indexer_topk_thd)))};
    }
};

}  // namespace ftrain

#endif
