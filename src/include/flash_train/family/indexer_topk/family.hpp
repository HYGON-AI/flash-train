// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_TOPK_FAMILY_HPP_
#define FTRAIN_FAMILY_INDEXER_TOPK_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/indexer_topk/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the IndexerTopK family contributes to its engine.
struct IndexerTopKFamily {
    using Problem = IndexerTopKProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId input_values;
        FTrainTensorId seq_lens;
        FTrainTensorId out_indices;
        FTrainTensorId out_values;
        FTrainTensorId top_k;
        FTrainTensorId next_n;
        FTrainTensorId order_values;
        FTrainIndexerTopKOpId indexer_topk;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId input_values        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId seq_lens            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId out_indices         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId out_values          = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId top_k               = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId next_n              = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId order_values        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainIndexerTopKOpId indexer_topk = pattern.addOperation<OperationKind::kIndexerTopK>(
                input_values, seq_lens, out_indices, out_values, top_k, next_n, order_values);
            return Roles{pattern.buildPattern(), input_values, seq_lens, out_indices, out_values, top_k, next_n,
                         order_values,           indexer_topk};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<IndexerTopKProblem>>> makeRecords();

    static IndexerTopKProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return IndexerTopKProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.input_values))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.seq_lens))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.out_indices))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.out_values))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.top_k))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.next_n))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.order_values))),
            std::get<IndexerTopKAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kIndexerTopK>::createPatternOperationId(roles.indexer_topk)))};
    }
};

}  // namespace ftrain

#endif
