// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_FWD_INDEXER_FAMILY_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_FWD_INDEXER_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/sparse_attn_fwd_indexer/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the SparseAttnFwdIndexer family contributes to its engine.
struct SparseAttnFwdIndexerFamily {
    using Problem = SparseAttnFwdIndexerProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId q;
        FTrainTensorId kv;
        FTrainTensorId attn_sink;
        FTrainTensorId topk_indices;
        FTrainTensorId topk_length;
        FTrainTensorId out;
        FTrainTensorId max_logits;
        FTrainTensorId lse;
        FTrainTensorId lse_indexer;
        FTrainTensorId softmax_scale;
        FTrainTensorId indexer_topk;
        FTrainSparseAttnFwdIndexerOpId sparse_attn_fwd_indexer;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId q             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId kv            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId attn_sink     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_indices  = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_length   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId out           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId max_logits    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId lse           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId lse_indexer   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId softmax_scale = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId indexer_topk  = pattern.addOperand<OperandKind::kTensor>();
            const FTrainSparseAttnFwdIndexerOpId sparse_attn_fwd_indexer =
                pattern.addOperation<OperationKind::kSparseAttnFwdIndexer>(q, kv, attn_sink, topk_indices, topk_length,
                                                                           out, max_logits, lse, lse_indexer,
                                                                           softmax_scale, indexer_topk);
            return Roles{pattern.buildPattern(),
                         q,
                         kv,
                         attn_sink,
                         topk_indices,
                         topk_length,
                         out,
                         max_logits,
                         lse,
                         lse_indexer,
                         softmax_scale,
                         indexer_topk,
                         sparse_attn_fwd_indexer};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<SparseAttnFwdIndexerProblem>>> makeRecords();

    static SparseAttnFwdIndexerProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return SparseAttnFwdIndexerProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.q))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.kv))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.attn_sink))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_indices))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_length))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.out))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.max_logits))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.lse))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.lse_indexer))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.softmax_scale))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.indexer_topk))),
            std::get<SparseAttnFwdIndexerAttributes>(
                *args.getOpArgument(OperationTraits<OperationKind::kSparseAttnFwdIndexer>::createPatternOperationId(
                    roles.sparse_attn_fwd_indexer)))};
    }
};

}  // namespace ftrain

#endif
