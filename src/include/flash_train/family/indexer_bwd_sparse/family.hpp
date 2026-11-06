// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_BWD_SPARSE_FAMILY_HPP_
#define FTRAIN_FAMILY_INDEXER_BWD_SPARSE_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/indexer_bwd_sparse/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the IndexerBwdSparse family contributes to its engine.
struct IndexerBwdSparseFamily {
    using Problem = IndexerBwdSparseProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId index_q;
        FTrainTensorId weights;
        FTrainTensorId index_k;
        FTrainTensorId attn_score;
        FTrainTensorId index_score;
        FTrainTensorId topk_indices;
        FTrainTensorId grad_attn_score;
        FTrainTensorId dq;
        FTrainTensorId dw;
        FTrainTensorId dk;
        FTrainTensorId sm_scale;
        FTrainTensorId loss_coeff;
        FTrainTensorId grad_loss;
        FTrainTensorId topk_indices_global;
        FTrainIndexerBwdSparseOpId indexer_bwd_sparse;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId index_q             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId weights             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId index_k             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId attn_score          = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId index_score         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_indices        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_attn_score     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId dq                  = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId dw                  = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId dk                  = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId sm_scale            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId loss_coeff          = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_loss           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_indices_global = pattern.addOperand<OperandKind::kTensor>();
            const FTrainIndexerBwdSparseOpId indexer_bwd_sparse =
                pattern.addOperation<OperationKind::kIndexerBwdSparse>(
                    index_q, weights, index_k, attn_score, index_score, topk_indices, grad_attn_score, dq, dw, dk,
                    sm_scale, loss_coeff, grad_loss, topk_indices_global);
            return Roles{pattern.buildPattern(),
                         index_q,
                         weights,
                         index_k,
                         attn_score,
                         index_score,
                         topk_indices,
                         grad_attn_score,
                         dq,
                         dw,
                         dk,
                         sm_scale,
                         loss_coeff,
                         grad_loss,
                         topk_indices_global,
                         indexer_bwd_sparse};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<IndexerBwdSparseProblem>>> makeRecords();

    static IndexerBwdSparseProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return IndexerBwdSparseProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.index_q))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.weights))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.index_k))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.attn_score))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.index_score))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_indices))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_attn_score))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.dq))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.dw))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.dk))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.sm_scale))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.loss_coeff))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_loss))),
            std::get<Tensor>(*args.getOperand(
                OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_indices_global))),
            std::get<IndexerBwdSparseAttributes>(
                *args.getOpArgument(OperationTraits<OperationKind::kIndexerBwdSparse>::createPatternOperationId(
                    roles.indexer_bwd_sparse)))};
    }
};

}  // namespace ftrain

#endif
