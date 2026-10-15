// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_BWD_DENSE_FAMILY_HPP_
#define FTRAIN_FAMILY_INDEXER_BWD_DENSE_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/indexer_bwd_dense/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the IndexerBwdDense family contributes to its engine.
struct IndexerBwdDenseFamily {
    using Problem = IndexerBwdDenseProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId index_q;
        FTrainTensorId weights;
        FTrainTensorId index_k;
        FTrainTensorId attn_score;
        FTrainTensorId attn_l1norm;
        FTrainTensorId index_score;
        FTrainTensorId index_lse;
        FTrainTensorId grad_index_score;
        FTrainTensorId dq;
        FTrainTensorId dw;
        FTrainTensorId dk;
        FTrainTensorId sm_scale;
        FTrainTensorId loss_coeff;
        FTrainTensorId grad_loss;
        FTrainTensorId ratio;
        FTrainIndexerBwdDenseOpId indexer_bwd_dense;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId index_q                      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId weights                      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId index_k                      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId attn_score                   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId attn_l1norm                  = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId index_score                  = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId index_lse                    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_index_score             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId dq                           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId dw                           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId dk                           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId sm_scale                     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId loss_coeff                   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_loss                    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId ratio                        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainIndexerBwdDenseOpId indexer_bwd_dense = pattern.addOperation<OperationKind::kIndexerBwdDense>(
                index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse, grad_index_score, dq, dw,
                dk, sm_scale, loss_coeff, grad_loss, ratio);
            return Roles{
                pattern.buildPattern(), index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse,
                grad_index_score,       dq,      dw,      dk,      sm_scale,   loss_coeff,  grad_loss,   ratio,
                indexer_bwd_dense};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<IndexerBwdDenseProblem>>> makeRecords();

    static IndexerBwdDenseProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return IndexerBwdDenseProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.index_q))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.weights))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.index_k))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.attn_score))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.attn_l1norm))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.index_score))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.index_lse))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_index_score))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.dq))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.dw))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.dk))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.sm_scale))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.loss_coeff))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_loss))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.ratio))),
            std::get<IndexerBwdDenseAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kIndexerBwdDense>::createPatternOperationId(roles.indexer_bwd_dense)))};
    }
};

}  // namespace ftrain

#endif
