// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_INDEXER_PREDICT_FAMILY_HPP_
#define FTRAIN_FAMILY_SPARSE_INDEXER_PREDICT_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/sparse_indexer_predict/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the SparseIndexerPredict family contributes to its engine.
struct SparseIndexerPredictFamily {
    using Problem = SparseIndexerPredictProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId q;
        FTrainTensorId k;
        FTrainTensorId w;
        FTrainTensorId topk_indices;
        FTrainTensorId predict;
        FTrainTensorId sm_scale;
        FTrainTensorId topk_indices_global;
        FTrainSparseIndexerPredictOpId sparse_indexer_predict;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId q                   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId k                   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId w                   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_indices        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId predict             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId sm_scale            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_indices_global = pattern.addOperand<OperandKind::kTensor>();
            const FTrainSparseIndexerPredictOpId sparse_indexer_predict =
                pattern.addOperation<OperationKind::kSparseIndexerPredict>(q, k, w, topk_indices, predict, sm_scale,
                                                                           topk_indices_global);
            return Roles{pattern.buildPattern(), q, k, w, topk_indices, predict, sm_scale, topk_indices_global,
                         sparse_indexer_predict};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<SparseIndexerPredictProblem>>> makeRecords();

    static SparseIndexerPredictProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return SparseIndexerPredictProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.q))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.k))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.w))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_indices))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.predict))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.sm_scale))),
            std::get<Tensor>(*args.getOperand(
                OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_indices_global))),
            std::get<SparseIndexerPredictAttributes>(
                *args.getOpArgument(OperationTraits<OperationKind::kSparseIndexerPredict>::createPatternOperationId(
                    roles.sparse_indexer_predict)))};
    }
};

}  // namespace ftrain

#endif
