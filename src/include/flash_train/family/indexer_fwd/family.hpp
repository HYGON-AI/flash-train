// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_FWD_FAMILY_HPP_
#define FTRAIN_FAMILY_INDEXER_FWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/indexer_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the IndexerFwd family contributes to its engine; OpsEngine
// consumes exactly these static functions (the OpsEngine comment carries the
// contract).
struct IndexerFwdFamily {
    using Problem = IndexerFwdProblem;

    // The supported Pattern together with the typed role IDs its builder
    // produced, built once and shared by every engine of this family.
    struct Roles {
        Pattern pattern;
        FTrainTensorId q;
        FTrainTensorId k;
        FTrainTensorId w;
        FTrainTensorId scores;
        FTrainTensorId ratio;
        FTrainTensorId sm_scale;
        FTrainIndexerFwdOpId indexer_fwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId q        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId k        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId w        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId scores   = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId ratio    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId sm_scale = pattern.addOperand<OperandKind::kTensor>();
            const FTrainIndexerFwdOpId indexer_fwd =
                pattern.addOperation<OperationKind::kIndexerFwd>(q, k, w, scores, ratio, sm_scale);
            return Roles{pattern.buildPattern(), q, k, w, scores, ratio, sm_scale, indexer_fwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    // The implementation catalog, defined in family.cpp so neither the
    // Primitive list nor their includes grow this header.
    static std::vector<std::shared_ptr<const Primitive<IndexerFwdProblem>>> makeRecords();

    static IndexerFwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return IndexerFwdProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.q))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.k))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.w))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.scores))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.ratio))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.sm_scale))),
            std::get<IndexerFwdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kIndexerFwd>::createPatternOperationId(roles.indexer_fwd)))};
    }
};

}  // namespace ftrain

#endif
