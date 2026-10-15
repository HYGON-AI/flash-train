// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_COMPACTIFY_FAMILY_HPP_
#define FTRAIN_FAMILY_COMPACTIFY_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/compactify/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the Compactify family contributes to its engine.
struct CompactifyFamily {
    using Problem = CompactifyProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId indices;
        FTrainTensorId output;
        FTrainTensorId topk_length;
        FTrainCompactifyOpId compactify;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId indices     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId output      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_length = pattern.addOperand<OperandKind::kTensor>();
            const FTrainCompactifyOpId compactify =
                pattern.addOperation<OperationKind::kCompactify>(indices, output, topk_length);
            return Roles{pattern.buildPattern(), indices, output, topk_length, compactify};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<CompactifyProblem>>> makeRecords();

    static CompactifyProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return CompactifyProblem{
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.indices))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.output))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_length))),
            std::get<CompactifyAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kCompactify>::createPatternOperationId(roles.compactify)))};
    }
};

}  // namespace ftrain

#endif
