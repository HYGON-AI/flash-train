// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_GDN_FAMILY_HPP_
#define FTRAIN_FAMILY_GDN_FAMILY_HPP_

#include <array>
#include <memory>
#include <vector>

#include "flash_train/family/gdn/problem.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<OperationKind Kind>
struct GdnFamilyBase {
    using Problem = GdnProblem<Kind>;
    using Traits = OperationTraits<Kind>;
    static constexpr std::size_t kPorts =
        Kind == OperationKind::kGdnForward ? 8 : Kind == OperationKind::kGdnBackward ? 14 :
        Kind == OperationKind::kGdn2Forward ? 7 : 13;

    struct Roles {
        Pattern pattern;
        std::array<FTrainTensorId, kPorts> tensors;
        typename Traits::Id op;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder builder;
            std::array<FTrainTensorId, kPorts> ids{};
            for (auto& id : ids) id = builder.addOperand<OperandKind::kTensor>();
            typename Traits::Id op{};
            if constexpr (Kind == OperationKind::kGdnForward) {
                op = builder.addOperation<Kind>(ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7]);
            } else if constexpr (Kind == OperationKind::kGdnBackward) {
                op = builder.addOperation<Kind>(ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
                                                ids[8], ids[9], ids[10], ids[11], ids[12], ids[13]);
            } else if constexpr (Kind == OperationKind::kGdn2Forward) {
                op = builder.addOperation<Kind>(ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6]);
            } else {
                op = builder.addOperation<Kind>(ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6],
                                                ids[7], ids[8], ids[9], ids[10], ids[11], ids[12]);
            }
            return Roles{builder.buildPattern(), ids, op};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }
    static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords() { return {}; }

    static Problem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        std::vector<Tensor> ports;
        ports.reserve(kPorts);
        for (const FTrainTensorId id : roles.tensors) {
            ports.push_back(std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(id))));
        }
        return Problem{std::move(ports), std::get<typename Traits::Type>(
            *args.getOpArgument(Traits::createPatternOperationId(roles.op)))};
    }
};

struct GdnForwardFamily : GdnFamilyBase<OperationKind::kGdnForward> {
    static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords();
};
struct GdnBackwardFamily : GdnFamilyBase<OperationKind::kGdnBackward> {
    static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords();
};
struct Gdn2ForwardFamily : GdnFamilyBase<OperationKind::kGdn2Forward> {
    static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords();
};
struct Gdn2BackwardFamily : GdnFamilyBase<OperationKind::kGdn2Backward> {
    static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords();
};

}  // namespace ftrain

#endif
