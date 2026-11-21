// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_KDA_FAMILY_HPP_
#define FTRAIN_FAMILY_KDA_FAMILY_HPP_

#include <array>
#include <memory>
#include <vector>

#include "flash_train/family/kda/problem.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

namespace detail {
inline Tensor kdaPort(const Args& args, FTrainTensorId id) {
    return std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(id)));
}
}  // namespace detail

// Kimi Delta Attention forward: o = KDA(q, k, v, g, beta) plus the tensors its backward
// needs (see KdaForwardProblem for the port contract).
struct KdaForwardFamily {
    using Problem = KdaForwardProblem;

    struct Roles {
        Pattern pattern;
        std::array<FTrainTensorId, KdaForwardProblem::kCount> ports;
        FTrainKdaForwardOpId op;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            std::array<FTrainTensorId, KdaForwardProblem::kCount> ports{};
            for (FTrainTensorId& id : ports) { id = pattern.addOperand<OperandKind::kTensor>(); }
            using P                       = KdaForwardProblem;
            const FTrainKdaForwardOpId op = pattern.addOperation<OperationKind::kKdaForward>(
                ports[P::kQ], ports[P::kK], ports[P::kV], ports[P::kG], ports[P::kBeta], ports[P::kALog],
                ports[P::kDtBias], ports[P::kCuSeqlens], ports[P::kO], ports[P::kQRstd], ports[P::kKRstd],
                ports[P::kAqk], ports[P::kAkk], ports[P::kQn], ports[P::kKn], ports[P::kH], ports[P::kState]);
            return Roles{pattern.buildPattern(), ports, op};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<KdaForwardProblem>>> makeRecords();

    static KdaForwardProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        std::vector<Tensor> ports;
        ports.reserve(KdaForwardProblem::kCount);
        for (const FTrainTensorId id : roles.ports) { ports.push_back(detail::kdaPort(args, id)); }
        return KdaForwardProblem{std::move(ports),
                                 std::get<KdaAttributes>(*args.getOpArgument(
                                     OperationTraits<OperationKind::kKdaForward>::createPatternOperationId(roles.op)))};
    }
};

// Kimi Delta Attention backward: the gradients of q, k, v, g, beta (and a_log, dt_bias with
// gate_in_kernel) from d_o, the forward's inputs and its saved tensors.
struct KdaBackwardFamily {
    using Problem = KdaBackwardProblem;

    struct Roles {
        Pattern pattern;
        std::array<FTrainTensorId, KdaBackwardProblem::kCount> ports;
        FTrainKdaBackwardOpId op;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            std::array<FTrainTensorId, KdaBackwardProblem::kCount> ports{};
            for (FTrainTensorId& id : ports) { id = pattern.addOperand<OperandKind::kTensor>(); }
            using P                        = KdaBackwardProblem;
            const FTrainKdaBackwardOpId op = pattern.addOperation<OperationKind::kKdaBackward>(
                ports[P::kDO], ports[P::kQ], ports[P::kK], ports[P::kV], ports[P::kG], ports[P::kBeta], ports[P::kALog],
                ports[P::kDtBias], ports[P::kCuSeqlens], ports[P::kQRstd], ports[P::kKRstd], ports[P::kAqk],
                ports[P::kAkk], ports[P::kQn], ports[P::kKn], ports[P::kDq], ports[P::kDk], ports[P::kDv],
                ports[P::kDg], ports[P::kDBeta], ports[P::kDALog], ports[P::kDDtBias], ports[P::kH], ports[P::kState]);
            return Roles{pattern.buildPattern(), ports, op};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<KdaBackwardProblem>>> makeRecords();

    static KdaBackwardProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        std::vector<Tensor> ports;
        ports.reserve(KdaBackwardProblem::kCount);
        for (const FTrainTensorId id : roles.ports) { ports.push_back(detail::kdaPort(args, id)); }
        return KdaBackwardProblem{
            std::move(ports), std::get<KdaAttributes>(*args.getOpArgument(
                                  OperationTraits<OperationKind::kKdaBackward>::createPatternOperationId(roles.op)))};
    }
};

}  // namespace ftrain

#endif
