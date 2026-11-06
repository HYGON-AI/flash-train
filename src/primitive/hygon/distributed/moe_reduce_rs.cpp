// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <flash_train/primitive/hygon/distributed/moe_reduce_rs.hpp>

#include <flash_train/error.hpp>
#include <flash_train/session/distributed.hpp>
#include <flash_train/constraints.hpp>

#include <utility>

namespace ftrain {
namespace {

std::string primitiveName(FTrainMoERSRoute route) {
    switch (route) {
        case FTRAIN_MOERS_ROUTE_A2A: return "HygonMoEReduceRSA2a";
        case FTRAIN_MOERS_ROUTE_A2A_RAGGED_SIGNAL: return "HygonMoEReduceRSRaggedSignal";
        default: return "HygonMoEReduceRSInvalid";
    }
}

}  // namespace

HygonMoEReduceRSPrimitive::HygonMoEReduceRSPrimitive(FTrainMoERSRoute route)
    : route_(route), name_(primitiveName(route)) {}

Result HygonMoEReduceRSPrimitive::isApplicable(const MoEReduceRSProblem& problem,
                                              const Constraints&) const {
    const auto& attrs = problem.attributes();
    if (attrs.getRoute() != route_) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MoEReduceRS session route does not match this primitive");
    }

    const auto session = DistributedSessionRegistry::get().getMoEReduceRS(attrs.getSessionId());
    if (session == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MoEReduceRS session was not found");
    }
    const auto& info = session->info();
    if (info.route != route_ || info.gemm_variant != attrs.getGemmVariant()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MoEReduceRS session route or variant does not match");
    }
    if (info.input_type != FTRAIN_NUMERIC_TYPE_FP16 && info.input_type != FTRAIN_NUMERIC_TYPE_BF16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MoEReduceRS supports only FP16 and BF16 activations");
    }
    if (info.world_size < 2 || info.nnodes != 1) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MoEReduceRS requires single-node multi-rank ETP");
    }

    const auto& input = problem.getInput().getStorageView();
    const auto& weight = problem.getWeight().getStorageView();
    const auto& output = problem.getOutput().getStorageView();
    if (input.getNumericType() != info.input_type || weight.getNumericType() != info.input_type ||
        output.getNumericType() != info.input_type) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MoEReduceRS tensor dtype does not match the session");
    }
    const auto input_k = input.getDims().at(1);
    const auto output_n = output.getDims().at(1);
    const auto weight_first = weight.getDims().at(1);
    const auto weight_second = weight.getDims().at(2);
    if (input.getDims().at(0) > info.max_ntokens * info.topk || input_k != info.k_dim ||
        output_n != info.n_dim) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MoEReduceRS shape does not match the session");
    }
    if (info.gemm_variant == FTRAIN_MOERS_GEMM_NK_DIRECT) {
        if (input_k != weight_second || output_n != weight_first) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "MoEReduceRS NK weight layout does not match the session");
        }
    } else if (info.gemm_variant == FTRAIN_MOERS_GEMM_KN_DIRECT) {
        if (input_k != weight_first || output_n != weight_second) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "MoEReduceRS KN weight layout does not match the session");
        }
    } else {
        const bool matches_nk = input_k == weight_second && output_n == weight_first;
        const bool matches_kn = input_k == weight_first && output_n == weight_second;
        if (!matches_nk && !matches_kn) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "MoEReduceRS layout-auto weight does not match session");
        }
    }
    return Result{};
}

void HygonMoEReduceRSPrimitive::configure(const MoEReduceRSProblem& problem) {
    session_ = DistributedSessionRegistry::get().getMoEReduceRS(problem.attributes().getSessionId());
    if (session_ == nullptr) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS session was not found");
    }
    const auto applicable = isApplicable(problem, Constraints{getCurrentDeviceId()});
    if (!applicable.isSuccess()) { throw Exception(applicable.getStatus(), "%s", applicable.getMessage()); }
    problem_.emplace(problem);
}

void HygonMoEReduceRSPrimitive::executeImpl(const Resources& resources) {
    if (session_ == nullptr || !problem_) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "MoEReduceRS primitive is not configured");
    }
    session_->execute(*problem_, resources.getStream());
}

}  // namespace ftrain
