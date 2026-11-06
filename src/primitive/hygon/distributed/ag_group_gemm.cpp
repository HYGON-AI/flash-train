// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <flash_train/primitive/hygon/distributed/ag_group_gemm.hpp>

#include <flash_train/error.hpp>
#include <flash_train/session/distributed.hpp>
#include <flash_train/constraints.hpp>

#include <cctype>
#include <utility>

namespace ftrain {
namespace {

std::string primitiveName(FTrainAGRoute route) {
    switch (route) {
        case FTRAIN_AG_ROUTE_NATIVE_NK_A2A: return "HygonAGGroupGemmNativeNkA2a";
        case FTRAIN_AG_ROUTE_NATIVE_KN_A2A: return "HygonAGGroupGemmNativeKnA2a";
        case FTRAIN_AG_ROUTE_NATIVE_KN_RING_FIXED: return "HygonAGGroupGemmNativeKnRingFixed";
        case FTRAIN_AG_ROUTE_NATIVE_KN_RING: return "HygonAGGroupGemmNativeKnRing";
        case FTRAIN_AG_ROUTE_NATIVE_KN_RING_READY_MASK: return "HygonAGGroupGemmNativeKnRingReadyMask";
        default: return "HygonAGGroupGemmInvalid";
    }
}

bool supportedNumericType(FTrainNumericType type) {
    return type == FTRAIN_NUMERIC_TYPE_FP16 || type == FTRAIN_NUMERIC_TYPE_BF16;
}

}  // namespace

HygonAGGroupGemmPrimitive::HygonAGGroupGemmPrimitive(FTrainAGRoute route) : route_(route), name_(primitiveName(route)) {}

Result HygonAGGroupGemmPrimitive::isApplicable(const AGGroupGemmProblem& problem,
                                              const Constraints&) const {
    const auto& attrs = problem.attributes();
    if (attrs.getRoute() != route_) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "AGGroupGemm session route does not match this primitive");
    }

    const auto session = DistributedSessionRegistry::get().getAGGroupGemm(attrs.getSessionId());
    if (session == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "AGGroupGemm session was not found");
    }
    const auto& info = session->info();
    if (info.route != route_) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "AGGroupGemm session route does not match this primitive");
    }
    if (!supportedNumericType(info.input_type) || !supportedNumericType(info.output_type)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "AGGroupGemm supports only FP16 and BF16 activations");
    }
    if (info.world_size < 2) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "AGGroupGemm requires at least two ETP ranks");
    }

    const auto& input = problem.getInput().getStorageView();
    const auto& weight = problem.getWeight().getStorageView();
    const auto& topk_ids = problem.getTopkIds().getStorageView();
    const auto& output = problem.getOutput().getStorageView();
    if (input.getNumericType() != info.input_type || weight.getNumericType() != info.input_type ||
        output.getNumericType() != info.output_type) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "AGGroupGemm tensor dtype does not match the session");
    }
    if (input.getDims().at(1) != info.k_dim || weight.getDims().at(1) != info.k_dim ||
        weight.getDims().at(2) != info.n_dim || output.getDims().at(1) != info.n_dim) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "AGGroupGemm shape does not match the session");
    }
    if (route_ == FTRAIN_AG_ROUTE_NATIVE_NK_A2A && topk_ids.getDims().at(0) != info.full_m) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "AGGroupGemm routing rows do not match session full_m");
    }
    if (route_ != FTRAIN_AG_ROUTE_NATIVE_NK_A2A && topk_ids.getDims().at(0) == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "AGGroupGemm expanded routing is empty");
    }
    if (route_ == FTRAIN_AG_ROUTE_NATIVE_NK_A2A && topk_ids.getDims().at(1) != info.topk) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "native-NK A2A requires compact session routing");
    }
    if (route_ != FTRAIN_AG_ROUTE_NATIVE_NK_A2A &&
        (info.topk != 1 || topk_ids.getDims().at(1) != 1)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "native-KN routes require operator topk equal to one");
    }
    std::uint64_t expected_output_rows = 0;
    if (route_ == FTRAIN_AG_ROUTE_NATIVE_NK_A2A) {
        expected_output_rows = info.full_m * topk_ids.getDims().at(1);
    } else {
        // Native-KN routes consume globally gathered, expert-sorted rows.  The
        // optional rank_m_sizes describe each rank's contribution to full_m;
        // unlike the inputs, the output is not a rank-local buffer.
        expected_output_rows = info.full_m;
    }
    if (output.getDims().at(0) != expected_output_rows) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "AGGroupGemm output rows do not match global expanded rows");
    }
    return Result{};
}

void HygonAGGroupGemmPrimitive::configure(const AGGroupGemmProblem& problem) {
    session_ = DistributedSessionRegistry::get().getAGGroupGemm(problem.attributes().getSessionId());
    if (session_ == nullptr) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm session was not found");
    }
    const auto applicable = isApplicable(problem, Constraints{getCurrentDeviceId()});
    if (!applicable.isSuccess()) { throw Exception(applicable.getStatus(), "%s", applicable.getMessage()); }
    problem_.emplace(problem);
}

void HygonAGGroupGemmPrimitive::executeImpl(const Resources& resources) {
    if (session_ == nullptr || !problem_) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "AGGroupGemm primitive is not configured");
    }
    session_->execute(*problem_, resources.getStream());
}

}  // namespace ftrain
