// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>

#include "flash_train/flash_train.h"

#include "flash_train/operation/gdn.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/api.hpp"

namespace ftrain {
namespace {

template<OperationKind Kind>
struct GdnEntry {
    static constexpr std::size_t kPorts = Kind == OperationKind::kGdnForward ? 8
                                          : Kind == OperationKind::kGdnBackward ? 14
                                          : Kind == OperationKind::kGdn2Forward ? 7 : 13;
    FTrainOps ops{nullptr};
    std::array<FTrainTensorId, kPorts> ports{};
    typename OperationTraits<Kind>::Id op{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

template<OperationKind Kind>
const GdnEntry<Kind>& entry() {
    static const GdnEntry<Kind> value = [] {
        GdnEntry<Kind> result;
        FTrainPattern pattern = nullptr;
        FTrainStatus status = ftrainPatternCreate(&pattern);
        for (auto& port : result.ports) {
            if (status == FTRAIN_STATUS_SUCCESS) status = ftrainPatternAddTensor(pattern, &port);
        }
        if (status == FTRAIN_STATUS_SUCCESS) {
            const auto& p = result.ports;
            if constexpr (Kind == OperationKind::kGdnForward) {
                status = ftrainPatternAddGdnForward(pattern, &result.op, p[0], p[1], p[2], p[3], p[4], p[5], p[6],
                                                   p[7]);
            } else if constexpr (Kind == OperationKind::kGdnBackward) {
                status = ftrainPatternAddGdnBackward(pattern, &result.op, p[0], p[1], p[2], p[3], p[4], p[5], p[6],
                                                    p[7], p[8], p[9], p[10], p[11], p[12], p[13]);
            } else if constexpr (Kind == OperationKind::kGdn2Forward) {
                status = ftrainPatternAddGdn2Forward(pattern, &result.op, p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
            } else {
                status = ftrainPatternAddGdn2Backward(pattern, &result.op, p[0], p[1], p[2], p[3], p[4], p[5],
                                                     p[6], p[7], p[8], p[9], p[10], p[11], p[12]);
            }
        }
        if (status == FTRAIN_STATUS_SUCCESS) status = ftrainOpsCreate(&result.ops, pattern);
        if (pattern != nullptr) ftrainPatternDestroy(pattern);
        result.status = status;
        return result;
    }();
    return value;
}

template<OperationKind Kind>
FTrainStatus run(const std::array<FTrainStorageView, GdnEntry<Kind>::kPorts>& views, float scale,
                 void* workspace, std::uint64_t workspace_bytes, FTrainStream stream) {
    const auto& role = entry<Kind>();
    if (role.status != FTRAIN_STATUS_SUCCESS) return role.status;

    FTrainArgs args = nullptr;
    FTrainPlan plan = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, role.ops);
    for (std::size_t i = 0; i < role.kPorts && status == FTRAIN_STATUS_SUCCESS; ++i) {
        status = ftrainArgsSetTensor(args, role.ports[i], views[i]);
    }
    if (status == FTRAIN_STATUS_SUCCESS) {
        if constexpr (Kind == OperationKind::kGdnForward) status = ftrainArgsSetGdnForward(args, role.op, scale);
        else if constexpr (Kind == OperationKind::kGdnBackward) status = ftrainArgsSetGdnBackward(args, role.op, scale);
        else if constexpr (Kind == OperationKind::kGdn2Forward) status = ftrainArgsSetGdn2Forward(args, role.op, scale);
        else status = ftrainArgsSetGdn2Backward(args, role.op, scale);
    }
    if (status == FTRAIN_STATUS_SUCCESS) status = ftrainPlanCreate(&plan, args);
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainPlanExecutePrimitive(plan, 0, workspace, workspace_bytes, stream);
    }

    const Result failure = status != FTRAIN_STATUS_SUCCESS ? getLastResult() : Result{};
    if (plan != nullptr) ftrainPlanDestroy(plan);
    if (args != nullptr) ftrainArgsDestroy(args);
    if (status != FTRAIN_STATUS_SUCCESS) setLastResult(failure.getStatus(), failure.getMessage());
    return status;
}

}  // namespace
}  // namespace ftrain

extern "C" FTrainStatus ftrainGdnForward(FTrainStorageView q, FTrainStorageView k, FTrainStorageView v,
                                        FTrainStorageView g, FTrainStorageView beta, FTrainStorageView h0,
                                        FTrainStorageView out, FTrainStorageView ht, float scale, void* workspace,
                                        uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdnForward"};
    return ftrain::run<ftrain::OperationKind::kGdnForward>({q, k, v, g, beta, h0, out, ht}, scale, workspace,
                                                          workspace_bytes, stream);
}

extern "C" FTrainStatus ftrainGdnBackward(FTrainStorageView q, FTrainStorageView k, FTrainStorageView v,
                                         FTrainStorageView g, FTrainStorageView beta, FTrainStorageView do_,
                                         FTrainStorageView h0, FTrainStorageView dht, FTrainStorageView dq,
                                         FTrainStorageView dk, FTrainStorageView dv, FTrainStorageView dg,
                                         FTrainStorageView dbeta, FTrainStorageView dh0, float scale, void* workspace,
                                         uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdnBackward"};
    return ftrain::run<ftrain::OperationKind::kGdnBackward>(
        {q, k, v, g, beta, do_, h0, dht, dq, dk, dv, dg, dbeta, dh0}, scale, workspace, workspace_bytes, stream);
}

extern "C" FTrainStatus ftrainGdn2Forward(FTrainStorageView q, FTrainStorageView k, FTrainStorageView v,
                                         FTrainStorageView g, FTrainStorageView b, FTrainStorageView w,
                                         FTrainStorageView out, float scale, void* workspace,
                                         uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdn2Forward"};
    return ftrain::run<ftrain::OperationKind::kGdn2Forward>({q, k, v, g, b, w, out}, scale, workspace,
                                                           workspace_bytes, stream);
}

extern "C" FTrainStatus ftrainGdn2Backward(FTrainStorageView q, FTrainStorageView k, FTrainStorageView v,
                                          FTrainStorageView g, FTrainStorageView b, FTrainStorageView w,
                                          FTrainStorageView do_, FTrainStorageView dq, FTrainStorageView dk,
                                          FTrainStorageView dv, FTrainStorageView dg, FTrainStorageView db,
                                          FTrainStorageView dw, float scale, void* workspace,
                                          uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainGdn2Backward"};
    return ftrain::run<ftrain::OperationKind::kGdn2Backward>(
        {q, k, v, g, b, w, do_, dq, dk, dv, dg, db, dw}, scale, workspace, workspace_bytes, stream);
}
