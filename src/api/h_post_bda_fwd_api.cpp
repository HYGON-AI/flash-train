// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct HPostBdaFwdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId h_res{};
    FTrainTensorId residual{};
    FTrainTensorId h_post{};
    FTrainTensorId x{};
    FTrainTensorId bias{};
    FTrainTensorId output{};
    FTrainHPostBdaFwdOpId h_post_bda_fwd{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const HPostBdaFwdEntry& hPostBdaFwdEntry() {
    static const HPostBdaFwdEntry entry = [] {
        HPostBdaFwdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.h_res); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.residual); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.h_post); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.x); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.bias); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.output); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddHPostBdaFwd(pattern, &entry.h_post_bda_fwd, entry.h_res, entry.residual,
                                                 entry.h_post, entry.x, entry.bias, entry.output);
        }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainOpsCreate(&entry.ops, pattern); }
        const Result failure = status != FTRAIN_STATUS_SUCCESS ? getLastResult() : Result{};
        if (pattern != nullptr) { ftrainPatternDestroy(pattern); }
        if (status != FTRAIN_STATUS_SUCCESS) { setLastResult(failure.getStatus(), failure.getMessage()); }
        entry.status = status;
        return entry;
    }();
    return entry;
}
}  // namespace
}  // namespace ftrain

extern "C" FTrainStatus ftrainHPostBdaFwd(FTrainStorageView h_res, FTrainStorageView residual, FTrainStorageView h_post,
                                          FTrainStorageView x, FTrainStorageView bias, FTrainStorageView output,
                                          FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes,
                                          FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainHPostBdaFwd"};
    const ftrain::HPostBdaFwdEntry& entry = ftrain::hPostBdaFwdEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.h_res, h_res); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.residual, residual); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.h_post, h_post); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.x, x); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.bias, bias); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.output, output); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetHPostBdaFwd(args, entry.h_post_bda_fwd, compute_type);
    }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPlanCreate(&plan, args); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainPlanExecutePrimitive(plan, 0, workspace, workspace_bytes, stream);
    }
    const ftrain::Result failure = status != FTRAIN_STATUS_SUCCESS ? ftrain::getLastResult() : ftrain::Result{};
    if (plan != nullptr) { ftrainPlanDestroy(plan); }
    if (args != nullptr) { ftrainArgsDestroy(args); }
    if (status != FTRAIN_STATUS_SUCCESS) { ftrain::setLastResult(failure.getStatus(), failure.getMessage()); }
    return status;
}
