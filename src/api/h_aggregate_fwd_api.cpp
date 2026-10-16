// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct HAggregateFwdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId x{};
    FTrainTensorId h_pre{};
    FTrainTensorId aggregated{};
    FTrainHAggregateFwdOpId h_aggregate_fwd{};
    Result result{};
};

const HAggregateFwdEntry& hAggregateFwdEntry() {
    static const HAggregateFwdEntry entry = [] {
        HAggregateFwdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.x); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.h_pre); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.aggregated); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status =
                ftrainPatternAddHAggregateFwd(pattern, &entry.h_aggregate_fwd, entry.x, entry.h_pre, entry.aggregated);
        }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainOpsCreate(&entry.ops, pattern); }
        const Result failure = status != FTRAIN_STATUS_SUCCESS ? getLastResult() : Result{};
        if (pattern != nullptr) { ftrainPatternDestroy(pattern); }
        entry.result = failure;
        return entry;
    }();
    return entry;
}
}  // namespace
}  // namespace ftrain

extern "C" FTrainStatus ftrainHAggregateFwd(FTrainStorageView x, FTrainStorageView h_pre, FTrainStorageView aggregated,
                                            FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes,
                                            FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainHAggregateFwd"};
    const ftrain::HAggregateFwdEntry& entry = ftrain::hAggregateFwdEntry();
    // Restore the cached initialization failure on every call, including calls from
    // another thread or after an API call has overwritten this thread's last result.
    if (!entry.result.isSuccess()) {
        ftrain::setLastResult(entry.result.getStatus(), entry.result.getMessage());
        return entry.result.getStatus();
    }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.x, x); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.h_pre, h_pre); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.aggregated, aggregated); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetHAggregateFwd(args, entry.h_aggregate_fwd, compute_type);
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
