// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct SinkhornFwdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId logits{};
    FTrainTensorId output{};
    FTrainTensorId m_init{};
    FTrainTensorId num_iterations{};
    FTrainTensorId eps{};
    FTrainSinkhornFwdOpId sinkhorn_fwd{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const SinkhornFwdEntry& sinkhornFwdEntry() {
    static const SinkhornFwdEntry entry = [] {
        SinkhornFwdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.logits); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.output); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.m_init); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.num_iterations); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.eps); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddSinkhornFwd(pattern, &entry.sinkhorn_fwd, entry.logits, entry.output, entry.m_init,
                                                 entry.num_iterations, entry.eps);
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

extern "C" FTrainStatus ftrainSinkhornFwd(FTrainStorageView logits, FTrainStorageView output, FTrainStorageView m_init,
                                          FTrainStorageView num_iterations, FTrainStorageView eps,
                                          FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes,
                                          FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainSinkhornFwd"};
    const ftrain::SinkhornFwdEntry& entry = ftrain::sinkhornFwdEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.logits, logits); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.output, output); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.m_init, m_init); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.num_iterations, num_iterations); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.eps, eps); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetSinkhornFwd(args, entry.sinkhorn_fwd, compute_type); }
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
