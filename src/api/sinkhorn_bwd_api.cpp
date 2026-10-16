// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct SinkhornBwdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId grad_output{};
    FTrainTensorId m_init{};
    FTrainTensorId grad_logits{};
    FTrainTensorId num_iterations{};
    FTrainTensorId eps{};
    FTrainSinkhornBwdOpId sinkhorn_bwd{};
    Result result{};
};

const SinkhornBwdEntry& sinkhornBwdEntry() {
    static const SinkhornBwdEntry entry = [] {
        SinkhornBwdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_output); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.m_init); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_logits); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.num_iterations); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.eps); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddSinkhornBwd(pattern, &entry.sinkhorn_bwd, entry.grad_output, entry.m_init,
                                                 entry.grad_logits, entry.num_iterations, entry.eps);
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

extern "C" FTrainStatus ftrainSinkhornBwd(FTrainStorageView grad_output, FTrainStorageView m_init,
                                          FTrainStorageView grad_logits, FTrainStorageView num_iterations,
                                          FTrainStorageView eps, FTrainNumericType compute_type, void* workspace,
                                          uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainSinkhornBwd"};
    const ftrain::SinkhornBwdEntry& entry = ftrain::sinkhornBwdEntry();
    if (!entry.result.isSuccess()) {
        ftrain::setLastResult(entry.result.getStatus(), entry.result.getMessage());
        return entry.result.getStatus();
    }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_output, grad_output); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.m_init, m_init); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_logits, grad_logits); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.num_iterations, num_iterations); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.eps, eps); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetSinkhornBwd(args, entry.sinkhorn_bwd, compute_type); }
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
