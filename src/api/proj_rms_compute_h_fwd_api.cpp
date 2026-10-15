// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct ProjRmsComputeHFwdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId x{};
    FTrainTensorId weight{};
    FTrainTensorId alpha_pre{};
    FTrainTensorId alpha_post{};
    FTrainTensorId alpha_res{};
    FTrainTensorId bias{};
    FTrainTensorId h_pre{};
    FTrainTensorId h_post{};
    FTrainTensorId h_res{};
    FTrainTensorId r{};
    FTrainTensorId proj{};
    FTrainTensorId n{};
    FTrainTensorId eps{};
    FTrainTensorId compute_h_eps{};
    FTrainProjRmsComputeHFwdOpId proj_rms_compute_h_fwd{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const ProjRmsComputeHFwdEntry& projRmsComputeHFwdEntry() {
    static const ProjRmsComputeHFwdEntry entry = [] {
        ProjRmsComputeHFwdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.x); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.weight); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.alpha_pre); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.alpha_post); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.alpha_res); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.bias); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.h_pre); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.h_post); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.h_res); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.r); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.proj); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.n); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.eps); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.compute_h_eps); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddProjRmsComputeHFwd(pattern, &entry.proj_rms_compute_h_fwd, entry.x, entry.weight,
                                                        entry.alpha_pre, entry.alpha_post, entry.alpha_res, entry.bias,
                                                        entry.h_pre, entry.h_post, entry.h_res, entry.r, entry.proj,
                                                        entry.n, entry.eps, entry.compute_h_eps);
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

extern "C" FTrainStatus ftrainProjRmsComputeHFwd(FTrainStorageView x, FTrainStorageView weight,
                                                 FTrainStorageView alpha_pre, FTrainStorageView alpha_post,
                                                 FTrainStorageView alpha_res, FTrainStorageView bias,
                                                 FTrainStorageView h_pre, FTrainStorageView h_post,
                                                 FTrainStorageView h_res, FTrainStorageView r, FTrainStorageView proj,
                                                 FTrainStorageView n, FTrainStorageView eps,
                                                 FTrainStorageView compute_h_eps, FTrainNumericType compute_type,
                                                 void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainProjRmsComputeHFwd"};
    const ftrain::ProjRmsComputeHFwdEntry& entry = ftrain::projRmsComputeHFwdEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.x, x); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.weight, weight); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.alpha_pre, alpha_pre); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.alpha_post, alpha_post); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.alpha_res, alpha_res); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.bias, bias); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.h_pre, h_pre); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.h_post, h_post); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.h_res, h_res); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.r, r); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.proj, proj); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.n, n); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.eps, eps); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.compute_h_eps, compute_h_eps); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetProjRmsComputeHFwd(args, entry.proj_rms_compute_h_fwd, compute_type);
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
