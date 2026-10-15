// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct ProjRmsComputeHBwdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId grad_h_pre{};
    FTrainTensorId grad_h_post{};
    FTrainTensorId grad_h_res{};
    FTrainTensorId grad_r{};
    FTrainTensorId h_pre{};
    FTrainTensorId h_post{};
    FTrainTensorId proj{};
    FTrainTensorId r_ch{};
    FTrainTensorId x{};
    FTrainTensorId weight{};
    FTrainTensorId alpha_pre{};
    FTrainTensorId alpha_post{};
    FTrainTensorId alpha_res{};
    FTrainTensorId grad_x{};
    FTrainTensorId grad_weight{};
    FTrainTensorId grad_alpha{};
    FTrainTensorId grad_bias{};
    FTrainTensorId n{};
    FTrainTensorId eps{};
    FTrainTensorId compute_h_eps{};
    FTrainProjRmsComputeHBwdOpId proj_rms_compute_h_bwd{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const ProjRmsComputeHBwdEntry& projRmsComputeHBwdEntry() {
    static const ProjRmsComputeHBwdEntry entry = [] {
        ProjRmsComputeHBwdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_h_pre); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_h_post); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_h_res); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_r); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.h_pre); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.h_post); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.proj); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.r_ch); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.x); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.weight); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.alpha_pre); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.alpha_post); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.alpha_res); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_x); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_weight); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_alpha); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_bias); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.n); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.eps); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.compute_h_eps); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddProjRmsComputeHBwd(
                pattern, &entry.proj_rms_compute_h_bwd, entry.grad_h_pre, entry.grad_h_post, entry.grad_h_res,
                entry.grad_r, entry.h_pre, entry.h_post, entry.proj, entry.r_ch, entry.x, entry.weight, entry.alpha_pre,
                entry.alpha_post, entry.alpha_res, entry.grad_x, entry.grad_weight, entry.grad_alpha, entry.grad_bias,
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

extern "C" FTrainStatus ftrainProjRmsComputeHBwd(
    FTrainStorageView grad_h_pre, FTrainStorageView grad_h_post, FTrainStorageView grad_h_res, FTrainStorageView grad_r,
    FTrainStorageView h_pre, FTrainStorageView h_post, FTrainStorageView proj, FTrainStorageView r_ch,
    FTrainStorageView x, FTrainStorageView weight, FTrainStorageView alpha_pre, FTrainStorageView alpha_post,
    FTrainStorageView alpha_res, FTrainStorageView grad_x, FTrainStorageView grad_weight, FTrainStorageView grad_alpha,
    FTrainStorageView grad_bias, FTrainStorageView n, FTrainStorageView eps, FTrainStorageView compute_h_eps,
    FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainProjRmsComputeHBwd"};
    const ftrain::ProjRmsComputeHBwdEntry& entry = ftrain::projRmsComputeHBwdEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_h_pre, grad_h_pre); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_h_post, grad_h_post); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_h_res, grad_h_res); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_r, grad_r); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.h_pre, h_pre); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.h_post, h_post); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.proj, proj); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.r_ch, r_ch); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.x, x); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.weight, weight); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.alpha_pre, alpha_pre); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.alpha_post, alpha_post); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.alpha_res, alpha_res); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_x, grad_x); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_weight, grad_weight); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_alpha, grad_alpha); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_bias, grad_bias); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.n, n); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.eps, eps); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.compute_h_eps, compute_h_eps); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetProjRmsComputeHBwd(args, entry.proj_rms_compute_h_bwd, compute_type);
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
