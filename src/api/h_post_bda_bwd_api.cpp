// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct HPostBdaBwdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId grad_output{};
    FTrainTensorId residual{};
    FTrainTensorId x{};
    FTrainTensorId bias{};
    FTrainTensorId h_res{};
    FTrainTensorId h_post{};
    FTrainTensorId grad_residual{};
    FTrainTensorId grad_x{};
    FTrainTensorId grad_h_res{};
    FTrainTensorId grad_h_post{};
    FTrainTensorId grad_bias{};
    FTrainHPostBdaBwdOpId h_post_bda_bwd{};
    Result result{};
};

const HPostBdaBwdEntry& hPostBdaBwdEntry() {
    static const HPostBdaBwdEntry entry = [] {
        HPostBdaBwdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_output); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.residual); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.x); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.bias); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.h_res); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.h_post); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_residual); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_x); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_h_res); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_h_post); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_bias); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddHPostBdaBwd(pattern, &entry.h_post_bda_bwd, entry.grad_output, entry.residual,
                                                 entry.x, entry.bias, entry.h_res, entry.h_post, entry.grad_residual,
                                                 entry.grad_x, entry.grad_h_res, entry.grad_h_post, entry.grad_bias);
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

extern "C" FTrainStatus ftrainHPostBdaBwd(FTrainStorageView grad_output, FTrainStorageView residual,
                                          FTrainStorageView x, FTrainStorageView bias, FTrainStorageView h_res,
                                          FTrainStorageView h_post, FTrainStorageView grad_residual,
                                          FTrainStorageView grad_x, FTrainStorageView grad_h_res,
                                          FTrainStorageView grad_h_post, FTrainStorageView grad_bias,
                                          FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes,
                                          FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainHPostBdaBwd"};
    const ftrain::HPostBdaBwdEntry& entry = ftrain::hPostBdaBwdEntry();
    if (!entry.result.isSuccess()) {
        ftrain::setLastResult(entry.result.getStatus(), entry.result.getMessage());
        return entry.result.getStatus();
    }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_output, grad_output); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.residual, residual); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.x, x); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.bias, bias); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.h_res, h_res); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.h_post, h_post); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_residual, grad_residual); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_x, grad_x); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_h_res, grad_h_res); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_h_post, grad_h_post); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_bias, grad_bias); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetHPostBdaBwd(args, entry.h_post_bda_bwd, compute_type);
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
