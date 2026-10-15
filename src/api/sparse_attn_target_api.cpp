// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct SparseAttnTargetEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId q{};
    FTrainTensorId k{};
    FTrainTensorId lse{};
    FTrainTensorId topk_indices{};
    FTrainTensorId target{};
    FTrainTensorId softmax_scale{};
    FTrainTensorId topk_indices_global{};
    FTrainSparseAttnTargetOpId sparse_attn_target{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const SparseAttnTargetEntry& sparseAttnTargetEntry() {
    static const SparseAttnTargetEntry entry = [] {
        SparseAttnTargetEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.k); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.lse); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.topk_indices); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.target); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.softmax_scale); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.topk_indices_global); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddSparseAttnTarget(pattern, &entry.sparse_attn_target, entry.q, entry.k, entry.lse,
                                                      entry.topk_indices, entry.target, entry.softmax_scale,
                                                      entry.topk_indices_global);
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

extern "C" FTrainStatus ftrainSparseAttnTarget(FTrainStorageView q, FTrainStorageView k, FTrainStorageView lse,
                                               FTrainStorageView topk_indices, FTrainStorageView target,
                                               FTrainStorageView softmax_scale, FTrainStorageView topk_indices_global,
                                               FTrainNumericType compute_type, void* workspace,
                                               uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainSparseAttnTarget"};
    const ftrain::SparseAttnTargetEntry& entry = ftrain::sparseAttnTargetEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.q, q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.k, k); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.lse, lse); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.topk_indices, topk_indices); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.target, target); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.softmax_scale, softmax_scale); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetTensor(args, entry.topk_indices_global, topk_indices_global);
    }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetSparseAttnTarget(args, entry.sparse_attn_target, compute_type);
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
