// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct SparseAttnFwdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId q{};
    FTrainTensorId kv{};
    FTrainTensorId attn_sink{};
    FTrainTensorId topk_indices{};
    FTrainTensorId topk_length{};
    FTrainTensorId out{};
    FTrainTensorId lse{};
    FTrainTensorId softmax_scale{};
    FTrainSparseAttnFwdOpId sparse_attn_fwd{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const SparseAttnFwdEntry& sparseAttnFwdEntry() {
    static const SparseAttnFwdEntry entry = [] {
        SparseAttnFwdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.kv); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.attn_sink); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.topk_indices); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.topk_length); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.out); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.lse); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.softmax_scale); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddSparseAttnFwd(pattern, &entry.sparse_attn_fwd, entry.q, entry.kv, entry.attn_sink,
                                                   entry.topk_indices, entry.topk_length, entry.out, entry.lse,
                                                   entry.softmax_scale);
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

extern "C" FTrainStatus ftrainSparseAttnFwd(FTrainStorageView q, FTrainStorageView kv, FTrainStorageView attn_sink,
                                            FTrainStorageView topk_indices, FTrainStorageView topk_length,
                                            FTrainStorageView out, FTrainStorageView lse,
                                            FTrainStorageView softmax_scale, FTrainNumericType compute_type,
                                            void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainSparseAttnFwd"};
    const ftrain::SparseAttnFwdEntry& entry = ftrain::sparseAttnFwdEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.q, q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.kv, kv); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.attn_sink, attn_sink); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.topk_indices, topk_indices); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.topk_length, topk_length); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.out, out); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.lse, lse); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.softmax_scale, softmax_scale); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetSparseAttnFwd(args, entry.sparse_attn_fwd, compute_type);
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
