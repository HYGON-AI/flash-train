// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct SparseIndexerPredictEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId q{};
    FTrainTensorId k{};
    FTrainTensorId w{};
    FTrainTensorId topk_indices{};
    FTrainTensorId predict{};
    FTrainTensorId sm_scale{};
    FTrainTensorId topk_indices_global{};
    FTrainSparseIndexerPredictOpId sparse_indexer_predict{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const SparseIndexerPredictEntry& sparseIndexerPredictEntry() {
    static const SparseIndexerPredictEntry entry = [] {
        SparseIndexerPredictEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.k); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.w); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.topk_indices); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.predict); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.sm_scale); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.topk_indices_global); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddSparseIndexerPredict(pattern, &entry.sparse_indexer_predict, entry.q, entry.k,
                                                          entry.w, entry.topk_indices, entry.predict, entry.sm_scale,
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

extern "C" FTrainStatus ftrainSparseIndexerPredict(FTrainStorageView q, FTrainStorageView k, FTrainStorageView w,
                                                   FTrainStorageView topk_indices, FTrainStorageView predict,
                                                   FTrainStorageView sm_scale, FTrainStorageView topk_indices_global,
                                                   FTrainNumericType compute_type, void* workspace,
                                                   uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainSparseIndexerPredict"};
    const ftrain::SparseIndexerPredictEntry& entry = ftrain::sparseIndexerPredictEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.q, q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.k, k); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.w, w); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.topk_indices, topk_indices); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.predict, predict); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.sm_scale, sm_scale); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetTensor(args, entry.topk_indices_global, topk_indices_global);
    }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetSparseIndexerPredict(args, entry.sparse_indexer_predict, compute_type);
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
