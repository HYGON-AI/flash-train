// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct IndexerTopKEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId input_values{};
    FTrainTensorId seq_lens{};
    FTrainTensorId out_indices{};
    FTrainTensorId out_values{};
    FTrainTensorId top_k{};
    FTrainTensorId next_n{};
    FTrainTensorId order_values{};
    FTrainIndexerTopKOpId indexer_topk{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const IndexerTopKEntry& indexerTopkEntry() {
    static const IndexerTopKEntry entry = [] {
        IndexerTopKEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.input_values); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.seq_lens); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.out_indices); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.out_values); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.top_k); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.next_n); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.order_values); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddIndexerTopK(pattern, &entry.indexer_topk, entry.input_values, entry.seq_lens,
                                                 entry.out_indices, entry.out_values, entry.top_k, entry.next_n,
                                                 entry.order_values);
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

extern "C" FTrainStatus ftrainIndexerTopK(FTrainStorageView input_values, FTrainStorageView seq_lens,
                                          FTrainStorageView out_indices, FTrainStorageView out_values,
                                          FTrainStorageView top_k, FTrainStorageView next_n,
                                          FTrainStorageView order_values, void* workspace, uint64_t workspace_bytes,
                                          FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainIndexerTopK"};
    const ftrain::IndexerTopKEntry& entry = ftrain::indexerTopkEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.input_values, input_values); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.seq_lens, seq_lens); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.out_indices, out_indices); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.out_values, out_values); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.top_k, top_k); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.next_n, next_n); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.order_values, order_values); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetIndexerTopK(args, entry.indexer_topk); }
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
