// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct IndexerTopKThdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId input_values{};
    FTrainTensorId cu_seqlens_q{};
    FTrainTensorId cu_seqlens_kv{};
    FTrainTensorId q_causal_offsets{};
    FTrainTensorId out_indices{};
    FTrainTensorId out_values{};
    FTrainTensorId ratio{};
    FTrainTensorId top_k{};
    FTrainTensorId order_values{};
    FTrainIndexerTopKThdOpId indexer_topk_thd{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const IndexerTopKThdEntry& indexerTopkThdEntry() {
    static const IndexerTopKThdEntry entry = [] {
        IndexerTopKThdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.input_values); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.cu_seqlens_q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.cu_seqlens_kv); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.q_causal_offsets); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.out_indices); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.out_values); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.ratio); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.top_k); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.order_values); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status =
                ftrainPatternAddIndexerTopKThd(pattern, &entry.indexer_topk_thd, entry.input_values, entry.cu_seqlens_q,
                                               entry.cu_seqlens_kv, entry.q_causal_offsets, entry.out_indices,
                                               entry.out_values, entry.ratio, entry.top_k, entry.order_values);
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

extern "C" FTrainStatus ftrainIndexerTopKThd(FTrainStorageView input_values, FTrainStorageView cu_seqlens_q,
                                             FTrainStorageView cu_seqlens_kv, FTrainStorageView q_causal_offsets,
                                             FTrainStorageView out_indices, FTrainStorageView out_values,
                                             FTrainStorageView ratio, FTrainStorageView top_k,
                                             FTrainStorageView order_values, void* workspace, uint64_t workspace_bytes,
                                             FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainIndexerTopKThd"};
    const ftrain::IndexerTopKThdEntry& entry = ftrain::indexerTopkThdEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.input_values, input_values); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.cu_seqlens_q, cu_seqlens_q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.cu_seqlens_kv, cu_seqlens_kv); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetTensor(args, entry.q_causal_offsets, q_causal_offsets);
    }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.out_indices, out_indices); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.out_values, out_values); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.ratio, ratio); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.top_k, top_k); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.order_values, order_values); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetIndexerTopKThd(args, entry.indexer_topk_thd); }
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
