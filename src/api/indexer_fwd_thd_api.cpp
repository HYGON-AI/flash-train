// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct IndexerFwdThdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId q{};
    FTrainTensorId k{};
    FTrainTensorId w{};
    FTrainTensorId cu_seqlens_q{};
    FTrainTensorId cu_seqlens_kv{};
    FTrainTensorId q_causal_offsets{};
    FTrainTensorId scores{};
    FTrainTensorId ratio{};
    FTrainTensorId sm_scale{};
    FTrainTensorId max_seqlen_kv{};
    FTrainIndexerFwdThdOpId indexer_fwd_thd{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const IndexerFwdThdEntry& indexerFwdThdEntry() {
    static const IndexerFwdThdEntry entry = [] {
        IndexerFwdThdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.k); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.w); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.cu_seqlens_q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.cu_seqlens_kv); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.q_causal_offsets); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.scores); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.ratio); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.sm_scale); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.max_seqlen_kv); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddIndexerFwdThd(pattern, &entry.indexer_fwd_thd, entry.q, entry.k, entry.w,
                                                   entry.cu_seqlens_q, entry.cu_seqlens_kv, entry.q_causal_offsets,
                                                   entry.scores, entry.ratio, entry.sm_scale, entry.max_seqlen_kv);
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

extern "C" FTrainStatus ftrainIndexerFwdThd(FTrainStorageView q, FTrainStorageView k, FTrainStorageView w,
                                            FTrainStorageView cu_seqlens_q, FTrainStorageView cu_seqlens_kv,
                                            FTrainStorageView q_causal_offsets, FTrainStorageView scores,
                                            FTrainStorageView ratio, FTrainStorageView sm_scale,
                                            FTrainStorageView max_seqlen_kv, FTrainNumericType compute_type,
                                            void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainIndexerFwdThd"};
    const ftrain::IndexerFwdThdEntry& entry = ftrain::indexerFwdThdEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.q, q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.k, k); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.w, w); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.cu_seqlens_q, cu_seqlens_q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.cu_seqlens_kv, cu_seqlens_kv); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetTensor(args, entry.q_causal_offsets, q_causal_offsets);
    }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.scores, scores); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.ratio, ratio); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.sm_scale, sm_scale); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.max_seqlen_kv, max_seqlen_kv); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetIndexerFwdThd(args, entry.indexer_fwd_thd, compute_type);
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
