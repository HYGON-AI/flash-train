// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct SparseAttnFwdIndexerEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId q{};
    FTrainTensorId kv{};
    FTrainTensorId attn_sink{};
    FTrainTensorId topk_indices{};
    FTrainTensorId topk_length{};
    FTrainTensorId out{};
    FTrainTensorId max_logits{};
    FTrainTensorId lse{};
    FTrainTensorId lse_indexer{};
    FTrainTensorId softmax_scale{};
    FTrainTensorId indexer_topk{};
    FTrainSparseAttnFwdIndexerOpId sparse_attn_fwd_indexer{};
    Result result{};
};

const SparseAttnFwdIndexerEntry& sparseAttnFwdIndexerEntry() {
    static const SparseAttnFwdIndexerEntry entry = [] {
        SparseAttnFwdIndexerEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.kv); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.attn_sink); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.topk_indices); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.topk_length); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.out); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.max_logits); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.lse); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.lse_indexer); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.softmax_scale); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.indexer_topk); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddSparseAttnFwdIndexer(pattern, &entry.sparse_attn_fwd_indexer, entry.q, entry.kv,
                                                          entry.attn_sink, entry.topk_indices, entry.topk_length,
                                                          entry.out, entry.max_logits, entry.lse, entry.lse_indexer,
                                                          entry.softmax_scale, entry.indexer_topk);
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

extern "C" FTrainStatus ftrainSparseAttnFwdIndexer(FTrainStorageView q, FTrainStorageView kv,
                                                   FTrainStorageView attn_sink, FTrainStorageView topk_indices,
                                                   FTrainStorageView topk_length, FTrainStorageView out,
                                                   FTrainStorageView max_logits, FTrainStorageView lse,
                                                   FTrainStorageView lse_indexer, FTrainStorageView softmax_scale,
                                                   FTrainStorageView indexer_topk, FTrainNumericType compute_type,
                                                   void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainSparseAttnFwdIndexer"};
    const ftrain::SparseAttnFwdIndexerEntry& entry = ftrain::sparseAttnFwdIndexerEntry();
    if (!entry.result.isSuccess()) {
        ftrain::setLastResult(entry.result.getStatus(), entry.result.getMessage());
        return entry.result.getStatus();
    }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.q, q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.kv, kv); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.attn_sink, attn_sink); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.topk_indices, topk_indices); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.topk_length, topk_length); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.out, out); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.max_logits, max_logits); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.lse, lse); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.lse_indexer, lse_indexer); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.softmax_scale, softmax_scale); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.indexer_topk, indexer_topk); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetSparseAttnFwdIndexer(args, entry.sparse_attn_fwd_indexer, compute_type);
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
