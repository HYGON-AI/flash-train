// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct IndexerBwdSparseEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId index_q{};
    FTrainTensorId weights{};
    FTrainTensorId index_k{};
    FTrainTensorId attn_score{};
    FTrainTensorId index_score{};
    FTrainTensorId topk_indices{};
    FTrainTensorId grad_attn_score{};
    FTrainTensorId dq{};
    FTrainTensorId dw{};
    FTrainTensorId dk{};
    FTrainTensorId sm_scale{};
    FTrainTensorId loss_coeff{};
    FTrainTensorId grad_loss{};
    FTrainTensorId topk_indices_global{};
    FTrainIndexerBwdSparseOpId indexer_bwd_sparse{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const IndexerBwdSparseEntry& indexerBwdSparseEntry() {
    static const IndexerBwdSparseEntry entry = [] {
        IndexerBwdSparseEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.index_q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.weights); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.index_k); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.attn_score); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.index_score); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.topk_indices); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_attn_score); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.dq); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.dw); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.dk); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.sm_scale); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.loss_coeff); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_loss); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.topk_indices_global); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddIndexerBwdSparse(
                pattern, &entry.indexer_bwd_sparse, entry.index_q, entry.weights, entry.index_k, entry.attn_score,
                entry.index_score, entry.topk_indices, entry.grad_attn_score, entry.dq, entry.dw, entry.dk,
                entry.sm_scale, entry.loss_coeff, entry.grad_loss, entry.topk_indices_global);
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

extern "C" FTrainStatus ftrainIndexerBwdSparse(FTrainStorageView index_q, FTrainStorageView weights,
                                               FTrainStorageView index_k, FTrainStorageView attn_score,
                                               FTrainStorageView index_score, FTrainStorageView topk_indices,
                                               FTrainStorageView grad_attn_score, FTrainStorageView dq,
                                               FTrainStorageView dw, FTrainStorageView dk, FTrainStorageView sm_scale,
                                               FTrainStorageView loss_coeff, FTrainStorageView grad_loss,
                                               FTrainStorageView topk_indices_global, FTrainNumericType compute_type,
                                               void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainIndexerBwdSparse"};
    const ftrain::IndexerBwdSparseEntry& entry = ftrain::indexerBwdSparseEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.index_q, index_q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.weights, weights); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.index_k, index_k); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.attn_score, attn_score); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.index_score, index_score); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.topk_indices, topk_indices); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_attn_score, grad_attn_score); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.dq, dq); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.dw, dw); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.dk, dk); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.sm_scale, sm_scale); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.loss_coeff, loss_coeff); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_loss, grad_loss); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetTensor(args, entry.topk_indices_global, topk_indices_global);
    }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetIndexerBwdSparse(args, entry.indexer_bwd_sparse, compute_type);
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
