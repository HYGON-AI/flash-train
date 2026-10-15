// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct IndexerBwdDenseThdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId index_q{};
    FTrainTensorId weights{};
    FTrainTensorId index_k{};
    FTrainTensorId attn_score{};
    FTrainTensorId attn_l1norm{};
    FTrainTensorId index_score{};
    FTrainTensorId index_lse{};
    FTrainTensorId cu_seqlens_q{};
    FTrainTensorId cu_seqlens_k{};
    FTrainTensorId q_causal_offsets{};
    FTrainTensorId grad_index_score{};
    FTrainTensorId dq{};
    FTrainTensorId dw{};
    FTrainTensorId dk{};
    FTrainTensorId sm_scale{};
    FTrainTensorId loss_coeff{};
    FTrainTensorId grad_loss{};
    FTrainTensorId ratio{};
    FTrainIndexerBwdDenseThdOpId indexer_bwd_dense_thd{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const IndexerBwdDenseThdEntry& indexerBwdDenseThdEntry() {
    static const IndexerBwdDenseThdEntry entry = [] {
        IndexerBwdDenseThdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.index_q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.weights); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.index_k); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.attn_score); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.attn_l1norm); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.index_score); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.index_lse); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.cu_seqlens_q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.cu_seqlens_k); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.q_causal_offsets); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_index_score); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.dq); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.dw); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.dk); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.sm_scale); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.loss_coeff); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.grad_loss); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.ratio); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddIndexerBwdDenseThd(
                pattern, &entry.indexer_bwd_dense_thd, entry.index_q, entry.weights, entry.index_k, entry.attn_score,
                entry.attn_l1norm, entry.index_score, entry.index_lse, entry.cu_seqlens_q, entry.cu_seqlens_k,
                entry.q_causal_offsets, entry.grad_index_score, entry.dq, entry.dw, entry.dk, entry.sm_scale,
                entry.loss_coeff, entry.grad_loss, entry.ratio);
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

extern "C" FTrainStatus ftrainIndexerBwdDenseThd(
    FTrainStorageView index_q, FTrainStorageView weights, FTrainStorageView index_k, FTrainStorageView attn_score,
    FTrainStorageView attn_l1norm, FTrainStorageView index_score, FTrainStorageView index_lse,
    FTrainStorageView cu_seqlens_q, FTrainStorageView cu_seqlens_k, FTrainStorageView q_causal_offsets,
    FTrainStorageView grad_index_score, FTrainStorageView dq, FTrainStorageView dw, FTrainStorageView dk,
    FTrainStorageView sm_scale, FTrainStorageView loss_coeff, FTrainStorageView grad_loss, FTrainStorageView ratio,
    FTrainNumericType compute_type, void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainIndexerBwdDenseThd"};
    const ftrain::IndexerBwdDenseThdEntry& entry = ftrain::indexerBwdDenseThdEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.index_q, index_q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.weights, weights); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.index_k, index_k); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.attn_score, attn_score); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.attn_l1norm, attn_l1norm); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.index_score, index_score); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.index_lse, index_lse); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.cu_seqlens_q, cu_seqlens_q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.cu_seqlens_k, cu_seqlens_k); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetTensor(args, entry.q_causal_offsets, q_causal_offsets);
    }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetTensor(args, entry.grad_index_score, grad_index_score);
    }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.dq, dq); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.dw, dw); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.dk, dk); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.sm_scale, sm_scale); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.loss_coeff, loss_coeff); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.grad_loss, grad_loss); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.ratio, ratio); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetIndexerBwdDenseThd(args, entry.indexer_bwd_dense_thd, compute_type);
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
