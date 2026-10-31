// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct DenseAttnScoreThdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId q{};
    FTrainTensorId k{};
    FTrainTensorId lse{};
    FTrainTensorId cu_seqlens_q{};
    FTrainTensorId cu_seqlens_kv{};
    FTrainTensorId q_causal_offsets{};
    FTrainTensorId out{};
    FTrainTensorId denom{};
    FTrainTensorId ratio{};
    FTrainTensorId softmax_scale{};
    FTrainTensorId max_seqlen_kv{};
    FTrainDenseAttnScoreThdOpId dense_attn_score_thd{};
    Result result{};
};

const DenseAttnScoreThdEntry& denseAttnScoreThdEntry() {
    static const DenseAttnScoreThdEntry entry = [] {
        DenseAttnScoreThdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.k); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.lse); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.cu_seqlens_q); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.cu_seqlens_kv); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.q_causal_offsets); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.out); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.denom); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.ratio); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.softmax_scale); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.max_seqlen_kv); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddDenseAttnScoreThd(pattern, &entry.dense_attn_score_thd, entry.q, entry.k,
                                                       entry.lse, entry.cu_seqlens_q, entry.cu_seqlens_kv,
                                                       entry.q_causal_offsets, entry.out, entry.denom, entry.ratio,
                                                       entry.softmax_scale, entry.max_seqlen_kv);
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

extern "C" FTrainStatus ftrainDenseAttnScoreThd(FTrainStorageView q, FTrainStorageView k, FTrainStorageView lse,
                                                FTrainStorageView cu_seqlens_q, FTrainStorageView cu_seqlens_kv,
                                                FTrainStorageView q_causal_offsets, FTrainStorageView out,
                                                FTrainStorageView denom, FTrainStorageView ratio,
                                                FTrainStorageView softmax_scale, FTrainStorageView max_seqlen_kv,
                                                FTrainNumericType compute_type, void* workspace,
                                                uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainDenseAttnScoreThd"};
    const ftrain::DenseAttnScoreThdEntry& entry = ftrain::denseAttnScoreThdEntry();
    if (!entry.result.isSuccess()) {
        ftrain::setLastResult(entry.result.getStatus(), entry.result.getMessage());
        return entry.result.getStatus();
    }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.q, q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.k, k); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.lse, lse); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.cu_seqlens_q, cu_seqlens_q); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.cu_seqlens_kv, cu_seqlens_kv); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetTensor(args, entry.q_causal_offsets, q_causal_offsets);
    }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.out, out); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.denom, denom); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.ratio, ratio); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.softmax_scale, softmax_scale); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.max_seqlen_kv, max_seqlen_kv); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetDenseAttnScoreThd(args, entry.dense_attn_score_thd, compute_type);
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
