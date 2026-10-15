// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/flash_train.h"

namespace ftrain {
namespace {
struct CSACompressorFwdEntry {
    FTrainOps ops{nullptr};
    FTrainTensorId kv{};
    FTrainTensorId score{};
    FTrainTensorId ape{};
    FTrainTensorId cu_seqlens{};
    FTrainTensorId cu_seqlens_comp{};
    FTrainTensorId compressed{};
    FTrainTensorId ratio{};
    FTrainTensorId head_dim{};
    FTrainTensorId coff{};
    FTrainTensorId total_comp{};
    FTrainCSACompressorFwdOpId csa_compressor_fwd{};
    FTrainStatus status{FTRAIN_STATUS_SUCCESS};
};

const CSACompressorFwdEntry& csaCompressorFwdEntry() {
    static const CSACompressorFwdEntry entry = [] {
        CSACompressorFwdEntry entry;
        FTrainPattern pattern = nullptr;
        FTrainStatus status   = ftrainPatternCreate(&pattern);
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.kv); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.score); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.ape); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.cu_seqlens); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.cu_seqlens_comp); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.compressed); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.ratio); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.head_dim); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.coff); }
        if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPatternAddTensor(pattern, &entry.total_comp); }
        if (status == FTRAIN_STATUS_SUCCESS) {
            status = ftrainPatternAddCSACompressorFwd(
                pattern, &entry.csa_compressor_fwd, entry.kv, entry.score, entry.ape, entry.cu_seqlens,
                entry.cu_seqlens_comp, entry.compressed, entry.ratio, entry.head_dim, entry.coff, entry.total_comp);
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

extern "C" FTrainStatus ftrainCSACompressorFwd(FTrainStorageView kv, FTrainStorageView score, FTrainStorageView ape,
                                               FTrainStorageView cu_seqlens, FTrainStorageView cu_seqlens_comp,
                                               FTrainStorageView compressed, FTrainStorageView ratio,
                                               FTrainStorageView head_dim, FTrainStorageView coff,
                                               FTrainStorageView total_comp, FTrainNumericType compute_type,
                                               void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainCSACompressorFwd"};
    const ftrain::CSACompressorFwdEntry& entry = ftrain::csaCompressorFwdEntry();
    if (entry.status != FTRAIN_STATUS_SUCCESS) { return entry.status; }
    FTrainArgs args     = nullptr;
    FTrainPlan plan     = nullptr;
    FTrainStatus status = ftrainArgsCreate(&args, entry.ops);
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.kv, kv); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.score, score); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.ape, ape); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.cu_seqlens, cu_seqlens); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.cu_seqlens_comp, cu_seqlens_comp); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.compressed, compressed); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.ratio, ratio); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.head_dim, head_dim); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.coff, coff); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsSetTensor(args, entry.total_comp, total_comp); }
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainArgsSetCSACompressorFwd(args, entry.csa_compressor_fwd, compute_type);
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
