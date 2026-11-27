// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>

#include <flash_train/flash_train.h>

#include "kda_bridge.h"

namespace flash_train_torch {
namespace {

FTrainStorageView makeDeviceMatrixView(const void* memory, const std::int64_t* dims) {
    return FTrainStorageView{const_cast<void*>(memory),    dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_FP32,
                             FTRAIN_INDEX_TYPE_CONTINUOUS, false};
}

FTrainStorageView makeHostScalarView(const void* value) {
    return FTrainStorageView{const_cast<void*>(value),     nullptr, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32,
                             FTRAIN_INDEX_TYPE_CONTINUOUS, true};
}

}  // namespace

// The HIP-flavored half of the binding: builds StorageViews from plain
// pointers and calls the staged C API. See flash_train_torch.cpp for why the
// two worlds stay in separate translation units.
unsigned char callFp32Gemm(const void* a, const std::int64_t* a_dims, const void* b, const std::int64_t* b_dims,
                           const void* c, const std::int64_t* c_dims, void* d, const std::int64_t* d_dims,
                           const void* alpha, const void* beta, void* stream) {
    return static_cast<unsigned char>(
        ftrainGemm(makeDeviceMatrixView(a, a_dims), makeDeviceMatrixView(b, b_dims), makeDeviceMatrixView(c, c_dims),
                   makeDeviceMatrixView(d, d_dims), makeHostScalarView(alpha), makeHostScalarView(beta),
                   FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0, static_cast<FTrainStream>(stream)));
}

namespace {
thread_local std::string gdn_error;

unsigned char recordGdnStatus(FTrainStatus status) {
    if (status != FTRAIN_STATUS_SUCCESS) gdn_error = ftrainGetLastMessage();
    return status;
}
}  // namespace

const char* lastGdnMessage() { return gdn_error.c_str(); }

void* prepareGdnPlan(int kind, const void* const* memory, const std::int64_t* const* dims,
                     const std::uint8_t* ranks, const std::uint8_t* types, float scale,
                     std::uint64_t* workspace_bytes, unsigned char* result) {
    const int count = kind == 0 ? 8 : kind == 1 ? 14 : kind == 2 ? 7 : 13;
    FTrainPattern pattern = nullptr;
    FTrainOps ops = nullptr;
    FTrainArgs args = nullptr;
    FTrainPlan plan = nullptr;
    FTrainTensorId ids[14]{};
    FTrainGdnForwardOpId gdn_fwd{};
    FTrainGdnBackwardOpId gdn_bwd{};
    FTrainGdn2ForwardOpId gdn2_fwd{};
    FTrainGdn2BackwardOpId gdn2_bwd{};
    FTrainStatus status = ftrainPatternCreate(&pattern);
    for (int i = 0; i < count && status == FTRAIN_STATUS_SUCCESS; ++i)
        status = ftrainPatternAddTensor(pattern, &ids[i]);
    if (status == FTRAIN_STATUS_SUCCESS) {
        if (kind == 0) status = ftrainPatternAddGdnForward(
            pattern, &gdn_fwd, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7]);
        else if (kind == 1) status = ftrainPatternAddGdnBackward(
            pattern, &gdn_bwd, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
            ids[8], ids[9], ids[10], ids[11], ids[12], ids[13]);
        else if (kind == 2) status = ftrainPatternAddGdn2Forward(
            pattern, &gdn2_fwd, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6]);
        else status = ftrainPatternAddGdn2Backward(
            pattern, &gdn2_bwd, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6],
            ids[7], ids[8], ids[9], ids[10], ids[11], ids[12]);
    }
    if (status == FTRAIN_STATUS_SUCCESS) status = ftrainOpsCreate(&ops, pattern);
    if (status == FTRAIN_STATUS_SUCCESS) status = ftrainArgsCreate(&args, ops);
    for (int i = 0; i < count && status == FTRAIN_STATUS_SUCCESS; ++i) {
        const FTrainStorageView view{const_cast<void*>(memory[i]), dims[i], nullptr, ranks[i], types[i],
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false};
        status = ftrainArgsSetTensor(args, ids[i], view);
    }
    if (status == FTRAIN_STATUS_SUCCESS) {
        if (kind == 0) status = ftrainArgsSetGdnForward(args, gdn_fwd, scale);
        else if (kind == 1) status = ftrainArgsSetGdnBackward(args, gdn_bwd, scale);
        else if (kind == 2) status = ftrainArgsSetGdn2Forward(args, gdn2_fwd, scale);
        else status = ftrainArgsSetGdn2Backward(args, gdn2_bwd, scale);
    }
    if (status == FTRAIN_STATUS_SUCCESS) status = ftrainPlanCreate(&plan, args);
    if (status == FTRAIN_STATUS_SUCCESS)
        status = ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, workspace_bytes);
    *result = recordGdnStatus(status);
    if (args) ftrainArgsDestroy(args);
    if (ops) ftrainOpsDestroy(ops);
    if (pattern) ftrainPatternDestroy(pattern);
    if (status != FTRAIN_STATUS_SUCCESS && plan) ftrainPlanDestroy(plan);
    return status == FTRAIN_STATUS_SUCCESS ? plan : nullptr;
}

unsigned char executeGdnPlan(void* plan, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    return recordGdnStatus(ftrainPlanExecutePrimitive(
        static_cast<FTrainPlan>(plan), 0, workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}

void destroyGdnPlan(void* plan) {
    if (plan) ftrainPlanDestroy(static_cast<FTrainPlan>(plan));

FTrainStorageView makeKdaView(const KdaBridgeView& v) {
    static const std::int64_t kEmpty[1] = {0};
    const FTrainNumericType type        = v.type == kKdaBf16   ? FTRAIN_NUMERIC_TYPE_BF16
                                          : v.type == kKdaFp32 ? FTRAIN_NUMERIC_TYPE_FP32
                                                               : FTRAIN_NUMERIC_TYPE_INT32;
    if (v.memory == nullptr) {  // unused optional port
        return FTrainStorageView{nullptr, kEmpty, nullptr, 1, type, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    }
    return FTrainStorageView{const_cast<void*>(v.memory),  v.dims, nullptr, static_cast<std::uint8_t>(v.num_dims), type,
                             FTRAIN_INDEX_TYPE_CONTINUOUS, false};
}

FTrainKdaAttributes makeKdaAttributes(const KdaBridgeAttributes& a) {
    return FTrainKdaAttributes{a.scale, a.lower_bound, a.gate_in_kernel, a.l2norm_qk, a.cp_phase};
}

template<typename Tensors, std::size_t N, typename Query, typename Exec>
unsigned char callKda(const KdaBridgeView* views, KdaBridgeAttributes attributes, void* workspace,
                      std::uint64_t workspace_bytes, std::uint64_t* query_bytes, void* stream, const char** message,
                      Query query, Exec exec) {
    static_assert(sizeof(Tensors) == N * sizeof(FTrainStorageView));
    FTrainStorageView ports[N];
    for (std::size_t i = 0; i < N; ++i) { ports[i] = makeKdaView(views[i]); }
    const Tensors* tensors    = reinterpret_cast<const Tensors*>(ports);
    const FTrainStatus status = query_bytes != nullptr ? query(tensors, makeKdaAttributes(attributes), query_bytes)
                                                       : exec(tensors, makeKdaAttributes(attributes), workspace,
                                                              workspace_bytes, static_cast<FTrainStream>(stream));
    if (status != FTRAIN_STATUS_SUCCESS && message != nullptr) { *message = ftrainGetLastMessage(); }
    return static_cast<unsigned char>(status);
}

}  // namespace

std::uint64_t kdaStateElements(std::int64_t tokens, std::int64_t heads, std::int64_t seqs, bool has_cu_seqlens) {
    return ftrainKdaStateElements(tokens, heads, seqs, has_cu_seqlens ? 1 : 0);
}

std::uint64_t kdaSavedHElements(std::int64_t tokens, std::int64_t heads, std::int64_t seqs, bool has_cu_seqlens) {
    return ftrainKdaSavedHElements(tokens, heads, seqs, has_cu_seqlens ? 1 : 0);
}

unsigned char callKdaForward(const KdaBridgeView* views, KdaBridgeAttributes attributes, void* workspace,
                             std::uint64_t workspace_bytes, std::uint64_t* query_bytes, void* stream,
                             const char** message) {
    return callKda<FTrainKdaForwardTensors, 19>(views, attributes, workspace, workspace_bytes, query_bytes, stream,
                                                message, ftrainKdaForwardGetWorkspaceBytes, ftrainKdaForward);
}

unsigned char callKdaBackward(const KdaBridgeView* views, KdaBridgeAttributes attributes, void* workspace,
                              std::uint64_t workspace_bytes, std::uint64_t* query_bytes, void* stream,
                              const char** message) {
    return callKda<FTrainKdaBackwardTensors, 27>(views, attributes, workspace, workspace_bytes, query_bytes, stream,
                                                 message, ftrainKdaBackwardGetWorkspaceBytes, ftrainKdaBackward);
}

}  // namespace flash_train_torch
