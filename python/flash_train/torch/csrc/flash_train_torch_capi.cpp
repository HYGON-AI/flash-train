// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

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
    return FTrainKdaAttributes{a.scale, a.lower_bound, a.gate_in_kernel, a.l2norm_qk};
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
    return callKda<FTrainKdaForwardTensors, 17>(views, attributes, workspace, workspace_bytes, query_bytes, stream,
                                                message, ftrainKdaForwardGetWorkspaceBytes, ftrainKdaForward);
}

unsigned char callKdaBackward(const KdaBridgeView* views, KdaBridgeAttributes attributes, void* workspace,
                              std::uint64_t workspace_bytes, std::uint64_t* query_bytes, void* stream,
                              const char** message) {
    return callKda<FTrainKdaBackwardTensors, 24>(views, attributes, workspace, workspace_bytes, query_bytes, stream,
                                                 message, ftrainKdaBackwardGetWorkspaceBytes, ftrainKdaBackward);
}

}  // namespace flash_train_torch
