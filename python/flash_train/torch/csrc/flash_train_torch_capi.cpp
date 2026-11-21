// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <flash_train/flash_train.h>

namespace flash_train_torch {
namespace {
// Adapted operators accept arbitrary tensor ranks and storage dtypes.
// Like GEMM's matrix helper, this only describes contiguous device storage.
FTrainStorageView makeDeviceTensorView(const void* memory, const std::int64_t* dims, std::uint8_t rank,
                                       FTrainNumericType dtype) {
    return FTrainStorageView{const_cast<void*>(memory),    dims, nullptr, rank, dtype,
                             FTRAIN_INDEX_TYPE_CONTINUOUS, false};
}

// Scalar views borrow stack storage only during synchronous argument assembly/configure;
// primitives copy host scalar values before kernels are submitted asynchronously.
FTrainStorageView makeHostScalarView(const void* value, FTrainNumericType dtype) {
    return FTrainStorageView{const_cast<void*>(value), nullptr, nullptr, 0, dtype, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
}

// Only operators with dynamic workspace need a staged query. Execution uses
// the convenience C API, as GEMM does. Query primitive 0, which that API executes.
template<std::uint64_t N, typename AddOperation, typename SetAttributes>
unsigned char queryWorkspace(const FTrainStorageView (&views)[N], AddOperation add_operation,
                             SetAttributes set_attributes, std::uint64_t* bytes) {
    FTrainPattern pattern = nullptr;
    FTrainOps ops         = nullptr;
    FTrainArgs args       = nullptr;
    FTrainPlan plan       = nullptr;
    FTrainTensorId ids[N]{};
    FTrainStatus status = ftrainPatternCreate(&pattern);
    for (std::uint64_t i = 0; i < N && status == FTRAIN_STATUS_SUCCESS; ++i) {
        status = ftrainPatternAddTensor(pattern, &ids[i]);
    }
    if (status == FTRAIN_STATUS_SUCCESS) { status = add_operation(pattern, ids); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainOpsCreate(&ops, pattern); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainArgsCreate(&args, ops); }
    for (std::uint64_t i = 0; i < N && status == FTRAIN_STATUS_SUCCESS; ++i) {
        status = ftrainArgsSetTensor(args, ids[i], views[i]);
    }
    if (status == FTRAIN_STATUS_SUCCESS) { status = set_attributes(args); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPlanCreate(&plan, args); }
    if (status == FTRAIN_STATUS_SUCCESS) { status = ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, bytes); }
    if (plan != nullptr) { ftrainPlanDestroy(plan); }
    if (args != nullptr) { ftrainArgsDestroy(args); }
    if (ops != nullptr) { ftrainOpsDestroy(ops); }
    if (pattern != nullptr) { ftrainPatternDestroy(pattern); }
    // This private query reports only its returned status; cleanup may replace the public TLS result.
    return static_cast<unsigned char>(status);
}
}  // namespace

std::uint8_t numericTypeCodeFp32() { return FTRAIN_NUMERIC_TYPE_FP32; }
std::uint8_t numericTypeCodeFp16() { return FTRAIN_NUMERIC_TYPE_FP16; }
std::uint8_t numericTypeCodeBf16() { return FTRAIN_NUMERIC_TYPE_BF16; }
std::uint8_t numericTypeCodeInt32() { return FTRAIN_NUMERIC_TYPE_INT32; }

unsigned char callSparseAttnBwd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, std::uint8_t kv_type, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, std::uint8_t out_type, const void* dout_data,
    const std::int64_t* dout_dims, std::uint8_t dout_rank, std::uint8_t dout_type, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank,
    std::uint8_t dq_type, const void* dkv_data, const std::int64_t* dkv_dims, std::uint8_t dkv_rank,
    std::uint8_t dkv_type, const void* dsink_data, const std::int64_t* dsink_dims, std::uint8_t dsink_rank,
    double softmax_scale, void* workspace, std::uint64_t workspace_bytes, void* stream, std::uint8_t compute_type) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(ftrainSparseAttnBwd(
        makeDeviceTensorView(q_data, q_dims, q_rank, static_cast<FTrainNumericType>(q_type)),
        makeDeviceTensorView(kv_data, kv_dims, kv_rank, static_cast<FTrainNumericType>(kv_type)),
        makeDeviceTensorView(out_data, out_dims, out_rank, static_cast<FTrainNumericType>(out_type)),
        makeDeviceTensorView(dout_data, dout_dims, dout_rank, static_cast<FTrainNumericType>(dout_type)),
        makeDeviceTensorView(lse_data, lse_dims, lse_rank, FTRAIN_NUMERIC_TYPE_FP32),
        makeDeviceTensorView(attn_sink_data, attn_sink_dims, attn_sink_rank, FTRAIN_NUMERIC_TYPE_FP32),
        makeDeviceTensorView(topk_indices_data, topk_indices_dims, topk_indices_rank, FTRAIN_NUMERIC_TYPE_INT32),
        makeDeviceTensorView(topk_length_data, topk_length_dims, topk_length_rank, FTRAIN_NUMERIC_TYPE_INT32),
        makeDeviceTensorView(dq_data, dq_dims, dq_rank, static_cast<FTrainNumericType>(dq_type)),
        makeDeviceTensorView(dkv_data, dkv_dims, dkv_rank, static_cast<FTrainNumericType>(dkv_type)),
        makeDeviceTensorView(dsink_data, dsink_dims, dsink_rank, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        static_cast<FTrainNumericType>(compute_type), workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}

unsigned char querySparseAttnBwd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, std::uint8_t kv_type, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, std::uint8_t out_type, const void* dout_data,
    const std::int64_t* dout_dims, std::uint8_t dout_rank, std::uint8_t dout_type, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank,
    std::uint8_t dq_type, const void* dkv_data, const std::int64_t* dkv_dims, std::uint8_t dkv_rank,
    std::uint8_t dkv_type, const void* dsink_data, const std::int64_t* dsink_dims, std::uint8_t dsink_rank,
    double softmax_scale, std::uint64_t* workspace_bytes, std::uint8_t compute_type) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        makeDeviceTensorView(q_data, q_dims, q_rank, static_cast<FTrainNumericType>(q_type)),
        makeDeviceTensorView(kv_data, kv_dims, kv_rank, static_cast<FTrainNumericType>(kv_type)),
        makeDeviceTensorView(out_data, out_dims, out_rank, static_cast<FTrainNumericType>(out_type)),
        makeDeviceTensorView(dout_data, dout_dims, dout_rank, static_cast<FTrainNumericType>(dout_type)),
        makeDeviceTensorView(lse_data, lse_dims, lse_rank, FTRAIN_NUMERIC_TYPE_FP32),
        makeDeviceTensorView(attn_sink_data, attn_sink_dims, attn_sink_rank, FTRAIN_NUMERIC_TYPE_FP32),
        makeDeviceTensorView(topk_indices_data, topk_indices_dims, topk_indices_rank, FTRAIN_NUMERIC_TYPE_INT32),
        makeDeviceTensorView(topk_length_data, topk_length_dims, topk_length_rank, FTRAIN_NUMERIC_TYPE_INT32),
        makeDeviceTensorView(dq_data, dq_dims, dq_rank, static_cast<FTrainNumericType>(dq_type)),
        makeDeviceTensorView(dkv_data, dkv_dims, dkv_rank, static_cast<FTrainNumericType>(dkv_type)),
        makeDeviceTensorView(dsink_data, dsink_dims, dsink_rank, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)};
    FTrainSparseAttnBwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnBwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7], ids[8], ids[9], ids[10], ids[11]);
        },
        [&](FTrainArgs args) {
            return ftrainArgsSetSparseAttnBwd(args, operation, static_cast<FTrainNumericType>(compute_type));
        },
        workspace_bytes);
}

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

}  // namespace flash_train_torch
