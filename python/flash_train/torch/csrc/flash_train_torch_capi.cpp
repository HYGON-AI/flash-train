// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <flash_train/flash_train.h>

// The HIP-flavored half of the binding: builds StorageViews from plain
// pointers and calls the C API. See flash_train_torch.cpp for why the two
// worlds stay in separate translation units.
namespace flash_train_torch {
namespace {
// Adapted operators accept arbitrary tensor ranks and storage dtypes. This
// only describes contiguous device storage.
FTrainStorageView makeDeviceTensorView(const void* memory, const std::int64_t* dims, std::uint8_t rank,
                                       FTrainNumericType dtype) {
    return FTrainStorageView{const_cast<void*>(memory),    dims, nullptr, rank, dtype,
                             FTRAIN_INDEX_TYPE_CONTINUOUS, false};
}

// Scalar views borrow stack storage only during synchronous selection;
// primitives copy host scalar values before kernels are submitted asynchronously.
FTrainStorageView makeHostScalarView(const void* value, FTrainNumericType dtype) {
    return FTrainStorageView{const_cast<void*>(value), nullptr, nullptr, 0, dtype, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
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

// Execution goes through the convenience C API, which runs primitive 0; the
// workspace query therefore asks for that primitive's requirement.
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
    FTrainSparseAttnBwdPlan plan = nullptr;
    FTrainStatus status = ftrainSparseAttnBwdCreate(&plan, views[0], views[1], views[2], views[3], views[4], views[5],
                                                    views[6], views[7], views[8], views[9], views[10], views[11],
                                                    static_cast<FTrainNumericType>(compute_type));
    if (status == FTRAIN_STATUS_SUCCESS) {
        status = ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes(plan, 0, workspace_bytes);
    }
    if (plan != nullptr) { ftrainSparseAttnBwdDestroy(plan); }
    // This private query reports only its returned status; cleanup may replace the public TLS result.
    return static_cast<unsigned char>(status);
}

}  // namespace flash_train_torch
