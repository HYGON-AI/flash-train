// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <flash_train/flash_train.h>

namespace flash_train_torch {
namespace {
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

unsigned char callBf16SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                           const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                           const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                           const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                           std::uint8_t topk_indices_rank, const void* predict_data,
                                           const std::int64_t* predict_dims, std::uint8_t predict_rank, double sm_scale,
                                           bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                           void* stream) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainSparseIndexerPredict(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(predict_data), predict_dims, nullptr, predict_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_BF16,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                           const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                           const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                           const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                           std::uint8_t topk_indices_rank, const void* predict_data,
                                           const std::int64_t* predict_dims, std::uint8_t predict_rank, double sm_scale,
                                           bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                           void* stream) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainSparseIndexerPredict(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(predict_data), predict_dims, nullptr, predict_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_FP16,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                           const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                           const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                           const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                           std::uint8_t topk_indices_rank, const void* predict_data,
                                           const std::int64_t* predict_dims, std::uint8_t predict_rank, double sm_scale,
                                           bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                           void* stream) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainSparseIndexerPredict(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(predict_data), predict_dims, nullptr, predict_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_FP32,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                                std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                                std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                                const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                                const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                                std::uint8_t topk_indices_rank, const void* predict_data,
                                                const std::int64_t* predict_dims, std::uint8_t predict_rank,
                                                double sm_scale, bool topk_indices_global, void* workspace,
                                                std::uint64_t workspace_bytes, void* stream) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainSparseIndexerPredict(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, static_cast<FTrainNumericType>(w_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(predict_data), predict_dims, nullptr, predict_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_FP32,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                            const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                            const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                            const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                            std::uint8_t topk_indices_rank, const void* predict_data,
                                            const std::int64_t* predict_dims, std::uint8_t predict_rank,
                                            double sm_scale, bool topk_indices_global, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(k_data),            k_dims,            nullptr, k_rank,            FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(w_data),            w_dims,            nullptr, w_rank,            FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(predict_data),      predict_dims,      nullptr, predict_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainSparseIndexerPredictOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseIndexerPredict(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseIndexerPredict(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                            const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                            const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                            const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                            std::uint8_t topk_indices_rank, const void* predict_data,
                                            const std::int64_t* predict_dims, std::uint8_t predict_rank,
                                            double sm_scale, bool topk_indices_global, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(k_data),            k_dims,            nullptr, k_rank,            FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(w_data),            w_dims,            nullptr, w_rank,            FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(predict_data),      predict_dims,      nullptr, predict_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainSparseIndexerPredictOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseIndexerPredict(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseIndexerPredict(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                            const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                            const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                            const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                            std::uint8_t topk_indices_rank, const void* predict_data,
                                            const std::int64_t* predict_dims, std::uint8_t predict_rank,
                                            double sm_scale, bool topk_indices_global, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(k_data),            k_dims,            nullptr, k_rank,            FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(w_data),            w_dims,            nullptr, w_rank,            FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(predict_data),      predict_dims,      nullptr, predict_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainSparseIndexerPredictOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseIndexerPredict(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseIndexerPredict(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32SparseIndexerPredict(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                                 std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                                 std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                                 const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                                 const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                                 std::uint8_t topk_indices_rank, const void* predict_data,
                                                 const std::int64_t* predict_dims, std::uint8_t predict_rank,
                                                 double sm_scale, bool topk_indices_global,
                                                 std::uint64_t* workspace_bytes) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        FTrainStorageView{const_cast<void*>(k_data),            k_dims,            nullptr, k_rank,            static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        FTrainStorageView{const_cast<void*>(w_data),            w_dims,            nullptr, w_rank,            static_cast<FTrainNumericType>(w_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(predict_data),      predict_dims,      nullptr, predict_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainSparseIndexerPredictOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseIndexerPredict(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseIndexerPredict(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                 const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                 const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                 const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                 int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                 void* stream) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    return static_cast<unsigned char>(
        ftrainIndexerFwd(FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank,
                                           FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
                         makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_BF16,
                         workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                 const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                 const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                 const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                 int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                 void* stream) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    return static_cast<unsigned char>(
        ftrainIndexerFwd(FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_FP16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank,
                                           FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
                         makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP16,
                         workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                 const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                 const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                 const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                 int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                 void* stream) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    return static_cast<unsigned char>(
        ftrainIndexerFwd(FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank,
                                           FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
                         makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32,
                         workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                      std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                      std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                      const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                      const void* scores_data, const std::int64_t* scores_dims,
                                      std::uint8_t scores_rank, int64_t ratio, double sm_scale, void* workspace,
                                      std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    return static_cast<unsigned char>(
        ftrainIndexerFwd(FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank,
                                           static_cast<FTrainNumericType>(q_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank,
                                           static_cast<FTrainNumericType>(k_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank,
                                           static_cast<FTrainNumericType>(w_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank,
                                           FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
                         makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32,
                         workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                  const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                  const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                  const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                  int64_t ratio, double sm_scale, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),      q_dims,      nullptr, q_rank,      FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),      k_dims,      nullptr, k_rank,      FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data),      w_dims,      nullptr, w_rank,      FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainIndexerFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerFwd(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                  const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                  const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                  const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                  int64_t ratio, double sm_scale, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),      q_dims,      nullptr, q_rank,      FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),      k_dims,      nullptr, k_rank,      FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data),      w_dims,      nullptr, w_rank,      FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainIndexerFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                  const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                  const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                  const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank,
                                  int64_t ratio, double sm_scale, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),      q_dims,      nullptr, q_rank,      FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),      k_dims,      nullptr, k_rank,      FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data),      w_dims,      nullptr, w_rank,      FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainIndexerFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32IndexerFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                       std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                       std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                       const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                       const void* scores_data, const std::int64_t* scores_dims,
                                       std::uint8_t scores_rank, int64_t ratio, double sm_scale,
                                       std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),      q_dims,      nullptr, q_rank,      static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),      k_dims,      nullptr, k_rank,      static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data),      w_dims,      nullptr, w_rank,      static_cast<FTrainNumericType>(w_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainIndexerFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16IndexerFwdThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio, double sm_scale,
    int64_t max_seqlen_kv, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainIndexerFwdThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_BF16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16IndexerFwdThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio, double sm_scale,
    int64_t max_seqlen_kv, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainIndexerFwdThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32IndexerFwdThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* scores_data, const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio, double sm_scale,
    int64_t max_seqlen_kv, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainIndexerFwdThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32IndexerFwdThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* k_data,
    const std::int64_t* k_dims, std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
    const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
    const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* scores_data,
    const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainIndexerFwdThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, static_cast<FTrainNumericType>(w_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data), scores_dims, nullptr, scores_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16IndexerFwdThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                     const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                     std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                     const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                     const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                     std::uint8_t q_causal_offsets_rank, const void* scores_data,
                                     const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio,
                                     double sm_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(w_data),                w_dims,                nullptr, w_rank,             FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data),           scores_dims,           nullptr, scores_rank,        FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerFwdThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerFwdThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7], ids[8], ids[9]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerFwdThd(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16IndexerFwdThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                     const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                     std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                     const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                     const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                     std::uint8_t q_causal_offsets_rank, const void* scores_data,
                                     const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio,
                                     double sm_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(w_data),                w_dims,                nullptr, w_rank,             FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data),           scores_dims,           nullptr, scores_rank,        FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerFwdThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerFwdThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7], ids[8], ids[9]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerFwdThd(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32IndexerFwdThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                     const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                     std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                     const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                     const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                     std::uint8_t q_causal_offsets_rank, const void* scores_data,
                                     const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio,
                                     double sm_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(w_data),                w_dims,                nullptr, w_rank,             FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data),           scores_dims,           nullptr, scores_rank,        FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerFwdThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerFwdThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7], ids[8], ids[9]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerFwdThd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32IndexerFwdThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                          std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                          std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                          const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                          const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                          std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                          const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                          const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                          std::uint8_t q_causal_offsets_rank, const void* scores_data,
                                          const std::int64_t* scores_dims, std::uint8_t scores_rank, int64_t ratio,
                                          double sm_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(w_data),                w_dims,                nullptr, w_rank,             static_cast<FTrainNumericType>(w_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32,              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(scores_data),           scores_dims,           nullptr, scores_rank,        FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerFwdThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerFwdThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7], ids[8], ids[9]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerFwdThd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                        const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                        const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                        int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                        void* stream) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    return static_cast<unsigned char>(ftrainDenseIndexerScore(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_BF16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                        const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                        const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                        int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                        void* stream) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    return static_cast<unsigned char>(ftrainDenseIndexerScore(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                        const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                        const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                        int64_t ratio, double sm_scale, void* workspace, std::uint64_t workspace_bytes,
                                        void* stream) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    return static_cast<unsigned char>(ftrainDenseIndexerScore(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                             std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                             std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                             const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                             const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                             const void* denom_data, const std::int64_t* denom_dims,
                                             std::uint8_t denom_rank, int64_t ratio, double sm_scale, void* workspace,
                                             std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    return static_cast<unsigned char>(ftrainDenseIndexerScore(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, static_cast<FTrainNumericType>(w_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                         const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                         const void* denom_data, const std::int64_t* denom_dims,
                                         std::uint8_t denom_rank, int64_t ratio, double sm_scale,
                                         std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),     q_dims,     nullptr, q_rank,     FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),     k_dims,     nullptr, k_rank,     FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data),     w_dims,     nullptr, w_rank,     FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),   out_dims,   nullptr, out_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainDenseIndexerScoreOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseIndexerScore(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                     ids[5], ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseIndexerScore(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                         const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                         const void* denom_data, const std::int64_t* denom_dims,
                                         std::uint8_t denom_rank, int64_t ratio, double sm_scale,
                                         std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),     q_dims,     nullptr, q_rank,     FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),     k_dims,     nullptr, k_rank,     FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data),     w_dims,     nullptr, w_rank,     FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),   out_dims,   nullptr, out_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainDenseIndexerScoreOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseIndexerScore(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                     ids[5], ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseIndexerScore(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
                                         const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                         const void* denom_data, const std::int64_t* denom_dims,
                                         std::uint8_t denom_rank, int64_t ratio, double sm_scale,
                                         std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),     q_dims,     nullptr, q_rank,     FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),     k_dims,     nullptr, k_rank,     FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data),     w_dims,     nullptr, w_rank,     FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),   out_dims,   nullptr, out_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainDenseIndexerScoreOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseIndexerScore(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                     ids[5], ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseIndexerScore(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32DenseIndexerScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                              std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                              std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
                                              const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type,
                                              const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                              const void* denom_data, const std::int64_t* denom_dims,
                                              std::uint8_t denom_rank, int64_t ratio, double sm_scale,
                                              std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage = ratio;
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),     q_dims,     nullptr, q_rank,     static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),     k_dims,     nullptr, k_rank,     static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data),     w_dims,     nullptr, w_rank,     static_cast<FTrainNumericType>(w_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),   out_dims,   nullptr, out_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainDenseIndexerScoreOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseIndexerScore(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                     ids[5], ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseIndexerScore(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainDenseIndexerScoreThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_BF16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainDenseIndexerScoreThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainDenseIndexerScoreThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* k_data,
    const std::int64_t* k_dims, std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
    const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
    const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data, const std::int64_t* denom_dims,
    std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv, void* workspace,
    std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainDenseIndexerScoreThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(w_data), w_dims, nullptr, w_rank, static_cast<FTrainNumericType>(w_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(w_data),                w_dims,                nullptr, w_rank,             FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),              out_dims,              nullptr, out_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(denom_data),            denom_dims,            nullptr, denom_rank,         FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainDenseIndexerScoreThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseIndexerScoreThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseIndexerScoreThd(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(w_data),                w_dims,                nullptr, w_rank,             FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),              out_dims,              nullptr, out_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(denom_data),            denom_dims,            nullptr, denom_rank,         FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainDenseIndexerScoreThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseIndexerScoreThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseIndexerScoreThd(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* w_data, const std::int64_t* w_dims, std::uint8_t w_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv,
    std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(w_data),                w_dims,                nullptr, w_rank,             FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),              out_dims,              nullptr, out_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(denom_data),            denom_dims,            nullptr, denom_rank,         FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainDenseIndexerScoreThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseIndexerScoreThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseIndexerScoreThd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32DenseIndexerScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* k_data,
    const std::int64_t* k_dims, std::uint8_t k_rank, std::uint8_t k_type, const void* w_data,
    const std::int64_t* w_dims, std::uint8_t w_rank, std::uint8_t w_type, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
    const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data, const std::int64_t* denom_dims,
    std::uint8_t denom_rank, int64_t ratio, double sm_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float sm_scale_storage             = static_cast<float>(sm_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(w_data),                w_dims,                nullptr, w_rank,             static_cast<FTrainNumericType>(w_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32,              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),              out_dims,              nullptr, out_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(denom_data),            denom_dims,            nullptr, denom_rank,         FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainDenseIndexerScoreThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseIndexerScoreThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseIndexerScoreThd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                    const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                    const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                    std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                    const void* topk_length_data, const std::int64_t* topk_length_dims,
                                    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                    std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                    std::uint8_t lse_rank, double softmax_scale, void* workspace,
                                    std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(ftrainSparseAttnFwd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_BF16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                    const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                    const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                    std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                    const void* topk_length_data, const std::int64_t* topk_length_dims,
                                    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                    std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                    std::uint8_t lse_rank, double softmax_scale, void* workspace,
                                    std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(ftrainSparseAttnFwd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                    const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                    const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                    std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                    const void* topk_length_data, const std::int64_t* topk_length_dims,
                                    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                    std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                    std::uint8_t lse_rank, double softmax_scale, void* workspace,
                                    std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(ftrainSparseAttnFwd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         std::uint8_t q_type, const void* kv_data, const std::int64_t* kv_dims,
                                         std::uint8_t kv_rank, std::uint8_t kv_type, const void* attn_sink_data,
                                         const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank,
                                         const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                         std::uint8_t topk_indices_rank, const void* topk_length_data,
                                         const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank,
                                         const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                         std::uint8_t out_type, const void* lse_data, const std::int64_t* lse_dims,
                                         std::uint8_t lse_rank, double softmax_scale, void* workspace,
                                         std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(ftrainSparseAttnFwd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank,
                          static_cast<FTrainNumericType>(kv_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank,
                          static_cast<FTrainNumericType>(out_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                     std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                     std::uint8_t lse_rank, double softmax_scale, std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,           FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,          FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainSparseAttnFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnFwd(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                     std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                     std::uint8_t lse_rank, double softmax_scale, std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,           FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,          FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainSparseAttnFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims,
                                     std::uint8_t out_rank, const void* lse_data, const std::int64_t* lse_dims,
                                     std::uint8_t lse_rank, double softmax_scale, std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainSparseAttnFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32SparseAttnFwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                          std::uint8_t q_type, const void* kv_data, const std::int64_t* kv_dims,
                                          std::uint8_t kv_rank, std::uint8_t kv_type, const void* attn_sink_data,
                                          const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank,
                                          const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                          std::uint8_t topk_indices_rank, const void* topk_length_data,
                                          const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank,
                                          const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                          std::uint8_t out_type, const void* lse_data, const std::int64_t* lse_dims,
                                          std::uint8_t lse_rank, double softmax_scale, std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,
                          static_cast<FTrainNumericType>(kv_type),                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,
                          static_cast<FTrainNumericType>(out_type),                                                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainSparseAttnFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage       = static_cast<float>(softmax_scale);
    const std::int64_t indexer_topk_storage = indexer_topk;
    return static_cast<unsigned char>(ftrainSparseAttnFwdIndexer(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(max_logits_data), max_logits_dims, nullptr, max_logits_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_indexer_data), lse_indexer_dims, nullptr, lse_indexer_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&indexer_topk_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_BF16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage       = static_cast<float>(softmax_scale);
    const std::int64_t indexer_topk_storage = indexer_topk;
    return static_cast<unsigned char>(ftrainSparseAttnFwdIndexer(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(max_logits_data), max_logits_dims, nullptr, max_logits_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_indexer_data), lse_indexer_dims, nullptr, lse_indexer_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&indexer_topk_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage       = static_cast<float>(softmax_scale);
    const std::int64_t indexer_topk_storage = indexer_topk;
    return static_cast<unsigned char>(ftrainSparseAttnFwdIndexer(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(max_logits_data), max_logits_dims, nullptr, max_logits_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_indexer_data), lse_indexer_dims, nullptr, lse_indexer_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&indexer_topk_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, std::uint8_t kv_type, const void* attn_sink_data,
    const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* topk_length_data,
    const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, std::uint8_t out_type, const void* max_logits_data,
    const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage       = static_cast<float>(softmax_scale);
    const std::int64_t indexer_topk_storage = indexer_topk;
    return static_cast<unsigned char>(ftrainSparseAttnFwdIndexer(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank,
                          static_cast<FTrainNumericType>(kv_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank,
                          static_cast<FTrainNumericType>(out_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(max_logits_data), max_logits_dims, nullptr, max_logits_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_indexer_data), lse_indexer_dims, nullptr, lse_indexer_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&indexer_topk_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage       = static_cast<float>(softmax_scale);
    const std::int64_t indexer_topk_storage = indexer_topk;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,           FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,          FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(max_logits_data),   max_logits_dims,   nullptr, max_logits_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_indexer_data),  lse_indexer_dims,  nullptr, lse_indexer_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&indexer_topk_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainSparseAttnFwdIndexerOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnFwdIndexer(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnFwdIndexer(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage       = static_cast<float>(softmax_scale);
    const std::int64_t indexer_topk_storage = indexer_topk;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,           FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,          FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(max_logits_data),   max_logits_dims,   nullptr, max_logits_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_indexer_data),  lse_indexer_dims,  nullptr, lse_indexer_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&indexer_topk_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainSparseAttnFwdIndexerOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnFwdIndexer(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnFwdIndexer(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* attn_sink_data, const std::int64_t* attn_sink_dims,
    std::uint8_t attn_sink_rank, const void* topk_indices_data, const std::int64_t* topk_indices_dims,
    std::uint8_t topk_indices_rank, const void* topk_length_data, const std::int64_t* topk_length_dims,
    std::uint8_t topk_length_rank, const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
    const void* max_logits_data, const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage       = static_cast<float>(softmax_scale);
    const std::int64_t indexer_topk_storage = indexer_topk;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(max_logits_data),   max_logits_dims,   nullptr, max_logits_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_indexer_data),  lse_indexer_dims,  nullptr, lse_indexer_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&indexer_topk_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainSparseAttnFwdIndexerOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnFwdIndexer(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnFwdIndexer(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32SparseAttnFwdIndexer(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, std::uint8_t kv_type, const void* attn_sink_data,
    const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* topk_length_data,
    const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, std::uint8_t out_type, const void* max_logits_data,
    const std::int64_t* max_logits_dims, std::uint8_t max_logits_rank, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* lse_indexer_data,
    const std::int64_t* lse_indexer_dims, std::uint8_t lse_indexer_rank, double softmax_scale, int64_t indexer_topk,
    std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage       = static_cast<float>(softmax_scale);
    const std::int64_t indexer_topk_storage = indexer_topk;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,
                          static_cast<FTrainNumericType>(kv_type),                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,
                          static_cast<FTrainNumericType>(out_type),                                                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(max_logits_data),   max_logits_dims,   nullptr, max_logits_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        FTrainStorageView{const_cast<void*>(lse_indexer_data),  lse_indexer_dims,  nullptr, lse_indexer_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&indexer_topk_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainSparseAttnFwdIndexerOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnFwdIndexer(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                        ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnFwdIndexer(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16SparseAttnBwd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* out_data, const std::int64_t* out_dims,
    std::uint8_t out_rank, const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* attn_sink_data,
    const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* topk_length_data,
    const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
    std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims, std::uint8_t dsink_rank,
    double softmax_scale, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(ftrainSparseAttnBwd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dout_data), dout_dims, nullptr, dout_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dkv_data), dkv_dims, nullptr, dkv_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dsink_data), dsink_dims, nullptr, dsink_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_BF16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16SparseAttnBwd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* out_data, const std::int64_t* out_dims,
    std::uint8_t out_rank, const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* attn_sink_data,
    const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* topk_length_data,
    const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
    std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims, std::uint8_t dsink_rank,
    double softmax_scale, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(ftrainSparseAttnBwd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dout_data), dout_dims, nullptr, dout_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dkv_data), dkv_dims, nullptr, dkv_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dsink_data), dsink_dims, nullptr, dsink_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32SparseAttnBwd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* kv_data,
    const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* out_data, const std::int64_t* out_dims,
    std::uint8_t out_rank, const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
    const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* attn_sink_data,
    const std::int64_t* attn_sink_dims, std::uint8_t attn_sink_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* topk_length_data,
    const std::int64_t* topk_length_dims, std::uint8_t topk_length_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
    std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims, std::uint8_t dsink_rank,
    double softmax_scale, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(ftrainSparseAttnBwd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dout_data), dout_dims, nullptr, dout_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dkv_data), dkv_dims, nullptr, dkv_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dsink_data), dsink_dims, nullptr, dsink_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32SparseAttnBwd(
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
    double softmax_scale, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(ftrainSparseAttnBwd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank,
                          static_cast<FTrainNumericType>(kv_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank,
                          static_cast<FTrainNumericType>(out_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dout_data), dout_dims, nullptr, dout_rank,
                          static_cast<FTrainNumericType>(dout_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_sink_data), attn_sink_dims, nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank,
                          static_cast<FTrainNumericType>(dq_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dkv_data), dkv_dims, nullptr, dkv_rank,
                          static_cast<FTrainNumericType>(dkv_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dsink_data), dsink_dims, nullptr, dsink_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16SparseAttnBwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* dq_data, const std::int64_t* dq_dims,
                                     std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
                                     std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims,
                                     std::uint8_t dsink_rank, double softmax_scale, std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,           FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,          FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(dout_data),         dout_dims,         nullptr, dout_rank,         FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),           dq_dims,           nullptr, dq_rank,           FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(dkv_data),          dkv_dims,          nullptr, dkv_rank,          FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(dsink_data),        dsink_dims,        nullptr, dsink_rank,        FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainSparseAttnBwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnBwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7], ids[8], ids[9], ids[10], ids[11]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnBwd(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16SparseAttnBwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* dq_data, const std::int64_t* dq_dims,
                                     std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
                                     std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims,
                                     std::uint8_t dsink_rank, double softmax_scale, std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,           FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,          FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(dout_data),         dout_dims,         nullptr, dout_rank,         FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),           dq_dims,           nullptr, dq_rank,           FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(dkv_data),          dkv_dims,          nullptr, dkv_rank,          FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(dsink_data),        dsink_dims,        nullptr, dsink_rank,        FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainSparseAttnBwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnBwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7], ids[8], ids[9], ids[10], ids[11]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnBwd(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32SparseAttnBwd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* dout_data, const std::int64_t* dout_dims, std::uint8_t dout_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* attn_sink_data, const std::int64_t* attn_sink_dims,
                                     std::uint8_t attn_sink_rank, const void* topk_indices_data,
                                     const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
                                     const void* topk_length_data, const std::int64_t* topk_length_dims,
                                     std::uint8_t topk_length_rank, const void* dq_data, const std::int64_t* dq_dims,
                                     std::uint8_t dq_rank, const void* dkv_data, const std::int64_t* dkv_dims,
                                     std::uint8_t dkv_rank, const void* dsink_data, const std::int64_t* dsink_dims,
                                     std::uint8_t dsink_rank, double softmax_scale, std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(dout_data),         dout_dims,         nullptr, dout_rank,         FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),           dq_dims,           nullptr, dq_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(dkv_data),          dkv_dims,          nullptr, dkv_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(dsink_data),        dsink_dims,        nullptr, dsink_rank,        FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainSparseAttnBwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnBwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7], ids[8], ids[9], ids[10], ids[11]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnBwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32SparseAttnBwd(
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
    double softmax_scale, std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        FTrainStorageView{const_cast<void*>(kv_data),           kv_dims,           nullptr, kv_rank,
                          static_cast<FTrainNumericType>(kv_type),                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),          out_dims,          nullptr, out_rank,
                          static_cast<FTrainNumericType>(out_type),                                                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dout_data),         dout_dims,         nullptr, dout_rank,
                          static_cast<FTrainNumericType>(dout_type),                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        FTrainStorageView{const_cast<void*>(attn_sink_data),    attn_sink_dims,    nullptr, attn_sink_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data),  topk_length_dims,  nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),           dq_dims,           nullptr, dq_rank,
                          static_cast<FTrainNumericType>(dq_type),                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dkv_data),          dkv_dims,          nullptr, dkv_rank,
                          static_cast<FTrainNumericType>(dkv_type),                                                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dsink_data),        dsink_dims,        nullptr, dsink_rank,        FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainSparseAttnBwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnBwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                 ids[6], ids[7], ids[8], ids[9], ids[10], ids[11]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnBwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                     int64_t ratio, double softmax_scale, void* workspace,
                                     std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage  = ratio;
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(
        ftrainDenseAttnScore(FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank,
                                               FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank,
                                               FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank,
                                               FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank,
                                               FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank,
                                               FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
                             makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
                             FTRAIN_NUMERIC_TYPE_BF16, workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                     int64_t ratio, double softmax_scale, void* workspace,
                                     std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage  = ratio;
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(
        ftrainDenseAttnScore(FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank,
                                               FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank,
                                               FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank,
                                               FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank,
                                               FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank,
                                               FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
                             makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
                             FTRAIN_NUMERIC_TYPE_FP16, workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                     const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                     const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                     const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                     const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                     int64_t ratio, double softmax_scale, void* workspace,
                                     std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage  = ratio;
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(
        ftrainDenseAttnScore(FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank,
                                               FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank,
                                               FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank,
                                               FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank,
                                               FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank,
                                               FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                             makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
                             makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
                             FTRAIN_NUMERIC_TYPE_FP32, workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                          std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                          std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
                                          const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* out_data,
                                          const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
                                          const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio,
                                          double softmax_scale, void* workspace, std::uint64_t workspace_bytes,
                                          void* stream) {
    const std::int64_t ratio_storage  = ratio;
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    return static_cast<unsigned char>(ftrainDenseAttnScore(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                      const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                      const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                      const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                      const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                      int64_t ratio, double softmax_scale, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage  = ratio;
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),     q_dims,     nullptr, q_rank,     FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),     k_dims,     nullptr, k_rank,     FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data),   lse_dims,   nullptr, lse_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),   out_dims,   nullptr, out_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainDenseAttnScoreOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseAttnScore(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                  ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseAttnScore(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                      const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                      const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                      const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                      const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                      int64_t ratio, double softmax_scale, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage  = ratio;
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),     q_dims,     nullptr, q_rank,     FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),     k_dims,     nullptr, k_rank,     FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data),   lse_dims,   nullptr, lse_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),   out_dims,   nullptr, out_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainDenseAttnScoreOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseAttnScore(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                  ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseAttnScore(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                      const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                      const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                      const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank,
                                      const void* denom_data, const std::int64_t* denom_dims, std::uint8_t denom_rank,
                                      int64_t ratio, double softmax_scale, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage  = ratio;
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),     q_dims,     nullptr, q_rank,     FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),     k_dims,     nullptr, k_rank,     FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data),   lse_dims,   nullptr, lse_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),   out_dims,   nullptr, out_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainDenseAttnScoreOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseAttnScore(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                  ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseAttnScore(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32DenseAttnScore(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                           std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                           std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
                                           const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* out_data,
                                           const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
                                           const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio,
                                           double softmax_scale, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage  = ratio;
    const float softmax_scale_storage = static_cast<float>(softmax_scale);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),     q_dims,     nullptr, q_rank,     static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data),     k_dims,     nullptr, k_rank,     static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data),   lse_dims,   nullptr, lse_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),   out_dims,   nullptr, out_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainDenseAttnScoreOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseAttnScore(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                  ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseAttnScore(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16DenseAttnScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double softmax_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float softmax_scale_storage        = static_cast<float>(softmax_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainDenseAttnScoreThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_BF16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16DenseAttnScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double softmax_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float softmax_scale_storage        = static_cast<float>(softmax_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainDenseAttnScoreThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32DenseAttnScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, const void* k_data, const std::int64_t* k_dims,
    std::uint8_t k_rank, const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* out_data, const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
    const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio, double softmax_scale, int64_t max_seqlen_kv,
    void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float softmax_scale_storage        = static_cast<float>(softmax_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainDenseAttnScoreThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32DenseAttnScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* k_data,
    const std::int64_t* k_dims, std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
    const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data, const std::int64_t* denom_dims,
    std::uint8_t denom_rank, int64_t ratio, double softmax_scale, int64_t max_seqlen_kv, void* workspace,
    std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t ratio_storage         = ratio;
    const float softmax_scale_storage        = static_cast<float>(softmax_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    return static_cast<unsigned char>(ftrainDenseAttnScoreThd(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data), out_dims, nullptr, out_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(denom_data), denom_dims, nullptr, denom_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16DenseAttnScoreThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                         const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                         std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                         const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                         const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                         std::uint8_t q_causal_offsets_rank, const void* out_data,
                                         const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
                                         const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio,
                                         double softmax_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float softmax_scale_storage        = static_cast<float>(softmax_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(lse_data),              lse_dims,              nullptr, lse_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),              out_dims,              nullptr, out_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(denom_data),            denom_dims,            nullptr, denom_rank,         FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainDenseAttnScoreThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseAttnScoreThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                     ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseAttnScoreThd(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16DenseAttnScoreThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                         const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                         std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                         const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                         const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                         std::uint8_t q_causal_offsets_rank, const void* out_data,
                                         const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
                                         const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio,
                                         double softmax_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float softmax_scale_storage        = static_cast<float>(softmax_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(lse_data),              lse_dims,              nullptr, lse_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),              out_dims,              nullptr, out_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(denom_data),            denom_dims,            nullptr, denom_rank,         FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainDenseAttnScoreThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseAttnScoreThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                     ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseAttnScoreThd(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32DenseAttnScoreThd(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                         const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                         const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                         const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims,
                                         std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
                                         const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank,
                                         const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims,
                                         std::uint8_t q_causal_offsets_rank, const void* out_data,
                                         const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data,
                                         const std::int64_t* denom_dims, std::uint8_t denom_rank, int64_t ratio,
                                         double softmax_scale, int64_t max_seqlen_kv, std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float softmax_scale_storage        = static_cast<float>(softmax_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(lse_data),              lse_dims,              nullptr, lse_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),              out_dims,              nullptr, out_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        FTrainStorageView{const_cast<void*>(denom_data),            denom_dims,            nullptr, denom_rank,         FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                          false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainDenseAttnScoreThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseAttnScoreThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                     ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseAttnScoreThd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32DenseAttnScoreThd(
    const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank, std::uint8_t q_type, const void* k_data,
    const std::int64_t* k_dims, std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
    const std::int64_t* lse_dims, std::uint8_t lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_kv_data,
    const std::int64_t* cu_seqlens_kv_dims, std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* out_data,
    const std::int64_t* out_dims, std::uint8_t out_rank, const void* denom_data, const std::int64_t* denom_dims,
    std::uint8_t denom_rank, int64_t ratio, double softmax_scale, int64_t max_seqlen_kv,
    std::uint64_t* workspace_bytes) {
    const std::int64_t ratio_storage         = ratio;
    const float softmax_scale_storage        = static_cast<float>(softmax_scale);
    const std::int64_t max_seqlen_kv_storage = max_seqlen_kv;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),                q_dims,                nullptr, q_rank,             static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(k_data),                k_dims,                nullptr, k_rank,             static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(lse_data),              lse_dims,              nullptr, lse_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data),    cu_seqlens_kv_dims,    nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                        FTRAIN_NUMERIC_TYPE_INT32,              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_data),              out_dims,              nullptr, out_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(denom_data),            denom_dims,            nullptr, denom_rank,         FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                       false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&max_seqlen_kv_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainDenseAttnScoreThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddDenseAttnScoreThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                     ids[5], ids[6], ids[7], ids[8], ids[9], ids[10]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetDenseAttnScoreThd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                       const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                       const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                       const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                       std::uint8_t topk_indices_rank, const void* target_data,
                                       const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                       bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                       void* stream) {
    const float softmax_scale_storage              = static_cast<float>(softmax_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainSparseAttnTarget(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(target_data), target_dims, nullptr, target_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_BF16,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                       const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                       const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                       const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                       std::uint8_t topk_indices_rank, const void* target_data,
                                       const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                       bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                       void* stream) {
    const float softmax_scale_storage              = static_cast<float>(softmax_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainSparseAttnTarget(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(target_data), target_dims, nullptr, target_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_FP16,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                       const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                       const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                       const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                       std::uint8_t topk_indices_rank, const void* target_data,
                                       const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                       bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
                                       void* stream) {
    const float softmax_scale_storage              = static_cast<float>(softmax_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainSparseAttnTarget(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(target_data), target_dims, nullptr, target_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_FP32,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                            std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                            std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
                                            const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                            const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                            std::uint8_t topk_indices_rank, const void* target_data,
                                            const std::int64_t* target_dims, std::uint8_t target_rank,
                                            double softmax_scale, bool topk_indices_global, void* workspace,
                                            std::uint64_t workspace_bytes, void* stream) {
    const float softmax_scale_storage              = static_cast<float>(softmax_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainSparseAttnTarget(
        FTrainStorageView{const_cast<void*>(q_data), q_dims, nullptr, q_rank, static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(k_data), k_dims, nullptr, k_rank, static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(lse_data), lse_dims, nullptr, lse_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(target_data), target_dims, nullptr, target_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_FP32,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                        const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                        std::uint8_t topk_indices_rank, const void* target_data,
                                        const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                        bool topk_indices_global, std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage              = static_cast<float>(softmax_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(k_data),            k_dims,            nullptr, k_rank,            FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(target_data),       target_dims,       nullptr, target_rank,       FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainSparseAttnTargetOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnTarget(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                    ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnTarget(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                        const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                        std::uint8_t topk_indices_rank, const void* target_data,
                                        const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                        bool topk_indices_global, std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage              = static_cast<float>(softmax_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(k_data),            k_dims,            nullptr, k_rank,            FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(target_data),       target_dims,       nullptr, target_rank,       FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainSparseAttnTargetOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnTarget(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                    ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnTarget(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                        const void* k_data, const std::int64_t* k_dims, std::uint8_t k_rank,
                                        const void* lse_data, const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                        const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                        std::uint8_t topk_indices_rank, const void* target_data,
                                        const std::int64_t* target_dims, std::uint8_t target_rank, double softmax_scale,
                                        bool topk_indices_global, std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage              = static_cast<float>(softmax_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(k_data),            k_dims,            nullptr, k_rank,            FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(target_data),       target_dims,       nullptr, target_rank,       FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainSparseAttnTargetOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnTarget(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                    ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnTarget(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32SparseAttnTarget(const void* q_data, const std::int64_t* q_dims, std::uint8_t q_rank,
                                             std::uint8_t q_type, const void* k_data, const std::int64_t* k_dims,
                                             std::uint8_t k_rank, std::uint8_t k_type, const void* lse_data,
                                             const std::int64_t* lse_dims, std::uint8_t lse_rank,
                                             const void* topk_indices_data, const std::int64_t* topk_indices_dims,
                                             std::uint8_t topk_indices_rank, const void* target_data,
                                             const std::int64_t* target_dims, std::uint8_t target_rank,
                                             double softmax_scale, bool topk_indices_global,
                                             std::uint64_t* workspace_bytes) {
    const float softmax_scale_storage              = static_cast<float>(softmax_scale);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(q_data),            q_dims,            nullptr, q_rank,            static_cast<FTrainNumericType>(q_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        FTrainStorageView{const_cast<void*>(k_data),            k_dims,            nullptr, k_rank,            static_cast<FTrainNumericType>(k_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        FTrainStorageView{const_cast<void*>(lse_data),          lse_dims,          nullptr, lse_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(target_data),       target_dims,       nullptr, target_rank,       FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                              false},
        makeHostScalarView(&softmax_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainSparseAttnTargetOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddSparseAttnTarget(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                    ids[6]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetSparseAttnTarget(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callFp32IndexerTopK(const void* input_values_data, const std::int64_t* input_values_dims,
                                  std::uint8_t input_values_rank, const void* seq_lens_data,
                                  const std::int64_t* seq_lens_dims, std::uint8_t seq_lens_rank,
                                  const void* out_indices_data, const std::int64_t* out_indices_dims,
                                  std::uint8_t out_indices_rank, const void* out_values_data,
                                  const std::int64_t* out_values_dims, std::uint8_t out_values_rank, int64_t top_k,
                                  int64_t next_n, bool order_values, void* stream) {
    const std::int64_t top_k_storage        = top_k;
    const std::int64_t next_n_storage       = next_n;
    const std::uint8_t order_values_storage = static_cast<std::uint8_t>(order_values);
    return static_cast<unsigned char>(ftrainIndexerTopK(
        FTrainStorageView{const_cast<void*>(input_values_data), input_values_dims, nullptr, input_values_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(seq_lens_data), seq_lens_dims, nullptr, seq_lens_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_indices_data), out_indices_dims, nullptr, out_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_values_data), out_values_dims, nullptr, out_values_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&top_k_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&next_n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&order_values_storage, FTRAIN_NUMERIC_TYPE_UINT8), nullptr, 0,
        static_cast<FTrainStream>(stream)));
}
unsigned char callFp32IndexerTopKThd(const void* input_values_data, const std::int64_t* input_values_dims,
                                     std::uint8_t input_values_rank, const void* cu_seqlens_q_data,
                                     const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
                                     const void* cu_seqlens_kv_data, const std::int64_t* cu_seqlens_kv_dims,
                                     std::uint8_t cu_seqlens_kv_rank, const void* q_causal_offsets_data,
                                     const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
                                     const void* out_indices_data, const std::int64_t* out_indices_dims,
                                     std::uint8_t out_indices_rank, const void* out_values_data,
                                     const std::int64_t* out_values_dims, std::uint8_t out_values_rank, int64_t ratio,
                                     int64_t top_k, bool order_values, void* stream) {
    const std::int64_t ratio_storage        = ratio;
    const std::int64_t top_k_storage        = top_k;
    const std::uint8_t order_values_storage = static_cast<std::uint8_t>(order_values);
    return static_cast<unsigned char>(ftrainIndexerTopKThd(
        FTrainStorageView{const_cast<void*>(input_values_data), input_values_dims, nullptr, input_values_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_kv_data), cu_seqlens_kv_dims, nullptr, cu_seqlens_kv_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_indices_data), out_indices_dims, nullptr, out_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(out_values_data), out_values_dims, nullptr, out_values_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&top_k_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&order_values_storage, FTRAIN_NUMERIC_TYPE_UINT8), nullptr, 0,
        static_cast<FTrainStream>(stream)));
}
unsigned char callInt32Compactify(const void* indices_data, const std::int64_t* indices_dims, std::uint8_t indices_rank,
                                  const void* output_data, const std::int64_t* output_dims, std::uint8_t output_rank,
                                  const void* topk_length_data, const std::int64_t* topk_length_dims,
                                  std::uint8_t topk_length_rank, void* stream) {
    return static_cast<unsigned char>(ftrainCompactify(
        FTrainStorageView{const_cast<void*>(indices_data), indices_dims, nullptr, indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(output_data), output_dims, nullptr, output_rank, FTRAIN_NUMERIC_TYPE_INT32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_length_data), topk_length_dims, nullptr, topk_length_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        nullptr, 0, static_cast<FTrainStream>(stream)));
}
unsigned char callBf16IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
    void* stream, const void* grad_loss_data) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const float loss_coeff_storage                 = static_cast<float>(loss_coeff);
    const float grad_loss_storage                  = static_cast<float>(grad_loss);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainIndexerBwdSparse(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_attn_score_data), grad_attn_score_dims, nullptr, grad_attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        FTrainStorageView{const_cast<void*>(grad_loss_data != nullptr ? grad_loss_data : &grad_loss_storage), nullptr,
                          nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS,
                          grad_loss_data == nullptr},
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_BF16,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
    void* stream) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const float loss_coeff_storage                 = static_cast<float>(loss_coeff);
    const float grad_loss_storage                  = static_cast<float>(grad_loss);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainIndexerBwdSparse(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_attn_score_data), grad_attn_score_dims, nullptr, grad_attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_FP16,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes,
    void* stream) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const float loss_coeff_storage                 = static_cast<float>(loss_coeff);
    const float grad_loss_storage                  = static_cast<float>(grad_loss);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainIndexerBwdSparse(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_attn_score_data), grad_attn_score_dims, nullptr, grad_attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_FP32,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* topk_indices_data, const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
    const void* grad_attn_score_data, const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, bool topk_indices_global, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const float loss_coeff_storage                 = static_cast<float>(loss_coeff);
    const float grad_loss_storage                  = static_cast<float>(grad_loss);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    return static_cast<unsigned char>(ftrainIndexerBwdSparse(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          static_cast<FTrainNumericType>(index_q_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          static_cast<FTrainNumericType>(weights_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          static_cast<FTrainNumericType>(index_k_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data), topk_indices_dims, nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_attn_score_data), grad_attn_score_dims, nullptr, grad_attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank,
                          static_cast<FTrainNumericType>(dq_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank,
                          static_cast<FTrainNumericType>(dw_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank,
                          static_cast<FTrainNumericType>(dk_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8), FTRAIN_NUMERIC_TYPE_FP32,
        workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, std::uint64_t* workspace_bytes,
    const void* grad_loss_data) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const float loss_coeff_storage                 = static_cast<float>(loss_coeff);
    const float grad_loss_storage                  = static_cast<float>(grad_loss);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),                                                    index_q_dims,         nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                                                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),                                                    weights_dims,         nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                                                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),                                                    index_k_dims,         nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                                                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),                                                 attn_score_dims,      nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),                                                index_score_dims,     nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data),                                               topk_indices_dims,    nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                                                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_attn_score_data),                                            grad_attn_score_dims, nullptr, grad_attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),                                                         dq_dims,              nullptr, dq_rank,              FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                                                    false},
        FTrainStorageView{const_cast<void*>(dw_data),                                                         dw_dims,              nullptr, dw_rank,              FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                                                    false},
        FTrainStorageView{const_cast<void*>(dk_data),                                                         dk_dims,              nullptr, dk_rank,              FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                                                                    false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        FTrainStorageView{const_cast<void*>(grad_loss_data != nullptr ? grad_loss_data : &grad_loss_storage), nullptr,
                          nullptr,                                                                                                           0,                    FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS,
                          grad_loss_data == nullptr                                                                                                                                                                             },
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainIndexerBwdSparseOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdSparse(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                    ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdSparse(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const float loss_coeff_storage                 = static_cast<float>(loss_coeff);
    const float grad_loss_storage                  = static_cast<float>(grad_loss);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),         index_q_dims,         nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),         weights_dims,         nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),         index_k_dims,         nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),      attn_score_dims,      nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),     index_score_dims,     nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data),    topk_indices_dims,    nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_attn_score_data), grad_attn_score_dims, nullptr, grad_attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),              dq_dims,              nullptr, dq_rank,              FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        FTrainStorageView{const_cast<void*>(dw_data),              dw_dims,              nullptr, dw_rank,              FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        FTrainStorageView{const_cast<void*>(dk_data),              dk_dims,              nullptr, dk_rank,              FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainIndexerBwdSparseOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdSparse(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                    ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdSparse(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* topk_indices_data,
    const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank, const void* grad_attn_score_data,
    const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, bool topk_indices_global, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const float loss_coeff_storage                 = static_cast<float>(loss_coeff);
    const float grad_loss_storage                  = static_cast<float>(grad_loss);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),         index_q_dims,         nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),         weights_dims,         nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),         index_k_dims,         nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),      attn_score_dims,      nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),     index_score_dims,     nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data),    topk_indices_dims,    nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_attn_score_data), grad_attn_score_dims, nullptr, grad_attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),              dq_dims,              nullptr, dq_rank,              FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        FTrainStorageView{const_cast<void*>(dw_data),              dw_dims,              nullptr, dw_rank,              FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        FTrainStorageView{const_cast<void*>(dk_data),              dk_dims,              nullptr, dk_rank,              FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainIndexerBwdSparseOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdSparse(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                    ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdSparse(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32IndexerBwdSparse(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* topk_indices_data, const std::int64_t* topk_indices_dims, std::uint8_t topk_indices_rank,
    const void* grad_attn_score_data, const std::int64_t* grad_attn_score_dims, std::uint8_t grad_attn_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, bool topk_indices_global, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage                   = static_cast<float>(sm_scale);
    const float loss_coeff_storage                 = static_cast<float>(loss_coeff);
    const float grad_loss_storage                  = static_cast<float>(grad_loss);
    const std::uint8_t topk_indices_global_storage = static_cast<std::uint8_t>(topk_indices_global);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),         index_q_dims,         nullptr, index_q_rank,
                          static_cast<FTrainNumericType>(index_q_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),         weights_dims,         nullptr, weights_rank,
                          static_cast<FTrainNumericType>(weights_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),         index_k_dims,         nullptr, index_k_rank,
                          static_cast<FTrainNumericType>(index_k_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),      attn_score_dims,      nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),     index_score_dims,     nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(topk_indices_data),    topk_indices_dims,    nullptr, topk_indices_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                    FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_attn_score_data), grad_attn_score_dims, nullptr, grad_attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                     FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),              dq_dims,              nullptr, dq_rank,
                          static_cast<FTrainNumericType>(dq_type),      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data),              dw_dims,              nullptr, dw_rank,
                          static_cast<FTrainNumericType>(dw_type),      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data),              dk_dims,              nullptr, dk_rank,
                          static_cast<FTrainNumericType>(dk_type),      FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&topk_indices_global_storage, FTRAIN_NUMERIC_TYPE_UINT8)
    };
    FTrainIndexerBwdSparseOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdSparse(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                    ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdSparse(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    return static_cast<unsigned char>(ftrainIndexerBwdDense(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data), attn_l1norm_dims, nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data), index_lse_dims, nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_BF16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    return static_cast<unsigned char>(ftrainIndexerBwdDense(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data), attn_l1norm_dims, nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data), index_lse_dims, nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    return static_cast<unsigned char>(ftrainIndexerBwdDense(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data), attn_l1norm_dims, nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data), index_lse_dims, nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* attn_l1norm_data, const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* index_lse_data, const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank,
    const void* grad_index_score_data, const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    return static_cast<unsigned char>(ftrainIndexerBwdDense(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          static_cast<FTrainNumericType>(index_q_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          static_cast<FTrainNumericType>(weights_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          static_cast<FTrainNumericType>(index_k_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data), attn_l1norm_dims, nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data), index_lse_dims, nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank,
                          static_cast<FTrainNumericType>(dq_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank,
                          static_cast<FTrainNumericType>(dw_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank,
                          static_cast<FTrainNumericType>(dk_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),          index_q_dims,          nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),          weights_dims,          nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),          index_k_dims,          nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),       attn_score_dims,       nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data),      attn_l1norm_dims,      nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),      index_score_dims,      nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data),        index_lse_dims,        nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank,                                                                      FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),               dq_dims,               nullptr, dq_rank,          FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(dw_data),               dw_dims,               nullptr, dw_rank,          FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(dk_data),               dk_dims,               nullptr, dk_rank,          FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                       false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerBwdDenseOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdDense(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                   ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13],
                                                   ids[14]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdDense(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),          index_q_dims,          nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),          weights_dims,          nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),          index_k_dims,          nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),       attn_score_dims,       nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data),      attn_l1norm_dims,      nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),      index_score_dims,      nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data),        index_lse_dims,        nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank,                                                                      FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),               dq_dims,               nullptr, dq_rank,          FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(dw_data),               dw_dims,               nullptr, dw_rank,          FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(dk_data),               dk_dims,               nullptr, dk_rank,          FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                       false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerBwdDenseOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdDense(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                   ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13],
                                                   ids[14]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdDense(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),          index_q_dims,          nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),          weights_dims,          nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),          index_k_dims,          nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),       attn_score_dims,       nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data),      attn_l1norm_dims,      nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),      index_score_dims,      nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data),        index_lse_dims,        nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank,                                                                      FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),               dq_dims,               nullptr, dq_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(dw_data),               dw_dims,               nullptr, dw_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                       false},
        FTrainStorageView{const_cast<void*>(dk_data),               dk_dims,               nullptr, dk_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                       false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerBwdDenseOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdDense(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                   ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13],
                                                   ids[14]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdDense(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32IndexerBwdDense(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* attn_l1norm_data, const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* index_lse_data, const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank,
    const void* grad_index_score_data, const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),          index_q_dims,          nullptr, index_q_rank,
                          static_cast<FTrainNumericType>(index_q_type),                                                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),          weights_dims,          nullptr, weights_rank,
                          static_cast<FTrainNumericType>(weights_type),                                                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),          index_k_dims,          nullptr, index_k_rank,
                          static_cast<FTrainNumericType>(index_k_type),                                                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),       attn_score_dims,       nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data),      attn_l1norm_dims,      nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),      index_score_dims,      nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data),        index_lse_dims,        nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank,                                                                      FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),               dq_dims,               nullptr, dq_rank,
                          static_cast<FTrainNumericType>(dq_type),                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data),               dw_dims,               nullptr, dw_rank,
                          static_cast<FTrainNumericType>(dw_type),                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data),               dk_dims,               nullptr, dk_rank,
                          static_cast<FTrainNumericType>(dk_type),                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerBwdDenseOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdDense(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5],
                                                   ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12], ids[13],
                                                   ids[14]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdDense(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    return static_cast<unsigned char>(ftrainIndexerBwdDenseThd(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data), attn_l1norm_dims, nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data), index_lse_dims, nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_k_data), cu_seqlens_k_dims, nullptr, cu_seqlens_k_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_BF16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    return static_cast<unsigned char>(ftrainIndexerBwdDenseThd(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data), attn_l1norm_dims, nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data), index_lse_dims, nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_k_data), cu_seqlens_k_dims, nullptr, cu_seqlens_k_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    return static_cast<unsigned char>(ftrainIndexerBwdDenseThd(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data), attn_l1norm_dims, nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data), index_lse_dims, nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_k_data), cu_seqlens_k_dims, nullptr, cu_seqlens_k_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* attn_l1norm_data, const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* index_lse_data, const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_k_data, const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* grad_index_score_data, const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, int64_t ratio, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    return static_cast<unsigned char>(ftrainIndexerBwdDenseThd(
        FTrainStorageView{const_cast<void*>(index_q_data), index_q_dims, nullptr, index_q_rank,
                          static_cast<FTrainNumericType>(index_q_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data), weights_dims, nullptr, weights_rank,
                          static_cast<FTrainNumericType>(weights_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data), index_k_dims, nullptr, index_k_rank,
                          static_cast<FTrainNumericType>(index_k_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data), attn_score_dims, nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data), attn_l1norm_dims, nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data), index_score_dims, nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data), index_lse_dims, nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data), cu_seqlens_q_dims, nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_k_data), cu_seqlens_k_dims, nullptr, cu_seqlens_k_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data), dq_dims, nullptr, dq_rank,
                          static_cast<FTrainNumericType>(dq_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data), dw_dims, nullptr, dw_rank,
                          static_cast<FTrainNumericType>(dw_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data), dk_dims, nullptr, dk_rank,
                          static_cast<FTrainNumericType>(dk_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),          index_q_dims,          nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),          weights_dims,          nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),          index_k_dims,          nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),       attn_score_dims,       nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data),      attn_l1norm_dims,      nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),      index_score_dims,      nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data),        index_lse_dims,        nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_k_data),     cu_seqlens_k_dims,     nullptr, cu_seqlens_k_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                       FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank,                                                                       FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),               dq_dims,               nullptr, dq_rank,           FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        FTrainStorageView{const_cast<void*>(dw_data),               dw_dims,               nullptr, dw_rank,           FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        FTrainStorageView{const_cast<void*>(dk_data),               dk_dims,               nullptr, dk_rank,           FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerBwdDenseThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdDenseThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13], ids[14], ids[15], ids[16], ids[17]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdDenseThd(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),          index_q_dims,          nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),          weights_dims,          nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),          index_k_dims,          nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),       attn_score_dims,       nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data),      attn_l1norm_dims,      nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),      index_score_dims,      nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data),        index_lse_dims,        nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_k_data),     cu_seqlens_k_dims,     nullptr, cu_seqlens_k_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                       FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank,                                                                       FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),               dq_dims,               nullptr, dq_rank,           FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        FTrainStorageView{const_cast<void*>(dw_data),               dw_dims,               nullptr, dw_rank,           FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        FTrainStorageView{const_cast<void*>(dk_data),               dk_dims,               nullptr, dk_rank,           FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerBwdDenseThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdDenseThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13], ids[14], ids[15], ids[16], ids[17]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdDenseThd(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, const void* weights_data,
    const std::int64_t* weights_dims, std::uint8_t weights_rank, const void* index_k_data,
    const std::int64_t* index_k_dims, std::uint8_t index_k_rank, const void* attn_score_data,
    const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank, const void* attn_l1norm_data,
    const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank, const void* index_score_data,
    const std::int64_t* index_score_dims, std::uint8_t index_score_rank, const void* index_lse_data,
    const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank, const void* cu_seqlens_q_data,
    const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank, const void* cu_seqlens_k_data,
    const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank, const void* q_causal_offsets_data,
    const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank, const void* grad_index_score_data,
    const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank, const void* dq_data,
    const std::int64_t* dq_dims, std::uint8_t dq_rank, const void* dw_data, const std::int64_t* dw_dims,
    std::uint8_t dw_rank, const void* dk_data, const std::int64_t* dk_dims, std::uint8_t dk_rank, double sm_scale,
    double loss_coeff, double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),          index_q_dims,          nullptr, index_q_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),          weights_dims,          nullptr, weights_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),          index_k_dims,          nullptr, index_k_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),       attn_score_dims,       nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data),      attn_l1norm_dims,      nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),      index_score_dims,      nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data),        index_lse_dims,        nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_k_data),     cu_seqlens_k_dims,     nullptr, cu_seqlens_k_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                       FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank,                                                                       FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),               dq_dims,               nullptr, dq_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        FTrainStorageView{const_cast<void*>(dw_data),               dw_dims,               nullptr, dw_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        FTrainStorageView{const_cast<void*>(dk_data),               dk_dims,               nullptr, dk_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                         false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerBwdDenseThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdDenseThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13], ids[14], ids[15], ids[16], ids[17]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdDenseThd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32IndexerBwdDenseThd(
    const void* index_q_data, const std::int64_t* index_q_dims, std::uint8_t index_q_rank, std::uint8_t index_q_type,
    const void* weights_data, const std::int64_t* weights_dims, std::uint8_t weights_rank, std::uint8_t weights_type,
    const void* index_k_data, const std::int64_t* index_k_dims, std::uint8_t index_k_rank, std::uint8_t index_k_type,
    const void* attn_score_data, const std::int64_t* attn_score_dims, std::uint8_t attn_score_rank,
    const void* attn_l1norm_data, const std::int64_t* attn_l1norm_dims, std::uint8_t attn_l1norm_rank,
    const void* index_score_data, const std::int64_t* index_score_dims, std::uint8_t index_score_rank,
    const void* index_lse_data, const std::int64_t* index_lse_dims, std::uint8_t index_lse_rank,
    const void* cu_seqlens_q_data, const std::int64_t* cu_seqlens_q_dims, std::uint8_t cu_seqlens_q_rank,
    const void* cu_seqlens_k_data, const std::int64_t* cu_seqlens_k_dims, std::uint8_t cu_seqlens_k_rank,
    const void* q_causal_offsets_data, const std::int64_t* q_causal_offsets_dims, std::uint8_t q_causal_offsets_rank,
    const void* grad_index_score_data, const std::int64_t* grad_index_score_dims, std::uint8_t grad_index_score_rank,
    const void* dq_data, const std::int64_t* dq_dims, std::uint8_t dq_rank, std::uint8_t dq_type, const void* dw_data,
    const std::int64_t* dw_dims, std::uint8_t dw_rank, std::uint8_t dw_type, const void* dk_data,
    const std::int64_t* dk_dims, std::uint8_t dk_rank, std::uint8_t dk_type, double sm_scale, double loss_coeff,
    double grad_loss, int64_t ratio, std::uint64_t* workspace_bytes) {
    const float sm_scale_storage     = static_cast<float>(sm_scale);
    const float loss_coeff_storage   = static_cast<float>(loss_coeff);
    const float grad_loss_storage    = static_cast<float>(grad_loss);
    const std::int64_t ratio_storage = ratio;
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(index_q_data),          index_q_dims,          nullptr, index_q_rank,
                          static_cast<FTrainNumericType>(index_q_type),                                                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weights_data),          weights_dims,          nullptr, weights_rank,
                          static_cast<FTrainNumericType>(weights_type),                                                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_k_data),          index_k_dims,          nullptr, index_k_rank,
                          static_cast<FTrainNumericType>(index_k_type),                                                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_score_data),       attn_score_dims,       nullptr, attn_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(attn_l1norm_data),      attn_l1norm_dims,      nullptr, attn_l1norm_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_score_data),      index_score_dims,      nullptr, index_score_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(index_lse_data),        index_lse_dims,        nullptr, index_lse_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_q_data),     cu_seqlens_q_dims,     nullptr, cu_seqlens_q_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_k_data),     cu_seqlens_k_dims,     nullptr, cu_seqlens_k_rank,
                          FTRAIN_NUMERIC_TYPE_INT32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(q_causal_offsets_data), q_causal_offsets_dims, nullptr,
                          q_causal_offsets_rank,                                                                       FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_index_score_data), grad_index_score_dims, nullptr,
                          grad_index_score_rank,                                                                       FTRAIN_NUMERIC_TYPE_FP32,  FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dq_data),               dq_dims,               nullptr, dq_rank,
                          static_cast<FTrainNumericType>(dq_type),                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dw_data),               dw_dims,               nullptr, dw_rank,
                          static_cast<FTrainNumericType>(dw_type),                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(dk_data),               dk_dims,               nullptr, dk_rank,
                          static_cast<FTrainNumericType>(dk_type),                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&sm_scale_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&loss_coeff_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&grad_loss_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64)
    };
    FTrainIndexerBwdDenseThdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddIndexerBwdDenseThd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13], ids[14], ids[15], ids[16], ids[17]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetIndexerBwdDenseThd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16CSACompressorFwd(const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank,
                                       const void* score_data, const std::int64_t* score_dims, std::uint8_t score_rank,
                                       const void* ape_data, const std::int64_t* ape_dims, std::uint8_t ape_rank,
                                       const void* cu_seqlens_data, const std::int64_t* cu_seqlens_dims,
                                       std::uint8_t cu_seqlens_rank, const void* cu_seqlens_comp_data,
                                       const std::int64_t* cu_seqlens_comp_dims, std::uint8_t cu_seqlens_comp_rank,
                                       const void* compressed_data, const std::int64_t* compressed_dims,
                                       std::uint8_t compressed_rank, int64_t ratio, int64_t head_dim, int64_t coff,
                                       int64_t total_comp, void* stream) {
    const std::int64_t ratio_storage      = ratio;
    const std::int64_t head_dim_storage   = head_dim;
    const std::int64_t coff_storage       = coff;
    const std::int64_t total_comp_storage = total_comp;
    return static_cast<unsigned char>(ftrainCSACompressorFwd(
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(score_data), score_dims, nullptr, score_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(ape_data), ape_dims, nullptr, ape_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_data), cu_seqlens_dims, nullptr, cu_seqlens_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_comp_data), cu_seqlens_comp_dims, nullptr, cu_seqlens_comp_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(compressed_data), compressed_dims, nullptr, compressed_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&head_dim_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&coff_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&total_comp_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_BF16, nullptr, 0,
        static_cast<FTrainStream>(stream)));
}
unsigned char callBf16CSACompressorBwd(
    const void* kv_data, const std::int64_t* kv_dims, std::uint8_t kv_rank, const void* score_data,
    const std::int64_t* score_dims, std::uint8_t score_rank, const void* ape_data, const std::int64_t* ape_dims,
    std::uint8_t ape_rank, const void* cu_seqlens_data, const std::int64_t* cu_seqlens_dims,
    std::uint8_t cu_seqlens_rank, const void* cu_seqlens_comp_data, const std::int64_t* cu_seqlens_comp_dims,
    std::uint8_t cu_seqlens_comp_rank, const void* grad_compressed_data, const std::int64_t* grad_compressed_dims,
    std::uint8_t grad_compressed_rank, const void* grad_kv_data, const std::int64_t* grad_kv_dims,
    std::uint8_t grad_kv_rank, const void* grad_score_data, const std::int64_t* grad_score_dims,
    std::uint8_t grad_score_rank, const void* grad_ape_data, const std::int64_t* grad_ape_dims,
    std::uint8_t grad_ape_rank, int64_t ratio, int64_t head_dim, int64_t coff, void* stream) {
    const std::int64_t ratio_storage    = ratio;
    const std::int64_t head_dim_storage = head_dim;
    const std::int64_t coff_storage     = coff;
    return static_cast<unsigned char>(ftrainCSACompressorBwd(
        FTrainStorageView{const_cast<void*>(kv_data), kv_dims, nullptr, kv_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(score_data), score_dims, nullptr, score_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(ape_data), ape_dims, nullptr, ape_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_data), cu_seqlens_dims, nullptr, cu_seqlens_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(cu_seqlens_comp_data), cu_seqlens_comp_dims, nullptr, cu_seqlens_comp_rank,
                          FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_compressed_data), grad_compressed_dims, nullptr, grad_compressed_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_kv_data), grad_kv_dims, nullptr, grad_kv_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_score_data), grad_score_dims, nullptr, grad_score_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_ape_data), grad_ape_dims, nullptr, grad_ape_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&ratio_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&head_dim_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&coff_storage, FTRAIN_NUMERIC_TYPE_INT64), FTRAIN_NUMERIC_TYPE_BF16, nullptr, 0,
        static_cast<FTrainStream>(stream)));
}
unsigned char callBf16SinkhornFwd(const void* logits_data, const std::int64_t* logits_dims, std::uint8_t logits_rank,
                                  const void* output_data, const std::int64_t* output_dims, std::uint8_t output_rank,
                                  const void* m_init_data, const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  int64_t num_iterations, double eps, void* stream) {
    const std::int64_t num_iterations_storage = num_iterations;
    const float eps_storage                   = static_cast<float>(eps);
    return static_cast<unsigned char>(
        ftrainSinkhornFwd(FTrainStorageView{const_cast<void*>(logits_data), logits_dims, nullptr, logits_rank,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(output_data), output_dims, nullptr, output_rank,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(m_init_data), m_init_dims, nullptr, m_init_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          makeHostScalarView(&num_iterations_storage, FTRAIN_NUMERIC_TYPE_INT64),
                          makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_BF16, nullptr,
                          0, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16SinkhornFwd(const void* logits_data, const std::int64_t* logits_dims, std::uint8_t logits_rank,
                                  const void* output_data, const std::int64_t* output_dims, std::uint8_t output_rank,
                                  const void* m_init_data, const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  int64_t num_iterations, double eps, void* stream) {
    const std::int64_t num_iterations_storage = num_iterations;
    const float eps_storage                   = static_cast<float>(eps);
    return static_cast<unsigned char>(
        ftrainSinkhornFwd(FTrainStorageView{const_cast<void*>(logits_data), logits_dims, nullptr, logits_rank,
                                            FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(output_data), output_dims, nullptr, output_rank,
                                            FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(m_init_data), m_init_dims, nullptr, m_init_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          makeHostScalarView(&num_iterations_storage, FTRAIN_NUMERIC_TYPE_INT64),
                          makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP16, nullptr,
                          0, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32SinkhornFwd(const void* logits_data, const std::int64_t* logits_dims, std::uint8_t logits_rank,
                                  const void* output_data, const std::int64_t* output_dims, std::uint8_t output_rank,
                                  const void* m_init_data, const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  int64_t num_iterations, double eps, void* stream) {
    const std::int64_t num_iterations_storage = num_iterations;
    const float eps_storage                   = static_cast<float>(eps);
    return static_cast<unsigned char>(
        ftrainSinkhornFwd(FTrainStorageView{const_cast<void*>(logits_data), logits_dims, nullptr, logits_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(output_data), output_dims, nullptr, output_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(m_init_data), m_init_dims, nullptr, m_init_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          makeHostScalarView(&num_iterations_storage, FTRAIN_NUMERIC_TYPE_INT64),
                          makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, nullptr,
                          0, static_cast<FTrainStream>(stream)));
}
unsigned char callBf16SinkhornBwd(const void* grad_output_data, const std::int64_t* grad_output_dims,
                                  std::uint8_t grad_output_rank, const void* m_init_data,
                                  const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  const void* grad_logits_data, const std::int64_t* grad_logits_dims,
                                  std::uint8_t grad_logits_rank, int64_t num_iterations, double eps, void* stream) {
    const std::int64_t num_iterations_storage = num_iterations;
    const float eps_storage                   = static_cast<float>(eps);
    return static_cast<unsigned char>(ftrainSinkhornBwd(
        FTrainStorageView{const_cast<void*>(grad_output_data), grad_output_dims, nullptr, grad_output_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(m_init_data), m_init_dims, nullptr, m_init_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_logits_data), grad_logits_dims, nullptr, grad_logits_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&num_iterations_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_BF16, nullptr, 0,
        static_cast<FTrainStream>(stream)));
}
unsigned char callFp16SinkhornBwd(const void* grad_output_data, const std::int64_t* grad_output_dims,
                                  std::uint8_t grad_output_rank, const void* m_init_data,
                                  const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  const void* grad_logits_data, const std::int64_t* grad_logits_dims,
                                  std::uint8_t grad_logits_rank, int64_t num_iterations, double eps, void* stream) {
    const std::int64_t num_iterations_storage = num_iterations;
    const float eps_storage                   = static_cast<float>(eps);
    return static_cast<unsigned char>(ftrainSinkhornBwd(
        FTrainStorageView{const_cast<void*>(grad_output_data), grad_output_dims, nullptr, grad_output_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(m_init_data), m_init_dims, nullptr, m_init_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_logits_data), grad_logits_dims, nullptr, grad_logits_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&num_iterations_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP16, nullptr, 0,
        static_cast<FTrainStream>(stream)));
}
unsigned char callFp32SinkhornBwd(const void* grad_output_data, const std::int64_t* grad_output_dims,
                                  std::uint8_t grad_output_rank, const void* m_init_data,
                                  const std::int64_t* m_init_dims, std::uint8_t m_init_rank,
                                  const void* grad_logits_data, const std::int64_t* grad_logits_dims,
                                  std::uint8_t grad_logits_rank, int64_t num_iterations, double eps, void* stream) {
    const std::int64_t num_iterations_storage = num_iterations;
    const float eps_storage                   = static_cast<float>(eps);
    return static_cast<unsigned char>(ftrainSinkhornBwd(
        FTrainStorageView{const_cast<void*>(grad_output_data), grad_output_dims, nullptr, grad_output_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(m_init_data), m_init_dims, nullptr, m_init_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_logits_data), grad_logits_dims, nullptr, grad_logits_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&num_iterations_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0,
        static_cast<FTrainStream>(stream)));
}
unsigned char callFp32HAggregateFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                    const void* h_pre_data, const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank,
                                    const void* aggregated_data, const std::int64_t* aggregated_dims,
                                    std::uint8_t aggregated_rank, void* stream) {
    return static_cast<unsigned char>(ftrainHAggregateFwd(
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(aggregated_data), aggregated_dims, nullptr, aggregated_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32HAggregateFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                         std::uint8_t x_type, const void* h_pre_data, const std::int64_t* h_pre_dims,
                                         std::uint8_t h_pre_rank, const void* aggregated_data,
                                         const std::int64_t* aggregated_dims, std::uint8_t aggregated_rank,
                                         std::uint8_t aggregated_type, void* stream) {
    return static_cast<unsigned char>(ftrainHAggregateFwd(
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, static_cast<FTrainNumericType>(x_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(aggregated_data), aggregated_dims, nullptr, aggregated_rank,
                          static_cast<FTrainNumericType>(aggregated_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32HAggregateBwd(const void* grad_aggregated_data, const std::int64_t* grad_aggregated_dims,
                                    std::uint8_t grad_aggregated_rank, const void* x_data, const std::int64_t* x_dims,
                                    std::uint8_t x_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
                                    std::uint8_t h_pre_rank, const void* grad_x_data, const std::int64_t* grad_x_dims,
                                    std::uint8_t grad_x_rank, const void* grad_h_pre_data,
                                    const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank, void* stream) {
    return static_cast<unsigned char>(ftrainHAggregateBwd(
        FTrainStorageView{const_cast<void*>(grad_aggregated_data), grad_aggregated_dims, nullptr, grad_aggregated_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_pre_data), grad_h_pre_dims, nullptr, grad_h_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32HAggregateBwd(
    const void* grad_aggregated_data, const std::int64_t* grad_aggregated_dims, std::uint8_t grad_aggregated_rank,
    std::uint8_t grad_aggregated_type, const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
    std::uint8_t x_type, const void* h_pre_data, const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank,
    const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, std::uint8_t grad_x_type,
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank, void* stream) {
    return static_cast<unsigned char>(ftrainHAggregateBwd(
        FTrainStorageView{const_cast<void*>(grad_aggregated_data), grad_aggregated_dims, nullptr, grad_aggregated_rank,
                          static_cast<FTrainNumericType>(grad_aggregated_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, static_cast<FTrainNumericType>(x_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank,
                          static_cast<FTrainNumericType>(grad_x_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_pre_data), grad_h_pre_dims, nullptr, grad_h_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0, static_cast<FTrainStream>(stream)));
}
unsigned char callBf16HPostBdaFwd(const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
                                  const void* residual_data, const std::int64_t* residual_dims,
                                  std::uint8_t residual_rank, const void* h_post_data, const std::int64_t* h_post_dims,
                                  std::uint8_t h_post_rank, const void* x_data, const std::int64_t* x_dims,
                                  std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
                                  std::uint8_t bias_rank, const void* output_data, const std::int64_t* output_dims,
                                  std::uint8_t output_rank, void* stream) {
    return static_cast<unsigned char>(
        ftrainHPostBdaFwd(FTrainStorageView{const_cast<void*>(h_res_data), h_res_dims, nullptr, h_res_rank,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(residual_data), residual_dims, nullptr, residual_rank,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(bias_data), bias_dims, nullptr, bias_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(output_data), output_dims, nullptr, output_rank,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTRAIN_NUMERIC_TYPE_BF16, nullptr, 0, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16HPostBdaFwd(const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
                                  const void* residual_data, const std::int64_t* residual_dims,
                                  std::uint8_t residual_rank, const void* h_post_data, const std::int64_t* h_post_dims,
                                  std::uint8_t h_post_rank, const void* x_data, const std::int64_t* x_dims,
                                  std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
                                  std::uint8_t bias_rank, const void* output_data, const std::int64_t* output_dims,
                                  std::uint8_t output_rank, void* stream) {
    return static_cast<unsigned char>(
        ftrainHPostBdaFwd(FTrainStorageView{const_cast<void*>(h_res_data), h_res_dims, nullptr, h_res_rank,
                                            FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(residual_data), residual_dims, nullptr, residual_rank,
                                            FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank,
                                            FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank,
                                            FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(bias_data), bias_dims, nullptr, bias_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(output_data), output_dims, nullptr, output_rank,
                                            FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTRAIN_NUMERIC_TYPE_FP16, nullptr, 0, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32HPostBdaFwd(const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
                                  const void* residual_data, const std::int64_t* residual_dims,
                                  std::uint8_t residual_rank, const void* h_post_data, const std::int64_t* h_post_dims,
                                  std::uint8_t h_post_rank, const void* x_data, const std::int64_t* x_dims,
                                  std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
                                  std::uint8_t bias_rank, const void* output_data, const std::int64_t* output_dims,
                                  std::uint8_t output_rank, void* stream) {
    return static_cast<unsigned char>(
        ftrainHPostBdaFwd(FTrainStorageView{const_cast<void*>(h_res_data), h_res_dims, nullptr, h_res_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(residual_data), residual_dims, nullptr, residual_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(bias_data), bias_dims, nullptr, bias_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTrainStorageView{const_cast<void*>(output_data), output_dims, nullptr, output_rank,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                          FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0, static_cast<FTrainStream>(stream)));
}
unsigned char callBf16HPostBdaBwd(
    const void* grad_output_data, const std::int64_t* grad_output_dims, std::uint8_t grad_output_rank,
    const void* residual_data, const std::int64_t* residual_dims, std::uint8_t residual_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
    std::uint8_t bias_rank, const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
    const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank, const void* grad_residual_data,
    const std::int64_t* grad_residual_dims, std::uint8_t grad_residual_rank, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, const void* grad_h_res_data,
    const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank, const void* grad_h_post_data,
    const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank, const void* grad_bias_data,
    const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, void* stream) {
    return static_cast<unsigned char>(ftrainHPostBdaBwd(
        FTrainStorageView{const_cast<void*>(grad_output_data), grad_output_dims, nullptr, grad_output_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(residual_data), residual_dims, nullptr, residual_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(bias_data), bias_dims, nullptr, bias_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_res_data), h_res_dims, nullptr, h_res_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_residual_data), grad_residual_dims, nullptr, grad_residual_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_res_data), grad_h_res_dims, nullptr, grad_h_res_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_post_data), grad_h_post_dims, nullptr, grad_h_post_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_bias_data), grad_bias_dims, nullptr, grad_bias_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTRAIN_NUMERIC_TYPE_BF16, nullptr, 0, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16HPostBdaBwd(
    const void* grad_output_data, const std::int64_t* grad_output_dims, std::uint8_t grad_output_rank,
    const void* residual_data, const std::int64_t* residual_dims, std::uint8_t residual_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
    std::uint8_t bias_rank, const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
    const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank, const void* grad_residual_data,
    const std::int64_t* grad_residual_dims, std::uint8_t grad_residual_rank, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, const void* grad_h_res_data,
    const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank, const void* grad_h_post_data,
    const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank, const void* grad_bias_data,
    const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, void* stream) {
    return static_cast<unsigned char>(ftrainHPostBdaBwd(
        FTrainStorageView{const_cast<void*>(grad_output_data), grad_output_dims, nullptr, grad_output_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(residual_data), residual_dims, nullptr, residual_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(bias_data), bias_dims, nullptr, bias_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_res_data), h_res_dims, nullptr, h_res_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_residual_data), grad_residual_dims, nullptr, grad_residual_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_res_data), grad_h_res_dims, nullptr, grad_h_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_post_data), grad_h_post_dims, nullptr, grad_h_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_bias_data), grad_bias_dims, nullptr, grad_bias_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTRAIN_NUMERIC_TYPE_FP16, nullptr, 0, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32HPostBdaBwd(
    const void* grad_output_data, const std::int64_t* grad_output_dims, std::uint8_t grad_output_rank,
    const void* residual_data, const std::int64_t* residual_dims, std::uint8_t residual_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* bias_data, const std::int64_t* bias_dims,
    std::uint8_t bias_rank, const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank,
    const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank, const void* grad_residual_data,
    const std::int64_t* grad_residual_dims, std::uint8_t grad_residual_rank, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, const void* grad_h_res_data,
    const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank, const void* grad_h_post_data,
    const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank, const void* grad_bias_data,
    const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, void* stream) {
    return static_cast<unsigned char>(ftrainHPostBdaBwd(
        FTrainStorageView{const_cast<void*>(grad_output_data), grad_output_dims, nullptr, grad_output_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(residual_data), residual_dims, nullptr, residual_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(bias_data), bias_dims, nullptr, bias_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_res_data), h_res_dims, nullptr, h_res_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_residual_data), grad_residual_dims, nullptr, grad_residual_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_res_data), grad_h_res_dims, nullptr, grad_h_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_post_data), grad_h_post_dims, nullptr, grad_h_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_bias_data), grad_bias_dims, nullptr, grad_bias_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0, static_cast<FTrainStream>(stream)));
}
unsigned char callBf16ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                 const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                 const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                 const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                 void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float eps_storage = static_cast<float>(eps);
    return static_cast<unsigned char>(
        ftrainProjRmsFwd(FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank,
                                           FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank,
                                           FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_BF16,
                         workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                 const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                 const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                 const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                 void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float eps_storage = static_cast<float>(eps);
    return static_cast<unsigned char>(
        ftrainProjRmsFwd(FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank,
                                           FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank,
                                           FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, FTRAIN_NUMERIC_TYPE_FP16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP16,
                         workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                 const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                 const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                 const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                 void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const float eps_storage = static_cast<float>(eps);
    return static_cast<unsigned char>(
        ftrainProjRmsFwd(FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank,
                                           FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank,
                                           FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
                         makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32,
                         workspace, workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                      std::uint8_t x_type, const void* weight_data, const std::int64_t* weight_dims,
                                      std::uint8_t weight_rank, std::uint8_t weight_type, const void* proj_data,
                                      const std::int64_t* proj_dims, std::uint8_t proj_rank, std::uint8_t proj_type,
                                      const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank,
                                      std::uint8_t r_type, double eps, void* workspace, std::uint64_t workspace_bytes,
                                      void* stream) {
    const float eps_storage = static_cast<float>(eps);
    return static_cast<unsigned char>(ftrainProjRmsFwd(
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, static_cast<FTrainNumericType>(x_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank,
                          static_cast<FTrainNumericType>(weight_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank,
                          static_cast<FTrainNumericType>(proj_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, static_cast<FTrainNumericType>(r_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                  const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                  const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                  const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                  std::uint64_t* workspace_bytes) {
    const float eps_storage = static_cast<float>(eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(x_data),      x_dims,      nullptr, x_rank,      FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data),   proj_dims,   nullptr, proj_rank,   FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data),      r_dims,      nullptr, r_rank,      FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsFwd(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                  const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                  const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                  const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                  std::uint64_t* workspace_bytes) {
    const float eps_storage = static_cast<float>(eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(x_data),      x_dims,      nullptr, x_rank,      FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data),   proj_dims,   nullptr, proj_rank,   FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data),      r_dims,      nullptr, r_rank,      FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                  const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank,
                                  const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
                                  const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, double eps,
                                  std::uint64_t* workspace_bytes) {
    const float eps_storage = static_cast<float>(eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(x_data),      x_dims,      nullptr, x_rank,      FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data),   proj_dims,   nullptr, proj_rank,   FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data),      r_dims,      nullptr, r_rank,      FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32ProjRmsFwd(const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank,
                                       std::uint8_t x_type, const void* weight_data, const std::int64_t* weight_dims,
                                       std::uint8_t weight_rank, std::uint8_t weight_type, const void* proj_data,
                                       const std::int64_t* proj_dims, std::uint8_t proj_rank, std::uint8_t proj_type,
                                       const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank,
                                       std::uint8_t r_type, double eps, std::uint64_t* workspace_bytes) {
    const float eps_storage = static_cast<float>(eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(x_data),      x_dims,      nullptr, x_rank,      static_cast<FTrainNumericType>(x_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                            false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank,
                          static_cast<FTrainNumericType>(weight_type),                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data),   proj_dims,   nullptr, proj_rank,
                          static_cast<FTrainNumericType>(proj_type),                                                                 FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data),      r_dims,      nullptr, r_rank,      static_cast<FTrainNumericType>(r_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                            false},
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16ProjRmsBwd(const void* grad_proj_data, const std::int64_t* grad_proj_dims,
                                 std::uint8_t grad_proj_rank, const void* grad_r_data, const std::int64_t* grad_r_dims,
                                 std::uint8_t grad_r_rank, const void* x_data, const std::int64_t* x_dims,
                                 std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
                                 std::uint8_t weight_rank, const void* r_data, const std::int64_t* r_dims,
                                 std::uint8_t r_rank, const void* grad_x_data, const std::int64_t* grad_x_dims,
                                 std::uint8_t grad_x_rank, const void* grad_weight_data,
                                 const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, double eps,
                                 void* stream) {
    const float eps_storage = static_cast<float>(eps);
    return static_cast<unsigned char>(ftrainProjRmsBwd(
        FTrainStorageView{const_cast<void*>(grad_proj_data), grad_proj_dims, nullptr, grad_proj_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data), grad_r_dims, nullptr, grad_r_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_BF16, nullptr, 0,
        static_cast<FTrainStream>(stream)));
}
unsigned char callFp16ProjRmsBwd(const void* grad_proj_data, const std::int64_t* grad_proj_dims,
                                 std::uint8_t grad_proj_rank, const void* grad_r_data, const std::int64_t* grad_r_dims,
                                 std::uint8_t grad_r_rank, const void* x_data, const std::int64_t* x_dims,
                                 std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
                                 std::uint8_t weight_rank, const void* r_data, const std::int64_t* r_dims,
                                 std::uint8_t r_rank, const void* grad_x_data, const std::int64_t* grad_x_dims,
                                 std::uint8_t grad_x_rank, const void* grad_weight_data,
                                 const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, double eps,
                                 void* stream) {
    const float eps_storage = static_cast<float>(eps);
    return static_cast<unsigned char>(ftrainProjRmsBwd(
        FTrainStorageView{const_cast<void*>(grad_proj_data), grad_proj_dims, nullptr, grad_proj_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data), grad_r_dims, nullptr, grad_r_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP16, nullptr, 0,
        static_cast<FTrainStream>(stream)));
}
unsigned char callFp32ProjRmsBwd(const void* grad_proj_data, const std::int64_t* grad_proj_dims,
                                 std::uint8_t grad_proj_rank, const void* grad_r_data, const std::int64_t* grad_r_dims,
                                 std::uint8_t grad_r_rank, const void* x_data, const std::int64_t* x_dims,
                                 std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
                                 std::uint8_t weight_rank, const void* r_data, const std::int64_t* r_dims,
                                 std::uint8_t r_rank, const void* grad_x_data, const std::int64_t* grad_x_dims,
                                 std::uint8_t grad_x_rank, const void* grad_weight_data,
                                 const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, double eps,
                                 void* stream) {
    const float eps_storage = static_cast<float>(eps);
    return static_cast<unsigned char>(ftrainProjRmsBwd(
        FTrainStorageView{const_cast<void*>(grad_proj_data), grad_proj_dims, nullptr, grad_proj_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data), grad_r_dims, nullptr, grad_r_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0,
        static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32ProjRmsBwd(
    const void* grad_proj_data, const std::int64_t* grad_proj_dims, std::uint8_t grad_proj_rank,
    std::uint8_t grad_proj_type, const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank,
    std::uint8_t grad_r_type, const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, std::uint8_t x_type,
    const void* weight_data, const std::int64_t* weight_dims, std::uint8_t weight_rank, std::uint8_t weight_type,
    const void* r_data, const std::int64_t* r_dims, std::uint8_t r_rank, std::uint8_t r_type, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, std::uint8_t grad_x_type, const void* grad_weight_data,
    const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, std::uint8_t grad_weight_type, double eps,
    void* stream) {
    const float eps_storage = static_cast<float>(eps);
    return static_cast<unsigned char>(ftrainProjRmsBwd(
        FTrainStorageView{const_cast<void*>(grad_proj_data), grad_proj_dims, nullptr, grad_proj_rank,
                          static_cast<FTrainNumericType>(grad_proj_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data), grad_r_dims, nullptr, grad_r_rank,
                          static_cast<FTrainNumericType>(grad_r_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, static_cast<FTrainNumericType>(x_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank,
                          static_cast<FTrainNumericType>(weight_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, static_cast<FTrainNumericType>(r_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank,
                          static_cast<FTrainNumericType>(grad_x_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          static_cast<FTrainNumericType>(grad_weight_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0,
        static_cast<FTrainStream>(stream)));
}
unsigned char callBf16ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, void* workspace, std::uint64_t workspace_bytes,
    void* stream) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    return static_cast<unsigned char>(ftrainProjRmsComputeHFwd(
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data), alpha_pre_dims, nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data), alpha_res_dims, nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(bias_data), bias_dims, nullptr, bias_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_res_data), h_res_dims, nullptr, h_res_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_BF16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, void* workspace, std::uint64_t workspace_bytes,
    void* stream) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    return static_cast<unsigned char>(ftrainProjRmsComputeHFwd(
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data), alpha_pre_dims, nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data), alpha_res_dims, nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(bias_data), bias_dims, nullptr, bias_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_res_data), h_res_dims, nullptr, h_res_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, void* workspace, std::uint64_t workspace_bytes,
    void* stream) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    return static_cast<unsigned char>(ftrainProjRmsComputeHFwd(
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data), alpha_pre_dims, nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data), alpha_res_dims, nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(bias_data), bias_dims, nullptr, bias_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_res_data), h_res_dims, nullptr, h_res_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, std::uint8_t x_type, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, std::uint8_t weight_type, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, std::uint8_t h_pre_type, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, std::uint8_t h_post_type, const void* h_res_data, const std::int64_t* h_res_dims,
    std::uint8_t h_res_rank, std::uint8_t h_res_type, const void* r_data, const std::int64_t* r_dims,
    std::uint8_t r_rank, std::uint8_t r_type, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, std::uint8_t proj_type, int64_t n, double eps, double compute_h_eps, void* workspace,
    std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    return static_cast<unsigned char>(ftrainProjRmsComputeHFwd(
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, static_cast<FTrainNumericType>(x_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank,
                          static_cast<FTrainNumericType>(weight_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data), alpha_pre_dims, nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data), alpha_res_dims, nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(bias_data), bias_dims, nullptr, bias_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank,
                          static_cast<FTrainNumericType>(h_pre_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank,
                          static_cast<FTrainNumericType>(h_post_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_res_data), h_res_dims, nullptr, h_res_rank,
                          static_cast<FTrainNumericType>(h_res_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data), r_dims, nullptr, r_rank, static_cast<FTrainNumericType>(r_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank,
                          static_cast<FTrainNumericType>(proj_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, std::uint64_t* workspace_bytes) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(x_data),          x_dims,          nullptr, x_rank,          FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(weight_data),     weight_dims,     nullptr, weight_rank,     FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data),  alpha_pre_dims,  nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data),  alpha_res_dims,  nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(bias_data),       bias_dims,       nullptr, bias_rank,       FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(h_pre_data),      h_pre_dims,      nullptr, h_pre_rank,      FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(h_post_data),     h_post_dims,     nullptr, h_post_rank,     FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(h_res_data),      h_res_dims,      nullptr, h_res_rank,      FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(r_data),          r_dims,          nullptr, r_rank,          FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(proj_data),       proj_dims,       nullptr, proj_rank,       FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsComputeHFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsComputeHFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsComputeHFwd(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, std::uint64_t* workspace_bytes) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(x_data),          x_dims,          nullptr, x_rank,          FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(weight_data),     weight_dims,     nullptr, weight_rank,     FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data),  alpha_pre_dims,  nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data),  alpha_res_dims,  nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(bias_data),       bias_dims,       nullptr, bias_rank,       FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(h_pre_data),      h_pre_dims,      nullptr, h_pre_rank,      FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(h_post_data),     h_post_dims,     nullptr, h_post_rank,     FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(h_res_data),      h_res_dims,      nullptr, h_res_rank,      FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(r_data),          r_dims,          nullptr, r_rank,          FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(proj_data),       proj_dims,       nullptr, proj_rank,       FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsComputeHFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsComputeHFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsComputeHFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank,
    const void* h_res_data, const std::int64_t* h_res_dims, std::uint8_t h_res_rank, const void* r_data,
    const std::int64_t* r_dims, std::uint8_t r_rank, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, int64_t n, double eps, double compute_h_eps, std::uint64_t* workspace_bytes) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(x_data),          x_dims,          nullptr, x_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(weight_data),     weight_dims,     nullptr, weight_rank,     FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data),  alpha_pre_dims,  nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data),  alpha_res_dims,  nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(bias_data),       bias_dims,       nullptr, bias_rank,       FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(h_pre_data),      h_pre_dims,      nullptr, h_pre_rank,      FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(h_post_data),     h_post_dims,     nullptr, h_post_rank,     FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(h_res_data),      h_res_dims,      nullptr, h_res_rank,      FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(r_data),          r_dims,          nullptr, r_rank,          FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        FTrainStorageView{const_cast<void*>(proj_data),       proj_dims,       nullptr, proj_rank,       FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                          false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsComputeHFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsComputeHFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsComputeHFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32ProjRmsComputeHFwd(
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, std::uint8_t x_type, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, std::uint8_t weight_type, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* bias_data,
    const std::int64_t* bias_dims, std::uint8_t bias_rank, const void* h_pre_data, const std::int64_t* h_pre_dims,
    std::uint8_t h_pre_rank, std::uint8_t h_pre_type, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, std::uint8_t h_post_type, const void* h_res_data, const std::int64_t* h_res_dims,
    std::uint8_t h_res_rank, std::uint8_t h_res_type, const void* r_data, const std::int64_t* r_dims,
    std::uint8_t r_rank, std::uint8_t r_type, const void* proj_data, const std::int64_t* proj_dims,
    std::uint8_t proj_rank, std::uint8_t proj_type, int64_t n, double eps, double compute_h_eps,
    std::uint64_t* workspace_bytes) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(x_data),          x_dims,          nullptr, x_rank,          static_cast<FTrainNumericType>(x_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                        false},
        FTrainStorageView{const_cast<void*>(weight_data),     weight_dims,     nullptr, weight_rank,
                          static_cast<FTrainNumericType>(weight_type),                                                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data),  alpha_pre_dims,  nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data),  alpha_res_dims,  nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(bias_data),       bias_dims,       nullptr, bias_rank,       FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                        false},
        FTrainStorageView{const_cast<void*>(h_pre_data),      h_pre_dims,      nullptr, h_pre_rank,
                          static_cast<FTrainNumericType>(h_pre_type),                                                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data),     h_post_dims,     nullptr, h_post_rank,
                          static_cast<FTrainNumericType>(h_post_type),                                                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_res_data),      h_res_dims,      nullptr, h_res_rank,
                          static_cast<FTrainNumericType>(h_res_type),                                                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_data),          r_dims,          nullptr, r_rank,          static_cast<FTrainNumericType>(r_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                        false},
        FTrainStorageView{const_cast<void*>(proj_data),       proj_dims,       nullptr, proj_rank,
                          static_cast<FTrainNumericType>(proj_type),                                                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsComputeHFwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsComputeHFwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsComputeHFwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char callBf16ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    return static_cast<unsigned char>(ftrainProjRmsComputeHBwd(
        FTrainStorageView{const_cast<void*>(grad_h_pre_data), grad_h_pre_dims, nullptr, grad_h_pre_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_post_data), grad_h_post_dims, nullptr, grad_h_post_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_res_data), grad_h_res_dims, nullptr, grad_h_res_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data), grad_r_dims, nullptr, grad_r_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_ch_data), r_ch_dims, nullptr, r_ch_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data), alpha_pre_dims, nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data), alpha_res_dims, nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank, FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_alpha_data), grad_alpha_dims, nullptr, grad_alpha_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_bias_data), grad_bias_dims, nullptr, grad_bias_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_BF16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp16ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    return static_cast<unsigned char>(ftrainProjRmsComputeHBwd(
        FTrainStorageView{const_cast<void*>(grad_h_pre_data), grad_h_pre_dims, nullptr, grad_h_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_post_data), grad_h_post_dims, nullptr, grad_h_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_res_data), grad_h_res_dims, nullptr, grad_h_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data), grad_r_dims, nullptr, grad_r_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_ch_data), r_ch_dims, nullptr, r_ch_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data), alpha_pre_dims, nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data), alpha_res_dims, nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank, FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_alpha_data), grad_alpha_dims, nullptr, grad_alpha_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_bias_data), grad_bias_dims, nullptr, grad_bias_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP16, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callFp32ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    return static_cast<unsigned char>(ftrainProjRmsComputeHBwd(
        FTrainStorageView{const_cast<void*>(grad_h_pre_data), grad_h_pre_dims, nullptr, grad_h_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_post_data), grad_h_post_dims, nullptr, grad_h_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_res_data), grad_h_res_dims, nullptr, grad_h_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data), grad_r_dims, nullptr, grad_r_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_ch_data), r_ch_dims, nullptr, r_ch_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data), alpha_pre_dims, nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data), alpha_res_dims, nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank, FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_alpha_data), grad_alpha_dims, nullptr, grad_alpha_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_bias_data), grad_bias_dims, nullptr, grad_bias_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char callMixedFp32ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    std::uint8_t grad_h_pre_type, const void* grad_h_post_data, const std::int64_t* grad_h_post_dims,
    std::uint8_t grad_h_post_rank, std::uint8_t grad_h_post_type, const void* grad_h_res_data,
    const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank, std::uint8_t grad_h_res_type,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, std::uint8_t grad_r_type,
    const void* h_pre_data, const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, std::uint8_t h_pre_type,
    const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank, std::uint8_t h_post_type,
    const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank, std::uint8_t proj_type,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, std::uint8_t r_ch_type,
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, std::uint8_t x_type, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, std::uint8_t weight_type, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, std::uint8_t grad_x_type, const void* grad_weight_data,
    const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, std::uint8_t grad_weight_type,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, void* workspace, std::uint64_t workspace_bytes, void* stream) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    return static_cast<unsigned char>(ftrainProjRmsComputeHBwd(
        FTrainStorageView{const_cast<void*>(grad_h_pre_data), grad_h_pre_dims, nullptr, grad_h_pre_rank,
                          static_cast<FTrainNumericType>(grad_h_pre_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_post_data), grad_h_post_dims, nullptr, grad_h_post_rank,
                          static_cast<FTrainNumericType>(grad_h_post_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_res_data), grad_h_res_dims, nullptr, grad_h_res_rank,
                          static_cast<FTrainNumericType>(grad_h_res_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data), grad_r_dims, nullptr, grad_r_rank,
                          static_cast<FTrainNumericType>(grad_r_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data), h_pre_dims, nullptr, h_pre_rank,
                          static_cast<FTrainNumericType>(h_pre_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data), h_post_dims, nullptr, h_post_rank,
                          static_cast<FTrainNumericType>(h_post_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data), proj_dims, nullptr, proj_rank,
                          static_cast<FTrainNumericType>(proj_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_ch_data), r_ch_dims, nullptr, r_ch_rank,
                          static_cast<FTrainNumericType>(r_ch_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data), x_dims, nullptr, x_rank, static_cast<FTrainNumericType>(x_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(weight_data), weight_dims, nullptr, weight_rank,
                          static_cast<FTrainNumericType>(weight_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data), alpha_pre_dims, nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data), alpha_post_dims, nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data), alpha_res_dims, nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data), grad_x_dims, nullptr, grad_x_rank,
                          static_cast<FTrainNumericType>(grad_x_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          static_cast<FTrainNumericType>(grad_weight_type), FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_alpha_data), grad_alpha_dims, nullptr, grad_alpha_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_bias_data), grad_bias_dims, nullptr, grad_bias_rank,
                          FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, workspace,
        workspace_bytes, static_cast<FTrainStream>(stream)));
}
unsigned char queryBf16ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, std::uint64_t* workspace_bytes) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(grad_h_pre_data),  grad_h_pre_dims,  nullptr, grad_h_pre_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_post_data), grad_h_post_dims, nullptr, grad_h_post_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_res_data),  grad_h_res_dims,  nullptr, grad_h_res_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data),      grad_r_dims,      nullptr, grad_r_rank,      FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(h_pre_data),       h_pre_dims,       nullptr, h_pre_rank,       FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(h_post_data),      h_post_dims,      nullptr, h_post_rank,      FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(proj_data),        proj_dims,        nullptr, proj_rank,        FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(r_ch_data),        r_ch_dims,        nullptr, r_ch_rank,        FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(x_data),           x_dims,           nullptr, x_rank,           FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(weight_data),      weight_dims,      nullptr, weight_rank,      FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data),   alpha_pre_dims,   nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data),  alpha_post_dims,  nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data),   alpha_res_dims,   nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data),      grad_x_dims,      nullptr, grad_x_rank,      FTRAIN_NUMERIC_TYPE_BF16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          FTRAIN_NUMERIC_TYPE_BF16,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_alpha_data),  grad_alpha_dims,  nullptr, grad_alpha_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_bias_data),   grad_bias_dims,   nullptr, grad_bias_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsComputeHBwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsComputeHBwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13], ids[14], ids[15], ids[16], ids[17], ids[18], ids[19]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsComputeHBwd(args, operation, FTRAIN_NUMERIC_TYPE_BF16); },
        workspace_bytes);
}
unsigned char queryFp16ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, std::uint64_t* workspace_bytes) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(grad_h_pre_data),  grad_h_pre_dims,  nullptr, grad_h_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_post_data), grad_h_post_dims, nullptr, grad_h_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_res_data),  grad_h_res_dims,  nullptr, grad_h_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data),      grad_r_dims,      nullptr, grad_r_rank,      FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(h_pre_data),       h_pre_dims,       nullptr, h_pre_rank,       FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(h_post_data),      h_post_dims,      nullptr, h_post_rank,      FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(proj_data),        proj_dims,        nullptr, proj_rank,        FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(r_ch_data),        r_ch_dims,        nullptr, r_ch_rank,        FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(x_data),           x_dims,           nullptr, x_rank,           FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(weight_data),      weight_dims,      nullptr, weight_rank,      FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data),   alpha_pre_dims,   nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data),  alpha_post_dims,  nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data),   alpha_res_dims,   nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data),      grad_x_dims,      nullptr, grad_x_rank,      FTRAIN_NUMERIC_TYPE_FP16,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          FTRAIN_NUMERIC_TYPE_FP16,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_alpha_data),  grad_alpha_dims,  nullptr, grad_alpha_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_bias_data),   grad_bias_dims,   nullptr, grad_bias_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsComputeHBwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsComputeHBwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13], ids[14], ids[15], ids[16], ids[17], ids[18], ids[19]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsComputeHBwd(args, operation, FTRAIN_NUMERIC_TYPE_FP16); },
        workspace_bytes);
}
unsigned char queryFp32ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    const void* grad_h_post_data, const std::int64_t* grad_h_post_dims, std::uint8_t grad_h_post_rank,
    const void* grad_h_res_data, const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, const void* h_pre_data,
    const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, const void* h_post_data, const std::int64_t* h_post_dims,
    std::uint8_t h_post_rank, const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, const void* x_data,
    const std::int64_t* x_dims, std::uint8_t x_rank, const void* weight_data, const std::int64_t* weight_dims,
    std::uint8_t weight_rank, const void* alpha_pre_data, const std::int64_t* alpha_pre_dims,
    std::uint8_t alpha_pre_rank, const void* alpha_post_data, const std::int64_t* alpha_post_dims,
    std::uint8_t alpha_post_rank, const void* alpha_res_data, const std::int64_t* alpha_res_dims,
    std::uint8_t alpha_res_rank, const void* grad_x_data, const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank,
    const void* grad_weight_data, const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, std::uint64_t* workspace_bytes) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(grad_h_pre_data),  grad_h_pre_dims,  nullptr, grad_h_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_post_data), grad_h_post_dims, nullptr, grad_h_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_res_data),  grad_h_res_dims,  nullptr, grad_h_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data),      grad_r_dims,      nullptr, grad_r_rank,      FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(h_pre_data),       h_pre_dims,       nullptr, h_pre_rank,       FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(h_post_data),      h_post_dims,      nullptr, h_post_rank,      FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(proj_data),        proj_dims,        nullptr, proj_rank,        FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(r_ch_data),        r_ch_dims,        nullptr, r_ch_rank,        FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(x_data),           x_dims,           nullptr, x_rank,           FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(weight_data),      weight_dims,      nullptr, weight_rank,      FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data),   alpha_pre_dims,   nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data),  alpha_post_dims,  nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data),   alpha_res_dims,   nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data),      grad_x_dims,      nullptr, grad_x_rank,      FTRAIN_NUMERIC_TYPE_FP32,
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                             false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_alpha_data),  grad_alpha_dims,  nullptr, grad_alpha_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_bias_data),   grad_bias_dims,   nullptr, grad_bias_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                   FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsComputeHBwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsComputeHBwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13], ids[14], ids[15], ids[16], ids[17], ids[18], ids[19]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsComputeHBwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
        workspace_bytes);
}
unsigned char queryMixedFp32ProjRmsComputeHBwd(
    const void* grad_h_pre_data, const std::int64_t* grad_h_pre_dims, std::uint8_t grad_h_pre_rank,
    std::uint8_t grad_h_pre_type, const void* grad_h_post_data, const std::int64_t* grad_h_post_dims,
    std::uint8_t grad_h_post_rank, std::uint8_t grad_h_post_type, const void* grad_h_res_data,
    const std::int64_t* grad_h_res_dims, std::uint8_t grad_h_res_rank, std::uint8_t grad_h_res_type,
    const void* grad_r_data, const std::int64_t* grad_r_dims, std::uint8_t grad_r_rank, std::uint8_t grad_r_type,
    const void* h_pre_data, const std::int64_t* h_pre_dims, std::uint8_t h_pre_rank, std::uint8_t h_pre_type,
    const void* h_post_data, const std::int64_t* h_post_dims, std::uint8_t h_post_rank, std::uint8_t h_post_type,
    const void* proj_data, const std::int64_t* proj_dims, std::uint8_t proj_rank, std::uint8_t proj_type,
    const void* r_ch_data, const std::int64_t* r_ch_dims, std::uint8_t r_ch_rank, std::uint8_t r_ch_type,
    const void* x_data, const std::int64_t* x_dims, std::uint8_t x_rank, std::uint8_t x_type, const void* weight_data,
    const std::int64_t* weight_dims, std::uint8_t weight_rank, std::uint8_t weight_type, const void* alpha_pre_data,
    const std::int64_t* alpha_pre_dims, std::uint8_t alpha_pre_rank, const void* alpha_post_data,
    const std::int64_t* alpha_post_dims, std::uint8_t alpha_post_rank, const void* alpha_res_data,
    const std::int64_t* alpha_res_dims, std::uint8_t alpha_res_rank, const void* grad_x_data,
    const std::int64_t* grad_x_dims, std::uint8_t grad_x_rank, std::uint8_t grad_x_type, const void* grad_weight_data,
    const std::int64_t* grad_weight_dims, std::uint8_t grad_weight_rank, std::uint8_t grad_weight_type,
    const void* grad_alpha_data, const std::int64_t* grad_alpha_dims, std::uint8_t grad_alpha_rank,
    const void* grad_bias_data, const std::int64_t* grad_bias_dims, std::uint8_t grad_bias_rank, int64_t n, double eps,
    double compute_h_eps, std::uint64_t* workspace_bytes) {
    const std::int64_t n_storage      = n;
    const float eps_storage           = static_cast<float>(eps);
    const float compute_h_eps_storage = static_cast<float>(compute_h_eps);
    const FTrainStorageView views[]{
        FTrainStorageView{const_cast<void*>(grad_h_pre_data),  grad_h_pre_dims,  nullptr, grad_h_pre_rank,
                          static_cast<FTrainNumericType>(grad_h_pre_type),                                                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_post_data), grad_h_post_dims, nullptr, grad_h_post_rank,
                          static_cast<FTrainNumericType>(grad_h_post_type),                                                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_h_res_data),  grad_h_res_dims,  nullptr, grad_h_res_rank,
                          static_cast<FTrainNumericType>(grad_h_res_type),                                                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_r_data),      grad_r_dims,      nullptr, grad_r_rank,
                          static_cast<FTrainNumericType>(grad_r_type),                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_pre_data),       h_pre_dims,       nullptr, h_pre_rank,
                          static_cast<FTrainNumericType>(h_pre_type),                                                                               FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(h_post_data),      h_post_dims,      nullptr, h_post_rank,
                          static_cast<FTrainNumericType>(h_post_type),                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(proj_data),        proj_dims,        nullptr, proj_rank,
                          static_cast<FTrainNumericType>(proj_type),                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(r_ch_data),        r_ch_dims,        nullptr, r_ch_rank,
                          static_cast<FTrainNumericType>(r_ch_type),                                                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(x_data),           x_dims,           nullptr, x_rank,           static_cast<FTrainNumericType>(x_type),
                          FTRAIN_INDEX_TYPE_CONTINUOUS,                                                                                                                           false},
        FTrainStorageView{const_cast<void*>(weight_data),      weight_dims,      nullptr, weight_rank,
                          static_cast<FTrainNumericType>(weight_type),                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_pre_data),   alpha_pre_dims,   nullptr, alpha_pre_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                 FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_post_data),  alpha_post_dims,  nullptr, alpha_post_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                 FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(alpha_res_data),   alpha_res_dims,   nullptr, alpha_res_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                 FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_x_data),      grad_x_dims,      nullptr, grad_x_rank,
                          static_cast<FTrainNumericType>(grad_x_type),                                                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_weight_data), grad_weight_dims, nullptr, grad_weight_rank,
                          static_cast<FTrainNumericType>(grad_weight_type),                                                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_alpha_data),  grad_alpha_dims,  nullptr, grad_alpha_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                 FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        FTrainStorageView{const_cast<void*>(grad_bias_data),   grad_bias_dims,   nullptr, grad_bias_rank,
                          FTRAIN_NUMERIC_TYPE_FP32,                                                                                                 FTRAIN_INDEX_TYPE_CONTINUOUS, false},
        makeHostScalarView(&n_storage, FTRAIN_NUMERIC_TYPE_INT64),
        makeHostScalarView(&eps_storage, FTRAIN_NUMERIC_TYPE_FP32),
        makeHostScalarView(&compute_h_eps_storage, FTRAIN_NUMERIC_TYPE_FP32)
    };
    FTrainProjRmsComputeHBwdOpId operation{};
    return queryWorkspace(
        views,
        [&](FTrainPattern pattern, const auto& ids) {
            return ftrainPatternAddProjRmsComputeHBwd(pattern, &operation, ids[0], ids[1], ids[2], ids[3], ids[4],
                                                      ids[5], ids[6], ids[7], ids[8], ids[9], ids[10], ids[11], ids[12],
                                                      ids[13], ids[14], ids[15], ids[16], ids[17], ids[18], ids[19]);
        },
        [&](FTrainArgs args) { return ftrainArgsSetProjRmsComputeHBwd(args, operation, FTRAIN_NUMERIC_TYPE_FP32); },
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
