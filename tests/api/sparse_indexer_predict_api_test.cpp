// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <flash_train/env.h>
#include <flash_train/flash_train.h>

#include "sparse_indexer_predict_staged_fixture.hpp"

namespace {

using ftrain_test::SparseIndexerPredictStagedTest;

FTrainStorageView hostScalar(const void* memory, FTrainNumericType numeric_type) {
    return FTrainStorageView{const_cast<void*>(memory),    nullptr, nullptr, 0, numeric_type,
                             FTRAIN_INDEX_TYPE_CONTINUOUS, true};
}

TEST_F(SparseIndexerPredictStagedTest, RunsTheStagedPathWithItsRowCounterWorkspace) {
    constexpr std::int64_t kBatch = 2, kQueryLength = 8, kKeyLength = 16, kHeads = 4, kHeadDim = 64, kTopK = 6;
    const std::int64_t q_dims[4]    = {kBatch, kQueryLength, kHeads, kHeadDim};
    const std::int64_t k_dims[3]    = {kBatch, kKeyLength, kHeadDim};
    const std::int64_t w_dims[3]    = {kBatch, kQueryLength, kHeads};
    const std::int64_t topk_dims[3] = {kBatch, kQueryLength, kTopK};

    FTrainPattern pattern = nullptr;
    FTrainOps ops         = nullptr;
    ASSERT_EQ(ftrainPatternCreate(&pattern), FTRAIN_STATUS_SUCCESS);
    FTrainTensorId q{}, k{}, w{}, topk{}, predict{}, sm_scale{}, topk_indices_global{};
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &q), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &k), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &w), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &topk), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &predict), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &sm_scale), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &topk_indices_global), FTRAIN_STATUS_SUCCESS);
    FTrainSparseIndexerPredictOpId op{};
    ASSERT_EQ(ftrainPatternAddSparseIndexerPredict(pattern, &op, q, k, w, topk, predict, sm_scale, topk_indices_global),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainOpsCreate(&ops, pattern), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    void* q_memory       = nullptr;
    void* k_memory       = nullptr;
    void* w_memory       = nullptr;
    void* topk_memory    = nullptr;
    void* predict_memory = nullptr;
    ASSERT_EQ(hipMalloc(&q_memory, kBatch * kQueryLength * kHeads * kHeadDim * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&k_memory, kBatch * kKeyLength * kHeadDim * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&w_memory, kBatch * kQueryLength * kHeads * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&topk_memory, kBatch * kQueryLength * kTopK * sizeof(std::int32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&predict_memory, kBatch * kQueryLength * kTopK * sizeof(float)), hipSuccess);

    const std::vector<float> host_q(kBatch * kQueryLength * kHeads * kHeadDim, 0.25F);
    const std::vector<float> host_k(kBatch * kKeyLength * kHeadDim, 0.5F);
    const std::vector<float> host_w(kBatch * kQueryLength * kHeads, 0.75F);
    std::vector<std::int32_t> host_topk(kBatch * kQueryLength * kTopK);
    for (std::size_t i = 0; i < host_topk.size(); ++i) { host_topk[i] = i % kTopK; }
    ASSERT_EQ(hipMemcpyAsync(q_memory, host_q.data(), host_q.size() * sizeof(float), hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(k_memory, host_k.data(), host_k.size() * sizeof(float), hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(w_memory, host_w.data(), host_w.size() * sizeof(float), hipMemcpyHostToDevice, stream),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(topk_memory, host_topk.data(), host_topk.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream),
              hipSuccess);

    FTrainArgs args = nullptr;
    ASSERT_EQ(ftrainArgsCreate(&args, ops), FTRAIN_STATUS_SUCCESS);
    auto fp32_view = [](void* memory, const std::int64_t* dims, std::uint8_t rank) {
        return FTrainStorageView{memory, dims, nullptr, rank, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                 false};
    };
    ASSERT_EQ(ftrainArgsSetTensor(args, q, fp32_view(q_memory, q_dims, 4)), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetTensor(args, k, fp32_view(k_memory, k_dims, 3)), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetTensor(args, w, fp32_view(w_memory, w_dims, 3)), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetTensor(args, topk,
                                  FTrainStorageView{topk_memory, topk_dims, nullptr, 3, FTRAIN_NUMERIC_TYPE_INT32,
                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, false}),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetTensor(args, predict, fp32_view(predict_memory, topk_dims, 3)), FTRAIN_STATUS_SUCCESS);
    const float scale                 = 1.0F;
    const std::uint8_t global_indices = 0;
    ASSERT_EQ(ftrainArgsSetTensor(args, sm_scale, hostScalar(&scale, FTRAIN_NUMERIC_TYPE_FP32)), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetTensor(args, topk_indices_global, hostScalar(&global_indices, FTRAIN_NUMERIC_TYPE_UINT8)),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetSparseIndexerPredict(args, op, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_STATUS_SUCCESS);

    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    std::uint64_t required = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &required), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(required, 0U);
    void* workspace = nullptr;
    ASSERT_EQ(hipMalloc(&workspace, required), hipSuccess);
    const FTrainStatus status = ftrainPlanExecutePrimitive(plan, 0, workspace, required, stream);
    EXPECT_EQ(status, FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::vector<float> actual(host_topk.size());
    ASSERT_EQ(hipMemcpy(actual.data(), predict_memory, actual.size() * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    // All selected scores are equal, so their softmax is uniform.
    const float expected = 1.0F / kTopK;
    for (float value : actual) { EXPECT_NEAR(value, expected, 1e-4F); }

    EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainOpsDestroy(ops), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(hipFree(workspace), hipSuccess);
    EXPECT_EQ(hipFree(q_memory), hipSuccess);
    EXPECT_EQ(hipFree(k_memory), hipSuccess);
    EXPECT_EQ(hipFree(w_memory), hipSuccess);
    EXPECT_EQ(hipFree(topk_memory), hipSuccess);
    EXPECT_EQ(hipFree(predict_memory), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

TEST_F(SparseIndexerPredictStagedTest, Bf16SparseIndexerPredictStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> q(256, 0x3e80);
    auto* device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 32};
    const std::vector<std::uint16_t> k(128, 0x3f00);
    auto* device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> w(8, 0x3f80);
    auto* device_w = allocate<std::uint16_t>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t topk_indices_dims[]{1, 4, 4};
    const std::vector<std::int32_t> topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(device_topk_indices, nullptr);
    const std::int64_t predict_dims[]{1, 4, 4};
    const std::vector<float> predict(16, 123.0F);
    auto* device_predict = allocate<float>(16);
    ASSERT_NE(device_predict, nullptr);
    float sm_scale                   = 0.25F;
    std::uint8_t topk_indices_global = 0;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_indices, topk_indices.data(), topk_indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_predict, predict.data(), predict.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 3, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto predict_view =
        makeView(device_predict, predict_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto topk_indices_global_view =
        makeView(&topk_indices_global, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                        topk_indices_global_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Bf16SparseIndexerPredict");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_predict, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
            }
        }
        ASSERT_EQ(
            ftrainSparseIndexerPredict(q_view, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                       topk_indices_global_view, FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
            FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_predict, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseIndexerPredict(invalid_q, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                         topk_indices_global_view, FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseIndexerPredict(invalid_q, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                         topk_indices_global_view, FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseIndexerPredictStagedTest, Fp16SparseIndexerPredictStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<__half> q(256, __float2half(0.25F));
    auto* device_q = allocate<__half>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 32};
    const std::vector<__half> k(128, __float2half(0.5F));
    auto* device_k = allocate<__half>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{1, 4, 2};
    const std::vector<__half> w(8, __float2half(1.0F));
    auto* device_w = allocate<__half>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t topk_indices_dims[]{1, 4, 4};
    const std::vector<std::int32_t> topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(device_topk_indices, nullptr);
    const std::int64_t predict_dims[]{1, 4, 4};
    const std::vector<float> predict(16, 123.0F);
    auto* device_predict = allocate<float>(16);
    ASSERT_NE(device_predict, nullptr);
    float sm_scale                   = 0.25F;
    std::uint8_t topk_indices_global = 0;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_indices, topk_indices.data(), topk_indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_predict, predict.data(), predict.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 3, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto predict_view =
        makeView(device_predict, predict_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto topk_indices_global_view =
        makeView(&topk_indices_global, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                        topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp16SparseIndexerPredict");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_predict, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
            }
        }
        ASSERT_EQ(
            ftrainSparseIndexerPredict(q_view, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                       topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
            FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_predict, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseIndexerPredict(invalid_q, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                         topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseIndexerPredict(invalid_q, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                         topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseIndexerPredictStagedTest, Fp32SparseIndexerPredictStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<float> q(256, 0.25F);
    auto* device_q = allocate<float>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 32};
    const std::vector<float> k(128, 0.5F);
    auto* device_k = allocate<float>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{1, 4, 2};
    const std::vector<float> w(8, 1.0F);
    auto* device_w = allocate<float>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t topk_indices_dims[]{1, 4, 4};
    const std::vector<std::int32_t> topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(device_topk_indices, nullptr);
    const std::int64_t predict_dims[]{1, 4, 4};
    const std::vector<float> predict(16, 123.0F);
    auto* device_predict = allocate<float>(16);
    ASSERT_NE(device_predict, nullptr);
    float sm_scale                   = 0.25F;
    std::uint8_t topk_indices_global = 0;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_indices, topk_indices.data(), topk_indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_predict, predict.data(), predict.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 3, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto predict_view =
        makeView(device_predict, predict_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto topk_indices_global_view =
        makeView(&topk_indices_global, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                        topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp32SparseIndexerPredict");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_predict, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
            }
        }
        ASSERT_EQ(
            ftrainSparseIndexerPredict(q_view, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                       topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
            FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_predict, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseIndexerPredict(invalid_q, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                         topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseIndexerPredict(invalid_q, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                         topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseIndexerPredictStagedTest, MixedFp32SparseIndexerPredictStorageMode3RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> q(256, 0x3e80);
    auto* device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 32};
    const std::vector<float> k(128, 0.5F);
    auto* device_k = allocate<float>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{1, 4, 2};
    const std::vector<float> w(8, 1.0F);
    auto* device_w = allocate<float>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t topk_indices_dims[]{1, 4, 4};
    const std::vector<std::int32_t> topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(device_topk_indices, nullptr);
    const std::int64_t predict_dims[]{1, 4, 4};
    const std::vector<float> predict(16, 123.0F);
    auto* device_predict = allocate<float>(16);
    ASSERT_NE(device_predict, nullptr);
    float sm_scale                   = 0.25F;
    std::uint8_t topk_indices_global = 0;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_indices, topk_indices.data(), topk_indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_predict, predict.data(), predict.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 3, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto predict_view =
        makeView(device_predict, predict_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto topk_indices_global_view =
        makeView(&topk_indices_global, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                        topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32SparseIndexerPredict");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_predict, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
            }
        }
        ASSERT_EQ(
            ftrainSparseIndexerPredict(q_view, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                       topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
            FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_predict, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseIndexerPredict(invalid_q, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                         topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseIndexerPredict(invalid_q, k_view, w_view, topk_indices_view, predict_view, sm_scale_view,
                                         topk_indices_global_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseIndexerPredictStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> first_q(256, 0x3e80);
    auto* first_device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(first_device_q, nullptr);
    const std::int64_t first_k_dims[]{1, 4, 32};
    const std::vector<std::uint16_t> first_k(128, 0x3f00);
    auto* first_device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(first_device_k, nullptr);
    const std::int64_t first_w_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> first_w(8, 0x3f80);
    auto* first_device_w = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_w, nullptr);
    const std::int64_t first_topk_indices_dims[]{1, 4, 4};
    const std::vector<std::int32_t> first_topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* first_device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(first_device_topk_indices, nullptr);
    const std::int64_t first_predict_dims[]{1, 4, 4};
    const std::vector<float> first_predict(16, 123.0F);
    auto* first_device_predict = allocate<float>(16);
    ASSERT_NE(first_device_predict, nullptr);
    float first_sm_scale                   = 0.25F;
    std::uint8_t first_topk_indices_global = 0;
    ASSERT_EQ(hipMemcpyAsync(first_device_q, first_q.data(), first_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_k, first_k.data(), first_k.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_w, first_w.data(), first_w.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_topk_indices, first_topk_indices.data(),
                             first_topk_indices.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_predict, first_predict.data(), first_predict.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_q_view =
        makeView(first_device_q, first_q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_k_view =
        makeView(first_device_k, first_k_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_w_view =
        makeView(first_device_w, first_w_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_topk_indices_view = makeView(first_device_topk_indices, first_topk_indices_dims, 3,
                                            FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_predict_view      = makeView(first_device_predict, first_predict_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_sm_scale_view =
        makeView(&first_sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_topk_indices_global_view =
        makeView(&first_topk_indices_global, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> second_q(256, 0x3e80);
    auto* second_device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(second_device_q, nullptr);
    const std::int64_t second_k_dims[]{1, 4, 32};
    const std::vector<std::uint16_t> second_k(128, 0x3f00);
    auto* second_device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(second_device_k, nullptr);
    const std::int64_t second_w_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> second_w(8, 0x3f80);
    auto* second_device_w = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_w, nullptr);
    const std::int64_t second_topk_indices_dims[]{1, 4, 4};
    const std::vector<std::int32_t> second_topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* second_device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(second_device_topk_indices, nullptr);
    const std::int64_t second_predict_dims[]{1, 4, 4};
    const std::vector<float> second_predict(16, 123.0F);
    auto* second_device_predict = allocate<float>(16);
    ASSERT_NE(second_device_predict, nullptr);
    float second_sm_scale                   = 0.25F;
    std::uint8_t second_topk_indices_global = 0;
    ASSERT_EQ(hipMemcpyAsync(second_device_q, second_q.data(), second_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_k, second_k.data(), second_k.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_w, second_w.data(), second_w.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_topk_indices, second_topk_indices.data(),
                             second_topk_indices.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_predict, second_predict.data(), second_predict.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_q_view =
        makeView(second_device_q, second_q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_k_view =
        makeView(second_device_k, second_k_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_w_view =
        makeView(second_device_w, second_w_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_topk_indices_view = makeView(second_device_topk_indices, second_topk_indices_dims, 3,
                                             FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_predict_view      = makeView(second_device_predict, second_predict_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_sm_scale_view =
        makeView(&second_sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_topk_indices_global_view = makeView(&second_topk_indices_global, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8,
                                                    FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args                      = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_q_view, first_k_view, first_w_view, first_topk_indices_view, first_predict_view,
                        first_sm_scale_view, first_topk_indices_global_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_q_view, second_k_view, second_w_view, second_topk_indices_view,
                        second_predict_view, second_sm_scale_view, second_topk_indices_global_view,
                        FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan second_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&second_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(second_plan, nullptr);
    trackPlan(second_plan);
    std::uint64_t second_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(second_plan, 0, &second_bytes), FTRAIN_STATUS_SUCCESS);
    void* second_workspace = second_bytes ? allocate<unsigned char>(second_bytes) : nullptr;
    if (second_bytes) { ASSERT_NE(second_workspace, nullptr); }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    ASSERT_EQ(ftrainOpsDestroy(ops_), FTRAIN_STATUS_SUCCESS);
    ops_ = nullptr;
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_predict, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_predict, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_predict, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
        }
    }
}

}  // namespace
