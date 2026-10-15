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

#include "indexer_fwd_thd_staged_fixture.hpp"

namespace {

using ftrain_test::IndexerFwdThdStagedTest;

TEST_F(IndexerFwdThdStagedTest, Bf16IndexerFwdThdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 32};
    const std::vector<std::uint16_t> q(256, 0x3e80);
    auto* device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{4, 1, 32};
    const std::vector<std::uint16_t> k(128, 0x3f00);
    auto* device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{4, 2};
    const std::vector<std::uint16_t> w(8, 0x3f80);
    auto* device_w = allocate<std::uint16_t>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t cu_seqlens_q_dims[]{2};
    const std::vector<std::int32_t> cu_seqlens_q{0, 4};
    auto* device_cu_seqlens_q = allocate<std::int32_t>(2);
    ASSERT_NE(device_cu_seqlens_q, nullptr);
    const std::int64_t cu_seqlens_kv_dims[]{2};
    const std::vector<std::int32_t> cu_seqlens_kv{0, 4};
    auto* device_cu_seqlens_kv = allocate<std::int32_t>(2);
    ASSERT_NE(device_cu_seqlens_kv, nullptr);
    const std::int64_t q_causal_offsets_dims[]{1};
    const std::vector<std::int32_t> q_causal_offsets(1, 0);
    auto* device_q_causal_offsets = allocate<std::int32_t>(1);
    ASSERT_NE(device_q_causal_offsets, nullptr);
    const std::int64_t scores_dims[]{4, 4};
    const std::vector<float> scores(16, 123.0F);
    auto* device_scores = allocate<float>(16);
    ASSERT_NE(device_scores, nullptr);
    std::int64_t ratio         = 1;
    float sm_scale             = 0.25F;
    std::int64_t max_seqlen_kv = 4;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_cu_seqlens_q, cu_seqlens_q.data(), cu_seqlens_q.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_cu_seqlens_kv, cu_seqlens_kv.data(), cu_seqlens_kv.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_q_causal_offsets, q_causal_offsets.data(),
                             q_causal_offsets.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_q_view     = makeView(device_cu_seqlens_q, cu_seqlens_q_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_kv_view    = makeView(device_cu_seqlens_kv, cu_seqlens_kv_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto q_causal_offsets_view = makeView(device_q_causal_offsets, q_causal_offsets_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto scores_view =
        makeView(device_scores, scores_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view    = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto max_seqlen_kv_view =
        makeView(&max_seqlen_kv, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view, q_causal_offsets_view,
                        scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Bf16IndexerFwdThd");
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
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        ASSERT_EQ(ftrainIndexerFwdThd(q_view, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                      q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                      FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerFwdThd(invalid_q, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                  q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                  FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerFwdThd(invalid_q, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                  q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                  FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerFwdThdStagedTest, Fp16IndexerFwdThdStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 32};
    const std::vector<__half> q(256, __float2half(0.25F));
    auto* device_q = allocate<__half>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{4, 1, 32};
    const std::vector<__half> k(128, __float2half(0.5F));
    auto* device_k = allocate<__half>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{4, 2};
    const std::vector<__half> w(8, __float2half(1.0F));
    auto* device_w = allocate<__half>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t cu_seqlens_q_dims[]{2};
    const std::vector<std::int32_t> cu_seqlens_q{0, 4};
    auto* device_cu_seqlens_q = allocate<std::int32_t>(2);
    ASSERT_NE(device_cu_seqlens_q, nullptr);
    const std::int64_t cu_seqlens_kv_dims[]{2};
    const std::vector<std::int32_t> cu_seqlens_kv{0, 4};
    auto* device_cu_seqlens_kv = allocate<std::int32_t>(2);
    ASSERT_NE(device_cu_seqlens_kv, nullptr);
    const std::int64_t q_causal_offsets_dims[]{1};
    const std::vector<std::int32_t> q_causal_offsets(1, 0);
    auto* device_q_causal_offsets = allocate<std::int32_t>(1);
    ASSERT_NE(device_q_causal_offsets, nullptr);
    const std::int64_t scores_dims[]{4, 4};
    const std::vector<float> scores(16, 123.0F);
    auto* device_scores = allocate<float>(16);
    ASSERT_NE(device_scores, nullptr);
    std::int64_t ratio         = 1;
    float sm_scale             = 0.25F;
    std::int64_t max_seqlen_kv = 4;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_cu_seqlens_q, cu_seqlens_q.data(), cu_seqlens_q.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_cu_seqlens_kv, cu_seqlens_kv.data(), cu_seqlens_kv.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_q_causal_offsets, q_causal_offsets.data(),
                             q_causal_offsets.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_q_view     = makeView(device_cu_seqlens_q, cu_seqlens_q_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_kv_view    = makeView(device_cu_seqlens_kv, cu_seqlens_kv_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto q_causal_offsets_view = makeView(device_q_causal_offsets, q_causal_offsets_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto scores_view =
        makeView(device_scores, scores_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view    = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto max_seqlen_kv_view =
        makeView(&max_seqlen_kv, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view, q_causal_offsets_view,
                        scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view, FTRAIN_NUMERIC_TYPE_FP16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp16IndexerFwdThd");
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
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        ASSERT_EQ(ftrainIndexerFwdThd(q_view, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                      q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                      FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerFwdThd(invalid_q, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                  q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                  FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerFwdThd(invalid_q, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                  q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                  FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerFwdThdStagedTest, Fp32IndexerFwdThdStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 32};
    const std::vector<float> q(256, 0.25F);
    auto* device_q = allocate<float>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{4, 1, 32};
    const std::vector<float> k(128, 0.5F);
    auto* device_k = allocate<float>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{4, 2};
    const std::vector<float> w(8, 1.0F);
    auto* device_w = allocate<float>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t cu_seqlens_q_dims[]{2};
    const std::vector<std::int32_t> cu_seqlens_q{0, 4};
    auto* device_cu_seqlens_q = allocate<std::int32_t>(2);
    ASSERT_NE(device_cu_seqlens_q, nullptr);
    const std::int64_t cu_seqlens_kv_dims[]{2};
    const std::vector<std::int32_t> cu_seqlens_kv{0, 4};
    auto* device_cu_seqlens_kv = allocate<std::int32_t>(2);
    ASSERT_NE(device_cu_seqlens_kv, nullptr);
    const std::int64_t q_causal_offsets_dims[]{1};
    const std::vector<std::int32_t> q_causal_offsets(1, 0);
    auto* device_q_causal_offsets = allocate<std::int32_t>(1);
    ASSERT_NE(device_q_causal_offsets, nullptr);
    const std::int64_t scores_dims[]{4, 4};
    const std::vector<float> scores(16, 123.0F);
    auto* device_scores = allocate<float>(16);
    ASSERT_NE(device_scores, nullptr);
    std::int64_t ratio         = 1;
    float sm_scale             = 0.25F;
    std::int64_t max_seqlen_kv = 4;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_cu_seqlens_q, cu_seqlens_q.data(), cu_seqlens_q.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_cu_seqlens_kv, cu_seqlens_kv.data(), cu_seqlens_kv.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_q_causal_offsets, q_causal_offsets.data(),
                             q_causal_offsets.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_q_view     = makeView(device_cu_seqlens_q, cu_seqlens_q_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_kv_view    = makeView(device_cu_seqlens_kv, cu_seqlens_kv_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto q_causal_offsets_view = makeView(device_q_causal_offsets, q_causal_offsets_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto scores_view =
        makeView(device_scores, scores_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view    = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto max_seqlen_kv_view =
        makeView(&max_seqlen_kv, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view, q_causal_offsets_view,
                        scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp32IndexerFwdThd");
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
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        ASSERT_EQ(ftrainIndexerFwdThd(q_view, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                      q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                      FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerFwdThd(invalid_q, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                  q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerFwdThd(invalid_q, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                  q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerFwdThdStagedTest, MixedFp32IndexerFwdThdStorageMode3RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 32};
    const std::vector<std::uint16_t> q(256, 0x3e80);
    auto* device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{4, 1, 32};
    const std::vector<float> k(128, 0.5F);
    auto* device_k = allocate<float>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{4, 2};
    const std::vector<float> w(8, 1.0F);
    auto* device_w = allocate<float>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t cu_seqlens_q_dims[]{2};
    const std::vector<std::int32_t> cu_seqlens_q{0, 4};
    auto* device_cu_seqlens_q = allocate<std::int32_t>(2);
    ASSERT_NE(device_cu_seqlens_q, nullptr);
    const std::int64_t cu_seqlens_kv_dims[]{2};
    const std::vector<std::int32_t> cu_seqlens_kv{0, 4};
    auto* device_cu_seqlens_kv = allocate<std::int32_t>(2);
    ASSERT_NE(device_cu_seqlens_kv, nullptr);
    const std::int64_t q_causal_offsets_dims[]{1};
    const std::vector<std::int32_t> q_causal_offsets(1, 0);
    auto* device_q_causal_offsets = allocate<std::int32_t>(1);
    ASSERT_NE(device_q_causal_offsets, nullptr);
    const std::int64_t scores_dims[]{4, 4};
    const std::vector<float> scores(16, 123.0F);
    auto* device_scores = allocate<float>(16);
    ASSERT_NE(device_scores, nullptr);
    std::int64_t ratio         = 1;
    float sm_scale             = 0.25F;
    std::int64_t max_seqlen_kv = 4;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_cu_seqlens_q, cu_seqlens_q.data(), cu_seqlens_q.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_cu_seqlens_kv, cu_seqlens_kv.data(), cu_seqlens_kv.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_q_causal_offsets, q_causal_offsets.data(),
                             q_causal_offsets.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_q_view     = makeView(device_cu_seqlens_q, cu_seqlens_q_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_kv_view    = makeView(device_cu_seqlens_kv, cu_seqlens_kv_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto q_causal_offsets_view = makeView(device_q_causal_offsets, q_causal_offsets_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto scores_view =
        makeView(device_scores, scores_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view    = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto max_seqlen_kv_view =
        makeView(&max_seqlen_kv, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view, q_causal_offsets_view,
                        scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32IndexerFwdThd");
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
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        ASSERT_EQ(ftrainIndexerFwdThd(q_view, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                      q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                      FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerFwdThd(invalid_q, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                  q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerFwdThd(invalid_q, k_view, w_view, cu_seqlens_q_view, cu_seqlens_kv_view,
                                  q_causal_offsets_view, scores_view, ratio_view, sm_scale_view, max_seqlen_kv_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerFwdThdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_q_dims[]{4, 2, 32};
    const std::vector<std::uint16_t> first_q(256, 0x3e80);
    auto* first_device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(first_device_q, nullptr);
    const std::int64_t first_k_dims[]{4, 1, 32};
    const std::vector<std::uint16_t> first_k(128, 0x3f00);
    auto* first_device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(first_device_k, nullptr);
    const std::int64_t first_w_dims[]{4, 2};
    const std::vector<std::uint16_t> first_w(8, 0x3f80);
    auto* first_device_w = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_w, nullptr);
    const std::int64_t first_cu_seqlens_q_dims[]{2};
    const std::vector<std::int32_t> first_cu_seqlens_q{0, 4};
    auto* first_device_cu_seqlens_q = allocate<std::int32_t>(2);
    ASSERT_NE(first_device_cu_seqlens_q, nullptr);
    const std::int64_t first_cu_seqlens_kv_dims[]{2};
    const std::vector<std::int32_t> first_cu_seqlens_kv{0, 4};
    auto* first_device_cu_seqlens_kv = allocate<std::int32_t>(2);
    ASSERT_NE(first_device_cu_seqlens_kv, nullptr);
    const std::int64_t first_q_causal_offsets_dims[]{1};
    const std::vector<std::int32_t> first_q_causal_offsets(1, 0);
    auto* first_device_q_causal_offsets = allocate<std::int32_t>(1);
    ASSERT_NE(first_device_q_causal_offsets, nullptr);
    const std::int64_t first_scores_dims[]{4, 4};
    const std::vector<float> first_scores(16, 123.0F);
    auto* first_device_scores = allocate<float>(16);
    ASSERT_NE(first_device_scores, nullptr);
    std::int64_t first_ratio         = 1;
    float first_sm_scale             = 0.25F;
    std::int64_t first_max_seqlen_kv = 4;
    ASSERT_EQ(hipMemcpyAsync(first_device_q, first_q.data(), first_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_k, first_k.data(), first_k.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_w, first_w.data(), first_w.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_cu_seqlens_q, first_cu_seqlens_q.data(),
                             first_cu_seqlens_q.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_cu_seqlens_kv, first_cu_seqlens_kv.data(),
                             first_cu_seqlens_kv.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_q_causal_offsets, first_q_causal_offsets.data(),
                             first_q_causal_offsets.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_scores, first_scores.data(), first_scores.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_q_view =
        makeView(first_device_q, first_q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_k_view =
        makeView(first_device_k, first_k_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_w_view =
        makeView(first_device_w, first_w_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_cu_seqlens_q_view     = makeView(first_device_cu_seqlens_q, first_cu_seqlens_q_dims, 1,
                                                FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_cu_seqlens_kv_view    = makeView(first_device_cu_seqlens_kv, first_cu_seqlens_kv_dims, 1,
                                                FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_q_causal_offsets_view = makeView(first_device_q_causal_offsets, first_q_causal_offsets_dims, 1,
                                                FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_scores_view           = makeView(first_device_scores, first_scores_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                                FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_ratio_view =
        makeView(&first_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_sm_scale_view =
        makeView(&first_sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_max_seqlen_kv_view =
        makeView(&first_max_seqlen_kv, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_q_dims[]{4, 2, 32};
    const std::vector<std::uint16_t> second_q(256, 0x3e80);
    auto* second_device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(second_device_q, nullptr);
    const std::int64_t second_k_dims[]{4, 1, 32};
    const std::vector<std::uint16_t> second_k(128, 0x3f00);
    auto* second_device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(second_device_k, nullptr);
    const std::int64_t second_w_dims[]{4, 2};
    const std::vector<std::uint16_t> second_w(8, 0x3f80);
    auto* second_device_w = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_w, nullptr);
    const std::int64_t second_cu_seqlens_q_dims[]{2};
    const std::vector<std::int32_t> second_cu_seqlens_q{0, 4};
    auto* second_device_cu_seqlens_q = allocate<std::int32_t>(2);
    ASSERT_NE(second_device_cu_seqlens_q, nullptr);
    const std::int64_t second_cu_seqlens_kv_dims[]{2};
    const std::vector<std::int32_t> second_cu_seqlens_kv{0, 4};
    auto* second_device_cu_seqlens_kv = allocate<std::int32_t>(2);
    ASSERT_NE(second_device_cu_seqlens_kv, nullptr);
    const std::int64_t second_q_causal_offsets_dims[]{1};
    const std::vector<std::int32_t> second_q_causal_offsets(1, 0);
    auto* second_device_q_causal_offsets = allocate<std::int32_t>(1);
    ASSERT_NE(second_device_q_causal_offsets, nullptr);
    const std::int64_t second_scores_dims[]{4, 4};
    const std::vector<float> second_scores(16, 123.0F);
    auto* second_device_scores = allocate<float>(16);
    ASSERT_NE(second_device_scores, nullptr);
    std::int64_t second_ratio         = 1;
    float second_sm_scale             = 0.25F;
    std::int64_t second_max_seqlen_kv = 4;
    ASSERT_EQ(hipMemcpyAsync(second_device_q, second_q.data(), second_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_k, second_k.data(), second_k.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_w, second_w.data(), second_w.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_cu_seqlens_q, second_cu_seqlens_q.data(),
                             second_cu_seqlens_q.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_cu_seqlens_kv, second_cu_seqlens_kv.data(),
                             second_cu_seqlens_kv.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_q_causal_offsets, second_q_causal_offsets.data(),
                             second_q_causal_offsets.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_scores, second_scores.data(), second_scores.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_q_view =
        makeView(second_device_q, second_q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_k_view =
        makeView(second_device_k, second_k_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_w_view =
        makeView(second_device_w, second_w_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_cu_seqlens_q_view     = makeView(second_device_cu_seqlens_q, second_cu_seqlens_q_dims, 1,
                                                 FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_cu_seqlens_kv_view    = makeView(second_device_cu_seqlens_kv, second_cu_seqlens_kv_dims, 1,
                                                 FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_q_causal_offsets_view = makeView(second_device_q_causal_offsets, second_q_causal_offsets_dims, 1,
                                                 FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_scores_view           = makeView(second_device_scores, second_scores_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                                 FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_ratio_view =
        makeView(&second_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_sm_scale_view =
        makeView(&second_sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_max_seqlen_kv_view =
        makeView(&second_max_seqlen_kv, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_q_view, first_k_view, first_w_view, first_cu_seqlens_q_view,
                        first_cu_seqlens_kv_view, first_q_causal_offsets_view, first_scores_view, first_ratio_view,
                        first_sm_scale_view, first_max_seqlen_kv_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_q_view, second_k_view, second_w_view, second_cu_seqlens_q_view,
                        second_cu_seqlens_kv_view, second_q_causal_offsets_view, second_scores_view, second_ratio_view,
                        second_sm_scale_view, second_max_seqlen_kv_view, FTRAIN_NUMERIC_TYPE_BF16));
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
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_scores, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(16, 2.0F);
        std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
        std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
        expected[11] = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) continue;
            if (std::isinf(expected[i])) {
                EXPECT_EQ(actual[i], expected[i]) << i;
            } else {
                EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
            }
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_scores, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(16, 2.0F);
        std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
        std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
        expected[11] = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) continue;
            if (std::isinf(expected[i])) {
                EXPECT_EQ(actual[i], expected[i]) << i;
            } else {
                EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
            }
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_scores, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(16, 2.0F);
        std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
        std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
        expected[11] = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) continue;
            if (std::isinf(expected[i])) {
                EXPECT_EQ(actual[i], expected[i]) << i;
            } else {
                EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
            }
        }
    }
}

}  // namespace
