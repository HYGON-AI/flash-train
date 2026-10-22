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

#include "dense_attn_score_staged_fixture.hpp"

namespace {

using ftrain_test::DenseAttnScoreStagedTest;

TEST_F(DenseAttnScoreStagedTest, Bf16DenseAttnScoreStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> q(256, 0x3e80);
    auto* device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 1, 32};
    const std::vector<std::uint16_t> k(128, 0x3f00);
    auto* device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t lse_dims[]{1, 4, 2};
    const std::vector<float> lse(8, 0.0F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
    const std::int64_t out_dims[]{1, 4, 4};
    const std::vector<float> out(16, 123.0F);
    auto* device_out = allocate<float>(16);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t denom_dims[]{1, 4};
    const std::vector<float> denom(4, 123.0F);
    auto* device_denom = allocate<float>(4);
    ASSERT_NE(device_denom, nullptr);
    std::int64_t ratio  = 1;
    float softmax_scale = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view   = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view   = makeView(device_k, k_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view = makeView(device_lse, lse_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto denom_view =
        makeView(device_denom, denom_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                        FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Bf16DenseAttnScore");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 5.43656365691809F);
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
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_denom, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(4, 5.43656365691809F);
            expected[1] = 10.87312731383618F;
            expected[2] = 16.30969097075427F;
            expected[3] = 21.74625462767236F;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainDenseAttnScore(q_view, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                       FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 5.43656365691809F);
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
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_denom, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(4, 5.43656365691809F);
            expected[1] = 10.87312731383618F;
            expected[2] = 16.30969097075427F;
            expected[3] = 21.74625462767236F;
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
    EXPECT_EQ(ftrainDenseAttnScore(invalid_q, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                   FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainDenseAttnScore(invalid_q, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                   FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(DenseAttnScoreStagedTest, Fp16DenseAttnScoreStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<__half> q(256, __float2half(0.25F));
    auto* device_q = allocate<__half>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 1, 32};
    const std::vector<__half> k(128, __float2half(0.5F));
    auto* device_k = allocate<__half>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t lse_dims[]{1, 4, 2};
    const std::vector<float> lse(8, 0.0F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
    const std::int64_t out_dims[]{1, 4, 4};
    const std::vector<float> out(16, 123.0F);
    auto* device_out = allocate<float>(16);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t denom_dims[]{1, 4};
    const std::vector<float> denom(4, 123.0F);
    auto* device_denom = allocate<float>(4);
    ASSERT_NE(device_denom, nullptr);
    std::int64_t ratio  = 1;
    float softmax_scale = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view   = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view   = makeView(device_k, k_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view = makeView(device_lse, lse_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto denom_view =
        makeView(device_denom, denom_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                        FTRAIN_NUMERIC_TYPE_FP16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp16DenseAttnScore");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 5.43656365691809F);
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
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_denom, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(4, 5.43656365691809F);
            expected[1] = 10.87312731383618F;
            expected[2] = 16.30969097075427F;
            expected[3] = 21.74625462767236F;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainDenseAttnScore(q_view, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                       FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 5.43656365691809F);
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
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_denom, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(4, 5.43656365691809F);
            expected[1] = 10.87312731383618F;
            expected[2] = 16.30969097075427F;
            expected[3] = 21.74625462767236F;
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
    EXPECT_EQ(ftrainDenseAttnScore(invalid_q, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                   FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainDenseAttnScore(invalid_q, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                   FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(DenseAttnScoreStagedTest, Fp32DenseAttnScoreStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<float> q(256, 0.25F);
    auto* device_q = allocate<float>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 1, 32};
    const std::vector<float> k(128, 0.5F);
    auto* device_k = allocate<float>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t lse_dims[]{1, 4, 2};
    const std::vector<float> lse(8, 0.0F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
    const std::int64_t out_dims[]{1, 4, 4};
    const std::vector<float> out(16, 123.0F);
    auto* device_out = allocate<float>(16);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t denom_dims[]{1, 4};
    const std::vector<float> denom(4, 123.0F);
    auto* device_denom = allocate<float>(4);
    ASSERT_NE(device_denom, nullptr);
    std::int64_t ratio  = 1;
    float softmax_scale = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view   = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view   = makeView(device_k, k_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view = makeView(device_lse, lse_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto denom_view =
        makeView(device_denom, denom_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                        FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp32DenseAttnScore");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 5.43656365691809F);
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
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_denom, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(4, 5.43656365691809F);
            expected[1] = 10.87312731383618F;
            expected[2] = 16.30969097075427F;
            expected[3] = 21.74625462767236F;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainDenseAttnScore(q_view, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                       FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 5.43656365691809F);
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
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_denom, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(4, 5.43656365691809F);
            expected[1] = 10.87312731383618F;
            expected[2] = 16.30969097075427F;
            expected[3] = 21.74625462767236F;
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
    EXPECT_EQ(ftrainDenseAttnScore(invalid_q, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                   FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainDenseAttnScore(invalid_q, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                   FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(DenseAttnScoreStagedTest, MixedFp32DenseAttnScoreStorageMode3RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> q(256, 0x3e80);
    auto* device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 1, 32};
    const std::vector<float> k(128, 0.5F);
    auto* device_k = allocate<float>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t lse_dims[]{1, 4, 2};
    const std::vector<float> lse(8, 0.0F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
    const std::int64_t out_dims[]{1, 4, 4};
    const std::vector<float> out(16, 123.0F);
    auto* device_out = allocate<float>(16);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t denom_dims[]{1, 4};
    const std::vector<float> denom(4, 123.0F);
    auto* device_denom = allocate<float>(4);
    ASSERT_NE(device_denom, nullptr);
    std::int64_t ratio  = 1;
    float softmax_scale = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view   = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view   = makeView(device_k, k_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view = makeView(device_lse, lse_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto denom_view =
        makeView(device_denom, denom_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                        FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32DenseAttnScore");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 5.43656365691809F);
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
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_denom, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(4, 5.43656365691809F);
            expected[1] = 10.87312731383618F;
            expected[2] = 16.30969097075427F;
            expected[3] = 21.74625462767236F;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_denom, denom.data(), denom.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainDenseAttnScore(q_view, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                       FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 5.43656365691809F);
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
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_denom, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(4, 5.43656365691809F);
            expected[1] = 10.87312731383618F;
            expected[2] = 16.30969097075427F;
            expected[3] = 21.74625462767236F;
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
    EXPECT_EQ(ftrainDenseAttnScore(invalid_q, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                   FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainDenseAttnScore(invalid_q, k_view, lse_view, out_view, denom_view, ratio_view, softmax_scale_view,
                                   FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(DenseAttnScoreStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> first_q(256, 0x3e80);
    auto* first_device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(first_device_q, nullptr);
    const std::int64_t first_k_dims[]{1, 4, 1, 32};
    const std::vector<std::uint16_t> first_k(128, 0x3f00);
    auto* first_device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(first_device_k, nullptr);
    const std::int64_t first_lse_dims[]{1, 4, 2};
    const std::vector<float> first_lse(8, 0.0F);
    auto* first_device_lse = allocate<float>(8);
    ASSERT_NE(first_device_lse, nullptr);
    const std::int64_t first_out_dims[]{1, 4, 4};
    const std::vector<float> first_out(16, 123.0F);
    auto* first_device_out = allocate<float>(16);
    ASSERT_NE(first_device_out, nullptr);
    const std::int64_t first_denom_dims[]{1, 4};
    const std::vector<float> first_denom(4, 123.0F);
    auto* first_device_denom = allocate<float>(4);
    ASSERT_NE(first_device_denom, nullptr);
    std::int64_t first_ratio  = 1;
    float first_softmax_scale = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(first_device_q, first_q.data(), first_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_k, first_k.data(), first_k.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_lse, first_lse.data(), first_lse.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_out, first_out.data(), first_out.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_denom, first_denom.data(), first_denom.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_q_view =
        makeView(first_device_q, first_q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_k_view =
        makeView(first_device_k, first_k_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_lse_view =
        makeView(first_device_lse, first_lse_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_out_view =
        makeView(first_device_out, first_out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_denom_view = makeView(first_device_denom, first_denom_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_ratio_view =
        makeView(&first_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_softmax_scale_view =
        makeView(&first_softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> second_q(256, 0x3e80);
    auto* second_device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(second_device_q, nullptr);
    const std::int64_t second_k_dims[]{1, 4, 1, 32};
    const std::vector<std::uint16_t> second_k(128, 0x3f00);
    auto* second_device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(second_device_k, nullptr);
    const std::int64_t second_lse_dims[]{1, 4, 2};
    const std::vector<float> second_lse(8, 0.0F);
    auto* second_device_lse = allocate<float>(8);
    ASSERT_NE(second_device_lse, nullptr);
    const std::int64_t second_out_dims[]{1, 4, 4};
    const std::vector<float> second_out(16, 123.0F);
    auto* second_device_out = allocate<float>(16);
    ASSERT_NE(second_device_out, nullptr);
    const std::int64_t second_denom_dims[]{1, 4};
    const std::vector<float> second_denom(4, 123.0F);
    auto* second_device_denom = allocate<float>(4);
    ASSERT_NE(second_device_denom, nullptr);
    std::int64_t second_ratio  = 1;
    float second_softmax_scale = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(second_device_q, second_q.data(), second_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_k, second_k.data(), second_k.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_lse, second_lse.data(), second_lse.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_out, second_out.data(), second_out.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_denom, second_denom.data(), second_denom.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_q_view =
        makeView(second_device_q, second_q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_k_view =
        makeView(second_device_k, second_k_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_lse_view =
        makeView(second_device_lse, second_lse_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_out_view =
        makeView(second_device_out, second_out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_denom_view = makeView(second_device_denom, second_denom_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_ratio_view =
        makeView(&second_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_softmax_scale_view =
        makeView(&second_softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_q_view, first_k_view, first_lse_view, first_out_view, first_denom_view,
                        first_ratio_view, first_softmax_scale_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_q_view, second_k_view, second_lse_view, second_out_view, second_denom_view,
                        second_ratio_view, second_softmax_scale_view, FTRAIN_NUMERIC_TYPE_BF16));
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
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                 stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(16, 5.43656365691809F);
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
    {
        std::vector<float> actual(4);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_denom, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(4, 5.43656365691809F);
        expected[1] = 10.87312731383618F;
        expected[2] = 16.30969097075427F;
        expected[3] = 21.74625462767236F;
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
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                 stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(16, 5.43656365691809F);
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
    {
        std::vector<float> actual(4);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_denom, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(4, 5.43656365691809F);
        expected[1] = 10.87312731383618F;
        expected[2] = 16.30969097075427F;
        expected[3] = 21.74625462767236F;
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
