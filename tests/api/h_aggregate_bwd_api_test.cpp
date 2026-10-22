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

#include "h_aggregate_bwd_staged_fixture.hpp"

namespace {

using ftrain_test::HAggregateBwdStagedTest;

TEST_F(HAggregateBwdStagedTest, MixedFp32HAggregateBwdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t grad_aggregated_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> grad_aggregated(32, 0x3f80);
    auto* device_grad_aggregated = allocate<std::uint16_t>(32);
    ASSERT_NE(device_grad_aggregated, nullptr);
    const std::int64_t x_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> x(64, 0x3f00);
    auto* device_x = allocate<std::uint16_t>(64);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t h_pre_dims[]{2, 1, 2};
    const std::vector<float> h_pre(4, 0.25F);
    auto* device_h_pre = allocate<float>(4);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t grad_x_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> grad_x(64, 0x42f6);
    auto* device_grad_x = allocate<std::uint16_t>(64);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_h_pre_dims[]{2, 1, 2};
    const std::vector<float> grad_h_pre(4, 123.0F);
    auto* device_grad_h_pre = allocate<float>(4);
    ASSERT_NE(device_grad_h_pre, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_grad_aggregated, grad_aggregated.data(),
                             grad_aggregated.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_aggregated_view = makeView(device_grad_aggregated, grad_aggregated_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view = makeView(device_x, x_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_aggregated_view, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
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
    EXPECT_STREQ(primitive, "MixedFp32HAggregateBwd");
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
        ASSERT_EQ(hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(std::uint16_t),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.25F, 2e-3 + 1e-2 * std::abs(0.25F)) << i;
            }
        }
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_pre, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 8.0F, 2e-5 + 2e-4 * std::abs(8.0F)) << i;
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(std::uint16_t),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainHAggregateBwd(grad_aggregated_view, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
                                      FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.25F, 2e-3 + 1e-2 * std::abs(0.25F)) << i;
            }
        }
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_pre, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 8.0F, 2e-5 + 2e-4 * std::abs(8.0F)) << i;
            }
        }
    }
    auto invalid_grad_aggregated           = grad_aggregated_view;
    invalid_grad_aggregated.is_host_memory = true;
    EXPECT_EQ(ftrainHAggregateBwd(invalid_grad_aggregated, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_aggregated              = grad_aggregated_view;
    invalid_grad_aggregated.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainHAggregateBwd(invalid_grad_aggregated, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(HAggregateBwdStagedTest, MixedFp32HAggregateBwdStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t grad_aggregated_dims[]{2, 1, 16};
    const std::vector<__half> grad_aggregated(32, __float2half(1.0F));
    auto* device_grad_aggregated = allocate<__half>(32);
    ASSERT_NE(device_grad_aggregated, nullptr);
    const std::int64_t x_dims[]{2, 1, 2, 16};
    const std::vector<__half> x(64, __float2half(0.5F));
    auto* device_x = allocate<__half>(64);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t h_pre_dims[]{2, 1, 2};
    const std::vector<float> h_pre(4, 0.25F);
    auto* device_h_pre = allocate<float>(4);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t grad_x_dims[]{2, 1, 2, 16};
    const std::vector<__half> grad_x(64, __float2half(123.0F));
    auto* device_grad_x = allocate<__half>(64);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_h_pre_dims[]{2, 1, 2};
    const std::vector<float> grad_h_pre(4, 123.0F);
    auto* device_grad_h_pre = allocate<float>(4);
    ASSERT_NE(device_grad_h_pre, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_grad_aggregated, grad_aggregated.data(), grad_aggregated.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_aggregated_view = makeView(device_grad_aggregated, grad_aggregated_dims, 3, FTRAIN_NUMERIC_TYPE_FP16,
                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view = makeView(device_x, x_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_aggregated_view, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
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
    EXPECT_STREQ(primitive, "MixedFp32HAggregateBwd");
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
        ASSERT_EQ(hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(__half), hipMemcpyHostToDevice,
                                 stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 0.25F, 2e-4 + 2e-3 * std::abs(0.25F)) << i;
            }
        }
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_pre, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 8.0F, 2e-5 + 2e-4 * std::abs(8.0F)) << i;
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(__half), hipMemcpyHostToDevice,
                                 stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainHAggregateBwd(grad_aggregated_view, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
                                      FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 0.25F, 2e-4 + 2e-3 * std::abs(0.25F)) << i;
            }
        }
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_pre, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 8.0F, 2e-5 + 2e-4 * std::abs(8.0F)) << i;
            }
        }
    }
    auto invalid_grad_aggregated           = grad_aggregated_view;
    invalid_grad_aggregated.is_host_memory = true;
    EXPECT_EQ(ftrainHAggregateBwd(invalid_grad_aggregated, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_aggregated              = grad_aggregated_view;
    invalid_grad_aggregated.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainHAggregateBwd(invalid_grad_aggregated, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(HAggregateBwdStagedTest, Fp32HAggregateBwdStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t grad_aggregated_dims[]{2, 1, 16};
    const std::vector<float> grad_aggregated(32, 1.0F);
    auto* device_grad_aggregated = allocate<float>(32);
    ASSERT_NE(device_grad_aggregated, nullptr);
    const std::int64_t x_dims[]{2, 1, 2, 16};
    const std::vector<float> x(64, 0.5F);
    auto* device_x = allocate<float>(64);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t h_pre_dims[]{2, 1, 2};
    const std::vector<float> h_pre(4, 0.25F);
    auto* device_h_pre = allocate<float>(4);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t grad_x_dims[]{2, 1, 2, 16};
    const std::vector<float> grad_x(64, 123.0F);
    auto* device_grad_x = allocate<float>(64);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_h_pre_dims[]{2, 1, 2};
    const std::vector<float> grad_h_pre(4, 123.0F);
    auto* device_grad_h_pre = allocate<float>(4);
    ASSERT_NE(device_grad_h_pre, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_grad_aggregated, grad_aggregated.data(), grad_aggregated.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_aggregated_view = makeView(device_grad_aggregated, grad_aggregated_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view = makeView(device_x, x_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_aggregated_view, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
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
    EXPECT_STREQ(primitive, "Fp32HAggregateBwd");
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
        ASSERT_EQ(
            hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
            }
        }
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_pre, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 8.0F, 2e-5 + 2e-4 * std::abs(8.0F)) << i;
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(
            hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainHAggregateBwd(grad_aggregated_view, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
                                      FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.25F, 2e-5 + 2e-4 * std::abs(0.25F)) << i;
            }
        }
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_pre, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 8.0F, 2e-5 + 2e-4 * std::abs(8.0F)) << i;
            }
        }
    }
    auto invalid_grad_aggregated           = grad_aggregated_view;
    invalid_grad_aggregated.is_host_memory = true;
    EXPECT_EQ(ftrainHAggregateBwd(invalid_grad_aggregated, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_aggregated              = grad_aggregated_view;
    invalid_grad_aggregated.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainHAggregateBwd(invalid_grad_aggregated, x_view, h_pre_view, grad_x_view, grad_h_pre_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(HAggregateBwdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_grad_aggregated_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> first_grad_aggregated(32, 0x3f80);
    auto* first_device_grad_aggregated = allocate<std::uint16_t>(32);
    ASSERT_NE(first_device_grad_aggregated, nullptr);
    const std::int64_t first_x_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> first_x(64, 0x3f00);
    auto* first_device_x = allocate<std::uint16_t>(64);
    ASSERT_NE(first_device_x, nullptr);
    const std::int64_t first_h_pre_dims[]{2, 1, 2};
    const std::vector<float> first_h_pre(4, 0.25F);
    auto* first_device_h_pre = allocate<float>(4);
    ASSERT_NE(first_device_h_pre, nullptr);
    const std::int64_t first_grad_x_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> first_grad_x(64, 0x42f6);
    auto* first_device_grad_x = allocate<std::uint16_t>(64);
    ASSERT_NE(first_device_grad_x, nullptr);
    const std::int64_t first_grad_h_pre_dims[]{2, 1, 2};
    const std::vector<float> first_grad_h_pre(4, 123.0F);
    auto* first_device_grad_h_pre = allocate<float>(4);
    ASSERT_NE(first_device_grad_h_pre, nullptr);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_aggregated, first_grad_aggregated.data(),
                             first_grad_aggregated.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_x, first_x.data(), first_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_h_pre, first_h_pre.data(), first_h_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_x, first_grad_x.data(), first_grad_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_h_pre, first_grad_h_pre.data(), first_grad_h_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_grad_aggregated_view = makeView(first_device_grad_aggregated, first_grad_aggregated_dims, 3,
                                               FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_x_view =
        makeView(first_device_x, first_x_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_h_pre_view      = makeView(first_device_h_pre, first_h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_x_view     = makeView(first_device_grad_x, first_grad_x_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_h_pre_view = makeView(first_device_grad_h_pre, first_grad_h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    const std::int64_t second_grad_aggregated_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> second_grad_aggregated(32, 0x3f80);
    auto* second_device_grad_aggregated = allocate<std::uint16_t>(32);
    ASSERT_NE(second_device_grad_aggregated, nullptr);
    const std::int64_t second_x_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> second_x(64, 0x3f00);
    auto* second_device_x = allocate<std::uint16_t>(64);
    ASSERT_NE(second_device_x, nullptr);
    const std::int64_t second_h_pre_dims[]{2, 1, 2};
    const std::vector<float> second_h_pre(4, 0.25F);
    auto* second_device_h_pre = allocate<float>(4);
    ASSERT_NE(second_device_h_pre, nullptr);
    const std::int64_t second_grad_x_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> second_grad_x(64, 0x42f6);
    auto* second_device_grad_x = allocate<std::uint16_t>(64);
    ASSERT_NE(second_device_grad_x, nullptr);
    const std::int64_t second_grad_h_pre_dims[]{2, 1, 2};
    const std::vector<float> second_grad_h_pre(4, 123.0F);
    auto* second_device_grad_h_pre = allocate<float>(4);
    ASSERT_NE(second_device_grad_h_pre, nullptr);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_aggregated, second_grad_aggregated.data(),
                             second_grad_aggregated.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_x, second_x.data(), second_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_h_pre, second_h_pre.data(), second_h_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_x, second_grad_x.data(), second_grad_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_h_pre, second_grad_h_pre.data(),
                             second_grad_h_pre.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_grad_aggregated_view = makeView(second_device_grad_aggregated, second_grad_aggregated_dims, 3,
                                                FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_x_view =
        makeView(second_device_x, second_x_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_h_pre_view      = makeView(second_device_h_pre, second_h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_x_view     = makeView(second_device_grad_x, second_grad_x_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_h_pre_view = makeView(second_device_grad_h_pre, second_grad_h_pre_dims, 3,
                                           FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args             = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_grad_aggregated_view, first_x_view, first_h_pre_view, first_grad_x_view,
                        first_grad_h_pre_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_grad_aggregated_view, second_x_view, second_h_pre_view, second_grad_x_view,
                        second_grad_h_pre_view, FTRAIN_NUMERIC_TYPE_FP32));
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
        std::vector<std::uint16_t> actual(64);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_x, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.25F, 2e-3 + 1e-2 * std::abs(0.25F)) << i;
        }
    }
    {
        std::vector<float> actual(4);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_h_pre, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 8.0F, 2e-5 + 2e-4 * std::abs(8.0F)) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(64);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_x, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.25F, 2e-3 + 1e-2 * std::abs(0.25F)) << i;
        }
    }
    {
        std::vector<float> actual(4);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_h_pre, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 8.0F, 2e-5 + 2e-4 * std::abs(8.0F)) << i;
        }
    }
}

}  // namespace
