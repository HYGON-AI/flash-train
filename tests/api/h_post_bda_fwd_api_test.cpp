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

#include "h_post_bda_fwd_staged_fixture.hpp"

namespace {

using ftrain_test::HPostBdaFwdStagedTest;

TEST_F(HPostBdaFwdStagedTest, Bf16HPostBdaFwdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t h_res_dims[]{2, 1, 2, 2};
    const std::vector<std::uint16_t> h_res(8, 0x3e80);
    auto* device_h_res = allocate<std::uint16_t>(8);
    ASSERT_NE(device_h_res, nullptr);
    const std::int64_t residual_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> residual(64, 0x3f00);
    auto* device_residual = allocate<std::uint16_t>(64);
    ASSERT_NE(device_residual, nullptr);
    const std::int64_t h_post_dims[]{2, 1, 2};
    const std::vector<std::uint16_t> h_post(4, 0x3f00);
    auto* device_h_post = allocate<std::uint16_t>(4);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t x_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> x(32, 0x3f00);
    auto* device_x = allocate<std::uint16_t>(32);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t bias_dims[]{16};
    const std::vector<float> bias(16, 0.25F);
    auto* device_bias = allocate<float>(16);
    ASSERT_NE(device_bias, nullptr);
    const std::int64_t output_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> output(64, 0x42f6);
    auto* device_output = allocate<std::uint16_t>(64);
    ASSERT_NE(device_output, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_h_res, h_res.data(), h_res.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_residual, residual.data(), residual.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_bias, bias.data(), bias.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_output, output.data(), output.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto h_res_view =
        makeView(device_h_res, h_res_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto residual_view =
        makeView(device_residual, residual_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto bias_view = makeView(device_bias, bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto output_view =
        makeView(device_output, output_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, h_res_view, residual_view, h_post_view, x_view, bias_view, output_view,
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
    EXPECT_STREQ(primitive, "Bf16HPostBdaFwd");
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
            std::vector<std::uint16_t> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(0.625F)) {
                    EXPECT_EQ(actual_value, 0.625F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 0.625F, 2e-3 + 1e-2 * std::abs(0.625F)) << i;
                }
            }
        }
        ASSERT_EQ(ftrainHPostBdaFwd(h_res_view, residual_view, h_post_view, x_view, bias_view, output_view,
                                    FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(0.625F)) {
                    EXPECT_EQ(actual_value, 0.625F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 0.625F, 2e-3 + 1e-2 * std::abs(0.625F)) << i;
                }
            }
        }
    }
    auto invalid_h_res           = h_res_view;
    invalid_h_res.is_host_memory = true;
    EXPECT_EQ(ftrainHPostBdaFwd(invalid_h_res, residual_view, h_post_view, x_view, bias_view, output_view,
                                FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_h_res              = h_res_view;
    invalid_h_res.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainHPostBdaFwd(invalid_h_res, residual_view, h_post_view, x_view, bias_view, output_view,
                                FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(HPostBdaFwdStagedTest, Fp16HPostBdaFwdStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t h_res_dims[]{2, 1, 2, 2};
    const std::vector<__half> h_res(8, __float2half(0.25F));
    auto* device_h_res = allocate<__half>(8);
    ASSERT_NE(device_h_res, nullptr);
    const std::int64_t residual_dims[]{2, 1, 2, 16};
    const std::vector<__half> residual(64, __float2half(0.5F));
    auto* device_residual = allocate<__half>(64);
    ASSERT_NE(device_residual, nullptr);
    const std::int64_t h_post_dims[]{2, 1, 2};
    const std::vector<__half> h_post(4, __float2half(0.5F));
    auto* device_h_post = allocate<__half>(4);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t x_dims[]{2, 1, 16};
    const std::vector<__half> x(32, __float2half(0.5F));
    auto* device_x = allocate<__half>(32);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t bias_dims[]{16};
    const std::vector<float> bias(16, 0.25F);
    auto* device_bias = allocate<float>(16);
    ASSERT_NE(device_bias, nullptr);
    const std::int64_t output_dims[]{2, 1, 2, 16};
    const std::vector<__half> output(64, __float2half(123.0F));
    auto* device_output = allocate<__half>(64);
    ASSERT_NE(device_output, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_h_res, h_res.data(), h_res.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_residual, residual.data(), residual.size() * sizeof(__half), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_bias, bias.data(), bias.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_output, output.data(), output.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto h_res_view =
        makeView(device_h_res, h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto residual_view =
        makeView(device_residual, residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto bias_view = makeView(device_bias, bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto output_view =
        makeView(device_output, output_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, h_res_view, residual_view, h_post_view, x_view, bias_view, output_view,
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
    EXPECT_STREQ(primitive, "Fp16HPostBdaFwd");
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
            std::vector<__half> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(0.625F)) {
                    EXPECT_EQ(__half2float(actual[i]), 0.625F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 0.625F, 2e-4 + 2e-3 * std::abs(0.625F)) << i;
                }
            }
        }
        ASSERT_EQ(ftrainHPostBdaFwd(h_res_view, residual_view, h_post_view, x_view, bias_view, output_view,
                                    FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(0.625F)) {
                    EXPECT_EQ(__half2float(actual[i]), 0.625F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 0.625F, 2e-4 + 2e-3 * std::abs(0.625F)) << i;
                }
            }
        }
    }
    auto invalid_h_res           = h_res_view;
    invalid_h_res.is_host_memory = true;
    EXPECT_EQ(ftrainHPostBdaFwd(invalid_h_res, residual_view, h_post_view, x_view, bias_view, output_view,
                                FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_h_res              = h_res_view;
    invalid_h_res.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainHPostBdaFwd(invalid_h_res, residual_view, h_post_view, x_view, bias_view, output_view,
                                FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(HPostBdaFwdStagedTest, Fp32HPostBdaFwdStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t h_res_dims[]{2, 1, 2, 2};
    const std::vector<float> h_res(8, 0.25F);
    auto* device_h_res = allocate<float>(8);
    ASSERT_NE(device_h_res, nullptr);
    const std::int64_t residual_dims[]{2, 1, 2, 16};
    const std::vector<float> residual(64, 0.5F);
    auto* device_residual = allocate<float>(64);
    ASSERT_NE(device_residual, nullptr);
    const std::int64_t h_post_dims[]{2, 1, 2};
    const std::vector<float> h_post(4, 0.5F);
    auto* device_h_post = allocate<float>(4);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t x_dims[]{2, 1, 16};
    const std::vector<float> x(32, 0.5F);
    auto* device_x = allocate<float>(32);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t bias_dims[]{16};
    const std::vector<float> bias(16, 0.25F);
    auto* device_bias = allocate<float>(16);
    ASSERT_NE(device_bias, nullptr);
    const std::int64_t output_dims[]{2, 1, 2, 16};
    const std::vector<float> output(64, 123.0F);
    auto* device_output = allocate<float>(64);
    ASSERT_NE(device_output, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_h_res, h_res.data(), h_res.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_residual, residual.data(), residual.size() * sizeof(float), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_bias, bias.data(), bias.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_output, output.data(), output.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto h_res_view =
        makeView(device_h_res, h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto residual_view =
        makeView(device_residual, residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto bias_view = makeView(device_bias, bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto output_view =
        makeView(device_output, output_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, h_res_view, residual_view, h_post_view, x_view, bias_view, output_view,
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
    EXPECT_STREQ(primitive, "Fp32HPostBdaFwd");
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
            std::vector<float> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(0.625F)) {
                    EXPECT_EQ(actual[i], 0.625F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 0.625F, 2e-5 + 2e-4 * std::abs(0.625F)) << i;
                }
            }
        }
        ASSERT_EQ(ftrainHPostBdaFwd(h_res_view, residual_view, h_post_view, x_view, bias_view, output_view,
                                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(0.625F)) {
                    EXPECT_EQ(actual[i], 0.625F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 0.625F, 2e-5 + 2e-4 * std::abs(0.625F)) << i;
                }
            }
        }
    }
    auto invalid_h_res           = h_res_view;
    invalid_h_res.is_host_memory = true;
    EXPECT_EQ(ftrainHPostBdaFwd(invalid_h_res, residual_view, h_post_view, x_view, bias_view, output_view,
                                FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_h_res              = h_res_view;
    invalid_h_res.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainHPostBdaFwd(invalid_h_res, residual_view, h_post_view, x_view, bias_view, output_view,
                                FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(HPostBdaFwdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_h_res_dims[]{2, 1, 2, 2};
    const std::vector<std::uint16_t> first_h_res(8, 0x3e80);
    auto* first_device_h_res = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_h_res, nullptr);
    const std::int64_t first_residual_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> first_residual(64, 0x3f00);
    auto* first_device_residual = allocate<std::uint16_t>(64);
    ASSERT_NE(first_device_residual, nullptr);
    const std::int64_t first_h_post_dims[]{2, 1, 2};
    const std::vector<std::uint16_t> first_h_post(4, 0x3f00);
    auto* first_device_h_post = allocate<std::uint16_t>(4);
    ASSERT_NE(first_device_h_post, nullptr);
    const std::int64_t first_x_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> first_x(32, 0x3f00);
    auto* first_device_x = allocate<std::uint16_t>(32);
    ASSERT_NE(first_device_x, nullptr);
    const std::int64_t first_bias_dims[]{16};
    const std::vector<float> first_bias(16, 0.25F);
    auto* first_device_bias = allocate<float>(16);
    ASSERT_NE(first_device_bias, nullptr);
    const std::int64_t first_output_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> first_output(64, 0x42f6);
    auto* first_device_output = allocate<std::uint16_t>(64);
    ASSERT_NE(first_device_output, nullptr);
    ASSERT_EQ(hipMemcpyAsync(first_device_h_res, first_h_res.data(), first_h_res.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_residual, first_residual.data(),
                             first_residual.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_h_post, first_h_post.data(), first_h_post.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_x, first_x.data(), first_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_bias, first_bias.data(), first_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_output, first_output.data(), first_output.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_h_res_view    = makeView(first_device_h_res, first_h_res_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                        FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_residual_view = makeView(first_device_residual, first_residual_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                        FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_h_post_view   = makeView(first_device_h_post, first_h_post_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                        FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_x_view =
        makeView(first_device_x, first_x_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_bias_view =
        makeView(first_device_bias, first_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_output_view = makeView(first_device_output, first_output_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    const std::int64_t second_h_res_dims[]{2, 1, 2, 2};
    const std::vector<std::uint16_t> second_h_res(8, 0x3e80);
    auto* second_device_h_res = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_h_res, nullptr);
    const std::int64_t second_residual_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> second_residual(64, 0x3f00);
    auto* second_device_residual = allocate<std::uint16_t>(64);
    ASSERT_NE(second_device_residual, nullptr);
    const std::int64_t second_h_post_dims[]{2, 1, 2};
    const std::vector<std::uint16_t> second_h_post(4, 0x3f00);
    auto* second_device_h_post = allocate<std::uint16_t>(4);
    ASSERT_NE(second_device_h_post, nullptr);
    const std::int64_t second_x_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> second_x(32, 0x3f00);
    auto* second_device_x = allocate<std::uint16_t>(32);
    ASSERT_NE(second_device_x, nullptr);
    const std::int64_t second_bias_dims[]{16};
    const std::vector<float> second_bias(16, 0.25F);
    auto* second_device_bias = allocate<float>(16);
    ASSERT_NE(second_device_bias, nullptr);
    const std::int64_t second_output_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> second_output(64, 0x42f6);
    auto* second_device_output = allocate<std::uint16_t>(64);
    ASSERT_NE(second_device_output, nullptr);
    ASSERT_EQ(hipMemcpyAsync(second_device_h_res, second_h_res.data(), second_h_res.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_residual, second_residual.data(),
                             second_residual.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_h_post, second_h_post.data(), second_h_post.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_x, second_x.data(), second_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_bias, second_bias.data(), second_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_output, second_output.data(), second_output.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_h_res_view    = makeView(second_device_h_res, second_h_res_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_residual_view = makeView(second_device_residual, second_residual_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_h_post_view   = makeView(second_device_h_post, second_h_post_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_x_view =
        makeView(second_device_x, second_x_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_bias_view   = makeView(second_device_bias, second_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_output_view = makeView(second_device_output, second_output_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args         = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_h_res_view, first_residual_view, first_h_post_view, first_x_view, first_bias_view,
                        first_output_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_h_res_view, second_residual_view, second_h_post_view, second_x_view,
                        second_bias_view, second_output_view, FTRAIN_NUMERIC_TYPE_BF16));
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
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_output, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            if (std::isinf(0.625F)) {
                EXPECT_EQ(actual_value, 0.625F) << i;
            } else {
                EXPECT_NEAR(actual_value, 0.625F, 2e-3 + 1e-2 * std::abs(0.625F)) << i;
            }
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(64);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_output, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            if (std::isinf(0.625F)) {
                EXPECT_EQ(actual_value, 0.625F) << i;
            } else {
                EXPECT_NEAR(actual_value, 0.625F, 2e-3 + 1e-2 * std::abs(0.625F)) << i;
            }
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(64);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_output, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            if (std::isinf(0.625F)) {
                EXPECT_EQ(actual_value, 0.625F) << i;
            } else {
                EXPECT_NEAR(actual_value, 0.625F, 2e-3 + 1e-2 * std::abs(0.625F)) << i;
            }
        }
    }
}

}  // namespace
