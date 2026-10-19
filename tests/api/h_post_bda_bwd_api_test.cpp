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

#include "h_post_bda_bwd_staged_fixture.hpp"

namespace {

using ftrain_test::HPostBdaBwdStagedTest;

TEST_F(HPostBdaBwdStagedTest, Bf16HPostBdaBwdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t grad_output_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> grad_output(64, 0x3f80);
    auto* device_grad_output = allocate<std::uint16_t>(64);
    ASSERT_NE(device_grad_output, nullptr);
    const std::int64_t residual_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> residual(64, 0x3f00);
    auto* device_residual = allocate<std::uint16_t>(64);
    ASSERT_NE(device_residual, nullptr);
    const std::int64_t x_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> x(32, 0x3f00);
    auto* device_x = allocate<std::uint16_t>(32);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t bias_dims[]{16};
    const std::vector<float> bias(16, 0.25F);
    auto* device_bias = allocate<float>(16);
    ASSERT_NE(device_bias, nullptr);
    const std::int64_t h_res_dims[]{2, 1, 2, 2};
    const std::vector<std::uint16_t> h_res(8, 0x3e80);
    auto* device_h_res = allocate<std::uint16_t>(8);
    ASSERT_NE(device_h_res, nullptr);
    const std::int64_t h_post_dims[]{2, 1, 2};
    const std::vector<std::uint16_t> h_post(4, 0x3f00);
    auto* device_h_post = allocate<std::uint16_t>(4);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t grad_residual_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> grad_residual(64, 0x42f6);
    auto* device_grad_residual = allocate<std::uint16_t>(64);
    ASSERT_NE(device_grad_residual, nullptr);
    const std::int64_t grad_x_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> grad_x(32, 0x42f6);
    auto* device_grad_x = allocate<std::uint16_t>(32);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_h_res_dims[]{2, 1, 2, 2};
    const std::vector<std::uint16_t> grad_h_res(8, 0x42f6);
    auto* device_grad_h_res = allocate<std::uint16_t>(8);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_h_post_dims[]{2, 1, 2};
    const std::vector<std::uint16_t> grad_h_post(4, 0x42f6);
    auto* device_grad_h_post = allocate<std::uint16_t>(4);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_bias_dims[]{16};
    const std::vector<float> grad_bias(16, 123.0F);
    auto* device_grad_bias = allocate<float>(16);
    ASSERT_NE(device_grad_bias, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_grad_output, grad_output.data(), grad_output.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_residual, residual.data(), residual.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_bias, bias.data(), bias.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_res, h_res.data(), h_res.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_residual, grad_residual.data(), grad_residual.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_output_view = makeView(device_grad_output, grad_output_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto residual_view =
        makeView(device_residual, residual_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto bias_view = makeView(device_bias, bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_res_view =
        makeView(device_h_res, h_res_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_residual_view = makeView(device_grad_residual, grad_residual_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_output_view, residual_view, x_view, bias_view, h_res_view, h_post_view,
                        grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
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
    EXPECT_STREQ(primitive, "Bf16HPostBdaBwd");
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
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_residual, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.5F, 2e-3 + 1e-2 * std::abs(0.5F)) << i;
            }
        }
        {
            std::vector<std::uint16_t> actual(32);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 1.0F, 2e-3 + 1e-2 * std::abs(1.0F)) << i;
            }
        }
        {
            std::vector<std::uint16_t> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_res, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 8.0F, 2e-3 + 1e-2 * std::abs(8.0F)) << i;
            }
        }
        {
            std::vector<std::uint16_t> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_post, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 12.0F, 2e-3 + 1e-2 * std::abs(12.0F)) << i;
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 2.0F, 2e-5 + 2e-4 * std::abs(2.0F)) << i;
            }
        }
        ASSERT_EQ(ftrainHPostBdaBwd(grad_output_view, residual_view, x_view, bias_view, h_res_view, h_post_view,
                                    grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
                                    FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_residual, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.5F, 2e-3 + 1e-2 * std::abs(0.5F)) << i;
            }
        }
        {
            std::vector<std::uint16_t> actual(32);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 1.0F, 2e-3 + 1e-2 * std::abs(1.0F)) << i;
            }
        }
        {
            std::vector<std::uint16_t> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_res, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 8.0F, 2e-3 + 1e-2 * std::abs(8.0F)) << i;
            }
        }
        {
            std::vector<std::uint16_t> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_post, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 12.0F, 2e-3 + 1e-2 * std::abs(12.0F)) << i;
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 2.0F, 2e-5 + 2e-4 * std::abs(2.0F)) << i;
            }
        }
    }
    auto invalid_grad_output           = grad_output_view;
    invalid_grad_output.is_host_memory = true;
    EXPECT_EQ(ftrainHPostBdaBwd(invalid_grad_output, residual_view, x_view, bias_view, h_res_view, h_post_view,
                                grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
                                FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_output              = grad_output_view;
    invalid_grad_output.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainHPostBdaBwd(invalid_grad_output, residual_view, x_view, bias_view, h_res_view, h_post_view,
                                grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
                                FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(HPostBdaBwdStagedTest, Fp16HPostBdaBwdStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t grad_output_dims[]{2, 1, 2, 16};
    const std::vector<__half> grad_output(64, __float2half(1.0F));
    auto* device_grad_output = allocate<__half>(64);
    ASSERT_NE(device_grad_output, nullptr);
    const std::int64_t residual_dims[]{2, 1, 2, 16};
    const std::vector<__half> residual(64, __float2half(0.5F));
    auto* device_residual = allocate<__half>(64);
    ASSERT_NE(device_residual, nullptr);
    const std::int64_t x_dims[]{2, 1, 16};
    const std::vector<__half> x(32, __float2half(0.5F));
    auto* device_x = allocate<__half>(32);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t bias_dims[]{16};
    const std::vector<float> bias(16, 0.25F);
    auto* device_bias = allocate<float>(16);
    ASSERT_NE(device_bias, nullptr);
    const std::int64_t h_res_dims[]{2, 1, 2, 2};
    const std::vector<__half> h_res(8, __float2half(0.25F));
    auto* device_h_res = allocate<__half>(8);
    ASSERT_NE(device_h_res, nullptr);
    const std::int64_t h_post_dims[]{2, 1, 2};
    const std::vector<__half> h_post(4, __float2half(0.5F));
    auto* device_h_post = allocate<__half>(4);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t grad_residual_dims[]{2, 1, 2, 16};
    const std::vector<__half> grad_residual(64, __float2half(123.0F));
    auto* device_grad_residual = allocate<__half>(64);
    ASSERT_NE(device_grad_residual, nullptr);
    const std::int64_t grad_x_dims[]{2, 1, 16};
    const std::vector<__half> grad_x(32, __float2half(123.0F));
    auto* device_grad_x = allocate<__half>(32);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_h_res_dims[]{2, 1, 2, 2};
    const std::vector<__half> grad_h_res(8, __float2half(123.0F));
    auto* device_grad_h_res = allocate<__half>(8);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_h_post_dims[]{2, 1, 2};
    const std::vector<__half> grad_h_post(4, __float2half(123.0F));
    auto* device_grad_h_post = allocate<__half>(4);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_bias_dims[]{16};
    const std::vector<float> grad_bias(16, 123.0F);
    auto* device_grad_bias = allocate<float>(16);
    ASSERT_NE(device_grad_bias, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_grad_output, grad_output.data(), grad_output.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_residual, residual.data(), residual.size() * sizeof(__half), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_bias, bias.data(), bias.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_res, h_res.data(), h_res.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_residual, grad_residual.data(), grad_residual.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_output_view = makeView(device_grad_output, grad_output_dims, 4, FTRAIN_NUMERIC_TYPE_FP16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto residual_view =
        makeView(device_residual, residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto bias_view = makeView(device_bias, bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_res_view =
        makeView(device_h_res, h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_residual_view = makeView(device_grad_residual, grad_residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP16,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_output_view, residual_view, x_view, bias_view, h_res_view, h_post_view,
                        grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
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
    EXPECT_STREQ(primitive, "Fp16HPostBdaBwd");
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
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_residual, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 0.5F, 2e-4 + 2e-3 * std::abs(0.5F)) << i;
            }
        }
        {
            std::vector<__half> actual(32);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 1.0F, 2e-4 + 2e-3 * std::abs(1.0F)) << i;
            }
        }
        {
            std::vector<__half> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_res, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 8.0F, 2e-4 + 2e-3 * std::abs(8.0F)) << i;
            }
        }
        {
            std::vector<__half> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_post, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 12.0F, 2e-4 + 2e-3 * std::abs(12.0F)) << i;
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 2.0F, 2e-5 + 2e-4 * std::abs(2.0F)) << i;
            }
        }
        ASSERT_EQ(ftrainHPostBdaBwd(grad_output_view, residual_view, x_view, bias_view, h_res_view, h_post_view,
                                    grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
                                    FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_residual, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 0.5F, 2e-4 + 2e-3 * std::abs(0.5F)) << i;
            }
        }
        {
            std::vector<__half> actual(32);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 1.0F, 2e-4 + 2e-3 * std::abs(1.0F)) << i;
            }
        }
        {
            std::vector<__half> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_res, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 8.0F, 2e-4 + 2e-3 * std::abs(8.0F)) << i;
            }
        }
        {
            std::vector<__half> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_post, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 12.0F, 2e-4 + 2e-3 * std::abs(12.0F)) << i;
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 2.0F, 2e-5 + 2e-4 * std::abs(2.0F)) << i;
            }
        }
    }
    auto invalid_grad_output           = grad_output_view;
    invalid_grad_output.is_host_memory = true;
    EXPECT_EQ(ftrainHPostBdaBwd(invalid_grad_output, residual_view, x_view, bias_view, h_res_view, h_post_view,
                                grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
                                FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_output              = grad_output_view;
    invalid_grad_output.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainHPostBdaBwd(invalid_grad_output, residual_view, x_view, bias_view, h_res_view, h_post_view,
                                grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
                                FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(HPostBdaBwdStagedTest, Fp32HPostBdaBwdStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t grad_output_dims[]{2, 1, 2, 16};
    const std::vector<float> grad_output(64, 1.0F);
    auto* device_grad_output = allocate<float>(64);
    ASSERT_NE(device_grad_output, nullptr);
    const std::int64_t residual_dims[]{2, 1, 2, 16};
    const std::vector<float> residual(64, 0.5F);
    auto* device_residual = allocate<float>(64);
    ASSERT_NE(device_residual, nullptr);
    const std::int64_t x_dims[]{2, 1, 16};
    const std::vector<float> x(32, 0.5F);
    auto* device_x = allocate<float>(32);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t bias_dims[]{16};
    const std::vector<float> bias(16, 0.25F);
    auto* device_bias = allocate<float>(16);
    ASSERT_NE(device_bias, nullptr);
    const std::int64_t h_res_dims[]{2, 1, 2, 2};
    const std::vector<float> h_res(8, 0.25F);
    auto* device_h_res = allocate<float>(8);
    ASSERT_NE(device_h_res, nullptr);
    const std::int64_t h_post_dims[]{2, 1, 2};
    const std::vector<float> h_post(4, 0.5F);
    auto* device_h_post = allocate<float>(4);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t grad_residual_dims[]{2, 1, 2, 16};
    const std::vector<float> grad_residual(64, 123.0F);
    auto* device_grad_residual = allocate<float>(64);
    ASSERT_NE(device_grad_residual, nullptr);
    const std::int64_t grad_x_dims[]{2, 1, 16};
    const std::vector<float> grad_x(32, 123.0F);
    auto* device_grad_x = allocate<float>(32);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_h_res_dims[]{2, 1, 2, 2};
    const std::vector<float> grad_h_res(8, 123.0F);
    auto* device_grad_h_res = allocate<float>(8);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_h_post_dims[]{2, 1, 2};
    const std::vector<float> grad_h_post(4, 123.0F);
    auto* device_grad_h_post = allocate<float>(4);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_bias_dims[]{16};
    const std::vector<float> grad_bias(16, 123.0F);
    auto* device_grad_bias = allocate<float>(16);
    ASSERT_NE(device_grad_bias, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_grad_output, grad_output.data(), grad_output.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_residual, residual.data(), residual.size() * sizeof(float), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_bias, bias.data(), bias.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_res, h_res.data(), h_res.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_residual, grad_residual.data(), grad_residual.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_output_view = makeView(device_grad_output, grad_output_dims, 4, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto residual_view =
        makeView(device_residual, residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto bias_view = makeView(device_bias, bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_res_view =
        makeView(device_h_res, h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_residual_view = makeView(device_grad_residual, grad_residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP32,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_output_view, residual_view, x_view, bias_view, h_res_view, h_post_view,
                        grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
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
    EXPECT_STREQ(primitive, "Fp32HPostBdaBwd");
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
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_residual, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
            }
        }
        {
            std::vector<float> actual(32);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.0F, 2e-5 + 2e-4 * std::abs(1.0F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_res, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 8.0F, 2e-5 + 2e-4 * std::abs(8.0F)) << i;
            }
        }
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_post, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 12.0F, 2e-5 + 2e-4 * std::abs(12.0F)) << i;
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 2.0F, 2e-5 + 2e-4 * std::abs(2.0F)) << i;
            }
        }
        ASSERT_EQ(ftrainHPostBdaBwd(grad_output_view, residual_view, x_view, bias_view, h_res_view, h_post_view,
                                    grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
                                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_residual, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
            }
        }
        {
            std::vector<float> actual(32);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.0F, 2e-5 + 2e-4 * std::abs(1.0F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_res, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 8.0F, 2e-5 + 2e-4 * std::abs(8.0F)) << i;
            }
        }
        {
            std::vector<float> actual(4);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_h_post, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 12.0F, 2e-5 + 2e-4 * std::abs(12.0F)) << i;
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 2.0F, 2e-5 + 2e-4 * std::abs(2.0F)) << i;
            }
        }
    }
    auto invalid_grad_output           = grad_output_view;
    invalid_grad_output.is_host_memory = true;
    EXPECT_EQ(ftrainHPostBdaBwd(invalid_grad_output, residual_view, x_view, bias_view, h_res_view, h_post_view,
                                grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
                                FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_output              = grad_output_view;
    invalid_grad_output.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainHPostBdaBwd(invalid_grad_output, residual_view, x_view, bias_view, h_res_view, h_post_view,
                                grad_residual_view, grad_x_view, grad_h_res_view, grad_h_post_view, grad_bias_view,
                                FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(HPostBdaBwdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_grad_output_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> first_grad_output(64, 0x3f80);
    auto* first_device_grad_output = allocate<std::uint16_t>(64);
    ASSERT_NE(first_device_grad_output, nullptr);
    const std::int64_t first_residual_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> first_residual(64, 0x3f00);
    auto* first_device_residual = allocate<std::uint16_t>(64);
    ASSERT_NE(first_device_residual, nullptr);
    const std::int64_t first_x_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> first_x(32, 0x3f00);
    auto* first_device_x = allocate<std::uint16_t>(32);
    ASSERT_NE(first_device_x, nullptr);
    const std::int64_t first_bias_dims[]{16};
    const std::vector<float> first_bias(16, 0.25F);
    auto* first_device_bias = allocate<float>(16);
    ASSERT_NE(first_device_bias, nullptr);
    const std::int64_t first_h_res_dims[]{2, 1, 2, 2};
    const std::vector<std::uint16_t> first_h_res(8, 0x3e80);
    auto* first_device_h_res = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_h_res, nullptr);
    const std::int64_t first_h_post_dims[]{2, 1, 2};
    const std::vector<std::uint16_t> first_h_post(4, 0x3f00);
    auto* first_device_h_post = allocate<std::uint16_t>(4);
    ASSERT_NE(first_device_h_post, nullptr);
    const std::int64_t first_grad_residual_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> first_grad_residual(64, 0x42f6);
    auto* first_device_grad_residual = allocate<std::uint16_t>(64);
    ASSERT_NE(first_device_grad_residual, nullptr);
    const std::int64_t first_grad_x_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> first_grad_x(32, 0x42f6);
    auto* first_device_grad_x = allocate<std::uint16_t>(32);
    ASSERT_NE(first_device_grad_x, nullptr);
    const std::int64_t first_grad_h_res_dims[]{2, 1, 2, 2};
    const std::vector<std::uint16_t> first_grad_h_res(8, 0x42f6);
    auto* first_device_grad_h_res = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_grad_h_res, nullptr);
    const std::int64_t first_grad_h_post_dims[]{2, 1, 2};
    const std::vector<std::uint16_t> first_grad_h_post(4, 0x42f6);
    auto* first_device_grad_h_post = allocate<std::uint16_t>(4);
    ASSERT_NE(first_device_grad_h_post, nullptr);
    const std::int64_t first_grad_bias_dims[]{16};
    const std::vector<float> first_grad_bias(16, 123.0F);
    auto* first_device_grad_bias = allocate<float>(16);
    ASSERT_NE(first_device_grad_bias, nullptr);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_output, first_grad_output.data(),
                             first_grad_output.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_residual, first_residual.data(),
                             first_residual.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_x, first_x.data(), first_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_bias, first_bias.data(), first_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_h_res, first_h_res.data(), first_h_res.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_h_post, first_h_post.data(), first_h_post.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_residual, first_grad_residual.data(),
                             first_grad_residual.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_x, first_grad_x.data(), first_grad_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_h_res, first_grad_h_res.data(),
                             first_grad_h_res.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_h_post, first_grad_h_post.data(),
                             first_grad_h_post.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_bias, first_grad_bias.data(), first_grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_grad_output_view = makeView(first_device_grad_output, first_grad_output_dims, 4,
                                           FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_residual_view    = makeView(first_device_residual, first_residual_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_x_view =
        makeView(first_device_x, first_x_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_bias_view =
        makeView(first_device_bias, first_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_h_res_view         = makeView(first_device_h_res, first_h_res_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_h_post_view        = makeView(first_device_h_post, first_h_post_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_residual_view = makeView(first_device_grad_residual, first_grad_residual_dims, 4,
                                             FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_x_view        = makeView(first_device_grad_x, first_grad_x_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                             FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_h_res_view  = makeView(first_device_grad_h_res, first_grad_h_res_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_h_post_view = makeView(first_device_grad_h_post, first_grad_h_post_dims, 3,
                                           FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_bias_view   = makeView(first_device_grad_bias, first_grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    const std::int64_t second_grad_output_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> second_grad_output(64, 0x3f80);
    auto* second_device_grad_output = allocate<std::uint16_t>(64);
    ASSERT_NE(second_device_grad_output, nullptr);
    const std::int64_t second_residual_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> second_residual(64, 0x3f00);
    auto* second_device_residual = allocate<std::uint16_t>(64);
    ASSERT_NE(second_device_residual, nullptr);
    const std::int64_t second_x_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> second_x(32, 0x3f00);
    auto* second_device_x = allocate<std::uint16_t>(32);
    ASSERT_NE(second_device_x, nullptr);
    const std::int64_t second_bias_dims[]{16};
    const std::vector<float> second_bias(16, 0.25F);
    auto* second_device_bias = allocate<float>(16);
    ASSERT_NE(second_device_bias, nullptr);
    const std::int64_t second_h_res_dims[]{2, 1, 2, 2};
    const std::vector<std::uint16_t> second_h_res(8, 0x3e80);
    auto* second_device_h_res = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_h_res, nullptr);
    const std::int64_t second_h_post_dims[]{2, 1, 2};
    const std::vector<std::uint16_t> second_h_post(4, 0x3f00);
    auto* second_device_h_post = allocate<std::uint16_t>(4);
    ASSERT_NE(second_device_h_post, nullptr);
    const std::int64_t second_grad_residual_dims[]{2, 1, 2, 16};
    const std::vector<std::uint16_t> second_grad_residual(64, 0x42f6);
    auto* second_device_grad_residual = allocate<std::uint16_t>(64);
    ASSERT_NE(second_device_grad_residual, nullptr);
    const std::int64_t second_grad_x_dims[]{2, 1, 16};
    const std::vector<std::uint16_t> second_grad_x(32, 0x42f6);
    auto* second_device_grad_x = allocate<std::uint16_t>(32);
    ASSERT_NE(second_device_grad_x, nullptr);
    const std::int64_t second_grad_h_res_dims[]{2, 1, 2, 2};
    const std::vector<std::uint16_t> second_grad_h_res(8, 0x42f6);
    auto* second_device_grad_h_res = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_grad_h_res, nullptr);
    const std::int64_t second_grad_h_post_dims[]{2, 1, 2};
    const std::vector<std::uint16_t> second_grad_h_post(4, 0x42f6);
    auto* second_device_grad_h_post = allocate<std::uint16_t>(4);
    ASSERT_NE(second_device_grad_h_post, nullptr);
    const std::int64_t second_grad_bias_dims[]{16};
    const std::vector<float> second_grad_bias(16, 123.0F);
    auto* second_device_grad_bias = allocate<float>(16);
    ASSERT_NE(second_device_grad_bias, nullptr);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_output, second_grad_output.data(),
                             second_grad_output.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_residual, second_residual.data(),
                             second_residual.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_x, second_x.data(), second_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_bias, second_bias.data(), second_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_h_res, second_h_res.data(), second_h_res.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_h_post, second_h_post.data(), second_h_post.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_residual, second_grad_residual.data(),
                             second_grad_residual.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_x, second_grad_x.data(), second_grad_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_h_res, second_grad_h_res.data(),
                             second_grad_h_res.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_h_post, second_grad_h_post.data(),
                             second_grad_h_post.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_bias, second_grad_bias.data(), second_grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_grad_output_view = makeView(second_device_grad_output, second_grad_output_dims, 4,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_residual_view    = makeView(second_device_residual, second_residual_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_x_view =
        makeView(second_device_x, second_x_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_bias_view          = makeView(second_device_bias, second_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_h_res_view         = makeView(second_device_h_res, second_h_res_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_h_post_view        = makeView(second_device_h_post, second_h_post_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_residual_view = makeView(second_device_grad_residual, second_grad_residual_dims, 4,
                                              FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_x_view        = makeView(second_device_grad_x, second_grad_x_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                              FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_h_res_view    = makeView(second_device_grad_h_res, second_grad_h_res_dims, 4,
                                              FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_h_post_view   = makeView(second_device_grad_h_post, second_grad_h_post_dims, 3,
                                              FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_bias_view = makeView(second_device_grad_bias, second_grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args            = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_grad_output_view, first_residual_view, first_x_view, first_bias_view,
                        first_h_res_view, first_h_post_view, first_grad_residual_view, first_grad_x_view,
                        first_grad_h_res_view, first_grad_h_post_view, first_grad_bias_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_grad_output_view, second_residual_view, second_x_view, second_bias_view,
                        second_h_res_view, second_h_post_view, second_grad_residual_view, second_grad_x_view,
                        second_grad_h_res_view, second_grad_h_post_view, second_grad_bias_view,
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
        std::vector<std::uint16_t> actual(64);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_residual, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.5F, 2e-3 + 1e-2 * std::abs(0.5F)) << i;
        }
    }
    {
        std::vector<std::uint16_t> actual(32);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_x, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 1.0F, 2e-3 + 1e-2 * std::abs(1.0F)) << i;
        }
    }
    {
        std::vector<std::uint16_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_h_res, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 8.0F, 2e-3 + 1e-2 * std::abs(8.0F)) << i;
        }
    }
    {
        std::vector<std::uint16_t> actual(4);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_h_post, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 12.0F, 2e-3 + 1e-2 * std::abs(12.0F)) << i;
        }
    }
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_bias, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 2.0F, 2e-5 + 2e-4 * std::abs(2.0F)) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(64);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_residual, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.5F, 2e-3 + 1e-2 * std::abs(0.5F)) << i;
        }
    }
    {
        std::vector<std::uint16_t> actual(32);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_x, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 1.0F, 2e-3 + 1e-2 * std::abs(1.0F)) << i;
        }
    }
    {
        std::vector<std::uint16_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_h_res, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 8.0F, 2e-3 + 1e-2 * std::abs(8.0F)) << i;
        }
    }
    {
        std::vector<std::uint16_t> actual(4);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_h_post, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 12.0F, 2e-3 + 1e-2 * std::abs(12.0F)) << i;
        }
    }
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_bias, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 2.0F, 2e-5 + 2e-4 * std::abs(2.0F)) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(64);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_residual, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.5F, 2e-3 + 1e-2 * std::abs(0.5F)) << i;
        }
    }
    {
        std::vector<std::uint16_t> actual(32);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_x, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 1.0F, 2e-3 + 1e-2 * std::abs(1.0F)) << i;
        }
    }
    {
        std::vector<std::uint16_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_h_res, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 8.0F, 2e-3 + 1e-2 * std::abs(8.0F)) << i;
        }
    }
    {
        std::vector<std::uint16_t> actual(4);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_h_post, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 12.0F, 2e-3 + 1e-2 * std::abs(12.0F)) << i;
        }
    }
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_bias, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 2.0F, 2e-5 + 2e-4 * std::abs(2.0F)) << i;
        }
    }
}

}  // namespace
