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

#include "proj_rms_compute_h_bwd_staged_fixture.hpp"

namespace {

using ftrain_test::ProjRmsComputeHBwdStagedTest;

TEST_F(ProjRmsComputeHBwdStagedTest, MixedFp32ProjRmsComputeHBwdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t grad_h_pre_dims[]{16, 2};
    const std::vector<std::uint16_t> grad_h_pre(32, 0x0000);
    auto* device_grad_h_pre = allocate<std::uint16_t>(32);
    ASSERT_NE(device_grad_h_pre, nullptr);
    const std::int64_t grad_h_post_dims[]{16, 2};
    const std::vector<std::uint16_t> grad_h_post(32, 0x0000);
    auto* device_grad_h_post = allocate<std::uint16_t>(32);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_h_res_dims[]{16, 4};
    const std::vector<std::uint16_t> grad_h_res(64, 0x0000);
    auto* device_grad_h_res = allocate<std::uint16_t>(64);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_r_dims[]{16, 1};
    const std::vector<std::uint16_t> grad_r(16, 0x0000);
    auto* device_grad_r = allocate<std::uint16_t>(16);
    ASSERT_NE(device_grad_r, nullptr);
    const std::int64_t h_pre_dims[]{16, 2};
    const std::vector<std::uint16_t> h_pre(32, 0x3f00);
    auto* device_h_pre = allocate<std::uint16_t>(32);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t h_post_dims[]{16, 2};
    const std::vector<std::uint16_t> h_post(32, 0x3f80);
    auto* device_h_post = allocate<std::uint16_t>(32);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<std::uint16_t> proj(128, 0x4080);
    auto* device_proj = allocate<std::uint16_t>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<std::uint16_t> r(16, 0x4000);
    auto* device_r = allocate<std::uint16_t>(16);
    ASSERT_NE(device_r, nullptr);
    const std::int64_t x_dims[]{16, 32};
    const std::vector<std::uint16_t> x(512, 0x3f00);
    auto* device_x = allocate<std::uint16_t>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<__half> weight(256, __float2half(0.25F));
    auto* device_weight = allocate<__half>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t alpha_pre_dims[]{1};
    const std::vector<float> alpha_pre(1, 1.0F);
    auto* device_alpha_pre = allocate<float>(1);
    ASSERT_NE(device_alpha_pre, nullptr);
    const std::int64_t alpha_post_dims[]{1};
    const std::vector<float> alpha_post(1, 1.0F);
    auto* device_alpha_post = allocate<float>(1);
    ASSERT_NE(device_alpha_post, nullptr);
    const std::int64_t alpha_res_dims[]{1};
    const std::vector<float> alpha_res(1, 1.0F);
    auto* device_alpha_res = allocate<float>(1);
    ASSERT_NE(device_alpha_res, nullptr);
    const std::int64_t grad_x_dims[]{16, 32};
    const std::vector<std::uint16_t> grad_x(512, 0x42f6);
    auto* device_grad_x = allocate<std::uint16_t>(512);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_weight_dims[]{8, 32};
    const std::vector<__half> grad_weight(256, __float2half(123.0F));
    auto* device_grad_weight = allocate<__half>(256);
    ASSERT_NE(device_grad_weight, nullptr);
    const std::int64_t grad_alpha_dims[]{3};
    const std::vector<float> grad_alpha(3, 123.0F);
    auto* device_grad_alpha = allocate<float>(3);
    ASSERT_NE(device_grad_alpha, nullptr);
    const std::int64_t grad_bias_dims[]{8};
    const std::vector<float> grad_bias(8, 123.0F);
    auto* device_grad_bias = allocate<float>(8);
    ASSERT_NE(device_grad_bias, nullptr);
    std::int64_t n      = 2;
    float eps           = 1e-06F;
    float compute_h_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_r, grad_r.data(), grad_r.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_pre, alpha_pre.data(), alpha_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_post, alpha_post.data(), alpha_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_res, alpha_res.data(), alpha_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_weight, grad_weight.data(), grad_weight.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_alpha, grad_alpha.data(), grad_alpha.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_r_view =
        makeView(device_grad_r, grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_pre_view =
        makeView(device_alpha_pre, alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_post_view =
        makeView(device_alpha_post, alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_res_view =
        makeView(device_alpha_res, alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_weight_view = makeView(device_grad_weight, grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_alpha_view =
        makeView(device_grad_alpha, grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto n_view   = makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto compute_h_eps_view =
        makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view, h_post_view,
                        proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view, alpha_res_view,
                        grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view, eps_view,
                        compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsComputeHBwd");
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
            std::vector<std::uint16_t> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<__half> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        ASSERT_EQ(ftrainProjRmsComputeHBwd(grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                           h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view,
                                           alpha_post_view, alpha_res_view, grad_x_view, grad_weight_view,
                                           grad_alpha_view, grad_bias_view, n_view, eps_view, compute_h_eps_view,
                                           FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<__half> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
    }
    auto invalid_grad_h_pre           = grad_h_pre_view;
    invalid_grad_h_pre.is_host_memory = true;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_h_pre              = grad_h_pre_view;
    invalid_grad_h_pre.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsComputeHBwdStagedTest, MixedFp32ProjRmsComputeHBwdStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t grad_h_pre_dims[]{16, 2};
    const std::vector<std::uint16_t> grad_h_pre(32, 0x0000);
    auto* device_grad_h_pre = allocate<std::uint16_t>(32);
    ASSERT_NE(device_grad_h_pre, nullptr);
    const std::int64_t grad_h_post_dims[]{16, 2};
    const std::vector<std::uint16_t> grad_h_post(32, 0x0000);
    auto* device_grad_h_post = allocate<std::uint16_t>(32);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_h_res_dims[]{16, 4};
    const std::vector<std::uint16_t> grad_h_res(64, 0x0000);
    auto* device_grad_h_res = allocate<std::uint16_t>(64);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_r_dims[]{16, 1};
    const std::vector<std::uint16_t> grad_r(16, 0x0000);
    auto* device_grad_r = allocate<std::uint16_t>(16);
    ASSERT_NE(device_grad_r, nullptr);
    const std::int64_t h_pre_dims[]{16, 2};
    const std::vector<std::uint16_t> h_pre(32, 0x3f00);
    auto* device_h_pre = allocate<std::uint16_t>(32);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t h_post_dims[]{16, 2};
    const std::vector<std::uint16_t> h_post(32, 0x3f80);
    auto* device_h_post = allocate<std::uint16_t>(32);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<std::uint16_t> proj(128, 0x4080);
    auto* device_proj = allocate<std::uint16_t>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<std::uint16_t> r(16, 0x4000);
    auto* device_r = allocate<std::uint16_t>(16);
    ASSERT_NE(device_r, nullptr);
    const std::int64_t x_dims[]{16, 32};
    const std::vector<std::uint16_t> x(512, 0x3f00);
    auto* device_x = allocate<std::uint16_t>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<float> weight(256, 0.25F);
    auto* device_weight = allocate<float>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t alpha_pre_dims[]{1};
    const std::vector<float> alpha_pre(1, 1.0F);
    auto* device_alpha_pre = allocate<float>(1);
    ASSERT_NE(device_alpha_pre, nullptr);
    const std::int64_t alpha_post_dims[]{1};
    const std::vector<float> alpha_post(1, 1.0F);
    auto* device_alpha_post = allocate<float>(1);
    ASSERT_NE(device_alpha_post, nullptr);
    const std::int64_t alpha_res_dims[]{1};
    const std::vector<float> alpha_res(1, 1.0F);
    auto* device_alpha_res = allocate<float>(1);
    ASSERT_NE(device_alpha_res, nullptr);
    const std::int64_t grad_x_dims[]{16, 32};
    const std::vector<std::uint16_t> grad_x(512, 0x42f6);
    auto* device_grad_x = allocate<std::uint16_t>(512);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_weight_dims[]{8, 32};
    const std::vector<float> grad_weight(256, 123.0F);
    auto* device_grad_weight = allocate<float>(256);
    ASSERT_NE(device_grad_weight, nullptr);
    const std::int64_t grad_alpha_dims[]{3};
    const std::vector<float> grad_alpha(3, 123.0F);
    auto* device_grad_alpha = allocate<float>(3);
    ASSERT_NE(device_grad_alpha, nullptr);
    const std::int64_t grad_bias_dims[]{8};
    const std::vector<float> grad_bias(8, 123.0F);
    auto* device_grad_bias = allocate<float>(8);
    ASSERT_NE(device_grad_bias, nullptr);
    std::int64_t n      = 2;
    float eps           = 1e-06F;
    float compute_h_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_r, grad_r.data(), grad_r.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_pre, alpha_pre.data(), alpha_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_post, alpha_post.data(), alpha_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_res, alpha_res.data(), alpha_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_weight, grad_weight.data(), grad_weight.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_alpha, grad_alpha.data(), grad_alpha.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_r_view =
        makeView(device_grad_r, grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_pre_view =
        makeView(device_alpha_pre, alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_post_view =
        makeView(device_alpha_post, alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_res_view =
        makeView(device_alpha_res, alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_weight_view = makeView(device_grad_weight, grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_alpha_view =
        makeView(device_grad_alpha, grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto n_view   = makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto compute_h_eps_view =
        makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view, h_post_view,
                        proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view, alpha_res_view,
                        grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view, eps_view,
                        compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsComputeHBwd");
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
            std::vector<std::uint16_t> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        ASSERT_EQ(ftrainProjRmsComputeHBwd(grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                           h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view,
                                           alpha_post_view, alpha_res_view, grad_x_view, grad_weight_view,
                                           grad_alpha_view, grad_bias_view, n_view, eps_view, compute_h_eps_view,
                                           FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
    }
    auto invalid_grad_h_pre           = grad_h_pre_view;
    invalid_grad_h_pre.is_host_memory = true;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_h_pre              = grad_h_pre_view;
    invalid_grad_h_pre.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsComputeHBwdStagedTest, Bf16ProjRmsComputeHBwdStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t grad_h_pre_dims[]{16, 2};
    const std::vector<std::uint16_t> grad_h_pre(32, 0x0000);
    auto* device_grad_h_pre = allocate<std::uint16_t>(32);
    ASSERT_NE(device_grad_h_pre, nullptr);
    const std::int64_t grad_h_post_dims[]{16, 2};
    const std::vector<std::uint16_t> grad_h_post(32, 0x0000);
    auto* device_grad_h_post = allocate<std::uint16_t>(32);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_h_res_dims[]{16, 4};
    const std::vector<std::uint16_t> grad_h_res(64, 0x0000);
    auto* device_grad_h_res = allocate<std::uint16_t>(64);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_r_dims[]{16, 1};
    const std::vector<std::uint16_t> grad_r(16, 0x0000);
    auto* device_grad_r = allocate<std::uint16_t>(16);
    ASSERT_NE(device_grad_r, nullptr);
    const std::int64_t h_pre_dims[]{16, 2};
    const std::vector<std::uint16_t> h_pre(32, 0x3f00);
    auto* device_h_pre = allocate<std::uint16_t>(32);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t h_post_dims[]{16, 2};
    const std::vector<std::uint16_t> h_post(32, 0x3f80);
    auto* device_h_post = allocate<std::uint16_t>(32);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<std::uint16_t> proj(128, 0x4080);
    auto* device_proj = allocate<std::uint16_t>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<std::uint16_t> r(16, 0x4000);
    auto* device_r = allocate<std::uint16_t>(16);
    ASSERT_NE(device_r, nullptr);
    const std::int64_t x_dims[]{16, 32};
    const std::vector<std::uint16_t> x(512, 0x3f00);
    auto* device_x = allocate<std::uint16_t>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<std::uint16_t> weight(256, 0x3e80);
    auto* device_weight = allocate<std::uint16_t>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t alpha_pre_dims[]{1};
    const std::vector<float> alpha_pre(1, 1.0F);
    auto* device_alpha_pre = allocate<float>(1);
    ASSERT_NE(device_alpha_pre, nullptr);
    const std::int64_t alpha_post_dims[]{1};
    const std::vector<float> alpha_post(1, 1.0F);
    auto* device_alpha_post = allocate<float>(1);
    ASSERT_NE(device_alpha_post, nullptr);
    const std::int64_t alpha_res_dims[]{1};
    const std::vector<float> alpha_res(1, 1.0F);
    auto* device_alpha_res = allocate<float>(1);
    ASSERT_NE(device_alpha_res, nullptr);
    const std::int64_t grad_x_dims[]{16, 32};
    const std::vector<std::uint16_t> grad_x(512, 0x42f6);
    auto* device_grad_x = allocate<std::uint16_t>(512);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_weight_dims[]{8, 32};
    const std::vector<std::uint16_t> grad_weight(256, 0x42f6);
    auto* device_grad_weight = allocate<std::uint16_t>(256);
    ASSERT_NE(device_grad_weight, nullptr);
    const std::int64_t grad_alpha_dims[]{3};
    const std::vector<float> grad_alpha(3, 123.0F);
    auto* device_grad_alpha = allocate<float>(3);
    ASSERT_NE(device_grad_alpha, nullptr);
    const std::int64_t grad_bias_dims[]{8};
    const std::vector<float> grad_bias(8, 123.0F);
    auto* device_grad_bias = allocate<float>(8);
    ASSERT_NE(device_grad_bias, nullptr);
    std::int64_t n      = 2;
    float eps           = 1e-06F;
    float compute_h_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_r, grad_r.data(), grad_r.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_pre, alpha_pre.data(), alpha_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_post, alpha_post.data(), alpha_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_res, alpha_res.data(), alpha_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_weight, grad_weight.data(), grad_weight.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_alpha, grad_alpha.data(), grad_alpha.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_r_view =
        makeView(device_grad_r, grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_pre_view =
        makeView(device_alpha_pre, alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_post_view =
        makeView(device_alpha_post, alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_res_view =
        makeView(device_alpha_res, alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_weight_view = makeView(device_grad_weight, grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_alpha_view =
        makeView(device_grad_alpha, grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto n_view   = makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto compute_h_eps_view =
        makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view, h_post_view,
                        proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view, alpha_res_view,
                        grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view, eps_view,
                        compute_h_eps_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Bf16ProjRmsComputeHBwd");
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
            std::vector<std::uint16_t> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<std::uint16_t> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        ASSERT_EQ(ftrainProjRmsComputeHBwd(grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                           h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view,
                                           alpha_post_view, alpha_res_view, grad_x_view, grad_weight_view,
                                           grad_alpha_view, grad_bias_view, n_view, eps_view, compute_h_eps_view,
                                           FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<std::uint16_t> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
    }
    auto invalid_grad_h_pre           = grad_h_pre_view;
    invalid_grad_h_pre.is_host_memory = true;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
        FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_h_pre              = grad_h_pre_view;
    invalid_grad_h_pre.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
        FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsComputeHBwdStagedTest, MixedFp32ProjRmsComputeHBwdStorageMode3RunsConvenienceAndStagedApis) {
    const std::int64_t grad_h_pre_dims[]{16, 2};
    const std::vector<__half> grad_h_pre(32, __float2half(0.0F));
    auto* device_grad_h_pre = allocate<__half>(32);
    ASSERT_NE(device_grad_h_pre, nullptr);
    const std::int64_t grad_h_post_dims[]{16, 2};
    const std::vector<__half> grad_h_post(32, __float2half(0.0F));
    auto* device_grad_h_post = allocate<__half>(32);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_h_res_dims[]{16, 4};
    const std::vector<__half> grad_h_res(64, __float2half(0.0F));
    auto* device_grad_h_res = allocate<__half>(64);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_r_dims[]{16, 1};
    const std::vector<__half> grad_r(16, __float2half(0.0F));
    auto* device_grad_r = allocate<__half>(16);
    ASSERT_NE(device_grad_r, nullptr);
    const std::int64_t h_pre_dims[]{16, 2};
    const std::vector<__half> h_pre(32, __float2half(0.5F));
    auto* device_h_pre = allocate<__half>(32);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t h_post_dims[]{16, 2};
    const std::vector<__half> h_post(32, __float2half(1.0F));
    auto* device_h_post = allocate<__half>(32);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<__half> proj(128, __float2half(4.0F));
    auto* device_proj = allocate<__half>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<__half> r(16, __float2half(2.0F));
    auto* device_r = allocate<__half>(16);
    ASSERT_NE(device_r, nullptr);
    const std::int64_t x_dims[]{16, 32};
    const std::vector<__half> x(512, __float2half(0.5F));
    auto* device_x = allocate<__half>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<std::uint16_t> weight(256, 0x3e80);
    auto* device_weight = allocate<std::uint16_t>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t alpha_pre_dims[]{1};
    const std::vector<float> alpha_pre(1, 1.0F);
    auto* device_alpha_pre = allocate<float>(1);
    ASSERT_NE(device_alpha_pre, nullptr);
    const std::int64_t alpha_post_dims[]{1};
    const std::vector<float> alpha_post(1, 1.0F);
    auto* device_alpha_post = allocate<float>(1);
    ASSERT_NE(device_alpha_post, nullptr);
    const std::int64_t alpha_res_dims[]{1};
    const std::vector<float> alpha_res(1, 1.0F);
    auto* device_alpha_res = allocate<float>(1);
    ASSERT_NE(device_alpha_res, nullptr);
    const std::int64_t grad_x_dims[]{16, 32};
    const std::vector<__half> grad_x(512, __float2half(123.0F));
    auto* device_grad_x = allocate<__half>(512);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_weight_dims[]{8, 32};
    const std::vector<std::uint16_t> grad_weight(256, 0x42f6);
    auto* device_grad_weight = allocate<std::uint16_t>(256);
    ASSERT_NE(device_grad_weight, nullptr);
    const std::int64_t grad_alpha_dims[]{3};
    const std::vector<float> grad_alpha(3, 123.0F);
    auto* device_grad_alpha = allocate<float>(3);
    ASSERT_NE(device_grad_alpha, nullptr);
    const std::int64_t grad_bias_dims[]{8};
    const std::vector<float> grad_bias(8, 123.0F);
    auto* device_grad_bias = allocate<float>(8);
    ASSERT_NE(device_grad_bias, nullptr);
    std::int64_t n      = 2;
    float eps           = 1e-06F;
    float compute_h_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_r, grad_r.data(), grad_r.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_pre, alpha_pre.data(), alpha_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_post, alpha_post.data(), alpha_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_res, alpha_res.data(), alpha_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_weight, grad_weight.data(), grad_weight.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_alpha, grad_alpha.data(), grad_alpha.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_r_view =
        makeView(device_grad_r, grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_pre_view =
        makeView(device_alpha_pre, alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_post_view =
        makeView(device_alpha_post, alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_res_view =
        makeView(device_alpha_res, alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_weight_view = makeView(device_grad_weight, grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_alpha_view =
        makeView(device_grad_alpha, grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto n_view   = makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto compute_h_eps_view =
        makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view, h_post_view,
                        proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view, alpha_res_view,
                        grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view, eps_view,
                        compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsComputeHBwd");
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
            std::vector<__half> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<std::uint16_t> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        ASSERT_EQ(ftrainProjRmsComputeHBwd(grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                           h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view,
                                           alpha_post_view, alpha_res_view, grad_x_view, grad_weight_view,
                                           grad_alpha_view, grad_bias_view, n_view, eps_view, compute_h_eps_view,
                                           FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<std::uint16_t> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
    }
    auto invalid_grad_h_pre           = grad_h_pre_view;
    invalid_grad_h_pre.is_host_memory = true;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_h_pre              = grad_h_pre_view;
    invalid_grad_h_pre.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsComputeHBwdStagedTest, MixedFp32ProjRmsComputeHBwdStorageMode4RunsConvenienceAndStagedApis) {
    const std::int64_t grad_h_pre_dims[]{16, 2};
    const std::vector<__half> grad_h_pre(32, __float2half(0.0F));
    auto* device_grad_h_pre = allocate<__half>(32);
    ASSERT_NE(device_grad_h_pre, nullptr);
    const std::int64_t grad_h_post_dims[]{16, 2};
    const std::vector<__half> grad_h_post(32, __float2half(0.0F));
    auto* device_grad_h_post = allocate<__half>(32);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_h_res_dims[]{16, 4};
    const std::vector<__half> grad_h_res(64, __float2half(0.0F));
    auto* device_grad_h_res = allocate<__half>(64);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_r_dims[]{16, 1};
    const std::vector<__half> grad_r(16, __float2half(0.0F));
    auto* device_grad_r = allocate<__half>(16);
    ASSERT_NE(device_grad_r, nullptr);
    const std::int64_t h_pre_dims[]{16, 2};
    const std::vector<__half> h_pre(32, __float2half(0.5F));
    auto* device_h_pre = allocate<__half>(32);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t h_post_dims[]{16, 2};
    const std::vector<__half> h_post(32, __float2half(1.0F));
    auto* device_h_post = allocate<__half>(32);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<__half> proj(128, __float2half(4.0F));
    auto* device_proj = allocate<__half>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<__half> r(16, __float2half(2.0F));
    auto* device_r = allocate<__half>(16);
    ASSERT_NE(device_r, nullptr);
    const std::int64_t x_dims[]{16, 32};
    const std::vector<__half> x(512, __float2half(0.5F));
    auto* device_x = allocate<__half>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<float> weight(256, 0.25F);
    auto* device_weight = allocate<float>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t alpha_pre_dims[]{1};
    const std::vector<float> alpha_pre(1, 1.0F);
    auto* device_alpha_pre = allocate<float>(1);
    ASSERT_NE(device_alpha_pre, nullptr);
    const std::int64_t alpha_post_dims[]{1};
    const std::vector<float> alpha_post(1, 1.0F);
    auto* device_alpha_post = allocate<float>(1);
    ASSERT_NE(device_alpha_post, nullptr);
    const std::int64_t alpha_res_dims[]{1};
    const std::vector<float> alpha_res(1, 1.0F);
    auto* device_alpha_res = allocate<float>(1);
    ASSERT_NE(device_alpha_res, nullptr);
    const std::int64_t grad_x_dims[]{16, 32};
    const std::vector<__half> grad_x(512, __float2half(123.0F));
    auto* device_grad_x = allocate<__half>(512);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_weight_dims[]{8, 32};
    const std::vector<float> grad_weight(256, 123.0F);
    auto* device_grad_weight = allocate<float>(256);
    ASSERT_NE(device_grad_weight, nullptr);
    const std::int64_t grad_alpha_dims[]{3};
    const std::vector<float> grad_alpha(3, 123.0F);
    auto* device_grad_alpha = allocate<float>(3);
    ASSERT_NE(device_grad_alpha, nullptr);
    const std::int64_t grad_bias_dims[]{8};
    const std::vector<float> grad_bias(8, 123.0F);
    auto* device_grad_bias = allocate<float>(8);
    ASSERT_NE(device_grad_bias, nullptr);
    std::int64_t n      = 2;
    float eps           = 1e-06F;
    float compute_h_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_r, grad_r.data(), grad_r.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_pre, alpha_pre.data(), alpha_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_post, alpha_post.data(), alpha_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_res, alpha_res.data(), alpha_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_weight, grad_weight.data(), grad_weight.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_alpha, grad_alpha.data(), grad_alpha.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_r_view =
        makeView(device_grad_r, grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_pre_view =
        makeView(device_alpha_pre, alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_post_view =
        makeView(device_alpha_post, alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_res_view =
        makeView(device_alpha_res, alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_weight_view = makeView(device_grad_weight, grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_alpha_view =
        makeView(device_grad_alpha, grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto n_view   = makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto compute_h_eps_view =
        makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view, h_post_view,
                        proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view, alpha_res_view,
                        grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view, eps_view,
                        compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsComputeHBwd");
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
            std::vector<__half> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<float> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        ASSERT_EQ(ftrainProjRmsComputeHBwd(grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                           h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view,
                                           alpha_post_view, alpha_res_view, grad_x_view, grad_weight_view,
                                           grad_alpha_view, grad_bias_view, n_view, eps_view, compute_h_eps_view,
                                           FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<float> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
    }
    auto invalid_grad_h_pre           = grad_h_pre_view;
    invalid_grad_h_pre.is_host_memory = true;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_h_pre              = grad_h_pre_view;
    invalid_grad_h_pre.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsComputeHBwdStagedTest, Fp16ProjRmsComputeHBwdStorageMode5RunsConvenienceAndStagedApis) {
    const std::int64_t grad_h_pre_dims[]{16, 2};
    const std::vector<__half> grad_h_pre(32, __float2half(0.0F));
    auto* device_grad_h_pre = allocate<__half>(32);
    ASSERT_NE(device_grad_h_pre, nullptr);
    const std::int64_t grad_h_post_dims[]{16, 2};
    const std::vector<__half> grad_h_post(32, __float2half(0.0F));
    auto* device_grad_h_post = allocate<__half>(32);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_h_res_dims[]{16, 4};
    const std::vector<__half> grad_h_res(64, __float2half(0.0F));
    auto* device_grad_h_res = allocate<__half>(64);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_r_dims[]{16, 1};
    const std::vector<__half> grad_r(16, __float2half(0.0F));
    auto* device_grad_r = allocate<__half>(16);
    ASSERT_NE(device_grad_r, nullptr);
    const std::int64_t h_pre_dims[]{16, 2};
    const std::vector<__half> h_pre(32, __float2half(0.5F));
    auto* device_h_pre = allocate<__half>(32);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t h_post_dims[]{16, 2};
    const std::vector<__half> h_post(32, __float2half(1.0F));
    auto* device_h_post = allocate<__half>(32);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<__half> proj(128, __float2half(4.0F));
    auto* device_proj = allocate<__half>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<__half> r(16, __float2half(2.0F));
    auto* device_r = allocate<__half>(16);
    ASSERT_NE(device_r, nullptr);
    const std::int64_t x_dims[]{16, 32};
    const std::vector<__half> x(512, __float2half(0.5F));
    auto* device_x = allocate<__half>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<__half> weight(256, __float2half(0.25F));
    auto* device_weight = allocate<__half>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t alpha_pre_dims[]{1};
    const std::vector<float> alpha_pre(1, 1.0F);
    auto* device_alpha_pre = allocate<float>(1);
    ASSERT_NE(device_alpha_pre, nullptr);
    const std::int64_t alpha_post_dims[]{1};
    const std::vector<float> alpha_post(1, 1.0F);
    auto* device_alpha_post = allocate<float>(1);
    ASSERT_NE(device_alpha_post, nullptr);
    const std::int64_t alpha_res_dims[]{1};
    const std::vector<float> alpha_res(1, 1.0F);
    auto* device_alpha_res = allocate<float>(1);
    ASSERT_NE(device_alpha_res, nullptr);
    const std::int64_t grad_x_dims[]{16, 32};
    const std::vector<__half> grad_x(512, __float2half(123.0F));
    auto* device_grad_x = allocate<__half>(512);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_weight_dims[]{8, 32};
    const std::vector<__half> grad_weight(256, __float2half(123.0F));
    auto* device_grad_weight = allocate<__half>(256);
    ASSERT_NE(device_grad_weight, nullptr);
    const std::int64_t grad_alpha_dims[]{3};
    const std::vector<float> grad_alpha(3, 123.0F);
    auto* device_grad_alpha = allocate<float>(3);
    ASSERT_NE(device_grad_alpha, nullptr);
    const std::int64_t grad_bias_dims[]{8};
    const std::vector<float> grad_bias(8, 123.0F);
    auto* device_grad_bias = allocate<float>(8);
    ASSERT_NE(device_grad_bias, nullptr);
    std::int64_t n      = 2;
    float eps           = 1e-06F;
    float compute_h_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_r, grad_r.data(), grad_r.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_pre, alpha_pre.data(), alpha_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_post, alpha_post.data(), alpha_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_res, alpha_res.data(), alpha_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_weight, grad_weight.data(), grad_weight.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_alpha, grad_alpha.data(), grad_alpha.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_r_view =
        makeView(device_grad_r, grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_pre_view =
        makeView(device_alpha_pre, alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_post_view =
        makeView(device_alpha_post, alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_res_view =
        makeView(device_alpha_res, alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_weight_view = makeView(device_grad_weight, grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_alpha_view =
        makeView(device_grad_alpha, grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto n_view   = makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto compute_h_eps_view =
        makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view, h_post_view,
                        proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view, alpha_res_view,
                        grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view, eps_view,
                        compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp16ProjRmsComputeHBwd");
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
            std::vector<__half> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<__half> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        ASSERT_EQ(ftrainProjRmsComputeHBwd(grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                           h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view,
                                           alpha_post_view, alpha_res_view, grad_x_view, grad_weight_view,
                                           grad_alpha_view, grad_bias_view, n_view, eps_view, compute_h_eps_view,
                                           FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<__half> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
    }
    auto invalid_grad_h_pre           = grad_h_pre_view;
    invalid_grad_h_pre.is_host_memory = true;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
        FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_h_pre              = grad_h_pre_view;
    invalid_grad_h_pre.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
        FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsComputeHBwdStagedTest, MixedFp32ProjRmsComputeHBwdStorageMode6RunsConvenienceAndStagedApis) {
    const std::int64_t grad_h_pre_dims[]{16, 2};
    const std::vector<float> grad_h_pre(32, 0.0F);
    auto* device_grad_h_pre = allocate<float>(32);
    ASSERT_NE(device_grad_h_pre, nullptr);
    const std::int64_t grad_h_post_dims[]{16, 2};
    const std::vector<float> grad_h_post(32, 0.0F);
    auto* device_grad_h_post = allocate<float>(32);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_h_res_dims[]{16, 4};
    const std::vector<float> grad_h_res(64, 0.0F);
    auto* device_grad_h_res = allocate<float>(64);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_r_dims[]{16, 1};
    const std::vector<float> grad_r(16, 0.0F);
    auto* device_grad_r = allocate<float>(16);
    ASSERT_NE(device_grad_r, nullptr);
    const std::int64_t h_pre_dims[]{16, 2};
    const std::vector<float> h_pre(32, 0.5F);
    auto* device_h_pre = allocate<float>(32);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t h_post_dims[]{16, 2};
    const std::vector<float> h_post(32, 1.0F);
    auto* device_h_post = allocate<float>(32);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<float> proj(128, 4.0F);
    auto* device_proj = allocate<float>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<float> r(16, 2.0F);
    auto* device_r = allocate<float>(16);
    ASSERT_NE(device_r, nullptr);
    const std::int64_t x_dims[]{16, 32};
    const std::vector<float> x(512, 0.5F);
    auto* device_x = allocate<float>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<std::uint16_t> weight(256, 0x3e80);
    auto* device_weight = allocate<std::uint16_t>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t alpha_pre_dims[]{1};
    const std::vector<float> alpha_pre(1, 1.0F);
    auto* device_alpha_pre = allocate<float>(1);
    ASSERT_NE(device_alpha_pre, nullptr);
    const std::int64_t alpha_post_dims[]{1};
    const std::vector<float> alpha_post(1, 1.0F);
    auto* device_alpha_post = allocate<float>(1);
    ASSERT_NE(device_alpha_post, nullptr);
    const std::int64_t alpha_res_dims[]{1};
    const std::vector<float> alpha_res(1, 1.0F);
    auto* device_alpha_res = allocate<float>(1);
    ASSERT_NE(device_alpha_res, nullptr);
    const std::int64_t grad_x_dims[]{16, 32};
    const std::vector<float> grad_x(512, 123.0F);
    auto* device_grad_x = allocate<float>(512);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_weight_dims[]{8, 32};
    const std::vector<std::uint16_t> grad_weight(256, 0x42f6);
    auto* device_grad_weight = allocate<std::uint16_t>(256);
    ASSERT_NE(device_grad_weight, nullptr);
    const std::int64_t grad_alpha_dims[]{3};
    const std::vector<float> grad_alpha(3, 123.0F);
    auto* device_grad_alpha = allocate<float>(3);
    ASSERT_NE(device_grad_alpha, nullptr);
    const std::int64_t grad_bias_dims[]{8};
    const std::vector<float> grad_bias(8, 123.0F);
    auto* device_grad_bias = allocate<float>(8);
    ASSERT_NE(device_grad_bias, nullptr);
    std::int64_t n      = 2;
    float eps           = 1e-06F;
    float compute_h_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_r, grad_r.data(), grad_r.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_pre, alpha_pre.data(), alpha_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_post, alpha_post.data(), alpha_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_res, alpha_res.data(), alpha_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_weight, grad_weight.data(), grad_weight.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_alpha, grad_alpha.data(), grad_alpha.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_r_view =
        makeView(device_grad_r, grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_pre_view =
        makeView(device_alpha_pre, alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_post_view =
        makeView(device_alpha_post, alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_res_view =
        makeView(device_alpha_res, alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_weight_view = makeView(device_grad_weight, grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_alpha_view =
        makeView(device_grad_alpha, grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto n_view   = makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto compute_h_eps_view =
        makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view, h_post_view,
                        proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view, alpha_res_view,
                        grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view, eps_view,
                        compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsComputeHBwd");
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
            std::vector<float> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<std::uint16_t> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        ASSERT_EQ(ftrainProjRmsComputeHBwd(grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                           h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view,
                                           alpha_post_view, alpha_res_view, grad_x_view, grad_weight_view,
                                           grad_alpha_view, grad_bias_view, n_view, eps_view, compute_h_eps_view,
                                           FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<std::uint16_t> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
    }
    auto invalid_grad_h_pre           = grad_h_pre_view;
    invalid_grad_h_pre.is_host_memory = true;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_h_pre              = grad_h_pre_view;
    invalid_grad_h_pre.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsComputeHBwdStagedTest, MixedFp32ProjRmsComputeHBwdStorageMode7RunsConvenienceAndStagedApis) {
    const std::int64_t grad_h_pre_dims[]{16, 2};
    const std::vector<float> grad_h_pre(32, 0.0F);
    auto* device_grad_h_pre = allocate<float>(32);
    ASSERT_NE(device_grad_h_pre, nullptr);
    const std::int64_t grad_h_post_dims[]{16, 2};
    const std::vector<float> grad_h_post(32, 0.0F);
    auto* device_grad_h_post = allocate<float>(32);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_h_res_dims[]{16, 4};
    const std::vector<float> grad_h_res(64, 0.0F);
    auto* device_grad_h_res = allocate<float>(64);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_r_dims[]{16, 1};
    const std::vector<float> grad_r(16, 0.0F);
    auto* device_grad_r = allocate<float>(16);
    ASSERT_NE(device_grad_r, nullptr);
    const std::int64_t h_pre_dims[]{16, 2};
    const std::vector<float> h_pre(32, 0.5F);
    auto* device_h_pre = allocate<float>(32);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t h_post_dims[]{16, 2};
    const std::vector<float> h_post(32, 1.0F);
    auto* device_h_post = allocate<float>(32);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<float> proj(128, 4.0F);
    auto* device_proj = allocate<float>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<float> r(16, 2.0F);
    auto* device_r = allocate<float>(16);
    ASSERT_NE(device_r, nullptr);
    const std::int64_t x_dims[]{16, 32};
    const std::vector<float> x(512, 0.5F);
    auto* device_x = allocate<float>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<__half> weight(256, __float2half(0.25F));
    auto* device_weight = allocate<__half>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t alpha_pre_dims[]{1};
    const std::vector<float> alpha_pre(1, 1.0F);
    auto* device_alpha_pre = allocate<float>(1);
    ASSERT_NE(device_alpha_pre, nullptr);
    const std::int64_t alpha_post_dims[]{1};
    const std::vector<float> alpha_post(1, 1.0F);
    auto* device_alpha_post = allocate<float>(1);
    ASSERT_NE(device_alpha_post, nullptr);
    const std::int64_t alpha_res_dims[]{1};
    const std::vector<float> alpha_res(1, 1.0F);
    auto* device_alpha_res = allocate<float>(1);
    ASSERT_NE(device_alpha_res, nullptr);
    const std::int64_t grad_x_dims[]{16, 32};
    const std::vector<float> grad_x(512, 123.0F);
    auto* device_grad_x = allocate<float>(512);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_weight_dims[]{8, 32};
    const std::vector<__half> grad_weight(256, __float2half(123.0F));
    auto* device_grad_weight = allocate<__half>(256);
    ASSERT_NE(device_grad_weight, nullptr);
    const std::int64_t grad_alpha_dims[]{3};
    const std::vector<float> grad_alpha(3, 123.0F);
    auto* device_grad_alpha = allocate<float>(3);
    ASSERT_NE(device_grad_alpha, nullptr);
    const std::int64_t grad_bias_dims[]{8};
    const std::vector<float> grad_bias(8, 123.0F);
    auto* device_grad_bias = allocate<float>(8);
    ASSERT_NE(device_grad_bias, nullptr);
    std::int64_t n      = 2;
    float eps           = 1e-06F;
    float compute_h_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_r, grad_r.data(), grad_r.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_pre, alpha_pre.data(), alpha_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_post, alpha_post.data(), alpha_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_res, alpha_res.data(), alpha_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_weight, grad_weight.data(), grad_weight.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_alpha, grad_alpha.data(), grad_alpha.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_r_view =
        makeView(device_grad_r, grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_pre_view =
        makeView(device_alpha_pre, alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_post_view =
        makeView(device_alpha_post, alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_res_view =
        makeView(device_alpha_res, alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_weight_view = makeView(device_grad_weight, grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_alpha_view =
        makeView(device_grad_alpha, grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto n_view   = makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto compute_h_eps_view =
        makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view, h_post_view,
                        proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view, alpha_res_view,
                        grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view, eps_view,
                        compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsComputeHBwd");
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
            std::vector<float> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<__half> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        ASSERT_EQ(ftrainProjRmsComputeHBwd(grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                           h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view,
                                           alpha_post_view, alpha_res_view, grad_x_view, grad_weight_view,
                                           grad_alpha_view, grad_bias_view, n_view, eps_view, compute_h_eps_view,
                                           FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<__half> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
    }
    auto invalid_grad_h_pre           = grad_h_pre_view;
    invalid_grad_h_pre.is_host_memory = true;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_h_pre              = grad_h_pre_view;
    invalid_grad_h_pre.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsComputeHBwdStagedTest, Fp32ProjRmsComputeHBwdStorageMode8RunsConvenienceAndStagedApis) {
    const std::int64_t grad_h_pre_dims[]{16, 2};
    const std::vector<float> grad_h_pre(32, 0.0F);
    auto* device_grad_h_pre = allocate<float>(32);
    ASSERT_NE(device_grad_h_pre, nullptr);
    const std::int64_t grad_h_post_dims[]{16, 2};
    const std::vector<float> grad_h_post(32, 0.0F);
    auto* device_grad_h_post = allocate<float>(32);
    ASSERT_NE(device_grad_h_post, nullptr);
    const std::int64_t grad_h_res_dims[]{16, 4};
    const std::vector<float> grad_h_res(64, 0.0F);
    auto* device_grad_h_res = allocate<float>(64);
    ASSERT_NE(device_grad_h_res, nullptr);
    const std::int64_t grad_r_dims[]{16, 1};
    const std::vector<float> grad_r(16, 0.0F);
    auto* device_grad_r = allocate<float>(16);
    ASSERT_NE(device_grad_r, nullptr);
    const std::int64_t h_pre_dims[]{16, 2};
    const std::vector<float> h_pre(32, 0.5F);
    auto* device_h_pre = allocate<float>(32);
    ASSERT_NE(device_h_pre, nullptr);
    const std::int64_t h_post_dims[]{16, 2};
    const std::vector<float> h_post(32, 1.0F);
    auto* device_h_post = allocate<float>(32);
    ASSERT_NE(device_h_post, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<float> proj(128, 4.0F);
    auto* device_proj = allocate<float>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<float> r(16, 2.0F);
    auto* device_r = allocate<float>(16);
    ASSERT_NE(device_r, nullptr);
    const std::int64_t x_dims[]{16, 32};
    const std::vector<float> x(512, 0.5F);
    auto* device_x = allocate<float>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<float> weight(256, 0.25F);
    auto* device_weight = allocate<float>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t alpha_pre_dims[]{1};
    const std::vector<float> alpha_pre(1, 1.0F);
    auto* device_alpha_pre = allocate<float>(1);
    ASSERT_NE(device_alpha_pre, nullptr);
    const std::int64_t alpha_post_dims[]{1};
    const std::vector<float> alpha_post(1, 1.0F);
    auto* device_alpha_post = allocate<float>(1);
    ASSERT_NE(device_alpha_post, nullptr);
    const std::int64_t alpha_res_dims[]{1};
    const std::vector<float> alpha_res(1, 1.0F);
    auto* device_alpha_res = allocate<float>(1);
    ASSERT_NE(device_alpha_res, nullptr);
    const std::int64_t grad_x_dims[]{16, 32};
    const std::vector<float> grad_x(512, 123.0F);
    auto* device_grad_x = allocate<float>(512);
    ASSERT_NE(device_grad_x, nullptr);
    const std::int64_t grad_weight_dims[]{8, 32};
    const std::vector<float> grad_weight(256, 123.0F);
    auto* device_grad_weight = allocate<float>(256);
    ASSERT_NE(device_grad_weight, nullptr);
    const std::int64_t grad_alpha_dims[]{3};
    const std::vector<float> grad_alpha(3, 123.0F);
    auto* device_grad_alpha = allocate<float>(3);
    ASSERT_NE(device_grad_alpha, nullptr);
    const std::int64_t grad_bias_dims[]{8};
    const std::vector<float> grad_bias(8, 123.0F);
    auto* device_grad_bias = allocate<float>(8);
    ASSERT_NE(device_grad_bias, nullptr);
    std::int64_t n      = 2;
    float eps           = 1e-06F;
    float compute_h_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_pre, grad_h_pre.data(), grad_h_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_post, grad_h_post.data(), grad_h_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_h_res, grad_h_res.data(), grad_h_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_r, grad_r.data(), grad_r.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_h_pre, h_pre.data(), h_pre.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_h_post, h_post.data(), h_post.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_pre, alpha_pre.data(), alpha_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_post, alpha_post.data(), alpha_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_alpha_res, alpha_res.data(), alpha_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_grad_x, grad_x.data(), grad_x.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_weight, grad_weight.data(), grad_weight.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_alpha, grad_alpha.data(), grad_alpha.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_bias, grad_bias.data(), grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_h_pre_view =
        makeView(device_grad_h_pre, grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_post_view = makeView(device_grad_h_post, grad_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_h_res_view =
        makeView(device_grad_h_res, grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_r_view =
        makeView(device_grad_r, grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_pre_view =
        makeView(device_h_pre, h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto h_post_view =
        makeView(device_h_post, h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto x_view    = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_pre_view =
        makeView(device_alpha_pre, alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_post_view =
        makeView(device_alpha_post, alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto alpha_res_view =
        makeView(device_alpha_res, alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_x_view =
        makeView(device_grad_x, grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_weight_view = makeView(device_grad_weight, grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_alpha_view =
        makeView(device_grad_alpha, grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_bias_view =
        makeView(device_grad_bias, grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto n_view   = makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto compute_h_eps_view =
        makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view, h_post_view,
                        proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view, alpha_res_view,
                        grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view, eps_view,
                        compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp32ProjRmsComputeHBwd");
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
            std::vector<float> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        ASSERT_EQ(ftrainProjRmsComputeHBwd(grad_h_pre_view, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                           h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view,
                                           alpha_post_view, alpha_res_view, grad_x_view, grad_weight_view,
                                           grad_alpha_view, grad_bias_view, n_view, eps_view, compute_h_eps_view,
                                           FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(512);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_x, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_weight, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(3);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_alpha, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_bias, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
    }
    auto invalid_grad_h_pre           = grad_h_pre_view;
    invalid_grad_h_pre.is_host_memory = true;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_h_pre              = grad_h_pre_view;
    invalid_grad_h_pre.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(
        ftrainProjRmsComputeHBwd(invalid_grad_h_pre, grad_h_post_view, grad_h_res_view, grad_r_view, h_pre_view,
                                 h_post_view, proj_view, r_view, x_view, weight_view, alpha_pre_view, alpha_post_view,
                                 alpha_res_view, grad_x_view, grad_weight_view, grad_alpha_view, grad_bias_view, n_view,
                                 eps_view, compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
        FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsComputeHBwdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_grad_h_pre_dims[]{16, 2};
    const std::vector<std::uint16_t> first_grad_h_pre(32, 0x0000);
    auto* first_device_grad_h_pre = allocate<std::uint16_t>(32);
    ASSERT_NE(first_device_grad_h_pre, nullptr);
    const std::int64_t first_grad_h_post_dims[]{16, 2};
    const std::vector<std::uint16_t> first_grad_h_post(32, 0x0000);
    auto* first_device_grad_h_post = allocate<std::uint16_t>(32);
    ASSERT_NE(first_device_grad_h_post, nullptr);
    const std::int64_t first_grad_h_res_dims[]{16, 4};
    const std::vector<std::uint16_t> first_grad_h_res(64, 0x0000);
    auto* first_device_grad_h_res = allocate<std::uint16_t>(64);
    ASSERT_NE(first_device_grad_h_res, nullptr);
    const std::int64_t first_grad_r_dims[]{16, 1};
    const std::vector<std::uint16_t> first_grad_r(16, 0x0000);
    auto* first_device_grad_r = allocate<std::uint16_t>(16);
    ASSERT_NE(first_device_grad_r, nullptr);
    const std::int64_t first_h_pre_dims[]{16, 2};
    const std::vector<std::uint16_t> first_h_pre(32, 0x3f00);
    auto* first_device_h_pre = allocate<std::uint16_t>(32);
    ASSERT_NE(first_device_h_pre, nullptr);
    const std::int64_t first_h_post_dims[]{16, 2};
    const std::vector<std::uint16_t> first_h_post(32, 0x3f80);
    auto* first_device_h_post = allocate<std::uint16_t>(32);
    ASSERT_NE(first_device_h_post, nullptr);
    const std::int64_t first_proj_dims[]{16, 8};
    const std::vector<std::uint16_t> first_proj(128, 0x4080);
    auto* first_device_proj = allocate<std::uint16_t>(128);
    ASSERT_NE(first_device_proj, nullptr);
    const std::int64_t first_r_dims[]{16, 1};
    const std::vector<std::uint16_t> first_r(16, 0x4000);
    auto* first_device_r = allocate<std::uint16_t>(16);
    ASSERT_NE(first_device_r, nullptr);
    const std::int64_t first_x_dims[]{16, 32};
    const std::vector<std::uint16_t> first_x(512, 0x3f00);
    auto* first_device_x = allocate<std::uint16_t>(512);
    ASSERT_NE(first_device_x, nullptr);
    const std::int64_t first_weight_dims[]{8, 32};
    const std::vector<__half> first_weight(256, __float2half(0.25F));
    auto* first_device_weight = allocate<__half>(256);
    ASSERT_NE(first_device_weight, nullptr);
    const std::int64_t first_alpha_pre_dims[]{1};
    const std::vector<float> first_alpha_pre(1, 1.0F);
    auto* first_device_alpha_pre = allocate<float>(1);
    ASSERT_NE(first_device_alpha_pre, nullptr);
    const std::int64_t first_alpha_post_dims[]{1};
    const std::vector<float> first_alpha_post(1, 1.0F);
    auto* first_device_alpha_post = allocate<float>(1);
    ASSERT_NE(first_device_alpha_post, nullptr);
    const std::int64_t first_alpha_res_dims[]{1};
    const std::vector<float> first_alpha_res(1, 1.0F);
    auto* first_device_alpha_res = allocate<float>(1);
    ASSERT_NE(first_device_alpha_res, nullptr);
    const std::int64_t first_grad_x_dims[]{16, 32};
    const std::vector<std::uint16_t> first_grad_x(512, 0x42f6);
    auto* first_device_grad_x = allocate<std::uint16_t>(512);
    ASSERT_NE(first_device_grad_x, nullptr);
    const std::int64_t first_grad_weight_dims[]{8, 32};
    const std::vector<__half> first_grad_weight(256, __float2half(123.0F));
    auto* first_device_grad_weight = allocate<__half>(256);
    ASSERT_NE(first_device_grad_weight, nullptr);
    const std::int64_t first_grad_alpha_dims[]{3};
    const std::vector<float> first_grad_alpha(3, 123.0F);
    auto* first_device_grad_alpha = allocate<float>(3);
    ASSERT_NE(first_device_grad_alpha, nullptr);
    const std::int64_t first_grad_bias_dims[]{8};
    const std::vector<float> first_grad_bias(8, 123.0F);
    auto* first_device_grad_bias = allocate<float>(8);
    ASSERT_NE(first_device_grad_bias, nullptr);
    std::int64_t first_n      = 2;
    float first_eps           = 1e-06F;
    float first_compute_h_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_h_pre, first_grad_h_pre.data(),
                             first_grad_h_pre.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_h_post, first_grad_h_post.data(),
                             first_grad_h_post.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_h_res, first_grad_h_res.data(),
                             first_grad_h_res.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_r, first_grad_r.data(), first_grad_r.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_h_pre, first_h_pre.data(), first_h_pre.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_h_post, first_h_post.data(), first_h_post.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_proj, first_proj.data(), first_proj.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_r, first_r.data(), first_r.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_x, first_x.data(), first_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_weight, first_weight.data(), first_weight.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_alpha_pre, first_alpha_pre.data(), first_alpha_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_alpha_post, first_alpha_post.data(), first_alpha_post.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_alpha_res, first_alpha_res.data(), first_alpha_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_x, first_grad_x.data(), first_grad_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_weight, first_grad_weight.data(),
                             first_grad_weight.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_alpha, first_grad_alpha.data(), first_grad_alpha.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_bias, first_grad_bias.data(), first_grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_grad_h_pre_view  = makeView(first_device_grad_h_pre, first_grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_h_post_view = makeView(first_device_grad_h_post, first_grad_h_post_dims, 2,
                                           FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_h_res_view  = makeView(first_device_grad_h_res, first_grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_r_view      = makeView(first_device_grad_r, first_grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_h_pre_view       = makeView(first_device_h_pre, first_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_h_post_view      = makeView(first_device_h_post, first_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_proj_view =
        makeView(first_device_proj, first_proj_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_r_view =
        makeView(first_device_r, first_r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_x_view =
        makeView(first_device_x, first_x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_weight_view      = makeView(first_device_weight, first_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_alpha_pre_view   = makeView(first_device_alpha_pre, first_alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_alpha_post_view  = makeView(first_device_alpha_post, first_alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_alpha_res_view   = makeView(first_device_alpha_res, first_alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_x_view      = makeView(first_device_grad_x, first_grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_weight_view = makeView(first_device_grad_weight, first_grad_weight_dims, 2,
                                           FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_alpha_view  = makeView(first_device_grad_alpha, first_grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_bias_view   = makeView(first_device_grad_bias, first_grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_n_view = makeView(&first_n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_eps_view =
        makeView(&first_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_compute_h_eps_view =
        makeView(&first_compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_grad_h_pre_dims[]{16, 2};
    const std::vector<std::uint16_t> second_grad_h_pre(32, 0x0000);
    auto* second_device_grad_h_pre = allocate<std::uint16_t>(32);
    ASSERT_NE(second_device_grad_h_pre, nullptr);
    const std::int64_t second_grad_h_post_dims[]{16, 2};
    const std::vector<std::uint16_t> second_grad_h_post(32, 0x0000);
    auto* second_device_grad_h_post = allocate<std::uint16_t>(32);
    ASSERT_NE(second_device_grad_h_post, nullptr);
    const std::int64_t second_grad_h_res_dims[]{16, 4};
    const std::vector<std::uint16_t> second_grad_h_res(64, 0x0000);
    auto* second_device_grad_h_res = allocate<std::uint16_t>(64);
    ASSERT_NE(second_device_grad_h_res, nullptr);
    const std::int64_t second_grad_r_dims[]{16, 1};
    const std::vector<std::uint16_t> second_grad_r(16, 0x0000);
    auto* second_device_grad_r = allocate<std::uint16_t>(16);
    ASSERT_NE(second_device_grad_r, nullptr);
    const std::int64_t second_h_pre_dims[]{16, 2};
    const std::vector<std::uint16_t> second_h_pre(32, 0x3f00);
    auto* second_device_h_pre = allocate<std::uint16_t>(32);
    ASSERT_NE(second_device_h_pre, nullptr);
    const std::int64_t second_h_post_dims[]{16, 2};
    const std::vector<std::uint16_t> second_h_post(32, 0x3f80);
    auto* second_device_h_post = allocate<std::uint16_t>(32);
    ASSERT_NE(second_device_h_post, nullptr);
    const std::int64_t second_proj_dims[]{16, 8};
    const std::vector<std::uint16_t> second_proj(128, 0x4080);
    auto* second_device_proj = allocate<std::uint16_t>(128);
    ASSERT_NE(second_device_proj, nullptr);
    const std::int64_t second_r_dims[]{16, 1};
    const std::vector<std::uint16_t> second_r(16, 0x4000);
    auto* second_device_r = allocate<std::uint16_t>(16);
    ASSERT_NE(second_device_r, nullptr);
    const std::int64_t second_x_dims[]{16, 32};
    const std::vector<std::uint16_t> second_x(512, 0x3f00);
    auto* second_device_x = allocate<std::uint16_t>(512);
    ASSERT_NE(second_device_x, nullptr);
    const std::int64_t second_weight_dims[]{8, 32};
    const std::vector<__half> second_weight(256, __float2half(0.25F));
    auto* second_device_weight = allocate<__half>(256);
    ASSERT_NE(second_device_weight, nullptr);
    const std::int64_t second_alpha_pre_dims[]{1};
    const std::vector<float> second_alpha_pre(1, 1.0F);
    auto* second_device_alpha_pre = allocate<float>(1);
    ASSERT_NE(second_device_alpha_pre, nullptr);
    const std::int64_t second_alpha_post_dims[]{1};
    const std::vector<float> second_alpha_post(1, 1.0F);
    auto* second_device_alpha_post = allocate<float>(1);
    ASSERT_NE(second_device_alpha_post, nullptr);
    const std::int64_t second_alpha_res_dims[]{1};
    const std::vector<float> second_alpha_res(1, 1.0F);
    auto* second_device_alpha_res = allocate<float>(1);
    ASSERT_NE(second_device_alpha_res, nullptr);
    const std::int64_t second_grad_x_dims[]{16, 32};
    const std::vector<std::uint16_t> second_grad_x(512, 0x42f6);
    auto* second_device_grad_x = allocate<std::uint16_t>(512);
    ASSERT_NE(second_device_grad_x, nullptr);
    const std::int64_t second_grad_weight_dims[]{8, 32};
    const std::vector<__half> second_grad_weight(256, __float2half(123.0F));
    auto* second_device_grad_weight = allocate<__half>(256);
    ASSERT_NE(second_device_grad_weight, nullptr);
    const std::int64_t second_grad_alpha_dims[]{3};
    const std::vector<float> second_grad_alpha(3, 123.0F);
    auto* second_device_grad_alpha = allocate<float>(3);
    ASSERT_NE(second_device_grad_alpha, nullptr);
    const std::int64_t second_grad_bias_dims[]{8};
    const std::vector<float> second_grad_bias(8, 123.0F);
    auto* second_device_grad_bias = allocate<float>(8);
    ASSERT_NE(second_device_grad_bias, nullptr);
    std::int64_t second_n      = 2;
    float second_eps           = 1e-06F;
    float second_compute_h_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_h_pre, second_grad_h_pre.data(),
                             second_grad_h_pre.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_h_post, second_grad_h_post.data(),
                             second_grad_h_post.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_h_res, second_grad_h_res.data(),
                             second_grad_h_res.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_r, second_grad_r.data(), second_grad_r.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_h_pre, second_h_pre.data(), second_h_pre.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_h_post, second_h_post.data(), second_h_post.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_proj, second_proj.data(), second_proj.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_r, second_r.data(), second_r.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_x, second_x.data(), second_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_weight, second_weight.data(), second_weight.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_alpha_pre, second_alpha_pre.data(), second_alpha_pre.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_alpha_post, second_alpha_post.data(),
                             second_alpha_post.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_alpha_res, second_alpha_res.data(), second_alpha_res.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_x, second_grad_x.data(), second_grad_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_weight, second_grad_weight.data(),
                             second_grad_weight.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_alpha, second_grad_alpha.data(),
                             second_grad_alpha.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_bias, second_grad_bias.data(), second_grad_bias.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_grad_h_pre_view  = makeView(second_device_grad_h_pre, second_grad_h_pre_dims, 2,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_h_post_view = makeView(second_device_grad_h_post, second_grad_h_post_dims, 2,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_h_res_view  = makeView(second_device_grad_h_res, second_grad_h_res_dims, 2,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_r_view      = makeView(second_device_grad_r, second_grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_h_pre_view       = makeView(second_device_h_pre, second_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_h_post_view      = makeView(second_device_h_post, second_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_proj_view        = makeView(second_device_proj, second_proj_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_r_view =
        makeView(second_device_r, second_r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_x_view =
        makeView(second_device_x, second_x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_weight_view      = makeView(second_device_weight, second_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_alpha_pre_view   = makeView(second_device_alpha_pre, second_alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_alpha_post_view  = makeView(second_device_alpha_post, second_alpha_post_dims, 1,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_alpha_res_view   = makeView(second_device_alpha_res, second_alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_x_view      = makeView(second_device_grad_x, second_grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_weight_view = makeView(second_device_grad_weight, second_grad_weight_dims, 2,
                                            FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_alpha_view  = makeView(second_device_grad_alpha, second_grad_alpha_dims, 1,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_bias_view   = makeView(second_device_grad_bias, second_grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_n_view = makeView(&second_n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_eps_view =
        makeView(&second_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_compute_h_eps_view =
        makeView(&second_compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_grad_h_pre_view, first_grad_h_post_view, first_grad_h_res_view, first_grad_r_view,
                        first_h_pre_view, first_h_post_view, first_proj_view, first_r_view, first_x_view,
                        first_weight_view, first_alpha_pre_view, first_alpha_post_view, first_alpha_res_view,
                        first_grad_x_view, first_grad_weight_view, first_grad_alpha_view, first_grad_bias_view,
                        first_n_view, first_eps_view, first_compute_h_eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_grad_h_pre_view, second_grad_h_post_view, second_grad_h_res_view,
                        second_grad_r_view, second_h_pre_view, second_h_post_view, second_proj_view, second_r_view,
                        second_x_view, second_weight_view, second_alpha_pre_view, second_alpha_post_view,
                        second_alpha_res_view, second_grad_x_view, second_grad_weight_view, second_grad_alpha_view,
                        second_grad_bias_view, second_n_view, second_eps_view, second_compute_h_eps_view,
                        FTRAIN_NUMERIC_TYPE_FP32));
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
        std::vector<std::uint16_t> actual(512);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_x, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
        }
    }
    {
        std::vector<__half> actual(256);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_weight, actual.size() * sizeof(__half),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
    }
    {
        std::vector<float> actual(3);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_alpha, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_bias, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(512);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_x, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
        }
    }
    {
        std::vector<__half> actual(256);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_weight, actual.size() * sizeof(__half),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
    }
    {
        std::vector<float> actual(3);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_alpha, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_bias, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(512);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_x, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.0F, 2e-3) << i;
        }
    }
    {
        std::vector<__half> actual(256);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_weight, actual.size() * sizeof(__half),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
    }
    {
        std::vector<float> actual(3);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_alpha, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_bias, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
    }
}

}  // namespace
