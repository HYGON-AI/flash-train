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

#include "sinkhorn_bwd_staged_fixture.hpp"

// Uniform m_init and constant upstream gradients give zero logit gradients by symmetry.
// Nonuniform gradients are checked against a host reference in sinkhorn_bwd_numerics_test.cpp.
namespace {

using ftrain_test::SinkhornBwdStagedTest;

TEST_F(SinkhornBwdStagedTest, Bf16SinkhornBwdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t grad_output_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> grad_output(8, 0x3f80);
    auto* device_grad_output = allocate<std::uint16_t>(8);
    ASSERT_NE(device_grad_output, nullptr);
    const std::int64_t m_init_dims[]{2, 2, 2};
    const std::vector<float> m_init(8, 0.5F);
    auto* device_m_init = allocate<float>(8);
    ASSERT_NE(device_m_init, nullptr);
    const std::int64_t grad_logits_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> grad_logits(8, 0x42f6);
    auto* device_grad_logits = allocate<std::uint16_t>(8);
    ASSERT_NE(device_grad_logits, nullptr);
    std::int64_t num_iterations = 2;
    float eps                   = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_output, grad_output.data(), grad_output.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_m_init, m_init.data(), m_init.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_logits, grad_logits.data(), grad_logits.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_output_view = makeView(device_grad_output, grad_output_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto m_init_view =
        makeView(device_m_init, m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_logits_view = makeView(device_grad_logits, grad_logits_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto num_iterations_view =
        makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view   = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_output_view, m_init_view, grad_logits_view, num_iterations_view, eps_view,
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
    EXPECT_STREQ(primitive, "Bf16SinkhornBwd");
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
            std::vector<std::uint16_t> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_logits, actual.size() * sizeof(std::uint16_t),
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
        ASSERT_EQ(ftrainSinkhornBwd(grad_output_view, m_init_view, grad_logits_view, num_iterations_view, eps_view,
                                    FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_logits, actual.size() * sizeof(std::uint16_t),
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
    }
    auto invalid_grad_output           = grad_output_view;
    invalid_grad_output.is_host_memory = true;
    EXPECT_EQ(ftrainSinkhornBwd(invalid_grad_output, m_init_view, grad_logits_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_output              = grad_output_view;
    invalid_grad_output.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSinkhornBwd(invalid_grad_output, m_init_view, grad_logits_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SinkhornBwdStagedTest, Fp16SinkhornBwdStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t grad_output_dims[]{2, 2, 2};
    const std::vector<__half> grad_output(8, __float2half(1.0F));
    auto* device_grad_output = allocate<__half>(8);
    ASSERT_NE(device_grad_output, nullptr);
    const std::int64_t m_init_dims[]{2, 2, 2};
    const std::vector<float> m_init(8, 0.5F);
    auto* device_m_init = allocate<float>(8);
    ASSERT_NE(device_m_init, nullptr);
    const std::int64_t grad_logits_dims[]{2, 2, 2};
    const std::vector<__half> grad_logits(8, __float2half(123.0F));
    auto* device_grad_logits = allocate<__half>(8);
    ASSERT_NE(device_grad_logits, nullptr);
    std::int64_t num_iterations = 2;
    float eps                   = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_output, grad_output.data(), grad_output.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_m_init, m_init.data(), m_init.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_logits, grad_logits.data(), grad_logits.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_output_view = makeView(device_grad_output, grad_output_dims, 3, FTRAIN_NUMERIC_TYPE_FP16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto m_init_view =
        makeView(device_m_init, m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_logits_view = makeView(device_grad_logits, grad_logits_dims, 3, FTRAIN_NUMERIC_TYPE_FP16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto num_iterations_view =
        makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view   = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_output_view, m_init_view, grad_logits_view, num_iterations_view, eps_view,
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
    EXPECT_STREQ(primitive, "Fp16SinkhornBwd");
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
            std::vector<__half> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_logits, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        ASSERT_EQ(ftrainSinkhornBwd(grad_output_view, m_init_view, grad_logits_view, num_iterations_view, eps_view,
                                    FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_logits, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
    }
    auto invalid_grad_output           = grad_output_view;
    invalid_grad_output.is_host_memory = true;
    EXPECT_EQ(ftrainSinkhornBwd(invalid_grad_output, m_init_view, grad_logits_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_output              = grad_output_view;
    invalid_grad_output.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSinkhornBwd(invalid_grad_output, m_init_view, grad_logits_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SinkhornBwdStagedTest, Fp32SinkhornBwdStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t grad_output_dims[]{2, 2, 2};
    const std::vector<float> grad_output(8, 1.0F);
    auto* device_grad_output = allocate<float>(8);
    ASSERT_NE(device_grad_output, nullptr);
    const std::int64_t m_init_dims[]{2, 2, 2};
    const std::vector<float> m_init(8, 0.5F);
    auto* device_m_init = allocate<float>(8);
    ASSERT_NE(device_m_init, nullptr);
    const std::int64_t grad_logits_dims[]{2, 2, 2};
    const std::vector<float> grad_logits(8, 123.0F);
    auto* device_grad_logits = allocate<float>(8);
    ASSERT_NE(device_grad_logits, nullptr);
    std::int64_t num_iterations = 2;
    float eps                   = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_grad_output, grad_output.data(), grad_output.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_m_init, m_init.data(), m_init.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_grad_logits, grad_logits.data(), grad_logits.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto grad_output_view = makeView(device_grad_output, grad_output_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto m_init_view =
        makeView(device_m_init, m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_logits_view = makeView(device_grad_logits, grad_logits_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto num_iterations_view =
        makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view   = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, grad_output_view, m_init_view, grad_logits_view, num_iterations_view, eps_view,
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
    EXPECT_STREQ(primitive, "Fp32SinkhornBwd");
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
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_logits, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        ASSERT_EQ(ftrainSinkhornBwd(grad_output_view, m_init_view, grad_logits_view, num_iterations_view, eps_view,
                                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_logits, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
    }
    auto invalid_grad_output           = grad_output_view;
    invalid_grad_output.is_host_memory = true;
    EXPECT_EQ(ftrainSinkhornBwd(invalid_grad_output, m_init_view, grad_logits_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_grad_output              = grad_output_view;
    invalid_grad_output.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSinkhornBwd(invalid_grad_output, m_init_view, grad_logits_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SinkhornBwdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_grad_output_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> first_grad_output(8, 0x3f80);
    auto* first_device_grad_output = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_grad_output, nullptr);
    const std::int64_t first_m_init_dims[]{2, 2, 2};
    const std::vector<float> first_m_init(8, 0.5F);
    auto* first_device_m_init = allocate<float>(8);
    ASSERT_NE(first_device_m_init, nullptr);
    const std::int64_t first_grad_logits_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> first_grad_logits(8, 0x42f6);
    auto* first_device_grad_logits = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_grad_logits, nullptr);
    std::int64_t first_num_iterations = 2;
    float first_eps                   = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_output, first_grad_output.data(),
                             first_grad_output.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_m_init, first_m_init.data(), first_m_init.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_grad_logits, first_grad_logits.data(),
                             first_grad_logits.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_grad_output_view = makeView(first_device_grad_output, first_grad_output_dims, 3,
                                           FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_m_init_view      = makeView(first_device_m_init, first_m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_logits_view = makeView(first_device_grad_logits, first_grad_logits_dims, 3,
                                           FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_num_iterations_view =
        makeView(&first_num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_eps_view =
        makeView(&first_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_grad_output_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> second_grad_output(8, 0x3f80);
    auto* second_device_grad_output = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_grad_output, nullptr);
    const std::int64_t second_m_init_dims[]{2, 2, 2};
    const std::vector<float> second_m_init(8, 0.5F);
    auto* second_device_m_init = allocate<float>(8);
    ASSERT_NE(second_device_m_init, nullptr);
    const std::int64_t second_grad_logits_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> second_grad_logits(8, 0x42f6);
    auto* second_device_grad_logits = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_grad_logits, nullptr);
    std::int64_t second_num_iterations = 2;
    float second_eps                   = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_output, second_grad_output.data(),
                             second_grad_output.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_m_init, second_m_init.data(), second_m_init.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_grad_logits, second_grad_logits.data(),
                             second_grad_logits.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_grad_output_view = makeView(second_device_grad_output, second_grad_output_dims, 3,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_m_init_view      = makeView(second_device_m_init, second_m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_logits_view = makeView(second_device_grad_logits, second_grad_logits_dims, 3,
                                            FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_num_iterations_view =
        makeView(&second_num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_eps_view =
        makeView(&second_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_grad_output_view, first_m_init_view, first_grad_logits_view,
                        first_num_iterations_view, first_eps_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_grad_output_view, second_m_init_view, second_grad_logits_view,
                        second_num_iterations_view, second_eps_view, FTRAIN_NUMERIC_TYPE_BF16));
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
        std::vector<std::uint16_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_logits, actual.size() * sizeof(std::uint16_t),
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
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_logits, actual.size() * sizeof(std::uint16_t),
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
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_logits, actual.size() * sizeof(std::uint16_t),
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
}

}  // namespace
