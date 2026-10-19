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

#include "sinkhorn_fwd_staged_fixture.hpp"

namespace {

using ftrain_test::SinkhornFwdStagedTest;

TEST_F(SinkhornFwdStagedTest, SinkhornConvenienceAndStagedMatchHostReference) {
    constexpr std::int64_t iterations = 3;
    constexpr float eps               = 1e-6F;
    const std::int64_t dims[]         = {1, 2, 2};
    std::array<float, 4> logits{0.2F, -0.3F, 0.7F, 0.1F}, expected{}, actual{}, staged{};
    for (int row = 0; row < 2; ++row) {
        const float sum = std::exp(logits[row * 2]) + std::exp(logits[row * 2 + 1]);
        for (int col = 0; col < 2; ++col) { expected[row * 2 + col] = std::exp(logits[row * 2 + col]) / sum + eps; }
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int row = 0; row < 2; ++row) {
            const float sum = expected[row * 2] + expected[row * 2 + 1] + eps;
            for (int col = 0; col < 2; ++col) { expected[row * 2 + col] /= sum; }
        }
        for (int col = 0; col < 2; ++col) {
            const float sum = expected[col] + expected[col + 2] + eps;
            for (int row = 0; row < 2; ++row) { expected[row * 2 + col] /= sum; }
        }
    }
    void* memory = nullptr;
    ASSERT_EQ(hipMalloc(&memory, 12 * sizeof(float)), hipSuccess);
    auto* input  = static_cast<float*>(memory);
    auto* output = input + 4;
    auto* state  = input + 8;
    hipStream_t stream{};
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(input, logits.data(), sizeof(logits), hipMemcpyHostToDevice, stream), hipSuccess);
    const auto tensor = [&](void* ptr) {
        return FTrainStorageView{ptr, dims, nullptr, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    };
    const FTrainStorageView count{
        const_cast<std::int64_t*>(&iterations), nullptr, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64,
        FTRAIN_INDEX_TYPE_CONTINUOUS,           true};
    const FTrainStorageView epsilon{const_cast<float*>(&eps),     nullptr, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32,
                                    FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    for (int repeat = 0; repeat < 3; ++repeat) {
        ASSERT_EQ(ftrainSinkhornFwd(tensor(input), tensor(output), tensor(state), count, epsilon,
                                    FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0, stream),
                  FTRAIN_STATUS_SUCCESS);
    }
    ASSERT_EQ(hipMemcpyAsync(actual.data(), output, sizeof(actual), hipMemcpyDeviceToHost, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    FTrainPattern pattern{};
    FTrainTensorId ports[5]{};
    FTrainSinkhornFwdOpId op{};
    ASSERT_EQ(ftrainPatternCreate(&pattern), FTRAIN_STATUS_SUCCESS);
    for (auto& port : ports) { ASSERT_EQ(ftrainPatternAddTensor(pattern, &port), FTRAIN_STATUS_SUCCESS); }
    ASSERT_EQ(ftrainPatternAddSinkhornFwd(pattern, &op, ports[0], ports[1], ports[2], ports[3], ports[4]),
              FTRAIN_STATUS_SUCCESS);
    FTrainOps ops{};
    FTrainArgs args{};
    FTrainPlan plan{};
    ASSERT_EQ(ftrainOpsCreate(&ops, pattern), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsCreate(&args, ops), FTRAIN_STATUS_SUCCESS);
    const FTrainStorageView views[] = {tensor(input), tensor(output), tensor(state), count, epsilon};
    for (int i = 0; i < 5; ++i) { ASSERT_EQ(ftrainArgsSetTensor(args, ports[i], views[i]), FTRAIN_STATUS_SUCCESS); }
    ASSERT_EQ(ftrainArgsSetSinkhornFwd(args, op, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainOpsDestroy(ops), FTRAIN_STATUS_SUCCESS);
    std::uint64_t bytes = 1;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(bytes, 0U);
    for (int repeat = 0; repeat < 3; ++repeat) {
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream), FTRAIN_STATUS_SUCCESS);
    }
    ASSERT_EQ(hipMemcpyAsync(staged.data(), output, sizeof(staged), hipMemcpyDeviceToHost, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-6F);
        EXPECT_NEAR(staged[i], expected[i], 1e-6F);
    }
    ASSERT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    ASSERT_EQ(hipFree(memory), hipSuccess);
}
TEST_F(SinkhornFwdStagedTest, Bf16SinkhornFwdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t logits_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> logits(8, 0x0000);
    auto* device_logits = allocate<std::uint16_t>(8);
    ASSERT_NE(device_logits, nullptr);
    const std::int64_t output_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> output(8, 0x42f6);
    auto* device_output = allocate<std::uint16_t>(8);
    ASSERT_NE(device_output, nullptr);
    const std::int64_t m_init_dims[]{2, 2, 2};
    const std::vector<float> m_init(8, 123.0F);
    auto* device_m_init = allocate<float>(8);
    ASSERT_NE(device_m_init, nullptr);
    std::int64_t num_iterations = 2;
    float eps                   = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_logits, logits.data(), logits.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_output, output.data(), output.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_m_init, m_init.data(), m_init.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto logits_view =
        makeView(device_logits, logits_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto output_view =
        makeView(device_output, output_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto m_init_view =
        makeView(device_m_init, m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto num_iterations_view =
        makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view   = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, logits_view, output_view, m_init_view, num_iterations_view, eps_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Bf16SinkhornFwd");
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
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(std::uint16_t),
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
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_m_init, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
            }
        }
        ASSERT_EQ(ftrainSinkhornFwd(logits_view, output_view, m_init_view, num_iterations_view, eps_view,
                                    FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(std::uint16_t),
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
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_m_init, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
            }
        }
    }
    auto invalid_logits           = logits_view;
    invalid_logits.is_host_memory = true;
    EXPECT_EQ(ftrainSinkhornFwd(invalid_logits, output_view, m_init_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_logits              = logits_view;
    invalid_logits.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSinkhornFwd(invalid_logits, output_view, m_init_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SinkhornFwdStagedTest, Fp16SinkhornFwdStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t logits_dims[]{2, 2, 2};
    const std::vector<__half> logits(8, __float2half(0.0F));
    auto* device_logits = allocate<__half>(8);
    ASSERT_NE(device_logits, nullptr);
    const std::int64_t output_dims[]{2, 2, 2};
    const std::vector<__half> output(8, __float2half(123.0F));
    auto* device_output = allocate<__half>(8);
    ASSERT_NE(device_output, nullptr);
    const std::int64_t m_init_dims[]{2, 2, 2};
    const std::vector<float> m_init(8, 123.0F);
    auto* device_m_init = allocate<float>(8);
    ASSERT_NE(device_m_init, nullptr);
    std::int64_t num_iterations = 2;
    float eps                   = 1e-06F;
    ASSERT_EQ(
        hipMemcpyAsync(device_logits, logits.data(), logits.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_output, output.data(), output.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_m_init, m_init.data(), m_init.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto logits_view =
        makeView(device_logits, logits_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto output_view =
        makeView(device_output, output_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto m_init_view =
        makeView(device_m_init, m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto num_iterations_view =
        makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view   = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, logits_view, output_view, m_init_view, num_iterations_view, eps_view, FTRAIN_NUMERIC_TYPE_FP16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp16SinkhornFwd");
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
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 0.5F, 2e-4 + 2e-3 * std::abs(0.5F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_m_init, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
            }
        }
        ASSERT_EQ(ftrainSinkhornFwd(logits_view, output_view, m_init_view, num_iterations_view, eps_view,
                                    FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(__half),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 0.5F, 2e-4 + 2e-3 * std::abs(0.5F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_m_init, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
            }
        }
    }
    auto invalid_logits           = logits_view;
    invalid_logits.is_host_memory = true;
    EXPECT_EQ(ftrainSinkhornFwd(invalid_logits, output_view, m_init_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_logits              = logits_view;
    invalid_logits.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSinkhornFwd(invalid_logits, output_view, m_init_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SinkhornFwdStagedTest, Fp32SinkhornFwdStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t logits_dims[]{2, 2, 2};
    const std::vector<float> logits(8, 0.0F);
    auto* device_logits = allocate<float>(8);
    ASSERT_NE(device_logits, nullptr);
    const std::int64_t output_dims[]{2, 2, 2};
    const std::vector<float> output(8, 123.0F);
    auto* device_output = allocate<float>(8);
    ASSERT_NE(device_output, nullptr);
    const std::int64_t m_init_dims[]{2, 2, 2};
    const std::vector<float> m_init(8, 123.0F);
    auto* device_m_init = allocate<float>(8);
    ASSERT_NE(device_m_init, nullptr);
    std::int64_t num_iterations = 2;
    float eps                   = 1e-06F;
    ASSERT_EQ(
        hipMemcpyAsync(device_logits, logits.data(), logits.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_output, output.data(), output.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_m_init, m_init.data(), m_init.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto logits_view =
        makeView(device_logits, logits_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto output_view =
        makeView(device_output, output_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto m_init_view =
        makeView(device_m_init, m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto num_iterations_view =
        makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto eps_view   = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, logits_view, output_view, m_init_view, num_iterations_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp32SinkhornFwd");
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
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_m_init, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
            }
        }
        ASSERT_EQ(ftrainSinkhornFwd(logits_view, output_view, m_init_view, num_iterations_view, eps_view,
                                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_output, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_m_init, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
            }
        }
    }
    auto invalid_logits           = logits_view;
    invalid_logits.is_host_memory = true;
    EXPECT_EQ(ftrainSinkhornFwd(invalid_logits, output_view, m_init_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_logits              = logits_view;
    invalid_logits.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSinkhornFwd(invalid_logits, output_view, m_init_view, num_iterations_view, eps_view,
                                FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SinkhornFwdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_logits_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> first_logits(8, 0x0000);
    auto* first_device_logits = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_logits, nullptr);
    const std::int64_t first_output_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> first_output(8, 0x42f6);
    auto* first_device_output = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_output, nullptr);
    const std::int64_t first_m_init_dims[]{2, 2, 2};
    const std::vector<float> first_m_init(8, 123.0F);
    auto* first_device_m_init = allocate<float>(8);
    ASSERT_NE(first_device_m_init, nullptr);
    std::int64_t first_num_iterations = 2;
    float first_eps                   = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(first_device_logits, first_logits.data(), first_logits.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_output, first_output.data(), first_output.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_m_init, first_m_init.data(), first_m_init.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_logits_view = makeView(first_device_logits, first_logits_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_output_view = makeView(first_device_output, first_output_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_m_init_view = makeView(first_device_m_init, first_m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_num_iterations_view =
        makeView(&first_num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_eps_view =
        makeView(&first_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_logits_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> second_logits(8, 0x0000);
    auto* second_device_logits = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_logits, nullptr);
    const std::int64_t second_output_dims[]{2, 2, 2};
    const std::vector<std::uint16_t> second_output(8, 0x42f6);
    auto* second_device_output = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_output, nullptr);
    const std::int64_t second_m_init_dims[]{2, 2, 2};
    const std::vector<float> second_m_init(8, 123.0F);
    auto* second_device_m_init = allocate<float>(8);
    ASSERT_NE(second_device_m_init, nullptr);
    std::int64_t second_num_iterations = 2;
    float second_eps                   = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(second_device_logits, second_logits.data(), second_logits.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_output, second_output.data(), second_output.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_m_init, second_m_init.data(), second_m_init.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_logits_view = makeView(second_device_logits, second_logits_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_output_view = makeView(second_device_output, second_output_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_m_init_view = makeView(second_device_m_init, second_m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_num_iterations_view =
        makeView(&second_num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_eps_view =
        makeView(&second_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_logits_view, first_output_view, first_m_init_view, first_num_iterations_view,
                        first_eps_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_logits_view, second_output_view, second_m_init_view, second_num_iterations_view,
                        second_eps_view, FTRAIN_NUMERIC_TYPE_BF16));
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
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_output, actual.size() * sizeof(std::uint16_t),
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
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_m_init, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_output, actual.size() * sizeof(std::uint16_t),
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
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_m_init, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_output, actual.size() * sizeof(std::uint16_t),
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
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_m_init, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 0.5F, 2e-5 + 2e-4 * std::abs(0.5F)) << i;
        }
    }
}

}  // namespace
