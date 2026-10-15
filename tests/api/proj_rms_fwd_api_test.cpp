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

#include "proj_rms_fwd_staged_fixture.hpp"

namespace {

using ftrain_test::ProjRmsFwdStagedTest;

constexpr float kEps = 1e-6F;

TEST_F(ProjRmsFwdStagedTest, CachedPlansKeepUniformAndMixedPrecisionSeparate) {
    FTrainPattern pattern{};
    ASSERT_EQ(ftrainPatternCreate(&pattern), FTRAIN_STATUS_SUCCESS);
    FTrainTensorId x{}, w{}, proj{}, r{}, eps{};
    FTrainProjRmsFwdOpId op{};
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &x), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &w), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &proj), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &r), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &eps), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddProjRmsFwd(pattern, &op, x, w, proj, r, eps), FTRAIN_STATUS_SUCCESS);
    FTrainOps ops{};
    ASSERT_EQ(ftrainOpsCreate(&ops, pattern), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS);
    alignas(16) std::array<char, 16> address{};
    const std::int64_t pd[] = {64, 8}, rd[] = {64, 1};
    const FTrainNumericType input_types[] = {FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP16,
                                             FTRAIN_NUMERIC_TYPE_FP32};
    for (int repeat = 0; repeat < 6; ++repeat) {
        const auto input_type        = input_types[repeat % 3];
        const std::int64_t reduction = repeat % 2 == 0 ? 32 : 4096;
        const std::int64_t xd[] = {64, reduction}, wd[] = {8, reduction};
        for (const auto weight_type : {FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP32}) {
            FTrainArgs args{};
            ASSERT_EQ(ftrainArgsCreate(&args, ops), FTRAIN_STATUS_SUCCESS);
            const auto bind = [&](FTrainTensorId id, const std::int64_t* dims, FTrainNumericType dtype) {
                return ftrainArgsSetTensor(
                    args, id,
                    FTrainStorageView{address.data(), dims, nullptr, 2, dtype, FTRAIN_INDEX_TYPE_CONTINUOUS, false});
            };
            ASSERT_EQ(bind(x, xd, input_type), FTRAIN_STATUS_SUCCESS);
            ASSERT_EQ(bind(w, wd, weight_type), FTRAIN_STATUS_SUCCESS);
            ASSERT_EQ(bind(proj, pd, input_type), FTRAIN_STATUS_SUCCESS);
            ASSERT_EQ(bind(r, rd, input_type), FTRAIN_STATUS_SUCCESS);
            ASSERT_EQ(
                ftrainArgsSetTensor(args, eps,
                                    FTrainStorageView{const_cast<float*>(&kEps), nullptr, nullptr, 0,
                                                      FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true}),
                FTRAIN_STATUS_SUCCESS);
            const FTrainNumericType compute_type = input_type == weight_type ? input_type : FTRAIN_NUMERIC_TYPE_FP32;
            ASSERT_EQ(ftrainArgsSetProjRmsFwd(args, op, compute_type), FTRAIN_STATUS_SUCCESS);
            FTrainPlan plan{};
            ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
            ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
            std::uint64_t count{};
            ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
            EXPECT_EQ(count, 1U);
            const char* name = nullptr;
            ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &name), FTRAIN_STATUS_SUCCESS);
            EXPECT_STREQ(name, input_type != weight_type                ? "MixedFp32ProjRmsFwd"
                               : input_type == FTRAIN_NUMERIC_TYPE_BF16 ? "Bf16ProjRmsFwd"
                                                                        : "Fp32ProjRmsFwd");
            std::uint64_t workspace_bytes{};
            ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &workspace_bytes), FTRAIN_STATUS_SUCCESS);
            EXPECT_EQ(workspace_bytes, reduction == 32 ? 0U : 8U * 64U * 9U * sizeof(float) + sizeof(int));
            ASSERT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
        }
    }
    ASSERT_EQ(ftrainOpsDestroy(ops), FTRAIN_STATUS_SUCCESS);
}
TEST_F(ProjRmsFwdStagedTest, MixedFp32ProjRmsFwdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t x_dims[]{16, 32};
    const std::vector<std::uint16_t> x(512, 0x3f00);
    auto* device_x = allocate<std::uint16_t>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<__half> weight(256, __float2half(0.25F));
    auto* device_weight = allocate<__half>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<std::uint16_t> proj(128, 0x42f6);
    auto* device_proj = allocate<std::uint16_t>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<std::uint16_t> r(16, 0x42f6);
    auto* device_r = allocate<std::uint16_t>(16);
    ASSERT_NE(device_r, nullptr);
    float eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto x_view = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto eps_view  = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsFwd");
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
            std::vector<std::uint16_t> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual_value, 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 4.0F, 2e-3 + 1e-2 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<std::uint16_t> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual_value, 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 1.9999960000120003F, 2e-3 + 1e-2 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
        ASSERT_EQ(ftrainProjRmsFwd(x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32,
                                   workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual_value, 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 4.0F, 2e-3 + 1e-2 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<std::uint16_t> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual_value, 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 1.9999960000120003F, 2e-3 + 1e-2 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
    }
    auto invalid_x           = x_view;
    invalid_x.is_host_memory = true;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_x              = x_view;
    invalid_x.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsFwdStagedTest, MixedFp32ProjRmsFwdStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t x_dims[]{16, 32};
    const std::vector<std::uint16_t> x(512, 0x3f00);
    auto* device_x = allocate<std::uint16_t>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<float> weight(256, 0.25F);
    auto* device_weight = allocate<float>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<std::uint16_t> proj(128, 0x42f6);
    auto* device_proj = allocate<std::uint16_t>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<std::uint16_t> r(16, 0x42f6);
    auto* device_r = allocate<std::uint16_t>(16);
    ASSERT_NE(device_r, nullptr);
    float eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto x_view = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto eps_view  = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsFwd");
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
            std::vector<std::uint16_t> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual_value, 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 4.0F, 2e-3 + 1e-2 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<std::uint16_t> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual_value, 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 1.9999960000120003F, 2e-3 + 1e-2 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
        ASSERT_EQ(ftrainProjRmsFwd(x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32,
                                   workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual_value, 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 4.0F, 2e-3 + 1e-2 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<std::uint16_t> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual_value, 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 1.9999960000120003F, 2e-3 + 1e-2 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
    }
    auto invalid_x           = x_view;
    invalid_x.is_host_memory = true;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_x              = x_view;
    invalid_x.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsFwdStagedTest, Bf16ProjRmsFwdStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t x_dims[]{16, 32};
    const std::vector<std::uint16_t> x(512, 0x3f00);
    auto* device_x = allocate<std::uint16_t>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<std::uint16_t> weight(256, 0x3e80);
    auto* device_weight = allocate<std::uint16_t>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<std::uint16_t> proj(128, 0x42f6);
    auto* device_proj = allocate<std::uint16_t>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<std::uint16_t> r(16, 0x42f6);
    auto* device_r = allocate<std::uint16_t>(16);
    ASSERT_NE(device_r, nullptr);
    float eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto x_view = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto eps_view  = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Bf16ProjRmsFwd");
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
            std::vector<std::uint16_t> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual_value, 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 4.0F, 2e-3 + 1e-2 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<std::uint16_t> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual_value, 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 1.9999960000120003F, 2e-3 + 1e-2 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
        ASSERT_EQ(ftrainProjRmsFwd(x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_BF16,
                                   workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual_value, 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 4.0F, 2e-3 + 1e-2 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<std::uint16_t> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual_value, 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual_value, 1.9999960000120003F, 2e-3 + 1e-2 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
    }
    auto invalid_x           = x_view;
    invalid_x.is_host_memory = true;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_BF16, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_x              = x_view;
    invalid_x.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_BF16, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsFwdStagedTest, MixedFp32ProjRmsFwdStorageMode3RunsConvenienceAndStagedApis) {
    const std::int64_t x_dims[]{16, 32};
    const std::vector<__half> x(512, __float2half(0.5F));
    auto* device_x = allocate<__half>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<std::uint16_t> weight(256, 0x3e80);
    auto* device_weight = allocate<std::uint16_t>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<__half> proj(128, __float2half(123.0F));
    auto* device_proj = allocate<__half>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<__half> r(16, __float2half(123.0F));
    auto* device_r = allocate<__half>(16);
    ASSERT_NE(device_r, nullptr);
    float eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto x_view = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto eps_view  = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsFwd");
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
            std::vector<__half> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(__half2float(actual[i]), 4.0F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 4.0F, 2e-4 + 2e-3 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<__half> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(__half), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(__half2float(actual[i]), 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 1.9999960000120003F,
                                2e-4 + 2e-3 * std::abs(1.9999960000120003F))
                        << i;
                }
            }
        }
        ASSERT_EQ(ftrainProjRmsFwd(x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32,
                                   workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(__half2float(actual[i]), 4.0F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 4.0F, 2e-4 + 2e-3 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<__half> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(__half), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(__half2float(actual[i]), 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 1.9999960000120003F,
                                2e-4 + 2e-3 * std::abs(1.9999960000120003F))
                        << i;
                }
            }
        }
    }
    auto invalid_x           = x_view;
    invalid_x.is_host_memory = true;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_x              = x_view;
    invalid_x.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsFwdStagedTest, MixedFp32ProjRmsFwdStorageMode4RunsConvenienceAndStagedApis) {
    const std::int64_t x_dims[]{16, 32};
    const std::vector<__half> x(512, __float2half(0.5F));
    auto* device_x = allocate<__half>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<float> weight(256, 0.25F);
    auto* device_weight = allocate<float>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<__half> proj(128, __float2half(123.0F));
    auto* device_proj = allocate<__half>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<__half> r(16, __float2half(123.0F));
    auto* device_r = allocate<__half>(16);
    ASSERT_NE(device_r, nullptr);
    float eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto x_view = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto eps_view  = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsFwd");
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
            std::vector<__half> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(__half2float(actual[i]), 4.0F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 4.0F, 2e-4 + 2e-3 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<__half> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(__half), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(__half2float(actual[i]), 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 1.9999960000120003F,
                                2e-4 + 2e-3 * std::abs(1.9999960000120003F))
                        << i;
                }
            }
        }
        ASSERT_EQ(ftrainProjRmsFwd(x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32,
                                   workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(__half2float(actual[i]), 4.0F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 4.0F, 2e-4 + 2e-3 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<__half> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(__half), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(__half2float(actual[i]), 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 1.9999960000120003F,
                                2e-4 + 2e-3 * std::abs(1.9999960000120003F))
                        << i;
                }
            }
        }
    }
    auto invalid_x           = x_view;
    invalid_x.is_host_memory = true;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_x              = x_view;
    invalid_x.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsFwdStagedTest, Fp16ProjRmsFwdStorageMode5RunsConvenienceAndStagedApis) {
    const std::int64_t x_dims[]{16, 32};
    const std::vector<__half> x(512, __float2half(0.5F));
    auto* device_x = allocate<__half>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<__half> weight(256, __float2half(0.25F));
    auto* device_weight = allocate<__half>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<__half> proj(128, __float2half(123.0F));
    auto* device_proj = allocate<__half>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<__half> r(16, __float2half(123.0F));
    auto* device_r = allocate<__half>(16);
    ASSERT_NE(device_r, nullptr);
    float eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto x_view = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto eps_view  = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp16ProjRmsFwd");
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
            std::vector<__half> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(__half2float(actual[i]), 4.0F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 4.0F, 2e-4 + 2e-3 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<__half> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(__half), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(__half2float(actual[i]), 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 1.9999960000120003F,
                                2e-4 + 2e-3 * std::abs(1.9999960000120003F))
                        << i;
                }
            }
        }
        ASSERT_EQ(ftrainProjRmsFwd(x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP16,
                                   workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(__half2float(actual[i]), 4.0F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 4.0F, 2e-4 + 2e-3 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<__half> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(__half), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(__half2float(actual[i]), 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), 1.9999960000120003F,
                                2e-4 + 2e-3 * std::abs(1.9999960000120003F))
                        << i;
                }
            }
        }
    }
    auto invalid_x           = x_view;
    invalid_x.is_host_memory = true;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP16, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_x              = x_view;
    invalid_x.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP16, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsFwdStagedTest, MixedFp32ProjRmsFwdStorageMode6RunsConvenienceAndStagedApis) {
    const std::int64_t x_dims[]{16, 32};
    const std::vector<float> x(512, 0.5F);
    auto* device_x = allocate<float>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<std::uint16_t> weight(256, 0x3e80);
    auto* device_weight = allocate<std::uint16_t>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<float> proj(128, 123.0F);
    auto* device_proj = allocate<float>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<float> r(16, 123.0F);
    auto* device_r = allocate<float>(16);
    ASSERT_NE(device_r, nullptr);
    float eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto x_view = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto eps_view  = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsFwd");
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
            std::vector<float> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual[i], 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 4.0F, 2e-5 + 2e-4 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual[i], 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 1.9999960000120003F, 2e-5 + 2e-4 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
        ASSERT_EQ(ftrainProjRmsFwd(x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32,
                                   workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual[i], 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 4.0F, 2e-5 + 2e-4 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual[i], 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 1.9999960000120003F, 2e-5 + 2e-4 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
    }
    auto invalid_x           = x_view;
    invalid_x.is_host_memory = true;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_x              = x_view;
    invalid_x.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsFwdStagedTest, MixedFp32ProjRmsFwdStorageMode7RunsConvenienceAndStagedApis) {
    const std::int64_t x_dims[]{16, 32};
    const std::vector<float> x(512, 0.5F);
    auto* device_x = allocate<float>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<__half> weight(256, __float2half(0.25F));
    auto* device_weight = allocate<__half>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<float> proj(128, 123.0F);
    auto* device_proj = allocate<float>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<float> r(16, 123.0F);
    auto* device_r = allocate<float>(16);
    ASSERT_NE(device_r, nullptr);
    float eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto x_view = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto eps_view  = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32ProjRmsFwd");
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
            std::vector<float> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual[i], 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 4.0F, 2e-5 + 2e-4 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual[i], 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 1.9999960000120003F, 2e-5 + 2e-4 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
        ASSERT_EQ(ftrainProjRmsFwd(x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32,
                                   workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual[i], 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 4.0F, 2e-5 + 2e-4 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual[i], 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 1.9999960000120003F, 2e-5 + 2e-4 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
    }
    auto invalid_x           = x_view;
    invalid_x.is_host_memory = true;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_x              = x_view;
    invalid_x.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsFwdStagedTest, Fp32ProjRmsFwdStorageMode8RunsConvenienceAndStagedApis) {
    const std::int64_t x_dims[]{16, 32};
    const std::vector<float> x(512, 0.5F);
    auto* device_x = allocate<float>(512);
    ASSERT_NE(device_x, nullptr);
    const std::int64_t weight_dims[]{8, 32};
    const std::vector<float> weight(256, 0.25F);
    auto* device_weight = allocate<float>(256);
    ASSERT_NE(device_weight, nullptr);
    const std::int64_t proj_dims[]{16, 8};
    const std::vector<float> proj(128, 123.0F);
    auto* device_proj = allocate<float>(128);
    ASSERT_NE(device_proj, nullptr);
    const std::int64_t r_dims[]{16, 1};
    const std::vector<float> r(16, 123.0F);
    auto* device_r = allocate<float>(16);
    ASSERT_NE(device_r, nullptr);
    float eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(device_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weight, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_proj, proj.data(), proj.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_r, r.data(), r.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto x_view = makeView(device_x, x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weight_view =
        makeView(device_weight, weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto proj_view = makeView(device_proj, proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto r_view    = makeView(device_r, r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto eps_view  = makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp32ProjRmsFwd");
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
            std::vector<float> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual[i], 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 4.0F, 2e-5 + 2e-4 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual[i], 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 1.9999960000120003F, 2e-5 + 2e-4 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
        ASSERT_EQ(ftrainProjRmsFwd(x_view, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32,
                                   workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_proj, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(4.0F)) {
                    EXPECT_EQ(actual[i], 4.0F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 4.0F, 2e-5 + 2e-4 * std::abs(4.0F)) << i;
                }
            }
        }
        {
            std::vector<float> actual(16);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_r, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isinf(1.9999960000120003F)) {
                    EXPECT_EQ(actual[i], 1.9999960000120003F) << i;
                } else {
                    EXPECT_NEAR(actual[i], 1.9999960000120003F, 2e-5 + 2e-4 * std::abs(1.9999960000120003F)) << i;
                }
            }
        }
    }
    auto invalid_x           = x_view;
    invalid_x.is_host_memory = true;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_x              = x_view;
    invalid_x.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainProjRmsFwd(invalid_x, weight_view, proj_view, r_view, eps_view, FTRAIN_NUMERIC_TYPE_FP32, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(ProjRmsFwdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_x_dims[]{16, 32};
    const std::vector<std::uint16_t> first_x(512, 0x3f00);
    auto* first_device_x = allocate<std::uint16_t>(512);
    ASSERT_NE(first_device_x, nullptr);
    const std::int64_t first_weight_dims[]{8, 32};
    const std::vector<__half> first_weight(256, __float2half(0.25F));
    auto* first_device_weight = allocate<__half>(256);
    ASSERT_NE(first_device_weight, nullptr);
    const std::int64_t first_proj_dims[]{16, 8};
    const std::vector<std::uint16_t> first_proj(128, 0x42f6);
    auto* first_device_proj = allocate<std::uint16_t>(128);
    ASSERT_NE(first_device_proj, nullptr);
    const std::int64_t first_r_dims[]{16, 1};
    const std::vector<std::uint16_t> first_r(16, 0x42f6);
    auto* first_device_r = allocate<std::uint16_t>(16);
    ASSERT_NE(first_device_r, nullptr);
    float first_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(first_device_x, first_x.data(), first_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_weight, first_weight.data(), first_weight.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_proj, first_proj.data(), first_proj.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_r, first_r.data(), first_r.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_x_view =
        makeView(first_device_x, first_x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_weight_view = makeView(first_device_weight, first_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_proj_view =
        makeView(first_device_proj, first_proj_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_r_view =
        makeView(first_device_r, first_r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_eps_view =
        makeView(&first_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_x_dims[]{16, 32};
    const std::vector<std::uint16_t> second_x(512, 0x3f00);
    auto* second_device_x = allocate<std::uint16_t>(512);
    ASSERT_NE(second_device_x, nullptr);
    const std::int64_t second_weight_dims[]{8, 32};
    const std::vector<__half> second_weight(256, __float2half(0.25F));
    auto* second_device_weight = allocate<__half>(256);
    ASSERT_NE(second_device_weight, nullptr);
    const std::int64_t second_proj_dims[]{16, 8};
    const std::vector<std::uint16_t> second_proj(128, 0x42f6);
    auto* second_device_proj = allocate<std::uint16_t>(128);
    ASSERT_NE(second_device_proj, nullptr);
    const std::int64_t second_r_dims[]{16, 1};
    const std::vector<std::uint16_t> second_r(16, 0x42f6);
    auto* second_device_r = allocate<std::uint16_t>(16);
    ASSERT_NE(second_device_r, nullptr);
    float second_eps = 1e-06F;
    ASSERT_EQ(hipMemcpyAsync(second_device_x, second_x.data(), second_x.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_weight, second_weight.data(), second_weight.size() * sizeof(__half),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_proj, second_proj.data(), second_proj.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_r, second_r.data(), second_r.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_x_view =
        makeView(second_device_x, second_x_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_weight_view = makeView(second_device_weight, second_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP16,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_proj_view   = makeView(second_device_proj, second_proj_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_r_view =
        makeView(second_device_r, second_r_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_eps_view =
        makeView(&second_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_x_view, first_weight_view, first_proj_view, first_r_view, first_eps_view,
                        FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_x_view, second_weight_view, second_proj_view, second_r_view, second_eps_view,
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
        std::vector<std::uint16_t> actual(128);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_proj, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            if (std::isinf(4.0F)) {
                EXPECT_EQ(actual_value, 4.0F) << i;
            } else {
                EXPECT_NEAR(actual_value, 4.0F, 2e-3 + 1e-2 * std::abs(4.0F)) << i;
            }
        }
    }
    {
        std::vector<std::uint16_t> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_r, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            if (std::isinf(1.9999960000120003F)) {
                EXPECT_EQ(actual_value, 1.9999960000120003F) << i;
            } else {
                EXPECT_NEAR(actual_value, 1.9999960000120003F, 2e-3 + 1e-2 * std::abs(1.9999960000120003F)) << i;
            }
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(128);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_proj, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            if (std::isinf(4.0F)) {
                EXPECT_EQ(actual_value, 4.0F) << i;
            } else {
                EXPECT_NEAR(actual_value, 4.0F, 2e-3 + 1e-2 * std::abs(4.0F)) << i;
            }
        }
    }
    {
        std::vector<std::uint16_t> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_r, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            if (std::isinf(1.9999960000120003F)) {
                EXPECT_EQ(actual_value, 1.9999960000120003F) << i;
            } else {
                EXPECT_NEAR(actual_value, 1.9999960000120003F, 2e-3 + 1e-2 * std::abs(1.9999960000120003F)) << i;
            }
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(128);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_proj, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            if (std::isinf(4.0F)) {
                EXPECT_EQ(actual_value, 4.0F) << i;
            } else {
                EXPECT_NEAR(actual_value, 4.0F, 2e-3 + 1e-2 * std::abs(4.0F)) << i;
            }
        }
    }
    {
        std::vector<std::uint16_t> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_r, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            if (std::isinf(1.9999960000120003F)) {
                EXPECT_EQ(actual_value, 1.9999960000120003F) << i;
            } else {
                EXPECT_NEAR(actual_value, 1.9999960000120003F, 2e-3 + 1e-2 * std::abs(1.9999960000120003F)) << i;
            }
        }
    }
}

}  // namespace
