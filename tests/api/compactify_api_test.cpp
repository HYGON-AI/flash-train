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

#include "compactify_staged_fixture.hpp"

namespace {

using ftrain_test::CompactifyStagedTest;

TEST_F(CompactifyStagedTest, CompactifyStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t indices_dims[]{2, 4};
    const std::vector<std::int32_t> indices{3, -1, 1, -1, 2, 0, -1, 1};
    auto* device_indices = allocate<std::int32_t>(8);
    ASSERT_NE(device_indices, nullptr);
    const std::int64_t output_dims[]{2, 4};
    const std::vector<std::int32_t> output(8, 0);
    auto* device_output = allocate<std::int32_t>(8);
    ASSERT_NE(device_output, nullptr);
    const std::int64_t topk_length_dims[]{2};
    const std::vector<std::int32_t> topk_length(2, 123);
    auto* device_topk_length = allocate<std::int32_t>(2);
    ASSERT_NE(device_topk_length, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_indices, indices.data(), indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_output, output.data(), output.size() * sizeof(std::int32_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_length, topk_length.data(), topk_length.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto indices_view =
        makeView(device_indices, indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto output_view =
        makeView(device_output, output_dims, 2, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_length_view = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args       = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, indices_view, output_view, topk_length_view));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Compactify");
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
        ASSERT_EQ(hipMemcpyAsync(device_topk_length, topk_length.data(), topk_length.size() * sizeof(std::int32_t),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::int32_t> actual(2);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_topk_length, actual.size() * sizeof(std::int32_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(2, 2.0F);
            expected[1] = 3.0F;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                EXPECT_EQ(actual[i], expected[i]) << i;
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_topk_length, topk_length.data(), topk_length.size() * sizeof(std::int32_t),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainCompactify(indices_view, output_view, topk_length_view, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::int32_t> actual(2);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_topk_length, actual.size() * sizeof(std::int32_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(2, 2.0F);
            expected[1] = 3.0F;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                EXPECT_EQ(actual[i], expected[i]) << i;
            }
        }
    }
    auto invalid_indices           = indices_view;
    invalid_indices.is_host_memory = true;
    EXPECT_EQ(ftrainCompactify(invalid_indices, output_view, topk_length_view, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_indices              = indices_view;
    invalid_indices.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainCompactify(invalid_indices, output_view, topk_length_view, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(CompactifyStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_indices_dims[]{2, 4};
    const std::vector<std::int32_t> first_indices{3, -1, 1, -1, 2, 0, -1, 1};
    auto* first_device_indices = allocate<std::int32_t>(8);
    ASSERT_NE(first_device_indices, nullptr);
    const std::int64_t first_output_dims[]{2, 4};
    const std::vector<std::int32_t> first_output(8, 0);
    auto* first_device_output = allocate<std::int32_t>(8);
    ASSERT_NE(first_device_output, nullptr);
    const std::int64_t first_topk_length_dims[]{2};
    const std::vector<std::int32_t> first_topk_length(2, 123);
    auto* first_device_topk_length = allocate<std::int32_t>(2);
    ASSERT_NE(first_device_topk_length, nullptr);
    ASSERT_EQ(hipMemcpyAsync(first_device_indices, first_indices.data(), first_indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_output, first_output.data(), first_output.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_topk_length, first_topk_length.data(),
                             first_topk_length.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_indices_view     = makeView(first_device_indices, first_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_output_view      = makeView(first_device_output, first_output_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_topk_length_view = makeView(first_device_topk_length, first_topk_length_dims, 1,
                                           FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    const std::int64_t second_indices_dims[]{2, 4};
    const std::vector<std::int32_t> second_indices{3, -1, 1, -1, 2, 0, -1, 1};
    auto* second_device_indices = allocate<std::int32_t>(8);
    ASSERT_NE(second_device_indices, nullptr);
    const std::int64_t second_output_dims[]{2, 4};
    const std::vector<std::int32_t> second_output(8, 0);
    auto* second_device_output = allocate<std::int32_t>(8);
    ASSERT_NE(second_device_output, nullptr);
    const std::int64_t second_topk_length_dims[]{2};
    const std::vector<std::int32_t> second_topk_length(2, 123);
    auto* second_device_topk_length = allocate<std::int32_t>(2);
    ASSERT_NE(second_device_topk_length, nullptr);
    ASSERT_EQ(hipMemcpyAsync(second_device_indices, second_indices.data(), second_indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_output, second_output.data(), second_output.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_topk_length, second_topk_length.data(),
                             second_topk_length.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_indices_view     = makeView(second_device_indices, second_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_output_view      = makeView(second_device_output, second_output_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_topk_length_view = makeView(second_device_topk_length, second_topk_length_dims, 1,
                                            FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    FTrainArgs args              = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_indices_view, first_output_view, first_topk_length_view));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_indices_view, second_output_view, second_topk_length_view));
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
        std::vector<std::int32_t> actual(2);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_topk_length, actual.size() * sizeof(std::int32_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(2, 2.0F);
        expected[1] = 3.0F;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) continue;
            EXPECT_EQ(actual[i], expected[i]) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::int32_t> actual(2);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_topk_length, actual.size() * sizeof(std::int32_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(2, 2.0F);
        expected[1] = 3.0F;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) continue;
            EXPECT_EQ(actual[i], expected[i]) << i;
        }
    }
}

}  // namespace
