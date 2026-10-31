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

#include "indexer_topk_thd_staged_fixture.hpp"

namespace {

using ftrain_test::IndexerTopKThdStagedTest;

TEST_F(IndexerTopKThdStagedTest, IndexerTopKThdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t input_values_dims[]{4, 4};
    const std::vector<float> input_values{0.0F, 1.0F, 2.0F, 3.0F, 0.0F, 1.0F, 2.0F, 3.0F,
                                          0.0F, 1.0F, 2.0F, 3.0F, 0.0F, 1.0F, 2.0F, 3.0F};
    auto* device_input_values = allocate<float>(16);
    ASSERT_NE(device_input_values, nullptr);
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
    const std::int64_t out_indices_dims[]{4, 2};
    const std::vector<std::int32_t> out_indices(8, 123);
    auto* device_out_indices = allocate<std::int32_t>(8);
    ASSERT_NE(device_out_indices, nullptr);
    const std::int64_t out_values_dims[]{4, 2};
    const std::vector<float> out_values(8, 123.0F);
    auto* device_out_values = allocate<float>(8);
    ASSERT_NE(device_out_values, nullptr);
    std::int64_t ratio        = 1;
    std::int64_t top_k        = 2;
    std::uint8_t order_values = 1;
    ASSERT_EQ(hipMemcpyAsync(device_input_values, input_values.data(), input_values.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
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
    ASSERT_EQ(hipMemcpyAsync(device_out_indices, out_indices.data(), out_indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_out_values, out_values.data(), out_values.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto input_values_view     = makeView(device_input_values, input_values_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_q_view     = makeView(device_cu_seqlens_q, cu_seqlens_q_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_kv_view    = makeView(device_cu_seqlens_kv, cu_seqlens_kv_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto q_causal_offsets_view = makeView(device_q_causal_offsets, q_causal_offsets_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_indices_view      = makeView(device_out_indices, out_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_values_view =
        makeView(device_out_values, out_values_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto top_k_view = makeView(&top_k, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto order_values_view =
        makeView(&order_values, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, input_values_view, cu_seqlens_q_view, cu_seqlens_kv_view, q_causal_offsets_view,
                        out_indices_view, out_values_view, ratio_view, top_k_view, order_values_view));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "IndexerTopKThd");
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
        ASSERT_EQ(hipMemcpyAsync(device_out_indices, out_indices.data(), out_indices.size() * sizeof(std::int32_t),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_out_values, out_values.data(), out_values.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::int32_t> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out_indices, actual.size() * sizeof(std::int32_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(8, std::numeric_limits<float>::quiet_NaN());
            expected[6] = 3.0F;
            expected[7] = 2.0F;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                EXPECT_EQ(actual[i], expected[i]) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out_values, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(8, std::numeric_limits<float>::quiet_NaN());
            expected[6] = 3.0F;
            expected[7] = 2.0F;
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
        ASSERT_EQ(hipMemcpyAsync(device_out_indices, out_indices.data(), out_indices.size() * sizeof(std::int32_t),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_out_values, out_values.data(), out_values.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainIndexerTopKThd(input_values_view, cu_seqlens_q_view, cu_seqlens_kv_view, q_causal_offsets_view,
                                       out_indices_view, out_values_view, ratio_view, top_k_view, order_values_view,
                                       workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::int32_t> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out_indices, actual.size() * sizeof(std::int32_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(8, std::numeric_limits<float>::quiet_NaN());
            expected[6] = 3.0F;
            expected[7] = 2.0F;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                EXPECT_EQ(actual[i], expected[i]) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out_values, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(8, std::numeric_limits<float>::quiet_NaN());
            expected[6] = 3.0F;
            expected[7] = 2.0F;
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
    auto invalid_input_values           = input_values_view;
    invalid_input_values.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerTopKThd(invalid_input_values, cu_seqlens_q_view, cu_seqlens_kv_view, q_causal_offsets_view,
                                   out_indices_view, out_values_view, ratio_view, top_k_view, order_values_view,
                                   workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_input_values              = input_values_view;
    invalid_input_values.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerTopKThd(invalid_input_values, cu_seqlens_q_view, cu_seqlens_kv_view, q_causal_offsets_view,
                                   out_indices_view, out_values_view, ratio_view, top_k_view, order_values_view,
                                   workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerTopKThdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_input_values_dims[]{4, 4};
    const std::vector<float> first_input_values{0.0F, 1.0F, 2.0F, 3.0F, 0.0F, 1.0F, 2.0F, 3.0F,
                                                0.0F, 1.0F, 2.0F, 3.0F, 0.0F, 1.0F, 2.0F, 3.0F};
    auto* first_device_input_values = allocate<float>(16);
    ASSERT_NE(first_device_input_values, nullptr);
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
    const std::int64_t first_out_indices_dims[]{4, 2};
    const std::vector<std::int32_t> first_out_indices(8, 123);
    auto* first_device_out_indices = allocate<std::int32_t>(8);
    ASSERT_NE(first_device_out_indices, nullptr);
    const std::int64_t first_out_values_dims[]{4, 2};
    const std::vector<float> first_out_values(8, 123.0F);
    auto* first_device_out_values = allocate<float>(8);
    ASSERT_NE(first_device_out_values, nullptr);
    std::int64_t first_ratio        = 1;
    std::int64_t first_top_k        = 2;
    std::uint8_t first_order_values = 1;
    ASSERT_EQ(hipMemcpyAsync(first_device_input_values, first_input_values.data(),
                             first_input_values.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
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
    ASSERT_EQ(hipMemcpyAsync(first_device_out_indices, first_out_indices.data(),
                             first_out_indices.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_out_values, first_out_values.data(), first_out_values.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_input_values_view     = makeView(first_device_input_values, first_input_values_dims, 2,
                                                FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_cu_seqlens_q_view     = makeView(first_device_cu_seqlens_q, first_cu_seqlens_q_dims, 1,
                                                FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_cu_seqlens_kv_view    = makeView(first_device_cu_seqlens_kv, first_cu_seqlens_kv_dims, 1,
                                                FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_q_causal_offsets_view = makeView(first_device_q_causal_offsets, first_q_causal_offsets_dims, 1,
                                                FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_out_indices_view      = makeView(first_device_out_indices, first_out_indices_dims, 2,
                                                FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_out_values_view = makeView(first_device_out_values, first_out_values_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_ratio_view =
        makeView(&first_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_top_k_view =
        makeView(&first_top_k, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_order_values_view =
        makeView(&first_order_values, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_input_values_dims[]{4, 4};
    const std::vector<float> second_input_values{0.0F, 1.0F, 2.0F, 3.0F, 0.0F, 1.0F, 2.0F, 3.0F,
                                                 0.0F, 1.0F, 2.0F, 3.0F, 0.0F, 1.0F, 2.0F, 3.0F};
    auto* second_device_input_values = allocate<float>(16);
    ASSERT_NE(second_device_input_values, nullptr);
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
    const std::int64_t second_out_indices_dims[]{4, 2};
    const std::vector<std::int32_t> second_out_indices(8, 123);
    auto* second_device_out_indices = allocate<std::int32_t>(8);
    ASSERT_NE(second_device_out_indices, nullptr);
    const std::int64_t second_out_values_dims[]{4, 2};
    const std::vector<float> second_out_values(8, 123.0F);
    auto* second_device_out_values = allocate<float>(8);
    ASSERT_NE(second_device_out_values, nullptr);
    std::int64_t second_ratio        = 1;
    std::int64_t second_top_k        = 2;
    std::uint8_t second_order_values = 1;
    ASSERT_EQ(hipMemcpyAsync(second_device_input_values, second_input_values.data(),
                             second_input_values.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
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
    ASSERT_EQ(hipMemcpyAsync(second_device_out_indices, second_out_indices.data(),
                             second_out_indices.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_out_values, second_out_values.data(),
                             second_out_values.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_input_values_view     = makeView(second_device_input_values, second_input_values_dims, 2,
                                                 FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_cu_seqlens_q_view     = makeView(second_device_cu_seqlens_q, second_cu_seqlens_q_dims, 1,
                                                 FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_cu_seqlens_kv_view    = makeView(second_device_cu_seqlens_kv, second_cu_seqlens_kv_dims, 1,
                                                 FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_q_causal_offsets_view = makeView(second_device_q_causal_offsets, second_q_causal_offsets_dims, 1,
                                                 FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_out_indices_view      = makeView(second_device_out_indices, second_out_indices_dims, 2,
                                                 FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_out_values_view       = makeView(second_device_out_values, second_out_values_dims, 2,
                                                 FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_ratio_view =
        makeView(&second_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_top_k_view =
        makeView(&second_top_k, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_order_values_view =
        makeView(&second_order_values, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_input_values_view, first_cu_seqlens_q_view, first_cu_seqlens_kv_view,
                        first_q_causal_offsets_view, first_out_indices_view, first_out_values_view, first_ratio_view,
                        first_top_k_view, first_order_values_view));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_input_values_view, second_cu_seqlens_q_view, second_cu_seqlens_kv_view,
                        second_q_causal_offsets_view, second_out_indices_view, second_out_values_view,
                        second_ratio_view, second_top_k_view, second_order_values_view));
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
        std::vector<std::int32_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_out_indices, actual.size() * sizeof(std::int32_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(8, std::numeric_limits<float>::quiet_NaN());
        expected[6] = 3.0F;
        expected[7] = 2.0F;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) continue;
            EXPECT_EQ(actual[i], expected[i]) << i;
        }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_out_values, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(8, std::numeric_limits<float>::quiet_NaN());
        expected[6] = 3.0F;
        expected[7] = 2.0F;
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
        std::vector<std::int32_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_out_indices, actual.size() * sizeof(std::int32_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(8, std::numeric_limits<float>::quiet_NaN());
        expected[6] = 3.0F;
        expected[7] = 2.0F;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) continue;
            EXPECT_EQ(actual[i], expected[i]) << i;
        }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_out_values, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(8, std::numeric_limits<float>::quiet_NaN());
        expected[6] = 3.0F;
        expected[7] = 2.0F;
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
