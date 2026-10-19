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

#include "sparse_attn_fwd_indexer_staged_fixture.hpp"

namespace {

using ftrain_test::SparseAttnFwdIndexerStagedTest;

TEST_F(SparseAttnFwdIndexerStagedTest, Bf16SparseAttnFwdIndexerStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> q(4096, 0x0000);
    auto* device_q = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t kv_dims[]{4, 512};
    const std::vector<std::uint16_t> kv(2048, 0x3f00);
    auto* device_kv = allocate<std::uint16_t>(2048);
    ASSERT_NE(device_kv, nullptr);
    const std::int64_t attn_sink_dims[]{2};
    const std::vector<float> attn_sink(2, 0.0F);
    auto* device_attn_sink = allocate<float>(2);
    ASSERT_NE(device_attn_sink, nullptr);
    const std::int64_t topk_indices_dims[]{4, 4};
    const std::vector<std::int32_t> topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(device_topk_indices, nullptr);
    const std::int64_t topk_length_dims[]{4};
    const std::vector<std::int32_t> topk_length(4, 4);
    auto* device_topk_length = allocate<std::int32_t>(4);
    ASSERT_NE(device_topk_length, nullptr);
    const std::int64_t out_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> out(4096, 0x42f6);
    auto* device_out = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t max_logits_dims[]{4, 2};
    const std::vector<float> max_logits(8, 123.0F);
    auto* device_max_logits = allocate<float>(8);
    ASSERT_NE(device_max_logits, nullptr);
    const std::int64_t lse_dims[]{4, 2};
    const std::vector<float> lse(8, 123.0F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
    const std::int64_t lse_indexer_dims[]{4, 2};
    const std::vector<float> lse_indexer(8, 123.0F);
    auto* device_lse_indexer = allocate<float>(8);
    ASSERT_NE(device_lse_indexer, nullptr);
    float softmax_scale       = 0.125F;
    std::int64_t indexer_topk = 2;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_kv, kv.data(), kv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_sink, attn_sink.data(), attn_sink.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_indices, topk_indices.data(), topk_indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_length, topk_length.data(), topk_length.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_max_logits, max_logits.data(), max_logits.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse_indexer, lse_indexer.data(), lse_indexer.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view  = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto kv_view = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_sink_view =
        makeView(device_attn_sink, attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_length_view  = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto max_logits_view =
        makeView(device_max_logits, max_logits_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view = makeView(device_lse, lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_indexer_view = makeView(device_lse_indexer, lse_indexer_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto indexer_topk_view =
        makeView(&indexer_topk, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, kv_view, attn_sink_view, topk_indices_view, topk_length_view, out_view,
                        max_logits_view, lse_view, lse_indexer_view, softmax_scale_view, indexer_topk_view,
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
    EXPECT_STREQ(primitive, "Bf16SparseAttnFwdIndexer");
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
            std::vector<std::uint16_t> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.4F, 2e-3 + 1e-2 * std::abs(0.4F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_max_logits, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.3862943611198906F, 2e-5 + 2e-4 * std::abs(1.3862943611198906F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse_indexer, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.0986122886681098F, 2e-5 + 2e-4 * std::abs(1.0986122886681098F)) << i;
            }
        }
        ASSERT_EQ(ftrainSparseAttnFwdIndexer(q_view, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                             out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                             indexer_topk_view, FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.4F, 2e-3 + 1e-2 * std::abs(0.4F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_max_logits, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.3862943611198906F, 2e-5 + 2e-4 * std::abs(1.3862943611198906F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse_indexer, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.0986122886681098F, 2e-5 + 2e-4 * std::abs(1.0986122886681098F)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseAttnFwdIndexer(invalid_q, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                         out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                         indexer_topk_view, FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseAttnFwdIndexer(invalid_q, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                         out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                         indexer_topk_view, FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseAttnFwdIndexerStagedTest, Fp16SparseAttnFwdIndexerStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 512};
    const std::vector<__half> q(4096, __float2half(0.0F));
    auto* device_q = allocate<__half>(4096);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t kv_dims[]{4, 512};
    const std::vector<__half> kv(2048, __float2half(0.5F));
    auto* device_kv = allocate<__half>(2048);
    ASSERT_NE(device_kv, nullptr);
    const std::int64_t attn_sink_dims[]{2};
    const std::vector<float> attn_sink(2, 0.0F);
    auto* device_attn_sink = allocate<float>(2);
    ASSERT_NE(device_attn_sink, nullptr);
    const std::int64_t topk_indices_dims[]{4, 4};
    const std::vector<std::int32_t> topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(device_topk_indices, nullptr);
    const std::int64_t topk_length_dims[]{4};
    const std::vector<std::int32_t> topk_length(4, 4);
    auto* device_topk_length = allocate<std::int32_t>(4);
    ASSERT_NE(device_topk_length, nullptr);
    const std::int64_t out_dims[]{4, 2, 512};
    const std::vector<__half> out(4096, __float2half(123.0F));
    auto* device_out = allocate<__half>(4096);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t max_logits_dims[]{4, 2};
    const std::vector<float> max_logits(8, 123.0F);
    auto* device_max_logits = allocate<float>(8);
    ASSERT_NE(device_max_logits, nullptr);
    const std::int64_t lse_dims[]{4, 2};
    const std::vector<float> lse(8, 123.0F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
    const std::int64_t lse_indexer_dims[]{4, 2};
    const std::vector<float> lse_indexer(8, 123.0F);
    auto* device_lse_indexer = allocate<float>(8);
    ASSERT_NE(device_lse_indexer, nullptr);
    float softmax_scale       = 0.125F;
    std::int64_t indexer_topk = 2;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_kv, kv.data(), kv.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_sink, attn_sink.data(), attn_sink.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_indices, topk_indices.data(), topk_indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_length, topk_length.data(), topk_length.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_max_logits, max_logits.data(), max_logits.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse_indexer, lse_indexer.data(), lse_indexer.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view  = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto kv_view = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_sink_view =
        makeView(device_attn_sink, attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_length_view  = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto max_logits_view =
        makeView(device_max_logits, max_logits_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view = makeView(device_lse, lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_indexer_view = makeView(device_lse_indexer, lse_indexer_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto indexer_topk_view =
        makeView(&indexer_topk, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, kv_view, attn_sink_view, topk_indices_view, topk_length_view, out_view,
                        max_logits_view, lse_view, lse_indexer_view, softmax_scale_view, indexer_topk_view,
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
    EXPECT_STREQ(primitive, "Fp16SparseAttnFwdIndexer");
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
            std::vector<__half> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 0.4F, 2e-4 + 2e-3 * std::abs(0.4F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_max_logits, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.3862943611198906F, 2e-5 + 2e-4 * std::abs(1.3862943611198906F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse_indexer, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.0986122886681098F, 2e-5 + 2e-4 * std::abs(1.0986122886681098F)) << i;
            }
        }
        ASSERT_EQ(ftrainSparseAttnFwdIndexer(q_view, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                             out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                             indexer_topk_view, FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), 0.4F, 2e-4 + 2e-3 * std::abs(0.4F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_max_logits, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.3862943611198906F, 2e-5 + 2e-4 * std::abs(1.3862943611198906F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse_indexer, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.0986122886681098F, 2e-5 + 2e-4 * std::abs(1.0986122886681098F)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseAttnFwdIndexer(invalid_q, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                         out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                         indexer_topk_view, FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseAttnFwdIndexer(invalid_q, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                         out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                         indexer_topk_view, FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseAttnFwdIndexerStagedTest, Fp32SparseAttnFwdIndexerStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 512};
    const std::vector<float> q(4096, 0.0F);
    auto* device_q = allocate<float>(4096);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t kv_dims[]{4, 512};
    const std::vector<float> kv(2048, 0.5F);
    auto* device_kv = allocate<float>(2048);
    ASSERT_NE(device_kv, nullptr);
    const std::int64_t attn_sink_dims[]{2};
    const std::vector<float> attn_sink(2, 0.0F);
    auto* device_attn_sink = allocate<float>(2);
    ASSERT_NE(device_attn_sink, nullptr);
    const std::int64_t topk_indices_dims[]{4, 4};
    const std::vector<std::int32_t> topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(device_topk_indices, nullptr);
    const std::int64_t topk_length_dims[]{4};
    const std::vector<std::int32_t> topk_length(4, 4);
    auto* device_topk_length = allocate<std::int32_t>(4);
    ASSERT_NE(device_topk_length, nullptr);
    const std::int64_t out_dims[]{4, 2, 512};
    const std::vector<float> out(4096, 123.0F);
    auto* device_out = allocate<float>(4096);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t max_logits_dims[]{4, 2};
    const std::vector<float> max_logits(8, 123.0F);
    auto* device_max_logits = allocate<float>(8);
    ASSERT_NE(device_max_logits, nullptr);
    const std::int64_t lse_dims[]{4, 2};
    const std::vector<float> lse(8, 123.0F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
    const std::int64_t lse_indexer_dims[]{4, 2};
    const std::vector<float> lse_indexer(8, 123.0F);
    auto* device_lse_indexer = allocate<float>(8);
    ASSERT_NE(device_lse_indexer, nullptr);
    float softmax_scale       = 0.125F;
    std::int64_t indexer_topk = 2;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_kv, kv.data(), kv.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_sink, attn_sink.data(), attn_sink.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_indices, topk_indices.data(), topk_indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_length, topk_length.data(), topk_length.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_max_logits, max_logits.data(), max_logits.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse_indexer, lse_indexer.data(), lse_indexer.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view  = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto kv_view = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_sink_view =
        makeView(device_attn_sink, attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_length_view  = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto max_logits_view =
        makeView(device_max_logits, max_logits_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view = makeView(device_lse, lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_indexer_view = makeView(device_lse_indexer, lse_indexer_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto indexer_topk_view =
        makeView(&indexer_topk, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, kv_view, attn_sink_view, topk_indices_view, topk_length_view, out_view,
                        max_logits_view, lse_view, lse_indexer_view, softmax_scale_view, indexer_topk_view,
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
    EXPECT_STREQ(primitive, "Fp32SparseAttnFwdIndexer");
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
            std::vector<float> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.4F, 2e-5 + 2e-4 * std::abs(0.4F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_max_logits, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.3862943611198906F, 2e-5 + 2e-4 * std::abs(1.3862943611198906F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse_indexer, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.0986122886681098F, 2e-5 + 2e-4 * std::abs(1.0986122886681098F)) << i;
            }
        }
        ASSERT_EQ(ftrainSparseAttnFwdIndexer(q_view, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                             out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                             indexer_topk_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 0.4F, 2e-5 + 2e-4 * std::abs(0.4F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_max_logits, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.3862943611198906F, 2e-5 + 2e-4 * std::abs(1.3862943611198906F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse_indexer, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.0986122886681098F, 2e-5 + 2e-4 * std::abs(1.0986122886681098F)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseAttnFwdIndexer(invalid_q, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                         out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                         indexer_topk_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseAttnFwdIndexer(invalid_q, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                         out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                         indexer_topk_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseAttnFwdIndexerStagedTest, MixedFp32SparseAttnFwdIndexerStorageMode3RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> q(4096, 0x0000);
    auto* device_q = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t kv_dims[]{4, 512};
    const std::vector<float> kv(2048, 0.5F);
    auto* device_kv = allocate<float>(2048);
    ASSERT_NE(device_kv, nullptr);
    const std::int64_t attn_sink_dims[]{2};
    const std::vector<float> attn_sink(2, 0.0F);
    auto* device_attn_sink = allocate<float>(2);
    ASSERT_NE(device_attn_sink, nullptr);
    const std::int64_t topk_indices_dims[]{4, 4};
    const std::vector<std::int32_t> topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(device_topk_indices, nullptr);
    const std::int64_t topk_length_dims[]{4};
    const std::vector<std::int32_t> topk_length(4, 4);
    auto* device_topk_length = allocate<std::int32_t>(4);
    ASSERT_NE(device_topk_length, nullptr);
    const std::int64_t out_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> out(4096, 0x42f6);
    auto* device_out = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t max_logits_dims[]{4, 2};
    const std::vector<float> max_logits(8, 123.0F);
    auto* device_max_logits = allocate<float>(8);
    ASSERT_NE(device_max_logits, nullptr);
    const std::int64_t lse_dims[]{4, 2};
    const std::vector<float> lse(8, 123.0F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
    const std::int64_t lse_indexer_dims[]{4, 2};
    const std::vector<float> lse_indexer(8, 123.0F);
    auto* device_lse_indexer = allocate<float>(8);
    ASSERT_NE(device_lse_indexer, nullptr);
    float softmax_scale       = 0.125F;
    std::int64_t indexer_topk = 2;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_kv, kv.data(), kv.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_sink, attn_sink.data(), attn_sink.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_indices, topk_indices.data(), topk_indices.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_topk_length, topk_length.data(), topk_length.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_max_logits, max_logits.data(), max_logits.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse_indexer, lse_indexer.data(), lse_indexer.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view  = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto kv_view = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_sink_view =
        makeView(device_attn_sink, attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_length_view  = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto max_logits_view =
        makeView(device_max_logits, max_logits_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view = makeView(device_lse, lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_indexer_view = makeView(device_lse_indexer, lse_indexer_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto indexer_topk_view =
        makeView(&indexer_topk, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, kv_view, attn_sink_view, topk_indices_view, topk_length_view, out_view,
                        max_logits_view, lse_view, lse_indexer_view, softmax_scale_view, indexer_topk_view,
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
    EXPECT_STREQ(primitive, "MixedFp32SparseAttnFwdIndexer");
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
            std::vector<std::uint16_t> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.4F, 2e-3 + 1e-2 * std::abs(0.4F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_max_logits, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.3862943611198906F, 2e-5 + 2e-4 * std::abs(1.3862943611198906F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse_indexer, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.0986122886681098F, 2e-5 + 2e-4 * std::abs(1.0986122886681098F)) << i;
            }
        }
        ASSERT_EQ(ftrainSparseAttnFwdIndexer(q_view, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                             out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                             indexer_topk_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_out, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.4F, 2e-3 + 1e-2 * std::abs(0.4F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_max_logits, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.3862943611198906F, 2e-5 + 2e-4 * std::abs(1.3862943611198906F)) << i;
            }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_lse_indexer, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], 1.0986122886681098F, 2e-5 + 2e-4 * std::abs(1.0986122886681098F)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseAttnFwdIndexer(invalid_q, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                         out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                         indexer_topk_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseAttnFwdIndexer(invalid_q, kv_view, attn_sink_view, topk_indices_view, topk_length_view,
                                         out_view, max_logits_view, lse_view, lse_indexer_view, softmax_scale_view,
                                         indexer_topk_view, FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseAttnFwdIndexerStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_q_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> first_q(4096, 0x0000);
    auto* first_device_q = allocate<std::uint16_t>(4096);
    ASSERT_NE(first_device_q, nullptr);
    const std::int64_t first_kv_dims[]{4, 512};
    const std::vector<std::uint16_t> first_kv(2048, 0x3f00);
    auto* first_device_kv = allocate<std::uint16_t>(2048);
    ASSERT_NE(first_device_kv, nullptr);
    const std::int64_t first_attn_sink_dims[]{2};
    const std::vector<float> first_attn_sink(2, 0.0F);
    auto* first_device_attn_sink = allocate<float>(2);
    ASSERT_NE(first_device_attn_sink, nullptr);
    const std::int64_t first_topk_indices_dims[]{4, 4};
    const std::vector<std::int32_t> first_topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* first_device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(first_device_topk_indices, nullptr);
    const std::int64_t first_topk_length_dims[]{4};
    const std::vector<std::int32_t> first_topk_length(4, 4);
    auto* first_device_topk_length = allocate<std::int32_t>(4);
    ASSERT_NE(first_device_topk_length, nullptr);
    const std::int64_t first_out_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> first_out(4096, 0x42f6);
    auto* first_device_out = allocate<std::uint16_t>(4096);
    ASSERT_NE(first_device_out, nullptr);
    const std::int64_t first_max_logits_dims[]{4, 2};
    const std::vector<float> first_max_logits(8, 123.0F);
    auto* first_device_max_logits = allocate<float>(8);
    ASSERT_NE(first_device_max_logits, nullptr);
    const std::int64_t first_lse_dims[]{4, 2};
    const std::vector<float> first_lse(8, 123.0F);
    auto* first_device_lse = allocate<float>(8);
    ASSERT_NE(first_device_lse, nullptr);
    const std::int64_t first_lse_indexer_dims[]{4, 2};
    const std::vector<float> first_lse_indexer(8, 123.0F);
    auto* first_device_lse_indexer = allocate<float>(8);
    ASSERT_NE(first_device_lse_indexer, nullptr);
    float first_softmax_scale       = 0.125F;
    std::int64_t first_indexer_topk = 2;
    ASSERT_EQ(hipMemcpyAsync(first_device_q, first_q.data(), first_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_kv, first_kv.data(), first_kv.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_attn_sink, first_attn_sink.data(), first_attn_sink.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_topk_indices, first_topk_indices.data(),
                             first_topk_indices.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_topk_length, first_topk_length.data(),
                             first_topk_length.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_out, first_out.data(), first_out.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_max_logits, first_max_logits.data(), first_max_logits.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_lse, first_lse.data(), first_lse.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_lse_indexer, first_lse_indexer.data(),
                             first_lse_indexer.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_q_view =
        makeView(first_device_q, first_q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_kv_view =
        makeView(first_device_kv, first_kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_attn_sink_view    = makeView(first_device_attn_sink, first_attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_topk_indices_view = makeView(first_device_topk_indices, first_topk_indices_dims, 2,
                                            FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_topk_length_view  = makeView(first_device_topk_length, first_topk_length_dims, 1,
                                            FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_out_view =
        makeView(first_device_out, first_out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_max_logits_view = makeView(first_device_max_logits, first_max_logits_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_lse_view =
        makeView(first_device_lse, first_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_lse_indexer_view = makeView(first_device_lse_indexer, first_lse_indexer_dims, 2,
                                           FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_softmax_scale_view =
        makeView(&first_softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_indexer_topk_view =
        makeView(&first_indexer_topk, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_q_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> second_q(4096, 0x0000);
    auto* second_device_q = allocate<std::uint16_t>(4096);
    ASSERT_NE(second_device_q, nullptr);
    const std::int64_t second_kv_dims[]{4, 512};
    const std::vector<std::uint16_t> second_kv(2048, 0x3f00);
    auto* second_device_kv = allocate<std::uint16_t>(2048);
    ASSERT_NE(second_device_kv, nullptr);
    const std::int64_t second_attn_sink_dims[]{2};
    const std::vector<float> second_attn_sink(2, 0.0F);
    auto* second_device_attn_sink = allocate<float>(2);
    ASSERT_NE(second_device_attn_sink, nullptr);
    const std::int64_t second_topk_indices_dims[]{4, 4};
    const std::vector<std::int32_t> second_topk_indices{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    auto* second_device_topk_indices = allocate<std::int32_t>(16);
    ASSERT_NE(second_device_topk_indices, nullptr);
    const std::int64_t second_topk_length_dims[]{4};
    const std::vector<std::int32_t> second_topk_length(4, 4);
    auto* second_device_topk_length = allocate<std::int32_t>(4);
    ASSERT_NE(second_device_topk_length, nullptr);
    const std::int64_t second_out_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> second_out(4096, 0x42f6);
    auto* second_device_out = allocate<std::uint16_t>(4096);
    ASSERT_NE(second_device_out, nullptr);
    const std::int64_t second_max_logits_dims[]{4, 2};
    const std::vector<float> second_max_logits(8, 123.0F);
    auto* second_device_max_logits = allocate<float>(8);
    ASSERT_NE(second_device_max_logits, nullptr);
    const std::int64_t second_lse_dims[]{4, 2};
    const std::vector<float> second_lse(8, 123.0F);
    auto* second_device_lse = allocate<float>(8);
    ASSERT_NE(second_device_lse, nullptr);
    const std::int64_t second_lse_indexer_dims[]{4, 2};
    const std::vector<float> second_lse_indexer(8, 123.0F);
    auto* second_device_lse_indexer = allocate<float>(8);
    ASSERT_NE(second_device_lse_indexer, nullptr);
    float second_softmax_scale       = 0.125F;
    std::int64_t second_indexer_topk = 2;
    ASSERT_EQ(hipMemcpyAsync(second_device_q, second_q.data(), second_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_kv, second_kv.data(), second_kv.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_attn_sink, second_attn_sink.data(), second_attn_sink.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_topk_indices, second_topk_indices.data(),
                             second_topk_indices.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_topk_length, second_topk_length.data(),
                             second_topk_length.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_out, second_out.data(), second_out.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_max_logits, second_max_logits.data(),
                             second_max_logits.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_lse, second_lse.data(), second_lse.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_lse_indexer, second_lse_indexer.data(),
                             second_lse_indexer.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_q_view =
        makeView(second_device_q, second_q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_kv_view =
        makeView(second_device_kv, second_kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_attn_sink_view = makeView(second_device_attn_sink, second_attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_topk_indices_view = makeView(second_device_topk_indices, second_topk_indices_dims, 2,
                                             FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_topk_length_view  = makeView(second_device_topk_length, second_topk_length_dims, 1,
                                             FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_out_view =
        makeView(second_device_out, second_out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_max_logits_view = makeView(second_device_max_logits, second_max_logits_dims, 2,
                                           FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_lse_view =
        makeView(second_device_lse, second_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_lse_indexer_view = makeView(second_device_lse_indexer, second_lse_indexer_dims, 2,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_softmax_scale_view =
        makeView(&second_softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_indexer_topk_view =
        makeView(&second_indexer_topk, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_q_view, first_kv_view, first_attn_sink_view, first_topk_indices_view,
                        first_topk_length_view, first_out_view, first_max_logits_view, first_lse_view,
                        first_lse_indexer_view, first_softmax_scale_view, first_indexer_topk_view,
                        FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_q_view, second_kv_view, second_attn_sink_view, second_topk_indices_view,
                        second_topk_length_view, second_out_view, second_max_logits_view, second_lse_view,
                        second_lse_indexer_view, second_softmax_scale_view, second_indexer_topk_view,
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
        std::vector<std::uint16_t> actual(4096);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_out, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.4F, 2e-3 + 1e-2 * std::abs(0.4F)) << i;
        }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_max_logits, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_lse, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                 stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 1.3862943611198906F, 2e-5 + 2e-4 * std::abs(1.3862943611198906F)) << i;
        }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_lse_indexer, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 1.0986122886681098F, 2e-5 + 2e-4 * std::abs(1.0986122886681098F)) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(4096);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_out, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.4F, 2e-3 + 1e-2 * std::abs(0.4F)) << i;
        }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_max_logits, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_lse, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                 stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 1.3862943611198906F, 2e-5 + 2e-4 * std::abs(1.3862943611198906F)) << i;
        }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_lse_indexer, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 1.0986122886681098F, 2e-5 + 2e-4 * std::abs(1.0986122886681098F)) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(4096);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_out, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.4F, 2e-3 + 1e-2 * std::abs(0.4F)) << i;
        }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_max_logits, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_lse, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                 stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 1.3862943611198906F, 2e-5 + 2e-4 * std::abs(1.3862943611198906F)) << i;
        }
    }
    {
        std::vector<float> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_lse_indexer, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], 1.0986122886681098F, 2e-5 + 2e-4 * std::abs(1.0986122886681098F)) << i;
        }
    }
}

}  // namespace
