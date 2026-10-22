// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <flash_train/env.h>
#include <flash_train/flash_train.h>

#include "sparse_attn_bwd_staged_fixture.hpp"

namespace {

using ftrain_test::SparseAttnBwdStagedTest;

// dSink reduces 512 products per row; its nonzero result needs a relative
// FP32 rounding budget as well as the near-zero absolute tolerance.

TEST_F(SparseAttnBwdStagedTest, Bf16SparseAttnBwdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> q(4096, 0x0000);
    auto* device_q = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t kv_dims[]{4, 512};
    const std::vector<std::uint16_t> kv(2048, 0x3f00);
    auto* device_kv = allocate<std::uint16_t>(2048);
    ASSERT_NE(device_kv, nullptr);
    const std::int64_t out_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> out(4096, 0x3ecd);
    auto* device_out = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t dout_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> dout(4096, 0x3e80);
    auto* device_dout = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_dout, nullptr);
    const std::int64_t lse_dims[]{4, 2};
    const std::vector<float> lse(8, 1.3862943611198906F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
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
    const std::int64_t dq_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> dq(4096, 0x42f6);
    auto* device_dq = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_dq, nullptr);
    const std::int64_t dkv_dims[]{4, 512};
    const std::vector<std::uint16_t> dkv(2048, 0x42f6);
    auto* device_dkv = allocate<std::uint16_t>(2048);
    ASSERT_NE(device_dkv, nullptr);
    const std::int64_t dsink_dims[]{2};
    const std::vector<float> dsink(2, 123.0F);
    auto* device_dsink = allocate<float>(2);
    ASSERT_NE(device_dsink, nullptr);
    float softmax_scale = 0.125F;
    // Stored LSE excludes the sink; include it in the probability denominator.
    // Different output storage precisions also change dOut*out.
    const float delta          = 512 * 0.25F * 0.400390625F;
    const float probability    = 1 / (std::exp(lse[0]) + std::exp(attn_sink[0]));
    const float expected_dq    = 4 * probability * (64 - delta) * 0.5F * 0.125F;
    const float expected_dkv   = 8 * probability * 0.25F;
    const float expected_dsink = -4 * probability * delta;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_kv, kv.data(), kv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_dout, dout.data(), dout.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
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
    ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view    = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto kv_view   = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view  = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dout_view = makeView(device_dout, dout_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view  = makeView(device_lse, lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_sink_view =
        makeView(device_attn_sink, attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_length_view  = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dq_view  = makeView(device_dq, dq_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dkv_view = makeView(device_dkv, dkv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dsink_view =
        makeView(device_dsink, dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                        topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Bf16SparseAttnBwd");
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
            hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, expected_dq, 2e-3) << i;
            }
        }
        {
            std::vector<std::uint16_t> actual(2048);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dkv, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, expected_dkv, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(2);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dsink, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], expected_dsink, 2e-5 + 2e-6 * std::abs(expected_dsink)) << i;
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(
            hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainSparseAttnBwd(q_view, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                      topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                      FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, expected_dq, 2e-3) << i;
            }
        }
        {
            std::vector<std::uint16_t> actual(2048);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dkv, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, expected_dkv, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(2);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dsink, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], expected_dsink, 2e-5 + 2e-6 * std::abs(expected_dsink)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseAttnBwd(invalid_q, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                  topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                  FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseAttnBwd(invalid_q, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                  topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                  FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseAttnBwdStagedTest, Fp16SparseAttnBwdStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 512};
    const std::vector<__half> q(4096, __float2half(0.0F));
    auto* device_q = allocate<__half>(4096);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t kv_dims[]{4, 512};
    const std::vector<__half> kv(2048, __float2half(0.5F));
    auto* device_kv = allocate<__half>(2048);
    ASSERT_NE(device_kv, nullptr);
    const std::int64_t out_dims[]{4, 2, 512};
    const std::vector<__half> out(4096, __float2half(0.4F));
    auto* device_out = allocate<__half>(4096);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t dout_dims[]{4, 2, 512};
    const std::vector<__half> dout(4096, __float2half(0.25F));
    auto* device_dout = allocate<__half>(4096);
    ASSERT_NE(device_dout, nullptr);
    const std::int64_t lse_dims[]{4, 2};
    const std::vector<float> lse(8, 1.3862943611198906F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
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
    const std::int64_t dq_dims[]{4, 2, 512};
    const std::vector<__half> dq(4096, __float2half(123.0F));
    auto* device_dq = allocate<__half>(4096);
    ASSERT_NE(device_dq, nullptr);
    const std::int64_t dkv_dims[]{4, 512};
    const std::vector<__half> dkv(2048, __float2half(123.0F));
    auto* device_dkv = allocate<__half>(2048);
    ASSERT_NE(device_dkv, nullptr);
    const std::int64_t dsink_dims[]{2};
    const std::vector<float> dsink(2, 123.0F);
    auto* device_dsink = allocate<float>(2);
    ASSERT_NE(device_dsink, nullptr);
    float softmax_scale = 0.125F;
    // Stored LSE excludes the sink; include it in the probability denominator.
    // Different output storage precisions also change dOut*out.
    const float delta          = 512 * 0.25F * __half2float(out[0]);
    const float probability    = 1 / (std::exp(lse[0]) + std::exp(attn_sink[0]));
    const float expected_dq    = 4 * probability * (64 - delta) * 0.5F * 0.125F;
    const float expected_dkv   = 8 * probability * 0.25F;
    const float expected_dsink = -4 * probability * delta;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_kv, kv.data(), kv.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dout, dout.data(), dout.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
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
    ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view    = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto kv_view   = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view  = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dout_view = makeView(device_dout, dout_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view  = makeView(device_lse, lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_sink_view =
        makeView(device_attn_sink, attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_length_view  = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dq_view  = makeView(device_dq, dq_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dkv_view = makeView(device_dkv, dkv_dims, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dsink_view =
        makeView(device_dsink, dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                        topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view, FTRAIN_NUMERIC_TYPE_FP16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp16SparseAttnBwd");
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
        ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), expected_dq, 2e-4) << i;
            }
        }
        {
            std::vector<__half> actual(2048);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dkv, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), expected_dkv, 2e-4) << i;
            }
        }
        {
            std::vector<float> actual(2);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dsink, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], expected_dsink, 2e-5 + 2e-6 * std::abs(expected_dsink)) << i;
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainSparseAttnBwd(q_view, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                      topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                      FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<__half> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), expected_dq, 2e-4) << i;
            }
        }
        {
            std::vector<__half> actual(2048);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dkv, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(__half2float(actual[i]), expected_dkv, 2e-4) << i;
            }
        }
        {
            std::vector<float> actual(2);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dsink, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], expected_dsink, 2e-5 + 2e-6 * std::abs(expected_dsink)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseAttnBwd(invalid_q, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                  topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                  FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseAttnBwd(invalid_q, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                  topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                  FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseAttnBwdStagedTest, Fp32SparseAttnBwdStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 512};
    const std::vector<float> q(4096, 0.0F);
    auto* device_q = allocate<float>(4096);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t kv_dims[]{4, 512};
    const std::vector<float> kv(2048, 0.5F);
    auto* device_kv = allocate<float>(2048);
    ASSERT_NE(device_kv, nullptr);
    const std::int64_t out_dims[]{4, 2, 512};
    const std::vector<float> out(4096, 0.4F);
    auto* device_out = allocate<float>(4096);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t dout_dims[]{4, 2, 512};
    const std::vector<float> dout(4096, 0.25F);
    auto* device_dout = allocate<float>(4096);
    ASSERT_NE(device_dout, nullptr);
    const std::int64_t lse_dims[]{4, 2};
    const std::vector<float> lse(8, 1.3862943611198906F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
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
    const std::int64_t dq_dims[]{4, 2, 512};
    const std::vector<float> dq(4096, 123.0F);
    auto* device_dq = allocate<float>(4096);
    ASSERT_NE(device_dq, nullptr);
    const std::int64_t dkv_dims[]{4, 512};
    const std::vector<float> dkv(2048, 123.0F);
    auto* device_dkv = allocate<float>(2048);
    ASSERT_NE(device_dkv, nullptr);
    const std::int64_t dsink_dims[]{2};
    const std::vector<float> dsink(2, 123.0F);
    auto* device_dsink = allocate<float>(2);
    ASSERT_NE(device_dsink, nullptr);
    float softmax_scale = 0.125F;
    // Stored LSE excludes the sink; include it in the probability denominator.
    // Different output storage precisions also change dOut*out.
    const float delta          = 512 * 0.25F * out[0];
    const float probability    = 1 / (std::exp(lse[0]) + std::exp(attn_sink[0]));
    const float expected_dq    = 4 * probability * (64 - delta) * 0.5F * 0.125F;
    const float expected_dkv   = 8 * probability * 0.25F;
    const float expected_dsink = -4 * probability * delta;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_kv, kv.data(), kv.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dout, dout.data(), dout.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
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
    ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view    = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto kv_view   = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view  = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dout_view = makeView(device_dout, dout_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view  = makeView(device_lse, lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_sink_view =
        makeView(device_attn_sink, attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_length_view  = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dq_view  = makeView(device_dq, dq_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dkv_view = makeView(device_dkv, dkv_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dsink_view =
        makeView(device_dsink, dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                        topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp32SparseAttnBwd");
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
        ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(4096);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], expected_dq, 2e-5) << i; }
        }
        {
            std::vector<float> actual(2048);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dkv, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], expected_dkv, 2e-5) << i; }
        }
        {
            std::vector<float> actual(2);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dsink, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], expected_dsink, 2e-5 + 2e-6 * std::abs(expected_dsink)) << i;
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainSparseAttnBwd(q_view, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                      topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                      FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(4096);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], expected_dq, 2e-5) << i; }
        }
        {
            std::vector<float> actual(2048);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dkv, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], expected_dkv, 2e-5) << i; }
        }
        {
            std::vector<float> actual(2);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dsink, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], expected_dsink, 2e-5 + 2e-6 * std::abs(expected_dsink)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseAttnBwd(invalid_q, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                  topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseAttnBwd(invalid_q, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                  topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseAttnBwdStagedTest, MixedFp32SparseAttnBwdStorageMode3RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> q(4096, 0x0000);
    auto* device_q = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t kv_dims[]{4, 512};
    const std::vector<float> kv(2048, 0.5F);
    auto* device_kv = allocate<float>(2048);
    ASSERT_NE(device_kv, nullptr);
    const std::int64_t out_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> out(4096, 0x3ecd);
    auto* device_out = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t dout_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> dout(4096, 0x3e80);
    auto* device_dout = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_dout, nullptr);
    const std::int64_t lse_dims[]{4, 2};
    const std::vector<float> lse(8, 1.3862943611198906F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
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
    const std::int64_t dq_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> dq(4096, 0x42f6);
    auto* device_dq = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_dq, nullptr);
    const std::int64_t dkv_dims[]{4, 512};
    const std::vector<float> dkv(2048, 123.0F);
    auto* device_dkv = allocate<float>(2048);
    ASSERT_NE(device_dkv, nullptr);
    const std::int64_t dsink_dims[]{2};
    const std::vector<float> dsink(2, 123.0F);
    auto* device_dsink = allocate<float>(2);
    ASSERT_NE(device_dsink, nullptr);
    float softmax_scale = 0.125F;
    // Stored LSE excludes the sink; include it in the probability denominator.
    // Different output storage precisions also change dOut*out.
    const float delta          = 512 * 0.25F * 0.400390625F;
    const float probability    = 1 / (std::exp(lse[0]) + std::exp(attn_sink[0]));
    const float expected_dq    = 4 * probability * (64 - delta) * 0.5F * 0.125F;
    const float expected_dkv   = 8 * probability * 0.25F;
    const float expected_dsink = -4 * probability * delta;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_kv, kv.data(), kv.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_dout, dout.data(), dout.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
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
    ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view    = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto kv_view   = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view  = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dout_view = makeView(device_dout, dout_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view  = makeView(device_lse, lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_sink_view =
        makeView(device_attn_sink, attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_length_view  = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dq_view  = makeView(device_dq, dq_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dkv_view = makeView(device_dkv, dkv_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dsink_view =
        makeView(device_dsink, dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, q_view, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                        topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32SparseAttnBwd");
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
            hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, expected_dq, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(2048);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dkv, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], expected_dkv, 2e-5) << i; }
        }
        {
            std::vector<float> actual(2);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dsink, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], expected_dsink, 2e-5 + 2e-6 * std::abs(expected_dsink)) << i;
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(
            hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainSparseAttnBwd(q_view, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                      topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                      FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(4096);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, expected_dq, 2e-3) << i;
            }
        }
        {
            std::vector<float> actual(2048);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dkv, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], expected_dkv, 2e-5) << i; }
        }
        {
            std::vector<float> actual(2);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dsink, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], expected_dsink, 2e-5 + 2e-6 * std::abs(expected_dsink)) << i;
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseAttnBwd(invalid_q, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                  topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseAttnBwd(invalid_q, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                                  topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view,
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseAttnBwdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_q_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> first_q(4096, 0x0000);
    auto* first_device_q = allocate<std::uint16_t>(4096);
    ASSERT_NE(first_device_q, nullptr);
    const std::int64_t first_kv_dims[]{4, 512};
    const std::vector<std::uint16_t> first_kv(2048, 0x3f00);
    auto* first_device_kv = allocate<std::uint16_t>(2048);
    ASSERT_NE(first_device_kv, nullptr);
    const std::int64_t first_out_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> first_out(4096, 0x3ecd);
    auto* first_device_out = allocate<std::uint16_t>(4096);
    ASSERT_NE(first_device_out, nullptr);
    const std::int64_t first_dout_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> first_dout(4096, 0x3e80);
    auto* first_device_dout = allocate<std::uint16_t>(4096);
    ASSERT_NE(first_device_dout, nullptr);
    const std::int64_t first_lse_dims[]{4, 2};
    const std::vector<float> first_lse(8, 1.3862943611198906F);
    auto* first_device_lse = allocate<float>(8);
    ASSERT_NE(first_device_lse, nullptr);
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
    const std::int64_t first_dq_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> first_dq(4096, 0x42f6);
    auto* first_device_dq = allocate<std::uint16_t>(4096);
    ASSERT_NE(first_device_dq, nullptr);
    const std::int64_t first_dkv_dims[]{4, 512};
    const std::vector<std::uint16_t> first_dkv(2048, 0x42f6);
    auto* first_device_dkv = allocate<std::uint16_t>(2048);
    ASSERT_NE(first_device_dkv, nullptr);
    const std::int64_t first_dsink_dims[]{2};
    const std::vector<float> first_dsink(2, 123.0F);
    auto* first_device_dsink = allocate<float>(2);
    ASSERT_NE(first_device_dsink, nullptr);
    float first_softmax_scale = 0.125F;
    // Stored LSE excludes the sink; include it in the probability denominator.
    // Different output storage precisions also change dOut*out.
    const float first_delta          = 512 * 0.25F * 0.400390625F;
    const float first_probability    = 1 / (std::exp(first_lse[0]) + std::exp(first_attn_sink[0]));
    const float first_expected_dq    = 4 * first_probability * (64 - first_delta) * 0.5F * 0.125F;
    const float first_expected_dkv   = 8 * first_probability * 0.25F;
    const float first_expected_dsink = -4 * first_probability * first_delta;
    ASSERT_EQ(hipMemcpyAsync(first_device_q, first_q.data(), first_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_kv, first_kv.data(), first_kv.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_out, first_out.data(), first_out.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_dout, first_dout.data(), first_dout.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_lse, first_lse.data(), first_lse.size() * sizeof(float),
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
    ASSERT_EQ(hipMemcpyAsync(first_device_dq, first_dq.data(), first_dq.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_dkv, first_dkv.data(), first_dkv.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_dsink, first_dsink.data(), first_dsink.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_q_view =
        makeView(first_device_q, first_q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_kv_view =
        makeView(first_device_kv, first_kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_out_view =
        makeView(first_device_out, first_out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_dout_view =
        makeView(first_device_dout, first_dout_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_lse_view =
        makeView(first_device_lse, first_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_attn_sink_view    = makeView(first_device_attn_sink, first_attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_topk_indices_view = makeView(first_device_topk_indices, first_topk_indices_dims, 2,
                                            FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_topk_length_view  = makeView(first_device_topk_length, first_topk_length_dims, 1,
                                            FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_dq_view =
        makeView(first_device_dq, first_dq_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_dkv_view =
        makeView(first_device_dkv, first_dkv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_dsink_view = makeView(first_device_dsink, first_dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_softmax_scale_view =
        makeView(&first_softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_q_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> second_q(4096, 0x0000);
    auto* second_device_q = allocate<std::uint16_t>(4096);
    ASSERT_NE(second_device_q, nullptr);
    const std::int64_t second_kv_dims[]{4, 512};
    const std::vector<std::uint16_t> second_kv(2048, 0x3f00);
    auto* second_device_kv = allocate<std::uint16_t>(2048);
    ASSERT_NE(second_device_kv, nullptr);
    const std::int64_t second_out_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> second_out(4096, 0x3ecd);
    auto* second_device_out = allocate<std::uint16_t>(4096);
    ASSERT_NE(second_device_out, nullptr);
    const std::int64_t second_dout_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> second_dout(4096, 0x3e80);
    auto* second_device_dout = allocate<std::uint16_t>(4096);
    ASSERT_NE(second_device_dout, nullptr);
    const std::int64_t second_lse_dims[]{4, 2};
    const std::vector<float> second_lse(8, 1.3862943611198906F);
    auto* second_device_lse = allocate<float>(8);
    ASSERT_NE(second_device_lse, nullptr);
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
    const std::int64_t second_dq_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> second_dq(4096, 0x42f6);
    auto* second_device_dq = allocate<std::uint16_t>(4096);
    ASSERT_NE(second_device_dq, nullptr);
    const std::int64_t second_dkv_dims[]{4, 512};
    const std::vector<std::uint16_t> second_dkv(2048, 0x42f6);
    auto* second_device_dkv = allocate<std::uint16_t>(2048);
    ASSERT_NE(second_device_dkv, nullptr);
    const std::int64_t second_dsink_dims[]{2};
    const std::vector<float> second_dsink(2, 123.0F);
    auto* second_device_dsink = allocate<float>(2);
    ASSERT_NE(second_device_dsink, nullptr);
    float second_softmax_scale = 0.125F;
    // Stored LSE excludes the sink; include it in the probability denominator.
    // Different output storage precisions also change dOut*out.
    const float second_delta          = 512 * 0.25F * 0.400390625F;
    const float second_probability    = 1 / (std::exp(second_lse[0]) + std::exp(second_attn_sink[0]));
    const float second_expected_dq    = 4 * second_probability * (64 - second_delta) * 0.5F * 0.125F;
    const float second_expected_dkv   = 8 * second_probability * 0.25F;
    const float second_expected_dsink = -4 * second_probability * second_delta;
    ASSERT_EQ(hipMemcpyAsync(second_device_q, second_q.data(), second_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_kv, second_kv.data(), second_kv.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_out, second_out.data(), second_out.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_dout, second_dout.data(), second_dout.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_lse, second_lse.data(), second_lse.size() * sizeof(float),
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
    ASSERT_EQ(hipMemcpyAsync(second_device_dq, second_dq.data(), second_dq.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_dkv, second_dkv.data(), second_dkv.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_dsink, second_dsink.data(), second_dsink.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_q_view =
        makeView(second_device_q, second_q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_kv_view =
        makeView(second_device_kv, second_kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_out_view =
        makeView(second_device_out, second_out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_dout_view = makeView(second_device_dout, second_dout_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_lse_view =
        makeView(second_device_lse, second_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_attn_sink_view = makeView(second_device_attn_sink, second_attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_topk_indices_view = makeView(second_device_topk_indices, second_topk_indices_dims, 2,
                                             FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_topk_length_view  = makeView(second_device_topk_length, second_topk_length_dims, 1,
                                             FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_dq_view =
        makeView(second_device_dq, second_dq_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_dkv_view =
        makeView(second_device_dkv, second_dkv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_dsink_view = makeView(second_device_dsink, second_dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_softmax_scale_view =
        makeView(&second_softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_q_view, first_kv_view, first_out_view, first_dout_view, first_lse_view,
                        first_attn_sink_view, first_topk_indices_view, first_topk_length_view, first_dq_view,
                        first_dkv_view, first_dsink_view, first_softmax_scale_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_q_view, second_kv_view, second_out_view, second_dout_view, second_lse_view,
                        second_attn_sink_view, second_topk_indices_view, second_topk_length_view, second_dq_view,
                        second_dkv_view, second_dsink_view, second_softmax_scale_view, FTRAIN_NUMERIC_TYPE_BF16));
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
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_dq, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, first_expected_dq, 2e-3) << i;
        }
    }
    {
        std::vector<std::uint16_t> actual(2048);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_dkv, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, first_expected_dkv, 2e-3) << i;
        }
    }
    {
        std::vector<float> actual(2);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_dsink, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], first_expected_dsink, 2e-5 + 2e-6 * std::abs(first_expected_dsink)) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(4096);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_dq, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, second_expected_dq, 2e-3) << i;
        }
    }
    {
        std::vector<std::uint16_t> actual(2048);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_dkv, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, second_expected_dkv, 2e-3) << i;
        }
    }
    {
        std::vector<float> actual(2);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_dsink, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], second_expected_dsink, 2e-5 + 2e-6 * std::abs(second_expected_dsink)) << i;
        }
    }
}

class ScopedEnvironment {
  public:
    explicit ScopedEnvironment(const char* name)
        : name_(name), present_(std::getenv(name) != nullptr), saved_(present_ ? std::getenv(name) : "") {}
    ~ScopedEnvironment() {
        if (present_)
            setenv(name_, saved_.c_str(), 1);
        else
            unsetenv(name_);
    }

  private:
    const char* name_;
    bool present_;
    std::string saved_;
};

TEST_F(SparseAttnBwdStagedTest, WorkspaceModeSeparatesPlansAndCacheEntries) {
    ScopedEnvironment restore(DSA_BWD_WORKSPACE);
    // Metadata-only planning: these small borrowed allocations are never launched.
    const std::array<std::vector<std::int64_t>, 12> dims{
        {{4097, 64, 576},
         {4096, 576},
         {4097, 64, 512},
         {4097, 64, 512},
         {4097, 64},
         {64},
         {4097, 2048},
         {4097},
         {4097, 64, 576},
         {4096, 576},
         {64},
         {}}
    };
    const std::int64_t q_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> q(4096, 0x0000);
    auto* device_q = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t kv_dims[]{4, 512};
    const std::vector<std::uint16_t> kv(2048, 0x3f00);
    auto* device_kv = allocate<std::uint16_t>(2048);
    ASSERT_NE(device_kv, nullptr);
    const std::int64_t out_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> out(4096, 0x3ecd);
    auto* device_out = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t dout_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> dout(4096, 0x0000);
    auto* device_dout = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_dout, nullptr);
    const std::int64_t lse_dims[]{4, 2};
    const std::vector<float> lse(8, 1.3862943611198906F);
    auto* device_lse = allocate<float>(8);
    ASSERT_NE(device_lse, nullptr);
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
    const std::int64_t dq_dims[]{4, 2, 512};
    const std::vector<std::uint16_t> dq(4096, 0x42f6);
    auto* device_dq = allocate<std::uint16_t>(4096);
    ASSERT_NE(device_dq, nullptr);
    const std::int64_t dkv_dims[]{4, 512};
    const std::vector<std::uint16_t> dkv(2048, 0x42f6);
    auto* device_dkv = allocate<std::uint16_t>(2048);
    ASSERT_NE(device_dkv, nullptr);
    const std::int64_t dsink_dims[]{2};
    const std::vector<float> dsink(2, 123.0F);
    auto* device_dsink = allocate<float>(2);
    ASSERT_NE(device_dsink, nullptr);
    float softmax_scale = 0.125F;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_kv, kv.data(), kv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_dout, dout.data(), dout.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
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
    ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view    = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto kv_view   = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view  = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dout_view = makeView(device_dout, dout_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view  = makeView(device_lse, lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_sink_view =
        makeView(device_attn_sink, attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_length_view  = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dq_view  = makeView(device_dq, dq_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dkv_view = makeView(device_dkv, dkv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dsink_view =
        makeView(device_dsink, dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    q_view.dims                 = dims[0].data();
    q_view.num_dims             = dims[0].size();
    kv_view.dims                = dims[1].data();
    kv_view.num_dims            = dims[1].size();
    out_view.dims               = dims[2].data();
    out_view.num_dims           = dims[2].size();
    dout_view.dims              = dims[3].data();
    dout_view.num_dims          = dims[3].size();
    lse_view.dims               = dims[4].data();
    lse_view.num_dims           = dims[4].size();
    attn_sink_view.dims         = dims[5].data();
    attn_sink_view.num_dims     = dims[5].size();
    topk_indices_view.dims      = dims[6].data();
    topk_indices_view.num_dims  = dims[6].size();
    topk_length_view.dims       = dims[7].data();
    topk_length_view.num_dims   = dims[7].size();
    dq_view.dims                = dims[8].data();
    dq_view.num_dims            = dims[8].size();
    dkv_view.dims               = dims[9].data();
    dkv_view.num_dims           = dims[9].size();
    dsink_view.dims             = dims[10].data();
    dsink_view.num_dims         = dims[10].size();
    softmax_scale_view.dims     = dims[11].data();
    softmax_scale_view.num_dims = dims[11].size();
    FTrainArgs args             = createArgs();
    ASSERT_TRUE(setArgs(args, q_view, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                        topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view, FTRAIN_NUMERIC_TYPE_BF16));
    std::array<FTrainPlan, 3> plans{};
    std::array<std::uint64_t, 3> bytes{};
    const std::array<const char*, 3> modes{"0", "1", "0"};
    for (std::size_t i = 0; i < modes.size(); ++i) {
        setenv(DSA_BWD_WORKSPACE, modes[i], 1);
        ASSERT_EQ(ftrainPlanCreate(&plans[i], args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
        trackPlan(plans[i]);
        ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plans[i], 0, &bytes[i]), FTRAIN_STATUS_SUCCESS);
    }
    constexpr std::uint64_t fixed   = 4096ULL * 576 * sizeof(float) + 256;
    constexpr std::uint64_t per_row = (2048ULL * 64 + 4096 + 36ULL * 4 * 64 * 4) * sizeof(std::int32_t);
    EXPECT_EQ(bytes[0], fixed + 4097 * per_row);
    EXPECT_EQ(bytes[1], fixed + 2049 * per_row);
    EXPECT_EQ(bytes[2], bytes[0]);
    std::uint64_t retained = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plans[1], 0, &retained), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(retained, bytes[1]);
}

TEST_F(SparseAttnBwdStagedTest, CachedModeSeparatesSelectionAndRetainsConfiguredPlan) {
    ScopedEnvironment restore(DSA_SAB2_CACHED);
    const std::int64_t q_dims[]{4, 8, 576};
    const std::vector<std::uint16_t> q(18432, 0x0000);
    auto* device_q = allocate<std::uint16_t>(18432);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t kv_dims[]{4, 576};
    const std::vector<std::uint16_t> kv(2304, 0x3f00);
    auto* device_kv = allocate<std::uint16_t>(2304);
    ASSERT_NE(device_kv, nullptr);
    const std::int64_t out_dims[]{4, 8, 512};
    const std::vector<std::uint16_t> out(16384, 0x3ecd);
    auto* device_out = allocate<std::uint16_t>(16384);
    ASSERT_NE(device_out, nullptr);
    const std::int64_t dout_dims[]{4, 8, 512};
    const std::vector<std::uint16_t> dout(16384, 0x0000);
    auto* device_dout = allocate<std::uint16_t>(16384);
    ASSERT_NE(device_dout, nullptr);
    const std::int64_t lse_dims[]{4, 8};
    const std::vector<float> lse(32, 1.3862943611198906F);
    auto* device_lse = allocate<float>(32);
    ASSERT_NE(device_lse, nullptr);
    const std::int64_t attn_sink_dims[]{8};
    const std::vector<float> attn_sink(8, 0.0F);
    auto* device_attn_sink = allocate<float>(8);
    ASSERT_NE(device_attn_sink, nullptr);
    const std::int64_t topk_indices_dims[]{4, 128};
    const std::vector<std::int32_t> topk_indices(512, 0);
    auto* device_topk_indices = allocate<std::int32_t>(512);
    ASSERT_NE(device_topk_indices, nullptr);
    const std::int64_t topk_length_dims[]{4};
    const std::vector<std::int32_t> topk_length(4, 128);
    auto* device_topk_length = allocate<std::int32_t>(4);
    ASSERT_NE(device_topk_length, nullptr);
    const std::int64_t dq_dims[]{4, 8, 576};
    const std::vector<std::uint16_t> dq(18432, 0x42f6);
    auto* device_dq = allocate<std::uint16_t>(18432);
    ASSERT_NE(device_dq, nullptr);
    const std::int64_t dkv_dims[]{4, 576};
    const std::vector<std::uint16_t> dkv(2304, 0x42f6);
    auto* device_dkv = allocate<std::uint16_t>(2304);
    ASSERT_NE(device_dkv, nullptr);
    const std::int64_t dsink_dims[]{8};
    const std::vector<float> dsink(8, 123.0F);
    auto* device_dsink = allocate<float>(8);
    ASSERT_NE(device_dsink, nullptr);
    float softmax_scale = 0.125F;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_kv, kv.data(), kv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_out, out.data(), out.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_dout, dout.data(), dout.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_lse, lse.data(), lse.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
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
    ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_dkv, dkv.data(), dkv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dsink, dsink.data(), dsink.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view    = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto kv_view   = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto out_view  = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dout_view = makeView(device_dout, dout_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto lse_view  = makeView(device_lse, lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_sink_view =
        makeView(device_attn_sink, attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_indices_view = makeView(device_topk_indices, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto topk_length_view  = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dq_view  = makeView(device_dq, dq_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dkv_view = makeView(device_dkv, dkv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dsink_view =
        makeView(device_dsink, dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto softmax_scale_view =
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_TRUE(setArgs(args, q_view, kv_view, out_view, dout_view, lse_view, attn_sink_view, topk_indices_view,
                        topk_length_view, dq_view, dkv_view, dsink_view, softmax_scale_view, FTRAIN_NUMERIC_TYPE_BF16));
    setenv(DSA_SAB2_CACHED, "1", 1);
    FTrainPlan cached = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&cached, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(cached);
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(cached, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    auto* workspace = allocate<unsigned char>(bytes);
    ASSERT_NE(workspace, nullptr);
    setenv(DSA_SAB2_CACHED, "0", 1);
    FTrainPlan disabled = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&disabled, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(disabled);
    std::uint64_t disabled_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(disabled, 0, &disabled_bytes), FTRAIN_STATUS_SUCCESS);
    EXPECT_LT(disabled_bytes, bytes);
    ASSERT_EQ(ftrainPlanExecutePrimitive(disabled, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
    // The existing plan retains the configuration captured at creation.
    ASSERT_EQ(ftrainPlanExecutePrimitive(cached, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(18432);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost,
                                 stream_),
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
        std::vector<std::uint16_t> actual(2304);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dkv, actual.size() * sizeof(std::uint16_t),
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
        std::vector<float> actual(8);
        ASSERT_EQ(
            hipMemcpyAsync(actual.data(), device_dsink, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
    }
    setenv(DSA_SAB2_CACHED, "1", 1);
    ASSERT_EQ(ftrainPlanCreate(&disabled, args), FTRAIN_STATUS_SUCCESS);
    trackPlan(disabled);
}
}  // namespace
