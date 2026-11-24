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
#include <hip/hip_runtime.h>

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
    FTrainSparseAttnBwdArgs args{};
    args.q    = makeView(device_q, q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    args.kv   = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    args.out  = makeView(device_out, out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    args.dout = makeView(device_dout, dout_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    args.lse  = makeView(device_lse, lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    args.attn_sink =
        makeView(device_attn_sink, attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    args.topk_indices = makeView(device_topk_indices, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                 FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    args.topk_length  = makeView(device_topk_length, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                 FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    args.dq           = makeView(device_dq, dq_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    args.dkv   = makeView(device_dkv, dkv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    args.dsink = makeView(device_dsink, dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    args.softmax_scale           = 0.125F;
    args.compute_type            = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainSparseAttnBwdPlan plan = createPlan(args);
    ASSERT_NE(plan, nullptr);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainSparseAttnBwdGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainSparseAttnBwdGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    // The default may change when the Finder gains more named Primitives.
    ASSERT_NE(primitive, nullptr);
    ASSERT_EQ(ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainSparseAttnBwdExecute(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
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
        ASSERT_EQ(ftrainSparseAttnBwdExecute(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
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
        ASSERT_EQ(ftrainSparseAttnBwd(&args, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
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
    auto invalid_args             = args;
    invalid_args.q.is_host_memory = true;
    EXPECT_EQ(ftrainSparseAttnBwd(&invalid_args, workspace, bytes, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_args                = args;
    invalid_args.q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainSparseAttnBwd(&invalid_args, workspace, bytes, stream_), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(SparseAttnBwdStagedTest, BindsEachPlanToItsOwnBuffers) {
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
    FTrainSparseAttnBwdArgs first_args{};
    first_args.q =
        makeView(first_device_q, first_q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    first_args.kv =
        makeView(first_device_kv, first_kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    first_args.out =
        makeView(first_device_out, first_out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    first_args.dout =
        makeView(first_device_dout, first_dout_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    first_args.lse =
        makeView(first_device_lse, first_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    first_args.attn_sink    = makeView(first_device_attn_sink, first_attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    first_args.topk_indices = makeView(first_device_topk_indices, first_topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    first_args.topk_length  = makeView(first_device_topk_length, first_topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    first_args.dq =
        makeView(first_device_dq, first_dq_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    first_args.dkv =
        makeView(first_device_dkv, first_dkv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    first_args.dsink         = makeView(first_device_dsink, first_dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                        FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    first_args.softmax_scale = 0.125F;
    first_args.compute_type  = FTRAIN_NUMERIC_TYPE_BF16;
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
    FTrainSparseAttnBwdArgs second_args{};
    second_args.q =
        makeView(second_device_q, second_q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    second_args.kv =
        makeView(second_device_kv, second_kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    second_args.out =
        makeView(second_device_out, second_out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    second_args.dout = makeView(second_device_dout, second_dout_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    second_args.lse =
        makeView(second_device_lse, second_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    second_args.attn_sink    = makeView(second_device_attn_sink, second_attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                        FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    second_args.topk_indices = makeView(second_device_topk_indices, second_topk_indices_dims, 2,
                                        FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    second_args.topk_length = makeView(second_device_topk_length, second_topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    second_args.dq =
        makeView(second_device_dq, second_dq_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    second_args.dkv =
        makeView(second_device_dkv, second_dkv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    second_args.dsink                  = makeView(second_device_dsink, second_dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32,
                                                  FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    second_args.softmax_scale          = 0.125F;
    second_args.compute_type           = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainSparseAttnBwdPlan first_plan = createPlan(first_args);
    ASSERT_NE(first_plan, nullptr);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes),
              FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    FTrainSparseAttnBwdPlan second_plan = createPlan(second_args);
    ASSERT_NE(second_plan, nullptr);
    std::uint64_t second_bytes = 0;
    ASSERT_EQ(ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes(second_plan, 0, &second_bytes),
              FTRAIN_STATUS_SUCCESS);
    void* second_workspace = second_bytes ? allocate<unsigned char>(second_bytes) : nullptr;
    if (second_bytes) { ASSERT_NE(second_workspace, nullptr); }
    ASSERT_EQ(ftrainSparseAttnBwdExecute(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
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
    ASSERT_EQ(ftrainSparseAttnBwdExecute(second_plan, 0, second_workspace, second_bytes, stream_),
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

// Selection reads only port metadata, so every port may alias one allocation.
TEST_F(SparseAttnBwdStagedTest, RejectsNullHandlesOutputsAndInvalidComputeTypes) {
    void* storage = allocate<std::uint64_t>(2048);
    ASSERT_NE(storage, nullptr);
    const std::int64_t q_dims[]{2, 2, 512};
    const std::int64_t kv_dims[]{2, 512};
    const std::int64_t out_dims[]{2, 2, 512};
    const std::int64_t lse_dims[]{2, 2};
    const std::int64_t attn_sink_dims[]{2};
    const std::int64_t topk_indices_dims[]{2, 2};
    const std::int64_t topk_length_dims[]{2};
    const std::int64_t dsink_dims[]{2};
    FTrainSparseAttnBwdArgs args{};
    args.q             = makeView(storage, q_dims, 3, FTRAIN_NUMERIC_TYPE_BF16);
    args.kv            = makeView(storage, kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16);
    args.out           = makeView(storage, out_dims, 3, FTRAIN_NUMERIC_TYPE_BF16);
    args.dout          = args.out;
    args.lse           = makeView(storage, lse_dims, 2);
    args.attn_sink     = makeView(storage, attn_sink_dims, 1);
    args.topk_indices  = makeView(storage, topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32);
    args.topk_length   = makeView(storage, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32);
    args.dq            = args.q;
    args.dkv           = args.kv;
    args.dsink         = makeView(storage, dsink_dims, 1);
    args.softmax_scale = 0.125F;
    args.compute_type  = FTRAIN_NUMERIC_TYPE_BF16;

    FTrainSparseAttnBwdPlan plan = nullptr;
    EXPECT_EQ(ftrainSparseAttnBwdCreate(nullptr, &args), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainSparseAttnBwdCreate(&plan, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    // INVALID and COUNT fail the compute-type range check on different comparisons.
    for (const FTrainNumericType compute_type : {FTRAIN_NUMERIC_TYPE_INVALID, FTRAIN_NUMERIC_TYPE_COUNT}) {
        SCOPED_TRACE(static_cast<unsigned int>(compute_type));
        args.compute_type = compute_type;
        EXPECT_EQ(ftrainSparseAttnBwdCreate(&plan, &args), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    args.compute_type = FTRAIN_NUMERIC_TYPE_BF16;
    ASSERT_EQ(ftrainSparseAttnBwdCreate(&plan, &args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);

    std::uint64_t count   = 0;
    std::uint64_t bytes   = 0;
    const char* name      = nullptr;
    const auto outOfRange = std::numeric_limits<std::uint64_t>::max();
    EXPECT_EQ(ftrainSparseAttnBwdGetNumPrimitives(nullptr, &count), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainSparseAttnBwdGetNumPrimitives(plan, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainSparseAttnBwdGetPrimitiveName(nullptr, 0, &name), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainSparseAttnBwdGetPrimitiveName(plan, 0, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainSparseAttnBwdGetPrimitiveName(plan, outOfRange, &name), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes(nullptr, 0, &bytes),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes(plan, 0, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainSparseAttnBwdExecute(nullptr, 0, nullptr, 0, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainSparseAttnBwdDestroy(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
}

// Deferred workspace-policy/cached-schedule tests were removed with those
// optimized Primitives. Black-box tests retain default-path parameters,
// lifecycle, rebinding and error behavior. Query primitive 0 for the default;
// do not reintroduce fixed candidate-count assertions or pin its concrete name.
// Configuration-specific boundaries stay in white-box tests, while all
// applicable candidates are exercised by the Plan-based numerical suite.

}  // namespace
