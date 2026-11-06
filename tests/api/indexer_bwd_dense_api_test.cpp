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

#include "indexer_bwd_dense_staged_fixture.hpp"

namespace {

using ftrain_test::IndexerBwdDenseStagedTest;

TEST_F(IndexerBwdDenseStagedTest, Bf16IndexerBwdDenseStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t index_q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> index_q(256, 0x3e80);
    auto* device_index_q = allocate<std::uint16_t>(256);
    ASSERT_NE(device_index_q, nullptr);
    const std::int64_t weights_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> weights(8, 0x3f80);
    auto* device_weights = allocate<std::uint16_t>(8);
    ASSERT_NE(device_weights, nullptr);
    const std::int64_t index_k_dims[]{1, 4, 32};
    const std::vector<std::uint16_t> index_k(128, 0x3f00);
    auto* device_index_k = allocate<std::uint16_t>(128);
    ASSERT_NE(device_index_k, nullptr);
    const std::int64_t attn_score_dims[]{1, 4, 4};
    const std::vector<float> attn_score{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                                        1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F};
    auto* device_attn_score = allocate<float>(16);
    ASSERT_NE(device_attn_score, nullptr);
    const std::int64_t attn_l1norm_dims[]{1, 4};
    const std::vector<float> attn_l1norm(4, 1.0F);
    auto* device_attn_l1norm = allocate<float>(4);
    ASSERT_NE(device_attn_l1norm, nullptr);
    const std::int64_t index_score_dims[]{1, 4, 4};
    const std::vector<float> index_score(16, 2.0F);
    auto* device_index_score = allocate<float>(16);
    ASSERT_NE(device_index_score, nullptr);
    const std::int64_t index_lse_dims[]{1, 4};
    const std::vector<float> index_lse{2.0F, 2.6931471805599454F, 3.09861228866811F, 3.386294361119891F};
    auto* device_index_lse = allocate<float>(4);
    ASSERT_NE(device_index_lse, nullptr);
    const std::int64_t grad_index_score_dims[]{1, 4, 4};

    auto* device_grad_index_score = device_index_score;  // In-place gradient output.
    ASSERT_NE(device_grad_index_score, nullptr);
    const std::int64_t dq_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> dq(256, 0x42f6);
    auto* device_dq = allocate<std::uint16_t>(256);
    ASSERT_NE(device_dq, nullptr);
    const std::int64_t dw_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> dw(8, 0x42f6);
    auto* device_dw = allocate<std::uint16_t>(8);
    ASSERT_NE(device_dw, nullptr);
    const std::int64_t dk_dims[]{1, 4, 32};
    const std::vector<std::uint16_t> dk(128, 0x42f6);
    auto* device_dk = allocate<std::uint16_t>(128);
    ASSERT_NE(device_dk, nullptr);
    float sm_scale     = 0.25F;
    float loss_coeff   = 1.0F;
    float grad_loss    = 1.0F;
    std::int64_t ratio = 1;
    ASSERT_EQ(hipMemcpyAsync(device_index_q, index_q.data(), index_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_weights, weights.data(), weights.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_index_k, index_k.data(), index_k.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_score, attn_score.data(), attn_score.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_l1norm, attn_l1norm.data(), attn_l1norm.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_index_lse, index_lse.data(), index_lse.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto index_q_view =
        makeView(device_index_q, index_q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weights_view =
        makeView(device_weights, weights_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_k_view =
        makeView(device_index_k, index_k_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_score_view =
        makeView(device_attn_score, attn_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_l1norm_view = makeView(device_attn_l1norm, attn_l1norm_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_score_view = makeView(device_index_score, index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_lse_view =
        makeView(device_index_lse, index_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_index_score_view = makeView(device_grad_index_score, grad_index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dq_view       = makeView(device_dq, dq_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dw_view       = makeView(device_dw, dw_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dk_view       = makeView(device_dk, dk_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto loss_coeff_view =
        makeView(&loss_coeff, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto grad_loss_view =
        makeView(&grad_loss, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto ratio_view = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, index_q_view, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                        index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                        sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Bf16IndexerBwdDense");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_index_score, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 0.0F);
            expected[4] = -0.125F;
            expected[5] = 0.125F;
            expected[8] = -0.16666666666666669F;
            std::fill(expected.begin() + 9, expected.begin() + 11, 0.08333333333333333F);
            expected[12] = -0.1875F;
            std::fill(expected.begin() + 13, expected.begin() + 16, 0.0625F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        {
            std::vector<std::uint16_t> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(std::uint16_t),
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
            std::vector<std::uint16_t> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dw, actual.size() * sizeof(std::uint16_t),
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
            std::vector<std::uint16_t> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dk, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(128, -0.059895833333333336F);
            std::fill(expected.begin() + 32, expected.begin() + 64, 0.033854166666666664F);
            std::fill(expected.begin() + 64, expected.begin() + 96, 0.018229166666666664F);
            std::fill(expected.begin() + 96, expected.begin() + 128, 0.0078125F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual_value, expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual_value, expected[i], 2e-3 + 1e-2 * std::abs(expected[i])) << i;
                }
            }
        }
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainIndexerBwdDense(index_q_view, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                        index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view,
                                        dk_view, sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                        FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_index_score, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 0.0F);
            expected[4] = -0.125F;
            expected[5] = 0.125F;
            expected[8] = -0.16666666666666669F;
            std::fill(expected.begin() + 9, expected.begin() + 11, 0.08333333333333333F);
            expected[12] = -0.1875F;
            std::fill(expected.begin() + 13, expected.begin() + 16, 0.0625F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        {
            std::vector<std::uint16_t> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(std::uint16_t),
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
            std::vector<std::uint16_t> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dw, actual.size() * sizeof(std::uint16_t),
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
            std::vector<std::uint16_t> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dk, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(128, -0.059895833333333336F);
            std::fill(expected.begin() + 32, expected.begin() + 64, 0.033854166666666664F);
            std::fill(expected.begin() + 64, expected.begin() + 96, 0.018229166666666664F);
            std::fill(expected.begin() + 96, expected.begin() + 128, 0.0078125F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual_value, expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual_value, expected[i], 2e-3 + 1e-2 * std::abs(expected[i])) << i;
                }
            }
        }
    }
    auto invalid_index_q           = index_q_view;
    invalid_index_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerBwdDense(invalid_index_q, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                    index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                                    sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                    FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_index_q              = index_q_view;
    invalid_index_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerBwdDense(invalid_index_q, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                    index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                                    sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                    FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerBwdDenseStagedTest, Fp16IndexerBwdDenseStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t index_q_dims[]{1, 4, 2, 32};
    const std::vector<__half> index_q(256, __float2half(0.25F));
    auto* device_index_q = allocate<__half>(256);
    ASSERT_NE(device_index_q, nullptr);
    const std::int64_t weights_dims[]{1, 4, 2};
    const std::vector<__half> weights(8, __float2half(1.0F));
    auto* device_weights = allocate<__half>(8);
    ASSERT_NE(device_weights, nullptr);
    const std::int64_t index_k_dims[]{1, 4, 32};
    const std::vector<__half> index_k(128, __float2half(0.5F));
    auto* device_index_k = allocate<__half>(128);
    ASSERT_NE(device_index_k, nullptr);
    const std::int64_t attn_score_dims[]{1, 4, 4};
    const std::vector<float> attn_score{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                                        1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F};
    auto* device_attn_score = allocate<float>(16);
    ASSERT_NE(device_attn_score, nullptr);
    const std::int64_t attn_l1norm_dims[]{1, 4};
    const std::vector<float> attn_l1norm(4, 1.0F);
    auto* device_attn_l1norm = allocate<float>(4);
    ASSERT_NE(device_attn_l1norm, nullptr);
    const std::int64_t index_score_dims[]{1, 4, 4};
    const std::vector<float> index_score(16, 2.0F);
    auto* device_index_score = allocate<float>(16);
    ASSERT_NE(device_index_score, nullptr);
    const std::int64_t index_lse_dims[]{1, 4};
    const std::vector<float> index_lse{2.0F, 2.6931471805599454F, 3.09861228866811F, 3.386294361119891F};
    auto* device_index_lse = allocate<float>(4);
    ASSERT_NE(device_index_lse, nullptr);
    const std::int64_t grad_index_score_dims[]{1, 4, 4};

    auto* device_grad_index_score = device_index_score;  // In-place gradient output.
    ASSERT_NE(device_grad_index_score, nullptr);
    const std::int64_t dq_dims[]{1, 4, 2, 32};
    const std::vector<__half> dq(256, __float2half(123.0F));
    auto* device_dq = allocate<__half>(256);
    ASSERT_NE(device_dq, nullptr);
    const std::int64_t dw_dims[]{1, 4, 2};
    const std::vector<__half> dw(8, __float2half(123.0F));
    auto* device_dw = allocate<__half>(8);
    ASSERT_NE(device_dw, nullptr);
    const std::int64_t dk_dims[]{1, 4, 32};
    const std::vector<__half> dk(128, __float2half(123.0F));
    auto* device_dk = allocate<__half>(128);
    ASSERT_NE(device_dk, nullptr);
    float sm_scale     = 0.25F;
    float loss_coeff   = 1.0F;
    float grad_loss    = 1.0F;
    std::int64_t ratio = 1;
    ASSERT_EQ(
        hipMemcpyAsync(device_index_q, index_q.data(), index_q.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weights, weights.data(), weights.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_index_k, index_k.data(), index_k.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_score, attn_score.data(), attn_score.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_l1norm, attn_l1norm.data(), attn_l1norm.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_index_lse, index_lse.data(), index_lse.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto index_q_view =
        makeView(device_index_q, index_q_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weights_view =
        makeView(device_weights, weights_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_k_view =
        makeView(device_index_k, index_k_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_score_view =
        makeView(device_attn_score, attn_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_l1norm_view = makeView(device_attn_l1norm, attn_l1norm_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_score_view = makeView(device_index_score, index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_lse_view =
        makeView(device_index_lse, index_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_index_score_view = makeView(device_grad_index_score, grad_index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dq_view       = makeView(device_dq, dq_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dw_view       = makeView(device_dw, dw_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dk_view       = makeView(device_dk, dk_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto loss_coeff_view =
        makeView(&loss_coeff, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto grad_loss_view =
        makeView(&grad_loss, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto ratio_view = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, index_q_view, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                        index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                        sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view, FTRAIN_NUMERIC_TYPE_FP16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp16IndexerBwdDense");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_index_score, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 0.0F);
            expected[4] = -0.125F;
            expected[5] = 0.125F;
            expected[8] = -0.16666666666666669F;
            std::fill(expected.begin() + 9, expected.begin() + 11, 0.08333333333333333F);
            expected[12] = -0.1875F;
            std::fill(expected.begin() + 13, expected.begin() + 16, 0.0625F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        {
            std::vector<__half> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<__half> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dw, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<__half> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dk, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(128, -0.059895833333333336F);
            std::fill(expected.begin() + 32, expected.begin() + 64, 0.033854166666666664F);
            std::fill(expected.begin() + 64, expected.begin() + 96, 0.018229166666666664F);
            std::fill(expected.begin() + 96, expected.begin() + 128, 0.0078125F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(__half2float(actual[i]), expected[i]) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), expected[i], 2e-4 + 2e-3 * std::abs(expected[i])) << i;
                }
            }
        }
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainIndexerBwdDense(index_q_view, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                        index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view,
                                        dk_view, sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                        FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_index_score, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 0.0F);
            expected[4] = -0.125F;
            expected[5] = 0.125F;
            expected[8] = -0.16666666666666669F;
            std::fill(expected.begin() + 9, expected.begin() + 11, 0.08333333333333333F);
            expected[12] = -0.1875F;
            std::fill(expected.begin() + 13, expected.begin() + 16, 0.0625F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        {
            std::vector<__half> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<__half> actual(8);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dw, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(__half2float(actual[i]), 0.0F, 2e-4) << i; }
        }
        {
            std::vector<__half> actual(128);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dk, actual.size() * sizeof(__half), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(128, -0.059895833333333336F);
            std::fill(expected.begin() + 32, expected.begin() + 64, 0.033854166666666664F);
            std::fill(expected.begin() + 64, expected.begin() + 96, 0.018229166666666664F);
            std::fill(expected.begin() + 96, expected.begin() + 128, 0.0078125F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(__half2float(actual[i]), expected[i]) << i;
                } else {
                    EXPECT_NEAR(__half2float(actual[i]), expected[i], 2e-4 + 2e-3 * std::abs(expected[i])) << i;
                }
            }
        }
    }
    auto invalid_index_q           = index_q_view;
    invalid_index_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerBwdDense(invalid_index_q, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                    index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                                    sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                    FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_index_q              = index_q_view;
    invalid_index_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerBwdDense(invalid_index_q, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                    index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                                    sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                    FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerBwdDenseStagedTest, Fp32IndexerBwdDenseStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t index_q_dims[]{1, 4, 2, 32};
    const std::vector<float> index_q(256, 0.25F);
    auto* device_index_q = allocate<float>(256);
    ASSERT_NE(device_index_q, nullptr);
    const std::int64_t weights_dims[]{1, 4, 2};
    const std::vector<float> weights(8, 1.0F);
    auto* device_weights = allocate<float>(8);
    ASSERT_NE(device_weights, nullptr);
    const std::int64_t index_k_dims[]{1, 4, 32};
    const std::vector<float> index_k(128, 0.5F);
    auto* device_index_k = allocate<float>(128);
    ASSERT_NE(device_index_k, nullptr);
    const std::int64_t attn_score_dims[]{1, 4, 4};
    const std::vector<float> attn_score{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                                        1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F};
    auto* device_attn_score = allocate<float>(16);
    ASSERT_NE(device_attn_score, nullptr);
    const std::int64_t attn_l1norm_dims[]{1, 4};
    const std::vector<float> attn_l1norm(4, 1.0F);
    auto* device_attn_l1norm = allocate<float>(4);
    ASSERT_NE(device_attn_l1norm, nullptr);
    const std::int64_t index_score_dims[]{1, 4, 4};
    const std::vector<float> index_score(16, 2.0F);
    auto* device_index_score = allocate<float>(16);
    ASSERT_NE(device_index_score, nullptr);
    const std::int64_t index_lse_dims[]{1, 4};
    const std::vector<float> index_lse{2.0F, 2.6931471805599454F, 3.09861228866811F, 3.386294361119891F};
    auto* device_index_lse = allocate<float>(4);
    ASSERT_NE(device_index_lse, nullptr);
    const std::int64_t grad_index_score_dims[]{1, 4, 4};

    auto* device_grad_index_score = device_index_score;  // In-place gradient output.
    ASSERT_NE(device_grad_index_score, nullptr);
    const std::int64_t dq_dims[]{1, 4, 2, 32};
    const std::vector<float> dq(256, 123.0F);
    auto* device_dq = allocate<float>(256);
    ASSERT_NE(device_dq, nullptr);
    const std::int64_t dw_dims[]{1, 4, 2};
    const std::vector<float> dw(8, 123.0F);
    auto* device_dw = allocate<float>(8);
    ASSERT_NE(device_dw, nullptr);
    const std::int64_t dk_dims[]{1, 4, 32};
    const std::vector<float> dk(128, 123.0F);
    auto* device_dk = allocate<float>(128);
    ASSERT_NE(device_dk, nullptr);
    float sm_scale     = 0.25F;
    float loss_coeff   = 1.0F;
    float grad_loss    = 1.0F;
    std::int64_t ratio = 1;
    ASSERT_EQ(
        hipMemcpyAsync(device_index_q, index_q.data(), index_q.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weights, weights.data(), weights.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_index_k, index_k.data(), index_k.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_score, attn_score.data(), attn_score.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_l1norm, attn_l1norm.data(), attn_l1norm.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_index_lse, index_lse.data(), index_lse.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto index_q_view =
        makeView(device_index_q, index_q_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weights_view =
        makeView(device_weights, weights_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_k_view =
        makeView(device_index_k, index_k_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_score_view =
        makeView(device_attn_score, attn_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_l1norm_view = makeView(device_attn_l1norm, attn_l1norm_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_score_view = makeView(device_index_score, index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_lse_view =
        makeView(device_index_lse, index_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_index_score_view = makeView(device_grad_index_score, grad_index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dq_view       = makeView(device_dq, dq_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dw_view       = makeView(device_dw, dw_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dk_view       = makeView(device_dk, dk_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto loss_coeff_view =
        makeView(&loss_coeff, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto grad_loss_view =
        makeView(&grad_loss, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto ratio_view = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, index_q_view, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                        index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                        sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp32IndexerBwdDense");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_index_score, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 0.0F);
            expected[4] = -0.125F;
            expected[5] = 0.125F;
            expected[8] = -0.16666666666666669F;
            std::fill(expected.begin() + 9, expected.begin() + 11, 0.08333333333333333F);
            expected[12] = -0.1875F;
            std::fill(expected.begin() + 13, expected.begin() + 16, 0.0625F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        {
            std::vector<float> actual(256);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_dw, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(128);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_dk, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(128, -0.059895833333333336F);
            std::fill(expected.begin() + 32, expected.begin() + 64, 0.033854166666666664F);
            std::fill(expected.begin() + 64, expected.begin() + 96, 0.018229166666666664F);
            std::fill(expected.begin() + 96, expected.begin() + 128, 0.0078125F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainIndexerBwdDense(index_q_view, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                        index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view,
                                        dk_view, sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                        FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_index_score, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 0.0F);
            expected[4] = -0.125F;
            expected[5] = 0.125F;
            expected[8] = -0.16666666666666669F;
            std::fill(expected.begin() + 9, expected.begin() + 11, 0.08333333333333333F);
            expected[12] = -0.1875F;
            std::fill(expected.begin() + 13, expected.begin() + 16, 0.0625F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        {
            std::vector<float> actual(256);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(8);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_dw, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(128);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_dk, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(128, -0.059895833333333336F);
            std::fill(expected.begin() + 32, expected.begin() + 64, 0.033854166666666664F);
            std::fill(expected.begin() + 64, expected.begin() + 96, 0.018229166666666664F);
            std::fill(expected.begin() + 96, expected.begin() + 128, 0.0078125F);
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
    auto invalid_index_q           = index_q_view;
    invalid_index_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerBwdDense(invalid_index_q, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                    index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                                    sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_index_q              = index_q_view;
    invalid_index_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerBwdDense(invalid_index_q, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                    index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                                    sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerBwdDenseStagedTest, MixedFp32IndexerBwdDenseStorageMode3RunsConvenienceAndStagedApis) {
    const std::int64_t index_q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> index_q(256, 0x3e80);
    auto* device_index_q = allocate<std::uint16_t>(256);
    ASSERT_NE(device_index_q, nullptr);
    const std::int64_t weights_dims[]{1, 4, 2};
    const std::vector<float> weights(8, 1.0F);
    auto* device_weights = allocate<float>(8);
    ASSERT_NE(device_weights, nullptr);
    const std::int64_t index_k_dims[]{1, 4, 32};
    const std::vector<float> index_k(128, 0.5F);
    auto* device_index_k = allocate<float>(128);
    ASSERT_NE(device_index_k, nullptr);
    const std::int64_t attn_score_dims[]{1, 4, 4};
    const std::vector<float> attn_score{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                                        1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F};
    auto* device_attn_score = allocate<float>(16);
    ASSERT_NE(device_attn_score, nullptr);
    const std::int64_t attn_l1norm_dims[]{1, 4};
    const std::vector<float> attn_l1norm(4, 1.0F);
    auto* device_attn_l1norm = allocate<float>(4);
    ASSERT_NE(device_attn_l1norm, nullptr);
    const std::int64_t index_score_dims[]{1, 4, 4};
    const std::vector<float> index_score(16, 2.0F);
    auto* device_index_score = allocate<float>(16);
    ASSERT_NE(device_index_score, nullptr);
    const std::int64_t index_lse_dims[]{1, 4};
    const std::vector<float> index_lse{2.0F, 2.6931471805599454F, 3.09861228866811F, 3.386294361119891F};
    auto* device_index_lse = allocate<float>(4);
    ASSERT_NE(device_index_lse, nullptr);
    const std::int64_t grad_index_score_dims[]{1, 4, 4};

    auto* device_grad_index_score = device_index_score;  // In-place gradient output.
    ASSERT_NE(device_grad_index_score, nullptr);
    const std::int64_t dq_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> dq(256, 0x42f6);
    auto* device_dq = allocate<std::uint16_t>(256);
    ASSERT_NE(device_dq, nullptr);
    const std::int64_t dw_dims[]{1, 4, 2};
    const std::vector<float> dw(8, 123.0F);
    auto* device_dw = allocate<float>(8);
    ASSERT_NE(device_dw, nullptr);
    const std::int64_t dk_dims[]{1, 4, 32};
    const std::vector<float> dk(128, 123.0F);
    auto* device_dk = allocate<float>(128);
    ASSERT_NE(device_dk, nullptr);
    float sm_scale     = 0.25F;
    float loss_coeff   = 1.0F;
    float grad_loss    = 1.0F;
    std::int64_t ratio = 1;
    ASSERT_EQ(hipMemcpyAsync(device_index_q, index_q.data(), index_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_weights, weights.data(), weights.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_index_k, index_k.data(), index_k.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_score, attn_score.data(), attn_score.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_attn_l1norm, attn_l1norm.data(), attn_l1norm.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_index_lse, index_lse.data(), index_lse.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto index_q_view =
        makeView(device_index_q, index_q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto weights_view =
        makeView(device_weights, weights_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_k_view =
        makeView(device_index_k, index_k_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_score_view =
        makeView(device_attn_score, attn_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto attn_l1norm_view = makeView(device_attn_l1norm, attn_l1norm_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_score_view = makeView(device_index_score, index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto index_lse_view =
        makeView(device_index_lse, index_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto grad_index_score_view = makeView(device_grad_index_score, grad_index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dq_view       = makeView(device_dq, dq_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dw_view       = makeView(device_dw, dw_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto dk_view       = makeView(device_dk, dk_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto loss_coeff_view =
        makeView(&loss_coeff, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto grad_loss_view =
        makeView(&grad_loss, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto ratio_view = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, index_q_view, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                        index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                        sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32IndexerBwdDense");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_index_score, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 0.0F);
            expected[4] = -0.125F;
            expected[5] = 0.125F;
            expected[8] = -0.16666666666666669F;
            std::fill(expected.begin() + 9, expected.begin() + 11, 0.08333333333333333F);
            expected[12] = -0.1875F;
            std::fill(expected.begin() + 13, expected.begin() + 16, 0.0625F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        {
            std::vector<std::uint16_t> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(std::uint16_t),
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
                hipMemcpyAsync(actual.data(), device_dw, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(128);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_dk, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(128, -0.059895833333333336F);
            std::fill(expected.begin() + 32, expected.begin() + 64, 0.033854166666666664F);
            std::fill(expected.begin() + 64, expected.begin() + 96, 0.018229166666666664F);
            std::fill(expected.begin() + 96, expected.begin() + 128, 0.0078125F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_index_score, index_score.data(), index_score.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(device_dq, dq.data(), dq.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dw, dw.data(), dw.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(device_dk, dk.data(), dk.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainIndexerBwdDense(index_q_view, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                        index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view,
                                        dk_view, sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                        FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_grad_index_score, actual.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 0.0F);
            expected[4] = -0.125F;
            expected[5] = 0.125F;
            expected[8] = -0.16666666666666669F;
            std::fill(expected.begin() + 9, expected.begin() + 11, 0.08333333333333333F);
            expected[12] = -0.1875F;
            std::fill(expected.begin() + 13, expected.begin() + 16, 0.0625F);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        {
            std::vector<std::uint16_t> actual(256);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_dq, actual.size() * sizeof(std::uint16_t),
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
                hipMemcpyAsync(actual.data(), device_dw, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], 0.0F, 2e-5) << i; }
        }
        {
            std::vector<float> actual(128);
            ASSERT_EQ(
                hipMemcpyAsync(actual.data(), device_dk, actual.size() * sizeof(float), hipMemcpyDeviceToHost, stream_),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(128, -0.059895833333333336F);
            std::fill(expected.begin() + 32, expected.begin() + 64, 0.033854166666666664F);
            std::fill(expected.begin() + 64, expected.begin() + 96, 0.018229166666666664F);
            std::fill(expected.begin() + 96, expected.begin() + 128, 0.0078125F);
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
    auto invalid_index_q           = index_q_view;
    invalid_index_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerBwdDense(invalid_index_q, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                    index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                                    sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_index_q              = index_q_view;
    invalid_index_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerBwdDense(invalid_index_q, weights_view, index_k_view, attn_score_view, attn_l1norm_view,
                                    index_score_view, index_lse_view, grad_index_score_view, dq_view, dw_view, dk_view,
                                    sm_scale_view, loss_coeff_view, grad_loss_view, ratio_view,
                                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerBwdDenseStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_index_q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> first_index_q(256, 0x3e80);
    auto* first_device_index_q = allocate<std::uint16_t>(256);
    ASSERT_NE(first_device_index_q, nullptr);
    const std::int64_t first_weights_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> first_weights(8, 0x3f80);
    auto* first_device_weights = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_weights, nullptr);
    const std::int64_t first_index_k_dims[]{1, 4, 32};
    const std::vector<std::uint16_t> first_index_k(128, 0x3f00);
    auto* first_device_index_k = allocate<std::uint16_t>(128);
    ASSERT_NE(first_device_index_k, nullptr);
    const std::int64_t first_attn_score_dims[]{1, 4, 4};
    const std::vector<float> first_attn_score{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                                              1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F};
    auto* first_device_attn_score = allocate<float>(16);
    ASSERT_NE(first_device_attn_score, nullptr);
    const std::int64_t first_attn_l1norm_dims[]{1, 4};
    const std::vector<float> first_attn_l1norm(4, 1.0F);
    auto* first_device_attn_l1norm = allocate<float>(4);
    ASSERT_NE(first_device_attn_l1norm, nullptr);
    const std::int64_t first_index_score_dims[]{1, 4, 4};
    const std::vector<float> first_index_score(16, 2.0F);
    auto* first_device_index_score = allocate<float>(16);
    ASSERT_NE(first_device_index_score, nullptr);
    const std::int64_t first_index_lse_dims[]{1, 4};
    const std::vector<float> first_index_lse{2.0F, 2.6931471805599454F, 3.09861228866811F, 3.386294361119891F};
    auto* first_device_index_lse = allocate<float>(4);
    ASSERT_NE(first_device_index_lse, nullptr);
    const std::int64_t first_grad_index_score_dims[]{1, 4, 4};

    auto* first_device_grad_index_score = first_device_index_score;  // In-place gradient output.
    ASSERT_NE(first_device_grad_index_score, nullptr);
    const std::int64_t first_dq_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> first_dq(256, 0x42f6);
    auto* first_device_dq = allocate<std::uint16_t>(256);
    ASSERT_NE(first_device_dq, nullptr);
    const std::int64_t first_dw_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> first_dw(8, 0x42f6);
    auto* first_device_dw = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_dw, nullptr);
    const std::int64_t first_dk_dims[]{1, 4, 32};
    const std::vector<std::uint16_t> first_dk(128, 0x42f6);
    auto* first_device_dk = allocate<std::uint16_t>(128);
    ASSERT_NE(first_device_dk, nullptr);
    float first_sm_scale     = 0.25F;
    float first_loss_coeff   = 1.0F;
    float first_grad_loss    = 1.0F;
    std::int64_t first_ratio = 1;
    ASSERT_EQ(hipMemcpyAsync(first_device_index_q, first_index_q.data(), first_index_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_weights, first_weights.data(), first_weights.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_index_k, first_index_k.data(), first_index_k.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_attn_score, first_attn_score.data(), first_attn_score.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_attn_l1norm, first_attn_l1norm.data(),
                             first_attn_l1norm.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_index_score, first_index_score.data(),
                             first_index_score.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_index_lse, first_index_lse.data(), first_index_lse.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_dq, first_dq.data(), first_dq.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_dw, first_dw.data(), first_dw.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_dk, first_dk.data(), first_dk.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_index_q_view     = makeView(first_device_index_q, first_index_q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_weights_view     = makeView(first_device_weights, first_weights_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_index_k_view     = makeView(first_device_index_k, first_index_k_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_attn_score_view  = makeView(first_device_attn_score, first_attn_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_attn_l1norm_view = makeView(first_device_attn_l1norm, first_attn_l1norm_dims, 2,
                                           FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_index_score_view = makeView(first_device_index_score, first_index_score_dims, 3,
                                           FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_index_lse_view   = makeView(first_device_index_lse, first_index_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                           FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_grad_index_score_view = makeView(first_device_grad_index_score, first_grad_index_score_dims, 3,
                                                FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_dq_view =
        makeView(first_device_dq, first_dq_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_dw_view =
        makeView(first_device_dw, first_dw_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_dk_view =
        makeView(first_device_dk, first_dk_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_sm_scale_view =
        makeView(&first_sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_loss_coeff_view =
        makeView(&first_loss_coeff, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_grad_loss_view =
        makeView(&first_grad_loss, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_ratio_view =
        makeView(&first_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_index_q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> second_index_q(256, 0x3e80);
    auto* second_device_index_q = allocate<std::uint16_t>(256);
    ASSERT_NE(second_device_index_q, nullptr);
    const std::int64_t second_weights_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> second_weights(8, 0x3f80);
    auto* second_device_weights = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_weights, nullptr);
    const std::int64_t second_index_k_dims[]{1, 4, 32};
    const std::vector<std::uint16_t> second_index_k(128, 0x3f00);
    auto* second_device_index_k = allocate<std::uint16_t>(128);
    ASSERT_NE(second_device_index_k, nullptr);
    const std::int64_t second_attn_score_dims[]{1, 4, 4};
    const std::vector<float> second_attn_score{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                                               1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F};
    auto* second_device_attn_score = allocate<float>(16);
    ASSERT_NE(second_device_attn_score, nullptr);
    const std::int64_t second_attn_l1norm_dims[]{1, 4};
    const std::vector<float> second_attn_l1norm(4, 1.0F);
    auto* second_device_attn_l1norm = allocate<float>(4);
    ASSERT_NE(second_device_attn_l1norm, nullptr);
    const std::int64_t second_index_score_dims[]{1, 4, 4};
    const std::vector<float> second_index_score(16, 2.0F);
    auto* second_device_index_score = allocate<float>(16);
    ASSERT_NE(second_device_index_score, nullptr);
    const std::int64_t second_index_lse_dims[]{1, 4};
    const std::vector<float> second_index_lse{2.0F, 2.6931471805599454F, 3.09861228866811F, 3.386294361119891F};
    auto* second_device_index_lse = allocate<float>(4);
    ASSERT_NE(second_device_index_lse, nullptr);
    const std::int64_t second_grad_index_score_dims[]{1, 4, 4};

    auto* second_device_grad_index_score = second_device_index_score;  // In-place gradient output.
    ASSERT_NE(second_device_grad_index_score, nullptr);
    const std::int64_t second_dq_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> second_dq(256, 0x42f6);
    auto* second_device_dq = allocate<std::uint16_t>(256);
    ASSERT_NE(second_device_dq, nullptr);
    const std::int64_t second_dw_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> second_dw(8, 0x42f6);
    auto* second_device_dw = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_dw, nullptr);
    const std::int64_t second_dk_dims[]{1, 4, 32};
    const std::vector<std::uint16_t> second_dk(128, 0x42f6);
    auto* second_device_dk = allocate<std::uint16_t>(128);
    ASSERT_NE(second_device_dk, nullptr);
    float second_sm_scale     = 0.25F;
    float second_loss_coeff   = 1.0F;
    float second_grad_loss    = 1.0F;
    std::int64_t second_ratio = 1;
    ASSERT_EQ(hipMemcpyAsync(second_device_index_q, second_index_q.data(),
                             second_index_q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_weights, second_weights.data(),
                             second_weights.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_index_k, second_index_k.data(),
                             second_index_k.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_attn_score, second_attn_score.data(),
                             second_attn_score.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_attn_l1norm, second_attn_l1norm.data(),
                             second_attn_l1norm.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_index_score, second_index_score.data(),
                             second_index_score.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_index_lse, second_index_lse.data(), second_index_lse.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_dq, second_dq.data(), second_dq.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_dw, second_dw.data(), second_dw.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_dk, second_dk.data(), second_dk.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_index_q_view     = makeView(second_device_index_q, second_index_q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_weights_view     = makeView(second_device_weights, second_weights_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_index_k_view     = makeView(second_device_index_k, second_index_k_dims, 3, FTRAIN_NUMERIC_TYPE_BF16,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_attn_score_view  = makeView(second_device_attn_score, second_attn_score_dims, 3,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_attn_l1norm_view = makeView(second_device_attn_l1norm, second_attn_l1norm_dims, 2,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_index_score_view = makeView(second_device_index_score, second_index_score_dims, 3,
                                            FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_index_lse_view   = makeView(second_device_index_lse, second_index_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_grad_index_score_view = makeView(second_device_grad_index_score, second_grad_index_score_dims, 3,
                                                 FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_dq_view =
        makeView(second_device_dq, second_dq_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_dw_view =
        makeView(second_device_dw, second_dw_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_dk_view =
        makeView(second_device_dk, second_dk_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_sm_scale_view =
        makeView(&second_sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_loss_coeff_view =
        makeView(&second_loss_coeff, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_grad_loss_view =
        makeView(&second_grad_loss, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_ratio_view =
        makeView(&second_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_index_q_view, first_weights_view, first_index_k_view, first_attn_score_view,
                        first_attn_l1norm_view, first_index_score_view, first_index_lse_view,
                        first_grad_index_score_view, first_dq_view, first_dw_view, first_dk_view, first_sm_scale_view,
                        first_loss_coeff_view, first_grad_loss_view, first_ratio_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_index_q_view, second_weights_view, second_index_k_view, second_attn_score_view,
                        second_attn_l1norm_view, second_index_score_view, second_index_lse_view,
                        second_grad_index_score_view, second_dq_view, second_dw_view, second_dk_view,
                        second_sm_scale_view, second_loss_coeff_view, second_grad_loss_view, second_ratio_view,
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
    ASSERT_EQ(hipMemcpyAsync(first_device_index_score, first_index_score.data(),
                             first_index_score.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_grad_index_score, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(16, 0.0F);
        expected[4] = -0.125F;
        expected[5] = 0.125F;
        expected[8] = -0.16666666666666669F;
        std::fill(expected.begin() + 9, expected.begin() + 11, 0.08333333333333333F);
        expected[12] = -0.1875F;
        std::fill(expected.begin() + 13, expected.begin() + 16, 0.0625F);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) continue;
            if (std::isinf(expected[i])) {
                EXPECT_EQ(actual[i], expected[i]) << i;
            } else {
                EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
            }
        }
    }
    {
        std::vector<std::uint16_t> actual(256);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_dq, actual.size() * sizeof(std::uint16_t),
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
        std::vector<std::uint16_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_dw, actual.size() * sizeof(std::uint16_t),
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
        std::vector<std::uint16_t> actual(128);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_dk, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(128, -0.059895833333333336F);
        std::fill(expected.begin() + 32, expected.begin() + 64, 0.033854166666666664F);
        std::fill(expected.begin() + 64, expected.begin() + 96, 0.018229166666666664F);
        std::fill(expected.begin() + 96, expected.begin() + 128, 0.0078125F);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            if (std::isnan(expected[i])) continue;
            if (std::isinf(expected[i])) {
                EXPECT_EQ(actual_value, expected[i]) << i;
            } else {
                EXPECT_NEAR(actual_value, expected[i], 2e-3 + 1e-2 * std::abs(expected[i])) << i;
            }
        }
    }
    ASSERT_EQ(hipMemcpyAsync(second_device_index_score, second_index_score.data(),
                             second_index_score.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_grad_index_score, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(16, 0.0F);
        expected[4] = -0.125F;
        expected[5] = 0.125F;
        expected[8] = -0.16666666666666669F;
        std::fill(expected.begin() + 9, expected.begin() + 11, 0.08333333333333333F);
        expected[12] = -0.1875F;
        std::fill(expected.begin() + 13, expected.begin() + 16, 0.0625F);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) continue;
            if (std::isinf(expected[i])) {
                EXPECT_EQ(actual[i], expected[i]) << i;
            } else {
                EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
            }
        }
    }
    {
        std::vector<std::uint16_t> actual(256);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_dq, actual.size() * sizeof(std::uint16_t),
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
        std::vector<std::uint16_t> actual(8);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_dw, actual.size() * sizeof(std::uint16_t),
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
        std::vector<std::uint16_t> actual(128);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_dk, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(128, -0.059895833333333336F);
        std::fill(expected.begin() + 32, expected.begin() + 64, 0.033854166666666664F);
        std::fill(expected.begin() + 64, expected.begin() + 96, 0.018229166666666664F);
        std::fill(expected.begin() + 96, expected.begin() + 128, 0.0078125F);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            if (std::isnan(expected[i])) continue;
            if (std::isinf(expected[i])) {
                EXPECT_EQ(actual_value, expected[i]) << i;
            } else {
                EXPECT_NEAR(actual_value, expected[i], 2e-3 + 1e-2 * std::abs(expected[i])) << i;
            }
        }
    }
}

}  // namespace
