// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Staged and convenience execution against a host reference over the same shape grid.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <flash_train/flash_train.h>
#include "../support/numerics.hpp"
#include "dense_attn_score_staged_fixture.hpp"

namespace {
using ftrain_test::DeviceBuffer;
using ftrain_test::expectNear;

struct ShapeParam {
    std::int64_t tokens;
    std::int64_t heads;
    std::int64_t head_dim;
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

void hostDenseAttnScore(const ShapeParam& shape, const std::vector<float>& q, const std::vector<float>& k,
                        const std::vector<float>& lse, std::vector<float>& out, std::vector<float>& denom,
                        std::int64_t ratio, float softmax_scale) {
    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        denom[row] = 0;
        for (std::int64_t col = 0; col < shape.tokens; ++col) {
            float score = 0;
            for (std::int64_t h = 0; h < shape.heads; ++h) {
                float dot = 0;
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += q[(row * shape.heads + h) * shape.head_dim + d] * k[col * shape.head_dim + d];
                score += std::exp(dot * softmax_scale - lse[row * shape.heads + h]);
            }
            out[row * shape.tokens + col] = col <= row ? score : -std::numeric_limits<float>::infinity();
            if (col <= row) denom[row] += score;
        }
    }
}

class DenseAttnScoreNumericsTest : public ftrain_test::DenseAttnScoreStagedTest,
                                   public testing::WithParamInterface<ShapeParam> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(DenseAttnScoreNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> q(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads) * (shape.head_dim)));
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = elementValue(i + 7919);
    std::vector<float> k(static_cast<std::size_t>((1) * (shape.tokens) * (1) * (shape.head_dim)));
    for (std::size_t i = 0; i < k.size(); ++i) k[i] = elementValue(i + 15838);
    std::vector<float> lse(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads)));
    for (std::size_t i = 0; i < lse.size(); ++i) lse[i] = elementValue(i + 23757);
    std::vector<float> out(static_cast<std::size_t>((1) * (shape.tokens) * (shape.tokens)));
    std::fill(out.begin(), out.end(), 123);
    std::vector<float> denom(static_cast<std::size_t>((1) * (shape.tokens)));
    std::fill(denom.begin(), denom.end(), 123);
    std::int64_t ratio  = 1;
    float softmax_scale = 0.125;

    auto expected_out   = out;
    auto expected_denom = denom;
    hostDenseAttnScore(shape, q, k, lse, expected_out, expected_denom, ratio, softmax_scale);

    const std::int64_t q_dims[]{1, shape.tokens, shape.heads, shape.head_dim};
    DeviceBuffer<float> device_q(q.size());
    ASSERT_NE(device_q.get(), nullptr);
    const std::int64_t k_dims[]{1, shape.tokens, 1, shape.head_dim};
    DeviceBuffer<float> device_k(k.size());
    ASSERT_NE(device_k.get(), nullptr);
    const std::int64_t lse_dims[]{1, shape.tokens, shape.heads};
    DeviceBuffer<float> device_lse(lse.size());
    ASSERT_NE(device_lse.get(), nullptr);
    const std::int64_t out_dims[]{1, shape.tokens, shape.tokens};
    DeviceBuffer<float> device_out(out.size());
    ASSERT_NE(device_out.get(), nullptr);
    const std::int64_t denom_dims[]{1, shape.tokens};
    DeviceBuffer<float> device_denom(denom.size());
    ASSERT_NE(device_denom.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_q.get(), q_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_k.get(), k_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_lse.get(), lse_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_out.get(), out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_denom.get(), denom_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    auto* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) ASSERT_NE(workspace, nullptr);

    for (bool convenience : {false, true}) {
        device_q.upload(q, stream_);
        device_k.upload(k, stream_);
        device_lse.upload(lse, stream_);
        device_out.upload(out, stream_);
        device_denom.upload(denom, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainDenseAttnScore(
                    makeView(device_q.get(), q_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_k.get(), k_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_lse.get(), lse_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_out.get(), out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_denom.get(), denom_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_out   = device_out.download(stream_);
        const auto actual_denom = device_denom.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_out.size(), expected_out.size());
        bool detects_zero_out = false;
        for (std::size_t i = 0; i < expected_out.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_out[i])) << "out reference " << i;
            if (std::isfinite(expected_out[i])) {
                ASSERT_TRUE(std::isfinite(actual_out[i])) << "out " << i;
                detects_zero_out |=
                    std::abs(expected_out[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_out[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_out[i], expected_out[i]) << "out " << i;
            }
        }
        EXPECT_TRUE(detects_zero_out) << "out reference cannot reject an all-zero result";
        expectNear(actual_out, expected_out, kRelTolerance, kAbsTolerance, "out");
        ASSERT_EQ(actual_denom.size(), expected_denom.size());
        bool detects_zero_denom = false;
        for (std::size_t i = 0; i < expected_denom.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_denom[i])) << "denom reference " << i;
            if (std::isfinite(expected_denom[i])) {
                ASSERT_TRUE(std::isfinite(actual_denom[i])) << "denom " << i;
                detects_zero_denom |=
                    std::abs(expected_denom[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_denom[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_denom[i], expected_denom[i]) << "denom " << i;
            }
        }
        EXPECT_TRUE(detects_zero_denom) << "denom reference cannot reject an all-zero result";
        expectNear(actual_denom, expected_denom, kRelTolerance, kAbsTolerance, "denom");
    }
}

INSTANTIATE_TEST_SUITE_P(DenseAttnScoreShapeGrid, DenseAttnScoreNumericsTest,
                         testing::Values(ShapeParam{8, 2, 32}, ShapeParam{33, 2, 64}, ShapeParam{128, 4, 128}),
                         [](const testing::TestParamInfo<DenseAttnScoreNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.tokens) + "x" + std::to_string(info.param.heads) + "x" +
                                    std::to_string(info.param.head_dim);
                         });
}  // namespace
