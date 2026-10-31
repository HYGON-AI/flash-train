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
#include "sparse_indexer_predict_staged_fixture.hpp"

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

void hostSparseIndexerPredict(const ShapeParam& shape, const std::vector<float>& q, const std::vector<float>& k,
                              const std::vector<float>& w, const std::vector<std::int32_t>& topk_indices,
                              std::vector<float>& predict, float sm_scale, std::uint8_t topk_indices_global) {
    const auto topk = std::min<std::int64_t>(shape.tokens, 32);
    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::int64_t j = 0; j < topk; ++j) {
            float value    = 0;
            const auto col = topk_indices[row * topk + j];
            for (std::int64_t h = 0; h < shape.heads; ++h) {
                float dot = 0;
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += q[(row * shape.heads + h) * shape.head_dim + d] * k[col * shape.head_dim + d];
                value += std::max(dot * sm_scale, 0.0F) * w[row * shape.heads + h];
            }
            predict[row * topk + j] = value;
            maximum                 = std::max(maximum, value);
        }
        float total = 0;
        for (std::int64_t j = 0; j < topk; ++j) total += std::exp(predict[row * topk + j] - maximum);
        for (std::int64_t j = 0; j < topk; ++j)
            predict[row * topk + j] = std::exp(predict[row * topk + j] - maximum) / total;
    }
}

class SparseIndexerPredictNumericsTest : public ftrain_test::SparseIndexerPredictStagedTest,
                                         public testing::WithParamInterface<ShapeParam> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(SparseIndexerPredictNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> q(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads) * (shape.head_dim)));
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = elementValue(i + 7919);
    std::vector<float> k(static_cast<std::size_t>((1) * (shape.tokens) * (shape.head_dim)));
    for (std::size_t i = 0; i < k.size(); ++i) k[i] = elementValue(i + 15838);
    std::vector<float> w(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads)));
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = elementValue(i + 23757);
    std::vector<std::int32_t> topk_indices(
        static_cast<std::size_t>((1) * (shape.tokens) * (std::min<std::int64_t>(shape.tokens, 32))));
    for (std::size_t i = 0; i < topk_indices.size(); ++i)
        topk_indices[i] = static_cast<std::int32_t>((i % (std::min<std::int64_t>(shape.tokens, 32))) % shape.tokens);
    std::vector<float> predict(
        static_cast<std::size_t>((1) * (shape.tokens) * (std::min<std::int64_t>(shape.tokens, 32))));
    std::fill(predict.begin(), predict.end(), 123);
    float sm_scale                   = 0.25;
    std::uint8_t topk_indices_global = 0;

    auto expected_predict = predict;
    hostSparseIndexerPredict(shape, q, k, w, topk_indices, expected_predict, sm_scale, topk_indices_global);

    const std::int64_t q_dims[]{1, shape.tokens, shape.heads, shape.head_dim};
    DeviceBuffer<float> device_q(q.size());
    ASSERT_NE(device_q.get(), nullptr);
    const std::int64_t k_dims[]{1, shape.tokens, shape.head_dim};
    DeviceBuffer<float> device_k(k.size());
    ASSERT_NE(device_k.get(), nullptr);
    const std::int64_t w_dims[]{1, shape.tokens, shape.heads};
    DeviceBuffer<float> device_w(w.size());
    ASSERT_NE(device_w.get(), nullptr);
    const std::int64_t topk_indices_dims[]{1, shape.tokens, std::min<std::int64_t>(shape.tokens, 32)};
    DeviceBuffer<std::int32_t> device_topk_indices(topk_indices.size());
    ASSERT_NE(device_topk_indices.get(), nullptr);
    const std::int64_t predict_dims[]{1, shape.tokens, std::min<std::int64_t>(shape.tokens, 32)};
    DeviceBuffer<float> device_predict(predict.size());
    ASSERT_NE(device_predict.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, makeView(device_q.get(), q_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(device_k.get(), k_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(device_w.get(), w_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(device_topk_indices.get(), topk_indices_dims, 3, FTRAIN_NUMERIC_TYPE_INT32),
        makeView(device_predict.get(), predict_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
        makeView(&topk_indices_global, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
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
        device_w.upload(w, stream_);
        device_topk_indices.upload(topk_indices, stream_);
        device_predict.upload(predict, stream_);
        if (convenience) {
            ASSERT_EQ(ftrainSparseIndexerPredict(
                          makeView(device_q.get(), q_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_k.get(), k_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_w.get(), w_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_topk_indices.get(), topk_indices_dims, 3, FTRAIN_NUMERIC_TYPE_INT32),
                          makeView(device_predict.get(), predict_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                          makeView(&topk_indices_global, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8,
                                   FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                          FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_predict = device_predict.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_predict.size(), expected_predict.size());
        bool detects_zero_predict = false;
        for (std::size_t i = 0; i < expected_predict.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_predict[i])) << "predict reference " << i;
            if (std::isfinite(expected_predict[i])) {
                ASSERT_TRUE(std::isfinite(actual_predict[i])) << "predict " << i;
                detects_zero_predict |=
                    std::abs(expected_predict[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_predict[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_predict[i], expected_predict[i]) << "predict " << i;
            }
        }
        EXPECT_TRUE(detects_zero_predict) << "predict reference cannot reject an all-zero result";
        expectNear(actual_predict, expected_predict, kRelTolerance, kAbsTolerance, "predict");
    }
}

INSTANTIATE_TEST_SUITE_P(SparseIndexerPredictShapeGrid, SparseIndexerPredictNumericsTest,
                         testing::Values(ShapeParam{8, 2, 32}, ShapeParam{33, 2, 64}, ShapeParam{128, 4, 128}),
                         [](const testing::TestParamInfo<SparseIndexerPredictNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.tokens) + "x" + std::to_string(info.param.heads) + "x" +
                                    std::to_string(info.param.head_dim);
                         });
}  // namespace
