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
#include "sparse_attn_bwd_staged_fixture.hpp"

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

void hostSparseAttnBwd(const ShapeParam& shape, const std::vector<float>& q, const std::vector<float>& kv,
                       const std::vector<float>& out, const std::vector<float>& dout, const std::vector<float>& lse,
                       const std::vector<float>& attn_sink, const std::vector<std::int32_t>& topk_indices,
                       const std::vector<std::int32_t>& topk_length, std::vector<float>& dq, std::vector<float>& dkv,
                       std::vector<float>& dsink, float softmax_scale) {
    std::fill(dq.begin(), dq.end(), 0);
    std::fill(dkv.begin(), dkv.end(), 0);
    std::fill(dsink.begin(), dsink.end(), 0);

    for (std::int64_t row = 0; row < shape.tokens; ++row)
        for (std::int64_t h = 0; h < shape.heads; ++h) {
            std::vector<float> logits(128), probabilities(128);
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int64_t j = 0; j < 128; ++j) {
                float dot      = 0;
                const auto col = topk_indices[row * 128 + j];
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += q[(row * shape.heads + h) * shape.head_dim + d] * kv[col * shape.head_dim + d];
                logits[j] = dot * softmax_scale;
                maximum   = std::max(maximum, logits[j]);
            }
            float sum = 0;
            for (auto value : logits) sum += std::exp(value - maximum);
            const float with_sink = sum + std::exp(attn_sink[h] - maximum);
            for (std::int64_t j = 0; j < 128; ++j) probabilities[j] = std::exp(logits[j] - maximum) / with_sink;
            std::vector<float> value(shape.head_dim, 0);
            for (std::int64_t j = 0; j < 128; ++j)
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    value[d] += probabilities[j] * kv[topk_indices[row * 128 + j] * shape.head_dim + d];

            float output_dot = 0;
            for (std::int64_t d = 0; d < shape.head_dim; ++d)
                output_dot += value[d] * dout[(row * shape.heads + h) * shape.head_dim + d];
            dsink[h] -= std::exp(attn_sink[h] - maximum) / with_sink * output_dot;
            for (std::int64_t j = 0; j < 128; ++j) {
                const auto col = topk_indices[row * 128 + j];
                float dot      = 0;
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot +=
                        dout[(row * shape.heads + h) * shape.head_dim + d] * (kv[col * shape.head_dim + d] - value[d]);
                const float ds = probabilities[j] * dot;
                for (std::int64_t d = 0; d < shape.head_dim; ++d) {
                    dq[(row * shape.heads + h) * shape.head_dim + d] +=
                        ds * kv[col * shape.head_dim + d] * softmax_scale;
                    dkv[col * shape.head_dim + d] +=
                        probabilities[j] * dout[(row * shape.heads + h) * shape.head_dim + d] +
                        ds * q[(row * shape.heads + h) * shape.head_dim + d] * softmax_scale;
                }
            }
        }
}

class SparseAttnBwdNumericsTest : public ftrain_test::SparseAttnBwdStagedTest,
                                  public testing::WithParamInterface<ShapeParam> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(SparseAttnBwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> q(static_cast<std::size_t>((shape.tokens) * (shape.heads) * (shape.head_dim)));
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = elementValue(i + 7919);
    std::vector<float> kv(static_cast<std::size_t>((shape.tokens) * (shape.head_dim)));
    for (std::size_t i = 0; i < kv.size(); ++i) kv[i] = elementValue(i + 15838);
    std::vector<float> out(static_cast<std::size_t>((shape.tokens) * (shape.heads) * (shape.head_dim)));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = elementValue(i + 23757);
    std::vector<float> dout(static_cast<std::size_t>((shape.tokens) * (shape.heads) * (shape.head_dim)));
    for (std::size_t i = 0; i < dout.size(); ++i) dout[i] = elementValue(i + 31676);
    std::vector<float> lse(static_cast<std::size_t>((shape.tokens) * (shape.heads)));
    for (std::size_t i = 0; i < lse.size(); ++i) lse[i] = elementValue(i + 39595);
    std::vector<float> attn_sink(static_cast<std::size_t>((shape.heads)));
    for (std::size_t i = 0; i < attn_sink.size(); ++i) attn_sink[i] = elementValue(i + 47514);
    std::vector<std::int32_t> topk_indices(static_cast<std::size_t>((shape.tokens) * (128)));
    for (std::size_t i = 0; i < topk_indices.size(); ++i)
        topk_indices[i] = static_cast<std::int32_t>((i % (128)) % shape.tokens);
    std::vector<std::int32_t> topk_length(static_cast<std::size_t>((shape.tokens)));
    std::fill(topk_length.begin(), topk_length.end(), 128);
    std::vector<float> dq(static_cast<std::size_t>((shape.tokens) * (shape.heads) * (shape.head_dim)));
    std::fill(dq.begin(), dq.end(), 123);
    std::vector<float> dkv(static_cast<std::size_t>((shape.tokens) * (shape.head_dim)));
    std::fill(dkv.begin(), dkv.end(), 123);
    std::vector<float> dsink(static_cast<std::size_t>((shape.heads)));
    std::fill(dsink.begin(), dsink.end(), 123);
    float softmax_scale = 0.125;

    for (std::int64_t row = 0; row < shape.tokens; ++row)
        for (std::int64_t h = 0; h < shape.heads; ++h) {
            std::vector<float> logits(128), probabilities(128);
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int64_t j = 0; j < 128; ++j) {
                float dot      = 0;
                const auto col = topk_indices[row * 128 + j];
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += q[(row * shape.heads + h) * shape.head_dim + d] * kv[col * shape.head_dim + d];
                logits[j] = dot * softmax_scale;
                maximum   = std::max(maximum, logits[j]);
            }
            float sum = 0;
            for (auto value : logits) sum += std::exp(value - maximum);
            const float with_sink = sum + std::exp(attn_sink[h] - maximum);
            for (std::int64_t j = 0; j < 128; ++j) probabilities[j] = std::exp(logits[j] - maximum) / with_sink;
            std::vector<float> value(shape.head_dim, 0);
            for (std::int64_t j = 0; j < 128; ++j)
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    value[d] += probabilities[j] * kv[topk_indices[row * 128 + j] * shape.head_dim + d];

            lse[row * shape.heads + h] = maximum + std::log(sum);
            for (std::int64_t d = 0; d < shape.head_dim; ++d)
                out[(row * shape.heads + h) * shape.head_dim + d] = value[d];
        }

    auto expected_dq    = dq;
    auto expected_dkv   = dkv;
    auto expected_dsink = dsink;
    hostSparseAttnBwd(shape, q, kv, out, dout, lse, attn_sink, topk_indices, topk_length, expected_dq, expected_dkv,
                      expected_dsink, softmax_scale);

    const std::int64_t q_dims[]{shape.tokens, shape.heads, shape.head_dim};
    DeviceBuffer<float> device_q(q.size());
    ASSERT_NE(device_q.get(), nullptr);
    const std::int64_t kv_dims[]{shape.tokens, shape.head_dim};
    DeviceBuffer<float> device_kv(kv.size());
    ASSERT_NE(device_kv.get(), nullptr);
    const std::int64_t out_dims[]{shape.tokens, shape.heads, shape.head_dim};
    DeviceBuffer<float> device_out(out.size());
    ASSERT_NE(device_out.get(), nullptr);
    const std::int64_t dout_dims[]{shape.tokens, shape.heads, shape.head_dim};
    DeviceBuffer<float> device_dout(dout.size());
    ASSERT_NE(device_dout.get(), nullptr);
    const std::int64_t lse_dims[]{shape.tokens, shape.heads};
    DeviceBuffer<float> device_lse(lse.size());
    ASSERT_NE(device_lse.get(), nullptr);
    const std::int64_t attn_sink_dims[]{shape.heads};
    DeviceBuffer<float> device_attn_sink(attn_sink.size());
    ASSERT_NE(device_attn_sink.get(), nullptr);
    const std::int64_t topk_indices_dims[]{shape.tokens, 128};
    DeviceBuffer<std::int32_t> device_topk_indices(topk_indices.size());
    ASSERT_NE(device_topk_indices.get(), nullptr);
    const std::int64_t topk_length_dims[]{shape.tokens};
    DeviceBuffer<std::int32_t> device_topk_length(topk_length.size());
    ASSERT_NE(device_topk_length.get(), nullptr);
    const std::int64_t dq_dims[]{shape.tokens, shape.heads, shape.head_dim};
    DeviceBuffer<float> device_dq(dq.size());
    ASSERT_NE(device_dq.get(), nullptr);
    const std::int64_t dkv_dims[]{shape.tokens, shape.head_dim};
    DeviceBuffer<float> device_dkv(dkv.size());
    ASSERT_NE(device_dkv.get(), nullptr);
    const std::int64_t dsink_dims[]{shape.heads};
    DeviceBuffer<float> device_dsink(dsink.size());
    ASSERT_NE(device_dsink.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_q.get(), q_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_kv.get(), kv_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_out.get(), out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_dout.get(), dout_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_lse.get(), lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_attn_sink.get(), attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_topk_indices.get(), topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                makeView(device_topk_length.get(), topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                makeView(device_dq.get(), dq_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_dkv.get(), dkv_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_dsink.get(), dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
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
        device_kv.upload(kv, stream_);
        device_out.upload(out, stream_);
        device_dout.upload(dout, stream_);
        device_lse.upload(lse, stream_);
        device_attn_sink.upload(attn_sink, stream_);
        device_topk_indices.upload(topk_indices, stream_);
        device_topk_length.upload(topk_length, stream_);
        device_dq.upload(dq, stream_);
        device_dkv.upload(dkv, stream_);
        device_dsink.upload(dsink, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainSparseAttnBwd(
                    makeView(device_q.get(), q_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_kv.get(), kv_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_out.get(), out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_dout.get(), dout_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_lse.get(), lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_attn_sink.get(), attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_topk_indices.get(), topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(device_topk_length.get(), topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(device_dq.get(), dq_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_dkv.get(), dkv_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_dsink.get(), dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        expectNear(device_dq.download(stream_), expected_dq, kRelTolerance, kAbsTolerance, "dq");
        expectNear(device_dkv.download(stream_), expected_dkv, kRelTolerance, kAbsTolerance, "dkv");
        expectNear(device_dsink.download(stream_), expected_dsink, kRelTolerance, kAbsTolerance, "dsink");
    }
}

INSTANTIATE_TEST_SUITE_P(SparseAttnBwdShapeGrid, SparseAttnBwdNumericsTest,
                         testing::Values(ShapeParam{8, 8, 512}, ShapeParam{33, 8, 512}, ShapeParam{128, 16, 512}),
                         [](const testing::TestParamInfo<SparseAttnBwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.tokens) + "x" + std::to_string(info.param.heads) + "x" +
                                    std::to_string(info.param.head_dim);
                         });
}  // namespace
