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
#include "indexer_bwd_dense_thd_staged_fixture.hpp"

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

void hostIndexerBwdDenseThd(const ShapeParam& shape, const std::vector<float>& index_q,
                            const std::vector<float>& weights, const std::vector<float>& index_k,
                            const std::vector<float>& attn_score, const std::vector<float>& attn_l1norm,
                            const std::vector<float>& index_score, const std::vector<float>& index_lse,
                            const std::vector<std::int32_t>& cu_seqlens_q,
                            const std::vector<std::int32_t>& cu_seqlens_k,
                            const std::vector<std::int32_t>& q_causal_offsets, std::vector<float>& grad_index_score,
                            std::vector<float>& dq, std::vector<float>& dw, std::vector<float>& dk, float sm_scale,
                            float loss_coeff, float grad_loss, std::int64_t ratio) {
    std::fill(dq.begin(), dq.end(), 0);
    std::fill(dw.begin(), dw.end(), 0);
    std::fill(dk.begin(), dk.end(), 0);
    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        float direct_sum = 0;
        for (std::int64_t col = 0; col <= row; ++col)
            direct_sum -= attn_score[row * shape.tokens + col] / (attn_l1norm[row] + 1e-10F) / shape.tokens;
        for (std::int64_t col = 0; col < shape.tokens; ++col) {
            const float target = attn_score[row * shape.tokens + col] / (attn_l1norm[row] + 1e-10F);
            const float ds     = col <= row
                                     ? (-target / shape.tokens -
                                        std::exp(index_score[row * shape.tokens + col] - index_lse[row]) * direct_sum) *
                                           loss_coeff * grad_loss
                                     : 0.0F;
            grad_index_score[row * shape.tokens + col] = ds;
            for (std::int64_t h = 0; h < shape.heads; ++h) {
                float dot = 0;
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += index_q[(row * shape.heads + h) * shape.head_dim + d] * index_k[col * shape.head_dim + d];
                dot                       *= sm_scale;
                dw[row * shape.heads + h] += std::max(dot, 0.0F) * ds;
                const float dg             = dot > 0 ? ds * weights[row * shape.heads + h] * sm_scale : 0.0F;
                for (std::int64_t d = 0; d < shape.head_dim; ++d) {
                    dq[(row * shape.heads + h) * shape.head_dim + d] += dg * index_k[col * shape.head_dim + d];
                    dk[col * shape.head_dim + d] += dg * index_q[(row * shape.heads + h) * shape.head_dim + d];
                }
            }
        }
    }
}

class IndexerBwdDenseThdNumericsTest : public ftrain_test::IndexerBwdDenseThdStagedTest,
                                       public testing::WithParamInterface<ShapeParam> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(IndexerBwdDenseThdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> index_q(static_cast<std::size_t>((shape.tokens) * (shape.heads) * (shape.head_dim)));
    for (std::size_t i = 0; i < index_q.size(); ++i) index_q[i] = elementValue(i + 7919);
    std::vector<float> weights(static_cast<std::size_t>((shape.tokens) * (shape.heads)));
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = elementValue(i + 15838);
    std::vector<float> index_k(static_cast<std::size_t>((shape.tokens) * (shape.head_dim)));
    for (std::size_t i = 0; i < index_k.size(); ++i) index_k[i] = elementValue(i + 23757);
    std::vector<float> attn_score(static_cast<std::size_t>((shape.tokens) * (shape.tokens)));
    for (std::size_t i = 0; i < attn_score.size(); ++i) attn_score[i] = elementValue(i + 31676);
    std::vector<float> attn_l1norm(static_cast<std::size_t>((shape.tokens)));
    for (std::size_t i = 0; i < attn_l1norm.size(); ++i) attn_l1norm[i] = elementValue(i + 39595);
    std::vector<float> index_score(static_cast<std::size_t>((shape.tokens) * (shape.tokens)));
    for (std::size_t i = 0; i < index_score.size(); ++i) index_score[i] = elementValue(i + 47514);
    std::vector<float> index_lse(static_cast<std::size_t>((shape.tokens)));
    for (std::size_t i = 0; i < index_lse.size(); ++i) index_lse[i] = elementValue(i + 55433);
    std::vector<std::int32_t> cu_seqlens_q(static_cast<std::size_t>((2)));
    cu_seqlens_q[0] = 0;
    cu_seqlens_q[1] = static_cast<std::int32_t>(shape.tokens);
    std::vector<std::int32_t> cu_seqlens_k(static_cast<std::size_t>((2)));
    cu_seqlens_k[0] = 0;
    cu_seqlens_k[1] = static_cast<std::int32_t>(shape.tokens);
    std::vector<std::int32_t> q_causal_offsets(static_cast<std::size_t>((1)));
    q_causal_offsets[0] = 0;
    std::vector<float> grad_index_score(static_cast<std::size_t>((shape.tokens) * (shape.tokens)));
    std::fill(grad_index_score.begin(), grad_index_score.end(), 123);
    std::vector<float> dq(static_cast<std::size_t>((shape.tokens) * (shape.heads) * (shape.head_dim)));
    std::fill(dq.begin(), dq.end(), 123);
    std::vector<float> dw(static_cast<std::size_t>((shape.tokens) * (shape.heads)));
    std::fill(dw.begin(), dw.end(), 123);
    std::vector<float> dk(static_cast<std::size_t>((shape.tokens) * (shape.head_dim)));
    std::fill(dk.begin(), dk.end(), 123);
    float sm_scale     = 0.25;
    float loss_coeff   = 1;
    float grad_loss    = 1;
    std::int64_t ratio = 1;

    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::int64_t col = 0; col < shape.tokens; ++col) {
            float score = 0;
            for (std::int64_t h = 0; h < shape.heads; ++h) {
                float dot = 0;
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += index_q[(row * shape.heads + h) * shape.head_dim + d] * index_k[col * shape.head_dim + d];
                score += std::max(dot * sm_scale, 0.0F) * weights[row * shape.heads + h];
            }
            index_score[row * shape.tokens + col] = col <= row ? score : -std::numeric_limits<float>::infinity();
            maximum                               = std::max(maximum, index_score[row * shape.tokens + col]);
        }
        float sum = 0;
        for (std::int64_t col = 0; col <= row; ++col) sum += std::exp(index_score[row * shape.tokens + col] - maximum);
        index_lse[row] = maximum + std::log(sum);
    }

    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        attn_l1norm[row] = 0;
        for (std::int64_t col = 0; col < shape.tokens; ++col) {
            auto& value       = attn_score[row * shape.tokens + col];
            value             = col <= row ? std::abs(value) + 0.01F : 0.0F;
            attn_l1norm[row] += value;
        }
    }

    auto expected_grad_index_score = grad_index_score;
    auto expected_dq               = dq;
    auto expected_dw               = dw;
    auto expected_dk               = dk;
    hostIndexerBwdDenseThd(shape, index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse,
                           cu_seqlens_q, cu_seqlens_k, q_causal_offsets, expected_grad_index_score, expected_dq,
                           expected_dw, expected_dk, sm_scale, loss_coeff, grad_loss, ratio);

    const std::int64_t index_q_dims[]{shape.tokens, shape.heads, shape.head_dim};
    DeviceBuffer<float> device_index_q(index_q.size());
    ASSERT_NE(device_index_q.get(), nullptr);
    const std::int64_t weights_dims[]{shape.tokens, shape.heads};
    DeviceBuffer<float> device_weights(weights.size());
    ASSERT_NE(device_weights.get(), nullptr);
    const std::int64_t index_k_dims[]{shape.tokens, shape.head_dim};
    DeviceBuffer<float> device_index_k(index_k.size());
    ASSERT_NE(device_index_k.get(), nullptr);
    const std::int64_t attn_score_dims[]{shape.tokens, shape.tokens};
    DeviceBuffer<float> device_attn_score(attn_score.size());
    ASSERT_NE(device_attn_score.get(), nullptr);
    const std::int64_t attn_l1norm_dims[]{shape.tokens};
    DeviceBuffer<float> device_attn_l1norm(attn_l1norm.size());
    ASSERT_NE(device_attn_l1norm.get(), nullptr);
    const std::int64_t index_score_dims[]{shape.tokens, shape.tokens};
    DeviceBuffer<float> device_index_score(index_score.size());
    ASSERT_NE(device_index_score.get(), nullptr);
    const std::int64_t index_lse_dims[]{shape.tokens};
    DeviceBuffer<float> device_index_lse(index_lse.size());
    ASSERT_NE(device_index_lse.get(), nullptr);
    const std::int64_t cu_seqlens_q_dims[]{2};
    DeviceBuffer<std::int32_t> device_cu_seqlens_q(cu_seqlens_q.size());
    ASSERT_NE(device_cu_seqlens_q.get(), nullptr);
    const std::int64_t cu_seqlens_k_dims[]{2};
    DeviceBuffer<std::int32_t> device_cu_seqlens_k(cu_seqlens_k.size());
    ASSERT_NE(device_cu_seqlens_k.get(), nullptr);
    const std::int64_t q_causal_offsets_dims[]{1};
    DeviceBuffer<std::int32_t> device_q_causal_offsets(q_causal_offsets.size());
    ASSERT_NE(device_q_causal_offsets.get(), nullptr);
    const std::int64_t grad_index_score_dims[]{shape.tokens, shape.tokens};
    const std::int64_t dq_dims[]{shape.tokens, shape.heads, shape.head_dim};
    DeviceBuffer<float> device_dq(dq.size());
    ASSERT_NE(device_dq.get(), nullptr);
    const std::int64_t dw_dims[]{shape.tokens, shape.heads};
    DeviceBuffer<float> device_dw(dw.size());
    ASSERT_NE(device_dw.get(), nullptr);
    const std::int64_t dk_dims[]{shape.tokens, shape.head_dim};
    DeviceBuffer<float> device_dk(dk.size());
    ASSERT_NE(device_dk.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_index_q.get(), index_q_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_weights.get(), weights_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_index_k.get(), index_k_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_attn_score.get(), attn_score_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_attn_l1norm.get(), attn_l1norm_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_index_score.get(), index_score_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_index_lse.get(), index_lse_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_cu_seqlens_q.get(), cu_seqlens_q_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                        makeView(device_cu_seqlens_k.get(), cu_seqlens_k_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                        makeView(device_q_causal_offsets.get(), q_causal_offsets_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                        makeView(device_index_score.get(), grad_index_score_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_dq.get(), dq_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_dw.get(), dw_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_dk.get(), dk_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        makeView(&loss_coeff, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        makeView(&grad_loss, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    auto* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) ASSERT_NE(workspace, nullptr);

    for (bool convenience : {false, true}) {
        device_index_q.upload(index_q, stream_);
        device_weights.upload(weights, stream_);
        device_index_k.upload(index_k, stream_);
        device_attn_score.upload(attn_score, stream_);
        device_attn_l1norm.upload(attn_l1norm, stream_);
        device_index_score.upload(index_score, stream_);
        device_index_lse.upload(index_lse, stream_);
        device_cu_seqlens_q.upload(cu_seqlens_q, stream_);
        device_cu_seqlens_k.upload(cu_seqlens_k, stream_);
        device_q_causal_offsets.upload(q_causal_offsets, stream_);
        device_dq.upload(dq, stream_);
        device_dw.upload(dw, stream_);
        device_dk.upload(dk, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainIndexerBwdDenseThd(
                    makeView(device_index_q.get(), index_q_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_weights.get(), weights_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_index_k.get(), index_k_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_attn_score.get(), attn_score_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_attn_l1norm.get(), attn_l1norm_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_index_score.get(), index_score_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_index_lse.get(), index_lse_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_cu_seqlens_q.get(), cu_seqlens_q_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(device_cu_seqlens_k.get(), cu_seqlens_k_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(device_q_causal_offsets.get(), q_causal_offsets_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(device_index_score.get(), grad_index_score_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_dq.get(), dq_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_dw.get(), dw_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_dk.get(), dk_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&loss_coeff, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&grad_loss, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        expectNear(device_index_score.download(stream_), expected_grad_index_score, kRelTolerance, kAbsTolerance,
                   "grad_index_score");
        expectNear(device_dq.download(stream_), expected_dq, kRelTolerance, kAbsTolerance, "dq");
        expectNear(device_dw.download(stream_), expected_dw, kRelTolerance, kAbsTolerance, "dw");
        expectNear(device_dk.download(stream_), expected_dk, kRelTolerance, kAbsTolerance, "dk");
    }
}

INSTANTIATE_TEST_SUITE_P(IndexerBwdDenseThdShapeGrid, IndexerBwdDenseThdNumericsTest,
                         testing::Values(ShapeParam{8, 2, 32}, ShapeParam{33, 2, 64}, ShapeParam{128, 4, 128}),
                         [](const testing::TestParamInfo<IndexerBwdDenseThdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.tokens) + "x" + std::to_string(info.param.heads) + "x" +
                                    std::to_string(info.param.head_dim);
                         });
}  // namespace
