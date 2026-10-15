// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Staged and convenience execution against a host reference over the same shape grid.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <flash_train/flash_train.h>
#include "../support/numerics.hpp"
#include "h_post_bda_bwd_staged_fixture.hpp"

namespace {
using ftrain_test::DeviceBuffer;
using ftrain_test::expectNear;

struct ShapeParam {
    std::int64_t tokens;
    std::int64_t streams;
    std::int64_t width;
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

void hostHPostBdaBwd(const ShapeParam& shape, const std::vector<float>& grad_output, const std::vector<float>& residual,
                     const std::vector<float>& x, const std::vector<float>& bias, const std::vector<float>& h_res,
                     const std::vector<float>& h_post, std::vector<float>& grad_residual, std::vector<float>& grad_x,
                     std::vector<float>& grad_h_res, std::vector<float>& grad_h_post, std::vector<float>& grad_bias) {
    std::fill(grad_x.begin(), grad_x.end(), 0);
    std::fill(grad_bias.begin(), grad_bias.end(), 0);
    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        for (std::int64_t i = 0; i < shape.streams; ++i)
            for (std::int64_t d = 0; d < shape.width; ++d) {
                float sum = 0;
                for (std::int64_t j = 0; j < shape.streams; ++j)
                    sum += h_res[(row * shape.streams + i) * shape.streams + j] *
                           grad_output[(row * shape.streams + j) * shape.width + d];
                grad_residual[(row * shape.streams + i) * shape.width + d] = sum;
            }
        for (std::int64_t i = 0; i < shape.streams; ++i)
            for (std::int64_t j = 0; j < shape.streams; ++j) {
                float sum = 0;
                for (std::int64_t d = 0; d < shape.width; ++d)
                    sum += residual[(row * shape.streams + i) * shape.width + d] *
                           grad_output[(row * shape.streams + j) * shape.width + d];
                grad_h_res[(row * shape.streams + i) * shape.streams + j] = sum;
            }
        for (std::int64_t j = 0; j < shape.streams; ++j) {
            float sum = 0;
            for (std::int64_t d = 0; d < shape.width; ++d) {
                const float g                  = grad_output[(row * shape.streams + j) * shape.width + d];
                sum                           += g * (x[row * shape.width + d] + bias[d]);
                grad_x[row * shape.width + d] += g * h_post[row * shape.streams + j];
                grad_bias[d]                  += g * h_post[row * shape.streams + j];
            }
            grad_h_post[row * shape.streams + j] = sum;
        }
    }
}

class HPostBdaBwdNumericsTest : public ftrain_test::HPostBdaBwdStagedTest,
                                public testing::WithParamInterface<ShapeParam> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(HPostBdaBwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> grad_output(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.width)));
    for (std::size_t i = 0; i < grad_output.size(); ++i) grad_output[i] = elementValue(i + 7919);
    std::vector<float> residual(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.width)));
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = elementValue(i + 15838);
    std::vector<float> x(static_cast<std::size_t>((shape.tokens) * (1) * (shape.width)));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = elementValue(i + 23757);
    std::vector<float> bias(static_cast<std::size_t>((shape.width)));
    for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = elementValue(i + 31676);
    std::vector<float> h_res(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.streams)));
    for (std::size_t i = 0; i < h_res.size(); ++i) h_res[i] = elementValue(i + 39595);
    std::vector<float> h_post(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams)));
    for (std::size_t i = 0; i < h_post.size(); ++i) h_post[i] = elementValue(i + 47514);
    std::vector<float> grad_residual(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.width)));
    std::fill(grad_residual.begin(), grad_residual.end(), 123);
    std::vector<float> grad_x(static_cast<std::size_t>((shape.tokens) * (1) * (shape.width)));
    std::fill(grad_x.begin(), grad_x.end(), 123);
    std::vector<float> grad_h_res(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.streams)));
    std::fill(grad_h_res.begin(), grad_h_res.end(), 123);
    std::vector<float> grad_h_post(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams)));
    std::fill(grad_h_post.begin(), grad_h_post.end(), 123);
    std::vector<float> grad_bias(static_cast<std::size_t>((shape.width)));
    std::fill(grad_bias.begin(), grad_bias.end(), 123);

    auto expected_grad_residual = grad_residual;
    auto expected_grad_x        = grad_x;
    auto expected_grad_h_res    = grad_h_res;
    auto expected_grad_h_post   = grad_h_post;
    auto expected_grad_bias     = grad_bias;
    hostHPostBdaBwd(shape, grad_output, residual, x, bias, h_res, h_post, expected_grad_residual, expected_grad_x,
                    expected_grad_h_res, expected_grad_h_post, expected_grad_bias);

    const std::int64_t grad_output_dims[]{shape.tokens, 1, shape.streams, shape.width};
    DeviceBuffer<float> device_grad_output(grad_output.size());
    ASSERT_NE(device_grad_output.get(), nullptr);
    const std::int64_t residual_dims[]{shape.tokens, 1, shape.streams, shape.width};
    DeviceBuffer<float> device_residual(residual.size());
    ASSERT_NE(device_residual.get(), nullptr);
    const std::int64_t x_dims[]{shape.tokens, 1, shape.width};
    DeviceBuffer<float> device_x(x.size());
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t bias_dims[]{shape.width};
    DeviceBuffer<float> device_bias(bias.size());
    ASSERT_NE(device_bias.get(), nullptr);
    const std::int64_t h_res_dims[]{shape.tokens, 1, shape.streams, shape.streams};
    DeviceBuffer<float> device_h_res(h_res.size());
    ASSERT_NE(device_h_res.get(), nullptr);
    const std::int64_t h_post_dims[]{shape.tokens, 1, shape.streams};
    DeviceBuffer<float> device_h_post(h_post.size());
    ASSERT_NE(device_h_post.get(), nullptr);
    const std::int64_t grad_residual_dims[]{shape.tokens, 1, shape.streams, shape.width};
    DeviceBuffer<float> device_grad_residual(grad_residual.size());
    ASSERT_NE(device_grad_residual.get(), nullptr);
    const std::int64_t grad_x_dims[]{shape.tokens, 1, shape.width};
    DeviceBuffer<float> device_grad_x(grad_x.size());
    ASSERT_NE(device_grad_x.get(), nullptr);
    const std::int64_t grad_h_res_dims[]{shape.tokens, 1, shape.streams, shape.streams};
    DeviceBuffer<float> device_grad_h_res(grad_h_res.size());
    ASSERT_NE(device_grad_h_res.get(), nullptr);
    const std::int64_t grad_h_post_dims[]{shape.tokens, 1, shape.streams};
    DeviceBuffer<float> device_grad_h_post(grad_h_post.size());
    ASSERT_NE(device_grad_h_post.get(), nullptr);
    const std::int64_t grad_bias_dims[]{shape.width};
    DeviceBuffer<float> device_grad_bias(grad_bias.size());
    ASSERT_NE(device_grad_bias.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_grad_output.get(), grad_output_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_residual.get(), residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_x.get(), x_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_bias.get(), bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_h_res.get(), h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_h_post.get(), h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_grad_residual.get(), grad_residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_grad_x.get(), grad_x_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_grad_h_res.get(), grad_h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_grad_h_post.get(), grad_h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_grad_bias.get(), grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                        FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    auto* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) ASSERT_NE(workspace, nullptr);

    for (bool convenience : {false, true}) {
        device_grad_output.upload(grad_output, stream_);
        device_residual.upload(residual, stream_);
        device_x.upload(x, stream_);
        device_bias.upload(bias, stream_);
        device_h_res.upload(h_res, stream_);
        device_h_post.upload(h_post, stream_);
        device_grad_residual.upload(grad_residual, stream_);
        device_grad_x.upload(grad_x, stream_);
        device_grad_h_res.upload(grad_h_res, stream_);
        device_grad_h_post.upload(grad_h_post, stream_);
        device_grad_bias.upload(grad_bias, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainHPostBdaBwd(makeView(device_grad_output.get(), grad_output_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                                  makeView(device_residual.get(), residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                                  makeView(device_x.get(), x_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                                  makeView(device_bias.get(), bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                                  makeView(device_h_res.get(), h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                                  makeView(device_h_post.get(), h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                                  makeView(device_grad_residual.get(), grad_residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                                  makeView(device_grad_x.get(), grad_x_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                                  makeView(device_grad_h_res.get(), grad_h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                                  makeView(device_grad_h_post.get(), grad_h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                                  makeView(device_grad_bias.get(), grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                                  FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        expectNear(device_grad_residual.download(stream_), expected_grad_residual, kRelTolerance, kAbsTolerance,
                   "grad_residual");
        expectNear(device_grad_x.download(stream_), expected_grad_x, kRelTolerance, kAbsTolerance, "grad_x");
        expectNear(device_grad_h_res.download(stream_), expected_grad_h_res, kRelTolerance, kAbsTolerance,
                   "grad_h_res");
        expectNear(device_grad_h_post.download(stream_), expected_grad_h_post, kRelTolerance, kAbsTolerance,
                   "grad_h_post");
        expectNear(device_grad_bias.download(stream_), expected_grad_bias, kRelTolerance, kAbsTolerance, "grad_bias");
    }
}

INSTANTIATE_TEST_SUITE_P(HPostBdaBwdShapeGrid, HPostBdaBwdNumericsTest,
                         testing::Values(ShapeParam{8, 2, 32}, ShapeParam{33, 4, 64}, ShapeParam{128, 4, 128}),
                         [](const testing::TestParamInfo<HPostBdaBwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.tokens) + "x" + std::to_string(info.param.streams) + "x" +
                                    std::to_string(info.param.width);
                         });
}  // namespace
