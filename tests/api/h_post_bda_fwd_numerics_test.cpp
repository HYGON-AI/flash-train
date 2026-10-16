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
#include "h_post_bda_fwd_staged_fixture.hpp"

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

void hostHPostBdaFwd(const ShapeParam& shape, const std::vector<float>& h_res, const std::vector<float>& residual,
                     const std::vector<float>& h_post, const std::vector<float>& x, const std::vector<float>& bias,
                     std::vector<float>& output) {
    for (std::int64_t row = 0; row < shape.tokens; ++row)
        for (std::int64_t j = 0; j < shape.streams; ++j)
            for (std::int64_t d = 0; d < shape.width; ++d) {
                float sum = h_post[row * shape.streams + j] * (x[row * shape.width + d] + bias[d]);
                for (std::int64_t i = 0; i < shape.streams; ++i)
                    sum += h_res[(row * shape.streams + i) * shape.streams + j] *
                           residual[(row * shape.streams + i) * shape.width + d];
                output[(row * shape.streams + j) * shape.width + d] = sum;
            }
}

class HPostBdaFwdNumericsTest : public ftrain_test::HPostBdaFwdStagedTest,
                                public testing::WithParamInterface<ShapeParam> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(HPostBdaFwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> h_res(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.streams)));
    for (std::size_t i = 0; i < h_res.size(); ++i) h_res[i] = elementValue(i + 7919);
    std::vector<float> residual(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.width)));
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = elementValue(i + 15838);
    std::vector<float> h_post(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams)));
    for (std::size_t i = 0; i < h_post.size(); ++i) h_post[i] = elementValue(i + 23757);
    std::vector<float> x(static_cast<std::size_t>((shape.tokens) * (1) * (shape.width)));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = elementValue(i + 31676);
    std::vector<float> bias(static_cast<std::size_t>((shape.width)));
    for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = elementValue(i + 39595);
    std::vector<float> output(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.width)));
    std::fill(output.begin(), output.end(), 123);

    auto expected_output = output;
    hostHPostBdaFwd(shape, h_res, residual, h_post, x, bias, expected_output);

    const std::int64_t h_res_dims[]{shape.tokens, 1, shape.streams, shape.streams};
    DeviceBuffer<float> device_h_res(h_res.size());
    ASSERT_NE(device_h_res.get(), nullptr);
    const std::int64_t residual_dims[]{shape.tokens, 1, shape.streams, shape.width};
    DeviceBuffer<float> device_residual(residual.size());
    ASSERT_NE(device_residual.get(), nullptr);
    const std::int64_t h_post_dims[]{shape.tokens, 1, shape.streams};
    DeviceBuffer<float> device_h_post(h_post.size());
    ASSERT_NE(device_h_post.get(), nullptr);
    const std::int64_t x_dims[]{shape.tokens, 1, shape.width};
    DeviceBuffer<float> device_x(x.size());
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t bias_dims[]{shape.width};
    DeviceBuffer<float> device_bias(bias.size());
    ASSERT_NE(device_bias.get(), nullptr);
    const std::int64_t output_dims[]{shape.tokens, 1, shape.streams, shape.width};
    DeviceBuffer<float> device_output(output.size());
    ASSERT_NE(device_output.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_h_res.get(), h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_residual.get(), residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_h_post.get(), h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_x.get(), x_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_bias.get(), bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_output.get(), output_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                        FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    auto* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) ASSERT_NE(workspace, nullptr);

    for (bool convenience : {false, true}) {
        device_h_res.upload(h_res, stream_);
        device_residual.upload(residual, stream_);
        device_h_post.upload(h_post, stream_);
        device_x.upload(x, stream_);
        device_bias.upload(bias, stream_);
        device_output.upload(output, stream_);
        if (convenience) {
            ASSERT_EQ(ftrainHPostBdaFwd(makeView(device_h_res.get(), h_res_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                                        makeView(device_residual.get(), residual_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                                        makeView(device_h_post.get(), h_post_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                                        makeView(device_x.get(), x_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                                        makeView(device_bias.get(), bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                                        makeView(device_output.get(), output_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                                        FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        expectNear(device_output.download(stream_), expected_output, kRelTolerance, kAbsTolerance, "output");
    }
}

INSTANTIATE_TEST_SUITE_P(HPostBdaFwdShapeGrid, HPostBdaFwdNumericsTest,
                         testing::Values(ShapeParam{8, 2, 32}, ShapeParam{33, 4, 64}, ShapeParam{128, 4, 128}),
                         [](const testing::TestParamInfo<HPostBdaFwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.tokens) + "x" + std::to_string(info.param.streams) + "x" +
                                    std::to_string(info.param.width);
                         });
}  // namespace
