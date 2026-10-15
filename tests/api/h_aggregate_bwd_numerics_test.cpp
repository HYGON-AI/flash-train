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
#include "h_aggregate_bwd_staged_fixture.hpp"

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

void hostHAggregateBwd(const ShapeParam& shape, const std::vector<float>& grad_aggregated, const std::vector<float>& x,
                       const std::vector<float>& h_pre, std::vector<float>& grad_x, std::vector<float>& grad_h_pre) {
    for (std::int64_t row = 0; row < shape.tokens; ++row)
        for (std::int64_t i = 0; i < shape.streams; ++i) {
            float sum = 0;
            for (std::int64_t d = 0; d < shape.width; ++d) {
                grad_x[(row * shape.streams + i) * shape.width + d] =
                    grad_aggregated[row * shape.width + d] * h_pre[row * shape.streams + i];
                sum += grad_aggregated[row * shape.width + d] * x[(row * shape.streams + i) * shape.width + d];
            }
            grad_h_pre[row * shape.streams + i] = sum;
        }
}

class HAggregateBwdNumericsTest : public ftrain_test::HAggregateBwdStagedTest,
                                  public testing::WithParamInterface<ShapeParam> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(HAggregateBwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> grad_aggregated(static_cast<std::size_t>((shape.tokens) * (1) * (shape.width)));
    for (std::size_t i = 0; i < grad_aggregated.size(); ++i) grad_aggregated[i] = elementValue(i + 7919);
    std::vector<float> x(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.width)));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = elementValue(i + 15838);
    std::vector<float> h_pre(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams)));
    for (std::size_t i = 0; i < h_pre.size(); ++i) h_pre[i] = elementValue(i + 23757);
    std::vector<float> grad_x(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.width)));
    std::fill(grad_x.begin(), grad_x.end(), 123);
    std::vector<float> grad_h_pre(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams)));
    std::fill(grad_h_pre.begin(), grad_h_pre.end(), 123);

    auto expected_grad_x     = grad_x;
    auto expected_grad_h_pre = grad_h_pre;
    hostHAggregateBwd(shape, grad_aggregated, x, h_pre, expected_grad_x, expected_grad_h_pre);

    const std::int64_t grad_aggregated_dims[]{shape.tokens, 1, shape.width};
    DeviceBuffer<float> device_grad_aggregated(grad_aggregated.size());
    ASSERT_NE(device_grad_aggregated.get(), nullptr);
    const std::int64_t x_dims[]{shape.tokens, 1, shape.streams, shape.width};
    DeviceBuffer<float> device_x(x.size());
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t h_pre_dims[]{shape.tokens, 1, shape.streams};
    DeviceBuffer<float> device_h_pre(h_pre.size());
    ASSERT_NE(device_h_pre.get(), nullptr);
    const std::int64_t grad_x_dims[]{shape.tokens, 1, shape.streams, shape.width};
    DeviceBuffer<float> device_grad_x(grad_x.size());
    ASSERT_NE(device_grad_x.get(), nullptr);
    const std::int64_t grad_h_pre_dims[]{shape.tokens, 1, shape.streams};
    DeviceBuffer<float> device_grad_h_pre(grad_h_pre.size());
    ASSERT_NE(device_grad_h_pre.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_grad_aggregated.get(), grad_aggregated_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_x.get(), x_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_h_pre.get(), h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_grad_x.get(), grad_x_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_grad_h_pre.get(), grad_h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    auto* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) ASSERT_NE(workspace, nullptr);

    for (bool convenience : {false, true}) {
        device_grad_aggregated.upload(grad_aggregated, stream_);
        device_x.upload(x, stream_);
        device_h_pre.upload(h_pre, stream_);
        device_grad_x.upload(grad_x, stream_);
        device_grad_h_pre.upload(grad_h_pre, stream_);
        if (convenience) {
            ASSERT_EQ(ftrainHAggregateBwd(
                          makeView(device_grad_aggregated.get(), grad_aggregated_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_x.get(), x_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_h_pre.get(), h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_grad_x.get(), grad_x_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_grad_h_pre.get(), grad_h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        expectNear(device_grad_x.download(stream_), expected_grad_x, kRelTolerance, kAbsTolerance, "grad_x");
        expectNear(device_grad_h_pre.download(stream_), expected_grad_h_pre, kRelTolerance, kAbsTolerance,
                   "grad_h_pre");
    }
}

INSTANTIATE_TEST_SUITE_P(HAggregateBwdShapeGrid, HAggregateBwdNumericsTest,
                         testing::Values(ShapeParam{8, 2, 32}, ShapeParam{33, 4, 64}, ShapeParam{128, 4, 128}),
                         [](const testing::TestParamInfo<HAggregateBwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.tokens) + "x" + std::to_string(info.param.streams) + "x" +
                                    std::to_string(info.param.width);
                         });
}  // namespace
