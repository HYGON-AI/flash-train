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
#include "h_aggregate_fwd_staged_fixture.hpp"

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

void hostHAggregateFwd(const ShapeParam& shape, const std::vector<float>& x, const std::vector<float>& h_pre,
                       std::vector<float>& aggregated) {
    for (std::int64_t row = 0; row < shape.tokens; ++row)
        for (std::int64_t d = 0; d < shape.width; ++d) {
            float sum = 0;
            for (std::int64_t i = 0; i < shape.streams; ++i)
                sum += x[(row * shape.streams + i) * shape.width + d] * h_pre[row * shape.streams + i];
            aggregated[row * shape.width + d] = sum;
        }
}

class HAggregateFwdNumericsTest : public ftrain_test::HAggregateFwdStagedTest,
                                  public testing::WithParamInterface<ShapeParam> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(HAggregateFwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> x(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.width)));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = elementValue(i + 7919);
    std::vector<float> h_pre(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams)));
    for (std::size_t i = 0; i < h_pre.size(); ++i) h_pre[i] = elementValue(i + 15838);
    std::vector<float> aggregated(static_cast<std::size_t>((shape.tokens) * (1) * (shape.width)));
    std::fill(aggregated.begin(), aggregated.end(), 123);

    auto expected_aggregated = aggregated;
    hostHAggregateFwd(shape, x, h_pre, expected_aggregated);

    const std::int64_t x_dims[]{shape.tokens, 1, shape.streams, shape.width};
    DeviceBuffer<float> device_x(x.size());
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t h_pre_dims[]{shape.tokens, 1, shape.streams};
    DeviceBuffer<float> device_h_pre(h_pre.size());
    ASSERT_NE(device_h_pre.get(), nullptr);
    const std::int64_t aggregated_dims[]{shape.tokens, 1, shape.width};
    DeviceBuffer<float> device_aggregated(aggregated.size());
    ASSERT_NE(device_aggregated.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_x.get(), x_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_h_pre.get(), h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_aggregated.get(), aggregated_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    auto* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) ASSERT_NE(workspace, nullptr);

    for (bool convenience : {false, true}) {
        device_x.upload(x, stream_);
        device_h_pre.upload(h_pre, stream_);
        device_aggregated.upload(aggregated, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainHAggregateFwd(makeView(device_x.get(), x_dims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                                    makeView(device_h_pre.get(), h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                                    makeView(device_aggregated.get(), aggregated_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_aggregated = device_aggregated.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_aggregated.size(), expected_aggregated.size());
        bool detects_zero_aggregated = false;
        for (std::size_t i = 0; i < expected_aggregated.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_aggregated[i])) << "aggregated reference " << i;
            if (std::isfinite(expected_aggregated[i])) {
                ASSERT_TRUE(std::isfinite(actual_aggregated[i])) << "aggregated " << i;
                detects_zero_aggregated |=
                    std::abs(expected_aggregated[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_aggregated[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_aggregated[i], expected_aggregated[i]) << "aggregated " << i;
            }
        }
        EXPECT_TRUE(detects_zero_aggregated) << "aggregated reference cannot reject an all-zero result";
        expectNear(actual_aggregated, expected_aggregated, kRelTolerance, kAbsTolerance, "aggregated");
    }
}

INSTANTIATE_TEST_SUITE_P(HAggregateFwdShapeGrid, HAggregateFwdNumericsTest,
                         testing::Values(ShapeParam{8, 2, 32}, ShapeParam{33, 4, 64}, ShapeParam{128, 4, 128}),
                         [](const testing::TestParamInfo<HAggregateFwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.tokens) + "x" + std::to_string(info.param.streams) + "x" +
                                    std::to_string(info.param.width);
                         });
}  // namespace
