// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// Gemm numerics test: staged execution against a host reference over a shape
// grid (deliberately including non-power-of-two edges), for both a fresh
// output (beta = 0) and an accumulated output (beta != 0).

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "../support/numerics.hpp"
#include "gemm_staged_fixture.hpp"

namespace {

using ftrain_test::DeviceBuffer;
using ftrain_test::expectNear;
using ftrain_test::GemmStagedTest;

struct ShapeParam {
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
};

// Deterministic bounded fill in [-0.5, 0.5): the same rule feeds the device
// upload and the host reference, no RNG involved.
float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 199.0F - 0.5F;
}

std::vector<float> hostGemm(const std::vector<float>& a, const std::vector<float>& b,
                            const std::vector<float>& c, std::int64_t m, std::int64_t k, std::int64_t n,
                            float alpha, float beta) {
    std::vector<float> d(static_cast<std::size_t>(m * n));
    for (std::int64_t row = 0; row < m; ++row) {
        for (std::int64_t column = 0; column < n; ++column) {
            float accumulated = 0.0F;
            for (std::int64_t depth = 0; depth < k; ++depth) {
                accumulated += a[static_cast<std::size_t>(row * k + depth)] *
                               b[static_cast<std::size_t>(depth * n + column)];
            }
            d[static_cast<std::size_t>(row * n + column)] =
                alpha * accumulated +
                beta * c[static_cast<std::size_t>(row * n + column)];
        }
    }
    return d;
}

class GemmNumericsTest : public GemmStagedTest, public testing::WithParamInterface<ShapeParam> {};

// Tolerance policy: fp32 elements bounded to [-0.5, 0.5] and k <= 257. The
// mixed tolerance below keeps at least a 30x margin over the measured worst
// deviation on gfx936 across the grid (worst ratio 0.031 of the budget).
constexpr double kRelTolerance = 2e-5;
constexpr double kAbsTolerance = 2e-5;

TEST_P(GemmNumericsTest, MatchesHostReferenceForFreshAndAccumulatedOutput) {
    const ShapeParam shape = GetParam();
    const std::int64_t m = shape.m;
    const std::int64_t k = shape.k;
    const std::int64_t n = shape.n;
    const std::size_t mk = static_cast<std::size_t>(m * k);
    const std::size_t kn = static_cast<std::size_t>(k * n);
    const std::size_t mn = static_cast<std::size_t>(m * n);

    std::vector<float> a(mk);
    std::vector<float> b(kn);
    std::vector<float> c(mn);
    for (std::size_t i = 0; i < mk; ++i) { a[i] = elementValue(i); }
    for (std::size_t i = 0; i < kn; ++i) { b[i] = elementValue(i + 7919); }
    for (std::size_t i = 0; i < mn; ++i) { c[i] = elementValue(i + 104729); }

    DeviceBuffer<float> device_a(mk);
    DeviceBuffer<float> device_b(kn);
    DeviceBuffer<float> device_c(mn);
    DeviceBuffer<float> device_d(mn);
    ASSERT_NE(device_a.get(), nullptr);
    ASSERT_NE(device_b.get(), nullptr);
    ASSERT_NE(device_c.get(), nullptr);
    ASSERT_NE(device_d.get(), nullptr);
    device_a.upload(a, stream_);
    device_b.upload(b, stream_);
    device_c.upload(c, stream_);

    const std::int64_t a_dims[]{m, k};
    const std::int64_t b_dims[]{k, n};
    const std::int64_t d_dims[]{m, n};
    float alpha = 1.5F;
    float beta_zero = 0.0F;
    float beta_accumulate = 1.25F;

    // Fresh output: beta = 0, c carries no storage.
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_a.get(), a_dims, 2), makeView(device_b.get(), b_dims, 2),
                        makeView(nullptr, d_dims, 2), makeView(device_d.get(), d_dims, 2),
                        makeView(&alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        makeView(&beta_zero, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_SUCCESS);
    const std::vector<float> fresh = device_d.download(stream_);
    expectNear(fresh, hostGemm(a, b, c, m, k, n, alpha, 0.0F), kRelTolerance, kAbsTolerance,
               "fresh output (beta = 0)");

    // Accumulated output: beta = 1.25 on the c buffer.
    args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_a.get(), a_dims, 2), makeView(device_b.get(), b_dims, 2),
                        makeView(device_c.get(), d_dims, 2), makeView(device_d.get(), d_dims, 2),
                        makeView(&alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        makeView(&beta_accumulate, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    FTrainPlan accumulate_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&accumulate_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(accumulate_plan, nullptr);
    trackPlan(accumulate_plan);
    ASSERT_EQ(ftrainPlanExecutePrimitive(accumulate_plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_SUCCESS);
    const std::vector<float> accumulated = device_d.download(stream_);
    expectNear(accumulated, hostGemm(a, b, c, m, k, n, alpha, 1.25F), kRelTolerance, kAbsTolerance,
               "accumulated output (beta = 1.25)");
}

INSTANTIATE_TEST_SUITE_P(
    GemmShapeGrid, GemmNumericsTest,
    testing::Values(ShapeParam{1, 1, 1}, ShapeParam{3, 4, 2}, ShapeParam{16, 16, 16},
                    ShapeParam{64, 48, 32}, ShapeParam{129, 65, 17}, ShapeParam{33, 257, 65},
                    ShapeParam{256, 128, 64}),
    [](const testing::TestParamInfo<GemmNumericsTest::ParamType>& info) {
        return std::to_string(info.param.m) + "x" + std::to_string(info.param.k) + "x" +
               std::to_string(info.param.n);
    });

}  // namespace
