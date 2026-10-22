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
#include "proj_rms_fwd_staged_fixture.hpp"

namespace {
using ftrain_test::DeviceBuffer;
using ftrain_test::expectNear;

struct ShapeParam {
    std::int64_t rows;
    std::int64_t outputs;
    std::int64_t reduction;
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

void hostProjRmsFwd(const ShapeParam& shape, const std::vector<float>& x, const std::vector<float>& weight,
                    std::vector<float>& proj, std::vector<float>& r, float eps) {
    for (std::int64_t row = 0; row < shape.rows; ++row) {
        float squared = 0;
        for (std::int64_t d = 0; d < shape.reduction; ++d)
            squared += x[row * shape.reduction + d] * x[row * shape.reduction + d];
        r[row] = 1.0F / (std::sqrt(squared / shape.reduction) + eps);
        for (std::int64_t j = 0; j < shape.outputs; ++j) {
            float sum = 0;
            for (std::int64_t d = 0; d < shape.reduction; ++d)
                sum += x[row * shape.reduction + d] * weight[j * shape.reduction + d];
            proj[row * shape.outputs + j] = sum;
        }
    }
}

class ProjRmsFwdNumericsTest : public ftrain_test::ProjRmsFwdStagedTest,
                               public testing::WithParamInterface<ShapeParam> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(ProjRmsFwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> x(static_cast<std::size_t>((shape.rows) * (shape.reduction)));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = elementValue(i + 7919);
    std::vector<float> weight(static_cast<std::size_t>((shape.outputs) * (shape.reduction)));
    for (std::size_t i = 0; i < weight.size(); ++i) weight[i] = elementValue(i + 15838);
    std::vector<float> proj(static_cast<std::size_t>((shape.rows) * (shape.outputs)));
    std::fill(proj.begin(), proj.end(), 123);
    std::vector<float> r(static_cast<std::size_t>((shape.rows) * (1)));
    std::fill(r.begin(), r.end(), 123);
    float eps = 1e-6;

    auto expected_proj = proj;
    auto expected_r    = r;
    hostProjRmsFwd(shape, x, weight, expected_proj, expected_r, eps);

    const std::int64_t x_dims[]{shape.rows, shape.reduction};
    DeviceBuffer<float> device_x(x.size());
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t weight_dims[]{shape.outputs, shape.reduction};
    DeviceBuffer<float> device_weight(weight.size());
    ASSERT_NE(device_weight.get(), nullptr);
    const std::int64_t proj_dims[]{shape.rows, shape.outputs};
    DeviceBuffer<float> device_proj(proj.size());
    ASSERT_NE(device_proj.get(), nullptr);
    const std::int64_t r_dims[]{shape.rows, 1};
    DeviceBuffer<float> device_r(r.size());
    ASSERT_NE(device_r.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_x.get(), x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_weight.get(), weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_proj.get(), proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_r.get(), r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
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
        device_weight.upload(weight, stream_);
        device_proj.upload(proj, stream_);
        device_r.upload(r, stream_);
        if (convenience) {
            ASSERT_EQ(ftrainProjRmsFwd(
                          makeView(device_x.get(), x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_weight.get(), weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_proj.get(), proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_r.get(), r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                          FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_proj = device_proj.download(stream_);
        const auto actual_r    = device_r.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_proj.size(), expected_proj.size());
        bool detects_zero_proj = false;
        for (std::size_t i = 0; i < expected_proj.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_proj[i])) << "proj reference " << i;
            if (std::isfinite(expected_proj[i])) {
                ASSERT_TRUE(std::isfinite(actual_proj[i])) << "proj " << i;
                detects_zero_proj |=
                    std::abs(expected_proj[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_proj[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_proj[i], expected_proj[i]) << "proj " << i;
            }
        }
        EXPECT_TRUE(detects_zero_proj) << "proj reference cannot reject an all-zero result";
        expectNear(actual_proj, expected_proj, kRelTolerance, kAbsTolerance, "proj");
        ASSERT_EQ(actual_r.size(), expected_r.size());
        bool detects_zero_r = false;
        for (std::size_t i = 0; i < expected_r.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_r[i])) << "r reference " << i;
            if (std::isfinite(expected_r[i])) {
                ASSERT_TRUE(std::isfinite(actual_r[i])) << "r " << i;
                detects_zero_r |= std::abs(expected_r[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_r[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_r[i], expected_r[i]) << "r " << i;
            }
        }
        EXPECT_TRUE(detects_zero_r) << "r reference cannot reject an all-zero result";
        expectNear(actual_r, expected_r, kRelTolerance, kAbsTolerance, "r");
    }
}

INSTANTIATE_TEST_SUITE_P(ProjRmsFwdShapeGrid, ProjRmsFwdNumericsTest,
                         testing::Values(ShapeParam{16, 8, 32}, ShapeParam{33, 16, 64}, ShapeParam{128, 24, 128}),
                         [](const testing::TestParamInfo<ProjRmsFwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.rows) + "x" + std::to_string(info.param.outputs) + "x" +
                                    std::to_string(info.param.reduction);
                         });
}  // namespace
