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
#include "proj_rms_bwd_staged_fixture.hpp"

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

void hostProjRmsBwd(const ShapeParam& shape, const std::vector<float>& grad_proj, const std::vector<float>& grad_r,
                    const std::vector<float>& x, const std::vector<float>& weight, const std::vector<float>& r,
                    std::vector<float>& grad_x, std::vector<float>& grad_weight, float eps) {
    std::fill(grad_weight.begin(), grad_weight.end(), 0);
    for (std::int64_t row = 0; row < shape.rows; ++row)
        for (std::int64_t d = 0; d < shape.reduction; ++d) {
            float norm2 = 0;
            for (std::int64_t k = 0; k < shape.reduction; ++k)
                norm2 += x[row * shape.reduction + k] * x[row * shape.reduction + k];
            const float rms = std::sqrt(norm2 / shape.reduction);
            float gx        = -grad_r[row] * r[row] * r[row] * x[row * shape.reduction + d] / (shape.reduction * rms);
            for (std::int64_t j = 0; j < shape.outputs; ++j) {
                gx += grad_proj[row * shape.outputs + j] * weight[j * shape.reduction + d];
                grad_weight[j * shape.reduction + d] +=
                    grad_proj[row * shape.outputs + j] * x[row * shape.reduction + d];
            }
            grad_x[row * shape.reduction + d] = gx;
        }
}

class ProjRmsBwdNumericsTest : public ftrain_test::ProjRmsBwdStagedTest,
                               public testing::WithParamInterface<ShapeParam> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(ProjRmsBwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> grad_proj(static_cast<std::size_t>((shape.rows) * (shape.outputs)));
    for (std::size_t i = 0; i < grad_proj.size(); ++i) grad_proj[i] = elementValue(i + 7919);
    std::vector<float> grad_r(static_cast<std::size_t>((shape.rows) * (1)));
    for (std::size_t i = 0; i < grad_r.size(); ++i) grad_r[i] = elementValue(i + 15838);
    std::vector<float> x(static_cast<std::size_t>((shape.rows) * (shape.reduction)));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = elementValue(i + 23757);
    std::vector<float> weight(static_cast<std::size_t>((shape.outputs) * (shape.reduction)));
    for (std::size_t i = 0; i < weight.size(); ++i) weight[i] = elementValue(i + 31676);
    std::vector<float> r(static_cast<std::size_t>((shape.rows) * (1)));
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = elementValue(i + 39595);
    std::vector<float> grad_x(static_cast<std::size_t>((shape.rows) * (shape.reduction)));
    std::fill(grad_x.begin(), grad_x.end(), 123);
    std::vector<float> grad_weight(static_cast<std::size_t>((shape.outputs) * (shape.reduction)));
    std::fill(grad_weight.begin(), grad_weight.end(), 123);
    float eps = 1e-6;

    for (std::int64_t row = 0; row < shape.rows; ++row) {
        float sum = 0;
        for (std::int64_t d = 0; d < shape.reduction; ++d)
            sum += x[row * shape.reduction + d] * x[row * shape.reduction + d];
        r[row] = 1.0F / (std::sqrt(sum / shape.reduction) + eps);
    }

    auto expected_grad_x      = grad_x;
    auto expected_grad_weight = grad_weight;
    hostProjRmsBwd(shape, grad_proj, grad_r, x, weight, r, expected_grad_x, expected_grad_weight, eps);

    const std::int64_t grad_proj_dims[]{shape.rows, shape.outputs};
    DeviceBuffer<float> device_grad_proj(grad_proj.size());
    ASSERT_NE(device_grad_proj.get(), nullptr);
    const std::int64_t grad_r_dims[]{shape.rows, 1};
    DeviceBuffer<float> device_grad_r(grad_r.size());
    ASSERT_NE(device_grad_r.get(), nullptr);
    const std::int64_t x_dims[]{shape.rows, shape.reduction};
    DeviceBuffer<float> device_x(x.size());
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t weight_dims[]{shape.outputs, shape.reduction};
    DeviceBuffer<float> device_weight(weight.size());
    ASSERT_NE(device_weight.get(), nullptr);
    const std::int64_t r_dims[]{shape.rows, 1};
    DeviceBuffer<float> device_r(r.size());
    ASSERT_NE(device_r.get(), nullptr);
    const std::int64_t grad_x_dims[]{shape.rows, shape.reduction};
    DeviceBuffer<float> device_grad_x(grad_x.size());
    ASSERT_NE(device_grad_x.get(), nullptr);
    const std::int64_t grad_weight_dims[]{shape.outputs, shape.reduction};
    DeviceBuffer<float> device_grad_weight(grad_weight.size());
    ASSERT_NE(device_grad_weight.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_grad_proj.get(), grad_proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_grad_r.get(), grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_x.get(), x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_weight.get(), weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_r.get(), r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_grad_x.get(), grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_grad_weight.get(), grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
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
        device_grad_proj.upload(grad_proj, stream_);
        device_grad_r.upload(grad_r, stream_);
        device_x.upload(x, stream_);
        device_weight.upload(weight, stream_);
        device_r.upload(r, stream_);
        device_grad_x.upload(grad_x, stream_);
        device_grad_weight.upload(grad_weight, stream_);
        if (convenience) {
            ASSERT_EQ(ftrainProjRmsBwd(
                          makeView(device_grad_proj.get(), grad_proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_grad_r.get(), grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_x.get(), x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_weight.get(), weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_r.get(), r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_grad_x.get(), grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_grad_weight.get(), grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                          FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        expectNear(device_grad_x.download(stream_), expected_grad_x, kRelTolerance, kAbsTolerance, "grad_x");
        expectNear(device_grad_weight.download(stream_), expected_grad_weight, kRelTolerance, kAbsTolerance,
                   "grad_weight");
    }
}

INSTANTIATE_TEST_SUITE_P(ProjRmsBwdShapeGrid, ProjRmsBwdNumericsTest,
                         testing::Values(ShapeParam{16, 8, 32}, ShapeParam{33, 16, 64}, ShapeParam{128, 24, 128}),
                         [](const testing::TestParamInfo<ProjRmsBwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.rows) + "x" + std::to_string(info.param.outputs) + "x" +
                                    std::to_string(info.param.reduction);
                         });
}  // namespace
