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
#include "sinkhorn_fwd_staged_fixture.hpp"

namespace {
using ftrain_test::DeviceBuffer;
using ftrain_test::expectNear;

struct ShapeParam {
    std::int64_t batch;
    std::int64_t matrix_size;
    std::int64_t iterations;
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

void hostSinkhornFwd(const ShapeParam& shape, const std::vector<float>& logits, std::vector<float>& output,
                     std::vector<float>& m_init, std::int64_t num_iterations, float eps) {
    for (std::int64_t batch = 0; batch < shape.batch; ++batch) {
        const auto base = batch * shape.matrix_size * shape.matrix_size;
        for (std::int64_t row = 0; row < shape.matrix_size; ++row) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                maximum = std::max(maximum, logits[base + row * shape.matrix_size + col]);
            float total = 0;
            for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                total += std::exp(logits[base + row * shape.matrix_size + col] - maximum);
            for (std::int64_t col = 0; col < shape.matrix_size; ++col) {
                const auto i = base + row * shape.matrix_size + col;
                output[i] = m_init[i] = std::exp(logits[i] - maximum) / total + eps;
            }
        }
        for (std::int64_t iter = 0; iter < shape.iterations; ++iter) {
            for (std::int64_t row = 0; row < shape.matrix_size; ++row) {
                float total = eps;
                for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                    total += output[base + row * shape.matrix_size + col];
                for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                    output[base + row * shape.matrix_size + col] /= total;
            }
            for (std::int64_t col = 0; col < shape.matrix_size; ++col) {
                float total = eps;
                for (std::int64_t row = 0; row < shape.matrix_size; ++row)
                    total += output[base + row * shape.matrix_size + col];
                for (std::int64_t row = 0; row < shape.matrix_size; ++row)
                    output[base + row * shape.matrix_size + col] /= total;
            }
        }
    }
}

class SinkhornFwdNumericsTest : public ftrain_test::SinkhornFwdStagedTest,
                                public testing::WithParamInterface<ShapeParam> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(SinkhornFwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> logits(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    for (std::size_t i = 0; i < logits.size(); ++i) logits[i] = elementValue(i + 7919);
    std::vector<float> output(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    std::fill(output.begin(), output.end(), 123);
    std::vector<float> m_init(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    std::fill(m_init.begin(), m_init.end(), 123);
    std::int64_t num_iterations = shape.iterations;
    float eps                   = 1e-6;

    auto expected_output = output;
    auto expected_m_init = m_init;
    hostSinkhornFwd(shape, logits, expected_output, expected_m_init, num_iterations, eps);

    const std::int64_t logits_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    DeviceBuffer<float> device_logits(logits.size());
    ASSERT_NE(device_logits.get(), nullptr);
    const std::int64_t output_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    DeviceBuffer<float> device_output(output.size());
    ASSERT_NE(device_output.get(), nullptr);
    const std::int64_t m_init_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    DeviceBuffer<float> device_m_init(m_init.size());
    ASSERT_NE(device_m_init.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_logits.get(), logits_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_output.get(), output_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_m_init.get(), m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
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
        device_logits.upload(logits, stream_);
        device_output.upload(output, stream_);
        device_m_init.upload(m_init, stream_);
        if (convenience) {
            ASSERT_EQ(ftrainSinkhornFwd(
                          makeView(device_logits.get(), logits_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_output.get(), output_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_m_init.get(), m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                   true),
                          makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                          FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        expectNear(device_output.download(stream_), expected_output, kRelTolerance, kAbsTolerance, "output");
        expectNear(device_m_init.download(stream_), expected_m_init, kRelTolerance, kAbsTolerance, "m_init");
    }
}

INSTANTIATE_TEST_SUITE_P(SinkhornFwdShapeGrid, SinkhornFwdNumericsTest,
                         testing::Values(ShapeParam{8, 2, 3}, ShapeParam{33, 4, 5}, ShapeParam{128, 8, 5}),
                         [](const testing::TestParamInfo<SinkhornFwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.batch) + "x" + std::to_string(info.param.matrix_size) +
                                    "x" + std::to_string(info.param.iterations);
                         });
}  // namespace
