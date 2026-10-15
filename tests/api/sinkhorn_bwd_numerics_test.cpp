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
#include "sinkhorn_bwd_staged_fixture.hpp"

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

void hostSinkhornBwd(const ShapeParam& shape, const std::vector<float>& grad_output, const std::vector<float>& m_init,
                     std::vector<float>& grad_logits, std::int64_t num_iterations, float eps) {
    for (std::int64_t batch = 0; batch < shape.batch; ++batch) {
        const auto base = batch * shape.matrix_size * shape.matrix_size;
        std::vector<float> state(m_init.begin() + base, m_init.begin() + base + shape.matrix_size * shape.matrix_size);
        std::vector<std::vector<float>> before_rows, after_rows, after_cols;
        for (std::int64_t iter = 0; iter < shape.iterations; ++iter) {
            before_rows.push_back(state);
            for (std::int64_t row = 0; row < shape.matrix_size; ++row) {
                float sum = eps;
                for (std::int64_t col = 0; col < shape.matrix_size; ++col) sum += state[row * shape.matrix_size + col];
                for (std::int64_t col = 0; col < shape.matrix_size; ++col) state[row * shape.matrix_size + col] /= sum;
            }
            after_rows.push_back(state);
            for (std::int64_t col = 0; col < shape.matrix_size; ++col) {
                float sum = eps;
                for (std::int64_t row = 0; row < shape.matrix_size; ++row) sum += state[row * shape.matrix_size + col];
                for (std::int64_t row = 0; row < shape.matrix_size; ++row) state[row * shape.matrix_size + col] /= sum;
            }
            after_cols.push_back(state);
        }
        std::vector<float> gradient(grad_output.begin() + base,
                                    grad_output.begin() + base + shape.matrix_size * shape.matrix_size);
        for (std::int64_t iter = shape.iterations; iter-- > 0;) {
            for (std::int64_t col = 0; col < shape.matrix_size; ++col) {
                float sum = eps, dot = 0;
                for (std::int64_t row = 0; row < shape.matrix_size; ++row) {
                    sum += after_rows[iter][row * shape.matrix_size + col];
                    dot += gradient[row * shape.matrix_size + col] * after_cols[iter][row * shape.matrix_size + col];
                }
                for (std::int64_t row = 0; row < shape.matrix_size; ++row)
                    gradient[row * shape.matrix_size + col] = (gradient[row * shape.matrix_size + col] - dot) / sum;
            }
            for (std::int64_t row = 0; row < shape.matrix_size; ++row) {
                float sum = eps, dot = 0;
                for (std::int64_t col = 0; col < shape.matrix_size; ++col) {
                    sum += before_rows[iter][row * shape.matrix_size + col];
                    dot += gradient[row * shape.matrix_size + col] * after_rows[iter][row * shape.matrix_size + col];
                }
                for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                    gradient[row * shape.matrix_size + col] = (gradient[row * shape.matrix_size + col] - dot) / sum;
            }
        }
        for (std::int64_t row = 0; row < shape.matrix_size; ++row) {
            float dot = 0;
            for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                dot += gradient[row * shape.matrix_size + col] * (m_init[base + row * shape.matrix_size + col] - eps);
            for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                grad_logits[base + row * shape.matrix_size + col] =
                    (m_init[base + row * shape.matrix_size + col] - eps) *
                    (gradient[row * shape.matrix_size + col] - dot);
        }
    }
}

class SinkhornBwdNumericsTest : public ftrain_test::SinkhornBwdStagedTest,
                                public testing::WithParamInterface<ShapeParam> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(SinkhornBwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> grad_output(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    for (std::size_t i = 0; i < grad_output.size(); ++i) grad_output[i] = elementValue(i + 7919);
    std::vector<float> m_init(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    for (std::size_t i = 0; i < m_init.size(); ++i) m_init[i] = elementValue(i + 15838);
    std::vector<float> grad_logits(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    std::fill(grad_logits.begin(), grad_logits.end(), 123);
    std::int64_t num_iterations = shape.iterations;
    float eps                   = 1e-6;

    for (std::int64_t row = 0; row < shape.batch * shape.matrix_size; ++row) {
        float total = 0;
        for (std::int64_t col = 0; col < shape.matrix_size; ++col)
            total += std::exp(m_init[row * shape.matrix_size + col]);
        for (std::int64_t col = 0; col < shape.matrix_size; ++col)
            m_init[row * shape.matrix_size + col] = std::exp(m_init[row * shape.matrix_size + col]) / total + eps;
    }

    auto expected_grad_logits = grad_logits;
    hostSinkhornBwd(shape, grad_output, m_init, expected_grad_logits, num_iterations, eps);

    const std::int64_t grad_output_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    DeviceBuffer<float> device_grad_output(grad_output.size());
    ASSERT_NE(device_grad_output.get(), nullptr);
    const std::int64_t m_init_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    DeviceBuffer<float> device_m_init(m_init.size());
    ASSERT_NE(device_m_init.get(), nullptr);
    const std::int64_t grad_logits_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    DeviceBuffer<float> device_grad_logits(grad_logits.size());
    ASSERT_NE(device_grad_logits.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_grad_output.get(), grad_output_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_m_init.get(), m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_grad_logits.get(), grad_logits_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
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
        device_grad_output.upload(grad_output, stream_);
        device_m_init.upload(m_init, stream_);
        device_grad_logits.upload(grad_logits, stream_);
        if (convenience) {
            ASSERT_EQ(ftrainSinkhornBwd(
                          makeView(device_grad_output.get(), grad_output_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_m_init.get(), m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_grad_logits.get(), grad_logits_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                   true),
                          makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                          FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        expectNear(device_grad_logits.download(stream_), expected_grad_logits, kRelTolerance, kAbsTolerance,
                   "grad_logits");
    }
}

INSTANTIATE_TEST_SUITE_P(SinkhornBwdShapeGrid, SinkhornBwdNumericsTest,
                         testing::Values(ShapeParam{8, 2, 3}, ShapeParam{33, 4, 5}, ShapeParam{128, 8, 5}),
                         [](const testing::TestParamInfo<SinkhornBwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.batch) + "x" + std::to_string(info.param.matrix_size) +
                                    "x" + std::to_string(info.param.iterations);
                         });
}  // namespace
