// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Staged and convenience execution against a host reference over the same shape grid.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_fp16.h>

#include <flash_train/flash_train.h>
#include "../support/numerics.hpp"
#include "sinkhorn_bwd_staged_fixture.hpp"

namespace {
using ftrain_test::DeviceBuffer;
using ftrain_test::expectNear;

// Host references must consume the same rounded values as low-precision storage.
inline std::uint16_t encodeFloat16(float value, FTrainNumericType type) {
    if (type == FTRAIN_NUMERIC_TYPE_FP16) {
        const __half half = __float2half(value);
        std::uint16_t bits;
        std::memcpy(&bits, &half, sizeof(bits));
        return bits;
    }
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

inline float decodeFloat16(std::uint16_t bits, FTrainNumericType type) {
    if (type == FTRAIN_NUMERIC_TYPE_FP16) {
        __half half;
        std::memcpy(&half, &bits, sizeof(bits));
        return __half2float(half);
    }
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

inline void roundToStorage(std::vector<float>& values, FTrainNumericType type) {
    if (type == FTRAIN_NUMERIC_TYPE_FP32) { return; }
    for (auto& value : values) { value = decodeFloat16(encodeFloat16(value, type), type); }
}

// A floating tensor with runtime storage precision, for a shared FP32 host oracle.
// Upload staging lives until the next synchronized test iteration. Readback waits
// before converting the bytes; production execution remains asynchronous.
class FloatingDeviceBuffer {
  public:
    FloatingDeviceBuffer(std::size_t count, FTrainNumericType type)
        : count_(count), type_(type),
          bytes_(count * (type == FTRAIN_NUMERIC_TYPE_FP32 ? sizeof(float) : sizeof(std::uint16_t))),
          staging_(bytes_.count()) {}

    void* get() const { return bytes_.get(); }

    void upload(const std::vector<float>& values, hipStream_t stream) {
        ASSERT_EQ(values.size(), count_);
        if (type_ == FTRAIN_NUMERIC_TYPE_FP32) {
            std::memcpy(staging_.data(), values.data(), staging_.size());
        } else {
            for (std::size_t i = 0; i < count_; ++i) {
                const auto bits = encodeFloat16(values[i], type_);
                std::memcpy(staging_.data() + i * sizeof(bits), &bits, sizeof(bits));
            }
        }
        bytes_.upload(staging_, stream);
    }

    std::vector<float> download(hipStream_t stream) {
        const auto storage = bytes_.download(stream);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        std::vector<float> values(count_);
        if (type_ == FTRAIN_NUMERIC_TYPE_FP32) {
            std::memcpy(values.data(), storage.data(), storage.size());
        } else {
            for (std::size_t i = 0; i < count_; ++i) {
                std::uint16_t bits;
                std::memcpy(&bits, storage.data() + i * sizeof(bits), sizeof(bits));
                values[i] = decodeFloat16(bits, type_);
            }
        }
        return values;
    }

  private:
    std::size_t count_;
    FTrainNumericType type_;
    DeviceBuffer<unsigned char> bytes_;
    std::vector<unsigned char> staging_;
};

struct ShapeParam {
    std::int64_t batch;
    std::int64_t matrix_size;
    std::int64_t iterations;
    float eps = 1e-6F;
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
            if (iter > 0) {
                for (std::int64_t row = 0; row < shape.matrix_size; ++row) {
                    float sum = eps;
                    for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                        sum += state[row * shape.matrix_size + col];
                    for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                        state[row * shape.matrix_size + col] /= sum;
                }
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
            if (iter > 0) {
                for (std::int64_t row = 0; row < shape.matrix_size; ++row) {
                    float sum = eps, dot = 0;
                    for (std::int64_t col = 0; col < shape.matrix_size; ++col) {
                        sum += before_rows[iter][row * shape.matrix_size + col];
                        dot +=
                            gradient[row * shape.matrix_size + col] * after_rows[iter][row * shape.matrix_size + col];
                    }
                    for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                        gradient[row * shape.matrix_size + col] = (gradient[row * shape.matrix_size + col] - dot) / sum;
                }
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
                                public testing::WithParamInterface<std::tuple<ShapeParam, FTrainNumericType>> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kFp32RelTolerance = 2e-4;
constexpr double kFp32AbsTolerance = 2e-5;

TEST_P(SinkhornBwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    RecordProperty("family", "sinkhorn_bwd");
    std::string executed_primitives;
    const FTrainNumericType dtype = std::get<1>(GetParam());
    const ShapeParam shape        = std::get<0>(GetParam());
    // Low-precision storage rounding; FP32 retains its original reduction budget.
    const double kRelTolerance = dtype == FTRAIN_NUMERIC_TYPE_FP32   ? kFp32RelTolerance
                                 : dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-2
                                                                     : 5e-2;
    const double kAbsTolerance = dtype == FTRAIN_NUMERIC_TYPE_FP32   ? kFp32AbsTolerance
                                 : dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-3
                                                                     : 5e-3;

    // Amplify low-precision gradients so a zero kernel cannot pass the absolute budget.
    std::vector<float> grad_output(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    for (std::size_t i = 0; i < grad_output.size(); ++i)
        grad_output[i] = elementValue(i + 7919) * (dtype == FTRAIN_NUMERIC_TYPE_FP32 ? 1.0F : 64.0F);
    std::vector<float> m_init(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    for (std::size_t i = 0; i < m_init.size(); ++i) m_init[i] = elementValue(i + 15838);
    std::vector<float> grad_logits(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    std::fill(grad_logits.begin(), grad_logits.end(), 123);
    std::int64_t num_iterations = shape.iterations;
    float eps                   = shape.eps;

    for (auto* values : {&grad_output, &grad_logits}) { roundToStorage(*values, dtype); }

    for (std::int64_t row = 0; row < shape.batch * shape.matrix_size; ++row) {
        float total = 0;
        for (std::int64_t col = 0; col < shape.matrix_size; ++col)
            total += std::exp(m_init[row * shape.matrix_size + col]);
        for (std::int64_t col = 0; col < shape.matrix_size; ++col)
            m_init[row * shape.matrix_size + col] = std::exp(m_init[row * shape.matrix_size + col]) / total + eps;
    }

    for (auto* values : {&grad_output, &grad_logits}) { roundToStorage(*values, dtype); }

    auto expected_grad_logits = grad_logits;
    hostSinkhornBwd(shape, grad_output, m_init, expected_grad_logits, num_iterations, eps);

    roundToStorage(expected_grad_logits, dtype);

    const std::int64_t grad_output_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    FloatingDeviceBuffer device_grad_output(grad_output.size(), dtype);
    ASSERT_NE(device_grad_output.get(), nullptr);
    const std::int64_t m_init_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    DeviceBuffer<float> device_m_init(m_init.size());
    ASSERT_NE(device_m_init.get(), nullptr);
    const std::int64_t grad_logits_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    FloatingDeviceBuffer device_grad_logits(grad_logits.size(), dtype);
    ASSERT_NE(device_grad_logits.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_grad_output.get(), grad_output_dims, 3, dtype),
                makeView(device_m_init.get(), m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_grad_logits.get(), grad_logits_dims, 3, dtype),
                makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true), dtype));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);
    std::uint64_t count = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    std::vector<std::uint64_t> workspace_sizes(count);
    std::uint64_t bytes = 0;
    for (std::uint64_t i = 0; i < count; ++i) {
        ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, i, &workspace_sizes[i]), FTRAIN_STATUS_SUCCESS);
        bytes = std::max(bytes, workspace_sizes[i]);
    }
    auto* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) ASSERT_NE(workspace, nullptr);

    // Exercise every applicable implementation, then the default convenience API.
    for (std::uint64_t primitive = 0; primitive <= count; ++primitive) {
        const bool convenience = primitive == count;
        const char* name       = "convenience";
        if (!convenience) {
            ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, primitive, &name), FTRAIN_STATUS_SUCCESS);
            executed_primitives += std::string(name) + "\n";
            RecordProperty("executed_primitives", executed_primitives);
        }
        SCOPED_TRACE(name);
        device_grad_output.upload(grad_output, stream_);
        device_m_init.upload(m_init, stream_);
        device_grad_logits.upload(grad_logits, stream_);
        if (convenience) {
            ASSERT_EQ(ftrainSinkhornBwd(
                          makeView(device_grad_output.get(), grad_output_dims, 3, dtype),
                          makeView(device_m_init.get(), m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(device_grad_logits.get(), grad_logits_dims, 3, dtype),
                          makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                   true),
                          makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                          dtype, workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, primitive, workspace, workspace_sizes[primitive], stream_),
                      FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_grad_logits = device_grad_logits.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_grad_logits.size(), expected_grad_logits.size());
        bool detects_zero_grad_logits = false;
        for (std::size_t i = 0; i < expected_grad_logits.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_grad_logits[i])) << "grad_logits reference " << i;
            if (std::isfinite(expected_grad_logits[i])) {
                ASSERT_TRUE(std::isfinite(actual_grad_logits[i])) << "grad_logits " << i;
                detects_zero_grad_logits |= std::abs(expected_grad_logits[i]) >
                                            kAbsTolerance + kRelTolerance * std::abs(expected_grad_logits[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_grad_logits[i], expected_grad_logits[i]) << "grad_logits " << i;
            }
        }
        EXPECT_TRUE(detects_zero_grad_logits) << "grad_logits reference cannot reject an all-zero result";
        expectNear(actual_grad_logits, expected_grad_logits, kRelTolerance, kAbsTolerance, "grad_logits");
    }
}

INSTANTIATE_TEST_SUITE_P(
    SinkhornBwdShapeGrid, SinkhornBwdNumericsTest,
    testing::Combine(testing::Values(ShapeParam{8, 2, 3}, ShapeParam{33, 4, 5}, ShapeParam{128, 8, 5},
                                     ShapeParam{33, 4, 1, 0.5F}, ShapeParam{33, 8, 3, 0.5F}, ShapeParam{7, 2, 1, 0.5F}),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16)),
    [](const testing::TestParamInfo<SinkhornBwdNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).batch) + "x" +
               std::to_string(std::get<0>(info.param).matrix_size) + "x" +
               std::to_string(std::get<0>(info.param).iterations) + "_type" + std::to_string(std::get<1>(info.param));
    });
}  // namespace
