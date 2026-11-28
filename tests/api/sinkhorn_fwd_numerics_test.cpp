// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Staged and convenience execution against a host reference over the same shape grid.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_fp16.h>

#include <flash_train/flash_train.h>
#include "../support/numerics.hpp"
#include "sinkhorn_fwd_staged_fixture.hpp"

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
            if (iter > 0) {
                for (std::int64_t row = 0; row < shape.matrix_size; ++row) {
                    float total = eps;
                    for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                        total += output[base + row * shape.matrix_size + col];
                    for (std::int64_t col = 0; col < shape.matrix_size; ++col)
                        output[base + row * shape.matrix_size + col] /= total;
                }
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
                                public testing::WithParamInterface<std::tuple<ShapeParam, FTrainNumericType>> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kFp32RelTolerance = 2e-4;
constexpr double kFp32AbsTolerance = 2e-5;

TEST_P(SinkhornFwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    RecordProperty("family", "sinkhorn_fwd");
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

    std::vector<float> logits(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    for (std::size_t i = 0; i < logits.size(); ++i) logits[i] = elementValue(i + 7919);
    std::vector<float> output(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    std::fill(output.begin(), output.end(), 123);
    std::vector<float> m_init(static_cast<std::size_t>((shape.batch) * (shape.matrix_size) * (shape.matrix_size)));
    std::fill(m_init.begin(), m_init.end(), 123);
    std::int64_t num_iterations = shape.iterations;
    float eps                   = shape.eps;

    for (auto* values : {&logits, &output}) { roundToStorage(*values, dtype); }

    auto expected_output = output;
    auto expected_m_init = m_init;
    hostSinkhornFwd(shape, logits, expected_output, expected_m_init, num_iterations, eps);

    roundToStorage(expected_output, dtype);

    const std::int64_t logits_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    FloatingDeviceBuffer device_logits(logits.size(), dtype);
    ASSERT_NE(device_logits.get(), nullptr);
    const std::int64_t output_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    FloatingDeviceBuffer device_output(output.size(), dtype);
    ASSERT_NE(device_output.get(), nullptr);
    const std::int64_t m_init_dims[]{shape.batch, shape.matrix_size, shape.matrix_size};
    DeviceBuffer<float> device_m_init(m_init.size());
    ASSERT_NE(device_m_init.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_logits.get(), logits_dims, 3, dtype),
                makeView(device_output.get(), output_dims, 3, dtype),
                makeView(device_m_init.get(), m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
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
        device_logits.upload(logits, stream_);
        device_output.upload(output, stream_);
        device_m_init.upload(m_init, stream_);
        if (convenience) {
            ASSERT_EQ(ftrainSinkhornFwd(
                          makeView(device_logits.get(), logits_dims, 3, dtype),
                          makeView(device_output.get(), output_dims, 3, dtype),
                          makeView(device_m_init.get(), m_init_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                          makeView(&num_iterations, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                   true),
                          makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                          dtype, workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, primitive, workspace, workspace_sizes[primitive], stream_),
                      FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_output = device_output.download(stream_);
        const auto actual_m_init = device_m_init.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_output.size(), expected_output.size());
        bool detects_zero_output = false;
        for (std::size_t i = 0; i < expected_output.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_output[i])) << "output reference " << i;
            if (std::isfinite(expected_output[i])) {
                ASSERT_TRUE(std::isfinite(actual_output[i])) << "output " << i;
                detects_zero_output |=
                    std::abs(expected_output[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_output[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_output[i], expected_output[i]) << "output " << i;
            }
        }
        EXPECT_TRUE(detects_zero_output) << "output reference cannot reject an all-zero result";
        expectNear(actual_output, expected_output, kRelTolerance, kAbsTolerance, "output");
        ASSERT_EQ(actual_m_init.size(), expected_m_init.size());
        bool detects_zero_m_init = false;
        for (std::size_t i = 0; i < expected_m_init.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_m_init[i])) << "m_init reference " << i;
            if (std::isfinite(expected_m_init[i])) {
                ASSERT_TRUE(std::isfinite(actual_m_init[i])) << "m_init " << i;
                detects_zero_m_init |=
                    std::abs(expected_m_init[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_m_init[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_m_init[i], expected_m_init[i]) << "m_init " << i;
            }
        }
        EXPECT_TRUE(detects_zero_m_init) << "m_init reference cannot reject an all-zero result";
        expectNear(actual_m_init, expected_m_init, kRelTolerance, kAbsTolerance, "m_init");
    }
}

INSTANTIATE_TEST_SUITE_P(
    SinkhornFwdShapeGrid, SinkhornFwdNumericsTest,
    testing::Combine(testing::Values(ShapeParam{8, 2, 3}, ShapeParam{33, 4, 5}, ShapeParam{128, 8, 5},
                                     ShapeParam{33, 4, 1, 0.5F}, ShapeParam{33, 8, 3, 0.5F}, ShapeParam{7, 2, 1, 0.5F}),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16)),
    [](const testing::TestParamInfo<SinkhornFwdNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).batch) + "x" +
               std::to_string(std::get<0>(info.param).matrix_size) + "x" +
               std::to_string(std::get<0>(info.param).iterations) + "_type" + std::to_string(std::get<1>(info.param));
    });
}  // namespace
