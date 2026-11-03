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
#include "h_aggregate_bwd_staged_fixture.hpp"

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
                                  public testing::WithParamInterface<std::tuple<ShapeParam, FTrainNumericType>> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kFp32RelTolerance = 2e-4;
constexpr double kFp32AbsTolerance = 2e-5;

TEST_P(HAggregateBwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    RecordProperty("family", "h_aggregate_bwd");
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

    for (auto* values : {&grad_aggregated, &x, &grad_x}) { roundToStorage(*values, dtype); }

    auto expected_grad_x     = grad_x;
    auto expected_grad_h_pre = grad_h_pre;
    hostHAggregateBwd(shape, grad_aggregated, x, h_pre, expected_grad_x, expected_grad_h_pre);

    roundToStorage(expected_grad_x, dtype);

    const std::int64_t grad_aggregated_dims[]{shape.tokens, 1, shape.width};
    FloatingDeviceBuffer device_grad_aggregated(grad_aggregated.size(), dtype);
    ASSERT_NE(device_grad_aggregated.get(), nullptr);
    const std::int64_t x_dims[]{shape.tokens, 1, shape.streams, shape.width};
    FloatingDeviceBuffer device_x(x.size(), dtype);
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t h_pre_dims[]{shape.tokens, 1, shape.streams};
    DeviceBuffer<float> device_h_pre(h_pre.size());
    ASSERT_NE(device_h_pre.get(), nullptr);
    const std::int64_t grad_x_dims[]{shape.tokens, 1, shape.streams, shape.width};
    FloatingDeviceBuffer device_grad_x(grad_x.size(), dtype);
    ASSERT_NE(device_grad_x.get(), nullptr);
    const std::int64_t grad_h_pre_dims[]{shape.tokens, 1, shape.streams};
    DeviceBuffer<float> device_grad_h_pre(grad_h_pre.size());
    ASSERT_NE(device_grad_h_pre.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_grad_aggregated.get(), grad_aggregated_dims, 3, dtype),
                        makeView(device_x.get(), x_dims, 4, dtype),
                        makeView(device_h_pre.get(), h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_grad_x.get(), grad_x_dims, 4, dtype),
                        makeView(device_grad_h_pre.get(), grad_h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        FTRAIN_NUMERIC_TYPE_FP32));
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
        device_grad_aggregated.upload(grad_aggregated, stream_);
        device_x.upload(x, stream_);
        device_h_pre.upload(h_pre, stream_);
        device_grad_x.upload(grad_x, stream_);
        device_grad_h_pre.upload(grad_h_pre, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainHAggregateBwd(makeView(device_grad_aggregated.get(), grad_aggregated_dims, 3, dtype),
                                    makeView(device_x.get(), x_dims, 4, dtype),
                                    makeView(device_h_pre.get(), h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                                    makeView(device_grad_x.get(), grad_x_dims, 4, dtype),
                                    makeView(device_grad_h_pre.get(), grad_h_pre_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, primitive, workspace, workspace_sizes[primitive], stream_),
                      FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_grad_x     = device_grad_x.download(stream_);
        const auto actual_grad_h_pre = device_grad_h_pre.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_grad_x.size(), expected_grad_x.size());
        bool detects_zero_grad_x = false;
        for (std::size_t i = 0; i < expected_grad_x.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_grad_x[i])) << "grad_x reference " << i;
            if (std::isfinite(expected_grad_x[i])) {
                ASSERT_TRUE(std::isfinite(actual_grad_x[i])) << "grad_x " << i;
                detects_zero_grad_x |=
                    std::abs(expected_grad_x[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_grad_x[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_grad_x[i], expected_grad_x[i]) << "grad_x " << i;
            }
        }
        EXPECT_TRUE(detects_zero_grad_x) << "grad_x reference cannot reject an all-zero result";
        expectNear(actual_grad_x, expected_grad_x, kRelTolerance, kAbsTolerance, "grad_x");
        ASSERT_EQ(actual_grad_h_pre.size(), expected_grad_h_pre.size());
        bool detects_zero_grad_h_pre = false;
        for (std::size_t i = 0; i < expected_grad_h_pre.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_grad_h_pre[i])) << "grad_h_pre reference " << i;
            if (std::isfinite(expected_grad_h_pre[i])) {
                ASSERT_TRUE(std::isfinite(actual_grad_h_pre[i])) << "grad_h_pre " << i;
                detects_zero_grad_h_pre |=
                    std::abs(expected_grad_h_pre[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_grad_h_pre[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_grad_h_pre[i], expected_grad_h_pre[i]) << "grad_h_pre " << i;
            }
        }
        EXPECT_TRUE(detects_zero_grad_h_pre) << "grad_h_pre reference cannot reject an all-zero result";
        expectNear(actual_grad_h_pre, expected_grad_h_pre, kRelTolerance, kAbsTolerance, "grad_h_pre");
    }
}

INSTANTIATE_TEST_SUITE_P(
    HAggregateBwdShapeGrid, HAggregateBwdNumericsTest,
    testing::Combine(testing::Values(ShapeParam{8, 2, 32}, ShapeParam{33, 4, 64}, ShapeParam{128, 4, 128}),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16)),
    [](const testing::TestParamInfo<HAggregateBwdNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).tokens) + "x" + std::to_string(std::get<0>(info.param).streams) +
               "x" + std::to_string(std::get<0>(info.param).width) + "_type" + std::to_string(std::get<1>(info.param));
    });
}  // namespace
