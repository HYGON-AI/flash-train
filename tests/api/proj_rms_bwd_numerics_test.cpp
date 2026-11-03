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
#include "proj_rms_bwd_staged_fixture.hpp"

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

class ProjRmsBwdNumericsTest
    : public ftrain_test::ProjRmsBwdStagedTest,
      public testing::WithParamInterface<std::tuple<ShapeParam, FTrainNumericType, FTrainNumericType>> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kFp32RelTolerance = 2e-4;
constexpr double kFp32AbsTolerance = 2e-5;

TEST_P(ProjRmsBwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    RecordProperty("family", "proj_rms_bwd");
    std::string executed_primitives;
    const FTrainNumericType dtype           = std::get<1>(GetParam());
    const FTrainNumericType secondary_dtype = std::get<2>(GetParam());
    const FTrainNumericType compute_type    = dtype == secondary_dtype ? dtype : FTRAIN_NUMERIC_TYPE_FP32;
    const ShapeParam shape                  = std::get<0>(GetParam());
    // Low-precision storage rounding; FP32 retains its original reduction budget.
    const double kRelTolerance = dtype == FTRAIN_NUMERIC_TYPE_FP32   ? kFp32RelTolerance
                                 : dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-2
                                                                     : 5e-2;
    const double kAbsTolerance = dtype == FTRAIN_NUMERIC_TYPE_FP32   ? kFp32AbsTolerance
                                 : dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-3
                                                                     : 5e-3;

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

    for (auto* values : {&grad_proj, &grad_r, &x, &r, &grad_x}) { roundToStorage(*values, dtype); }
    for (auto* values : {&weight, &grad_weight}) { roundToStorage(*values, secondary_dtype); }

    for (std::int64_t row = 0; row < shape.rows; ++row) {
        float sum = 0;
        for (std::int64_t d = 0; d < shape.reduction; ++d)
            sum += x[row * shape.reduction + d] * x[row * shape.reduction + d];
        r[row] = 1.0F / (std::sqrt(sum / shape.reduction) + eps);
    }

    for (auto* values : {&grad_proj, &grad_r, &x, &r, &grad_x}) { roundToStorage(*values, dtype); }
    for (auto* values : {&weight, &grad_weight}) { roundToStorage(*values, secondary_dtype); }

    auto expected_grad_x      = grad_x;
    auto expected_grad_weight = grad_weight;
    hostProjRmsBwd(shape, grad_proj, grad_r, x, weight, r, expected_grad_x, expected_grad_weight, eps);

    roundToStorage(expected_grad_x, dtype);
    roundToStorage(expected_grad_weight, secondary_dtype);

    const std::int64_t grad_proj_dims[]{shape.rows, shape.outputs};
    FloatingDeviceBuffer device_grad_proj(grad_proj.size(), dtype);
    ASSERT_NE(device_grad_proj.get(), nullptr);
    const std::int64_t grad_r_dims[]{shape.rows, 1};
    FloatingDeviceBuffer device_grad_r(grad_r.size(), dtype);
    ASSERT_NE(device_grad_r.get(), nullptr);
    const std::int64_t x_dims[]{shape.rows, shape.reduction};
    FloatingDeviceBuffer device_x(x.size(), dtype);
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t weight_dims[]{shape.outputs, shape.reduction};
    FloatingDeviceBuffer device_weight(weight.size(), secondary_dtype);
    ASSERT_NE(device_weight.get(), nullptr);
    const std::int64_t r_dims[]{shape.rows, 1};
    FloatingDeviceBuffer device_r(r.size(), dtype);
    ASSERT_NE(device_r.get(), nullptr);
    const std::int64_t grad_x_dims[]{shape.rows, shape.reduction};
    FloatingDeviceBuffer device_grad_x(grad_x.size(), dtype);
    ASSERT_NE(device_grad_x.get(), nullptr);
    const std::int64_t grad_weight_dims[]{shape.outputs, shape.reduction};
    FloatingDeviceBuffer device_grad_weight(grad_weight.size(), secondary_dtype);
    ASSERT_NE(device_grad_weight.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, makeView(device_grad_proj.get(), grad_proj_dims, 2, dtype),
        makeView(device_grad_r.get(), grad_r_dims, 2, dtype), makeView(device_x.get(), x_dims, 2, dtype),
        makeView(device_weight.get(), weight_dims, 2, secondary_dtype), makeView(device_r.get(), r_dims, 2, dtype),
        makeView(device_grad_x.get(), grad_x_dims, 2, dtype),
        makeView(device_grad_weight.get(), grad_weight_dims, 2, secondary_dtype),
        makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true), compute_type));
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
        device_grad_proj.upload(grad_proj, stream_);
        device_grad_r.upload(grad_r, stream_);
        device_x.upload(x, stream_);
        device_weight.upload(weight, stream_);
        device_r.upload(r, stream_);
        device_grad_x.upload(grad_x, stream_);
        device_grad_weight.upload(grad_weight, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainProjRmsBwd(
                    makeView(device_grad_proj.get(), grad_proj_dims, 2, dtype),
                    makeView(device_grad_r.get(), grad_r_dims, 2, dtype), makeView(device_x.get(), x_dims, 2, dtype),
                    makeView(device_weight.get(), weight_dims, 2, secondary_dtype),
                    makeView(device_r.get(), r_dims, 2, dtype), makeView(device_grad_x.get(), grad_x_dims, 2, dtype),
                    makeView(device_grad_weight.get(), grad_weight_dims, 2, secondary_dtype),
                    makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    compute_type, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, primitive, workspace, workspace_sizes[primitive], stream_),
                      FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_grad_x      = device_grad_x.download(stream_);
        const auto actual_grad_weight = device_grad_weight.download(stream_);
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
        const double secondary_rel_tolerance = secondary_dtype == FTRAIN_NUMERIC_TYPE_BF16   ? 5e-2
                                               : secondary_dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-2
                                                                                             : kRelTolerance;
        const double secondary_abs_tolerance = secondary_dtype == FTRAIN_NUMERIC_TYPE_BF16   ? 5e-3
                                               : secondary_dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-3
                                                                                             : kAbsTolerance;
        ASSERT_EQ(actual_grad_weight.size(), expected_grad_weight.size());
        bool detects_zero_grad_weight = false;
        for (std::size_t i = 0; i < expected_grad_weight.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_grad_weight[i])) << "grad_weight reference " << i;
            if (std::isfinite(expected_grad_weight[i])) {
                ASSERT_TRUE(std::isfinite(actual_grad_weight[i])) << "grad_weight " << i;
                detects_zero_grad_weight |=
                    std::abs(expected_grad_weight[i]) >
                    secondary_abs_tolerance + secondary_rel_tolerance * std::abs(expected_grad_weight[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_grad_weight[i], expected_grad_weight[i]) << "grad_weight " << i;
            }
        }
        EXPECT_TRUE(detects_zero_grad_weight) << "grad_weight reference cannot reject an all-zero result";
        expectNear(actual_grad_weight, expected_grad_weight, secondary_rel_tolerance, secondary_abs_tolerance,
                   "grad_weight");
    }
}

INSTANTIATE_TEST_SUITE_P(
    ProjRmsBwdShapeGrid, ProjRmsBwdNumericsTest,
    testing::Combine(testing::Values(ShapeParam{16, 8, 32}, ShapeParam{33, 16, 64}, ShapeParam{128, 24, 128},
                                     ShapeParam{17, 8, 512}),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16)),
    [](const testing::TestParamInfo<ProjRmsBwdNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).rows) + "x" + std::to_string(std::get<0>(info.param).outputs) +
               "x" + std::to_string(std::get<0>(info.param).reduction) + "_type" +
               std::to_string(std::get<1>(info.param)) + "_secondary" + std::to_string(std::get<2>(info.param));
    });
}  // namespace
