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
#include "proj_rms_fwd_staged_fixture.hpp"

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

class ProjRmsFwdNumericsTest
    : public ftrain_test::ProjRmsFwdStagedTest,
      public testing::WithParamInterface<std::tuple<ShapeParam, FTrainNumericType, FTrainNumericType>> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kFp32RelTolerance = 2e-4;
constexpr double kFp32AbsTolerance = 2e-5;

TEST_P(ProjRmsFwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    RecordProperty("family", "proj_rms_fwd");
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

    std::vector<float> x(static_cast<std::size_t>((shape.rows) * (shape.reduction)));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = elementValue(i + 7919);
    std::vector<float> weight(static_cast<std::size_t>((shape.outputs) * (shape.reduction)));
    for (std::size_t i = 0; i < weight.size(); ++i) weight[i] = elementValue(i + 15838);
    std::vector<float> proj(static_cast<std::size_t>((shape.rows) * (shape.outputs)));
    std::fill(proj.begin(), proj.end(), 123);
    std::vector<float> r(static_cast<std::size_t>((shape.rows) * (1)));
    std::fill(r.begin(), r.end(), 123);
    float eps = 1e-6;

    for (auto* values : {&x, &proj, &r}) { roundToStorage(*values, dtype); }
    for (auto* values : {&weight}) { roundToStorage(*values, secondary_dtype); }

    auto expected_proj = proj;
    auto expected_r    = r;
    hostProjRmsFwd(shape, x, weight, expected_proj, expected_r, eps);

    roundToStorage(expected_proj, dtype);
    roundToStorage(expected_r, dtype);

    const std::int64_t x_dims[]{shape.rows, shape.reduction};
    FloatingDeviceBuffer device_x(x.size(), dtype);
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t weight_dims[]{shape.outputs, shape.reduction};
    FloatingDeviceBuffer device_weight(weight.size(), secondary_dtype);
    ASSERT_NE(device_weight.get(), nullptr);
    const std::int64_t proj_dims[]{shape.rows, shape.outputs};
    FloatingDeviceBuffer device_proj(proj.size(), dtype);
    ASSERT_NE(device_proj.get(), nullptr);
    const std::int64_t r_dims[]{shape.rows, 1};
    FloatingDeviceBuffer device_r(r.size(), dtype);
    ASSERT_NE(device_r.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_x.get(), x_dims, 2, dtype),
                        makeView(device_weight.get(), weight_dims, 2, secondary_dtype),
                        makeView(device_proj.get(), proj_dims, 2, dtype), makeView(device_r.get(), r_dims, 2, dtype),
                        makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        compute_type));
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
        device_x.upload(x, stream_);
        device_weight.upload(weight, stream_);
        device_proj.upload(proj, stream_);
        device_r.upload(r, stream_);
        if (convenience) {
            ASSERT_EQ(ftrainProjRmsFwd(
                          makeView(device_x.get(), x_dims, 2, dtype),
                          makeView(device_weight.get(), weight_dims, 2, secondary_dtype),
                          makeView(device_proj.get(), proj_dims, 2, dtype), makeView(device_r.get(), r_dims, 2, dtype),
                          makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                          compute_type, workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, primitive, workspace, workspace_sizes[primitive], stream_),
                      FTRAIN_STATUS_SUCCESS);
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

INSTANTIATE_TEST_SUITE_P(
    ProjRmsFwdShapeGrid, ProjRmsFwdNumericsTest,
    testing::Combine(testing::Values(ShapeParam{16, 8, 32}, ShapeParam{33, 16, 64}, ShapeParam{128, 24, 128}),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16)),
    [](const testing::TestParamInfo<ProjRmsFwdNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).rows) + "x" + std::to_string(std::get<0>(info.param).outputs) +
               "x" + std::to_string(std::get<0>(info.param).reduction) + "_type" +
               std::to_string(std::get<1>(info.param)) + "_secondary" + std::to_string(std::get<2>(info.param));
    });
}  // namespace
