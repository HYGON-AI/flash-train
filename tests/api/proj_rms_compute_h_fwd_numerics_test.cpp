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
#include "proj_rms_compute_h_fwd_staged_fixture.hpp"

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
    std::int64_t streams;
    std::int64_t reduction;
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

void hostProjRmsComputeHFwd(const ShapeParam& shape, const std::vector<float>& x, const std::vector<float>& weight,
                            const std::vector<float>& alpha_pre, const std::vector<float>& alpha_post,
                            const std::vector<float>& alpha_res, const std::vector<float>& bias,
                            std::vector<float>& h_pre, std::vector<float>& h_post, std::vector<float>& h_res,
                            std::vector<float>& r, std::vector<float>& proj, std::int64_t n, float eps,
                            float compute_h_eps) {
    for (std::int64_t row = 0; row < shape.rows; ++row) {
        float squared = 0;
        for (std::int64_t d = 0; d < shape.reduction; ++d)
            squared += x[row * shape.reduction + d] * x[row * shape.reduction + d];
        r[row] = std::sqrt(squared / shape.reduction);
        for (std::int64_t j = 0; j < (shape.streams * shape.streams + 2 * shape.streams); ++j) {
            float sum = 0;
            for (std::int64_t d = 0; d < shape.reduction; ++d)
                sum += x[row * shape.reduction + d] * weight[j * shape.reduction + d];
            proj[row * (shape.streams * shape.streams + 2 * shape.streams) + j] = sum;
        }

        for (std::int64_t i = 0; i < shape.streams; ++i) {
            h_pre[row * shape.streams + i] =
                1.0F / (1.0F + std::exp(-(proj[row * (shape.streams * shape.streams + 2 * shape.streams) + i] /
                                              (r[row] + eps) * alpha_pre[0] +
                                          bias[i]))) +
                compute_h_eps;
            h_post[row * shape.streams + i] =
                2.0F /
                (1.0F + std::exp(-(proj[row * (shape.streams * shape.streams + 2 * shape.streams) + shape.streams + i] /
                                       (r[row] + eps) * alpha_post[0] +
                                   bias[shape.streams + i])));
        }
        for (std::int64_t i = 0; i < shape.streams * shape.streams; ++i)
            h_res[row * shape.streams * shape.streams + i] =
                proj[row * (shape.streams * shape.streams + 2 * shape.streams) + 2 * shape.streams + i] /
                    (r[row] + eps) * alpha_res[0] +
                bias[2 * shape.streams + i];
    }
}

class ProjRmsComputeHFwdNumericsTest
    : public ftrain_test::ProjRmsComputeHFwdStagedTest,
      public testing::WithParamInterface<std::tuple<ShapeParam, FTrainNumericType, FTrainNumericType, bool>> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kFp32RelTolerance = 2e-4;
constexpr double kFp32AbsTolerance = 2e-5;

TEST_P(ProjRmsComputeHFwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    RecordProperty("family", "proj_rms_compute_h_fwd");
    std::string executed_primitives;
    const FTrainNumericType dtype           = std::get<1>(GetParam());
    const FTrainNumericType secondary_dtype = std::get<2>(GetParam());
    const FTrainNumericType compute_type    = dtype == secondary_dtype ? dtype : FTRAIN_NUMERIC_TYPE_FP32;
    const FTrainNumericType state_dtype     = std::get<3>(GetParam()) ? compute_type : dtype;
    const ShapeParam shape                  = std::get<0>(GetParam());
    // Low-precision storage rounding; FP32 retains its original reduction budget.
    const double kRelTolerance = state_dtype == FTRAIN_NUMERIC_TYPE_FP32   ? kFp32RelTolerance
                                 : state_dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-2
                                                                           : 5e-2;
    const double kAbsTolerance = state_dtype == FTRAIN_NUMERIC_TYPE_FP32   ? kFp32AbsTolerance
                                 : state_dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-3
                                                                           : 5e-3;

    std::vector<float> x(static_cast<std::size_t>((shape.rows) * (shape.reduction)));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = elementValue(i + 7919);
    std::vector<float> weight(
        static_cast<std::size_t>((shape.streams * shape.streams + 2 * shape.streams) * (shape.reduction)));
    for (std::size_t i = 0; i < weight.size(); ++i) weight[i] = elementValue(i + 15838);
    std::vector<float> alpha_pre(static_cast<std::size_t>((1)));
    std::fill(alpha_pre.begin(), alpha_pre.end(), 1.0F);
    std::vector<float> alpha_post(static_cast<std::size_t>((1)));
    std::fill(alpha_post.begin(), alpha_post.end(), 1.0F);
    std::vector<float> alpha_res(static_cast<std::size_t>((1)));
    std::fill(alpha_res.begin(), alpha_res.end(), 1.0F);
    std::vector<float> bias(static_cast<std::size_t>((shape.streams * shape.streams + 2 * shape.streams)));
    for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = elementValue(i + 47514);
    std::vector<float> h_pre(static_cast<std::size_t>((shape.rows) * (shape.streams)));
    std::fill(h_pre.begin(), h_pre.end(), 123);
    std::vector<float> h_post(static_cast<std::size_t>((shape.rows) * (shape.streams)));
    std::fill(h_post.begin(), h_post.end(), 123);
    std::vector<float> h_res(static_cast<std::size_t>((shape.rows) * (shape.streams * shape.streams)));
    std::fill(h_res.begin(), h_res.end(), 123);
    std::vector<float> r(static_cast<std::size_t>((shape.rows) * (1)));
    std::fill(r.begin(), r.end(), 123);
    std::vector<float> proj(
        static_cast<std::size_t>((shape.rows) * (shape.streams * shape.streams + 2 * shape.streams)));
    std::fill(proj.begin(), proj.end(), 123);
    std::int64_t n      = shape.streams;
    float eps           = 1e-6;
    float compute_h_eps = 1e-6;

    roundToStorage(x, dtype);
    for (auto* values : {&h_pre, &h_post, &h_res, &r, &proj}) { roundToStorage(*values, state_dtype); }
    for (auto* values : {&weight}) { roundToStorage(*values, secondary_dtype); }

    auto expected_h_pre  = h_pre;
    auto expected_h_post = h_post;
    auto expected_h_res  = h_res;
    auto expected_r      = r;
    auto expected_proj   = proj;
    hostProjRmsComputeHFwd(shape, x, weight, alpha_pre, alpha_post, alpha_res, bias, expected_h_pre, expected_h_post,
                           expected_h_res, expected_r, expected_proj, n, eps, compute_h_eps);

    roundToStorage(expected_h_pre, state_dtype);
    roundToStorage(expected_h_post, state_dtype);
    roundToStorage(expected_h_res, state_dtype);
    roundToStorage(expected_r, state_dtype);
    roundToStorage(expected_proj, state_dtype);

    const std::int64_t x_dims[]{shape.rows, shape.reduction};
    FloatingDeviceBuffer device_x(x.size(), dtype);
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t weight_dims[]{shape.streams * shape.streams + 2 * shape.streams, shape.reduction};
    FloatingDeviceBuffer device_weight(weight.size(), secondary_dtype);
    ASSERT_NE(device_weight.get(), nullptr);
    const std::int64_t alpha_pre_dims[]{1};
    DeviceBuffer<float> device_alpha_pre(alpha_pre.size());
    ASSERT_NE(device_alpha_pre.get(), nullptr);
    const std::int64_t alpha_post_dims[]{1};
    DeviceBuffer<float> device_alpha_post(alpha_post.size());
    ASSERT_NE(device_alpha_post.get(), nullptr);
    const std::int64_t alpha_res_dims[]{1};
    DeviceBuffer<float> device_alpha_res(alpha_res.size());
    ASSERT_NE(device_alpha_res.get(), nullptr);
    const std::int64_t bias_dims[]{shape.streams * shape.streams + 2 * shape.streams};
    DeviceBuffer<float> device_bias(bias.size());
    ASSERT_NE(device_bias.get(), nullptr);
    const std::int64_t h_pre_dims[]{shape.rows, shape.streams};
    FloatingDeviceBuffer device_h_pre(h_pre.size(), state_dtype);
    ASSERT_NE(device_h_pre.get(), nullptr);
    const std::int64_t h_post_dims[]{shape.rows, shape.streams};
    FloatingDeviceBuffer device_h_post(h_post.size(), state_dtype);
    ASSERT_NE(device_h_post.get(), nullptr);
    const std::int64_t h_res_dims[]{shape.rows, shape.streams * shape.streams};
    FloatingDeviceBuffer device_h_res(h_res.size(), state_dtype);
    ASSERT_NE(device_h_res.get(), nullptr);
    const std::int64_t r_dims[]{shape.rows, 1};
    FloatingDeviceBuffer device_r(r.size(), state_dtype);
    ASSERT_NE(device_r.get(), nullptr);
    const std::int64_t proj_dims[]{shape.rows, shape.streams * shape.streams + 2 * shape.streams};
    FloatingDeviceBuffer device_proj(proj.size(), state_dtype);
    ASSERT_NE(device_proj.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, makeView(device_x.get(), x_dims, 2, dtype),
        makeView(device_weight.get(), weight_dims, 2, secondary_dtype),
        makeView(device_alpha_pre.get(), alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(device_alpha_post.get(), alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(device_alpha_res.get(), alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(device_bias.get(), bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(device_h_pre.get(), h_pre_dims, 2, state_dtype),
        makeView(device_h_post.get(), h_post_dims, 2, state_dtype),
        makeView(device_h_res.get(), h_res_dims, 2, state_dtype), makeView(device_r.get(), r_dims, 2, state_dtype),
        makeView(device_proj.get(), proj_dims, 2, state_dtype),
        makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
        makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
        makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
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
        device_alpha_pre.upload(alpha_pre, stream_);
        device_alpha_post.upload(alpha_post, stream_);
        device_alpha_res.upload(alpha_res, stream_);
        device_bias.upload(bias, stream_);
        device_h_pre.upload(h_pre, stream_);
        device_h_post.upload(h_post, stream_);
        device_h_res.upload(h_res, stream_);
        device_r.upload(r, stream_);
        device_proj.upload(proj, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainProjRmsComputeHFwd(
                    makeView(device_x.get(), x_dims, 2, dtype),
                    makeView(device_weight.get(), weight_dims, 2, secondary_dtype),
                    makeView(device_alpha_pre.get(), alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_alpha_post.get(), alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_alpha_res.get(), alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_bias.get(), bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_h_pre.get(), h_pre_dims, 2, state_dtype),
                    makeView(device_h_post.get(), h_post_dims, 2, state_dtype),
                    makeView(device_h_res.get(), h_res_dims, 2, state_dtype),
                    makeView(device_r.get(), r_dims, 2, state_dtype),
                    makeView(device_proj.get(), proj_dims, 2, state_dtype),
                    makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    compute_type, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, primitive, workspace, workspace_sizes[primitive], stream_),
                      FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_h_pre  = device_h_pre.download(stream_);
        const auto actual_h_post = device_h_post.download(stream_);
        const auto actual_h_res  = device_h_res.download(stream_);
        const auto actual_r      = device_r.download(stream_);
        const auto actual_proj   = device_proj.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_h_pre.size(), expected_h_pre.size());
        bool detects_zero_h_pre = false;
        for (std::size_t i = 0; i < expected_h_pre.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_h_pre[i])) << "h_pre reference " << i;
            if (std::isfinite(expected_h_pre[i])) {
                ASSERT_TRUE(std::isfinite(actual_h_pre[i])) << "h_pre " << i;
                detects_zero_h_pre |=
                    std::abs(expected_h_pre[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_h_pre[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_h_pre[i], expected_h_pre[i]) << "h_pre " << i;
            }
        }
        EXPECT_TRUE(detects_zero_h_pre) << "h_pre reference cannot reject an all-zero result";
        expectNear(actual_h_pre, expected_h_pre, kRelTolerance, kAbsTolerance, "h_pre");
        ASSERT_EQ(actual_h_post.size(), expected_h_post.size());
        bool detects_zero_h_post = false;
        for (std::size_t i = 0; i < expected_h_post.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_h_post[i])) << "h_post reference " << i;
            if (std::isfinite(expected_h_post[i])) {
                ASSERT_TRUE(std::isfinite(actual_h_post[i])) << "h_post " << i;
                detects_zero_h_post |=
                    std::abs(expected_h_post[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_h_post[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_h_post[i], expected_h_post[i]) << "h_post " << i;
            }
        }
        EXPECT_TRUE(detects_zero_h_post) << "h_post reference cannot reject an all-zero result";
        expectNear(actual_h_post, expected_h_post, kRelTolerance, kAbsTolerance, "h_post");
        ASSERT_EQ(actual_h_res.size(), expected_h_res.size());
        bool detects_zero_h_res = false;
        for (std::size_t i = 0; i < expected_h_res.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_h_res[i])) << "h_res reference " << i;
            if (std::isfinite(expected_h_res[i])) {
                ASSERT_TRUE(std::isfinite(actual_h_res[i])) << "h_res " << i;
                detects_zero_h_res |=
                    std::abs(expected_h_res[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_h_res[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_h_res[i], expected_h_res[i]) << "h_res " << i;
            }
        }
        EXPECT_TRUE(detects_zero_h_res) << "h_res reference cannot reject an all-zero result";
        expectNear(actual_h_res, expected_h_res, kRelTolerance, kAbsTolerance, "h_res");
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
    }
}

INSTANTIATE_TEST_SUITE_P(
    ProjRmsComputeHFwdShapeGrid, ProjRmsComputeHFwdNumericsTest,
    testing::Combine(testing::Values(ShapeParam{16, 2, 32}, ShapeParam{33, 4, 64}, ShapeParam{128, 4, 128},
                                     ShapeParam{16, 2, 4096}),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16),
                     testing::Bool()),
    [](const testing::TestParamInfo<ProjRmsComputeHFwdNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).rows) + "x" + std::to_string(std::get<0>(info.param).streams) +
               "x" + std::to_string(std::get<0>(info.param).reduction) + "_type" +
               std::to_string(std::get<1>(info.param)) + "_secondary" + std::to_string(std::get<2>(info.param)) +
               "_promoted" + std::to_string(std::get<3>(info.param));
    });
}  // namespace
