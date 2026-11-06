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
#include "h_post_bda_bwd_staged_fixture.hpp"

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

void hostHPostBdaBwd(const ShapeParam& shape, const std::vector<float>& grad_output, const std::vector<float>& residual,
                     const std::vector<float>& x, const std::vector<float>& bias, const std::vector<float>& h_res,
                     const std::vector<float>& h_post, std::vector<float>& grad_residual, std::vector<float>& grad_x,
                     std::vector<float>& grad_h_res, std::vector<float>& grad_h_post, std::vector<float>& grad_bias) {
    std::fill(grad_x.begin(), grad_x.end(), 0);
    std::fill(grad_bias.begin(), grad_bias.end(), 0);
    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        for (std::int64_t i = 0; i < shape.streams; ++i)
            for (std::int64_t d = 0; d < shape.width; ++d) {
                float sum = 0;
                for (std::int64_t j = 0; j < shape.streams; ++j)
                    sum += h_res[(row * shape.streams + i) * shape.streams + j] *
                           grad_output[(row * shape.streams + j) * shape.width + d];
                grad_residual[(row * shape.streams + i) * shape.width + d] = sum;
            }
        for (std::int64_t i = 0; i < shape.streams; ++i)
            for (std::int64_t j = 0; j < shape.streams; ++j) {
                float sum = 0;
                for (std::int64_t d = 0; d < shape.width; ++d)
                    sum += residual[(row * shape.streams + i) * shape.width + d] *
                           grad_output[(row * shape.streams + j) * shape.width + d];
                grad_h_res[(row * shape.streams + i) * shape.streams + j] = sum;
            }
        for (std::int64_t j = 0; j < shape.streams; ++j) {
            float sum = 0;
            for (std::int64_t d = 0; d < shape.width; ++d) {
                const float g                  = grad_output[(row * shape.streams + j) * shape.width + d];
                sum                           += g * (x[row * shape.width + d] + bias[d]);
                grad_x[row * shape.width + d] += g * h_post[row * shape.streams + j];
                grad_bias[d]                  += g * h_post[row * shape.streams + j];
            }
            grad_h_post[row * shape.streams + j] = sum;
        }
    }
}

class HPostBdaBwdNumericsTest : public ftrain_test::HPostBdaBwdStagedTest,
                                public testing::WithParamInterface<std::tuple<ShapeParam, FTrainNumericType, bool>> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kFp32RelTolerance = 2e-4;
constexpr double kFp32AbsTolerance = 2e-5;

TEST_P(HPostBdaBwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    RecordProperty("family", "h_post_bda_bwd");
    std::string executed_primitives;
    const FTrainNumericType dtype = std::get<1>(GetParam());
    const auto mapping_dtype      = std::get<2>(GetParam()) ? FTRAIN_NUMERIC_TYPE_FP32 : dtype;
    const ShapeParam shape        = std::get<0>(GetParam());
    // Low-precision storage rounding; FP32 retains its original reduction budget.
    const double kRelTolerance = dtype == FTRAIN_NUMERIC_TYPE_FP32   ? kFp32RelTolerance
                                 : dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-2
                                                                     : 5e-2;
    const double kAbsTolerance = dtype == FTRAIN_NUMERIC_TYPE_FP32   ? kFp32AbsTolerance
                                 : dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-3
                                                                     : 5e-3;

    std::vector<float> grad_output(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.width)));
    for (std::size_t i = 0; i < grad_output.size(); ++i) grad_output[i] = elementValue(i + 7919);
    std::vector<float> residual(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.width)));
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = elementValue(i + 15838);
    std::vector<float> x(static_cast<std::size_t>((shape.tokens) * (1) * (shape.width)));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = elementValue(i + 23757);
    std::vector<float> bias(static_cast<std::size_t>((shape.width)));
    for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = elementValue(i + 31676);
    std::vector<float> h_res(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.streams)));
    for (std::size_t i = 0; i < h_res.size(); ++i) h_res[i] = elementValue(i + 39595);
    std::vector<float> h_post(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams)));
    for (std::size_t i = 0; i < h_post.size(); ++i) h_post[i] = elementValue(i + 47514);
    std::vector<float> grad_residual(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.width)));
    std::fill(grad_residual.begin(), grad_residual.end(), 123);
    std::vector<float> grad_x(static_cast<std::size_t>((shape.tokens) * (1) * (shape.width)));
    std::fill(grad_x.begin(), grad_x.end(), 123);
    std::vector<float> grad_h_res(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams) * (shape.streams)));
    std::fill(grad_h_res.begin(), grad_h_res.end(), 123);
    std::vector<float> grad_h_post(static_cast<std::size_t>((shape.tokens) * (1) * (shape.streams)));
    std::fill(grad_h_post.begin(), grad_h_post.end(), 123);
    std::vector<float> grad_bias(static_cast<std::size_t>((shape.width)));
    std::fill(grad_bias.begin(), grad_bias.end(), 123);

    for (auto* values : {&grad_output, &residual, &x, &grad_residual, &grad_x}) { roundToStorage(*values, dtype); }
    for (auto* values : {&h_res, &h_post, &grad_h_res, &grad_h_post}) { roundToStorage(*values, mapping_dtype); }

    auto expected_grad_residual = grad_residual;
    auto expected_grad_x        = grad_x;
    auto expected_grad_h_res    = grad_h_res;
    auto expected_grad_h_post   = grad_h_post;
    auto expected_grad_bias     = grad_bias;
    hostHPostBdaBwd(shape, grad_output, residual, x, bias, h_res, h_post, expected_grad_residual, expected_grad_x,
                    expected_grad_h_res, expected_grad_h_post, expected_grad_bias);

    roundToStorage(expected_grad_residual, dtype);
    roundToStorage(expected_grad_x, dtype);
    roundToStorage(expected_grad_h_res, mapping_dtype);
    roundToStorage(expected_grad_h_post, mapping_dtype);

    const std::int64_t grad_output_dims[]{shape.tokens, 1, shape.streams, shape.width};
    FloatingDeviceBuffer device_grad_output(grad_output.size(), dtype);
    ASSERT_NE(device_grad_output.get(), nullptr);
    const std::int64_t residual_dims[]{shape.tokens, 1, shape.streams, shape.width};
    FloatingDeviceBuffer device_residual(residual.size(), dtype);
    ASSERT_NE(device_residual.get(), nullptr);
    const std::int64_t x_dims[]{shape.tokens, 1, shape.width};
    FloatingDeviceBuffer device_x(x.size(), dtype);
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t bias_dims[]{shape.width};
    DeviceBuffer<float> device_bias(bias.size());
    ASSERT_NE(device_bias.get(), nullptr);
    const std::int64_t h_res_dims[]{shape.tokens, 1, shape.streams, shape.streams};
    FloatingDeviceBuffer device_h_res(h_res.size(), mapping_dtype);
    ASSERT_NE(device_h_res.get(), nullptr);
    const std::int64_t h_post_dims[]{shape.tokens, 1, shape.streams};
    FloatingDeviceBuffer device_h_post(h_post.size(), mapping_dtype);
    ASSERT_NE(device_h_post.get(), nullptr);
    const std::int64_t grad_residual_dims[]{shape.tokens, 1, shape.streams, shape.width};
    FloatingDeviceBuffer device_grad_residual(grad_residual.size(), dtype);
    ASSERT_NE(device_grad_residual.get(), nullptr);
    const std::int64_t grad_x_dims[]{shape.tokens, 1, shape.width};
    FloatingDeviceBuffer device_grad_x(grad_x.size(), dtype);
    ASSERT_NE(device_grad_x.get(), nullptr);
    const std::int64_t grad_h_res_dims[]{shape.tokens, 1, shape.streams, shape.streams};
    FloatingDeviceBuffer device_grad_h_res(grad_h_res.size(), mapping_dtype);
    ASSERT_NE(device_grad_h_res.get(), nullptr);
    const std::int64_t grad_h_post_dims[]{shape.tokens, 1, shape.streams};
    FloatingDeviceBuffer device_grad_h_post(grad_h_post.size(), mapping_dtype);
    ASSERT_NE(device_grad_h_post.get(), nullptr);
    const std::int64_t grad_bias_dims[]{shape.width};
    DeviceBuffer<float> device_grad_bias(grad_bias.size());
    ASSERT_NE(device_grad_bias.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_grad_output.get(), grad_output_dims, 4, dtype),
                        makeView(device_residual.get(), residual_dims, 4, dtype),
                        makeView(device_x.get(), x_dims, 3, dtype),
                        makeView(device_bias.get(), bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_h_res.get(), h_res_dims, 4, mapping_dtype),
                        makeView(device_h_post.get(), h_post_dims, 3, mapping_dtype),
                        makeView(device_grad_residual.get(), grad_residual_dims, 4, dtype),
                        makeView(device_grad_x.get(), grad_x_dims, 3, dtype),
                        makeView(device_grad_h_res.get(), grad_h_res_dims, 4, mapping_dtype),
                        makeView(device_grad_h_post.get(), grad_h_post_dims, 3, mapping_dtype),
                        makeView(device_grad_bias.get(), grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32), dtype));
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
        device_residual.upload(residual, stream_);
        device_x.upload(x, stream_);
        device_bias.upload(bias, stream_);
        device_h_res.upload(h_res, stream_);
        device_h_post.upload(h_post, stream_);
        device_grad_residual.upload(grad_residual, stream_);
        device_grad_x.upload(grad_x, stream_);
        device_grad_h_res.upload(grad_h_res, stream_);
        device_grad_h_post.upload(grad_h_post, stream_);
        device_grad_bias.upload(grad_bias, stream_);
        if (convenience) {
            ASSERT_EQ(ftrainHPostBdaBwd(makeView(device_grad_output.get(), grad_output_dims, 4, dtype),
                                        makeView(device_residual.get(), residual_dims, 4, dtype),
                                        makeView(device_x.get(), x_dims, 3, dtype),
                                        makeView(device_bias.get(), bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                                        makeView(device_h_res.get(), h_res_dims, 4, mapping_dtype),
                                        makeView(device_h_post.get(), h_post_dims, 3, mapping_dtype),
                                        makeView(device_grad_residual.get(), grad_residual_dims, 4, dtype),
                                        makeView(device_grad_x.get(), grad_x_dims, 3, dtype),
                                        makeView(device_grad_h_res.get(), grad_h_res_dims, 4, mapping_dtype),
                                        makeView(device_grad_h_post.get(), grad_h_post_dims, 3, mapping_dtype),
                                        makeView(device_grad_bias.get(), grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                                        dtype, workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, primitive, workspace, workspace_sizes[primitive], stream_),
                      FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_grad_residual = device_grad_residual.download(stream_);
        const auto actual_grad_x        = device_grad_x.download(stream_);
        const auto actual_grad_h_res    = device_grad_h_res.download(stream_);
        const auto actual_grad_h_post   = device_grad_h_post.download(stream_);
        const auto actual_grad_bias     = device_grad_bias.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_grad_residual.size(), expected_grad_residual.size());
        bool detects_zero_grad_residual = false;
        for (std::size_t i = 0; i < expected_grad_residual.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_grad_residual[i])) << "grad_residual reference " << i;
            if (std::isfinite(expected_grad_residual[i])) {
                ASSERT_TRUE(std::isfinite(actual_grad_residual[i])) << "grad_residual " << i;
                detects_zero_grad_residual |= std::abs(expected_grad_residual[i]) >
                                              kAbsTolerance + kRelTolerance * std::abs(expected_grad_residual[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_grad_residual[i], expected_grad_residual[i]) << "grad_residual " << i;
            }
        }
        EXPECT_TRUE(detects_zero_grad_residual) << "grad_residual reference cannot reject an all-zero result";
        expectNear(actual_grad_residual, expected_grad_residual, kRelTolerance, kAbsTolerance, "grad_residual");
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
        ASSERT_EQ(actual_grad_h_res.size(), expected_grad_h_res.size());
        bool detects_zero_grad_h_res = false;
        for (std::size_t i = 0; i < expected_grad_h_res.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_grad_h_res[i])) << "grad_h_res reference " << i;
            if (std::isfinite(expected_grad_h_res[i])) {
                ASSERT_TRUE(std::isfinite(actual_grad_h_res[i])) << "grad_h_res " << i;
                detects_zero_grad_h_res |=
                    std::abs(expected_grad_h_res[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_grad_h_res[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_grad_h_res[i], expected_grad_h_res[i]) << "grad_h_res " << i;
            }
        }
        EXPECT_TRUE(detects_zero_grad_h_res) << "grad_h_res reference cannot reject an all-zero result";
        expectNear(actual_grad_h_res, expected_grad_h_res, kRelTolerance, kAbsTolerance, "grad_h_res");
        ASSERT_EQ(actual_grad_h_post.size(), expected_grad_h_post.size());
        bool detects_zero_grad_h_post = false;
        for (std::size_t i = 0; i < expected_grad_h_post.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_grad_h_post[i])) << "grad_h_post reference " << i;
            if (std::isfinite(expected_grad_h_post[i])) {
                ASSERT_TRUE(std::isfinite(actual_grad_h_post[i])) << "grad_h_post " << i;
                detects_zero_grad_h_post |= std::abs(expected_grad_h_post[i]) >
                                            kAbsTolerance + kRelTolerance * std::abs(expected_grad_h_post[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_grad_h_post[i], expected_grad_h_post[i]) << "grad_h_post " << i;
            }
        }
        EXPECT_TRUE(detects_zero_grad_h_post) << "grad_h_post reference cannot reject an all-zero result";
        expectNear(actual_grad_h_post, expected_grad_h_post, kRelTolerance, kAbsTolerance, "grad_h_post");
        ASSERT_EQ(actual_grad_bias.size(), expected_grad_bias.size());
        bool detects_zero_grad_bias = false;
        for (std::size_t i = 0; i < expected_grad_bias.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_grad_bias[i])) << "grad_bias reference " << i;
            if (std::isfinite(expected_grad_bias[i])) {
                ASSERT_TRUE(std::isfinite(actual_grad_bias[i])) << "grad_bias " << i;
                detects_zero_grad_bias |=
                    std::abs(expected_grad_bias[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_grad_bias[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_grad_bias[i], expected_grad_bias[i]) << "grad_bias " << i;
            }
        }
        EXPECT_TRUE(detects_zero_grad_bias) << "grad_bias reference cannot reject an all-zero result";
        expectNear(actual_grad_bias, expected_grad_bias, kRelTolerance, kAbsTolerance, "grad_bias");
    }
}

INSTANTIATE_TEST_SUITE_P(
    HPostBdaBwdShapeGrid, HPostBdaBwdNumericsTest,
    testing::Combine(testing::Values(ShapeParam{3, 3, 17}, ShapeParam{8, 2, 32}, ShapeParam{33, 4, 64},
                                     ShapeParam{128, 4, 128}, ShapeParam{7, 4, 512}, ShapeParam{3, 8, 512}),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16),
                     testing::Bool()),
    [](const testing::TestParamInfo<HPostBdaBwdNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).tokens) + "x" + std::to_string(std::get<0>(info.param).streams) +
               "x" + std::to_string(std::get<0>(info.param).width) + "_type" + std::to_string(std::get<1>(info.param)) +
               "_fp32mapping" + std::to_string(std::get<2>(info.param));
    });
}  // namespace
