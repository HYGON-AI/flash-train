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
#include "dense_indexer_score_staged_fixture.hpp"

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
    std::int64_t heads;
    std::int64_t head_dim;
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

void hostDenseIndexerScore(const ShapeParam& shape, const std::vector<float>& q, const std::vector<float>& k,
                           const std::vector<float>& w, std::vector<float>& out, std::vector<float>& denom,
                           std::int64_t ratio, float sm_scale) {
    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::int64_t col = 0; col < shape.tokens; ++col) {
            float score = 0;
            for (std::int64_t h = 0; h < shape.heads; ++h) {
                float dot = 0;
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += q[(row * shape.heads + h) * shape.head_dim + d] * k[col * shape.head_dim + d];
                score += std::max(dot * sm_scale, 0.0F) * w[row * shape.heads + h];
            }
            out[row * shape.tokens + col] = col <= row ? score : -std::numeric_limits<float>::infinity();
            maximum                       = std::max(maximum, out[row * shape.tokens + col]);
        }
        float sum = 0;
        for (std::int64_t col = 0; col <= row; ++col) sum += std::exp(out[row * shape.tokens + col] - maximum);
        denom[row] = maximum + std::log(sum);
    }
}

class DenseIndexerScoreNumericsTest
    : public ftrain_test::DenseIndexerScoreStagedTest,
      public testing::WithParamInterface<std::tuple<ShapeParam, FTrainNumericType, FTrainNumericType>> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kFp32RelTolerance = 2e-4;
constexpr double kFp32AbsTolerance = 2e-5;

TEST_P(DenseIndexerScoreNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    RecordProperty("family", "dense_indexer_score");
    std::string executed_primitives;
    const FTrainNumericType dtype           = std::get<1>(GetParam());
    const FTrainNumericType secondary_dtype = std::get<2>(GetParam());
    const FTrainNumericType compute_type    = dtype == secondary_dtype ? dtype : FTRAIN_NUMERIC_TYPE_FP32;
    const ShapeParam shape                  = std::get<0>(GetParam());
    // Inputs are rounded before the host oracle; all score outputs remain FP32.
    const double kRelTolerance = kFp32RelTolerance;
    const double kAbsTolerance = kFp32AbsTolerance;

    std::vector<float> q(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads) * (shape.head_dim)));
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = elementValue(i + 7919);
    std::vector<float> k(static_cast<std::size_t>((1) * (shape.tokens) * (1) * (shape.head_dim)));
    for (std::size_t i = 0; i < k.size(); ++i) k[i] = elementValue(i + 15838);
    std::vector<float> w(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads)));
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = elementValue(i + 23757);
    std::vector<float> out(static_cast<std::size_t>((1) * (shape.tokens) * (shape.tokens)));
    std::fill(out.begin(), out.end(), 123);
    std::vector<float> denom(static_cast<std::size_t>((1) * (shape.tokens)));
    std::fill(denom.begin(), denom.end(), 123);
    std::int64_t ratio = 1;
    float sm_scale     = 0.25;

    for (auto* values : {&q, &w}) { roundToStorage(*values, dtype); }
    for (auto* values : {&k}) { roundToStorage(*values, secondary_dtype); }

    auto expected_out   = out;
    auto expected_denom = denom;
    hostDenseIndexerScore(shape, q, k, w, expected_out, expected_denom, ratio, sm_scale);

    const std::int64_t q_dims[]{1, shape.tokens, shape.heads, shape.head_dim};
    FloatingDeviceBuffer device_q(q.size(), dtype);
    ASSERT_NE(device_q.get(), nullptr);
    const std::int64_t k_dims[]{1, shape.tokens, 1, shape.head_dim};
    FloatingDeviceBuffer device_k(k.size(), secondary_dtype);
    ASSERT_NE(device_k.get(), nullptr);
    const std::int64_t w_dims[]{1, shape.tokens, shape.heads};
    FloatingDeviceBuffer device_w(w.size(), dtype);
    ASSERT_NE(device_w.get(), nullptr);
    const std::int64_t out_dims[]{1, shape.tokens, shape.tokens};
    DeviceBuffer<float> device_out(out.size());
    ASSERT_NE(device_out.get(), nullptr);
    const std::int64_t denom_dims[]{1, shape.tokens};
    DeviceBuffer<float> device_denom(denom.size());
    ASSERT_NE(device_denom.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, makeView(device_q.get(), q_dims, 4, dtype), makeView(device_k.get(), k_dims, 4, secondary_dtype),
        makeView(device_w.get(), w_dims, 3, dtype), makeView(device_out.get(), out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(device_denom.get(), denom_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
        makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true), compute_type));
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
        device_q.upload(q, stream_);
        device_k.upload(k, stream_);
        device_w.upload(w, stream_);
        device_out.upload(out, stream_);
        device_denom.upload(denom, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainDenseIndexerScore(
                    makeView(device_q.get(), q_dims, 4, dtype), makeView(device_k.get(), k_dims, 4, secondary_dtype),
                    makeView(device_w.get(), w_dims, 3, dtype),
                    makeView(device_out.get(), out_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_denom.get(), denom_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    compute_type, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, primitive, workspace, workspace_sizes[primitive], stream_),
                      FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_out   = device_out.download(stream_);
        const auto actual_denom = device_denom.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_out.size(), expected_out.size());
        bool detects_zero_out = false;
        for (std::size_t i = 0; i < expected_out.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_out[i])) << "out reference " << i;
            if (std::isfinite(expected_out[i])) {
                ASSERT_TRUE(std::isfinite(actual_out[i])) << "out " << i;
                detects_zero_out |=
                    std::abs(expected_out[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_out[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_out[i], expected_out[i]) << "out " << i;
            }
        }
        EXPECT_TRUE(detects_zero_out) << "out reference cannot reject an all-zero result";
        expectNear(actual_out, expected_out, kRelTolerance, kAbsTolerance, "out");
        ASSERT_EQ(actual_denom.size(), expected_denom.size());
        bool detects_zero_denom = false;
        for (std::size_t i = 0; i < expected_denom.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_denom[i])) << "denom reference " << i;
            if (std::isfinite(expected_denom[i])) {
                ASSERT_TRUE(std::isfinite(actual_denom[i])) << "denom " << i;
                detects_zero_denom |=
                    std::abs(expected_denom[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_denom[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_denom[i], expected_denom[i]) << "denom " << i;
            }
        }
        EXPECT_TRUE(detects_zero_denom) << "denom reference cannot reject an all-zero result";
        expectNear(actual_denom, expected_denom, kRelTolerance, kAbsTolerance, "denom");
    }
}

INSTANTIATE_TEST_SUITE_P(
    DenseIndexerScoreShapeGrid, DenseIndexerScoreNumericsTest,
    testing::Combine(testing::Values(ShapeParam{8, 2, 32}, ShapeParam{33, 2, 64}, ShapeParam{128, 4, 128},
                                     ShapeParam{17, 32, 128}),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16)),
    [](const testing::TestParamInfo<DenseIndexerScoreNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).tokens) + "x" + std::to_string(std::get<0>(info.param).heads) +
               "x" + std::to_string(std::get<0>(info.param).head_dim) + "_type" +
               std::to_string(std::get<1>(info.param)) + "_secondary" + std::to_string(std::get<2>(info.param));
    });
}  // namespace
