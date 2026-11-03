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
#include "sparse_attn_target_staged_fixture.hpp"

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
    std::int64_t topk       = 0;
    std::int64_t valid_keys = 0;
    std::int64_t selectionWidth() const { return topk ? topk : std::min<std::int64_t>(tokens, 32); }
    std::int64_t validKeys() const { return valid_keys ? valid_keys : selectionWidth(); }
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

void hostSparseAttnTarget(const ShapeParam& shape, const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& lse, const std::vector<std::int32_t>& topk_indices,
                          std::vector<float>& target, float softmax_scale, std::uint8_t topk_indices_global) {
    const auto topk = shape.selectionWidth();
    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        float total = 0;
        for (std::int64_t j = 0; j < topk; ++j) {
            float value    = 0;
            const auto col = topk_indices[row * topk + j];
            if (col < 0) {
                target[row * topk + j] = 0;
                continue;
            }
            for (std::int64_t h = 0; h < shape.heads; ++h) {
                float dot = 0;
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += q[(row * shape.heads + h) * shape.head_dim + d] * k[col * shape.head_dim + d];
                value += std::exp(dot * softmax_scale - lse[row * shape.heads + h]);
            }
            target[row * topk + j]  = value;
            total                  += value;
        }
        for (std::int64_t j = 0; j < topk; ++j) target[row * topk + j] /= total;
    }
}

class SparseAttnTargetNumericsTest
    : public ftrain_test::SparseAttnTargetStagedTest,
      public testing::WithParamInterface<std::tuple<ShapeParam, FTrainNumericType, FTrainNumericType>> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kFp32RelTolerance = 2e-4;
constexpr double kFp32AbsTolerance = 2e-5;

TEST_P(SparseAttnTargetNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    RecordProperty("family", "sparse_attn_target");
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

    std::vector<float> q(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads) * (shape.head_dim)));
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = elementValue(i + 7919);
    std::vector<float> k(static_cast<std::size_t>((1) * (shape.tokens) * (shape.head_dim)));
    for (std::size_t i = 0; i < k.size(); ++i) k[i] = elementValue(i + 15838);
    std::vector<float> lse(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads)));
    for (std::size_t i = 0; i < lse.size(); ++i) lse[i] = elementValue(i + 23757);
    std::vector<std::int32_t> topk_indices(static_cast<std::size_t>((1) * (shape.tokens) * (shape.selectionWidth())));
    for (std::size_t i = 0; i < topk_indices.size(); ++i)
        topk_indices[i] = i % shape.selectionWidth() < static_cast<std::size_t>(shape.validKeys())
                              ? static_cast<std::int32_t>((i % shape.selectionWidth()) % shape.tokens)
                              : -1;
    std::vector<float> target(static_cast<std::size_t>((1) * (shape.tokens) * (shape.selectionWidth())));
    std::fill(target.begin(), target.end(), 123);
    float softmax_scale              = 0.125;
    std::uint8_t topk_indices_global = 0;

    for (auto* values : {&q}) { roundToStorage(*values, dtype); }
    for (auto* values : {&k}) { roundToStorage(*values, secondary_dtype); }

    auto expected_target = target;
    hostSparseAttnTarget(shape, q, k, lse, topk_indices, expected_target, softmax_scale, topk_indices_global);

    const std::int64_t q_dims[]{1, shape.tokens, shape.heads, shape.head_dim};
    FloatingDeviceBuffer device_q(q.size(), dtype);
    ASSERT_NE(device_q.get(), nullptr);
    const std::int64_t k_dims[]{1, shape.tokens, shape.head_dim};
    FloatingDeviceBuffer device_k(k.size(), secondary_dtype);
    ASSERT_NE(device_k.get(), nullptr);
    const std::int64_t lse_dims[]{1, shape.tokens, shape.heads};
    DeviceBuffer<float> device_lse(lse.size());
    ASSERT_NE(device_lse.get(), nullptr);
    const std::int64_t topk_indices_dims[]{1, shape.tokens, shape.selectionWidth()};
    DeviceBuffer<std::int32_t> device_topk_indices(topk_indices.size());
    ASSERT_NE(device_topk_indices.get(), nullptr);
    const std::int64_t target_dims[]{1, shape.tokens, shape.selectionWidth()};
    DeviceBuffer<float> device_target(target.size());
    ASSERT_NE(device_target.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, makeView(device_q.get(), q_dims, 4, dtype), makeView(device_k.get(), k_dims, 3, secondary_dtype),
        makeView(device_lse.get(), lse_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(device_topk_indices.get(), topk_indices_dims, 3, FTRAIN_NUMERIC_TYPE_INT32),
        makeView(device_target.get(), target_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
        makeView(&topk_indices_global, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
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
        device_q.upload(q, stream_);
        device_k.upload(k, stream_);
        device_lse.upload(lse, stream_);
        device_topk_indices.upload(topk_indices, stream_);
        device_target.upload(target, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainSparseAttnTarget(
                    makeView(device_q.get(), q_dims, 4, dtype), makeView(device_k.get(), k_dims, 3, secondary_dtype),
                    makeView(device_lse.get(), lse_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_topk_indices.get(), topk_indices_dims, 3, FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(device_target.get(), target_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&topk_indices_global, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS,
                             true),
                    compute_type, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, primitive, workspace, workspace_sizes[primitive], stream_),
                      FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_target = device_target.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_target.size(), expected_target.size());
        bool detects_zero_target = false;
        for (std::size_t i = 0; i < expected_target.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_target[i])) << "target reference " << i;
            if (std::isfinite(expected_target[i])) {
                ASSERT_TRUE(std::isfinite(actual_target[i])) << "target " << i;
                detects_zero_target |=
                    std::abs(expected_target[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_target[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_target[i], expected_target[i]) << "target " << i;
            }
        }
        EXPECT_TRUE(detects_zero_target) << "target reference cannot reject an all-zero result";
        expectNear(actual_target, expected_target, kRelTolerance, kAbsTolerance, "target");
    }
}

INSTANTIATE_TEST_SUITE_P(
    SparseAttnTargetShapeGrid, SparseAttnTargetNumericsTest,
    testing::Combine(testing::Values(ShapeParam{8, 2, 32}, ShapeParam{33, 2, 64}, ShapeParam{128, 4, 128},
                                     ShapeParam{17, 64, 512}, ShapeParam{17, 64, 576}),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16)),
    [](const testing::TestParamInfo<SparseAttnTargetNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).tokens) + "x" + std::to_string(std::get<0>(info.param).heads) +
               "x" + std::to_string(std::get<0>(info.param).head_dim) + "_k" +
               std::to_string(std::get<0>(info.param).selectionWidth()) + "_type" +
               std::to_string(std::get<1>(info.param)) + "_secondary" + std::to_string(std::get<2>(info.param));
    });

// Shapes needed by specialized BF16 support domains.
INSTANTIATE_TEST_SUITE_P(
    OptimizedSparseAttnTargetShapeGrid, SparseAttnTargetNumericsTest,
    testing::Values(std::make_tuple(ShapeParam{1024, 64, 512, 512, 4}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16},
                                    FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}),
                    std::make_tuple(ShapeParam{4096, 32, 576, 2048, 4}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16},
                                    FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16})),
    [](const testing::TestParamInfo<SparseAttnTargetNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).tokens) + "x" + std::to_string(std::get<0>(info.param).heads) +
               "x" + std::to_string(std::get<0>(info.param).head_dim) + "_k" +
               std::to_string(std::get<0>(info.param).selectionWidth()) + "_type" +
               std::to_string(std::get<1>(info.param)) + "_secondary" + std::to_string(std::get<2>(info.param));
    });
}  // namespace
