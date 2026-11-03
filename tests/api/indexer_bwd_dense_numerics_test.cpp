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
#include "indexer_bwd_dense_staged_fixture.hpp"

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
    std::int64_t ratio      = 1;
    float sm_scale          = 0.25F;
    std::int64_t key_tokens = 0;
    std::int64_t keyTokens() const { return key_tokens ? key_tokens : tokens; }
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

void hostIndexerBwdDense(const ShapeParam& shape, const std::vector<float>& index_q, const std::vector<float>& weights,
                         const std::vector<float>& index_k, const std::vector<float>& attn_score,
                         const std::vector<float>& attn_l1norm, const std::vector<float>& index_score,
                         const std::vector<float>& index_lse, std::vector<float>& grad_index_score,
                         std::vector<float>& dq, std::vector<float>& dw, std::vector<float>& dk, float sm_scale,
                         float loss_coeff, float grad_loss, std::int64_t ratio) {
    std::fill(dq.begin(), dq.end(), 0);
    std::fill(dw.begin(), dw.end(), 0);
    std::fill(dk.begin(), dk.end(), 0);
    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        float direct_sum = 0;
        for (std::int64_t col = 0; col < std::min(shape.keyTokens(), (row + 1) / ratio); ++col)
            direct_sum -= attn_score[row * shape.keyTokens() + col] / (attn_l1norm[row] + 1e-10F) / shape.tokens;
        for (std::int64_t col = 0; col < shape.keyTokens(); ++col) {
            const float target = attn_score[row * shape.keyTokens() + col] / (attn_l1norm[row] + 1e-10F);
            const float ds =
                col < std::min(shape.keyTokens(), (row + 1) / ratio)
                    ? (-target / shape.tokens -
                       std::exp(index_score[row * shape.keyTokens() + col] - index_lse[row]) * direct_sum) *
                          loss_coeff * grad_loss
                    : 0.0F;
            grad_index_score[row * shape.keyTokens() + col] = ds;
            if (ds == 0) { continue; }
            for (std::int64_t h = 0; h < shape.heads; ++h) {
                float dot = 0;
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += index_q[(row * shape.heads + h) * shape.head_dim + d] * index_k[col * shape.head_dim + d];
                dot                       *= sm_scale;
                dw[row * shape.heads + h] += std::max(dot, 0.0F) * ds;
                const float dg             = dot > 0 ? ds * weights[row * shape.heads + h] * sm_scale : 0.0F;
                for (std::int64_t d = 0; d < shape.head_dim; ++d) {
                    dq[(row * shape.heads + h) * shape.head_dim + d] += dg * index_k[col * shape.head_dim + d];
                    dk[col * shape.head_dim + d] += dg * index_q[(row * shape.heads + h) * shape.head_dim + d];
                }
            }
        }
    }
}

class IndexerBwdDenseNumericsTest
    : public ftrain_test::IndexerBwdDenseStagedTest,
      public testing::WithParamInterface<std::tuple<ShapeParam, FTrainNumericType, FTrainNumericType>> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kFp32RelTolerance = 2e-4;
constexpr double kFp32AbsTolerance = 2e-5;

TEST_P(IndexerBwdDenseNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    RecordProperty("family", "indexer_bwd_dense");
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

    std::vector<float> index_q(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads) * (shape.head_dim)));
    for (std::size_t i = 0; i < index_q.size(); ++i) index_q[i] = elementValue(i + 7919);
    std::vector<float> weights(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads)));
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = elementValue(i + 15838);
    std::vector<float> index_k(static_cast<std::size_t>((1) * (shape.keyTokens()) * (shape.head_dim)));
    for (std::size_t i = 0; i < index_k.size(); ++i) index_k[i] = elementValue(i + 23757);
    std::vector<float> attn_score(static_cast<std::size_t>((1) * (shape.tokens) * (shape.keyTokens())));
    for (std::size_t i = 0; i < attn_score.size(); ++i) attn_score[i] = elementValue(i + 31676);
    std::vector<float> attn_l1norm(static_cast<std::size_t>((1) * (shape.tokens)));
    for (std::size_t i = 0; i < attn_l1norm.size(); ++i) attn_l1norm[i] = elementValue(i + 39595);
    std::vector<float> index_score(static_cast<std::size_t>((1) * (shape.tokens) * (shape.keyTokens())));
    for (std::size_t i = 0; i < index_score.size(); ++i) index_score[i] = elementValue(i + 47514);
    std::vector<float> index_lse(static_cast<std::size_t>((1) * (shape.tokens)));
    for (std::size_t i = 0; i < index_lse.size(); ++i) index_lse[i] = elementValue(i + 55433);
    std::vector<float> grad_index_score(static_cast<std::size_t>((1) * (shape.tokens) * (shape.keyTokens())));
    std::fill(grad_index_score.begin(), grad_index_score.end(), 123);
    std::vector<float> dq(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads) * (shape.head_dim)));
    std::fill(dq.begin(), dq.end(), 123);
    std::vector<float> dw(static_cast<std::size_t>((1) * (shape.tokens) * (shape.heads)));
    std::fill(dw.begin(), dw.end(), 123);
    std::vector<float> dk(static_cast<std::size_t>((1) * (shape.keyTokens()) * (shape.head_dim)));
    std::fill(dk.begin(), dk.end(), 123);
    float sm_scale   = shape.sm_scale;
    float loss_coeff = 0.5F;
    // Amplify upstream gradients so the absolute tolerance rejects zero outputs.
    float grad_loss    = 64.0F * shape.tokens;
    std::int64_t ratio = shape.ratio;

    for (auto* values : {&index_q, &weights, &dq, &dw}) { roundToStorage(*values, dtype); }
    for (auto* values : {&index_k, &dk}) { roundToStorage(*values, secondary_dtype); }

    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::int64_t col = 0; col < shape.keyTokens(); ++col) {
            float score = 0;
            for (std::int64_t h = 0; h < shape.heads; ++h) {
                float dot = 0;
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += index_q[(row * shape.heads + h) * shape.head_dim + d] * index_k[col * shape.head_dim + d];
                score += std::max(dot * sm_scale, 0.0F) * weights[row * shape.heads + h];
            }
            index_score[row * shape.keyTokens() + col] =
                col < std::min(shape.keyTokens(), (row + 1) / ratio) ? score : -std::numeric_limits<float>::infinity();
            maximum = std::max(maximum, index_score[row * shape.keyTokens() + col]);
        }
        float sum = 0;
        for (std::int64_t col = 0; col < std::min(shape.keyTokens(), (row + 1) / ratio); ++col)
            sum += std::exp(index_score[row * shape.keyTokens() + col] - maximum);
        index_lse[row] = maximum + std::log(sum);
    }

    for (std::int64_t row = 0; row < shape.tokens; ++row) {
        attn_l1norm[row] = 0;
        for (std::int64_t col = 0; col < shape.keyTokens(); ++col) {
            auto& value       = attn_score[row * shape.keyTokens() + col];
            value             = col < std::min(shape.keyTokens(), (row + 1) / ratio) ? std::abs(value) + 0.01F : 0.0F;
            attn_l1norm[row] += value;
        }
    }

    for (auto* values : {&index_q, &weights, &dq, &dw}) { roundToStorage(*values, dtype); }
    for (auto* values : {&index_k, &dk}) { roundToStorage(*values, secondary_dtype); }

    auto expected_grad_index_score = grad_index_score;
    auto expected_dq               = dq;
    auto expected_dw               = dw;
    auto expected_dk               = dk;
    hostIndexerBwdDense(shape, index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse,
                        expected_grad_index_score, expected_dq, expected_dw, expected_dk, sm_scale, loss_coeff,
                        grad_loss, ratio);

    roundToStorage(expected_dq, dtype);
    roundToStorage(expected_dw, dtype);
    roundToStorage(expected_dk, secondary_dtype);

    const std::int64_t index_q_dims[]{1, shape.tokens, shape.heads, shape.head_dim};
    FloatingDeviceBuffer device_index_q(index_q.size(), dtype);
    ASSERT_NE(device_index_q.get(), nullptr);
    const std::int64_t weights_dims[]{1, shape.tokens, shape.heads};
    FloatingDeviceBuffer device_weights(weights.size(), dtype);
    ASSERT_NE(device_weights.get(), nullptr);
    const std::int64_t index_k_dims[]{1, shape.keyTokens(), shape.head_dim};
    FloatingDeviceBuffer device_index_k(index_k.size(), secondary_dtype);
    ASSERT_NE(device_index_k.get(), nullptr);
    const std::int64_t attn_score_dims[]{1, shape.tokens, shape.keyTokens()};
    DeviceBuffer<float> device_attn_score(attn_score.size());
    ASSERT_NE(device_attn_score.get(), nullptr);
    const std::int64_t attn_l1norm_dims[]{1, shape.tokens};
    DeviceBuffer<float> device_attn_l1norm(attn_l1norm.size());
    ASSERT_NE(device_attn_l1norm.get(), nullptr);
    const std::int64_t index_score_dims[]{1, shape.tokens, shape.keyTokens()};
    DeviceBuffer<float> device_index_score(index_score.size());
    ASSERT_NE(device_index_score.get(), nullptr);
    const std::int64_t index_lse_dims[]{1, shape.tokens};
    DeviceBuffer<float> device_index_lse(index_lse.size());
    ASSERT_NE(device_index_lse.get(), nullptr);
    const std::int64_t grad_index_score_dims[]{1, shape.tokens, shape.keyTokens()};
    const std::int64_t dq_dims[]{1, shape.tokens, shape.heads, shape.head_dim};
    FloatingDeviceBuffer device_dq(dq.size(), dtype);
    ASSERT_NE(device_dq.get(), nullptr);
    const std::int64_t dw_dims[]{1, shape.tokens, shape.heads};
    FloatingDeviceBuffer device_dw(dw.size(), dtype);
    ASSERT_NE(device_dw.get(), nullptr);
    const std::int64_t dk_dims[]{1, shape.keyTokens(), shape.head_dim};
    FloatingDeviceBuffer device_dk(dk.size(), secondary_dtype);
    ASSERT_NE(device_dk.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_index_q.get(), index_q_dims, 4, dtype),
                        makeView(device_weights.get(), weights_dims, 3, dtype),
                        makeView(device_index_k.get(), index_k_dims, 3, secondary_dtype),
                        makeView(device_attn_score.get(), attn_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_attn_l1norm.get(), attn_l1norm_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_index_score.get(), index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_index_lse.get(), index_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_index_score.get(), grad_index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                        makeView(device_dq.get(), dq_dims, 4, dtype), makeView(device_dw.get(), dw_dims, 3, dtype),
                        makeView(device_dk.get(), dk_dims, 3, secondary_dtype),
                        makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        makeView(&loss_coeff, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        makeView(&grad_loss, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
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
        device_index_q.upload(index_q, stream_);
        device_weights.upload(weights, stream_);
        device_index_k.upload(index_k, stream_);
        device_attn_score.upload(attn_score, stream_);
        device_attn_l1norm.upload(attn_l1norm, stream_);
        device_index_score.upload(index_score, stream_);
        device_index_lse.upload(index_lse, stream_);
        device_dq.upload(dq, stream_);
        device_dw.upload(dw, stream_);
        device_dk.upload(dk, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainIndexerBwdDense(
                    makeView(device_index_q.get(), index_q_dims, 4, dtype),
                    makeView(device_weights.get(), weights_dims, 3, dtype),
                    makeView(device_index_k.get(), index_k_dims, 3, secondary_dtype),
                    makeView(device_attn_score.get(), attn_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_attn_l1norm.get(), attn_l1norm_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_index_score.get(), index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_index_lse.get(), index_lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_index_score.get(), grad_index_score_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_dq.get(), dq_dims, 4, dtype), makeView(device_dw.get(), dw_dims, 3, dtype),
                    makeView(device_dk.get(), dk_dims, 3, secondary_dtype),
                    makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&loss_coeff, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&grad_loss, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    compute_type, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, primitive, workspace, workspace_sizes[primitive], stream_),
                      FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_grad_index_score = device_index_score.download(stream_);
        const auto actual_dq               = device_dq.download(stream_);
        const auto actual_dw               = device_dw.download(stream_);
        const auto actual_dk               = device_dk.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(actual_grad_index_score.size(), expected_grad_index_score.size());
        bool detects_zero_grad_index_score = false;
        for (std::size_t i = 0; i < expected_grad_index_score.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_grad_index_score[i])) << "grad_index_score reference " << i;
            if (std::isfinite(expected_grad_index_score[i])) {
                ASSERT_TRUE(std::isfinite(actual_grad_index_score[i])) << "grad_index_score " << i;
                detects_zero_grad_index_score |= std::abs(expected_grad_index_score[i]) >
                                                 kAbsTolerance + kRelTolerance * std::abs(expected_grad_index_score[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_grad_index_score[i], expected_grad_index_score[i]) << "grad_index_score " << i;
            }
        }
        EXPECT_TRUE(detects_zero_grad_index_score) << "grad_index_score reference cannot reject an all-zero result";
        expectNear(actual_grad_index_score, expected_grad_index_score, kRelTolerance, kAbsTolerance,
                   "grad_index_score");
        ASSERT_EQ(actual_dq.size(), expected_dq.size());
        bool detects_zero_dq = false;
        for (std::size_t i = 0; i < expected_dq.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_dq[i])) << "dq reference " << i;
            if (std::isfinite(expected_dq[i])) {
                ASSERT_TRUE(std::isfinite(actual_dq[i])) << "dq " << i;
                detects_zero_dq |= std::abs(expected_dq[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_dq[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_dq[i], expected_dq[i]) << "dq " << i;
            }
        }
        EXPECT_TRUE(detects_zero_dq) << "dq reference cannot reject an all-zero result";
        expectNear(actual_dq, expected_dq, kRelTolerance, kAbsTolerance, "dq");
        ASSERT_EQ(actual_dw.size(), expected_dw.size());
        bool detects_zero_dw = false;
        for (std::size_t i = 0; i < expected_dw.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_dw[i])) << "dw reference " << i;
            if (std::isfinite(expected_dw[i])) {
                ASSERT_TRUE(std::isfinite(actual_dw[i])) << "dw " << i;
                detects_zero_dw |= std::abs(expected_dw[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_dw[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_dw[i], expected_dw[i]) << "dw " << i;
            }
        }
        EXPECT_TRUE(detects_zero_dw) << "dw reference cannot reject an all-zero result";
        expectNear(actual_dw, expected_dw, kRelTolerance, kAbsTolerance, "dw");
        const double secondary_rel_tolerance = secondary_dtype == FTRAIN_NUMERIC_TYPE_BF16   ? 5e-2
                                               : secondary_dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-2
                                                                                             : kRelTolerance;
        const double secondary_abs_tolerance = secondary_dtype == FTRAIN_NUMERIC_TYPE_BF16   ? 5e-3
                                               : secondary_dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-3
                                                                                             : kAbsTolerance;
        ASSERT_EQ(actual_dk.size(), expected_dk.size());
        bool detects_zero_dk = false;
        for (std::size_t i = 0; i < expected_dk.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_dk[i])) << "dk reference " << i;
            if (std::isfinite(expected_dk[i])) {
                ASSERT_TRUE(std::isfinite(actual_dk[i])) << "dk " << i;
                detects_zero_dk |= std::abs(expected_dk[i]) >
                                   secondary_abs_tolerance + secondary_rel_tolerance * std::abs(expected_dk[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_dk[i], expected_dk[i]) << "dk " << i;
            }
        }
        EXPECT_TRUE(detects_zero_dk) << "dk reference cannot reject an all-zero result";
        expectNear(actual_dk, expected_dk, secondary_rel_tolerance, secondary_abs_tolerance, "dk");
    }
}

INSTANTIATE_TEST_SUITE_P(
    IndexerBwdDenseShapeGrid, IndexerBwdDenseNumericsTest,
    testing::Combine(testing::Values(ShapeParam{8, 2, 32}, ShapeParam{33, 2, 64}, ShapeParam{128, 4, 128},
                                     ShapeParam{17, 64, 128}),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16)),
    [](const testing::TestParamInfo<IndexerBwdDenseNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).tokens) + "x" + std::to_string(std::get<0>(info.param).heads) +
               "x" + std::to_string(std::get<0>(info.param).head_dim) + "_ratio" +
               std::to_string(std::get<0>(info.param).ratio) + "_keys" +
               std::to_string(std::get<0>(info.param).keyTokens()) + "_unit" +
               std::to_string(std::get<0>(info.param).sm_scale == 1.0F) + "_type" +
               std::to_string(std::get<1>(info.param)) + "_secondary" + std::to_string(std::get<2>(info.param));
    });

// Shapes needed by specialized BF16 support domains.
INSTANTIATE_TEST_SUITE_P(
    OptimizedIndexerBwdDenseShapeGrid, IndexerBwdDenseNumericsTest,
    testing::Values(std::make_tuple(ShapeParam{33, 64, 128, 4, 0.25F, 16}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16},
                                    FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}),
                    std::make_tuple(ShapeParam{33, 64, 128, 1, 1.0F, 2048}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16},
                                    FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16})),
    [](const testing::TestParamInfo<IndexerBwdDenseNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).tokens) + "x" + std::to_string(std::get<0>(info.param).heads) +
               "x" + std::to_string(std::get<0>(info.param).head_dim) + "_ratio" +
               std::to_string(std::get<0>(info.param).ratio) + "_keys" +
               std::to_string(std::get<0>(info.param).keyTokens()) + "_unit" +
               std::to_string(std::get<0>(info.param).sm_scale == 1.0F) + "_type" +
               std::to_string(std::get<1>(info.param)) + "_secondary" + std::to_string(std::get<2>(info.param));
    });
}  // namespace
