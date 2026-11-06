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
#include "sparse_attn_bwd_staged_fixture.hpp"

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
    std::int64_t topk         = 128;
    std::int64_t kv_tokens    = 0;
    std::int64_t value_dim    = 0;
    bool lengths              = true;
    std::int64_t valid_keys   = 0;
    std::int64_t indexer_topk = 32;

    std::int64_t valueDim() const { return value_dim ? value_dim : head_dim; }
    std::int64_t kvTokens() const { return kv_tokens ? kv_tokens : tokens; }
    std::int64_t validKeys() const { return valid_keys ? valid_keys : topk; }
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

void hostSparseAttnBwd(const ShapeParam& shape, const std::vector<float>& q, const std::vector<float>& kv,
                       const std::vector<float>& out, const std::vector<float>& dout, const std::vector<float>& lse,
                       const std::vector<float>& attn_sink, const std::vector<std::int32_t>& topk_indices,
                       const std::vector<std::int32_t>& topk_length, std::vector<float>& dq, std::vector<float>& dkv,
                       std::vector<float>& dsink, float softmax_scale) {
    std::fill(dq.begin(), dq.end(), 0);
    std::fill(dkv.begin(), dkv.end(), 0);
    std::fill(dsink.begin(), dsink.end(), 0);

    for (std::int64_t row = 0; row < shape.tokens; ++row)
        for (std::int64_t h = 0; h < shape.heads; ++h) {
            std::vector<float> logits(shape.validKeys()), probabilities(shape.validKeys());
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int64_t j = 0; j < shape.validKeys(); ++j) {
                float dot      = 0;
                const auto col = topk_indices[row * shape.topk + j];
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += q[(row * shape.heads + h) * shape.head_dim + d] * kv[col * shape.head_dim + d];
                logits[j] = dot * softmax_scale;
                maximum   = std::max(maximum, logits[j]);
            }
            float sum = 0;
            for (auto value : logits) sum += std::exp(value - maximum);
            const float with_sink = sum + std::exp(attn_sink[h] - maximum);
            for (std::int64_t j = 0; j < shape.validKeys(); ++j)
                probabilities[j] = std::exp(logits[j] - maximum) / with_sink;
            float output_dot = 0;
            for (std::int64_t d = 0; d < shape.valueDim(); ++d)
                output_dot += out[(row * shape.heads + h) * shape.valueDim() + d] *
                              dout[(row * shape.heads + h) * shape.valueDim() + d];
            dsink[h] -= std::exp(attn_sink[h] - maximum) / with_sink * output_dot;
            for (std::int64_t j = 0; j < shape.validKeys(); ++j) {
                const auto col = topk_indices[row * shape.topk + j];
                float dot      = 0;
                for (std::int64_t d = 0; d < shape.valueDim(); ++d)
                    dot += dout[(row * shape.heads + h) * shape.valueDim() + d] * kv[col * shape.head_dim + d];
                const float ds = probabilities[j] * (dot - output_dot);
                for (std::int64_t d = 0; d < shape.head_dim; ++d) {
                    dq[(row * shape.heads + h) * shape.head_dim + d] +=
                        ds * kv[col * shape.head_dim + d] * softmax_scale;
                    dkv[col * shape.head_dim + d] +=
                        (d < shape.valueDim() ? probabilities[j] * dout[(row * shape.heads + h) * shape.valueDim() + d]
                                              : 0.0F) +
                        ds * q[(row * shape.heads + h) * shape.head_dim + d] * softmax_scale;
                }
            }
        }
}

class SparseAttnBwdNumericsTest
    : public ftrain_test::SparseAttnBwdStagedTest,
      public testing::WithParamInterface<std::tuple<ShapeParam, FTrainNumericType, FTrainNumericType>> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kFp32RelTolerance = 2e-4;
constexpr double kFp32AbsTolerance = 2e-5;

TEST_P(SparseAttnBwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    RecordProperty("family", "sparse_attn_bwd");
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

    std::vector<float> q(static_cast<std::size_t>((shape.tokens) * (shape.heads) * (shape.head_dim)));
    for (std::size_t i = 0; i < q.size(); ++i)
        q[i] = elementValue(i + 7919) * (dtype == FTRAIN_NUMERIC_TYPE_FP32 ? 1.0F : 2.0F);
    std::vector<float> kv(static_cast<std::size_t>((shape.kvTokens()) * (shape.head_dim)));
    for (std::size_t i = 0; i < kv.size(); ++i)
        kv[i] = elementValue(i + 15838) * (secondary_dtype == FTRAIN_NUMERIC_TYPE_FP32 ? 1.0F : 2.0F);
    std::vector<float> out(static_cast<std::size_t>((shape.tokens) * (shape.heads) * (shape.valueDim())));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = elementValue(i + 23757);
    std::vector<float> dout(static_cast<std::size_t>((shape.tokens) * (shape.heads) * (shape.valueDim())));
    // Positive incoming values avoid cancellation of the MMAC probability
    // products; wider Q/K magnitudes keep dQ above the storage error budget.
    for (std::size_t i = 0; i < dout.size(); ++i)
        dout[i] = elementValue(i + 31676) + (dtype == FTRAIN_NUMERIC_TYPE_FP32 ? 0.0F : 1.0F);
    std::vector<float> lse(static_cast<std::size_t>((shape.tokens) * (shape.heads)));
    for (std::size_t i = 0; i < lse.size(); ++i) lse[i] = elementValue(i + 39595);
    std::vector<float> attn_sink(static_cast<std::size_t>((shape.heads)));
    for (std::size_t i = 0; i < attn_sink.size(); ++i) attn_sink[i] = elementValue(i + 47514);
    std::vector<std::int32_t> topk_indices(static_cast<std::size_t>((shape.tokens) * (shape.topk)));
    for (std::size_t i = 0; i < topk_indices.size(); ++i)
        topk_indices[i] = i % shape.topk < static_cast<std::size_t>(shape.validKeys())
                              ? static_cast<std::int32_t>(((shape.valid_keys ? i / shape.topk : 0) + i % shape.topk) %
                                                          shape.kvTokens())
                              : -1;
    std::vector<std::int32_t> topk_length(static_cast<std::size_t>((shape.tokens)));
    std::fill(topk_length.begin(), topk_length.end(), static_cast<std::int32_t>(shape.validKeys()));
    std::vector<float> dq(static_cast<std::size_t>((shape.tokens) * (shape.heads) * (shape.head_dim)));
    std::fill(dq.begin(), dq.end(), 123);
    std::vector<float> dkv(static_cast<std::size_t>((shape.kvTokens()) * (shape.head_dim)));
    std::fill(dkv.begin(), dkv.end(), 123);
    std::vector<float> dsink(static_cast<std::size_t>((shape.heads)));
    std::fill(dsink.begin(), dsink.end(), 123);
    float softmax_scale = 0.125;

    for (auto* values : {&q, &out, &dout, &dq}) { roundToStorage(*values, dtype); }
    for (auto* values : {&kv, &dkv}) { roundToStorage(*values, secondary_dtype); }

    for (std::int64_t row = 0; row < shape.tokens; ++row)
        for (std::int64_t h = 0; h < shape.heads; ++h) {
            std::vector<float> logits(shape.validKeys()), probabilities(shape.validKeys());
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int64_t j = 0; j < shape.validKeys(); ++j) {
                float dot      = 0;
                const auto col = topk_indices[row * shape.topk + j];
                for (std::int64_t d = 0; d < shape.head_dim; ++d)
                    dot += q[(row * shape.heads + h) * shape.head_dim + d] * kv[col * shape.head_dim + d];
                logits[j] = dot * softmax_scale;
                maximum   = std::max(maximum, logits[j]);
            }
            float sum = 0;
            for (auto value : logits) sum += std::exp(value - maximum);
            const float with_sink = sum + std::exp(attn_sink[h] - maximum);
            for (std::int64_t j = 0; j < shape.validKeys(); ++j)
                probabilities[j] = std::exp(logits[j] - maximum) / with_sink;
            std::vector<float> value(shape.valueDim(), 0);
            for (std::int64_t j = 0; j < shape.validKeys(); ++j)
                for (std::int64_t d = 0; d < shape.valueDim(); ++d)
                    value[d] += probabilities[j] * kv[topk_indices[row * shape.topk + j] * shape.head_dim + d];

            lse[row * shape.heads + h] = maximum + std::log(sum);
            for (std::int64_t d = 0; d < shape.valueDim(); ++d)
                out[(row * shape.heads + h) * shape.valueDim() + d] = value[d];
        }

    for (auto* values : {&q, &out, &dout, &dq}) { roundToStorage(*values, dtype); }
    for (auto* values : {&kv, &dkv}) { roundToStorage(*values, secondary_dtype); }

    auto expected_dq    = dq;
    auto expected_dkv   = dkv;
    auto expected_dsink = dsink;
    hostSparseAttnBwd(shape, q, kv, out, dout, lse, attn_sink, topk_indices, topk_length, expected_dq, expected_dkv,
                      expected_dsink, softmax_scale);

    roundToStorage(expected_dq, dtype);
    roundToStorage(expected_dkv, secondary_dtype);

    const std::int64_t q_dims[]{shape.tokens, shape.heads, shape.head_dim};
    FloatingDeviceBuffer device_q(q.size(), dtype);
    ASSERT_NE(device_q.get(), nullptr);
    const std::int64_t kv_dims[]{shape.kvTokens(), shape.head_dim};
    FloatingDeviceBuffer device_kv(kv.size(), secondary_dtype);
    ASSERT_NE(device_kv.get(), nullptr);
    const std::int64_t out_dims[]{shape.tokens, shape.heads, shape.valueDim()};
    FloatingDeviceBuffer device_out(out.size(), dtype);
    ASSERT_NE(device_out.get(), nullptr);
    const std::int64_t dout_dims[]{shape.tokens, shape.heads, shape.valueDim()};
    FloatingDeviceBuffer device_dout(dout.size(), dtype);
    ASSERT_NE(device_dout.get(), nullptr);
    const std::int64_t lse_dims[]{shape.tokens, shape.heads};
    DeviceBuffer<float> device_lse(lse.size());
    ASSERT_NE(device_lse.get(), nullptr);
    const std::int64_t attn_sink_dims[]{shape.heads};
    DeviceBuffer<float> device_attn_sink(attn_sink.size());
    ASSERT_NE(device_attn_sink.get(), nullptr);
    const std::int64_t topk_indices_dims[]{shape.tokens, shape.topk};
    DeviceBuffer<std::int32_t> device_topk_indices(topk_indices.size());
    ASSERT_NE(device_topk_indices.get(), nullptr);
    const std::int64_t topk_length_dims[]{shape.tokens};
    DeviceBuffer<std::int32_t> device_topk_length(topk_length.size());
    ASSERT_NE(device_topk_length.get(), nullptr);
    const std::int64_t dq_dims[]{shape.tokens, shape.heads, shape.head_dim};
    FloatingDeviceBuffer device_dq(dq.size(), dtype);
    ASSERT_NE(device_dq.get(), nullptr);
    const std::int64_t dkv_dims[]{shape.kvTokens(), shape.head_dim};
    FloatingDeviceBuffer device_dkv(dkv.size(), secondary_dtype);
    ASSERT_NE(device_dkv.get(), nullptr);
    const std::int64_t dsink_dims[]{shape.heads};
    DeviceBuffer<float> device_dsink(dsink.size());
    ASSERT_NE(device_dsink.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, makeView(device_q.get(), q_dims, 3, dtype), makeView(device_kv.get(), kv_dims, 2, secondary_dtype),
        makeView(device_out.get(), out_dims, 3, dtype), makeView(device_dout.get(), dout_dims, 3, dtype),
        makeView(device_lse.get(), lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(device_attn_sink.get(), attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(device_topk_indices.get(), topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
        makeView(shape.lengths ? device_topk_length.get() : nullptr, topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
        makeView(device_dq.get(), dq_dims, 3, dtype), makeView(device_dkv.get(), dkv_dims, 2, secondary_dtype),
        makeView(device_dsink.get(), dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
        makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
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
        device_kv.upload(kv, stream_);
        device_out.upload(out, stream_);
        device_dout.upload(dout, stream_);
        device_lse.upload(lse, stream_);
        device_attn_sink.upload(attn_sink, stream_);
        device_topk_indices.upload(topk_indices, stream_);
        device_topk_length.upload(topk_length, stream_);
        device_dq.upload(dq, stream_);
        device_dkv.upload(dkv, stream_);
        device_dsink.upload(dsink, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainSparseAttnBwd(
                    makeView(device_q.get(), q_dims, 3, dtype), makeView(device_kv.get(), kv_dims, 2, secondary_dtype),
                    makeView(device_out.get(), out_dims, 3, dtype), makeView(device_dout.get(), dout_dims, 3, dtype),
                    makeView(device_lse.get(), lse_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_attn_sink.get(), attn_sink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_topk_indices.get(), topk_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(shape.lengths ? device_topk_length.get() : nullptr, topk_length_dims, 1,
                             FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(device_dq.get(), dq_dims, 3, dtype),
                    makeView(device_dkv.get(), dkv_dims, 2, secondary_dtype),
                    makeView(device_dsink.get(), dsink_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(&softmax_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    compute_type, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, primitive, workspace, workspace_sizes[primitive], stream_),
                      FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_dq    = device_dq.download(stream_);
        const auto actual_dkv   = device_dkv.download(stream_);
        const auto actual_dsink = device_dsink.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        // dQ is smaller than dKV for these bounded inputs. A tighter BF16
        // absolute budget keeps the zero-output rejection meaningful.
        const double dq_abs_tolerance = dtype == FTRAIN_NUMERIC_TYPE_BF16 ? 1e-3 : kAbsTolerance;
        ASSERT_EQ(actual_dq.size(), expected_dq.size());
        bool detects_zero_dq = false;
        for (std::size_t i = 0; i < expected_dq.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_dq[i])) << "dq reference " << i;
            if (std::isfinite(expected_dq[i])) {
                ASSERT_TRUE(std::isfinite(actual_dq[i])) << "dq " << i;
                detects_zero_dq |=
                    std::abs(expected_dq[i]) > dq_abs_tolerance + kRelTolerance * std::abs(expected_dq[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_dq[i], expected_dq[i]) << "dq " << i;
            }
        }
        EXPECT_TRUE(detects_zero_dq) << "dq reference cannot reject an all-zero result";
        expectNear(actual_dq, expected_dq, kRelTolerance, dq_abs_tolerance, "dq");
        const double secondary_rel_tolerance = secondary_dtype == FTRAIN_NUMERIC_TYPE_BF16   ? 5e-2
                                               : secondary_dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-2
                                                                                             : kRelTolerance;
        const double secondary_abs_tolerance = secondary_dtype == FTRAIN_NUMERIC_TYPE_BF16   ? 5e-3
                                               : secondary_dtype == FTRAIN_NUMERIC_TYPE_FP16 ? 2e-3
                                                                                             : kAbsTolerance;
        ASSERT_EQ(actual_dkv.size(), expected_dkv.size());
        bool detects_zero_dkv = false;
        for (std::size_t i = 0; i < expected_dkv.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_dkv[i])) << "dkv reference " << i;
            if (std::isfinite(expected_dkv[i])) {
                ASSERT_TRUE(std::isfinite(actual_dkv[i])) << "dkv " << i;
                detects_zero_dkv |= std::abs(expected_dkv[i]) >
                                    secondary_abs_tolerance + secondary_rel_tolerance * std::abs(expected_dkv[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_dkv[i], expected_dkv[i]) << "dkv " << i;
            }
        }
        EXPECT_TRUE(detects_zero_dkv) << "dkv reference cannot reject an all-zero result";
        expectNear(actual_dkv, expected_dkv, secondary_rel_tolerance, secondary_abs_tolerance, "dkv");
        ASSERT_EQ(actual_dsink.size(), expected_dsink.size());
        bool detects_zero_dsink = false;
        for (std::size_t i = 0; i < expected_dsink.size(); ++i) {
            ASSERT_FALSE(std::isnan(expected_dsink[i])) << "dsink reference " << i;
            if (std::isfinite(expected_dsink[i])) {
                ASSERT_TRUE(std::isfinite(actual_dsink[i])) << "dsink " << i;
                detects_zero_dsink |=
                    std::abs(expected_dsink[i]) > kAbsTolerance + kRelTolerance * std::abs(expected_dsink[i]);
            } else {
                // Masked infinities must match exactly; tolerance arithmetic cannot check them.
                ASSERT_EQ(actual_dsink[i], expected_dsink[i]) << "dsink " << i;
            }
        }
        EXPECT_TRUE(detects_zero_dsink) << "dsink reference cannot reject an all-zero result";
        expectNear(actual_dsink, expected_dsink, kRelTolerance, kAbsTolerance, "dsink");
    }
}

INSTANTIATE_TEST_SUITE_P(
    SparseAttnBwdShapeGrid, SparseAttnBwdNumericsTest,
    testing::Combine(testing::Values(ShapeParam{8, 8, 512}, ShapeParam{33, 8, 512}, ShapeParam{128, 16, 512}),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16),
                     testing::Values(FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16)),
    [](const testing::TestParamInfo<SparseAttnBwdNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).tokens) + "x" + std::to_string(std::get<0>(info.param).heads) +
               "x" + std::to_string(std::get<0>(info.param).head_dim) + "_k" +
               std::to_string(std::get<0>(info.param).topk) + "_s" +
               std::to_string(std::get<0>(info.param).kvTokens()) + "_len" +
               std::to_string(std::get<0>(info.param).lengths) + "_type" + std::to_string(std::get<1>(info.param)) +
               "_secondary" + std::to_string(std::get<2>(info.param));
    });

// BF16-only schedules: (T,H,D,K,S,V,lengths,valid_keys,indexer_topk).
// A short valid prefix bounds host-reference cost even for large classifier domains.
INSTANTIATE_TEST_SUITE_P(
    OptimizedSparseAttnBwdShapeGrid, SparseAttnBwdNumericsTest,
    testing::Values(
        std::make_tuple(ShapeParam{3, 64, 512, 256, 8, 512, true, 4}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16},
                        FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}),
        std::make_tuple(ShapeParam{3, 128, 512, 256, 8, 512, true, 4}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16},
                        FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}),
        std::make_tuple(ShapeParam{3, 32, 576, 128, 8, 512, true, 4}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16},
                        FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}),
        std::make_tuple(ShapeParam{256, 64, 512, 128, 8, 512, true, 4}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16},
                        FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}),
        std::make_tuple(ShapeParam{2048, 64, 512, 128, 129, 512, true, 4}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16},
                        FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}),
        std::make_tuple(ShapeParam{2048, 64, 512, 128, 128, 512, true, 4}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16},
                        FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}),
        std::make_tuple(ShapeParam{4096, 64, 512, 1152, 5120, 512, true, 4},
                        FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}),
        std::make_tuple(ShapeParam{4096, 32, 576, 2048, 4096, 512, true, 4},
                        FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}),
        std::make_tuple(ShapeParam{4096, 64, 576, 2048, 4096, 512, true, 4},
                        FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16}, FTrainNumericType{FTRAIN_NUMERIC_TYPE_BF16})),
    [](const testing::TestParamInfo<SparseAttnBwdNumericsTest::ParamType>& info) {
        return std::to_string(std::get<0>(info.param).tokens) + "x" + std::to_string(std::get<0>(info.param).heads) +
               "x" + std::to_string(std::get<0>(info.param).head_dim) + "_k" +
               std::to_string(std::get<0>(info.param).topk) + "_s" +
               std::to_string(std::get<0>(info.param).kvTokens()) + "_len" +
               std::to_string(std::get<0>(info.param).lengths) + "_type" + std::to_string(std::get<1>(info.param)) +
               "_secondary" + std::to_string(std::get<2>(info.param));
    });
}  // namespace
