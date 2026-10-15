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
#include <vector>

#include <gtest/gtest.h>

#include <flash_train/flash_train.h>
#include "../support/numerics.hpp"
#include "csa_compressor_fwd_staged_fixture.hpp"

namespace {
using ftrain_test::DeviceBuffer;
using ftrain_test::expectNear;

struct ShapeParam {
    std::int64_t tokens;
    std::int64_t ratio;
    std::int64_t head_dim;
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

std::vector<std::uint16_t> toBf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, &values[i], 4);
        bits      += 0x7fff + ((bits >> 16) & 1);
        result[i]  = static_cast<std::uint16_t>(bits >> 16);
    }
    return result;
}
std::vector<float> fromBf16(const std::vector<std::uint16_t>& values) {
    std::vector<float> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(values[i]) << 16;
        std::memcpy(&result[i], &bits, 4);
    }
    return result;
}

void hostCSACompressorFwd(const ShapeParam& shape, const std::vector<float>& kv, const std::vector<float>& score,
                          const std::vector<float>& ape, const std::vector<std::int32_t>& cu_seqlens,
                          const std::vector<std::int32_t>& cu_seqlens_comp, std::vector<float>& compressed,
                          std::int64_t ratio, std::int64_t head_dim, std::int64_t coff, std::int64_t total_comp) {
    for (std::int64_t group = 0; group < shape.tokens / shape.ratio; ++group)
        for (std::int64_t d = 0; d < shape.head_dim; ++d) {
            std::vector<float> probabilities(shape.ratio);
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::int64_t pos = 0; pos < shape.ratio; ++pos)
                maximum = std::max(
                    maximum, score[(group * shape.ratio + pos) * shape.head_dim + d] + ape[pos * shape.head_dim + d]);
            float sum = 0, average = 0;
            for (std::int64_t pos = 0; pos < shape.ratio; ++pos) {
                probabilities[pos]  = std::exp(score[(group * shape.ratio + pos) * shape.head_dim + d] +
                                               ape[pos * shape.head_dim + d] - maximum);
                sum                += probabilities[pos];
            }
            for (std::int64_t pos = 0; pos < shape.ratio; ++pos) {
                probabilities[pos] /= sum;
                average            += probabilities[pos] * kv[(group * shape.ratio + pos) * shape.head_dim + d];
            }
            compressed[group * shape.head_dim + d] = average;
        }
}

class CSACompressorFwdNumericsTest : public ftrain_test::CSACompressorFwdStagedTest,
                                     public testing::WithParamInterface<ShapeParam> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// BF16 has a 1/128 significand step: the 1% relative budget covers final storage rounding.
// The absolute budget handles near-zero outputs after FP32 accumulation.
constexpr double kRelTolerance = 1e-2;
constexpr double kAbsTolerance = 1e-3;

TEST_P(CSACompressorFwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> kv(static_cast<std::size_t>((shape.tokens) * (shape.head_dim)));
    for (std::size_t i = 0; i < kv.size(); ++i) kv[i] = elementValue(i + 7919);
    kv = fromBf16(toBf16(kv));
    std::vector<float> score(static_cast<std::size_t>((shape.tokens) * (shape.head_dim)));
    for (std::size_t i = 0; i < score.size(); ++i) score[i] = elementValue(i + 15838);
    score = fromBf16(toBf16(score));
    std::vector<float> ape(static_cast<std::size_t>((shape.ratio) * (shape.head_dim)));
    for (std::size_t i = 0; i < ape.size(); ++i) ape[i] = elementValue(i + 23757);
    std::vector<std::int32_t> cu_seqlens(static_cast<std::size_t>((2)));
    cu_seqlens[0] = 0;
    cu_seqlens[1] = static_cast<std::int32_t>(shape.tokens);
    std::vector<std::int32_t> cu_seqlens_comp(static_cast<std::size_t>((2)));
    cu_seqlens_comp[0] = 0;
    cu_seqlens_comp[1] = static_cast<std::int32_t>(shape.tokens / shape.ratio);
    std::vector<float> compressed(static_cast<std::size_t>((shape.tokens / shape.ratio) * (shape.head_dim)));
    std::fill(compressed.begin(), compressed.end(), 123);
    std::int64_t ratio      = shape.ratio;
    std::int64_t head_dim   = shape.head_dim;
    std::int64_t coff       = 1;
    std::int64_t total_comp = shape.tokens / shape.ratio;

    auto expected_compressed = compressed;
    hostCSACompressorFwd(shape, kv, score, ape, cu_seqlens, cu_seqlens_comp, expected_compressed, ratio, head_dim, coff,
                         total_comp);

    const std::int64_t kv_dims[]{shape.tokens, shape.head_dim};
    DeviceBuffer<std::uint16_t> device_kv(kv.size());
    ASSERT_NE(device_kv.get(), nullptr);
    const auto storage_kv = toBf16(kv);
    const std::int64_t score_dims[]{shape.tokens, shape.head_dim};
    DeviceBuffer<std::uint16_t> device_score(score.size());
    ASSERT_NE(device_score.get(), nullptr);
    const auto storage_score = toBf16(score);
    const std::int64_t ape_dims[]{shape.ratio, shape.head_dim};
    DeviceBuffer<float> device_ape(ape.size());
    ASSERT_NE(device_ape.get(), nullptr);
    const std::int64_t cu_seqlens_dims[]{2};
    DeviceBuffer<std::int32_t> device_cu_seqlens(cu_seqlens.size());
    ASSERT_NE(device_cu_seqlens.get(), nullptr);
    const std::int64_t cu_seqlens_comp_dims[]{2};
    DeviceBuffer<std::int32_t> device_cu_seqlens_comp(cu_seqlens_comp.size());
    ASSERT_NE(device_cu_seqlens_comp.get(), nullptr);
    const std::int64_t compressed_dims[]{shape.tokens / shape.ratio, shape.head_dim};
    DeviceBuffer<std::uint16_t> device_compressed(compressed.size());
    ASSERT_NE(device_compressed.get(), nullptr);
    const auto storage_compressed = toBf16(compressed);
    FTrainArgs args               = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_kv.get(), kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16),
                makeView(device_score.get(), score_dims, 2, FTRAIN_NUMERIC_TYPE_BF16),
                makeView(device_ape.get(), ape_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_cu_seqlens.get(), cu_seqlens_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                makeView(device_cu_seqlens_comp.get(), cu_seqlens_comp_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                makeView(device_compressed.get(), compressed_dims, 2, FTRAIN_NUMERIC_TYPE_BF16),
                makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&head_dim, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&coff, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&total_comp, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    auto* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) ASSERT_NE(workspace, nullptr);

    for (bool convenience : {false, true}) {
        device_kv.upload(storage_kv, stream_);
        device_score.upload(storage_score, stream_);
        device_ape.upload(ape, stream_);
        device_cu_seqlens.upload(cu_seqlens, stream_);
        device_cu_seqlens_comp.upload(cu_seqlens_comp, stream_);
        device_compressed.upload(storage_compressed, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainCSACompressorFwd(
                    makeView(device_kv.get(), kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16),
                    makeView(device_score.get(), score_dims, 2, FTRAIN_NUMERIC_TYPE_BF16),
                    makeView(device_ape.get(), ape_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_cu_seqlens.get(), cu_seqlens_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(device_cu_seqlens_comp.get(), cu_seqlens_comp_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(device_compressed.get(), compressed_dims, 2, FTRAIN_NUMERIC_TYPE_BF16),
                    makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&head_dim, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&coff, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&total_comp, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        expectNear(fromBf16(device_compressed.download(stream_)), expected_compressed, kRelTolerance, kAbsTolerance,
                   "compressed");
    }
}

INSTANTIATE_TEST_SUITE_P(CSACompressorFwdShapeGrid, CSACompressorFwdNumericsTest,
                         testing::Values(ShapeParam{8, 4, 32}, ShapeParam{36, 4, 64}, ShapeParam{128, 4, 128}),
                         [](const testing::TestParamInfo<CSACompressorFwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.tokens) + "x" + std::to_string(info.param.ratio) + "x" +
                                    std::to_string(info.param.head_dim);
                         });
}  // namespace
