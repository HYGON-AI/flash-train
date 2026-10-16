// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Staged and convenience execution against a host reference over the same shape grid.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <flash_train/flash_train.h>
#include "../support/numerics.hpp"
#include "indexer_topk_staged_fixture.hpp"

namespace {
using ftrain_test::DeviceBuffer;
using ftrain_test::expectNear;

struct ShapeParam {
    std::int64_t rows;
    std::int64_t columns;
    std::int64_t top_k;
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

void hostIndexerTopK(const ShapeParam& shape, const std::vector<float>& input_values,
                     const std::vector<std::int32_t>& seq_lens, std::vector<std::int32_t>& out_indices,
                     std::vector<float>& out_values, std::int64_t top_k, std::int64_t next_n,
                     std::uint8_t order_values) {
    for (std::int64_t row = 0; row < shape.rows; ++row) {
        std::vector<std::pair<float, std::int32_t>> items;
        for (std::int64_t col = 0; col < shape.columns; ++col) {
            const bool valid = col < seq_lens[row];
            items.emplace_back(
                valid ? input_values[row * shape.columns + col] : -std::numeric_limits<float>::infinity(),
                static_cast<std::int32_t>(col));
        }
        std::stable_sort(items.begin(), items.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        for (std::int64_t j = 0; j < shape.top_k; ++j) {
            out_values[row * shape.top_k + j]  = items[j].first;
            out_indices[row * shape.top_k + j] = std::isfinite(items[j].first) ? items[j].second : -1;
        }
    }
}

class IndexerTopKNumericsTest : public ftrain_test::IndexerTopKStagedTest,
                                public testing::WithParamInterface<ShapeParam> {};

// Exact equality is required for indices and selected FP32 values.
// Top-k copies input values without floating-point arithmetic.
constexpr double kRelTolerance = 0;
constexpr double kAbsTolerance = 0;

TEST_P(IndexerTopKNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> input_values(static_cast<std::size_t>((shape.rows) * (shape.columns)));
    for (std::size_t i = 0; i < input_values.size(); ++i) input_values[i] = elementValue(i + 7919);
    std::vector<std::int32_t> seq_lens(static_cast<std::size_t>((shape.rows)));
    std::fill(seq_lens.begin(), seq_lens.end(), static_cast<std::int32_t>(shape.columns));
    std::vector<std::int32_t> out_indices(static_cast<std::size_t>((shape.rows) * (shape.top_k)));
    std::fill(out_indices.begin(), out_indices.end(), 123);
    std::vector<float> out_values(static_cast<std::size_t>((shape.rows) * (shape.top_k)));
    std::fill(out_values.begin(), out_values.end(), 123);
    std::int64_t top_k        = shape.top_k;
    std::int64_t next_n       = 1;
    std::uint8_t order_values = 1;

    auto expected_out_indices = out_indices;
    auto expected_out_values  = out_values;
    hostIndexerTopK(shape, input_values, seq_lens, expected_out_indices, expected_out_values, top_k, next_n,
                    order_values);

    const std::int64_t input_values_dims[]{shape.rows, shape.columns};
    DeviceBuffer<float> device_input_values(input_values.size());
    ASSERT_NE(device_input_values.get(), nullptr);
    const std::int64_t seq_lens_dims[]{shape.rows};
    DeviceBuffer<std::int32_t> device_seq_lens(seq_lens.size());
    ASSERT_NE(device_seq_lens.get(), nullptr);
    const std::int64_t out_indices_dims[]{shape.rows, shape.top_k};
    DeviceBuffer<std::int32_t> device_out_indices(out_indices.size());
    ASSERT_NE(device_out_indices.get(), nullptr);
    const std::int64_t out_values_dims[]{shape.rows, shape.top_k};
    DeviceBuffer<float> device_out_values(out_values.size());
    ASSERT_NE(device_out_values.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_input_values.get(), input_values_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_seq_lens.get(), seq_lens_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                makeView(device_out_indices.get(), out_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                makeView(device_out_values.get(), out_values_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(&top_k, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&next_n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&order_values, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    auto* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) ASSERT_NE(workspace, nullptr);

    for (bool convenience : {false, true}) {
        device_input_values.upload(input_values, stream_);
        device_seq_lens.upload(seq_lens, stream_);
        device_out_indices.upload(out_indices, stream_);
        device_out_values.upload(out_values, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainIndexerTopK(
                    makeView(device_input_values.get(), input_values_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_seq_lens.get(), seq_lens_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(device_out_indices.get(), out_indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                    makeView(device_out_values.get(), out_values_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(&top_k, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&next_n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&order_values, nullptr, 0, FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        EXPECT_EQ(device_out_indices.download(stream_), expected_out_indices);
        expectNear(device_out_values.download(stream_), expected_out_values, kRelTolerance, kAbsTolerance,
                   "out_values");
    }
}

INSTANTIATE_TEST_SUITE_P(IndexerTopKShapeGrid, IndexerTopKNumericsTest,
                         testing::Values(ShapeParam{8, 16, 4}, ShapeParam{33, 64, 8}, ShapeParam{128, 128, 16}),
                         [](const testing::TestParamInfo<IndexerTopKNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.rows) + "x" + std::to_string(info.param.columns) + "x" +
                                    std::to_string(info.param.top_k);
                         });
}  // namespace
