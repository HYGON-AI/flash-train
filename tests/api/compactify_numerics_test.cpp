// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Staged and convenience execution against a host reference over the same shape grid.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <flash_train/flash_train.h>
#include "../support/numerics.hpp"
#include "compactify_staged_fixture.hpp"

namespace {
using ftrain_test::DeviceBuffer;
using ftrain_test::expectNear;

struct ShapeParam {
    std::int64_t rows;
    std::int64_t columns;
    std::int64_t invalid_stride;
};

float elementValue(std::uint64_t linear) {
    return static_cast<float>((linear * 2654435761ULL) % 199ULL) / 995.0F - 0.1F;
}

void hostCompactify(const ShapeParam& shape, const std::vector<std::int32_t>& indices,
                    std::vector<std::int32_t>& output, std::vector<std::int32_t>& topk_length) {
    std::fill(output.begin(), output.end(), -1);
    for (std::int64_t row = 0; row < shape.rows; ++row) {
        std::int32_t length = 0;
        for (std::int64_t col = 0; col < shape.columns; ++col)
            if (indices[row * shape.columns + col] >= 0)
                output[row * shape.columns + length++] = indices[row * shape.columns + col];
        topk_length[row] = length;
    }
}

class CompactifyNumericsTest : public ftrain_test::CompactifyStagedTest,
                               public testing::WithParamInterface<ShapeParam> {};

// Exact integer equality is required: compactification only copies indices
// and counts valid entries; it performs no floating-point arithmetic.

TEST_P(CompactifyNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<std::int32_t> indices(static_cast<std::size_t>((shape.rows) * (shape.columns)));
    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] =
            (i % shape.columns) % shape.invalid_stride == 0 ? -1 : static_cast<std::int32_t>(i % shape.columns);
    std::vector<std::int32_t> output(static_cast<std::size_t>((shape.rows) * (shape.columns)));
    std::fill(output.begin(), output.end(), 123);
    std::vector<std::int32_t> topk_length(static_cast<std::size_t>((shape.rows)));
    std::fill(topk_length.begin(), topk_length.end(), 123);

    auto expected_output      = output;
    auto expected_topk_length = topk_length;
    hostCompactify(shape, indices, expected_output, expected_topk_length);

    const std::int64_t indices_dims[]{shape.rows, shape.columns};
    DeviceBuffer<std::int32_t> device_indices(indices.size());
    ASSERT_NE(device_indices.get(), nullptr);
    const std::int64_t output_dims[]{shape.rows, shape.columns};
    DeviceBuffer<std::int32_t> device_output(output.size());
    ASSERT_NE(device_output.get(), nullptr);
    const std::int64_t topk_length_dims[]{shape.rows};
    DeviceBuffer<std::int32_t> device_topk_length(topk_length.size());
    ASSERT_NE(device_topk_length.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_indices.get(), indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                        makeView(device_output.get(), output_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                        makeView(device_topk_length.get(), topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32)));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    auto* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) ASSERT_NE(workspace, nullptr);

    for (bool convenience : {false, true}) {
        device_indices.upload(indices, stream_);
        device_output.upload(output, stream_);
        device_topk_length.upload(topk_length, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainCompactify(makeView(device_indices.get(), indices_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                                 makeView(device_output.get(), output_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                                 makeView(device_topk_length.get(), topk_length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                                 workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        const auto actual_output      = device_output.download(stream_);
        const auto actual_topk_length = device_topk_length.download(stream_);
        // Readbacks are asynchronous; finish them before inspecting host storage.
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        EXPECT_EQ(actual_output, expected_output);
        EXPECT_EQ(actual_topk_length, expected_topk_length);
    }
}

INSTANTIATE_TEST_SUITE_P(CompactifyShapeGrid, CompactifyNumericsTest,
                         testing::Values(ShapeParam{8, 16, 4}, ShapeParam{33, 64, 3}, ShapeParam{128, 128, 4}),
                         [](const testing::TestParamInfo<CompactifyNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.rows) + "x" + std::to_string(info.param.columns) + "x" +
                                    std::to_string(info.param.invalid_stride);
                         });
}  // namespace
