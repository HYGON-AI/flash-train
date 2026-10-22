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
#include "proj_rms_compute_h_fwd_staged_fixture.hpp"

namespace {
using ftrain_test::DeviceBuffer;
using ftrain_test::expectNear;

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

class ProjRmsComputeHFwdNumericsTest : public ftrain_test::ProjRmsComputeHFwdStagedTest,
                                       public testing::WithParamInterface<ShapeParam> {};

// Deterministic bounded inputs are shared by the device call and host reference.
// Relative error covers FP32 reductions; the absolute budget handles near-zero outputs.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(ProjRmsComputeHFwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
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

    auto expected_h_pre  = h_pre;
    auto expected_h_post = h_post;
    auto expected_h_res  = h_res;
    auto expected_r      = r;
    auto expected_proj   = proj;
    hostProjRmsComputeHFwd(shape, x, weight, alpha_pre, alpha_post, alpha_res, bias, expected_h_pre, expected_h_post,
                           expected_h_res, expected_r, expected_proj, n, eps, compute_h_eps);

    const std::int64_t x_dims[]{shape.rows, shape.reduction};
    DeviceBuffer<float> device_x(x.size());
    ASSERT_NE(device_x.get(), nullptr);
    const std::int64_t weight_dims[]{shape.streams * shape.streams + 2 * shape.streams, shape.reduction};
    DeviceBuffer<float> device_weight(weight.size());
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
    DeviceBuffer<float> device_h_pre(h_pre.size());
    ASSERT_NE(device_h_pre.get(), nullptr);
    const std::int64_t h_post_dims[]{shape.rows, shape.streams};
    DeviceBuffer<float> device_h_post(h_post.size());
    ASSERT_NE(device_h_post.get(), nullptr);
    const std::int64_t h_res_dims[]{shape.rows, shape.streams * shape.streams};
    DeviceBuffer<float> device_h_res(h_res.size());
    ASSERT_NE(device_h_res.get(), nullptr);
    const std::int64_t r_dims[]{shape.rows, 1};
    DeviceBuffer<float> device_r(r.size());
    ASSERT_NE(device_r.get(), nullptr);
    const std::int64_t proj_dims[]{shape.rows, shape.streams * shape.streams + 2 * shape.streams};
    DeviceBuffer<float> device_proj(proj.size());
    ASSERT_NE(device_proj.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_x.get(), x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_weight.get(), weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_alpha_pre.get(), alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_alpha_post.get(), alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_alpha_res.get(), alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_bias.get(), bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_h_pre.get(), h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_h_post.get(), h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_h_res.get(), h_res_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_r.get(), r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_proj.get(), proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    trackPlan(plan);
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    auto* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) ASSERT_NE(workspace, nullptr);

    for (bool convenience : {false, true}) {
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
                    makeView(device_x.get(), x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_weight.get(), weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_alpha_pre.get(), alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_alpha_post.get(), alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_alpha_res.get(), alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_bias.get(), bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_h_pre.get(), h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_h_post.get(), h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_h_res.get(), h_res_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_r.get(), r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_proj.get(), proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
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

INSTANTIATE_TEST_SUITE_P(ProjRmsComputeHFwdShapeGrid, ProjRmsComputeHFwdNumericsTest,
                         testing::Values(ShapeParam{16, 2, 32}, ShapeParam{33, 4, 64}, ShapeParam{128, 4, 128}),
                         [](const testing::TestParamInfo<ProjRmsComputeHFwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.rows) + "x" + std::to_string(info.param.streams) + "x" +
                                    std::to_string(info.param.reduction);
                         });
}  // namespace
