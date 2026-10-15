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
#include "proj_rms_compute_h_bwd_staged_fixture.hpp"

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

void hostProjRmsComputeHBwd(const ShapeParam& shape, const std::vector<float>& grad_h_pre,
                            const std::vector<float>& grad_h_post, const std::vector<float>& grad_h_res,
                            const std::vector<float>& grad_r, const std::vector<float>& h_pre,
                            const std::vector<float>& h_post, const std::vector<float>& proj,
                            const std::vector<float>& r, const std::vector<float>& x, const std::vector<float>& weight,
                            const std::vector<float>& alpha_pre, const std::vector<float>& alpha_post,
                            const std::vector<float>& alpha_res, std::vector<float>& grad_x,
                            std::vector<float>& grad_weight, std::vector<float>& grad_alpha,
                            std::vector<float>& grad_bias, std::int64_t n, float eps, float compute_h_eps) {
    const auto outputs = shape.streams * shape.streams + 2 * shape.streams;
    std::fill(grad_weight.begin(), grad_weight.end(), 0);
    std::fill(grad_bias.begin(), grad_bias.end(), 0);
    std::fill(grad_alpha.begin(), grad_alpha.end(), 0);
    for (std::int64_t row = 0; row < shape.rows; ++row) {
        std::vector<float> gp(outputs);
        float rms_gradient = grad_r[row];
        for (std::int64_t j = 0; j < outputs; ++j) {
            float g, alpha;
            std::size_t alpha_slot;
            if (j < shape.streams) {
                const float sig = h_pre[row * shape.streams + j] - compute_h_eps;
                g               = grad_h_pre[row * shape.streams + j] * sig * (1 - sig);
                alpha           = alpha_pre[0];
                alpha_slot      = 0;
            } else if (j < 2 * shape.streams) {
                const float sig = h_post[row * shape.streams + j - shape.streams] / 2;
                g               = grad_h_post[row * shape.streams + j - shape.streams] * 2 * sig * (1 - sig);
                alpha           = alpha_post[0];
                alpha_slot      = 1;
            } else {
                g          = grad_h_res[row * shape.streams * shape.streams + j - 2 * shape.streams];
                alpha      = alpha_res[0];
                alpha_slot = 2;
            }
            const float value       = proj[row * outputs + j];
            grad_alpha[alpha_slot] += g * value / (r[row] + eps);
            grad_bias[j]           += g;
            gp[j]                   = g * alpha / (r[row] + eps);
            rms_gradient           -= g * alpha * value / ((r[row] + eps) * (r[row] + eps));
        }
        for (std::int64_t d = 0; d < shape.reduction; ++d) {
            float gx = rms_gradient * x[row * shape.reduction + d] / (shape.reduction * r[row]);
            for (std::int64_t j = 0; j < outputs; ++j) {
                gx                                   += gp[j] * weight[j * shape.reduction + d];
                grad_weight[j * shape.reduction + d] += gp[j] * x[row * shape.reduction + d];
            }
            grad_x[row * shape.reduction + d] = gx;
        }
    }
}

class ProjRmsComputeHBwdNumericsTest : public ftrain_test::ProjRmsComputeHBwdStagedTest,
                                       public testing::WithParamInterface<ShapeParam> {};

// Deterministic main inputs in [-0.1,0.1], with nonzero incoming gradients.
// Relative error covers FP32 reductions; the absolute budget handles near-zero gradients.
constexpr double kRelTolerance = 2e-4;
constexpr double kAbsTolerance = 2e-5;

TEST_P(ProjRmsComputeHBwdNumericsTest, MatchesHostReferenceThroughStagedAndConvenienceApis) {
    const ShapeParam shape = GetParam();
    std::vector<float> grad_h_pre(static_cast<std::size_t>((shape.rows) * (shape.streams)));
    for (std::size_t i = 0; i < grad_h_pre.size(); ++i) grad_h_pre[i] = elementValue(i + 7919);
    std::vector<float> grad_h_post(static_cast<std::size_t>((shape.rows) * (shape.streams)));
    for (std::size_t i = 0; i < grad_h_post.size(); ++i) grad_h_post[i] = elementValue(i + 15838);
    std::vector<float> grad_h_res(static_cast<std::size_t>((shape.rows) * (shape.streams * shape.streams)));
    for (std::size_t i = 0; i < grad_h_res.size(); ++i) grad_h_res[i] = elementValue(i + 23757);
    std::vector<float> grad_r(static_cast<std::size_t>((shape.rows) * (1)));
    for (std::size_t i = 0; i < grad_r.size(); ++i) grad_r[i] = elementValue(i + 31676);
    std::vector<float> h_pre(static_cast<std::size_t>((shape.rows) * (shape.streams)));
    for (std::size_t i = 0; i < h_pre.size(); ++i) h_pre[i] = elementValue(i + 39595);
    std::vector<float> h_post(static_cast<std::size_t>((shape.rows) * (shape.streams)));
    for (std::size_t i = 0; i < h_post.size(); ++i) h_post[i] = elementValue(i + 47514);
    std::vector<float> proj(
        static_cast<std::size_t>((shape.rows) * (shape.streams * shape.streams + 2 * shape.streams)));
    for (std::size_t i = 0; i < proj.size(); ++i) proj[i] = elementValue(i + 55433);
    std::vector<float> r(static_cast<std::size_t>((shape.rows) * (1)));
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = elementValue(i + 63352);
    std::vector<float> x(static_cast<std::size_t>((shape.rows) * (shape.reduction)));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = elementValue(i + 71271);
    std::vector<float> weight(
        static_cast<std::size_t>((shape.streams * shape.streams + 2 * shape.streams) * (shape.reduction)));
    for (std::size_t i = 0; i < weight.size(); ++i) weight[i] = elementValue(i + 79190);
    std::vector<float> alpha_pre(static_cast<std::size_t>((1)));
    std::fill(alpha_pre.begin(), alpha_pre.end(), 1.0F);
    std::vector<float> alpha_post(static_cast<std::size_t>((1)));
    std::fill(alpha_post.begin(), alpha_post.end(), 1.0F);
    std::vector<float> alpha_res(static_cast<std::size_t>((1)));
    std::fill(alpha_res.begin(), alpha_res.end(), 1.0F);
    std::vector<float> grad_x(static_cast<std::size_t>((shape.rows) * (shape.reduction)));
    std::fill(grad_x.begin(), grad_x.end(), 123);
    std::vector<float> grad_weight(
        static_cast<std::size_t>((shape.streams * shape.streams + 2 * shape.streams) * (shape.reduction)));
    std::fill(grad_weight.begin(), grad_weight.end(), 123);
    std::vector<float> grad_alpha(static_cast<std::size_t>((3)));
    std::fill(grad_alpha.begin(), grad_alpha.end(), 123);
    std::vector<float> grad_bias(static_cast<std::size_t>((shape.streams * shape.streams + 2 * shape.streams)));
    std::fill(grad_bias.begin(), grad_bias.end(), 123);
    std::int64_t n      = shape.streams;
    float eps           = 1e-6;
    float compute_h_eps = 1e-6;

    for (std::int64_t row = 0; row < shape.rows; ++row) {
        float sum = 0;
        for (std::int64_t d = 0; d < shape.reduction; ++d)
            sum += x[row * shape.reduction + d] * x[row * shape.reduction + d];
        r[row] = std::sqrt(sum / shape.reduction);
        for (std::int64_t j = 0; j < shape.streams * shape.streams + 2 * shape.streams; ++j) {
            float dot = 0;
            for (std::int64_t d = 0; d < shape.reduction; ++d)
                dot += x[row * shape.reduction + d] * weight[j * shape.reduction + d];
            proj[row * (shape.streams * shape.streams + 2 * shape.streams) + j] = dot;
        }
        for (std::int64_t j = 0; j < shape.streams; ++j) {
            h_pre[row * shape.streams + j] =
                1.0F / (1.0F + std::exp(-proj[row * (shape.streams * shape.streams + 2 * shape.streams) + j] /
                                        (r[row] + eps) * alpha_pre[0])) +
                compute_h_eps;
            h_post[row * shape.streams + j] =
                2.0F /
                (1.0F + std::exp(-proj[row * (shape.streams * shape.streams + 2 * shape.streams) + shape.streams + j] /
                                 (r[row] + eps) * alpha_post[0]));
        }
    }

    auto expected_grad_x      = grad_x;
    auto expected_grad_weight = grad_weight;
    auto expected_grad_alpha  = grad_alpha;
    auto expected_grad_bias   = grad_bias;
    hostProjRmsComputeHBwd(shape, grad_h_pre, grad_h_post, grad_h_res, grad_r, h_pre, h_post, proj, r, x, weight,
                           alpha_pre, alpha_post, alpha_res, expected_grad_x, expected_grad_weight, expected_grad_alpha,
                           expected_grad_bias, n, eps, compute_h_eps);

    const std::int64_t grad_h_pre_dims[]{shape.rows, shape.streams};
    DeviceBuffer<float> device_grad_h_pre(grad_h_pre.size());
    ASSERT_NE(device_grad_h_pre.get(), nullptr);
    const std::int64_t grad_h_post_dims[]{shape.rows, shape.streams};
    DeviceBuffer<float> device_grad_h_post(grad_h_post.size());
    ASSERT_NE(device_grad_h_post.get(), nullptr);
    const std::int64_t grad_h_res_dims[]{shape.rows, shape.streams * shape.streams};
    DeviceBuffer<float> device_grad_h_res(grad_h_res.size());
    ASSERT_NE(device_grad_h_res.get(), nullptr);
    const std::int64_t grad_r_dims[]{shape.rows, 1};
    DeviceBuffer<float> device_grad_r(grad_r.size());
    ASSERT_NE(device_grad_r.get(), nullptr);
    const std::int64_t h_pre_dims[]{shape.rows, shape.streams};
    DeviceBuffer<float> device_h_pre(h_pre.size());
    ASSERT_NE(device_h_pre.get(), nullptr);
    const std::int64_t h_post_dims[]{shape.rows, shape.streams};
    DeviceBuffer<float> device_h_post(h_post.size());
    ASSERT_NE(device_h_post.get(), nullptr);
    const std::int64_t proj_dims[]{shape.rows, shape.streams * shape.streams + 2 * shape.streams};
    DeviceBuffer<float> device_proj(proj.size());
    ASSERT_NE(device_proj.get(), nullptr);
    const std::int64_t r_dims[]{shape.rows, 1};
    DeviceBuffer<float> device_r(r.size());
    ASSERT_NE(device_r.get(), nullptr);
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
    const std::int64_t grad_x_dims[]{shape.rows, shape.reduction};
    DeviceBuffer<float> device_grad_x(grad_x.size());
    ASSERT_NE(device_grad_x.get(), nullptr);
    const std::int64_t grad_weight_dims[]{shape.streams * shape.streams + 2 * shape.streams, shape.reduction};
    DeviceBuffer<float> device_grad_weight(grad_weight.size());
    ASSERT_NE(device_grad_weight.get(), nullptr);
    const std::int64_t grad_alpha_dims[]{3};
    DeviceBuffer<float> device_grad_alpha(grad_alpha.size());
    ASSERT_NE(device_grad_alpha.get(), nullptr);
    const std::int64_t grad_bias_dims[]{shape.streams * shape.streams + 2 * shape.streams};
    DeviceBuffer<float> device_grad_bias(grad_bias.size());
    ASSERT_NE(device_grad_bias.get(), nullptr);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_grad_h_pre.get(), grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_grad_h_post.get(), grad_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_grad_h_res.get(), grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_grad_r.get(), grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_h_pre.get(), h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_h_post.get(), h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_proj.get(), proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_r.get(), r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_x.get(), x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_weight.get(), weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_alpha_pre.get(), alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_alpha_post.get(), alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_alpha_res.get(), alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_grad_x.get(), grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_grad_weight.get(), grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_grad_alpha.get(), grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                makeView(device_grad_bias.get(), grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
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
        device_grad_h_pre.upload(grad_h_pre, stream_);
        device_grad_h_post.upload(grad_h_post, stream_);
        device_grad_h_res.upload(grad_h_res, stream_);
        device_grad_r.upload(grad_r, stream_);
        device_h_pre.upload(h_pre, stream_);
        device_h_post.upload(h_post, stream_);
        device_proj.upload(proj, stream_);
        device_r.upload(r, stream_);
        device_x.upload(x, stream_);
        device_weight.upload(weight, stream_);
        device_alpha_pre.upload(alpha_pre, stream_);
        device_alpha_post.upload(alpha_post, stream_);
        device_alpha_res.upload(alpha_res, stream_);
        device_grad_x.upload(grad_x, stream_);
        device_grad_weight.upload(grad_weight, stream_);
        device_grad_alpha.upload(grad_alpha, stream_);
        device_grad_bias.upload(grad_bias, stream_);
        if (convenience) {
            ASSERT_EQ(
                ftrainProjRmsComputeHBwd(
                    makeView(device_grad_h_pre.get(), grad_h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_grad_h_post.get(), grad_h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_grad_h_res.get(), grad_h_res_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_grad_r.get(), grad_r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_h_pre.get(), h_pre_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_h_post.get(), h_post_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_proj.get(), proj_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_r.get(), r_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_x.get(), x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_weight.get(), weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_alpha_pre.get(), alpha_pre_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_alpha_post.get(), alpha_post_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_alpha_res.get(), alpha_res_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_grad_x.get(), grad_x_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_grad_weight.get(), grad_weight_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_grad_alpha.get(), grad_alpha_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(device_grad_bias.get(), grad_bias_dims, 1, FTRAIN_NUMERIC_TYPE_FP32),
                    makeView(&n, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    makeView(&compute_h_eps, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                    FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        }
        expectNear(device_grad_x.download(stream_), expected_grad_x, kRelTolerance, kAbsTolerance, "grad_x");
        expectNear(device_grad_weight.download(stream_), expected_grad_weight, kRelTolerance, kAbsTolerance,
                   "grad_weight");
        expectNear(device_grad_alpha.download(stream_), expected_grad_alpha, kRelTolerance, kAbsTolerance,
                   "grad_alpha");
        expectNear(device_grad_bias.download(stream_), expected_grad_bias, kRelTolerance, kAbsTolerance, "grad_bias");
    }
}

INSTANTIATE_TEST_SUITE_P(ProjRmsComputeHBwdShapeGrid, ProjRmsComputeHBwdNumericsTest,
                         testing::Values(ShapeParam{16, 2, 32}, ShapeParam{33, 4, 64}, ShapeParam{128, 4, 128}),
                         [](const testing::TestParamInfo<ProjRmsComputeHBwdNumericsTest::ParamType>& info) {
                             return std::to_string(info.param.rows) + "x" + std::to_string(info.param.streams) + "x" +
                                    std::to_string(info.param.reduction);
                         });
}  // namespace
