// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <flash_train/env.h>
#include <flash_train/flash_train.h>

#include "indexer_fwd_staged_fixture.hpp"

namespace {

using ftrain_test::IndexerFwdStagedTest;

constexpr std::int64_t kBatch    = 1;
constexpr std::int64_t kQueryLen = 4;
constexpr std::int64_t kKeyLen   = 4;
constexpr std::int64_t kHeads    = 2;
constexpr std::int64_t kHeadDim  = 16;
constexpr std::int64_t kRatio    = 1;
constexpr float kSmScale         = 0.5F;

const std::int64_t kQDims[4]      = {kBatch, kQueryLen, kHeads, kHeadDim};
const std::int64_t kKDims[4]      = {kBatch, kKeyLen, 1, kHeadDim};
const std::int64_t kWDims[3]      = {kBatch, kQueryLen, kHeads};
const std::int64_t kScoresDims[3] = {kBatch, kQueryLen, kKeyLen};

FTrainStorageView view(void* memory, const std::int64_t* dims, std::uint8_t rank, FTrainNumericType numeric_type) {
    return FTrainStorageView{memory, dims, nullptr, rank, numeric_type, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
}

FTrainStorageView hostScalar(const void* memory, FTrainNumericType numeric_type) {
    return FTrainStorageView{const_cast<void*>(memory),    nullptr, nullptr, 0, numeric_type,
                             FTRAIN_INDEX_TYPE_CONTINUOUS, true};
}

// scores[b,q,k] = mask * sum_h ReLU(sm_scale * q·k) * w, mask: k < (q + 1)/ratio.
std::vector<float> referenceIndexerFwd(const std::vector<float>& q, const std::vector<float>& k,
                                       const std::vector<float>& w) {
    std::vector<float> scores(static_cast<std::size_t>(kBatch * kQueryLen * kKeyLen), 0.0F);
    for (std::int64_t b = 0; b < kBatch; ++b) {
        for (std::int64_t row = 0; row < kQueryLen; ++row) {
            for (std::int64_t column = 0; column < kKeyLen; ++column) {
                const bool valid = column < (row + 1) / kRatio;
                float value      = 0.0F;
                for (std::int64_t head = 0; head < kHeads; ++head) {
                    float dot = 0.0F;
                    for (std::int64_t dim = 0; dim < kHeadDim; ++dim) {
                        const std::size_t q_index =
                            static_cast<std::size_t>(((b * kQueryLen + row) * kHeads + head) * kHeadDim + dim);
                        const std::size_t k_index  = static_cast<std::size_t>((b * kKeyLen + column) * kHeadDim + dim);
                        dot                       += q[q_index] * k[k_index];
                    }
                    const float weight  = w[static_cast<std::size_t>((b * kQueryLen + row) * kHeads + head)];
                    value              += std::fmax(0.0F, kSmScale * dot) * weight;
                }
                scores[static_cast<std::size_t>((b * kQueryLen + row) * kKeyLen + column)] = valid ? value : -INFINITY;
            }
        }
    }
    return scores;
}

TEST_F(IndexerFwdStagedTest, Fp32ChainMatchesTheHostReference) {
    std::vector<float> q(static_cast<std::size_t>(kBatch * kQueryLen * kHeads * kHeadDim));
    std::vector<float> k(static_cast<std::size_t>(kBatch * kKeyLen * kHeadDim));
    std::vector<float> w(static_cast<std::size_t>(kBatch * kQueryLen * kHeads));
    for (std::size_t index = 0; index < q.size(); ++index) {
        q[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.25F;
    }
    for (std::size_t index = 0; index < k.size(); ++index) {
        k[index] = static_cast<float>(static_cast<int>(index % 5) - 2) * 0.5F;
    }
    for (std::size_t index = 0; index < w.size(); ++index) { w[index] = 1.0F + static_cast<float>(index % 3) * 0.5F; }

    auto* device_q      = allocate<float>(q.size());
    auto* device_k      = allocate<float>(k.size());
    auto* device_w      = allocate<float>(w.size());
    auto* device_scores = allocate<float>(kBatch * kQueryLen * kKeyLen);
    ASSERT_NE(device_q, nullptr);
    ASSERT_NE(device_k, nullptr);
    ASSERT_NE(device_w, nullptr);
    ASSERT_NE(device_scores, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, view(device_q, kQDims, 4, FTRAIN_NUMERIC_TYPE_FP32), view(device_k, kKDims, 4, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_w, kWDims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_scores, kScoresDims, 3, FTRAIN_NUMERIC_TYPE_FP32), hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64),
        hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32)));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    trackPlan(plan);
    ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);

    const std::vector<float> expected = referenceIndexerFwd(q, k, w);
    std::vector<float> actual(expected.size());
    ASSERT_EQ(hipMemcpy(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (std::isinf(expected[index])) {
            EXPECT_TRUE(std::isinf(actual[index])) << "at index " << index;
        } else {
            EXPECT_NEAR(actual[index], expected[index], 1e-3F) << "at index " << index;
        }
    }
}

TEST_F(IndexerFwdStagedTest, ConvenienceEntryMatchesReferenceAndRetainsFailureStatus) {
    std::vector<float> q(kBatch * kQueryLen * kHeads * kHeadDim, 0.25F);
    std::vector<float> k(kBatch * kKeyLen * kHeadDim, 0.5F);
    std::vector<float> w(kBatch * kQueryLen * kHeads, 0.75F);
    auto* device_q      = allocate<float>(q.size());
    auto* device_k      = allocate<float>(k.size());
    auto* device_w      = allocate<float>(w.size());
    auto* device_scores = allocate<float>(kBatch * kQueryLen * kKeyLen);
    ASSERT_NE(device_q, nullptr);
    ASSERT_NE(device_k, nullptr);
    ASSERT_NE(device_w, nullptr);
    ASSERT_NE(device_scores, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, view(device_q, kQDims, 4, FTRAIN_NUMERIC_TYPE_FP32), view(device_k, kKDims, 4, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_w, kWDims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_scores, kScoresDims, 3, FTRAIN_NUMERIC_TYPE_FP32), hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64),
        hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32)));
    // The buffers above were allocated in q/k/w/output order; call repeatedly to exercise cache hits.
    const auto qv = view(allocations_[0], kQDims, 4, FTRAIN_NUMERIC_TYPE_FP32);
    const auto kv = view(allocations_[1], kKDims, 4, FTRAIN_NUMERIC_TYPE_FP32);
    const auto wv = view(allocations_[2], kWDims, 3, FTRAIN_NUMERIC_TYPE_FP32);
    const auto sv = view(device_scores, kScoresDims, 3, FTRAIN_NUMERIC_TYPE_FP32);
    for (int repeat = 0; repeat < 3; ++repeat) {
        ASSERT_EQ(ftrainIndexerFwd(qv, kv, wv, sv, hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64),
                                   hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, nullptr,
                                   0, stream_),
                  FTRAIN_STATUS_SUCCESS);
    }
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    const auto expected = referenceIndexerFwd(q, k, w);
    std::vector<float> actual(expected.size());
    ASSERT_EQ(hipMemcpy(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::isinf(expected[i])) {
            EXPECT_EQ(actual[i], expected[i]);
        } else {
            EXPECT_NEAR(actual[i], expected[i], 1e-5F);
        }
    }
    auto malformed     = sv;
    malformed.num_dims = 2;
    EXPECT_EQ(ftrainIndexerFwd(qv, kv, wv, malformed, hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64),
                               hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32, nullptr, 0,
                               stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    const char* message = ftrainGetLastMessage();
    ASSERT_NE(message, nullptr);
    EXPECT_NE(std::string(message).find("rank"), std::string::npos);
}

TEST_F(IndexerFwdStagedTest, PlanNamesTheChainTheComputeTypeSelects) {
    const std::vector<float> q(static_cast<std::size_t>(kBatch * kQueryLen * kHeads * kHeadDim), 1.0F);
    const std::vector<float> k(static_cast<std::size_t>(kBatch * kKeyLen * kHeadDim), 1.0F);
    const std::vector<float> w(static_cast<std::size_t>(kBatch * kQueryLen * kHeads), 1.0F);
    auto* device_q      = allocate<float>(q.size());
    auto* device_k      = allocate<float>(k.size());
    auto* device_w      = allocate<float>(w.size());
    auto* device_scores = allocate<float>(kBatch * kQueryLen * kKeyLen);
    ASSERT_NE(device_q, nullptr);
    ASSERT_NE(device_k, nullptr);
    ASSERT_NE(device_w, nullptr);
    ASSERT_NE(device_scores, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, view(device_q, kQDims, 4, FTRAIN_NUMERIC_TYPE_FP32), view(device_k, kKDims, 4, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_w, kWDims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_scores, kScoresDims, 3, FTRAIN_NUMERIC_TYPE_FP32), hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64),
        hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32)));
    FTrainPlan plan = nullptr;

    // The FP32 compute type selects the FP32 chain; the FP16 chain wants FP16
    // storage, so nothing applies and the plan creation reports it.
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    trackPlan(plan);
    std::uint64_t count = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(count, 1U);
    const char* name = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &name), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "Fp32IndexerFwd");

    FTrainArgs other = nullptr;
    ASSERT_EQ(ftrainArgsCreate(&other, ops_), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetTensor(other, q_id_, view(device_scores, kQDims, 4, FTRAIN_NUMERIC_TYPE_FP32)),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetTensor(other, k_id_, view(device_scores, kKDims, 4, FTRAIN_NUMERIC_TYPE_FP32)),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetTensor(other, w_id_, view(device_scores, kWDims, 3, FTRAIN_NUMERIC_TYPE_FP32)),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetTensor(other, scores_id_, view(device_scores, kScoresDims, 3, FTRAIN_NUMERIC_TYPE_FP32)),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetTensor(other, ratio_id_, hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64)),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetTensor(other, sm_scale_id_, hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32)),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainArgsSetIndexerFwd(other, indexer_fwd_id_, FTRAIN_NUMERIC_TYPE_BF16), FTRAIN_STATUS_SUCCESS);
    FTrainPlan rejected = nullptr;
    EXPECT_EQ(ftrainPlanCreate(&rejected, other), FTRAIN_STATUS_UNSUPPORTED);
    if (rejected != nullptr) { EXPECT_EQ(ftrainPlanDestroy(rejected), FTRAIN_STATUS_SUCCESS); }
    EXPECT_EQ(ftrainArgsDestroy(other), FTRAIN_STATUS_SUCCESS);
}

TEST_F(IndexerFwdStagedTest, ReportsIncompleteArgumentsAndUnknownComputeTypes) {
    FTrainArgs incomplete = nullptr;
    ASSERT_EQ(ftrainArgsCreate(&incomplete, ops_), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsSetIndexerFwd(incomplete, indexer_fwd_id_, FTRAIN_NUMERIC_TYPE_COUNT),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    FTrainPlan plan = nullptr;
    EXPECT_EQ(ftrainPlanCreate(&plan, incomplete), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainArgsDestroy(incomplete), FTRAIN_STATUS_SUCCESS);
}

TEST_F(IndexerFwdStagedTest, ScalarPortsRequireHostRankZeroAndDeclaredDtype) {
    const std::vector<float> q(static_cast<std::size_t>(kBatch * kQueryLen * kHeads * kHeadDim), 1.0F);
    const std::vector<float> k(static_cast<std::size_t>(kBatch * kKeyLen * kHeadDim), 1.0F);
    const std::vector<float> w(static_cast<std::size_t>(kBatch * kQueryLen * kHeads), 1.0F);
    auto* device_q      = allocate<float>(q.size());
    auto* device_k      = allocate<float>(k.size());
    auto* device_w      = allocate<float>(w.size());
    auto* device_scores = allocate<float>(kBatch * kQueryLen * kKeyLen);
    ASSERT_NE(device_q, nullptr);
    ASSERT_NE(device_k, nullptr);
    ASSERT_NE(device_w, nullptr);
    ASSERT_NE(device_scores, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, view(device_q, kQDims, 4, FTRAIN_NUMERIC_TYPE_FP32), view(device_k, kKDims, 4, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_w, kWDims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_scores, kScoresDims, 3, FTRAIN_NUMERIC_TYPE_FP32), hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64),
        hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32)));
    FTrainPlan plan = nullptr;

    ASSERT_EQ(ftrainArgsSetTensor(args, ratio_id_, hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32)),
              FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(ftrainArgsSetTensor(args, ratio_id_, hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64)),
              FTRAIN_STATUS_SUCCESS);

    const std::int64_t one_dim[] = {1};
    ASSERT_EQ(ftrainArgsSetTensor(args, sm_scale_id_,
                                  FTrainStorageView{const_cast<float*>(&kSmScale), one_dim, nullptr, 1,
                                                    FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true}),
              FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(ftrainArgsSetTensor(args, sm_scale_id_, hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32)),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    trackPlan(plan);
}

TEST_F(IndexerFwdStagedTest, RebindsHostScaleAndOutlivesOpsAndArgs) {
    const std::vector<float> q(kBatch * kQueryLen * kHeads * kHeadDim, 1.0F);
    const std::vector<float> k(kBatch * kKeyLen * kHeadDim, 1.0F);
    const std::vector<float> w(kBatch * kQueryLen * kHeads, 1.0F);
    auto* device_q      = allocate<float>(q.size());
    auto* device_k      = allocate<float>(k.size());
    auto* device_w      = allocate<float>(w.size());
    auto* device_scores = allocate<float>(kBatch * kQueryLen * kKeyLen);
    ASSERT_NE(device_q, nullptr);
    ASSERT_NE(device_k, nullptr);
    ASSERT_NE(device_w, nullptr);
    ASSERT_NE(device_scores, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, view(device_q, kQDims, 4, FTRAIN_NUMERIC_TYPE_FP32), view(device_k, kKDims, 4, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_w, kWDims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_scores, kScoresDims, 3, FTRAIN_NUMERIC_TYPE_FP32), hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64),
        hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32)));
    FTrainPlan plan = nullptr;
    float scale     = 0.25F;
    ASSERT_EQ(ftrainArgsSetTensor(args, sm_scale_id_, hostScalar(&scale, FTRAIN_NUMERIC_TYPE_FP32)),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    trackPlan(plan);
    scale = 0.75F;
    FTrainPlan rebound{};
    ASSERT_EQ(ftrainPlanCreate(&rebound, args), FTRAIN_STATUS_SUCCESS);
    // Both plans own the scalar value captured by configure, as in Gemm.
    scale = -1.0F;
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    args = nullptr;
    ASSERT_EQ(ftrainOpsDestroy(ops_), FTRAIN_STATUS_SUCCESS);
    ops_ = nullptr;
    std::vector<float> actual(kBatch * kQueryLen * kKeyLen);
    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(ftrainPlanExecutePrimitive(i == 0 ? plan : rebound, 0, nullptr, 0, stream_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        ASSERT_EQ(hipMemcpy(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        for (std::int64_t row = 0; row < kQueryLen; ++row) {
            for (std::int64_t column = 0; column < kKeyLen; ++column) {
                const float value = actual[row * kKeyLen + column];
                if (column < (row + 1) / kRatio) {
                    EXPECT_FLOAT_EQ(value, kHeads * kHeadDim * (i == 0 ? 0.25F : 0.75F));
                } else {
                    EXPECT_EQ(value, -INFINITY);
                }
            }
        }
    }
    EXPECT_EQ(ftrainPlanDestroy(rebound), FTRAIN_STATUS_SUCCESS);
}

TEST_F(IndexerFwdStagedTest, DistinguishesScalarStructureFromImplementationSupport) {
    const std::vector<float> q(kBatch * kQueryLen * kHeads * kHeadDim, 1.0F);
    const std::vector<float> k(kBatch * kKeyLen * kHeadDim, 1.0F);
    const std::vector<float> w(kBatch * kQueryLen * kHeads, 1.0F);
    auto* device_q      = allocate<float>(q.size());
    auto* device_k      = allocate<float>(k.size());
    auto* device_w      = allocate<float>(w.size());
    auto* device_scores = allocate<float>(kBatch * kQueryLen * kKeyLen);
    ASSERT_NE(device_q, nullptr);
    ASSERT_NE(device_k, nullptr);
    ASSERT_NE(device_w, nullptr);
    ASSERT_NE(device_scores, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, view(device_q, kQDims, 4, FTRAIN_NUMERIC_TYPE_FP32), view(device_k, kKDims, 4, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_w, kWDims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_scores, kScoresDims, 3, FTRAIN_NUMERIC_TYPE_FP32), hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64),
        hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32)));
    FTrainPlan plan       = nullptr;
    auto scalar           = hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32);
    scalar.is_host_memory = false;
    ASSERT_EQ(ftrainArgsSetTensor(args, sm_scale_id_, scalar), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_INVALID_ARGUMENT);
    scalar.is_host_memory = true;
    scalar.numeric_type   = FTRAIN_NUMERIC_TYPE_INT32;
    ASSERT_EQ(ftrainArgsSetTensor(args, sm_scale_id_, scalar), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_UNSUPPORTED);
    alignas(float) unsigned char unaligned[2 * sizeof(float)]{};
    scalar.numeric_type = FTRAIN_NUMERIC_TYPE_FP32;
    scalar.memory       = unaligned + 1;
    ASSERT_EQ(ftrainArgsSetTensor(args, sm_scale_id_, scalar), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_UNSUPPORTED);
    scalar = hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32);
    ASSERT_EQ(ftrainArgsSetTensor(args, sm_scale_id_, scalar), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    trackPlan(plan);
}

TEST_F(IndexerFwdStagedTest, MixedStorageUsesFp32WithWorkspaceThroughBothApis) {
    const std::vector<float> q(kBatch * kQueryLen * kHeads * kHeadDim, 0.25F);
    const std::vector<float> k(kBatch * kKeyLen * kHeadDim, 0.5F);
    const std::vector<float> w(kBatch * kQueryLen * kHeads, 0.75F);
    auto* device_q      = allocate<float>(q.size());
    auto* device_k      = allocate<float>(k.size());
    auto* device_w      = allocate<float>(w.size());
    auto* device_scores = allocate<float>(kBatch * kQueryLen * kKeyLen);
    ASSERT_NE(device_q, nullptr);
    ASSERT_NE(device_k, nullptr);
    ASSERT_NE(device_w, nullptr);
    ASSERT_NE(device_scores, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, view(device_q, kQDims, 4, FTRAIN_NUMERIC_TYPE_FP32), view(device_k, kKDims, 4, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_w, kWDims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        view(device_scores, kScoresDims, 3, FTRAIN_NUMERIC_TYPE_FP32), hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64),
        hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32)));
    FTrainPlan plan = nullptr;
    // Exact BF16 bit representation of 0.25; avoids a HIP type in the public API test.
    const std::vector<std::uint16_t> q_bf16(q.size(), 0x3e80);
    auto* device_bf16_q = allocate<std::uint16_t>(q.size());
    ASSERT_NE(device_bf16_q, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_bf16_q, q_bf16.data(), q_bf16.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    const auto qv = view(device_bf16_q, kQDims, 4, FTRAIN_NUMERIC_TYPE_BF16);
    const auto kv = view(allocations_[1], kKDims, 4, FTRAIN_NUMERIC_TYPE_FP32);
    const auto wv = view(allocations_[2], kWDims, 3, FTRAIN_NUMERIC_TYPE_FP32);
    const auto sv = view(device_scores, kScoresDims, 3, FTRAIN_NUMERIC_TYPE_FP32);
    ASSERT_EQ(ftrainArgsSetTensor(args, q_id_, qv), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    trackPlan(plan);
    const char* name = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &name), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(name, "MixedFp32IndexerFwd");
    std::uint64_t bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(bytes, q.size() * sizeof(float));
    void* workspace = allocate<unsigned char>(bytes);
    ASSERT_NE(workspace, nullptr);
    EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    const auto expected = referenceIndexerFwd(q, k, w);
    for (int mode = 0; mode < 3; ++mode) {
        ASSERT_EQ(hipMemsetAsync(device_scores, 0, expected.size() * sizeof(float), stream_), hipSuccess);
        if (mode == 0) {
            ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(ftrainIndexerFwd(qv, kv, wv, sv, hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64),
                                       hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_FP32,
                                       workspace, bytes, stream_),
                      FTRAIN_STATUS_SUCCESS);
        }
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> actual(expected.size());
        ASSERT_EQ(hipMemcpy(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_EQ(actual[i], expected[i]); }
    }
    // A warmed cache must neither ignore compute_type nor accept an integer input.
    EXPECT_EQ(ftrainIndexerFwd(qv, kv, wv, sv, hostScalar(&kRatio, FTRAIN_NUMERIC_TYPE_INT64),
                               hostScalar(&kSmScale, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_NUMERIC_TYPE_BF16, workspace,
                               bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    auto invalid         = kv;
    invalid.numeric_type = FTRAIN_NUMERIC_TYPE_INT32;
    ASSERT_EQ(ftrainArgsSetTensor(args, k_id_, invalid), FTRAIN_STATUS_SUCCESS);
    FTrainPlan rejected = nullptr;
    EXPECT_EQ(ftrainPlanCreate(&rejected, args), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(rejected, nullptr);
}

TEST_F(IndexerFwdStagedTest, Bf16IndexerFwdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> q(256, 0x3e80);
    auto* device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 1, 32};
    const std::vector<std::uint16_t> k(128, 0x3f00);
    auto* device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> w(8, 0x3f80);
    auto* device_w = allocate<std::uint16_t>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t scores_dims[]{1, 4, 4};
    const std::vector<float> scores(16, 123.0F);
    auto* device_scores = allocate<float>(16);
    ASSERT_NE(device_scores, nullptr);
    std::int64_t ratio = 1;
    float sm_scale     = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto scores_view =
        makeView(device_scores, scores_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view    = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args    = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, q_view, k_view, w_view, scores_view, ratio_view, sm_scale_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Bf16IndexerFwd");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        // Restore every output before this independent execution.
        ASSERT_EQ(
            hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(
            hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainIndexerFwd(q_view, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                                   FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerFwd(invalid_q, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                               FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerFwd(invalid_q, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                               FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerFwdStagedTest, Fp16IndexerFwdStorageMode1RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<__half> q(256, __float2half(0.25F));
    auto* device_q = allocate<__half>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 1, 32};
    const std::vector<__half> k(128, __float2half(0.5F));
    auto* device_k = allocate<__half>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{1, 4, 2};
    const std::vector<__half> w(8, __float2half(1.0F));
    auto* device_w = allocate<__half>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t scores_dims[]{1, 4, 4};
    const std::vector<float> scores(16, 123.0F);
    auto* device_scores = allocate<float>(16);
    ASSERT_NE(device_scores, nullptr);
    std::int64_t ratio = 1;
    float sm_scale     = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(__half), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 4, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 3, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto scores_view =
        makeView(device_scores, scores_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view    = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args    = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, q_view, k_view, w_view, scores_view, ratio_view, sm_scale_view, FTRAIN_NUMERIC_TYPE_FP16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp16IndexerFwd");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        // Restore every output before this independent execution.
        ASSERT_EQ(
            hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(
            hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainIndexerFwd(q_view, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                                   FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerFwd(invalid_q, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                               FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerFwd(invalid_q, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                               FTRAIN_NUMERIC_TYPE_FP16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerFwdStagedTest, Fp32IndexerFwdStorageMode2RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<float> q(256, 0.25F);
    auto* device_q = allocate<float>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 1, 32};
    const std::vector<float> k(128, 0.5F);
    auto* device_k = allocate<float>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{1, 4, 2};
    const std::vector<float> w(8, 1.0F);
    auto* device_w = allocate<float>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t scores_dims[]{1, 4, 4};
    const std::vector<float> scores(16, 123.0F);
    auto* device_scores = allocate<float>(16);
    ASSERT_NE(device_scores, nullptr);
    std::int64_t ratio = 1;
    float sm_scale     = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto scores_view =
        makeView(device_scores, scores_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view    = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args    = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, q_view, k_view, w_view, scores_view, ratio_view, sm_scale_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "Fp32IndexerFwd");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        // Restore every output before this independent execution.
        ASSERT_EQ(
            hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(
            hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainIndexerFwd(q_view, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                                   FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerFwd(invalid_q, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                               FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerFwd(invalid_q, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                               FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerFwdStagedTest, MixedFp32IndexerFwdStorageMode3RunsConvenienceAndStagedApis) {
    const std::int64_t q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> q(256, 0x3e80);
    auto* device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(device_q, nullptr);
    const std::int64_t k_dims[]{1, 4, 1, 32};
    const std::vector<float> k(128, 0.5F);
    auto* device_k = allocate<float>(128);
    ASSERT_NE(device_k, nullptr);
    const std::int64_t w_dims[]{1, 4, 2};
    const std::vector<float> w(8, 1.0F);
    auto* device_w = allocate<float>(8);
    ASSERT_NE(device_w, nullptr);
    const std::int64_t scores_dims[]{1, 4, 4};
    const std::vector<float> scores(16, 123.0F);
    auto* device_scores = allocate<float>(16);
    ASSERT_NE(device_scores, nullptr);
    std::int64_t ratio = 1;
    float sm_scale     = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(device_q, q.data(), q.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_w, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto q_view = makeView(device_q, q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto k_view = makeView(device_k, k_dims, 4, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto w_view = makeView(device_w, w_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto scores_view =
        makeView(device_scores, scores_dims, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view    = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto sm_scale_view = makeView(&sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args    = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, q_view, k_view, w_view, scores_view, ratio_view, sm_scale_view, FTRAIN_NUMERIC_TYPE_FP32));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "MixedFp32IndexerFwd");
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &bytes), FTRAIN_STATUS_SUCCESS);
    void* workspace = bytes ? allocate<unsigned char>(bytes) : nullptr;
    if (bytes) {
        ASSERT_NE(workspace, nullptr);
        EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    for (int repeat = 0; repeat < 2; ++repeat) {
        // Restore every output before this independent execution.
        ASSERT_EQ(
            hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(
            hipMemcpyAsync(device_scores, scores.data(), scores.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
            hipSuccess);
        ASSERT_EQ(ftrainIndexerFwd(q_view, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                                   FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<float> actual(16);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_scores, actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                     stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            std::vector<float> expected(16, 2.0F);
            std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
            std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
            expected[11] = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::isnan(expected[i])) continue;
                if (std::isinf(expected[i])) {
                    EXPECT_EQ(actual[i], expected[i]) << i;
                } else {
                    EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
                }
            }
        }
    }
    auto invalid_q           = q_view;
    invalid_q.is_host_memory = true;
    EXPECT_EQ(ftrainIndexerFwd(invalid_q, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                               FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_q              = q_view;
    invalid_q.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainIndexerFwd(invalid_q, k_view, w_view, scores_view, ratio_view, sm_scale_view,
                               FTRAIN_NUMERIC_TYPE_FP32, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(IndexerFwdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> first_q(256, 0x3e80);
    auto* first_device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(first_device_q, nullptr);
    const std::int64_t first_k_dims[]{1, 4, 1, 32};
    const std::vector<std::uint16_t> first_k(128, 0x3f00);
    auto* first_device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(first_device_k, nullptr);
    const std::int64_t first_w_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> first_w(8, 0x3f80);
    auto* first_device_w = allocate<std::uint16_t>(8);
    ASSERT_NE(first_device_w, nullptr);
    const std::int64_t first_scores_dims[]{1, 4, 4};
    const std::vector<float> first_scores(16, 123.0F);
    auto* first_device_scores = allocate<float>(16);
    ASSERT_NE(first_device_scores, nullptr);
    std::int64_t first_ratio = 1;
    float first_sm_scale     = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(first_device_q, first_q.data(), first_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_k, first_k.data(), first_k.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_w, first_w.data(), first_w.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_scores, first_scores.data(), first_scores.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_q_view =
        makeView(first_device_q, first_q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_k_view =
        makeView(first_device_k, first_k_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_w_view =
        makeView(first_device_w, first_w_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_scores_view = makeView(first_device_scores, first_scores_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_ratio_view =
        makeView(&first_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_sm_scale_view =
        makeView(&first_sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_q_dims[]{1, 4, 2, 32};
    const std::vector<std::uint16_t> second_q(256, 0x3e80);
    auto* second_device_q = allocate<std::uint16_t>(256);
    ASSERT_NE(second_device_q, nullptr);
    const std::int64_t second_k_dims[]{1, 4, 1, 32};
    const std::vector<std::uint16_t> second_k(128, 0x3f00);
    auto* second_device_k = allocate<std::uint16_t>(128);
    ASSERT_NE(second_device_k, nullptr);
    const std::int64_t second_w_dims[]{1, 4, 2};
    const std::vector<std::uint16_t> second_w(8, 0x3f80);
    auto* second_device_w = allocate<std::uint16_t>(8);
    ASSERT_NE(second_device_w, nullptr);
    const std::int64_t second_scores_dims[]{1, 4, 4};
    const std::vector<float> second_scores(16, 123.0F);
    auto* second_device_scores = allocate<float>(16);
    ASSERT_NE(second_device_scores, nullptr);
    std::int64_t second_ratio = 1;
    float second_sm_scale     = 0.25F;
    ASSERT_EQ(hipMemcpyAsync(second_device_q, second_q.data(), second_q.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_k, second_k.data(), second_k.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_w, second_w.data(), second_w.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_scores, second_scores.data(), second_scores.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_q_view =
        makeView(second_device_q, second_q_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_k_view =
        makeView(second_device_k, second_k_dims, 4, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_w_view =
        makeView(second_device_w, second_w_dims, 3, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_scores_view = makeView(second_device_scores, second_scores_dims, 3, FTRAIN_NUMERIC_TYPE_FP32,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_ratio_view =
        makeView(&second_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_sm_scale_view =
        makeView(&second_sm_scale, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_q_view, first_k_view, first_w_view, first_scores_view, first_ratio_view,
                        first_sm_scale_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_q_view, second_k_view, second_w_view, second_scores_view, second_ratio_view,
                        second_sm_scale_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan second_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&second_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(second_plan, nullptr);
    trackPlan(second_plan);
    std::uint64_t second_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(second_plan, 0, &second_bytes), FTRAIN_STATUS_SUCCESS);
    void* second_workspace = second_bytes ? allocate<unsigned char>(second_bytes) : nullptr;
    if (second_bytes) { ASSERT_NE(second_workspace, nullptr); }
    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    ASSERT_EQ(ftrainOpsDestroy(ops_), FTRAIN_STATUS_SUCCESS);
    ops_ = nullptr;
    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, first_workspace, first_bytes, stream_), FTRAIN_STATUS_SUCCESS);
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_scores, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(16, 2.0F);
        std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
        std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
        expected[11] = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) continue;
            if (std::isinf(expected[i])) {
                EXPECT_EQ(actual[i], expected[i]) << i;
            } else {
                EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
            }
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<float> actual(16);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_scores, actual.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        std::vector<float> expected(16, 2.0F);
        std::fill(expected.begin() + 1, expected.begin() + 4, -std::numeric_limits<float>::infinity());
        std::fill(expected.begin() + 6, expected.begin() + 8, -std::numeric_limits<float>::infinity());
        expected[11] = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) continue;
            if (std::isinf(expected[i])) {
                EXPECT_EQ(actual[i], expected[i]) << i;
            } else {
                EXPECT_NEAR(actual[i], expected[i], 2e-5 + 2e-4 * std::abs(expected[i])) << i;
            }
        }
    }
}

}  // namespace
