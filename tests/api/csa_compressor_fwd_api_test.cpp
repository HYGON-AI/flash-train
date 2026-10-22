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

#include "csa_compressor_fwd_staged_fixture.hpp"

namespace {

using ftrain_test::CSACompressorFwdStagedTest;

TEST_F(CSACompressorFwdStagedTest, RejectsInvalidEnumsAndUnsupportedComputeOnBothCPaths) {
    void* memory = allocate<unsigned char>(8192);
    ASSERT_NE(memory, nullptr);
    ASSERT_EQ(hipMemset(memory, 0, 8192), hipSuccess);
    const FTrainTensorId kv              = kv_id_;
    const FTrainTensorId score           = score_id_;
    const FTrainTensorId ape             = ape_id_;
    const FTrainTensorId cu_seqlens      = cu_seqlens_id_;
    const FTrainTensorId cu_seqlens_comp = cu_seqlens_comp_id_;
    const FTrainTensorId compressed      = compressed_id_;
    const FTrainTensorId ratio           = ratio_id_;
    const FTrainTensorId head_dim        = head_dim_id_;
    const FTrainTensorId coff            = coff_id_;
    const FTrainTensorId total_comp      = total_comp_id_;
    const auto op                        = csa_compressor_fwd_id_;
    FTrainArgs args                      = createArgs();
    ASSERT_NE(args, nullptr);
    FTrainPlan plan = nullptr;
    const std::vector<std::int64_t> kv_dims{64, 32};
    const FTrainStorageView kv_view{memory,
                                    kv_dims.data(),
                                    nullptr,
                                    static_cast<std::uint8_t>(kv_dims.size()),
                                    FTRAIN_NUMERIC_TYPE_BF16,
                                    FTRAIN_INDEX_TYPE_CONTINUOUS,
                                    false};
    ASSERT_EQ(ftrainArgsSetTensor(args, kv, kv_view), FTRAIN_STATUS_SUCCESS);
    const std::vector<std::int64_t> score_dims{64, 32};
    const FTrainStorageView score_view{memory,
                                       score_dims.data(),
                                       nullptr,
                                       static_cast<std::uint8_t>(score_dims.size()),
                                       FTRAIN_NUMERIC_TYPE_BF16,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS,
                                       false};
    ASSERT_EQ(ftrainArgsSetTensor(args, score, score_view), FTRAIN_STATUS_SUCCESS);
    const std::vector<std::int64_t> ape_dims{4, 32};
    const FTrainStorageView ape_view{memory,
                                     ape_dims.data(),
                                     nullptr,
                                     static_cast<std::uint8_t>(ape_dims.size()),
                                     FTRAIN_NUMERIC_TYPE_FP32,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS,
                                     false};
    ASSERT_EQ(ftrainArgsSetTensor(args, ape, ape_view), FTRAIN_STATUS_SUCCESS);
    const std::vector<std::int64_t> cu_seqlens_dims{3};
    const FTrainStorageView cu_seqlens_view{memory,
                                            cu_seqlens_dims.data(),
                                            nullptr,
                                            static_cast<std::uint8_t>(cu_seqlens_dims.size()),
                                            FTRAIN_NUMERIC_TYPE_INT32,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS,
                                            false};
    ASSERT_EQ(ftrainArgsSetTensor(args, cu_seqlens, cu_seqlens_view), FTRAIN_STATUS_SUCCESS);
    const std::vector<std::int64_t> cu_seqlens_comp_dims{3};
    const FTrainStorageView cu_seqlens_comp_view{memory,
                                                 cu_seqlens_comp_dims.data(),
                                                 nullptr,
                                                 static_cast<std::uint8_t>(cu_seqlens_comp_dims.size()),
                                                 FTRAIN_NUMERIC_TYPE_INT32,
                                                 FTRAIN_INDEX_TYPE_CONTINUOUS,
                                                 false};
    ASSERT_EQ(ftrainArgsSetTensor(args, cu_seqlens_comp, cu_seqlens_comp_view), FTRAIN_STATUS_SUCCESS);
    const std::vector<std::int64_t> compressed_dims{16, 32};
    const FTrainStorageView compressed_view{memory,
                                            compressed_dims.data(),
                                            nullptr,
                                            static_cast<std::uint8_t>(compressed_dims.size()),
                                            FTRAIN_NUMERIC_TYPE_BF16,
                                            FTRAIN_INDEX_TYPE_CONTINUOUS,
                                            false};
    ASSERT_EQ(ftrainArgsSetTensor(args, compressed, compressed_view), FTRAIN_STATUS_SUCCESS);
    std::int64_t ratio_value = 4;
    const FTrainStorageView ratio_view{
        &ratio_value, nullptr, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    ASSERT_EQ(ftrainArgsSetTensor(args, ratio, ratio_view), FTRAIN_STATUS_SUCCESS);
    std::int64_t head_dim_value = 32;
    const FTrainStorageView head_dim_view{
        &head_dim_value, nullptr, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    ASSERT_EQ(ftrainArgsSetTensor(args, head_dim, head_dim_view), FTRAIN_STATUS_SUCCESS);
    std::int64_t coff_value = 1;
    const FTrainStorageView coff_view{
        &coff_value, nullptr, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    ASSERT_EQ(ftrainArgsSetTensor(args, coff, coff_view), FTRAIN_STATUS_SUCCESS);
    std::int64_t total_comp_value = 16;
    const FTrainStorageView total_comp_view{
        &total_comp_value, nullptr, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    ASSERT_EQ(ftrainArgsSetTensor(args, total_comp, total_comp_view), FTRAIN_STATUS_SUCCESS);

    for (const auto type : {FTRAIN_NUMERIC_TYPE_INVALID, FTRAIN_NUMERIC_TYPE_COUNT}) {
        EXPECT_EQ(ftrainArgsSetCSACompressorFwd(args, op, type), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    // Prime this Ops selection cache with BF16, then keep the exact same tensor metadata.
    ASSERT_EQ(ftrainArgsSetCSACompressorFwd(args, op, FTRAIN_NUMERIC_TYPE_BF16), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    trackPlan(plan);
    ASSERT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
    plans_.back() = nullptr;
    plan          = nullptr;
    for (const auto type : {FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_INT32}) {
        ASSERT_EQ(ftrainArgsSetCSACompressorFwd(args, op, type), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_UNSUPPORTED);
        EXPECT_EQ(ftrainCSACompressorFwd(kv_view, score_view, ape_view, cu_seqlens_view, cu_seqlens_comp_view,
                                         compressed_view, ratio_view, head_dim_view, coff_view, total_comp_view, type,
                                         nullptr, 0, nullptr),
                  FTRAIN_STATUS_UNSUPPORTED);
    }
    ASSERT_EQ(ftrainArgsSetCSACompressorFwd(args, op, FTRAIN_NUMERIC_TYPE_BF16), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    trackPlan(plan);
}

TEST_F(CSACompressorFwdStagedTest, CSACompressorFwdStorageMode0RunsConvenienceAndStagedApis) {
    const std::int64_t kv_dims[]{8, 32};
    const std::vector<std::uint16_t> kv(256, 0x3f00);
    auto* device_kv = allocate<std::uint16_t>(256);
    ASSERT_NE(device_kv, nullptr);
    const std::int64_t score_dims[]{8, 32};
    const std::vector<std::uint16_t> score(256, 0x0000);
    auto* device_score = allocate<std::uint16_t>(256);
    ASSERT_NE(device_score, nullptr);
    const std::int64_t ape_dims[]{4, 32};
    const std::vector<float> ape(128, 0.0F);
    auto* device_ape = allocate<float>(128);
    ASSERT_NE(device_ape, nullptr);
    const std::int64_t cu_seqlens_dims[]{2};
    const std::vector<std::int32_t> cu_seqlens{0, 8};
    auto* device_cu_seqlens = allocate<std::int32_t>(2);
    ASSERT_NE(device_cu_seqlens, nullptr);
    const std::int64_t cu_seqlens_comp_dims[]{2};
    const std::vector<std::int32_t> cu_seqlens_comp{0, 2};
    auto* device_cu_seqlens_comp = allocate<std::int32_t>(2);
    ASSERT_NE(device_cu_seqlens_comp, nullptr);
    const std::int64_t compressed_dims[]{2, 32};
    const std::vector<std::uint16_t> compressed(64, 0x42f6);
    auto* device_compressed = allocate<std::uint16_t>(64);
    ASSERT_NE(device_compressed, nullptr);
    std::int64_t ratio      = 4;
    std::int64_t head_dim   = 32;
    std::int64_t coff       = 1;
    std::int64_t total_comp = 2;
    ASSERT_EQ(hipMemcpyAsync(device_kv, kv.data(), kv.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_score, score.data(), score.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice,
                             stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_ape, ape.data(), ape.size() * sizeof(float), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_cu_seqlens, cu_seqlens.data(), cu_seqlens.size() * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_cu_seqlens_comp, cu_seqlens_comp.data(),
                             cu_seqlens_comp.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_compressed, compressed.data(), compressed.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto kv_view = makeView(device_kv, kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto score_view =
        makeView(device_score, score_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ape_view = makeView(device_ape, ape_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_view =
        makeView(device_cu_seqlens, cu_seqlens_dims, 1, FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto cu_seqlens_comp_view = makeView(device_cu_seqlens_comp, cu_seqlens_comp_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                         FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto compressed_view =
        makeView(device_compressed, compressed_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto ratio_view    = makeView(&ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto head_dim_view = makeView(&head_dim, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto coff_view     = makeView(&coff, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto total_comp_view =
        makeView(&total_comp, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, kv_view, score_view, ape_view, cu_seqlens_view, cu_seqlens_comp_view, compressed_view,
                        ratio_view, head_dim_view, coff_view, total_comp_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS) << ftrainGetLastMessage();
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    std::uint64_t count = 0, bytes = 0;
    ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
    ASSERT_GT(count, 0U);
    const char* primitive = nullptr;
    ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &primitive), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(primitive, "CSACompressorFwd");
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
        ASSERT_EQ(hipMemcpyAsync(device_compressed, compressed.data(), compressed.size() * sizeof(std::uint16_t),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, bytes, stream_), FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_compressed, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.5F, 2e-3 + 1e-2 * std::abs(0.5F)) << i;
            }
        }
        // Restore every output before this independent execution.
        ASSERT_EQ(hipMemcpyAsync(device_compressed, compressed.data(), compressed.size() * sizeof(std::uint16_t),
                                 hipMemcpyHostToDevice, stream_),
                  hipSuccess);
        ASSERT_EQ(ftrainCSACompressorFwd(kv_view, score_view, ape_view, cu_seqlens_view, cu_seqlens_comp_view,
                                         compressed_view, ratio_view, head_dim_view, coff_view, total_comp_view,
                                         FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
                  FTRAIN_STATUS_SUCCESS);
        {
            std::vector<std::uint16_t> actual(64);
            ASSERT_EQ(hipMemcpyAsync(actual.data(), device_compressed, actual.size() * sizeof(std::uint16_t),
                                     hipMemcpyDeviceToHost, stream_),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
                float actual_value;
                std::memcpy(&actual_value, &bits, sizeof(actual_value));
                EXPECT_NEAR(actual_value, 0.5F, 2e-3 + 1e-2 * std::abs(0.5F)) << i;
            }
        }
    }
    auto invalid_kv           = kv_view;
    invalid_kv.is_host_memory = true;
    EXPECT_EQ(ftrainCSACompressorFwd(invalid_kv, score_view, ape_view, cu_seqlens_view, cu_seqlens_comp_view,
                                     compressed_view, ratio_view, head_dim_view, coff_view, total_comp_view,
                                     FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    invalid_kv              = kv_view;
    invalid_kv.numeric_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_EQ(ftrainCSACompressorFwd(invalid_kv, score_view, ape_view, cu_seqlens_view, cu_seqlens_comp_view,
                                     compressed_view, ratio_view, head_dim_view, coff_view, total_comp_view,
                                     FTRAIN_NUMERIC_TYPE_BF16, workspace, bytes, stream_),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(CSACompressorFwdStagedTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    const std::int64_t first_kv_dims[]{8, 32};
    const std::vector<std::uint16_t> first_kv(256, 0x3f00);
    auto* first_device_kv = allocate<std::uint16_t>(256);
    ASSERT_NE(first_device_kv, nullptr);
    const std::int64_t first_score_dims[]{8, 32};
    const std::vector<std::uint16_t> first_score(256, 0x0000);
    auto* first_device_score = allocate<std::uint16_t>(256);
    ASSERT_NE(first_device_score, nullptr);
    const std::int64_t first_ape_dims[]{4, 32};
    const std::vector<float> first_ape(128, 0.0F);
    auto* first_device_ape = allocate<float>(128);
    ASSERT_NE(first_device_ape, nullptr);
    const std::int64_t first_cu_seqlens_dims[]{2};
    const std::vector<std::int32_t> first_cu_seqlens{0, 8};
    auto* first_device_cu_seqlens = allocate<std::int32_t>(2);
    ASSERT_NE(first_device_cu_seqlens, nullptr);
    const std::int64_t first_cu_seqlens_comp_dims[]{2};
    const std::vector<std::int32_t> first_cu_seqlens_comp{0, 2};
    auto* first_device_cu_seqlens_comp = allocate<std::int32_t>(2);
    ASSERT_NE(first_device_cu_seqlens_comp, nullptr);
    const std::int64_t first_compressed_dims[]{2, 32};
    const std::vector<std::uint16_t> first_compressed(64, 0x42f6);
    auto* first_device_compressed = allocate<std::uint16_t>(64);
    ASSERT_NE(first_device_compressed, nullptr);
    std::int64_t first_ratio      = 4;
    std::int64_t first_head_dim   = 32;
    std::int64_t first_coff       = 1;
    std::int64_t first_total_comp = 2;
    ASSERT_EQ(hipMemcpyAsync(first_device_kv, first_kv.data(), first_kv.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_score, first_score.data(), first_score.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_ape, first_ape.data(), first_ape.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_cu_seqlens, first_cu_seqlens.data(),
                             first_cu_seqlens.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_cu_seqlens_comp, first_cu_seqlens_comp.data(),
                             first_cu_seqlens_comp.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_compressed, first_compressed.data(),
                             first_compressed.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto first_kv_view =
        makeView(first_device_kv, first_kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_score_view = makeView(first_device_score, first_score_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_ape_view =
        makeView(first_device_ape, first_ape_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_cu_seqlens_view = makeView(first_device_cu_seqlens, first_cu_seqlens_dims, 1, FTRAIN_NUMERIC_TYPE_INT32,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_cu_seqlens_comp_view = makeView(first_device_cu_seqlens_comp, first_cu_seqlens_comp_dims, 1,
                                               FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_compressed_view = makeView(first_device_compressed, first_compressed_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                          FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto first_ratio_view =
        makeView(&first_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_head_dim_view =
        makeView(&first_head_dim, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_coff_view =
        makeView(&first_coff, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto first_total_comp_view =
        makeView(&first_total_comp, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    const std::int64_t second_kv_dims[]{8, 32};
    const std::vector<std::uint16_t> second_kv(256, 0x3f00);
    auto* second_device_kv = allocate<std::uint16_t>(256);
    ASSERT_NE(second_device_kv, nullptr);
    const std::int64_t second_score_dims[]{8, 32};
    const std::vector<std::uint16_t> second_score(256, 0x0000);
    auto* second_device_score = allocate<std::uint16_t>(256);
    ASSERT_NE(second_device_score, nullptr);
    const std::int64_t second_ape_dims[]{4, 32};
    const std::vector<float> second_ape(128, 0.0F);
    auto* second_device_ape = allocate<float>(128);
    ASSERT_NE(second_device_ape, nullptr);
    const std::int64_t second_cu_seqlens_dims[]{2};
    const std::vector<std::int32_t> second_cu_seqlens{0, 8};
    auto* second_device_cu_seqlens = allocate<std::int32_t>(2);
    ASSERT_NE(second_device_cu_seqlens, nullptr);
    const std::int64_t second_cu_seqlens_comp_dims[]{2};
    const std::vector<std::int32_t> second_cu_seqlens_comp{0, 2};
    auto* second_device_cu_seqlens_comp = allocate<std::int32_t>(2);
    ASSERT_NE(second_device_cu_seqlens_comp, nullptr);
    const std::int64_t second_compressed_dims[]{2, 32};
    const std::vector<std::uint16_t> second_compressed(64, 0x42f6);
    auto* second_device_compressed = allocate<std::uint16_t>(64);
    ASSERT_NE(second_device_compressed, nullptr);
    std::int64_t second_ratio      = 4;
    std::int64_t second_head_dim   = 32;
    std::int64_t second_coff       = 1;
    std::int64_t second_total_comp = 2;
    ASSERT_EQ(hipMemcpyAsync(second_device_kv, second_kv.data(), second_kv.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_score, second_score.data(), second_score.size() * sizeof(std::uint16_t),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_ape, second_ape.data(), second_ape.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_cu_seqlens, second_cu_seqlens.data(),
                             second_cu_seqlens.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_cu_seqlens_comp, second_cu_seqlens_comp.data(),
                             second_cu_seqlens_comp.size() * sizeof(std::int32_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_compressed, second_compressed.data(),
                             second_compressed.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    auto second_kv_view =
        makeView(second_device_kv, second_kv_dims, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_score_view = makeView(second_device_score, second_score_dims, 2, FTRAIN_NUMERIC_TYPE_BF16,
                                      FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_ape_view =
        makeView(second_device_ape, second_ape_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_cu_seqlens_view      = makeView(second_device_cu_seqlens, second_cu_seqlens_dims, 1,
                                                FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_cu_seqlens_comp_view = makeView(second_device_cu_seqlens_comp, second_cu_seqlens_comp_dims, 1,
                                                FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_compressed_view      = makeView(second_device_compressed, second_compressed_dims, 2,
                                                FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false);
    auto second_ratio_view =
        makeView(&second_ratio, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_head_dim_view =
        makeView(&second_head_dim, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_coff_view =
        makeView(&second_coff, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    auto second_total_comp_view =
        makeView(&second_total_comp, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true);
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, first_kv_view, first_score_view, first_ape_view, first_cu_seqlens_view,
                        first_cu_seqlens_comp_view, first_compressed_view, first_ratio_view, first_head_dim_view,
                        first_coff_view, first_total_comp_view, FTRAIN_NUMERIC_TYPE_BF16));
    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);
    std::uint64_t first_bytes = 0;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &first_bytes), FTRAIN_STATUS_SUCCESS);
    void* first_workspace = first_bytes ? allocate<unsigned char>(first_bytes) : nullptr;
    if (first_bytes) { ASSERT_NE(first_workspace, nullptr); }
    ASSERT_TRUE(setArgs(args, second_kv_view, second_score_view, second_ape_view, second_cu_seqlens_view,
                        second_cu_seqlens_comp_view, second_compressed_view, second_ratio_view, second_head_dim_view,
                        second_coff_view, second_total_comp_view, FTRAIN_NUMERIC_TYPE_BF16));
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
        std::vector<std::uint16_t> actual(64);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), first_device_compressed, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.5F, 2e-3 + 1e-2 * std::abs(0.5F)) << i;
        }
    }
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, second_workspace, second_bytes, stream_),
              FTRAIN_STATUS_SUCCESS);
    {
        std::vector<std::uint16_t> actual(64);
        ASSERT_EQ(hipMemcpyAsync(actual.data(), second_device_compressed, actual.size() * sizeof(std::uint16_t),
                                 hipMemcpyDeviceToHost, stream_),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(actual[i]) << 16;
            float actual_value;
            std::memcpy(&actual_value, &bits, sizeof(actual_value));
            EXPECT_NEAR(actual_value, 0.5F, 2e-3 + 1e-2 * std::abs(0.5F)) << i;
        }
    }
}

}  // namespace
