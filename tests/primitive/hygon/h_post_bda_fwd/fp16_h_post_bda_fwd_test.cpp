// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/h_post_bda_fwd/problem.hpp"
#include "flash_train/primitive/hygon/h_post_bda_fwd/fp16_h_post_bda_fwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kHResDims[4] = {2, 1, 2, 2};
constexpr std::int64_t kResidualDims[4] = {2, 1, 2, 16};
constexpr std::int64_t kHPostDims[3] = {2, 1, 2};
constexpr std::int64_t kXDims[3] = {2, 1, 16};
constexpr std::int64_t kBiasDims[1] = {16};
constexpr std::int64_t kOutputDims[4] = {2, 1, 2, 16};

struct ProblemPorts {
    void* h_res                       = kDefaultHResMemory;
    const std::int64_t* h_res_dims    = kHResDims;
    FTrainNumericType h_res_type      = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType h_res_index       = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* residual                    = kDefaultResidualMemory;
    const std::int64_t* residual_dims = kResidualDims;
    FTrainNumericType residual_type   = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType residual_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* h_post                      = kDefaultHPostMemory;
    const std::int64_t* h_post_dims   = kHPostDims;
    FTrainNumericType h_post_type     = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType h_post_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* x                           = kDefaultXMemory;
    const std::int64_t* x_dims        = kXDims;
    FTrainNumericType x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType x_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* bias                        = kDefaultBiasMemory;
    const std::int64_t* bias_dims     = kBiasDims;
    FTrainNumericType bias_type       = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType bias_index        = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* output                      = kDefaultOutputMemory;
    const std::int64_t* output_dims   = kOutputDims;
    FTrainNumericType output_type     = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType output_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type    = FTRAIN_NUMERIC_TYPE_FP16;
    alignas(16) static unsigned char kDefaultHResMemory[16];
    alignas(16) static unsigned char kDefaultResidualMemory[16];
    alignas(16) static unsigned char kDefaultHPostMemory[16];
    alignas(16) static unsigned char kDefaultXMemory[16];
    alignas(16) static unsigned char kDefaultBiasMemory[16];
    alignas(16) static unsigned char kDefaultOutputMemory[16];
};

alignas(16) unsigned char ProblemPorts::kDefaultHResMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultResidualMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultHPostMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultXMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultBiasMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultOutputMemory[16];

Tensor makeDeviceTensor(void* memory, const std::int64_t* dims, std::uint8_t rank, FTrainNumericType numeric_type,
                        FTrainIndexType index_type) {
    FTrainStorageView view{};
    view.memory         = memory;
    view.dims           = dims;
    view.strides        = nullptr;
    view.num_dims       = rank;
    view.numeric_type   = numeric_type;
    view.index_type     = index_type;
    view.is_host_memory = false;
    return Tensor{StorageView{view}};
}

HPostBdaFwdProblem makeProblem(const ProblemPorts& ports) {
    return HPostBdaFwdProblem{
        makeDeviceTensor(ports.h_res, ports.h_res_dims, 4, ports.h_res_type, ports.h_res_index),
        makeDeviceTensor(ports.residual, ports.residual_dims, 4, ports.residual_type, ports.residual_index),
        makeDeviceTensor(ports.h_post, ports.h_post_dims, 3, ports.h_post_type, ports.h_post_index),
        makeDeviceTensor(ports.x, ports.x_dims, 3, ports.x_type, ports.x_index),
        makeDeviceTensor(ports.bias, ports.bias_dims, 1, ports.bias_type, ports.bias_index),
        makeDeviceTensor(ports.output, ports.output_dims, 4, ports.output_type, ports.output_index),
        HPostBdaFwdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Fp16HPostBdaFwdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_h_res[16];
    rebound.h_res = other_h_res;
    alignas(16) static unsigned char other_residual[16];
    rebound.residual = other_residual;
    alignas(16) static unsigned char other_h_post[16];
    rebound.h_post = other_h_post;
    alignas(16) static unsigned char other_x[16];
    rebound.x = other_x;
    alignas(16) static unsigned char other_bias[16];
    rebound.bias = other_bias;
    alignas(16) static unsigned char other_output[16];
    rebound.output = other_output;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp16HPostBdaFwdTest, AcceptsContinuousAlignedOperands) {
    Fp16HPostBdaFwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp16HPostBdaFwd");
}

TEST(Fp16HPostBdaFwdTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp16HPostBdaFwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_h_res;
    unsupported_h_res.h_res_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_h_res), kConstraints).isSuccess());
    ProblemPorts unsupported_residual;
    unsupported_residual.residual_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_residual), kConstraints).isSuccess());
    ProblemPorts unsupported_h_post;
    unsupported_h_post.h_post_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_h_post), kConstraints).isSuccess());
    ProblemPorts unsupported_x;
    unsupported_x.x_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_x), kConstraints).isSuccess());
    ProblemPorts unsupported_bias;
    unsupported_bias.bias_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_bias), kConstraints).isSuccess());
    ProblemPorts unsupported_output;
    unsupported_output.output_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_output), kConstraints).isSuccess());
}

TEST(Fp16HPostBdaFwdTest, RejectsNonContinuousIndexTypes) {
    Fp16HPostBdaFwd primitive;
    ProblemPorts strided_h_res;
    strided_h_res.h_res_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_h_res), kConstraints).isSuccess());
    ProblemPorts strided_residual;
    strided_residual.residual_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_residual), kConstraints).isSuccess());
    ProblemPorts strided_h_post;
    strided_h_post.h_post_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_h_post), kConstraints).isSuccess());
    ProblemPorts strided_x;
    strided_x.x_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_x), kConstraints).isSuccess());
    ProblemPorts strided_bias;
    strided_bias.bias_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_bias), kConstraints).isSuccess());
    ProblemPorts strided_output;
    strided_output.output_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_output), kConstraints).isSuccess());
}

TEST(Fp16HPostBdaFwdTest, RejectsMisalignedAddresses) {
    Fp16HPostBdaFwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_h_res;
    misaligned_h_res.h_res = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_h_res), kConstraints).isSuccess());
    ProblemPorts misaligned_residual;
    misaligned_residual.residual = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_residual), kConstraints).isSuccess());
    ProblemPorts misaligned_h_post;
    misaligned_h_post.h_post = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_h_post), kConstraints).isSuccess());
    ProblemPorts misaligned_x;
    misaligned_x.x = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_x), kConstraints).isSuccess());
    ProblemPorts misaligned_bias;
    misaligned_bias.bias = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_bias), kConstraints).isSuccess());
    ProblemPorts misaligned_output;
    misaligned_output.output = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_output), kConstraints).isSuccess());
}

TEST(Fp16HPostBdaFwdTest, ChecksRequiredAndOptionalAddresses) {
    Fp16HPostBdaFwd primitive;
    ProblemPorts null_h_res;
    null_h_res.h_res = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_h_res), kConstraints).isSuccess());
    ProblemPorts null_residual;
    null_residual.residual = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_residual), kConstraints).isSuccess());
    ProblemPorts null_h_post;
    null_h_post.h_post = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_h_post), kConstraints).isSuccess());
    ProblemPorts null_x;
    null_x.x = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_x), kConstraints).isSuccess());
    ProblemPorts null_bias;
    null_bias.bias = nullptr;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(null_bias), kConstraints).isSuccess());
    ProblemPorts null_output;
    null_output.output = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_output), kConstraints).isSuccess());
}

TEST(Fp16HPostBdaFwdTest, ClonesIntoIndependentConfigurableCopies) {
    Fp16HPostBdaFwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp16HPostBdaFwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<HPostBdaFwdProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<HPostBdaFwdProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp16HPostBdaFwdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original                = makeProblem(ports).getProblemKey();
        const std::int64_t h_res_dims[]    = {2, 1, 33, 33};
        ports.h_res_dims                   = h_res_dims;
        const std::int64_t residual_dims[] = {2, 1, 33, 16};
        ports.residual_dims                = residual_dims;
        const std::int64_t h_post_dims[]   = {2, 1, 33};
        ports.h_post_dims                  = h_post_dims;
        const std::int64_t output_dims[]   = {2, 1, 33, 16};
        ports.output_dims                  = output_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp16HPostBdaFwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp16HPostBdaFwdTest, ChecksChannelTileArithmeticBoundary) {
    constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
    for (const std::int64_t channels : {limit - 255, limit}) {
        ProblemPorts ports;

        const std::int64_t h_res_dims[] = {
            ports.h_res_dims[0] == 16 ? channels : 1, ports.h_res_dims[1] == 16 ? channels : 1,
            ports.h_res_dims[2] == 16 ? channels : 1, ports.h_res_dims[3] == 16 ? channels : 1};
        ports.h_res_dims = h_res_dims;

        const std::int64_t residual_dims[] = {
            ports.residual_dims[0] == 16 ? channels : 1, ports.residual_dims[1] == 16 ? channels : 1,
            ports.residual_dims[2] == 16 ? channels : 1, ports.residual_dims[3] == 16 ? channels : 1};
        ports.residual_dims = residual_dims;

        const std::int64_t h_post_dims[] = {ports.h_post_dims[0] == 16 ? channels : 1,
                                            ports.h_post_dims[1] == 16 ? channels : 1,
                                            ports.h_post_dims[2] == 16 ? channels : 1};
        ports.h_post_dims                = h_post_dims;

        const std::int64_t x_dims[] = {ports.x_dims[0] == 16 ? channels : 1, ports.x_dims[1] == 16 ? channels : 1,
                                       ports.x_dims[2] == 16 ? channels : 1};
        ports.x_dims                = x_dims;

        const std::int64_t bias_dims[] = {ports.bias_dims[0] == 16 ? channels : 1};
        ports.bias_dims                = bias_dims;

        const std::int64_t output_dims[] = {
            ports.output_dims[0] == 16 ? channels : 1, ports.output_dims[1] == 16 ? channels : 1,
            ports.output_dims[2] == 16 ? channels : 1, ports.output_dims[3] == 16 ? channels : 1};
        ports.output_dims = output_dims;

        Fp16HPostBdaFwd primitive;
        EXPECT_EQ(primitive.isApplicable(makeProblem(ports), Constraints{0}).getStatus(),
                  channels == limit ? FTRAIN_STATUS_UNSUPPORTED : FTRAIN_STATUS_SUCCESS);
    }
}

TEST(Fp16HPostBdaFwdTest, ChecksEmptyBatchContract) {
    ProblemPorts ports;

    const std::int64_t h_res_dims[]    = {0, ports.h_res_dims[1], ports.h_res_dims[2], ports.h_res_dims[3]};
    ports.h_res_dims                   = h_res_dims;
    ports.h_res                        = nullptr;
    const std::int64_t residual_dims[] = {0, ports.residual_dims[1], ports.residual_dims[2], ports.residual_dims[3]};
    ports.residual_dims                = residual_dims;
    ports.residual                     = nullptr;
    const std::int64_t h_post_dims[]   = {0, ports.h_post_dims[1], ports.h_post_dims[2]};
    ports.h_post_dims                  = h_post_dims;
    ports.h_post                       = nullptr;
    const std::int64_t x_dims[]        = {0, ports.x_dims[1], ports.x_dims[2]};
    ports.x_dims                       = x_dims;
    ports.x                            = nullptr;
    ports.bias                         = nullptr;
    const std::int64_t output_dims[]   = {0, ports.output_dims[1], ports.output_dims[2], ports.output_dims[3]};
    ports.output_dims                  = output_dims;
    ports.output                       = nullptr;
    const auto empty                   = makeProblem(ports);
    Fp16HPostBdaFwd primitive;
    ASSERT_TRUE(primitive.isApplicable(empty, Constraints{0}).isSuccess());
    primitive.configure(empty);
    EXPECT_NO_THROW(primitive.execute(Resources{nullptr, 0, nullptr}));
}

}  // namespace ftrain
