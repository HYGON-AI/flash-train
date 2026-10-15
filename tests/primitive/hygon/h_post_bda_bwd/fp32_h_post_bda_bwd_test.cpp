// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/h_post_bda_bwd/problem.hpp"
#include "flash_train/primitive/hygon/h_post_bda_bwd/fp32_h_post_bda_bwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kGradOutputDims[4] = {2, 1, 2, 16};
constexpr std::int64_t kResidualDims[4] = {2, 1, 2, 16};
constexpr std::int64_t kXDims[3] = {2, 1, 16};
constexpr std::int64_t kBiasDims[1] = {16};
constexpr std::int64_t kHResDims[4] = {2, 1, 2, 2};
constexpr std::int64_t kHPostDims[3] = {2, 1, 2};
constexpr std::int64_t kGradResidualDims[4] = {2, 1, 2, 16};
constexpr std::int64_t kGradXDims[3] = {2, 1, 16};
constexpr std::int64_t kGradHResDims[4] = {2, 1, 2, 2};
constexpr std::int64_t kGradHPostDims[3] = {2, 1, 2};
constexpr std::int64_t kGradBiasDims[1] = {16};

struct ProblemPorts {
    void* grad_output                      = kDefaultGradOutputMemory;
    const std::int64_t* grad_output_dims   = kGradOutputDims;
    FTrainNumericType grad_output_type     = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_output_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* residual                         = kDefaultResidualMemory;
    const std::int64_t* residual_dims      = kResidualDims;
    FTrainNumericType residual_type        = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType residual_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* x                                = kDefaultXMemory;
    const std::int64_t* x_dims             = kXDims;
    FTrainNumericType x_type               = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType x_index                = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* bias                             = kDefaultBiasMemory;
    const std::int64_t* bias_dims          = kBiasDims;
    FTrainNumericType bias_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType bias_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* h_res                            = kDefaultHResMemory;
    const std::int64_t* h_res_dims         = kHResDims;
    FTrainNumericType h_res_type           = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType h_res_index            = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* h_post                           = kDefaultHPostMemory;
    const std::int64_t* h_post_dims        = kHPostDims;
    FTrainNumericType h_post_type          = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType h_post_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_residual                    = kDefaultGradResidualMemory;
    const std::int64_t* grad_residual_dims = kGradResidualDims;
    FTrainNumericType grad_residual_type   = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_residual_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_x                           = kDefaultGradXMemory;
    const std::int64_t* grad_x_dims        = kGradXDims;
    FTrainNumericType grad_x_type          = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_x_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_h_res                       = kDefaultGradHResMemory;
    const std::int64_t* grad_h_res_dims    = kGradHResDims;
    FTrainNumericType grad_h_res_type      = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_h_res_index       = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_h_post                      = kDefaultGradHPostMemory;
    const std::int64_t* grad_h_post_dims   = kGradHPostDims;
    FTrainNumericType grad_h_post_type     = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_h_post_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_bias                        = kDefaultGradBiasMemory;
    const std::int64_t* grad_bias_dims     = kGradBiasDims;
    FTrainNumericType grad_bias_type       = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_bias_index        = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type         = FTRAIN_NUMERIC_TYPE_FP32;
    alignas(16) static unsigned char kDefaultGradOutputMemory[16];
    alignas(16) static unsigned char kDefaultResidualMemory[16];
    alignas(16) static unsigned char kDefaultXMemory[16];
    alignas(16) static unsigned char kDefaultBiasMemory[16];
    alignas(16) static unsigned char kDefaultHResMemory[16];
    alignas(16) static unsigned char kDefaultHPostMemory[16];
    alignas(16) static unsigned char kDefaultGradResidualMemory[16];
    alignas(16) static unsigned char kDefaultGradXMemory[16];
    alignas(16) static unsigned char kDefaultGradHResMemory[16];
    alignas(16) static unsigned char kDefaultGradHPostMemory[16];
    alignas(16) static unsigned char kDefaultGradBiasMemory[16];
};

alignas(16) unsigned char ProblemPorts::kDefaultGradOutputMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultResidualMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultXMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultBiasMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultHResMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultHPostMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradResidualMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradXMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradHResMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradHPostMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradBiasMemory[16];

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

HPostBdaBwdProblem makeProblem(const ProblemPorts& ports) {
    return HPostBdaBwdProblem{
        makeDeviceTensor(ports.grad_output, ports.grad_output_dims, 4, ports.grad_output_type, ports.grad_output_index),
        makeDeviceTensor(ports.residual, ports.residual_dims, 4, ports.residual_type, ports.residual_index),
        makeDeviceTensor(ports.x, ports.x_dims, 3, ports.x_type, ports.x_index),
        makeDeviceTensor(ports.bias, ports.bias_dims, 1, ports.bias_type, ports.bias_index),
        makeDeviceTensor(ports.h_res, ports.h_res_dims, 4, ports.h_res_type, ports.h_res_index),
        makeDeviceTensor(ports.h_post, ports.h_post_dims, 3, ports.h_post_type, ports.h_post_index),
        makeDeviceTensor(ports.grad_residual, ports.grad_residual_dims, 4, ports.grad_residual_type,
                         ports.grad_residual_index),
        makeDeviceTensor(ports.grad_x, ports.grad_x_dims, 3, ports.grad_x_type, ports.grad_x_index),
        makeDeviceTensor(ports.grad_h_res, ports.grad_h_res_dims, 4, ports.grad_h_res_type, ports.grad_h_res_index),
        makeDeviceTensor(ports.grad_h_post, ports.grad_h_post_dims, 3, ports.grad_h_post_type, ports.grad_h_post_index),
        makeDeviceTensor(ports.grad_bias, ports.grad_bias_dims, 1, ports.grad_bias_type, ports.grad_bias_index),
        HPostBdaBwdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

Tensor tensor(void* memory, std::initializer_list<std::int64_t> shape) {
    return Tensor{StorageView{FTrainStorageView{memory, shape.begin(), nullptr, static_cast<std::uint8_t>(shape.size()),
                                                FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
}

HPostBdaBwdProblem makeProblem(void* memory, std::int64_t batch, std::int64_t channels) {
    const auto x          = tensor(memory, {batch, 1, 1, channels});
    const auto h_pre      = tensor(memory, {batch, 1, 1});
    const auto h_post     = tensor(memory, {batch, 1, 1});
    const auto h_res      = tensor(memory, {batch, 1, 1, 1});
    const auto aggregated = tensor(memory, {batch, 1, channels});
    const auto bias       = tensor(memory, {channels});
    return {x, x,          aggregated, bias,   h_res, h_post,
            x, aggregated, h_res,      h_post, bias,  HPostBdaBwdAttributes{FTRAIN_NUMERIC_TYPE_FP32}};
}

}  // namespace

TEST(Fp32HPostBdaBwdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_grad_output[16];
    rebound.grad_output = other_grad_output;
    alignas(16) static unsigned char other_residual[16];
    rebound.residual = other_residual;
    alignas(16) static unsigned char other_x[16];
    rebound.x = other_x;
    alignas(16) static unsigned char other_bias[16];
    rebound.bias = other_bias;
    alignas(16) static unsigned char other_h_res[16];
    rebound.h_res = other_h_res;
    alignas(16) static unsigned char other_h_post[16];
    rebound.h_post = other_h_post;
    alignas(16) static unsigned char other_grad_residual[16];
    rebound.grad_residual = other_grad_residual;
    alignas(16) static unsigned char other_grad_x[16];
    rebound.grad_x = other_grad_x;
    alignas(16) static unsigned char other_grad_h_res[16];
    rebound.grad_h_res = other_grad_h_res;
    alignas(16) static unsigned char other_grad_h_post[16];
    rebound.grad_h_post = other_grad_h_post;
    alignas(16) static unsigned char other_grad_bias[16];
    rebound.grad_bias = other_grad_bias;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp32HPostBdaBwdTest, AcceptsContinuousAlignedOperands) {
    Fp32HPostBdaBwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp32HPostBdaBwd");
}

TEST(Fp32HPostBdaBwdTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp32HPostBdaBwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_output;
    unsupported_grad_output.grad_output_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_output), kConstraints).isSuccess());
    ProblemPorts unsupported_residual;
    unsupported_residual.residual_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_residual), kConstraints).isSuccess());
    ProblemPorts unsupported_x;
    unsupported_x.x_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_x), kConstraints).isSuccess());
    ProblemPorts unsupported_bias;
    unsupported_bias.bias_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_bias), kConstraints).isSuccess());
    ProblemPorts unsupported_h_res;
    unsupported_h_res.h_res_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_h_res), kConstraints).isSuccess());
    ProblemPorts unsupported_h_post;
    unsupported_h_post.h_post_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_h_post), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_residual;
    unsupported_grad_residual.grad_residual_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_residual), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_x;
    unsupported_grad_x.grad_x_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_x), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_h_res;
    unsupported_grad_h_res.grad_h_res_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_h_res), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_h_post;
    unsupported_grad_h_post.grad_h_post_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_h_post), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_bias;
    unsupported_grad_bias.grad_bias_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_bias), kConstraints).isSuccess());
}

TEST(Fp32HPostBdaBwdTest, RejectsNonContinuousIndexTypes) {
    Fp32HPostBdaBwd primitive;
    ProblemPorts strided_grad_output;
    strided_grad_output.grad_output_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_output), kConstraints).isSuccess());
    ProblemPorts strided_residual;
    strided_residual.residual_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_residual), kConstraints).isSuccess());
    ProblemPorts strided_x;
    strided_x.x_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_x), kConstraints).isSuccess());
    ProblemPorts strided_bias;
    strided_bias.bias_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_bias), kConstraints).isSuccess());
    ProblemPorts strided_h_res;
    strided_h_res.h_res_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_h_res), kConstraints).isSuccess());
    ProblemPorts strided_h_post;
    strided_h_post.h_post_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_h_post), kConstraints).isSuccess());
    ProblemPorts strided_grad_residual;
    strided_grad_residual.grad_residual_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_residual), kConstraints).isSuccess());
    ProblemPorts strided_grad_x;
    strided_grad_x.grad_x_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_x), kConstraints).isSuccess());
    ProblemPorts strided_grad_h_res;
    strided_grad_h_res.grad_h_res_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_h_res), kConstraints).isSuccess());
    ProblemPorts strided_grad_h_post;
    strided_grad_h_post.grad_h_post_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_h_post), kConstraints).isSuccess());
    ProblemPorts strided_grad_bias;
    strided_grad_bias.grad_bias_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_bias), kConstraints).isSuccess());
}

TEST(Fp32HPostBdaBwdTest, RejectsMisalignedAddresses) {
    Fp32HPostBdaBwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_grad_output;
    misaligned_grad_output.grad_output = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_output), kConstraints).isSuccess());
    ProblemPorts misaligned_residual;
    misaligned_residual.residual = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_residual), kConstraints).isSuccess());
    ProblemPorts misaligned_x;
    misaligned_x.x = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_x), kConstraints).isSuccess());
    ProblemPorts misaligned_bias;
    misaligned_bias.bias = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_bias), kConstraints).isSuccess());
    ProblemPorts misaligned_h_res;
    misaligned_h_res.h_res = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_h_res), kConstraints).isSuccess());
    ProblemPorts misaligned_h_post;
    misaligned_h_post.h_post = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_h_post), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_residual;
    misaligned_grad_residual.grad_residual = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_residual), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_x;
    misaligned_grad_x.grad_x = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_x), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_h_res;
    misaligned_grad_h_res.grad_h_res = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_h_res), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_h_post;
    misaligned_grad_h_post.grad_h_post = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_h_post), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_bias;
    misaligned_grad_bias.grad_bias = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_bias), kConstraints).isSuccess());
}

TEST(Fp32HPostBdaBwdTest, ChecksRequiredAndOptionalAddresses) {
    Fp32HPostBdaBwd primitive;
    ProblemPorts null_grad_output;
    null_grad_output.grad_output = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_output), kConstraints).isSuccess());
    ProblemPorts null_residual;
    null_residual.residual = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_residual), kConstraints).isSuccess());
    ProblemPorts null_x;
    null_x.x = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_x), kConstraints).isSuccess());
    ProblemPorts null_bias;
    null_bias.bias = nullptr;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(null_bias), kConstraints).isSuccess());
    ProblemPorts null_h_res;
    null_h_res.h_res = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_h_res), kConstraints).isSuccess());
    ProblemPorts null_h_post;
    null_h_post.h_post = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_h_post), kConstraints).isSuccess());
    ProblemPorts null_grad_residual;
    null_grad_residual.grad_residual = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_residual), kConstraints).isSuccess());
    ProblemPorts null_grad_x;
    null_grad_x.grad_x = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_x), kConstraints).isSuccess());
    ProblemPorts null_grad_h_res;
    null_grad_h_res.grad_h_res = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_h_res), kConstraints).isSuccess());
    ProblemPorts null_grad_h_post;
    null_grad_h_post.grad_h_post = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_h_post), kConstraints).isSuccess());
    ProblemPorts null_grad_bias;
    null_grad_bias.grad_bias = nullptr;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(null_grad_bias), kConstraints).isSuccess());
}

TEST(Fp32HPostBdaBwdTest, ClonesIntoIndependentConfigurableCopies) {
    Fp32HPostBdaBwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp32HPostBdaBwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<HPostBdaBwdProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<HPostBdaBwdProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp32HPostBdaBwdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original                     = makeProblem(ports).getProblemKey();
        const std::int64_t grad_output_dims[]   = {2, 1, 33, 16};
        ports.grad_output_dims                  = grad_output_dims;
        const std::int64_t residual_dims[]      = {2, 1, 33, 16};
        ports.residual_dims                     = residual_dims;
        const std::int64_t h_res_dims[]         = {2, 1, 33, 33};
        ports.h_res_dims                        = h_res_dims;
        const std::int64_t h_post_dims[]        = {2, 1, 33};
        ports.h_post_dims                       = h_post_dims;
        const std::int64_t grad_residual_dims[] = {2, 1, 33, 16};
        ports.grad_residual_dims                = grad_residual_dims;
        const std::int64_t grad_h_res_dims[]    = {2, 1, 33, 33};
        ports.grad_h_res_dims                   = grad_h_res_dims;
        const std::int64_t grad_h_post_dims[]   = {2, 1, 33};
        ports.grad_h_post_dims                  = grad_h_post_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp32HPostBdaBwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp32HPostBdaBwdRegressionTest, ChecksChannelTileArithmeticAtTheIntBoundary) {
    alignas(16) unsigned char memory[16]{};
    const Fp32HPostBdaBwd kernel;
    constexpr std::int64_t max = std::numeric_limits<int>::max();
    const auto unsafe          = makeProblem(memory, 1, max);
    EXPECT_EQ(kernel.isApplicable(unsafe, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    const auto safe = makeProblem(memory, 1, max - 255);
    EXPECT_TRUE(kernel.isApplicable(safe, Constraints{0}).isSuccess());
}

TEST(Fp32HPostBdaBwdRegressionTest, RejectsOverflowingCombinedGridAndEmptyBatch) {
    alignas(16) unsigned char memory[16]{};
    const Fp32HPostBdaBwd kernel;
    const auto oversized_grid = makeProblem(memory, std::numeric_limits<int>::max(), 1);
    EXPECT_EQ(kernel.isApplicable(oversized_grid, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    const auto empty = makeProblem(memory, 0, 16);
    EXPECT_EQ(kernel.isApplicable(empty, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST(Fp32HPostBdaBwdTest, ChecksChannelTileArithmeticBoundary) {
    constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
    for (const std::int64_t channels : {limit - 255, limit}) {
        ProblemPorts ports;

        const std::int64_t grad_output_dims[] = {
            ports.grad_output_dims[0] == 16 ? channels : 1, ports.grad_output_dims[1] == 16 ? channels : 1,
            ports.grad_output_dims[2] == 16 ? channels : 1, ports.grad_output_dims[3] == 16 ? channels : 1};
        ports.grad_output_dims = grad_output_dims;

        const std::int64_t residual_dims[] = {
            ports.residual_dims[0] == 16 ? channels : 1, ports.residual_dims[1] == 16 ? channels : 1,
            ports.residual_dims[2] == 16 ? channels : 1, ports.residual_dims[3] == 16 ? channels : 1};
        ports.residual_dims = residual_dims;

        const std::int64_t x_dims[] = {ports.x_dims[0] == 16 ? channels : 1, ports.x_dims[1] == 16 ? channels : 1,
                                       ports.x_dims[2] == 16 ? channels : 1};
        ports.x_dims                = x_dims;

        const std::int64_t bias_dims[] = {ports.bias_dims[0] == 16 ? channels : 1};
        ports.bias_dims                = bias_dims;

        const std::int64_t h_res_dims[] = {
            ports.h_res_dims[0] == 16 ? channels : 1, ports.h_res_dims[1] == 16 ? channels : 1,
            ports.h_res_dims[2] == 16 ? channels : 1, ports.h_res_dims[3] == 16 ? channels : 1};
        ports.h_res_dims = h_res_dims;

        const std::int64_t h_post_dims[] = {ports.h_post_dims[0] == 16 ? channels : 1,
                                            ports.h_post_dims[1] == 16 ? channels : 1,
                                            ports.h_post_dims[2] == 16 ? channels : 1};
        ports.h_post_dims                = h_post_dims;

        const std::int64_t grad_residual_dims[] = {
            ports.grad_residual_dims[0] == 16 ? channels : 1, ports.grad_residual_dims[1] == 16 ? channels : 1,
            ports.grad_residual_dims[2] == 16 ? channels : 1, ports.grad_residual_dims[3] == 16 ? channels : 1};
        ports.grad_residual_dims = grad_residual_dims;

        const std::int64_t grad_x_dims[] = {ports.grad_x_dims[0] == 16 ? channels : 1,
                                            ports.grad_x_dims[1] == 16 ? channels : 1,
                                            ports.grad_x_dims[2] == 16 ? channels : 1};
        ports.grad_x_dims                = grad_x_dims;

        const std::int64_t grad_h_res_dims[] = {
            ports.grad_h_res_dims[0] == 16 ? channels : 1, ports.grad_h_res_dims[1] == 16 ? channels : 1,
            ports.grad_h_res_dims[2] == 16 ? channels : 1, ports.grad_h_res_dims[3] == 16 ? channels : 1};
        ports.grad_h_res_dims = grad_h_res_dims;

        const std::int64_t grad_h_post_dims[] = {ports.grad_h_post_dims[0] == 16 ? channels : 1,
                                                 ports.grad_h_post_dims[1] == 16 ? channels : 1,
                                                 ports.grad_h_post_dims[2] == 16 ? channels : 1};
        ports.grad_h_post_dims                = grad_h_post_dims;

        const std::int64_t grad_bias_dims[] = {ports.grad_bias_dims[0] == 16 ? channels : 1};
        ports.grad_bias_dims                = grad_bias_dims;

        Fp32HPostBdaBwd primitive;
        EXPECT_EQ(primitive.isApplicable(makeProblem(ports), Constraints{0}).getStatus(),
                  channels == limit ? FTRAIN_STATUS_UNSUPPORTED : FTRAIN_STATUS_SUCCESS);
    }
}

TEST(Fp32HPostBdaBwdTest, ChecksEmptyBatchContract) {
    ProblemPorts ports;

    const std::int64_t grad_output_dims[] = {0, ports.grad_output_dims[1], ports.grad_output_dims[2],
                                             ports.grad_output_dims[3]};
    ports.grad_output_dims                = grad_output_dims;
    ports.grad_output                     = nullptr;
    const std::int64_t residual_dims[]    = {0, ports.residual_dims[1], ports.residual_dims[2], ports.residual_dims[3]};
    ports.residual_dims                   = residual_dims;
    ports.residual                        = nullptr;
    const std::int64_t x_dims[]           = {0, ports.x_dims[1], ports.x_dims[2]};
    ports.x_dims                          = x_dims;
    ports.x                               = nullptr;
    ports.bias                            = nullptr;
    const std::int64_t h_res_dims[]       = {0, ports.h_res_dims[1], ports.h_res_dims[2], ports.h_res_dims[3]};
    ports.h_res_dims                      = h_res_dims;
    ports.h_res                           = nullptr;
    const std::int64_t h_post_dims[]      = {0, ports.h_post_dims[1], ports.h_post_dims[2]};
    ports.h_post_dims                     = h_post_dims;
    ports.h_post                          = nullptr;
    const std::int64_t grad_residual_dims[] = {0, ports.grad_residual_dims[1], ports.grad_residual_dims[2],
                                               ports.grad_residual_dims[3]};
    ports.grad_residual_dims                = grad_residual_dims;
    ports.grad_residual                     = nullptr;
    const std::int64_t grad_x_dims[]        = {0, ports.grad_x_dims[1], ports.grad_x_dims[2]};
    ports.grad_x_dims                       = grad_x_dims;
    ports.grad_x                            = nullptr;
    const std::int64_t grad_h_res_dims[]    = {0, ports.grad_h_res_dims[1], ports.grad_h_res_dims[2],
                                               ports.grad_h_res_dims[3]};
    ports.grad_h_res_dims                   = grad_h_res_dims;
    ports.grad_h_res                        = nullptr;
    const std::int64_t grad_h_post_dims[]   = {0, ports.grad_h_post_dims[1], ports.grad_h_post_dims[2]};
    ports.grad_h_post_dims                  = grad_h_post_dims;
    ports.grad_h_post                       = nullptr;
    ports.grad_bias                         = nullptr;
    const auto empty                        = makeProblem(ports);
    Fp32HPostBdaBwd primitive;
    EXPECT_EQ(primitive.isApplicable(empty, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

}  // namespace ftrain
