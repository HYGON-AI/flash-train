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
#include "flash_train/family/h_aggregate_bwd/problem.hpp"
#include "flash_train/primitive/hygon/h_aggregate_bwd/fp32_h_aggregate_bwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kGradAggregatedDims[3] = {2, 1, 16};
constexpr std::int64_t kXDims[4]              = {2, 1, 2, 16};
constexpr std::int64_t kHPreDims[3]           = {2, 1, 2};
constexpr std::int64_t kGradXDims[4]          = {2, 1, 2, 16};
constexpr std::int64_t kGradHPreDims[3]       = {2, 1, 2};

struct ProblemPorts {
    void* grad_aggregated                    = kDefaultGradAggregatedMemory;
    const std::int64_t* grad_aggregated_dims = kGradAggregatedDims;
    FTrainNumericType grad_aggregated_type   = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_aggregated_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* x                                  = kDefaultXMemory;
    const std::int64_t* x_dims               = kXDims;
    FTrainNumericType x_type                 = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType x_index                  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* h_pre                              = kDefaultHPreMemory;
    const std::int64_t* h_pre_dims           = kHPreDims;
    FTrainNumericType h_pre_type             = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType h_pre_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_x                             = kDefaultGradXMemory;
    const std::int64_t* grad_x_dims          = kGradXDims;
    FTrainNumericType grad_x_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_x_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_h_pre                         = kDefaultGradHPreMemory;
    const std::int64_t* grad_h_pre_dims      = kGradHPreDims;
    FTrainNumericType grad_h_pre_type        = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_h_pre_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type           = FTRAIN_NUMERIC_TYPE_FP32;
    alignas(16) static unsigned char kDefaultGradAggregatedMemory[16];
    alignas(16) static unsigned char kDefaultXMemory[16];
    alignas(16) static unsigned char kDefaultHPreMemory[16];
    alignas(16) static unsigned char kDefaultGradXMemory[16];
    alignas(16) static unsigned char kDefaultGradHPreMemory[16];
};

alignas(16) unsigned char ProblemPorts::kDefaultGradAggregatedMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultXMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultHPreMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradXMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradHPreMemory[16];

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

HAggregateBwdProblem makeProblem(const ProblemPorts& ports) {
    return HAggregateBwdProblem{
        makeDeviceTensor(ports.grad_aggregated, ports.grad_aggregated_dims, 3, ports.grad_aggregated_type,
                         ports.grad_aggregated_index),
        makeDeviceTensor(ports.x, ports.x_dims, 4, ports.x_type, ports.x_index),
        makeDeviceTensor(ports.h_pre, ports.h_pre_dims, 3, ports.h_pre_type, ports.h_pre_index),
        makeDeviceTensor(ports.grad_x, ports.grad_x_dims, 4, ports.grad_x_type, ports.grad_x_index),
        makeDeviceTensor(ports.grad_h_pre, ports.grad_h_pre_dims, 3, ports.grad_h_pre_type, ports.grad_h_pre_index),
        HAggregateBwdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

// StorageView copies dims into its own vector during construction; shape is not retained.
Tensor tensor(void* memory, std::initializer_list<std::int64_t> shape) {
    return Tensor{StorageView{FTrainStorageView{memory, shape.begin(), nullptr, static_cast<std::uint8_t>(shape.size()),
                                                FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
}

HAggregateBwdProblem makeProblem(void* memory, std::int64_t batch, std::int64_t channels) {
    const auto x          = tensor(memory, {batch, 1, 1, channels});
    const auto h_pre      = tensor(memory, {batch, 1, 1});
    const auto h_post     = tensor(memory, {batch, 1, 1});
    const auto h_res      = tensor(memory, {batch, 1, 1, 1});
    const auto aggregated = tensor(memory, {batch, 1, channels});
    const auto bias       = tensor(memory, {channels});
    return {aggregated, x, h_pre, x, h_pre, HAggregateBwdAttributes{FTRAIN_NUMERIC_TYPE_FP32}};
}

}  // namespace

TEST(Fp32HAggregateBwdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_grad_aggregated[16];
    rebound.grad_aggregated = other_grad_aggregated;
    alignas(16) static unsigned char other_x[16];
    rebound.x = other_x;
    alignas(16) static unsigned char other_h_pre[16];
    rebound.h_pre = other_h_pre;
    alignas(16) static unsigned char other_grad_x[16];
    rebound.grad_x = other_grad_x;
    alignas(16) static unsigned char other_grad_h_pre[16];
    rebound.grad_h_pre = other_grad_h_pre;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp32HAggregateBwdTest, AcceptsContinuousAlignedOperands) {
    Fp32HAggregateBwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp32HAggregateBwd");
}

TEST(Fp32HAggregateBwdTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp32HAggregateBwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_aggregated;
    unsupported_grad_aggregated.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_aggregated), kConstraints).isSuccess());
    ProblemPorts unsupported_x;
    unsupported_x.x_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_x), kConstraints).isSuccess());
    ProblemPorts unsupported_h_pre;
    unsupported_h_pre.h_pre_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_h_pre), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_x;
    unsupported_grad_x.grad_x_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_x), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_h_pre;
    unsupported_grad_h_pre.grad_h_pre_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_h_pre), kConstraints).isSuccess());
}

TEST(Fp32HAggregateBwdTest, RejectsNonContinuousIndexTypes) {
    Fp32HAggregateBwd primitive;
    ProblemPorts strided_grad_aggregated;
    strided_grad_aggregated.grad_aggregated_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_aggregated), kConstraints).isSuccess());
    ProblemPorts strided_x;
    strided_x.x_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_x), kConstraints).isSuccess());
    ProblemPorts strided_h_pre;
    strided_h_pre.h_pre_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_h_pre), kConstraints).isSuccess());
    ProblemPorts strided_grad_x;
    strided_grad_x.grad_x_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_x), kConstraints).isSuccess());
    ProblemPorts strided_grad_h_pre;
    strided_grad_h_pre.grad_h_pre_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_h_pre), kConstraints).isSuccess());
}

TEST(Fp32HAggregateBwdTest, RejectsMisalignedAddresses) {
    Fp32HAggregateBwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_grad_aggregated;
    misaligned_grad_aggregated.grad_aggregated = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_aggregated), kConstraints).isSuccess());
    ProblemPorts misaligned_x;
    misaligned_x.x = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_x), kConstraints).isSuccess());
    ProblemPorts misaligned_h_pre;
    misaligned_h_pre.h_pre = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_h_pre), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_x;
    misaligned_grad_x.grad_x = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_x), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_h_pre;
    misaligned_grad_h_pre.grad_h_pre = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_h_pre), kConstraints).isSuccess());
}

TEST(Fp32HAggregateBwdTest, ChecksRequiredAndOptionalAddresses) {
    Fp32HAggregateBwd primitive;
    ProblemPorts null_grad_aggregated;
    null_grad_aggregated.grad_aggregated = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_aggregated), kConstraints).isSuccess());
    ProblemPorts null_x;
    null_x.x = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_x), kConstraints).isSuccess());
    ProblemPorts null_h_pre;
    null_h_pre.h_pre = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_h_pre), kConstraints).isSuccess());
    ProblemPorts null_grad_x;
    null_grad_x.grad_x = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_x), kConstraints).isSuccess());
    ProblemPorts null_grad_h_pre;
    null_grad_h_pre.grad_h_pre = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_h_pre), kConstraints).isSuccess());
}

TEST(Fp32HAggregateBwdTest, ClonesIntoIndependentConfigurableCopies) {
    Fp32HAggregateBwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp32HAggregateBwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<HAggregateBwdProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<HAggregateBwdProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp32HAggregateBwdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original                  = makeProblem(ports).getProblemKey();
        const std::int64_t x_dims[]          = {2, 1, 33, 16};
        ports.x_dims                         = x_dims;
        const std::int64_t h_pre_dims[]      = {2, 1, 33};
        ports.h_pre_dims                     = h_pre_dims;
        const std::int64_t grad_x_dims[]     = {2, 1, 33, 16};
        ports.grad_x_dims                    = grad_x_dims;
        const std::int64_t grad_h_pre_dims[] = {2, 1, 33};
        ports.grad_h_pre_dims                = grad_h_pre_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp32HAggregateBwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp32HAggregateBwdRegressionTest, ChecksChannelTileArithmeticAtTheIntBoundary) {
    alignas(16) unsigned char memory[16]{};
    const Fp32HAggregateBwd kernel;
    constexpr std::int64_t max = std::numeric_limits<int>::max();
    const auto unsafe          = makeProblem(memory, 1, max);
    EXPECT_EQ(kernel.isApplicable(unsafe, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    const auto safe = makeProblem(memory, 1, max - 63);
    EXPECT_TRUE(kernel.isApplicable(safe, Constraints{0}).isSuccess());
}

TEST(Fp32HAggregateBwdRegressionTest, EmptyOutputAcceptsNullPointersAndExecutesWithoutLaunching) {
    const auto empty = makeProblem(nullptr, 0, 16);
    Fp32HAggregateBwd kernel;
    ASSERT_TRUE(empty.hasEmptyOutput());
    ASSERT_TRUE(kernel.isApplicable(empty, Constraints{0}).isSuccess());
    kernel.configure(empty);
    EXPECT_NO_THROW(kernel.execute(Resources{nullptr, 0, nullptr}));
}

TEST(Fp32HAggregateBwdTest, ChecksChannelTileArithmeticBoundary) {
    constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
    for (const std::int64_t channels : {limit - 63, limit}) {
        ProblemPorts ports;

        const std::int64_t grad_aggregated_dims[] = {ports.grad_aggregated_dims[0] == 16 ? channels : 1,
                                                     ports.grad_aggregated_dims[1] == 16 ? channels : 1,
                                                     ports.grad_aggregated_dims[2] == 16 ? channels : 1};
        ports.grad_aggregated_dims                = grad_aggregated_dims;

        const std::int64_t x_dims[] = {ports.x_dims[0] == 16 ? channels : 1, ports.x_dims[1] == 16 ? channels : 1,
                                       ports.x_dims[2] == 16 ? channels : 1, ports.x_dims[3] == 16 ? channels : 1};
        ports.x_dims                = x_dims;

        const std::int64_t h_pre_dims[] = {ports.h_pre_dims[0] == 16 ? channels : 1,
                                           ports.h_pre_dims[1] == 16 ? channels : 1,
                                           ports.h_pre_dims[2] == 16 ? channels : 1};
        ports.h_pre_dims                = h_pre_dims;

        const std::int64_t grad_x_dims[] = {
            ports.grad_x_dims[0] == 16 ? channels : 1, ports.grad_x_dims[1] == 16 ? channels : 1,
            ports.grad_x_dims[2] == 16 ? channels : 1, ports.grad_x_dims[3] == 16 ? channels : 1};
        ports.grad_x_dims = grad_x_dims;

        const std::int64_t grad_h_pre_dims[] = {ports.grad_h_pre_dims[0] == 16 ? channels : 1,
                                                ports.grad_h_pre_dims[1] == 16 ? channels : 1,
                                                ports.grad_h_pre_dims[2] == 16 ? channels : 1};
        ports.grad_h_pre_dims                = grad_h_pre_dims;

        Fp32HAggregateBwd primitive;
        EXPECT_EQ(primitive.isApplicable(makeProblem(ports), Constraints{0}).getStatus(),
                  channels == limit ? FTRAIN_STATUS_UNSUPPORTED : FTRAIN_STATUS_SUCCESS);
    }
}

TEST(Fp32HAggregateBwdTest, ChecksEmptyBatchContract) {
    ProblemPorts ports;

    const std::int64_t grad_aggregated_dims[] = {0, ports.grad_aggregated_dims[1], ports.grad_aggregated_dims[2]};
    ports.grad_aggregated_dims                = grad_aggregated_dims;
    ports.grad_aggregated                     = nullptr;
    const std::int64_t x_dims[]               = {0, ports.x_dims[1], ports.x_dims[2], ports.x_dims[3]};
    ports.x_dims                              = x_dims;
    ports.x                                   = nullptr;
    const std::int64_t h_pre_dims[]           = {0, ports.h_pre_dims[1], ports.h_pre_dims[2]};
    ports.h_pre_dims                          = h_pre_dims;
    ports.h_pre                               = nullptr;
    const std::int64_t grad_x_dims[]          = {0, ports.grad_x_dims[1], ports.grad_x_dims[2], ports.grad_x_dims[3]};
    ports.grad_x_dims                         = grad_x_dims;
    ports.grad_x                              = nullptr;
    const std::int64_t grad_h_pre_dims[]      = {0, ports.grad_h_pre_dims[1], ports.grad_h_pre_dims[2]};
    ports.grad_h_pre_dims                     = grad_h_pre_dims;
    ports.grad_h_pre                          = nullptr;
    const auto empty                          = makeProblem(ports);
    Fp32HAggregateBwd primitive;
    ASSERT_TRUE(primitive.isApplicable(empty, Constraints{0}).isSuccess());
    primitive.configure(empty);
    EXPECT_NO_THROW(primitive.execute(Resources{nullptr, 0, nullptr}));
}

}  // namespace ftrain
