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
#include "flash_train/family/h_aggregate_fwd/problem.hpp"
#include "flash_train/primitive/hygon/h_aggregate_fwd/fp32_h_aggregate_fwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kXDims[4]          = {2, 1, 2, 16};
constexpr std::int64_t kHPreDims[3]       = {2, 1, 2};
constexpr std::int64_t kAggregatedDims[3] = {2, 1, 16};

struct ProblemPorts {
    void* x                             = kDefaultXMemory;
    const std::int64_t* x_dims          = kXDims;
    FTrainNumericType x_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType x_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* h_pre                         = kDefaultHPreMemory;
    const std::int64_t* h_pre_dims      = kHPreDims;
    FTrainNumericType h_pre_type        = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType h_pre_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* aggregated                    = kDefaultAggregatedMemory;
    const std::int64_t* aggregated_dims = kAggregatedDims;
    FTrainNumericType aggregated_type   = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType aggregated_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type      = FTRAIN_NUMERIC_TYPE_FP32;
    alignas(16) static unsigned char kDefaultXMemory[16];
    alignas(16) static unsigned char kDefaultHPreMemory[16];
    alignas(16) static unsigned char kDefaultAggregatedMemory[16];
};

alignas(16) unsigned char ProblemPorts::kDefaultXMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultHPreMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultAggregatedMemory[16];

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

HAggregateFwdProblem makeProblem(const ProblemPorts& ports) {
    return HAggregateFwdProblem{
        makeDeviceTensor(ports.x, ports.x_dims, 4, ports.x_type, ports.x_index),
        makeDeviceTensor(ports.h_pre, ports.h_pre_dims, 3, ports.h_pre_type, ports.h_pre_index),
        makeDeviceTensor(ports.aggregated, ports.aggregated_dims, 3, ports.aggregated_type, ports.aggregated_index),
        HAggregateFwdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

// StorageView copies dims into its own vector during construction; shape is not retained.
Tensor tensor(void* memory, std::initializer_list<std::int64_t> shape,
              FTrainNumericType dtype = FTRAIN_NUMERIC_TYPE_FP32) {
    return Tensor{StorageView{FTrainStorageView{memory, shape.begin(), nullptr, static_cast<std::uint8_t>(shape.size()),
                                                dtype, FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
}

HAggregateFwdProblem makeProblem(void* memory, std::int64_t batch, std::int64_t channels) {
    const auto x          = tensor(memory, {batch, 1, 1, channels});
    const auto h_pre      = tensor(memory, {batch, 1, 1});
    const auto h_post     = tensor(memory, {batch, 1, 1});
    const auto h_res      = tensor(memory, {batch, 1, 1, 1});
    const auto aggregated = tensor(memory, {batch, 1, channels});
    const auto bias       = tensor(memory, {channels});
    return {x, h_pre, aggregated, HAggregateFwdAttributes{FTRAIN_NUMERIC_TYPE_FP32}};
}

}  // namespace

TEST(Fp32HAggregateFwdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_x[16];
    rebound.x = other_x;
    alignas(16) static unsigned char other_h_pre[16];
    rebound.h_pre = other_h_pre;
    alignas(16) static unsigned char other_aggregated[16];
    rebound.aggregated = other_aggregated;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp32HAggregateFwdTest, AcceptsContinuousAlignedOperands) {
    Fp32HAggregateFwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp32HAggregateFwd");
}

TEST(Fp32HAggregateFwdTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp32HAggregateFwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_x;
    unsupported_x.x_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_x), kConstraints).isSuccess());
    ProblemPorts unsupported_h_pre;
    unsupported_h_pre.h_pre_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_h_pre), kConstraints).isSuccess());
    ProblemPorts unsupported_aggregated;
    unsupported_aggregated.aggregated_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_aggregated), kConstraints).isSuccess());
}

TEST(Fp32HAggregateFwdTest, RejectsNonContinuousIndexTypes) {
    Fp32HAggregateFwd primitive;
    ProblemPorts strided_x;
    strided_x.x_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_x), kConstraints).isSuccess());
    ProblemPorts strided_h_pre;
    strided_h_pre.h_pre_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_h_pre), kConstraints).isSuccess());
    ProblemPorts strided_aggregated;
    strided_aggregated.aggregated_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_aggregated), kConstraints).isSuccess());
}

TEST(Fp32HAggregateFwdTest, RejectsMisalignedAddresses) {
    Fp32HAggregateFwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_x;
    misaligned_x.x = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_x), kConstraints).isSuccess());
    ProblemPorts misaligned_h_pre;
    misaligned_h_pre.h_pre = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_h_pre), kConstraints).isSuccess());
    ProblemPorts misaligned_aggregated;
    misaligned_aggregated.aggregated = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_aggregated), kConstraints).isSuccess());
}

TEST(Fp32HAggregateFwdTest, ChecksRequiredAndOptionalAddresses) {
    Fp32HAggregateFwd primitive;
    ProblemPorts null_x;
    null_x.x = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_x), kConstraints).isSuccess());
    ProblemPorts null_h_pre;
    null_h_pre.h_pre = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_h_pre), kConstraints).isSuccess());
    ProblemPorts null_aggregated;
    null_aggregated.aggregated = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_aggregated), kConstraints).isSuccess());
}

TEST(Fp32HAggregateFwdTest, ClonesIntoIndependentConfigurableCopies) {
    Fp32HAggregateFwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp32HAggregateFwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<HAggregateFwdProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<HAggregateFwdProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp32HAggregateFwdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original             = makeProblem(ports).getProblemKey();
        const std::int64_t x_dims[]     = {2, 1, 33, 16};
        ports.x_dims                    = x_dims;
        const std::int64_t h_pre_dims[] = {2, 1, 33};
        ports.h_pre_dims                = h_pre_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp32HAggregateFwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp32HAggregateFwdRegressionTest, ChecksChannelTileArithmeticAtTheIntBoundary) {
    alignas(16) unsigned char memory[16]{};
    const Fp32HAggregateFwd kernel;
    constexpr std::int64_t max = std::numeric_limits<int>::max();
    const auto unsafe          = makeProblem(memory, 1, max);
    EXPECT_EQ(kernel.isApplicable(unsafe, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    const auto safe = makeProblem(memory, 1, max - 255);
    EXPECT_TRUE(kernel.isApplicable(safe, Constraints{0}).isSuccess());
}

TEST(Fp32HAggregateFwdRegressionTest, EmptyOutputAcceptsNullPointersAndExecutesWithoutLaunching) {
    const auto empty = makeProblem(nullptr, 0, 16);
    Fp32HAggregateFwd kernel;
    ASSERT_TRUE(empty.hasEmptyOutput());
    ASSERT_TRUE(kernel.isApplicable(empty, Constraints{0}).isSuccess());
    kernel.configure(empty);
    EXPECT_NO_THROW(kernel.execute(Resources{nullptr, 0, nullptr}));
}

TEST(Fp32HAggregateFwdTest, ChecksChannelTileArithmeticBoundary) {
    constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
    for (const std::int64_t channels : {limit - 255, limit}) {
        ProblemPorts ports;

        const std::int64_t x_dims[] = {ports.x_dims[0] == 16 ? channels : 1, ports.x_dims[1] == 16 ? channels : 1,
                                       ports.x_dims[2] == 16 ? channels : 1, ports.x_dims[3] == 16 ? channels : 1};
        ports.x_dims                = x_dims;

        const std::int64_t h_pre_dims[] = {ports.h_pre_dims[0] == 16 ? channels : 1,
                                           ports.h_pre_dims[1] == 16 ? channels : 1,
                                           ports.h_pre_dims[2] == 16 ? channels : 1};
        ports.h_pre_dims                = h_pre_dims;

        const std::int64_t aggregated_dims[] = {ports.aggregated_dims[0] == 16 ? channels : 1,
                                                ports.aggregated_dims[1] == 16 ? channels : 1,
                                                ports.aggregated_dims[2] == 16 ? channels : 1};
        ports.aggregated_dims                = aggregated_dims;

        Fp32HAggregateFwd primitive;
        EXPECT_EQ(primitive.isApplicable(makeProblem(ports), Constraints{0}).getStatus(),
                  channels == limit ? FTRAIN_STATUS_UNSUPPORTED : FTRAIN_STATUS_SUCCESS);
    }
}

TEST(Fp32HAggregateFwdTest, ChecksEmptyBatchContract) {
    ProblemPorts ports;

    const std::int64_t x_dims[]          = {0, ports.x_dims[1], ports.x_dims[2], ports.x_dims[3]};
    ports.x_dims                         = x_dims;
    ports.x                              = nullptr;
    const std::int64_t h_pre_dims[]      = {0, ports.h_pre_dims[1], ports.h_pre_dims[2]};
    ports.h_pre_dims                     = h_pre_dims;
    ports.h_pre                          = nullptr;
    const std::int64_t aggregated_dims[] = {0, ports.aggregated_dims[1], ports.aggregated_dims[2]};
    ports.aggregated_dims                = aggregated_dims;
    ports.aggregated                     = nullptr;
    const auto empty                     = makeProblem(ports);
    Fp32HAggregateFwd primitive;
    ASSERT_TRUE(primitive.isApplicable(empty, Constraints{0}).isSuccess());
    primitive.configure(empty);
    EXPECT_NO_THROW(primitive.execute(Resources{nullptr, 0, nullptr}));
}

}  // namespace ftrain
