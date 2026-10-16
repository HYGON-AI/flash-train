// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/h_aggregate_fwd/problem.hpp"
#include "flash_train/primitive/hygon/h_aggregate_fwd/mixed_fp32_h_aggregate_fwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kXDims[4] = {2, 1, 2, 16};
constexpr std::int64_t kHPreDims[3] = {2, 1, 2};
constexpr std::int64_t kAggregatedDims[3] = {2, 1, 16};

struct ProblemPorts {
    void* x                             = kDefaultXMemory;
    const std::int64_t* x_dims          = kXDims;
    FTrainNumericType x_type            = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType x_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* h_pre                         = kDefaultHPreMemory;
    const std::int64_t* h_pre_dims      = kHPreDims;
    FTrainNumericType h_pre_type        = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType h_pre_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* aggregated                    = kDefaultAggregatedMemory;
    const std::int64_t* aggregated_dims = kAggregatedDims;
    FTrainNumericType aggregated_type   = FTRAIN_NUMERIC_TYPE_BF16;
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

}  // namespace

TEST(MixedFp32HAggregateFwdTest, KeyIgnoresAllocationAddresses) {
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

TEST(MixedFp32HAggregateFwdTest, Bf16AcceptsContinuousAlignedOperands) {
    MixedFp32HAggregateFwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "MixedFp32HAggregateFwd");
}

TEST(MixedFp32HAggregateFwdTest, Bf16RejectsUnsupportedComputeAndElementTypes) {
    MixedFp32HAggregateFwd primitive;
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

TEST(MixedFp32HAggregateFwdTest, Bf16RejectsNonContinuousIndexTypes) {
    MixedFp32HAggregateFwd primitive;
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

TEST(MixedFp32HAggregateFwdTest, Bf16RejectsMisalignedAddresses) {
    MixedFp32HAggregateFwd primitive;
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

TEST(MixedFp32HAggregateFwdTest, Bf16ChecksRequiredAndOptionalAddresses) {
    MixedFp32HAggregateFwd primitive;
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

TEST(MixedFp32HAggregateFwdTest, Bf16ClonesIntoIndependentConfigurableCopies) {
    MixedFp32HAggregateFwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "MixedFp32HAggregateFwd");
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

TEST(MixedFp32HAggregateFwdTest, Fp16AcceptsContinuousAlignedOperands) {
    MixedFp32HAggregateFwd primitive;
    ProblemPorts ports;
    ports.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    ports.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "MixedFp32HAggregateFwd");
}

TEST(MixedFp32HAggregateFwdTest, Fp16RejectsUnsupportedComputeAndElementTypes) {
    MixedFp32HAggregateFwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_compute.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_compute.compute_type    = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_x;
    unsupported_x.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_x.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_x.x_type          = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_x), kConstraints).isSuccess());
    ProblemPorts unsupported_h_pre;
    unsupported_h_pre.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_h_pre.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_h_pre.h_pre_type      = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_h_pre), kConstraints).isSuccess());
    ProblemPorts unsupported_aggregated;
    unsupported_aggregated.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_aggregated.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_aggregated.aggregated_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_aggregated), kConstraints).isSuccess());
}

TEST(MixedFp32HAggregateFwdTest, Fp16RejectsNonContinuousIndexTypes) {
    MixedFp32HAggregateFwd primitive;
    ProblemPorts strided_x;
    strided_x.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    strided_x.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    strided_x.x_index         = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_x), kConstraints).isSuccess());
    ProblemPorts strided_h_pre;
    strided_h_pre.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    strided_h_pre.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    strided_h_pre.h_pre_index     = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_h_pre), kConstraints).isSuccess());
    ProblemPorts strided_aggregated;
    strided_aggregated.x_type           = FTRAIN_NUMERIC_TYPE_FP16;
    strided_aggregated.aggregated_type  = FTRAIN_NUMERIC_TYPE_FP16;
    strided_aggregated.aggregated_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_aggregated), kConstraints).isSuccess());
}

TEST(MixedFp32HAggregateFwdTest, Fp16RejectsMisalignedAddresses) {
    MixedFp32HAggregateFwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_x;
    misaligned_x.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_x.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_x.x               = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_x), kConstraints).isSuccess());
    ProblemPorts misaligned_h_pre;
    misaligned_h_pre.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_h_pre.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_h_pre.h_pre           = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_h_pre), kConstraints).isSuccess());
    ProblemPorts misaligned_aggregated;
    misaligned_aggregated.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_aggregated.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_aggregated.aggregated      = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_aggregated), kConstraints).isSuccess());
}

TEST(MixedFp32HAggregateFwdTest, Fp16ChecksRequiredAndOptionalAddresses) {
    MixedFp32HAggregateFwd primitive;
    ProblemPorts null_x;
    null_x.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    null_x.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    null_x.x               = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_x), kConstraints).isSuccess());
    ProblemPorts null_h_pre;
    null_h_pre.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    null_h_pre.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    null_h_pre.h_pre           = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_h_pre), kConstraints).isSuccess());
    ProblemPorts null_aggregated;
    null_aggregated.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    null_aggregated.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    null_aggregated.aggregated      = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_aggregated), kConstraints).isSuccess());
}

TEST(MixedFp32HAggregateFwdTest, Fp16ClonesIntoIndependentConfigurableCopies) {
    MixedFp32HAggregateFwd prototype;
    ProblemPorts ports;
    ports.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    ports.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "MixedFp32HAggregateFwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<HAggregateFwdProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        other.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
        other.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
        static_cast<Primitive<HAggregateFwdProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(MixedFp32HAggregateFwdTest, Bf16fp32RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original             = makeProblem(ports).getProblemKey();
        const std::int64_t x_dims[]     = {2, 1, 33, 16};
        ports.x_dims                    = x_dims;
        const std::int64_t h_pre_dims[] = {2, 1, 33};
        ports.h_pre_dims                = h_pre_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        MixedFp32HAggregateFwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(MixedFp32HAggregateFwdTest, Bf16fp32ChecksChannelTileArithmeticBoundary) {
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

        MixedFp32HAggregateFwd primitive;
        EXPECT_EQ(primitive.isApplicable(makeProblem(ports), Constraints{0}).getStatus(),
                  channels == limit ? FTRAIN_STATUS_UNSUPPORTED : FTRAIN_STATUS_SUCCESS);
    }
}

TEST(MixedFp32HAggregateFwdTest, Bf16fp32ChecksEmptyBatchContract) {
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
    MixedFp32HAggregateFwd primitive;
    ASSERT_TRUE(primitive.isApplicable(empty, Constraints{0}).isSuccess());
    primitive.configure(empty);
    EXPECT_NO_THROW(primitive.execute(Resources{nullptr, 0, nullptr}));
}

TEST(MixedFp32HAggregateFwdTest, Fp16fp32RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original             = makeProblem(ports).getProblemKey();
        const std::int64_t x_dims[]     = {2, 1, 33, 16};
        ports.x_dims                    = x_dims;
        const std::int64_t h_pre_dims[] = {2, 1, 33};
        ports.h_pre_dims                = h_pre_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        MixedFp32HAggregateFwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(MixedFp32HAggregateFwdTest, Fp16fp32ChecksChannelTileArithmeticBoundary) {
    constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
    for (const std::int64_t channels : {limit - 255, limit}) {
        ProblemPorts ports;
        ports.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
        ports.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;

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

        MixedFp32HAggregateFwd primitive;
        EXPECT_EQ(primitive.isApplicable(makeProblem(ports), Constraints{0}).getStatus(),
                  channels == limit ? FTRAIN_STATUS_UNSUPPORTED : FTRAIN_STATUS_SUCCESS);
    }
}

TEST(MixedFp32HAggregateFwdTest, Fp16fp32ChecksEmptyBatchContract) {
    ProblemPorts ports;
    ports.x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    ports.aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;

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
    MixedFp32HAggregateFwd primitive;
    ASSERT_TRUE(primitive.isApplicable(empty, Constraints{0}).isSuccess());
    primitive.configure(empty);
    EXPECT_NO_THROW(primitive.execute(Resources{nullptr, 0, nullptr}));
}

}  // namespace ftrain
