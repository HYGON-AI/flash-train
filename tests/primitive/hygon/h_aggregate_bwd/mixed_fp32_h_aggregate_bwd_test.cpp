// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/h_aggregate_bwd/problem.hpp"
#include "flash_train/primitive/hygon/h_aggregate_bwd/mixed_fp32_h_aggregate_bwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kGradAggregatedDims[3] = {2, 1, 16};
constexpr std::int64_t kXDims[4] = {2, 1, 2, 16};
constexpr std::int64_t kHPreDims[3] = {2, 1, 2};
constexpr std::int64_t kGradXDims[4] = {2, 1, 2, 16};
constexpr std::int64_t kGradHPreDims[3] = {2, 1, 2};

struct ProblemPorts {
    void* grad_aggregated                    = kDefaultGradAggregatedMemory;
    const std::int64_t* grad_aggregated_dims = kGradAggregatedDims;
    FTrainNumericType grad_aggregated_type   = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_aggregated_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* x                                  = kDefaultXMemory;
    const std::int64_t* x_dims               = kXDims;
    FTrainNumericType x_type                 = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType x_index                  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* h_pre                              = kDefaultHPreMemory;
    const std::int64_t* h_pre_dims           = kHPreDims;
    FTrainNumericType h_pre_type             = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType h_pre_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_x                             = kDefaultGradXMemory;
    const std::int64_t* grad_x_dims          = kGradXDims;
    FTrainNumericType grad_x_type            = FTRAIN_NUMERIC_TYPE_BF16;
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

}  // namespace

TEST(MixedFp32HAggregateBwdTest, KeyIgnoresAllocationAddresses) {
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

TEST(MixedFp32HAggregateBwdTest, Bf16AcceptsContinuousAlignedOperands) {
    MixedFp32HAggregateBwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "MixedFp32HAggregateBwd");
}

TEST(MixedFp32HAggregateBwdTest, Bf16RejectsUnsupportedComputeAndElementTypes) {
    MixedFp32HAggregateBwd primitive;
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

TEST(MixedFp32HAggregateBwdTest, Bf16RejectsNonContinuousIndexTypes) {
    MixedFp32HAggregateBwd primitive;
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

TEST(MixedFp32HAggregateBwdTest, Bf16RejectsMisalignedAddresses) {
    MixedFp32HAggregateBwd primitive;
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

TEST(MixedFp32HAggregateBwdTest, Bf16ChecksRequiredAndOptionalAddresses) {
    MixedFp32HAggregateBwd primitive;
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

TEST(MixedFp32HAggregateBwdTest, Bf16ClonesIntoDistinctConfigurableCopies) {
    MixedFp32HAggregateBwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "MixedFp32HAggregateBwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<HAggregateBwdProblem>&>(*first).configure(makeProblem(ports));
    alignas(16) unsigned char other_memory[16]{};
    {
        ProblemPorts other;
        other.grad_aggregated = other_memory;
        static_cast<Primitive<HAggregateBwdProblem>&>(*second).configure(makeProblem(other));
    }
    // Workspace checks alone do not prove pointer independence.
    // RebindsAddressesAndOutlivesOpsAndArgs covers execution with separate plans.
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(MixedFp32HAggregateBwdTest, Fp16AcceptsContinuousAlignedOperands) {
    MixedFp32HAggregateBwd primitive;
    ProblemPorts ports;
    ports.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    ports.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    ports.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "MixedFp32HAggregateBwd");
}

TEST(MixedFp32HAggregateBwdTest, Fp16RejectsUnsupportedComputeAndElementTypes) {
    MixedFp32HAggregateBwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_compute.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_compute.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_compute.compute_type         = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_aggregated;
    unsupported_grad_aggregated.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_grad_aggregated.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_grad_aggregated.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_grad_aggregated.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_aggregated), kConstraints).isSuccess());
    ProblemPorts unsupported_x;
    unsupported_x.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_x.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_x.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_x.x_type               = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_x), kConstraints).isSuccess());
    ProblemPorts unsupported_h_pre;
    unsupported_h_pre.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_h_pre.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_h_pre.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_h_pre.h_pre_type           = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_h_pre), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_x;
    unsupported_grad_x.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_grad_x.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_grad_x.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_grad_x.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_x), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_h_pre;
    unsupported_grad_h_pre.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_grad_h_pre.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_grad_h_pre.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    unsupported_grad_h_pre.grad_h_pre_type      = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_h_pre), kConstraints).isSuccess());
}

TEST(MixedFp32HAggregateBwdTest, Fp16RejectsNonContinuousIndexTypes) {
    MixedFp32HAggregateBwd primitive;
    ProblemPorts strided_grad_aggregated;
    strided_grad_aggregated.grad_aggregated_type  = FTRAIN_NUMERIC_TYPE_FP16;
    strided_grad_aggregated.x_type                = FTRAIN_NUMERIC_TYPE_FP16;
    strided_grad_aggregated.grad_x_type           = FTRAIN_NUMERIC_TYPE_FP16;
    strided_grad_aggregated.grad_aggregated_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_aggregated), kConstraints).isSuccess());
    ProblemPorts strided_x;
    strided_x.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    strided_x.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    strided_x.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    strided_x.x_index              = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_x), kConstraints).isSuccess());
    ProblemPorts strided_h_pre;
    strided_h_pre.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    strided_h_pre.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    strided_h_pre.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    strided_h_pre.h_pre_index          = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_h_pre), kConstraints).isSuccess());
    ProblemPorts strided_grad_x;
    strided_grad_x.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    strided_grad_x.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    strided_grad_x.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    strided_grad_x.grad_x_index         = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_x), kConstraints).isSuccess());
    ProblemPorts strided_grad_h_pre;
    strided_grad_h_pre.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    strided_grad_h_pre.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    strided_grad_h_pre.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    strided_grad_h_pre.grad_h_pre_index     = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_h_pre), kConstraints).isSuccess());
}

TEST(MixedFp32HAggregateBwdTest, Fp16RejectsMisalignedAddresses) {
    MixedFp32HAggregateBwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_grad_aggregated;
    misaligned_grad_aggregated.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_grad_aggregated.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_grad_aggregated.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_grad_aggregated.grad_aggregated      = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_aggregated), kConstraints).isSuccess());
    ProblemPorts misaligned_x;
    misaligned_x.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_x.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_x.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_x.x                    = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_x), kConstraints).isSuccess());
    ProblemPorts misaligned_h_pre;
    misaligned_h_pre.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_h_pre.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_h_pre.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_h_pre.h_pre                = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_h_pre), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_x;
    misaligned_grad_x.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_grad_x.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_grad_x.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_grad_x.grad_x               = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_x), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_h_pre;
    misaligned_grad_h_pre.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_grad_h_pre.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_grad_h_pre.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    misaligned_grad_h_pre.grad_h_pre           = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_h_pre), kConstraints).isSuccess());
}

TEST(MixedFp32HAggregateBwdTest, Fp16ChecksRequiredAndOptionalAddresses) {
    MixedFp32HAggregateBwd primitive;
    ProblemPorts null_grad_aggregated;
    null_grad_aggregated.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    null_grad_aggregated.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    null_grad_aggregated.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    null_grad_aggregated.grad_aggregated      = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_aggregated), kConstraints).isSuccess());
    ProblemPorts null_x;
    null_x.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    null_x.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    null_x.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    null_x.x                    = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_x), kConstraints).isSuccess());
    ProblemPorts null_h_pre;
    null_h_pre.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    null_h_pre.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    null_h_pre.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    null_h_pre.h_pre                = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_h_pre), kConstraints).isSuccess());
    ProblemPorts null_grad_x;
    null_grad_x.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    null_grad_x.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    null_grad_x.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    null_grad_x.grad_x               = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_x), kConstraints).isSuccess());
    ProblemPorts null_grad_h_pre;
    null_grad_h_pre.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    null_grad_h_pre.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    null_grad_h_pre.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    null_grad_h_pre.grad_h_pre           = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_h_pre), kConstraints).isSuccess());
}

TEST(MixedFp32HAggregateBwdTest, Fp16ClonesIntoDistinctConfigurableCopies) {
    MixedFp32HAggregateBwd prototype;
    ProblemPorts ports;
    ports.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    ports.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    ports.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "MixedFp32HAggregateBwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<HAggregateBwdProblem>&>(*first).configure(makeProblem(ports));
    alignas(16) unsigned char other_memory[16]{};
    {
        ProblemPorts other;
        other.grad_aggregated      = other_memory;
        other.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
        other.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
        other.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;
        static_cast<Primitive<HAggregateBwdProblem>&>(*second).configure(makeProblem(other));
    }
    // Workspace checks alone do not prove pointer independence.
    // RebindsAddressesAndOutlivesOpsAndArgs covers execution with separate plans.
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(MixedFp32HAggregateBwdTest, Bf16fp32RejectsUnsupportedGeometryWithoutLaunching) {
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
        MixedFp32HAggregateBwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(MixedFp32HAggregateBwdTest, Bf16fp32ChecksChannelTileArithmeticBoundary) {
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

        MixedFp32HAggregateBwd primitive;
        EXPECT_EQ(primitive.isApplicable(makeProblem(ports), Constraints{0}).getStatus(),
                  channels == limit ? FTRAIN_STATUS_UNSUPPORTED : FTRAIN_STATUS_SUCCESS);
    }
}

TEST(MixedFp32HAggregateBwdTest, Bf16fp32ChecksEmptyBatchContract) {
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
    MixedFp32HAggregateBwd primitive;
    ASSERT_TRUE(primitive.isApplicable(empty, Constraints{0}).isSuccess());
    primitive.configure(empty);
    EXPECT_NO_THROW(primitive.execute(Resources{nullptr, 0, nullptr}));
}

TEST(MixedFp32HAggregateBwdTest, Fp16fp32RejectsUnsupportedGeometryWithoutLaunching) {
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
        MixedFp32HAggregateBwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(MixedFp32HAggregateBwdTest, Fp16fp32ChecksChannelTileArithmeticBoundary) {
    constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
    for (const std::int64_t channels : {limit - 63, limit}) {
        ProblemPorts ports;
        ports.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
        ports.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
        ports.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;

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

        MixedFp32HAggregateBwd primitive;
        EXPECT_EQ(primitive.isApplicable(makeProblem(ports), Constraints{0}).getStatus(),
                  channels == limit ? FTRAIN_STATUS_UNSUPPORTED : FTRAIN_STATUS_SUCCESS);
    }
}

TEST(MixedFp32HAggregateBwdTest, Fp16fp32ChecksEmptyBatchContract) {
    ProblemPorts ports;
    ports.grad_aggregated_type = FTRAIN_NUMERIC_TYPE_FP16;
    ports.x_type               = FTRAIN_NUMERIC_TYPE_FP16;
    ports.grad_x_type          = FTRAIN_NUMERIC_TYPE_FP16;

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
    MixedFp32HAggregateBwd primitive;
    ASSERT_TRUE(primitive.isApplicable(empty, Constraints{0}).isSuccess());
    primitive.configure(empty);
    EXPECT_NO_THROW(primitive.execute(Resources{nullptr, 0, nullptr}));
}

}  // namespace ftrain
