// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/proj_rms_fwd/problem.hpp"
#include "flash_train/primitive/hygon/proj_rms_fwd/bf16_proj_rms_fwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kXDims[2] = {16, 32};
constexpr std::int64_t kWeightDims[2] = {8, 32};
constexpr std::int64_t kProjDims[2] = {16, 8};
constexpr std::int64_t kRDims[2] = {16, 1};

struct ProblemPorts {
    void* x                         = kDefaultXMemory;
    const std::int64_t* x_dims      = kXDims;
    FTrainNumericType x_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType x_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* weight                    = kDefaultWeightMemory;
    const std::int64_t* weight_dims = kWeightDims;
    FTrainNumericType weight_type   = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType weight_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* proj                      = kDefaultProjMemory;
    const std::int64_t* proj_dims   = kProjDims;
    FTrainNumericType proj_type     = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType proj_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* r                         = kDefaultRMemory;
    const std::int64_t* r_dims      = kRDims;
    FTrainNumericType r_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType r_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* eps                       = &kDefaultEps;
    FTrainNumericType eps_type      = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType eps_index       = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type  = FTRAIN_NUMERIC_TYPE_BF16;
    alignas(16) static unsigned char kDefaultXMemory[16];
    alignas(16) static unsigned char kDefaultWeightMemory[16];
    alignas(16) static unsigned char kDefaultProjMemory[16];
    alignas(16) static unsigned char kDefaultRMemory[16];
    static float kDefaultEps;
};

alignas(16) unsigned char ProblemPorts::kDefaultXMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultWeightMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultProjMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultRMemory[16];
float ProblemPorts::kDefaultEps = 1e-06;

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

Tensor makeHostScalar(void* memory, FTrainNumericType numeric_type, FTrainIndexType index_type) {
    FTrainStorageView view{};
    view.memory         = memory;
    view.dims           = nullptr;
    view.strides        = nullptr;
    view.num_dims       = 0;
    view.numeric_type   = numeric_type;
    view.index_type     = index_type;
    view.is_host_memory = true;
    return Tensor{StorageView{view}};
}

ProjRmsFwdProblem makeProblem(const ProblemPorts& ports) {
    return ProjRmsFwdProblem{
        makeDeviceTensor(ports.x, ports.x_dims, 2, ports.x_type, ports.x_index),
        makeDeviceTensor(ports.weight, ports.weight_dims, 2, ports.weight_type, ports.weight_index),
        makeDeviceTensor(ports.proj, ports.proj_dims, 2, ports.proj_type, ports.proj_index),
        makeDeviceTensor(ports.r, ports.r_dims, 2, ports.r_type, ports.r_index),
        makeHostScalar(ports.eps, ports.eps_type, ports.eps_index),
        ProjRmsFwdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

Tensor matrix(void* address, std::int64_t rows, std::int64_t columns, FTrainNumericType dtype) {
    const std::int64_t dims[] = {rows, columns};
    return Tensor{
        StorageView{FTrainStorageView{address, dims, nullptr, 2, dtype, FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
}

constexpr float kEps = 1e-6F;

Tensor hostScalar(const float& value) {
    return Tensor{StorageView{FTrainStorageView{const_cast<float*>(&value), nullptr, nullptr, 0,
                                                FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true}}};
}

ProjRmsFwdProblem problem(void* address, FTrainNumericType x, FTrainNumericType w) {
    return {matrix(address, 64, 32, x), matrix(address, 8, 32, w), matrix(address, 64, 8, x),
            matrix(address, 64, 1, x),  hostScalar(kEps),          ProjRmsFwdAttributes{x}};
}

ProjRmsFwdProblem validatedProblem(void* address, FTrainNumericType x, FTrainNumericType w) {
    return {matrix(address, 64, 32, x), matrix(address, 8, 32, w),
            matrix(address, 64, 8, x),  matrix(address, 64, 1, x),
            hostScalar(kEps),           ProjRmsFwdAttributes{x == w ? x : FTRAIN_NUMERIC_TYPE_FP32}};
}

}  // namespace

TEST(Bf16ProjRmsFwdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_x[16];
    rebound.x = other_x;
    alignas(16) static unsigned char other_weight[16];
    rebound.weight = other_weight;
    alignas(16) static unsigned char other_proj[16];
    rebound.proj = other_proj;
    alignas(16) static unsigned char other_r[16];
    rebound.r       = other_r;
    float other_eps = 1e-06;
    rebound.eps     = &other_eps;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Bf16ProjRmsFwdTest, AcceptsContinuousAlignedOperands) {
    Bf16ProjRmsFwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Bf16ProjRmsFwd");
}

TEST(Bf16ProjRmsFwdTest, RejectsUnsupportedComputeAndElementTypes) {
    Bf16ProjRmsFwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_x;
    unsupported_x.x_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_x), kConstraints).isSuccess());
    ProblemPorts unsupported_weight;
    unsupported_weight.weight_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_weight), kConstraints).isSuccess());
    ProblemPorts unsupported_proj;
    unsupported_proj.proj_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_proj), kConstraints).isSuccess());
    ProblemPorts unsupported_r;
    unsupported_r.r_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_r), kConstraints).isSuccess());
    ProblemPorts unsupported_eps;
    unsupported_eps.eps_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
}

TEST(Bf16ProjRmsFwdTest, RejectsNonContinuousIndexTypes) {
    Bf16ProjRmsFwd primitive;
    ProblemPorts strided_x;
    strided_x.x_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_x), kConstraints).isSuccess());
    ProblemPorts strided_weight;
    strided_weight.weight_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_weight), kConstraints).isSuccess());
    ProblemPorts strided_proj;
    strided_proj.proj_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_proj), kConstraints).isSuccess());
    ProblemPorts strided_r;
    strided_r.r_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_r), kConstraints).isSuccess());
    ProblemPorts strided_eps;
    strided_eps.eps_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16ProjRmsFwdTest, RejectsMisalignedAddresses) {
    Bf16ProjRmsFwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_x;
    misaligned_x.x = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_x), kConstraints).isSuccess());
    ProblemPorts misaligned_weight;
    misaligned_weight.weight = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_weight), kConstraints).isSuccess());
    ProblemPorts misaligned_proj;
    misaligned_proj.proj = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_proj), kConstraints).isSuccess());
    ProblemPorts misaligned_r;
    misaligned_r.r = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_r), kConstraints).isSuccess());
    ProblemPorts misaligned_eps;
    misaligned_eps.eps = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16ProjRmsFwdTest, ChecksRequiredAndOptionalAddresses) {
    Bf16ProjRmsFwd primitive;
    ProblemPorts null_x;
    null_x.x = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_x), kConstraints).isSuccess());
    ProblemPorts null_weight;
    null_weight.weight = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_weight), kConstraints).isSuccess());
    ProblemPorts null_proj;
    null_proj.proj = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_proj), kConstraints).isSuccess());
    ProblemPorts null_r;
    null_r.r = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_r), kConstraints).isSuccess());
    ProblemPorts null_eps;
    null_eps.eps = nullptr;
    try {
        makeProblem(null_eps);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16ProjRmsFwdTest, ClonesIntoDistinctConfigurableCopies) {
    Bf16ProjRmsFwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Bf16ProjRmsFwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<ProjRmsFwdProblem>&>(*first).configure(makeProblem(ports));
    alignas(16) unsigned char other_memory[16]{};
    {
        const std::int64_t other_x_dims[]{16, 4096};
        const std::int64_t other_weight_dims[]{8, 4096};
        ProblemPorts other;
        other.x_dims      = other_x_dims;
        other.weight_dims = other_weight_dims;
        other.x = other_memory;
        static_cast<Primitive<ProjRmsFwdProblem>&>(*second).configure(makeProblem(other));
    }
    // Workspace checks alone do not prove pointer independence.
    // RebindsAddressesAndOutlivesOpsAndArgs covers execution with separate plans.
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_GT(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Bf16ProjRmsFwdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original              = makeProblem(ports).getProblemKey();
        const std::int64_t x_dims[]      = {16, 33};
        ports.x_dims                     = x_dims;
        const std::int64_t weight_dims[] = {8, 33};
        ports.weight_dims                = weight_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Bf16ProjRmsFwd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16ProjRmsFwdRegressionTest, RejectsUnsupportedWeightType) {
    alignas(16) unsigned char address[16]{};
    const auto unsupported = problem(address, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_INT32);
    {
        Bf16ProjRmsFwd primitive;
        EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    }
}

TEST(Bf16ProjRmsFwdRegressionTest, ChecksPaddedReductionBoundaryForThisStoragePair) {
    alignas(16) unsigned char address[16]{};
    {
        const auto x = FTRAIN_NUMERIC_TYPE_BF16;
        {
            const auto w                = FTRAIN_NUMERIC_TYPE_BF16;
            const std::int64_t chunk    = x == w && x != FTRAIN_NUMERIC_TYPE_FP32 ? 128 : 64;
            const std::int64_t limit    = std::numeric_limits<std::int32_t>::max() - (chunk - 1);
            const std::int64_t fragment = x == FTRAIN_NUMERIC_TYPE_FP32 ? 8 : 16;
            for (const std::int64_t reduction : {limit, limit + fragment}) {
                const ProjRmsFwdProblem p{matrix(address, 1, reduction, x),
                                          matrix(address, 1, reduction, w),
                                          matrix(address, 1, 1, x),
                                          matrix(address, 1, 1, x),
                                          hostScalar(kEps),
                                          ProjRmsFwdAttributes{x}};
                std::size_t supported = 0;
                {
                    Bf16ProjRmsFwd primitive;
                    supported += primitive.isApplicable(p, Constraints{0}).isSuccess();
                }
                EXPECT_EQ(supported, reduction == limit ? 1U : 0U);
            }
        }
    }
}

TEST(Bf16ProjRmsFwdTest, CacheKeyIncludesWeightTypeButNotAllocationAddress) {
    alignas(16) unsigned char first[16]{}, second[16]{};
    const auto bf16    = validatedProblem(first, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_BF16);
    const auto mixed   = validatedProblem(first, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP32);
    const auto rebound = validatedProblem(second, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP32);
    EXPECT_NE(bf16.getProblemKey(), mixed.getProblemKey());
    EXPECT_EQ(mixed.getProblemKey(), rebound.getProblemKey());
}

TEST(Bf16ProjRmsFwdTest, RejectsInconsistentReductionDimensions) {
    alignas(16) unsigned char address[16]{};
    EXPECT_THROW(
        (ProjRmsFwdProblem{
            matrix(address, 64, 32, FTRAIN_NUMERIC_TYPE_BF16), matrix(address, 8, 16, FTRAIN_NUMERIC_TYPE_FP32),
            matrix(address, 64, 8, FTRAIN_NUMERIC_TYPE_BF16), matrix(address, 64, 1, FTRAIN_NUMERIC_TYPE_BF16),
            hostScalar(kEps), ProjRmsFwdAttributes{FTRAIN_NUMERIC_TYPE_BF16}}),
        Exception);
}

TEST(Bf16ProjRmsFwdTest, ValidEpsilonValuesShareSelectionButInvalidEpsilonIsRejected) {
    alignas(16) unsigned char address[16]{};
    float eps       = 1e-6F;
    const auto make = [&] {
        return ProjRmsFwdProblem{matrix(address, 64, 32, FTRAIN_NUMERIC_TYPE_BF16),
                                 matrix(address, 8, 32, FTRAIN_NUMERIC_TYPE_BF16),
                                 matrix(address, 64, 8, FTRAIN_NUMERIC_TYPE_BF16),
                                 matrix(address, 64, 1, FTRAIN_NUMERIC_TYPE_BF16),
                                 hostScalar(eps),
                                 ProjRmsFwdAttributes{FTRAIN_NUMERIC_TYPE_BF16}};
    };
    const auto key = make().getProblemKey();
    eps            = 1e-3F;
    EXPECT_EQ(key, make().getProblemKey());
    eps = 0.0F;
    EXPECT_THROW(make(), Exception);
}

}  // namespace ftrain
