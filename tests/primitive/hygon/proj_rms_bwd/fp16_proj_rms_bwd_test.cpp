// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/proj_rms_bwd/problem.hpp"
#include "flash_train/primitive/hygon/proj_rms_bwd/fp16_proj_rms_bwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kGradProjDims[2] = {16, 8};
constexpr std::int64_t kGradRDims[2] = {16, 1};
constexpr std::int64_t kXDims[2] = {16, 32};
constexpr std::int64_t kWeightDims[2] = {8, 32};
constexpr std::int64_t kRDims[2] = {16, 1};
constexpr std::int64_t kGradXDims[2] = {16, 32};
constexpr std::int64_t kGradWeightDims[2] = {8, 32};

struct ProblemPorts {
    void* grad_proj                      = kDefaultGradProjMemory;
    const std::int64_t* grad_proj_dims   = kGradProjDims;
    FTrainNumericType grad_proj_type     = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType grad_proj_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_r                         = kDefaultGradRMemory;
    const std::int64_t* grad_r_dims      = kGradRDims;
    FTrainNumericType grad_r_type        = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType grad_r_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* x                              = kDefaultXMemory;
    const std::int64_t* x_dims           = kXDims;
    FTrainNumericType x_type             = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType x_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* weight                         = kDefaultWeightMemory;
    const std::int64_t* weight_dims      = kWeightDims;
    FTrainNumericType weight_type        = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType weight_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* r                              = kDefaultRMemory;
    const std::int64_t* r_dims           = kRDims;
    FTrainNumericType r_type             = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType r_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_x                         = kDefaultGradXMemory;
    const std::int64_t* grad_x_dims      = kGradXDims;
    FTrainNumericType grad_x_type        = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType grad_x_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_weight                    = kDefaultGradWeightMemory;
    const std::int64_t* grad_weight_dims = kGradWeightDims;
    FTrainNumericType grad_weight_type   = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType grad_weight_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* eps                            = &kDefaultEps;
    FTrainNumericType eps_type           = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType eps_index            = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type       = FTRAIN_NUMERIC_TYPE_FP16;
    alignas(16) static unsigned char kDefaultGradProjMemory[16];
    alignas(16) static unsigned char kDefaultGradRMemory[16];
    alignas(16) static unsigned char kDefaultXMemory[16];
    alignas(16) static unsigned char kDefaultWeightMemory[16];
    alignas(16) static unsigned char kDefaultRMemory[16];
    alignas(16) static unsigned char kDefaultGradXMemory[16];
    alignas(16) static unsigned char kDefaultGradWeightMemory[16];
    static float kDefaultEps;
};

alignas(16) unsigned char ProblemPorts::kDefaultGradProjMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradRMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultXMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultWeightMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultRMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradXMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradWeightMemory[16];
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

ProjRmsBwdProblem makeProblem(const ProblemPorts& ports) {
    return ProjRmsBwdProblem{
        makeDeviceTensor(ports.grad_proj, ports.grad_proj_dims, 2, ports.grad_proj_type, ports.grad_proj_index),
        makeDeviceTensor(ports.grad_r, ports.grad_r_dims, 2, ports.grad_r_type, ports.grad_r_index),
        makeDeviceTensor(ports.x, ports.x_dims, 2, ports.x_type, ports.x_index),
        makeDeviceTensor(ports.weight, ports.weight_dims, 2, ports.weight_type, ports.weight_index),
        makeDeviceTensor(ports.r, ports.r_dims, 2, ports.r_type, ports.r_index),
        makeDeviceTensor(ports.grad_x, ports.grad_x_dims, 2, ports.grad_x_type, ports.grad_x_index),
        makeDeviceTensor(ports.grad_weight, ports.grad_weight_dims, 2, ports.grad_weight_type, ports.grad_weight_index),
        makeHostScalar(ports.eps, ports.eps_type, ports.eps_index),
        ProjRmsBwdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Fp16ProjRmsBwdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_grad_proj[16];
    rebound.grad_proj = other_grad_proj;
    alignas(16) static unsigned char other_grad_r[16];
    rebound.grad_r = other_grad_r;
    alignas(16) static unsigned char other_x[16];
    rebound.x = other_x;
    alignas(16) static unsigned char other_weight[16];
    rebound.weight = other_weight;
    alignas(16) static unsigned char other_r[16];
    rebound.r = other_r;
    alignas(16) static unsigned char other_grad_x[16];
    rebound.grad_x = other_grad_x;
    alignas(16) static unsigned char other_grad_weight[16];
    rebound.grad_weight = other_grad_weight;
    float other_eps     = 1e-06;
    rebound.eps         = &other_eps;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp16ProjRmsBwdTest, AcceptsContinuousAlignedOperands) {
    Fp16ProjRmsBwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp16ProjRmsBwd");
}

TEST(Fp16ProjRmsBwdTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp16ProjRmsBwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_proj;
    unsupported_grad_proj.grad_proj_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_proj), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_r;
    unsupported_grad_r.grad_r_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_r), kConstraints).isSuccess());
    ProblemPorts unsupported_x;
    unsupported_x.x_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_x), kConstraints).isSuccess());
    ProblemPorts unsupported_weight;
    unsupported_weight.weight_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_weight), kConstraints).isSuccess());
    ProblemPorts unsupported_r;
    unsupported_r.r_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_r), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_x;
    unsupported_grad_x.grad_x_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_x), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_weight;
    unsupported_grad_weight.grad_weight_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_weight), kConstraints).isSuccess());
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

TEST(Fp16ProjRmsBwdTest, RejectsNonContinuousIndexTypes) {
    Fp16ProjRmsBwd primitive;
    ProblemPorts strided_grad_proj;
    strided_grad_proj.grad_proj_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_proj), kConstraints).isSuccess());
    ProblemPorts strided_grad_r;
    strided_grad_r.grad_r_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_r), kConstraints).isSuccess());
    ProblemPorts strided_x;
    strided_x.x_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_x), kConstraints).isSuccess());
    ProblemPorts strided_weight;
    strided_weight.weight_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_weight), kConstraints).isSuccess());
    ProblemPorts strided_r;
    strided_r.r_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_r), kConstraints).isSuccess());
    ProblemPorts strided_grad_x;
    strided_grad_x.grad_x_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_x), kConstraints).isSuccess());
    ProblemPorts strided_grad_weight;
    strided_grad_weight.grad_weight_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_weight), kConstraints).isSuccess());
    ProblemPorts strided_eps;
    strided_eps.eps_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp16ProjRmsBwdTest, RejectsMisalignedAddresses) {
    Fp16ProjRmsBwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_grad_proj;
    misaligned_grad_proj.grad_proj = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_proj), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_r;
    misaligned_grad_r.grad_r = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_r), kConstraints).isSuccess());
    ProblemPorts misaligned_x;
    misaligned_x.x = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_x), kConstraints).isSuccess());
    ProblemPorts misaligned_weight;
    misaligned_weight.weight = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_weight), kConstraints).isSuccess());
    ProblemPorts misaligned_r;
    misaligned_r.r = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_r), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_x;
    misaligned_grad_x.grad_x = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_x), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_weight;
    misaligned_grad_weight.grad_weight = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_weight), kConstraints).isSuccess());
    ProblemPorts misaligned_eps;
    misaligned_eps.eps = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp16ProjRmsBwdTest, ChecksRequiredAndOptionalAddresses) {
    Fp16ProjRmsBwd primitive;
    ProblemPorts null_grad_proj;
    null_grad_proj.grad_proj = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_proj), kConstraints).isSuccess());
    ProblemPorts null_grad_r;
    null_grad_r.grad_r = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_r), kConstraints).isSuccess());
    ProblemPorts null_x;
    null_x.x = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_x), kConstraints).isSuccess());
    ProblemPorts null_weight;
    null_weight.weight = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_weight), kConstraints).isSuccess());
    ProblemPorts null_r;
    null_r.r = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_r), kConstraints).isSuccess());
    ProblemPorts null_grad_x;
    null_grad_x.grad_x = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_x), kConstraints).isSuccess());
    ProblemPorts null_grad_weight;
    null_grad_weight.grad_weight = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_weight), kConstraints).isSuccess());
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

TEST(Fp16ProjRmsBwdTest, ClonesIntoDistinctConfigurableCopies) {
    Fp16ProjRmsBwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp16ProjRmsBwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<ProjRmsBwdProblem>&>(*first).configure(makeProblem(ports));
    alignas(16) unsigned char other_memory[16]{};
    {
        ProblemPorts other;
        other.grad_proj = other_memory;
        static_cast<Primitive<ProjRmsBwdProblem>&>(*second).configure(makeProblem(other));
    }
    // Workspace checks alone do not prove pointer independence.
    // RebindsAddressesAndOutlivesOpsAndArgs covers execution with separate plans.
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp16ProjRmsBwdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original                   = makeProblem(ports).getProblemKey();
        const std::int64_t x_dims[]           = {16, 33};
        ports.x_dims                          = x_dims;
        const std::int64_t weight_dims[]      = {8, 33};
        ports.weight_dims                     = weight_dims;
        const std::int64_t grad_x_dims[]      = {16, 33};
        ports.grad_x_dims                     = grad_x_dims;
        const std::int64_t grad_weight_dims[] = {8, 33};
        ports.grad_weight_dims                = grad_weight_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp16ProjRmsBwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

}  // namespace ftrain
