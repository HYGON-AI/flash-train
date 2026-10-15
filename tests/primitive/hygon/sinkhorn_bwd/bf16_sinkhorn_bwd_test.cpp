// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/sinkhorn_bwd/problem.hpp"
#include "flash_train/primitive/hygon/sinkhorn_bwd/bf16_sinkhorn_bwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kGradOutputDims[3] = {2, 2, 2};
constexpr std::int64_t kMInitDims[3] = {2, 2, 2};
constexpr std::int64_t kGradLogitsDims[3] = {2, 2, 2};

struct ProblemPorts {
    void* grad_output                     = kDefaultGradOutputMemory;
    const std::int64_t* grad_output_dims  = kGradOutputDims;
    FTrainNumericType grad_output_type    = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_output_index     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* m_init                          = kDefaultMInitMemory;
    const std::int64_t* m_init_dims       = kMInitDims;
    FTrainNumericType m_init_type         = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType m_init_index          = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_logits                     = kDefaultGradLogitsMemory;
    const std::int64_t* grad_logits_dims  = kGradLogitsDims;
    FTrainNumericType grad_logits_type    = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_logits_index     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* num_iterations                  = &kDefaultNumIterations;
    FTrainNumericType num_iterations_type = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType num_iterations_index  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* eps                             = &kDefaultEps;
    FTrainNumericType eps_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType eps_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type        = FTRAIN_NUMERIC_TYPE_BF16;
    alignas(16) static unsigned char kDefaultGradOutputMemory[16];
    alignas(16) static unsigned char kDefaultMInitMemory[16];
    alignas(16) static unsigned char kDefaultGradLogitsMemory[16];
    static std::int64_t kDefaultNumIterations;
    static float kDefaultEps;
};

alignas(16) unsigned char ProblemPorts::kDefaultGradOutputMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultMInitMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradLogitsMemory[16];

std::int64_t ProblemPorts::kDefaultNumIterations = 2;

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

SinkhornBwdProblem makeProblem(const ProblemPorts& ports) {
    return SinkhornBwdProblem{
        makeDeviceTensor(ports.grad_output, ports.grad_output_dims, 3, ports.grad_output_type, ports.grad_output_index),
        makeDeviceTensor(ports.m_init, ports.m_init_dims, 3, ports.m_init_type, ports.m_init_index),
        makeDeviceTensor(ports.grad_logits, ports.grad_logits_dims, 3, ports.grad_logits_type, ports.grad_logits_index),
        makeHostScalar(ports.num_iterations, ports.num_iterations_type, ports.num_iterations_index),
        makeHostScalar(ports.eps, ports.eps_type, ports.eps_index),
        SinkhornBwdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

Tensor tensor(void* memory, std::initializer_list<std::int64_t> shape) {
    return Tensor{StorageView{FTrainStorageView{memory, shape.begin(), nullptr, static_cast<std::uint8_t>(shape.size()),
                                                FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
}

SinkhornBwdProblem makeProblem(std::int64_t size, const std::int64_t& iterations) {
    alignas(16) static unsigned char memory[16]{};
    static float eps  = 1e-6F;
    const auto values = tensor(memory, {1, 1, 2, size, size});
    const auto state  = tensor(memory, {2, size, size});
    const auto count =
        Tensor{StorageView{FTrainStorageView{const_cast<std::int64_t*>(&iterations), nullptr, nullptr, 0,
                                             FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true}}};
    const auto epsilon = Tensor{StorageView{
        FTrainStorageView{&eps, nullptr, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true}}};
    return {values, state, values, count, epsilon, SinkhornBwdAttributes{FTRAIN_NUMERIC_TYPE_FP32}};
}

}  // namespace

TEST(Bf16SinkhornBwdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_grad_output[16];
    rebound.grad_output = other_grad_output;
    alignas(16) static unsigned char other_m_init[16];
    rebound.m_init = other_m_init;
    alignas(16) static unsigned char other_grad_logits[16];
    rebound.grad_logits               = other_grad_logits;
    std::int64_t other_num_iterations = 2;
    rebound.num_iterations            = &other_num_iterations;
    float other_eps                   = 1e-06;
    rebound.eps                       = &other_eps;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Bf16SinkhornBwdTest, AcceptsContinuousAlignedOperands) {
    Bf16SinkhornBwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Bf16SinkhornBwd");
}

TEST(Bf16SinkhornBwdTest, RejectsUnsupportedComputeAndElementTypes) {
    Bf16SinkhornBwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_output;
    unsupported_grad_output.grad_output_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_output), kConstraints).isSuccess());
    ProblemPorts unsupported_m_init;
    unsupported_m_init.m_init_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_m_init), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_logits;
    unsupported_grad_logits.grad_logits_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_logits), kConstraints).isSuccess());
    ProblemPorts unsupported_num_iterations;
    unsupported_num_iterations.num_iterations_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_num_iterations), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
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

TEST(Bf16SinkhornBwdTest, RejectsNonContinuousIndexTypes) {
    Bf16SinkhornBwd primitive;
    ProblemPorts strided_grad_output;
    strided_grad_output.grad_output_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_output), kConstraints).isSuccess());
    ProblemPorts strided_m_init;
    strided_m_init.m_init_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_m_init), kConstraints).isSuccess());
    ProblemPorts strided_grad_logits;
    strided_grad_logits.grad_logits_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_logits), kConstraints).isSuccess());
    ProblemPorts strided_num_iterations;
    strided_num_iterations.num_iterations_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_num_iterations), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_eps;
    strided_eps.eps_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16SinkhornBwdTest, RejectsMisalignedAddresses) {
    Bf16SinkhornBwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_grad_output;
    misaligned_grad_output.grad_output = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_output), kConstraints).isSuccess());
    ProblemPorts misaligned_m_init;
    misaligned_m_init.m_init = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_m_init), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_logits;
    misaligned_grad_logits.grad_logits = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_logits), kConstraints).isSuccess());
    ProblemPorts misaligned_num_iterations;
    misaligned_num_iterations.num_iterations = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_num_iterations), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_eps;
    misaligned_eps.eps = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16SinkhornBwdTest, ChecksRequiredAndOptionalAddresses) {
    Bf16SinkhornBwd primitive;
    ProblemPorts null_grad_output;
    null_grad_output.grad_output = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_output), kConstraints).isSuccess());
    ProblemPorts null_m_init;
    null_m_init.m_init = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_m_init), kConstraints).isSuccess());
    ProblemPorts null_grad_logits;
    null_grad_logits.grad_logits = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_logits), kConstraints).isSuccess());
    ProblemPorts null_num_iterations;
    null_num_iterations.num_iterations = nullptr;
    try {
        makeProblem(null_num_iterations);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
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

TEST(Bf16SinkhornBwdTest, ClonesIntoIndependentConfigurableCopies) {
    Bf16SinkhornBwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Bf16SinkhornBwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<SinkhornBwdProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<SinkhornBwdProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Bf16SinkhornBwdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original                   = makeProblem(ports).getProblemKey();
        const std::int64_t grad_output_dims[] = {2, 33, 33};
        ports.grad_output_dims                = grad_output_dims;
        const std::int64_t m_init_dims[]      = {2, 33, 33};
        ports.m_init_dims                     = m_init_dims;
        const std::int64_t grad_logits_dims[] = {2, 33, 33};
        ports.grad_logits_dims                = grad_logits_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Bf16SinkhornBwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16SinkhornBwdTest, ChecksIterationSharedMemoryBoundary) {
    for (const std::int64_t iterations : {408LL, 409LL, 2147483647LL}) {
        ProblemPorts ports;

        const std::int64_t grad_output_dims[] = {2, 4, 4};
        ports.grad_output_dims                = grad_output_dims;
        const std::int64_t m_init_dims[]      = {2, 4, 4};
        ports.m_init_dims                     = m_init_dims;
        const std::int64_t grad_logits_dims[] = {2, 4, 4};
        ports.grad_logits_dims                = grad_logits_dims;

        std::int64_t c_num_iterations_value = iterations;
        ports.num_iterations                = &c_num_iterations_value;

        Bf16SinkhornBwd primitive;
        EXPECT_EQ(primitive.isApplicable(makeProblem(ports), Constraints{0}).getStatus(),
                  iterations == 408 ? FTRAIN_STATUS_SUCCESS : FTRAIN_STATUS_UNSUPPORTED);
    }
}

TEST(SinkhornBwdProblemTest, AcceptsArbitraryBatchRankAndLeavesMatrixSizeSupportToPrimitive) {
    const std::int64_t iterations = 3;
    const auto problem            = makeProblem(4, iterations);
    EXPECT_EQ(problem.getBatch(), 2U);
    EXPECT_EQ(problem.getMatrixSize(), 4U);
    EXPECT_NO_THROW(makeProblem(33, iterations));
}

}  // namespace ftrain
