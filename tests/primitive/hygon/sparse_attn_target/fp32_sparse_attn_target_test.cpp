// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/sparse_attn_target/problem.hpp"
#include "flash_train/primitive/hygon/sparse_attn_target/fp32_sparse_attn_target.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kQDims[4] = {1, 4, 2, 32};
constexpr std::int64_t kKDims[3] = {1, 4, 32};
constexpr std::int64_t kLseDims[3] = {1, 4, 2};
constexpr std::int64_t kTopkIndicesDims[3] = {1, 4, 4};
constexpr std::int64_t kTargetDims[3] = {1, 4, 4};

struct ProblemPorts {
    void* q                                    = kDefaultQMemory;
    const std::int64_t* q_dims                 = kQDims;
    FTrainNumericType q_type                   = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType q_index                    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* k                                    = kDefaultKMemory;
    const std::int64_t* k_dims                 = kKDims;
    FTrainNumericType k_type                   = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType k_index                    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* lse                                  = kDefaultLseMemory;
    const std::int64_t* lse_dims               = kLseDims;
    FTrainNumericType lse_type                 = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType lse_index                  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* topk_indices                         = kDefaultTopkIndicesMemory;
    const std::int64_t* topk_indices_dims      = kTopkIndicesDims;
    FTrainNumericType topk_indices_type        = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType topk_indices_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* target                               = kDefaultTargetMemory;
    const std::int64_t* target_dims            = kTargetDims;
    FTrainNumericType target_type              = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType target_index               = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* softmax_scale                        = &kDefaultSoftmaxScale;
    FTrainNumericType softmax_scale_type       = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType softmax_scale_index        = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* topk_indices_global                  = &kDefaultTopkIndicesGlobal;
    FTrainNumericType topk_indices_global_type = FTRAIN_NUMERIC_TYPE_UINT8;
    FTrainIndexType topk_indices_global_index  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type             = FTRAIN_NUMERIC_TYPE_FP32;
    alignas(16) static unsigned char kDefaultQMemory[16];
    alignas(16) static unsigned char kDefaultKMemory[16];
    alignas(16) static unsigned char kDefaultLseMemory[16];
    alignas(16) static unsigned char kDefaultTopkIndicesMemory[16];
    alignas(16) static unsigned char kDefaultTargetMemory[16];
    static float kDefaultSoftmaxScale;
    static std::uint8_t kDefaultTopkIndicesGlobal;
};

alignas(16) unsigned char ProblemPorts::kDefaultQMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultKMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultLseMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultTopkIndicesMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultTargetMemory[16];
float ProblemPorts::kDefaultSoftmaxScale = 0.25F;

std::uint8_t ProblemPorts::kDefaultTopkIndicesGlobal = 0;

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

SparseAttnTargetProblem makeProblem(const ProblemPorts& ports) {
    return SparseAttnTargetProblem{
        makeDeviceTensor(ports.q, ports.q_dims, 4, ports.q_type, ports.q_index),
        makeDeviceTensor(ports.k, ports.k_dims, 3, ports.k_type, ports.k_index),
        makeDeviceTensor(ports.lse, ports.lse_dims, 3, ports.lse_type, ports.lse_index),
        makeDeviceTensor(ports.topk_indices, ports.topk_indices_dims, 3, ports.topk_indices_type,
                         ports.topk_indices_index),
        makeDeviceTensor(ports.target, ports.target_dims, 3, ports.target_type, ports.target_index),
        makeHostScalar(ports.softmax_scale, ports.softmax_scale_type, ports.softmax_scale_index),
        makeHostScalar(ports.topk_indices_global, ports.topk_indices_global_type, ports.topk_indices_global_index),
        SparseAttnTargetAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Fp32SparseAttnTargetTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_q[16];
    rebound.q = other_q;
    alignas(16) static unsigned char other_k[16];
    rebound.k = other_k;
    alignas(16) static unsigned char other_lse[16];
    rebound.lse = other_lse;
    alignas(16) static unsigned char other_topk_indices[16];
    rebound.topk_indices = other_topk_indices;
    alignas(16) static unsigned char other_target[16];
    rebound.target                         = other_target;
    float other_softmax_scale              = 0.25F;
    rebound.softmax_scale                  = &other_softmax_scale;
    std::uint8_t other_topk_indices_global = 0;
    rebound.topk_indices_global            = &other_topk_indices_global;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp32SparseAttnTargetTest, AcceptsContinuousAlignedOperands) {
    Fp32SparseAttnTarget primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp32SparseAttnTarget");
}

TEST(Fp32SparseAttnTargetTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp32SparseAttnTarget primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_q;
    unsupported_q.q_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_q), kConstraints).isSuccess());
    ProblemPorts unsupported_k;
    unsupported_k.k_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_k), kConstraints).isSuccess());
    ProblemPorts unsupported_lse;
    unsupported_lse.lse_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_lse), kConstraints).isSuccess());
    ProblemPorts unsupported_topk_indices;
    unsupported_topk_indices.topk_indices_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_topk_indices), kConstraints).isSuccess());
    ProblemPorts unsupported_target;
    unsupported_target.target_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_target), kConstraints).isSuccess());
    ProblemPorts unsupported_softmax_scale;
    unsupported_softmax_scale.softmax_scale_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_softmax_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_topk_indices_global;
    unsupported_topk_indices_global.topk_indices_global_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_topk_indices_global), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
}

TEST(Fp32SparseAttnTargetTest, RejectsNonContinuousIndexTypes) {
    Fp32SparseAttnTarget primitive;
    ProblemPorts strided_q;
    strided_q.q_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_q), kConstraints).isSuccess());
    ProblemPorts strided_k;
    strided_k.k_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_k), kConstraints).isSuccess());
    ProblemPorts strided_lse;
    strided_lse.lse_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_lse), kConstraints).isSuccess());
    ProblemPorts strided_topk_indices;
    strided_topk_indices.topk_indices_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_topk_indices), kConstraints).isSuccess());
    ProblemPorts strided_target;
    strided_target.target_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_target), kConstraints).isSuccess());
    ProblemPorts strided_softmax_scale;
    strided_softmax_scale.softmax_scale_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_softmax_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_topk_indices_global;
    strided_topk_indices_global.topk_indices_global_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_topk_indices_global), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32SparseAttnTargetTest, RejectsMisalignedAddresses) {
    Fp32SparseAttnTarget primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_q;
    misaligned_q.q = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_q), kConstraints).isSuccess());
    ProblemPorts misaligned_k;
    misaligned_k.k = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_k), kConstraints).isSuccess());
    ProblemPorts misaligned_lse;
    misaligned_lse.lse = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_lse), kConstraints).isSuccess());
    ProblemPorts misaligned_topk_indices;
    misaligned_topk_indices.topk_indices = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_topk_indices), kConstraints).isSuccess());
    ProblemPorts misaligned_target;
    misaligned_target.target = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_target), kConstraints).isSuccess());
    ProblemPorts misaligned_softmax_scale;
    misaligned_softmax_scale.softmax_scale = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_softmax_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32SparseAttnTargetTest, ChecksRequiredAndOptionalAddresses) {
    Fp32SparseAttnTarget primitive;
    ProblemPorts null_q;
    null_q.q = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_q), kConstraints).isSuccess());
    ProblemPorts null_k;
    null_k.k = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_k), kConstraints).isSuccess());
    ProblemPorts null_lse;
    null_lse.lse = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_lse), kConstraints).isSuccess());
    ProblemPorts null_topk_indices;
    null_topk_indices.topk_indices = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_topk_indices), kConstraints).isSuccess());
    ProblemPorts null_target;
    null_target.target = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_target), kConstraints).isSuccess());
    ProblemPorts null_softmax_scale;
    null_softmax_scale.softmax_scale = nullptr;
    try {
        makeProblem(null_softmax_scale);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_topk_indices_global;
    null_topk_indices_global.topk_indices_global = nullptr;
    try {
        makeProblem(null_topk_indices_global);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32SparseAttnTargetTest, ClonesIntoDistinctConfigurableCopies) {
    Fp32SparseAttnTarget prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp32SparseAttnTarget");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<SparseAttnTargetProblem>&>(*first).configure(makeProblem(ports));
    alignas(16) unsigned char other_memory[16]{};
    {
        ProblemPorts other;
        other.q = other_memory;
        static_cast<Primitive<SparseAttnTargetProblem>&>(*second).configure(makeProblem(other));
    }
    // Workspace checks alone do not prove pointer independence.
    // RebindsAddressesAndOutlivesOpsAndArgs covers execution with separate plans.
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp32SparseAttnTargetTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original         = makeProblem(ports).getProblemKey();
        const std::int64_t q_dims[] = {1, 4, 2, 33};
        ports.q_dims                = q_dims;
        const std::int64_t k_dims[] = {1, 4, 33};
        ports.k_dims                = k_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp32SparseAttnTarget primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

}  // namespace ftrain
