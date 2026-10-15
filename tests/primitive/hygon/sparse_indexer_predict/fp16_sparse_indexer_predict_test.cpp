// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/sparse_indexer_predict/problem.hpp"
#include "flash_train/primitive/hygon/sparse_indexer_predict/fp16_sparse_indexer_predict.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kQDims[4] = {1, 4, 2, 32};
constexpr std::int64_t kKDims[3] = {1, 4, 32};
constexpr std::int64_t kWDims[3] = {1, 4, 2};
constexpr std::int64_t kTopkIndicesDims[3] = {1, 4, 4};
constexpr std::int64_t kPredictDims[3] = {1, 4, 4};

struct ProblemPorts {
    void* q                                    = kDefaultQMemory;
    const std::int64_t* q_dims                 = kQDims;
    FTrainNumericType q_type                   = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType q_index                    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* k                                    = kDefaultKMemory;
    const std::int64_t* k_dims                 = kKDims;
    FTrainNumericType k_type                   = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType k_index                    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* w                                    = kDefaultWMemory;
    const std::int64_t* w_dims                 = kWDims;
    FTrainNumericType w_type                   = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType w_index                    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* topk_indices                         = kDefaultTopkIndicesMemory;
    const std::int64_t* topk_indices_dims      = kTopkIndicesDims;
    FTrainNumericType topk_indices_type        = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType topk_indices_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* predict                              = kDefaultPredictMemory;
    const std::int64_t* predict_dims           = kPredictDims;
    FTrainNumericType predict_type             = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType predict_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* sm_scale                             = &kDefaultSmScale;
    FTrainNumericType sm_scale_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType sm_scale_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* topk_indices_global                  = &kDefaultTopkIndicesGlobal;
    FTrainNumericType topk_indices_global_type = FTRAIN_NUMERIC_TYPE_UINT8;
    FTrainIndexType topk_indices_global_index  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type             = FTRAIN_NUMERIC_TYPE_FP16;
    alignas(16) static unsigned char kDefaultQMemory[16];
    alignas(16) static unsigned char kDefaultKMemory[16];
    alignas(16) static unsigned char kDefaultWMemory[16];
    alignas(16) static unsigned char kDefaultTopkIndicesMemory[16];
    alignas(16) static unsigned char kDefaultPredictMemory[16];
    static float kDefaultSmScale;
    static std::uint8_t kDefaultTopkIndicesGlobal;
};

alignas(16) unsigned char ProblemPorts::kDefaultQMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultKMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultWMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultTopkIndicesMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultPredictMemory[16];
float ProblemPorts::kDefaultSmScale = 0.25F;

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

SparseIndexerPredictProblem makeProblem(const ProblemPorts& ports) {
    return SparseIndexerPredictProblem{
        makeDeviceTensor(ports.q, ports.q_dims, 4, ports.q_type, ports.q_index),
        makeDeviceTensor(ports.k, ports.k_dims, 3, ports.k_type, ports.k_index),
        makeDeviceTensor(ports.w, ports.w_dims, 3, ports.w_type, ports.w_index),
        makeDeviceTensor(ports.topk_indices, ports.topk_indices_dims, 3, ports.topk_indices_type,
                         ports.topk_indices_index),
        makeDeviceTensor(ports.predict, ports.predict_dims, 3, ports.predict_type, ports.predict_index),
        makeHostScalar(ports.sm_scale, ports.sm_scale_type, ports.sm_scale_index),
        makeHostScalar(ports.topk_indices_global, ports.topk_indices_global_type, ports.topk_indices_global_index),
        SparseIndexerPredictAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Fp16SparseIndexerPredictTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_q[16];
    rebound.q = other_q;
    alignas(16) static unsigned char other_k[16];
    rebound.k = other_k;
    alignas(16) static unsigned char other_w[16];
    rebound.w = other_w;
    alignas(16) static unsigned char other_topk_indices[16];
    rebound.topk_indices = other_topk_indices;
    alignas(16) static unsigned char other_predict[16];
    rebound.predict                        = other_predict;
    float other_sm_scale                   = 0.25F;
    rebound.sm_scale                       = &other_sm_scale;
    std::uint8_t other_topk_indices_global = 0;
    rebound.topk_indices_global            = &other_topk_indices_global;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp16SparseIndexerPredictTest, AcceptsContinuousAlignedOperands) {
    Fp16SparseIndexerPredict primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp16SparseIndexerPredict");
}

TEST(Fp16SparseIndexerPredictTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp16SparseIndexerPredict primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_q;
    unsupported_q.q_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_q), kConstraints).isSuccess());
    ProblemPorts unsupported_k;
    unsupported_k.k_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_k), kConstraints).isSuccess());
    ProblemPorts unsupported_w;
    unsupported_w.w_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_w), kConstraints).isSuccess());
    ProblemPorts unsupported_topk_indices;
    unsupported_topk_indices.topk_indices_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_topk_indices), kConstraints).isSuccess());
    ProblemPorts unsupported_predict;
    unsupported_predict.predict_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_predict), kConstraints).isSuccess());
    ProblemPorts unsupported_sm_scale;
    unsupported_sm_scale.sm_scale_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_sm_scale), kConstraints).isSuccess());
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

TEST(Fp16SparseIndexerPredictTest, RejectsNonContinuousIndexTypes) {
    Fp16SparseIndexerPredict primitive;
    ProblemPorts strided_q;
    strided_q.q_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_q), kConstraints).isSuccess());
    ProblemPorts strided_k;
    strided_k.k_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_k), kConstraints).isSuccess());
    ProblemPorts strided_w;
    strided_w.w_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_w), kConstraints).isSuccess());
    ProblemPorts strided_topk_indices;
    strided_topk_indices.topk_indices_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_topk_indices), kConstraints).isSuccess());
    ProblemPorts strided_predict;
    strided_predict.predict_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_predict), kConstraints).isSuccess());
    ProblemPorts strided_sm_scale;
    strided_sm_scale.sm_scale_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_sm_scale), kConstraints).isSuccess());
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

TEST(Fp16SparseIndexerPredictTest, RejectsMisalignedAddresses) {
    Fp16SparseIndexerPredict primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_q;
    misaligned_q.q = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_q), kConstraints).isSuccess());
    ProblemPorts misaligned_k;
    misaligned_k.k = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_k), kConstraints).isSuccess());
    ProblemPorts misaligned_w;
    misaligned_w.w = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_w), kConstraints).isSuccess());
    ProblemPorts misaligned_topk_indices;
    misaligned_topk_indices.topk_indices = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_topk_indices), kConstraints).isSuccess());
    ProblemPorts misaligned_predict;
    misaligned_predict.predict = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_predict), kConstraints).isSuccess());
    ProblemPorts misaligned_sm_scale;
    misaligned_sm_scale.sm_scale = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_sm_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp16SparseIndexerPredictTest, ChecksRequiredAndOptionalAddresses) {
    Fp16SparseIndexerPredict primitive;
    ProblemPorts null_q;
    null_q.q = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_q), kConstraints).isSuccess());
    ProblemPorts null_k;
    null_k.k = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_k), kConstraints).isSuccess());
    ProblemPorts null_w;
    null_w.w = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_w), kConstraints).isSuccess());
    ProblemPorts null_topk_indices;
    null_topk_indices.topk_indices = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_topk_indices), kConstraints).isSuccess());
    ProblemPorts null_predict;
    null_predict.predict = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_predict), kConstraints).isSuccess());
    ProblemPorts null_sm_scale;
    null_sm_scale.sm_scale = nullptr;
    try {
        makeProblem(null_sm_scale);
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

TEST(Fp16SparseIndexerPredictTest, ClonesIntoIndependentConfigurableCopies) {
    Fp16SparseIndexerPredict prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp16SparseIndexerPredict");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<SparseIndexerPredictProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<SparseIndexerPredictProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp16SparseIndexerPredictTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original         = makeProblem(ports).getProblemKey();
        const std::int64_t q_dims[] = {1, 4, 2, 33};
        ports.q_dims                = q_dims;
        const std::int64_t k_dims[] = {1, 4, 33};
        ports.k_dims                = k_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp16SparseIndexerPredict primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

}  // namespace ftrain
