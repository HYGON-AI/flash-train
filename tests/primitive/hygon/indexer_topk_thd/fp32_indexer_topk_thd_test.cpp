// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/indexer_topk_thd/problem.hpp"
#include "flash_train/primitive/hygon/indexer_topk_thd/fp32_indexer_topk_thd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kInputValuesDims[2] = {4, 4};
constexpr std::int64_t kCuSeqlensQDims[1] = {2};
constexpr std::int64_t kCuSeqlensKvDims[1] = {2};
constexpr std::int64_t kQCausalOffsetsDims[1] = {1};
constexpr std::int64_t kOutIndicesDims[2] = {4, 2};
constexpr std::int64_t kOutValuesDims[2] = {4, 2};

struct ProblemPorts {
    void* input_values                        = kDefaultInputValuesMemory;
    const std::int64_t* input_values_dims     = kInputValuesDims;
    FTrainNumericType input_values_type       = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType input_values_index        = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* cu_seqlens_q                        = kDefaultCuSeqlensQMemory;
    const std::int64_t* cu_seqlens_q_dims     = kCuSeqlensQDims;
    FTrainNumericType cu_seqlens_q_type       = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType cu_seqlens_q_index        = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* cu_seqlens_kv                       = kDefaultCuSeqlensKvMemory;
    const std::int64_t* cu_seqlens_kv_dims    = kCuSeqlensKvDims;
    FTrainNumericType cu_seqlens_kv_type      = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType cu_seqlens_kv_index       = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* q_causal_offsets                    = kDefaultQCausalOffsetsMemory;
    const std::int64_t* q_causal_offsets_dims = kQCausalOffsetsDims;
    FTrainNumericType q_causal_offsets_type   = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType q_causal_offsets_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* out_indices                         = kDefaultOutIndicesMemory;
    const std::int64_t* out_indices_dims      = kOutIndicesDims;
    FTrainNumericType out_indices_type        = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType out_indices_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* out_values                          = kDefaultOutValuesMemory;
    const std::int64_t* out_values_dims       = kOutValuesDims;
    FTrainNumericType out_values_type         = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType out_values_index          = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* ratio                               = &kDefaultRatio;
    FTrainNumericType ratio_type              = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType ratio_index               = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* top_k                               = &kDefaultTopK;
    FTrainNumericType top_k_type              = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType top_k_index               = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* order_values                        = &kDefaultOrderValues;
    FTrainNumericType order_values_type       = FTRAIN_NUMERIC_TYPE_UINT8;
    FTrainIndexType order_values_index        = FTRAIN_INDEX_TYPE_CONTINUOUS;
    alignas(16) static unsigned char kDefaultInputValuesMemory[16];
    alignas(16) static unsigned char kDefaultCuSeqlensQMemory[16];
    alignas(16) static unsigned char kDefaultCuSeqlensKvMemory[16];
    alignas(16) static unsigned char kDefaultQCausalOffsetsMemory[16];
    alignas(16) static unsigned char kDefaultOutIndicesMemory[16];
    alignas(16) static unsigned char kDefaultOutValuesMemory[16];
    static std::int64_t kDefaultRatio;
    static std::int64_t kDefaultTopK;
    static std::uint8_t kDefaultOrderValues;
};

alignas(16) unsigned char ProblemPorts::kDefaultInputValuesMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultCuSeqlensQMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultCuSeqlensKvMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultQCausalOffsetsMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultOutIndicesMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultOutValuesMemory[16];

std::int64_t ProblemPorts::kDefaultRatio = 1;

std::int64_t ProblemPorts::kDefaultTopK = 2;

std::uint8_t ProblemPorts::kDefaultOrderValues = 1;

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

IndexerTopKThdProblem makeProblem(const ProblemPorts& ports) {
    return IndexerTopKThdProblem{
        makeDeviceTensor(ports.input_values, ports.input_values_dims, 2, ports.input_values_type,
                         ports.input_values_index),
        makeDeviceTensor(ports.cu_seqlens_q, ports.cu_seqlens_q_dims, 1, ports.cu_seqlens_q_type,
                         ports.cu_seqlens_q_index),
        makeDeviceTensor(ports.cu_seqlens_kv, ports.cu_seqlens_kv_dims, 1, ports.cu_seqlens_kv_type,
                         ports.cu_seqlens_kv_index),
        makeDeviceTensor(ports.q_causal_offsets, ports.q_causal_offsets_dims, 1, ports.q_causal_offsets_type,
                         ports.q_causal_offsets_index),
        makeDeviceTensor(ports.out_indices, ports.out_indices_dims, 2, ports.out_indices_type, ports.out_indices_index),
        makeDeviceTensor(ports.out_values, ports.out_values_dims, 2, ports.out_values_type, ports.out_values_index),
        makeHostScalar(ports.ratio, ports.ratio_type, ports.ratio_index),
        makeHostScalar(ports.top_k, ports.top_k_type, ports.top_k_index),
        makeHostScalar(ports.order_values, ports.order_values_type, ports.order_values_index),
        IndexerTopKThdAttributes{}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Fp32IndexerTopKThdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_input_values[16];
    rebound.input_values = other_input_values;
    alignas(16) static unsigned char other_cu_seqlens_q[16];
    rebound.cu_seqlens_q = other_cu_seqlens_q;
    alignas(16) static unsigned char other_cu_seqlens_kv[16];
    rebound.cu_seqlens_kv = other_cu_seqlens_kv;
    alignas(16) static unsigned char other_q_causal_offsets[16];
    rebound.q_causal_offsets = other_q_causal_offsets;
    alignas(16) static unsigned char other_out_indices[16];
    rebound.out_indices = other_out_indices;
    alignas(16) static unsigned char other_out_values[16];
    rebound.out_values              = other_out_values;
    std::int64_t other_ratio        = 1;
    rebound.ratio                   = &other_ratio;
    std::int64_t other_top_k        = 2;
    rebound.top_k                   = &other_top_k;
    std::uint8_t other_order_values = 1;
    rebound.order_values            = &other_order_values;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp32IndexerTopKThdTest, AcceptsContinuousAlignedOperands) {
    Fp32IndexerTopKThd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "IndexerTopKThd");
}

TEST(Fp32IndexerTopKThdTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp32IndexerTopKThd primitive;
    ProblemPorts unsupported_input_values;
    unsupported_input_values.input_values_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_input_values), kConstraints).isSuccess());
    ProblemPorts unsupported_cu_seqlens_q;
    unsupported_cu_seqlens_q.cu_seqlens_q_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_cu_seqlens_q), kConstraints).isSuccess());
    ProblemPorts unsupported_cu_seqlens_kv;
    unsupported_cu_seqlens_kv.cu_seqlens_kv_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_cu_seqlens_kv), kConstraints).isSuccess());
    ProblemPorts unsupported_q_causal_offsets;
    unsupported_q_causal_offsets.q_causal_offsets_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_q_causal_offsets), kConstraints).isSuccess());
    ProblemPorts unsupported_out_indices;
    unsupported_out_indices.out_indices_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_out_indices), kConstraints).isSuccess());
    ProblemPorts unsupported_out_values;
    unsupported_out_values.out_values_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_out_values), kConstraints).isSuccess());
    ProblemPorts unsupported_ratio;
    unsupported_ratio.ratio_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_top_k;
    unsupported_top_k.top_k_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_top_k), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_order_values;
    unsupported_order_values.order_values_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_order_values), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
}

TEST(Fp32IndexerTopKThdTest, RejectsNonContinuousIndexTypes) {
    Fp32IndexerTopKThd primitive;
    ProblemPorts strided_input_values;
    strided_input_values.input_values_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_input_values), kConstraints).isSuccess());
    ProblemPorts strided_cu_seqlens_q;
    strided_cu_seqlens_q.cu_seqlens_q_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_cu_seqlens_q), kConstraints).isSuccess());
    ProblemPorts strided_cu_seqlens_kv;
    strided_cu_seqlens_kv.cu_seqlens_kv_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_cu_seqlens_kv), kConstraints).isSuccess());
    ProblemPorts strided_q_causal_offsets;
    strided_q_causal_offsets.q_causal_offsets_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_q_causal_offsets), kConstraints).isSuccess());
    ProblemPorts strided_out_indices;
    strided_out_indices.out_indices_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_out_indices), kConstraints).isSuccess());
    ProblemPorts strided_out_values;
    strided_out_values.out_values_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_out_values), kConstraints).isSuccess());
    ProblemPorts strided_ratio;
    strided_ratio.ratio_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_top_k;
    strided_top_k.top_k_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_top_k), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_order_values;
    strided_order_values.order_values_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_order_values), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32IndexerTopKThdTest, RejectsMisalignedAddresses) {
    Fp32IndexerTopKThd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_input_values;
    misaligned_input_values.input_values = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_input_values), kConstraints).isSuccess());
    ProblemPorts misaligned_cu_seqlens_q;
    misaligned_cu_seqlens_q.cu_seqlens_q = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_cu_seqlens_q), kConstraints).isSuccess());
    ProblemPorts misaligned_cu_seqlens_kv;
    misaligned_cu_seqlens_kv.cu_seqlens_kv = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_cu_seqlens_kv), kConstraints).isSuccess());
    ProblemPorts misaligned_q_causal_offsets;
    misaligned_q_causal_offsets.q_causal_offsets = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_q_causal_offsets), kConstraints).isSuccess());
    ProblemPorts misaligned_out_indices;
    misaligned_out_indices.out_indices = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_out_indices), kConstraints).isSuccess());
    ProblemPorts misaligned_out_values;
    misaligned_out_values.out_values = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_out_values), kConstraints).isSuccess());
    ProblemPorts misaligned_ratio;
    misaligned_ratio.ratio = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_top_k;
    misaligned_top_k.top_k = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_top_k), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32IndexerTopKThdTest, ChecksRequiredAndOptionalAddresses) {
    Fp32IndexerTopKThd primitive;
    ProblemPorts null_input_values;
    null_input_values.input_values = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_input_values), kConstraints).isSuccess());
    ProblemPorts null_cu_seqlens_q;
    null_cu_seqlens_q.cu_seqlens_q = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_cu_seqlens_q), kConstraints).isSuccess());
    ProblemPorts null_cu_seqlens_kv;
    null_cu_seqlens_kv.cu_seqlens_kv = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_cu_seqlens_kv), kConstraints).isSuccess());
    ProblemPorts null_q_causal_offsets;
    null_q_causal_offsets.q_causal_offsets = nullptr;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(null_q_causal_offsets), kConstraints).isSuccess());
    ProblemPorts null_out_indices;
    null_out_indices.out_indices = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_out_indices), kConstraints).isSuccess());
    ProblemPorts null_out_values;
    null_out_values.out_values = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_out_values), kConstraints).isSuccess());
    ProblemPorts null_ratio;
    null_ratio.ratio = nullptr;
    try {
        makeProblem(null_ratio);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_top_k;
    null_top_k.top_k = nullptr;
    try {
        makeProblem(null_top_k);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_order_values;
    null_order_values.order_values = nullptr;
    try {
        makeProblem(null_order_values);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32IndexerTopKThdTest, ClonesIntoIndependentConfigurableCopies) {
    Fp32IndexerTopKThd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "IndexerTopKThd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<IndexerTopKThdProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<IndexerTopKThdProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp32IndexerTopKThdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original                    = makeProblem(ports).getProblemKey();
        const std::int64_t input_values_dims[] = {4, 2147483648};
        ports.input_values_dims                = input_values_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp32IndexerTopKThd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

}  // namespace ftrain
