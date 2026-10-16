// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/dense_indexer_score_thd/problem.hpp"
#include "flash_train/primitive/hygon/dense_indexer_score_thd/bf16_dense_indexer_score_thd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kQDims[3] = {4, 2, 32};
constexpr std::int64_t kKDims[3] = {4, 1, 32};
constexpr std::int64_t kWDims[2] = {4, 2};
constexpr std::int64_t kCuSeqlensQDims[1] = {2};
constexpr std::int64_t kCuSeqlensKvDims[1] = {2};
constexpr std::int64_t kQCausalOffsetsDims[1] = {1};
constexpr std::int64_t kOutDims[2] = {4, 4};
constexpr std::int64_t kDenomDims[1] = {4};

struct ProblemPorts {
    void* q                                   = kDefaultQMemory;
    const std::int64_t* q_dims                = kQDims;
    FTrainNumericType q_type                  = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType q_index                   = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* k                                   = kDefaultKMemory;
    const std::int64_t* k_dims                = kKDims;
    FTrainNumericType k_type                  = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType k_index                   = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* w                                   = kDefaultWMemory;
    const std::int64_t* w_dims                = kWDims;
    FTrainNumericType w_type                  = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType w_index                   = FTRAIN_INDEX_TYPE_CONTINUOUS;
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
    void* out                                 = kDefaultOutMemory;
    const std::int64_t* out_dims              = kOutDims;
    FTrainNumericType out_type                = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType out_index                 = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* denom                               = kDefaultDenomMemory;
    const std::int64_t* denom_dims            = kDenomDims;
    FTrainNumericType denom_type              = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType denom_index               = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* ratio                               = &kDefaultRatio;
    FTrainNumericType ratio_type              = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType ratio_index               = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* sm_scale                            = &kDefaultSmScale;
    FTrainNumericType sm_scale_type           = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType sm_scale_index            = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* max_seqlen_kv                       = &kDefaultMaxSeqlenKv;
    FTrainNumericType max_seqlen_kv_type      = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType max_seqlen_kv_index       = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type            = FTRAIN_NUMERIC_TYPE_BF16;
    alignas(16) static unsigned char kDefaultQMemory[16];
    alignas(16) static unsigned char kDefaultKMemory[16];
    alignas(16) static unsigned char kDefaultWMemory[16];
    alignas(16) static unsigned char kDefaultCuSeqlensQMemory[16];
    alignas(16) static unsigned char kDefaultCuSeqlensKvMemory[16];
    alignas(16) static unsigned char kDefaultQCausalOffsetsMemory[16];
    alignas(16) static unsigned char kDefaultOutMemory[16];
    alignas(16) static unsigned char kDefaultDenomMemory[16];
    static std::int64_t kDefaultRatio;
    static float kDefaultSmScale;
    static std::int64_t kDefaultMaxSeqlenKv;
};

alignas(16) unsigned char ProblemPorts::kDefaultQMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultKMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultWMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultCuSeqlensQMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultCuSeqlensKvMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultQCausalOffsetsMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultOutMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDenomMemory[16];

std::int64_t ProblemPorts::kDefaultRatio = 1;

float ProblemPorts::kDefaultSmScale = 0.25F;

std::int64_t ProblemPorts::kDefaultMaxSeqlenKv = 4;

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

DenseIndexerScoreThdProblem makeProblem(const ProblemPorts& ports) {
    return DenseIndexerScoreThdProblem{
        makeDeviceTensor(ports.q, ports.q_dims, 3, ports.q_type, ports.q_index),
        makeDeviceTensor(ports.k, ports.k_dims, 3, ports.k_type, ports.k_index),
        makeDeviceTensor(ports.w, ports.w_dims, 2, ports.w_type, ports.w_index),
        makeDeviceTensor(ports.cu_seqlens_q, ports.cu_seqlens_q_dims, 1, ports.cu_seqlens_q_type,
                         ports.cu_seqlens_q_index),
        makeDeviceTensor(ports.cu_seqlens_kv, ports.cu_seqlens_kv_dims, 1, ports.cu_seqlens_kv_type,
                         ports.cu_seqlens_kv_index),
        makeDeviceTensor(ports.q_causal_offsets, ports.q_causal_offsets_dims, 1, ports.q_causal_offsets_type,
                         ports.q_causal_offsets_index),
        makeDeviceTensor(ports.out, ports.out_dims, 2, ports.out_type, ports.out_index),
        makeDeviceTensor(ports.denom, ports.denom_dims, 1, ports.denom_type, ports.denom_index),
        makeHostScalar(ports.ratio, ports.ratio_type, ports.ratio_index),
        makeHostScalar(ports.sm_scale, ports.sm_scale_type, ports.sm_scale_index),
        makeHostScalar(ports.max_seqlen_kv, ports.max_seqlen_kv_type, ports.max_seqlen_kv_index),
        DenseIndexerScoreThdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

struct Ports {
    alignas(16) unsigned char memory[16]{};
    std::int64_t ratio           = 2;
    std::int64_t max_seqlen      = 8;
    float scale                  = 0.25F;
    FTrainNumericType scale_type = FTRAIN_NUMERIC_TYPE_FP32;
    bool scale_is_host           = true;
    std::int64_t output_width    = 8;
    std::int64_t kv_offsets      = 3;
};

Tensor device(void* memory, std::initializer_list<std::int64_t> shape, FTrainNumericType dtype) {
    return Tensor{StorageView{FTrainStorageView{memory, shape.begin(), nullptr, static_cast<std::uint8_t>(shape.size()),
                                                dtype, FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
}

Tensor scalar(void* memory, FTrainNumericType dtype, bool is_host = true) {
    return Tensor{
        StorageView{FTrainStorageView{memory, nullptr, nullptr, 0, dtype, FTRAIN_INDEX_TYPE_CONTINUOUS, is_host}}};
}

DenseIndexerScoreThdProblem makeProblem(Ports& p) {
    return {device(p.memory, {4, 4, 32}, FTRAIN_NUMERIC_TYPE_BF16),
            device(p.memory, {8, 1, 32}, FTRAIN_NUMERIC_TYPE_BF16),
            device(p.memory, {4, 4}, FTRAIN_NUMERIC_TYPE_BF16),
            device(p.memory, {3}, FTRAIN_NUMERIC_TYPE_INT32),
            device(p.memory, {p.kv_offsets}, FTRAIN_NUMERIC_TYPE_INT32),
            device(nullptr, {2}, FTRAIN_NUMERIC_TYPE_INT32),
            device(p.memory, {4, p.output_width}, FTRAIN_NUMERIC_TYPE_FP32),
            device(p.memory, {4}, FTRAIN_NUMERIC_TYPE_FP32),
            scalar(&p.ratio, FTRAIN_NUMERIC_TYPE_INT64),
            scalar(&p.scale, p.scale_type, p.scale_is_host),
            scalar(&p.max_seqlen, FTRAIN_NUMERIC_TYPE_INT64),
            DenseIndexerScoreThdAttributes{FTRAIN_NUMERIC_TYPE_BF16}};
}

}  // namespace

TEST(Bf16DenseIndexerScoreThdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_q[16];
    rebound.q = other_q;
    alignas(16) static unsigned char other_k[16];
    rebound.k = other_k;
    alignas(16) static unsigned char other_w[16];
    rebound.w = other_w;
    alignas(16) static unsigned char other_cu_seqlens_q[16];
    rebound.cu_seqlens_q = other_cu_seqlens_q;
    alignas(16) static unsigned char other_cu_seqlens_kv[16];
    rebound.cu_seqlens_kv = other_cu_seqlens_kv;
    alignas(16) static unsigned char other_q_causal_offsets[16];
    rebound.q_causal_offsets = other_q_causal_offsets;
    alignas(16) static unsigned char other_out[16];
    rebound.out = other_out;
    alignas(16) static unsigned char other_denom[16];
    rebound.denom                    = other_denom;
    std::int64_t other_ratio         = 1;
    rebound.ratio                    = &other_ratio;
    float other_sm_scale             = 0.25F;
    rebound.sm_scale                 = &other_sm_scale;
    std::int64_t other_max_seqlen_kv = 4;
    rebound.max_seqlen_kv            = &other_max_seqlen_kv;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Bf16DenseIndexerScoreThdTest, AcceptsContinuousAlignedOperands) {
    Bf16DenseIndexerScoreThd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Bf16DenseIndexerScoreThd");
}

TEST(Bf16DenseIndexerScoreThdTest, RejectsUnsupportedComputeAndElementTypes) {
    Bf16DenseIndexerScoreThd primitive;
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
    ProblemPorts unsupported_cu_seqlens_q;
    unsupported_cu_seqlens_q.cu_seqlens_q_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_cu_seqlens_q), kConstraints).isSuccess());
    ProblemPorts unsupported_cu_seqlens_kv;
    unsupported_cu_seqlens_kv.cu_seqlens_kv_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_cu_seqlens_kv), kConstraints).isSuccess());
    ProblemPorts unsupported_q_causal_offsets;
    unsupported_q_causal_offsets.q_causal_offsets_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_q_causal_offsets), kConstraints).isSuccess());
    ProblemPorts unsupported_out;
    unsupported_out.out_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_out), kConstraints).isSuccess());
    ProblemPorts unsupported_denom;
    unsupported_denom.denom_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_denom), kConstraints).isSuccess());
    ProblemPorts unsupported_ratio;
    unsupported_ratio.ratio_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_sm_scale;
    unsupported_sm_scale.sm_scale_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_sm_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_max_seqlen_kv;
    unsupported_max_seqlen_kv.max_seqlen_kv_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_max_seqlen_kv), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
}

TEST(Bf16DenseIndexerScoreThdTest, RejectsNonContinuousIndexTypes) {
    Bf16DenseIndexerScoreThd primitive;
    ProblemPorts strided_q;
    strided_q.q_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_q), kConstraints).isSuccess());
    ProblemPorts strided_k;
    strided_k.k_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_k), kConstraints).isSuccess());
    ProblemPorts strided_w;
    strided_w.w_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_w), kConstraints).isSuccess());
    ProblemPorts strided_cu_seqlens_q;
    strided_cu_seqlens_q.cu_seqlens_q_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_cu_seqlens_q), kConstraints).isSuccess());
    ProblemPorts strided_cu_seqlens_kv;
    strided_cu_seqlens_kv.cu_seqlens_kv_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_cu_seqlens_kv), kConstraints).isSuccess());
    ProblemPorts strided_q_causal_offsets;
    strided_q_causal_offsets.q_causal_offsets_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_q_causal_offsets), kConstraints).isSuccess());
    ProblemPorts strided_out;
    strided_out.out_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_out), kConstraints).isSuccess());
    ProblemPorts strided_denom;
    strided_denom.denom_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_denom), kConstraints).isSuccess());
    ProblemPorts strided_ratio;
    strided_ratio.ratio_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_sm_scale;
    strided_sm_scale.sm_scale_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_sm_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_max_seqlen_kv;
    strided_max_seqlen_kv.max_seqlen_kv_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_max_seqlen_kv), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16DenseIndexerScoreThdTest, RejectsMisalignedAddresses) {
    Bf16DenseIndexerScoreThd primitive;
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
    ProblemPorts misaligned_cu_seqlens_q;
    misaligned_cu_seqlens_q.cu_seqlens_q = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_cu_seqlens_q), kConstraints).isSuccess());
    ProblemPorts misaligned_cu_seqlens_kv;
    misaligned_cu_seqlens_kv.cu_seqlens_kv = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_cu_seqlens_kv), kConstraints).isSuccess());
    ProblemPorts misaligned_q_causal_offsets;
    misaligned_q_causal_offsets.q_causal_offsets = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_q_causal_offsets), kConstraints).isSuccess());
    ProblemPorts misaligned_out;
    misaligned_out.out = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_out), kConstraints).isSuccess());
    ProblemPorts misaligned_denom;
    misaligned_denom.denom = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_denom), kConstraints).isSuccess());
    ProblemPorts misaligned_ratio;
    misaligned_ratio.ratio = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_sm_scale;
    misaligned_sm_scale.sm_scale = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_sm_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_max_seqlen_kv;
    misaligned_max_seqlen_kv.max_seqlen_kv = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_max_seqlen_kv), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16DenseIndexerScoreThdTest, ChecksRequiredAndOptionalAddresses) {
    Bf16DenseIndexerScoreThd primitive;
    ProblemPorts null_q;
    null_q.q = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_q), kConstraints).isSuccess());
    ProblemPorts null_k;
    null_k.k = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_k), kConstraints).isSuccess());
    ProblemPorts null_w;
    null_w.w = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_w), kConstraints).isSuccess());
    ProblemPorts null_cu_seqlens_q;
    null_cu_seqlens_q.cu_seqlens_q = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_cu_seqlens_q), kConstraints).isSuccess());
    ProblemPorts null_cu_seqlens_kv;
    null_cu_seqlens_kv.cu_seqlens_kv = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_cu_seqlens_kv), kConstraints).isSuccess());
    ProblemPorts null_q_causal_offsets;
    null_q_causal_offsets.q_causal_offsets = nullptr;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(null_q_causal_offsets), kConstraints).isSuccess());
    ProblemPorts null_out;
    null_out.out = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_out), kConstraints).isSuccess());
    ProblemPorts null_denom;
    null_denom.denom = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_denom), kConstraints).isSuccess());
    ProblemPorts null_ratio;
    null_ratio.ratio = nullptr;
    try {
        makeProblem(null_ratio);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_sm_scale;
    null_sm_scale.sm_scale = nullptr;
    try {
        makeProblem(null_sm_scale);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_max_seqlen_kv;
    null_max_seqlen_kv.max_seqlen_kv = nullptr;
    try {
        makeProblem(null_max_seqlen_kv);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16DenseIndexerScoreThdTest, ClonesIntoIndependentConfigurableCopies) {
    Bf16DenseIndexerScoreThd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Bf16DenseIndexerScoreThd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<DenseIndexerScoreThdProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<DenseIndexerScoreThdProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Bf16DenseIndexerScoreThdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original         = makeProblem(ports).getProblemKey();
        const std::int64_t q_dims[] = {4, 2, 33};
        ports.q_dims                = q_dims;
        const std::int64_t k_dims[] = {4, 1, 33};
        ports.k_dims                = k_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Bf16DenseIndexerScoreThd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16DenseIndexerScoreThdTest, RejectsTooManyHeads) {
    {
        ProblemPorts ports;

        const std::int64_t q_dims[] = {ports.q_dims[0], 129, ports.q_dims[2]};
        ports.q_dims                = q_dims;

        const std::int64_t w_dims[] = {ports.w_dims[0], 129};
        ports.w_dims                = w_dims;

        auto problem = makeProblem(ports);
        Bf16DenseIndexerScoreThd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16DenseIndexerScoreThdTest, RejectsMissingBf16Instantiation) {
    {
        ProblemPorts ports;

        const std::int64_t q_dims[] = {ports.q_dims[0], ports.q_dims[1], 160};
        ports.q_dims                = q_dims;

        const std::int64_t k_dims[] = {ports.k_dims[0], ports.k_dims[1], 160};
        ports.k_dims                = k_dims;

        auto problem = makeProblem(ports);
        Bf16DenseIndexerScoreThd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16DenseIndexerScoreThdTest, RejectsExcessiveLaunchSharedMemory) {
    {
        ProblemPorts ports;

        const std::int64_t k_dims[] = {100000, ports.k_dims[1], ports.k_dims[2]};
        ports.k_dims                = k_dims;

        const std::int64_t out_dims[] = {ports.out_dims[0], 100000};
        ports.out_dims                = out_dims;

        std::int64_t c_max_seqlen_kv_value = 100000;
        ports.max_seqlen_kv                = &c_max_seqlen_kv_value;

        auto problem = makeProblem(ports);
        Bf16DenseIndexerScoreThd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16DenseIndexerScoreThdTest, PrecisionSpecificWidthCoverage) {
    {
        ProblemPorts ports;

        // BF16 has a D=256 specialization; FP16/FP32 use a runtime width.

        const std::int64_t q_dims[] = {ports.q_dims[0], ports.q_dims[1],
                                       ports.compute_type == FTRAIN_NUMERIC_TYPE_BF16 ? 256 : 160};
        ports.q_dims                = q_dims;

        const std::int64_t k_dims[] = {ports.k_dims[0], ports.k_dims[1],
                                       ports.compute_type == FTRAIN_NUMERIC_TYPE_BF16 ? 256 : 160};
        ports.k_dims                = k_dims;

        auto problem = makeProblem(ports);
        Bf16DenseIndexerScoreThd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_SUCCESS)
                << primitive.getName();
        }
    }
}

TEST(Bf16DenseIndexerScoreThdRegressionTest, ExecutionScalarTypeIsASupportDecision) {
    Ports p;
    p.scale_type = FTRAIN_NUMERIC_TYPE_INT32;
    auto problem = makeProblem(p);
    {
        Bf16DenseIndexerScoreThd primitive;
        EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    }
}

TEST(DenseIndexerScoreThdProblemTest, ValidatesPackedShapesAndHostScalarPlacement) {
    Ports p;
    EXPECT_NO_THROW(makeProblem(p));
    p.output_width = 7;
    EXPECT_THROW(makeProblem(p), Exception);
    p.output_width = 8;
    p.kv_offsets   = 2;
    EXPECT_THROW(makeProblem(p), Exception);
    p.kv_offsets    = 3;
    p.scale_is_host = false;
    EXPECT_THROW(makeProblem(p), Exception);
}

TEST(DenseIndexerScoreThdProblemTest, CacheTracksStructuralScalarsButNotExecutionScaleOrAddresses) {
    Ports p, rebound;
    const auto key = makeProblem(p).getProblemKey();
    EXPECT_EQ(key, makeProblem(rebound).getProblemKey());
    p.scale = 0.5F;
    EXPECT_EQ(key, makeProblem(p).getProblemKey());
    p.ratio = 4;
    EXPECT_NE(key, makeProblem(p).getProblemKey());
    p.ratio = 0;
    EXPECT_THROW(makeProblem(p), Exception);
}

TEST(Bf16DenseIndexerScoreThdTest, LauncherRejectsUnsupportedHeadDimensions) {
    for (int width : {144, 160, 192, 240, 272, 432, 464, 528, 560}) {
        EXPECT_FALSE(launchBf16DenseIndexerScoreThd(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                    nullptr, nullptr, 4, 4, 4, 2, 1, width, 1, 1, 0.25F))
            << width;
    }
}

}  // namespace ftrain
