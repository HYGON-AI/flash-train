// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/csa_compressor_bwd/problem.hpp"
#include "flash_train/primitive/hygon/csa_compressor_bwd/bf16_csa_compressor_bwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kKvDims[2]             = {8, 32};
constexpr std::int64_t kScoreDims[2]          = {8, 32};
constexpr std::int64_t kApeDims[2]            = {4, 32};
constexpr std::int64_t kCuSeqlensDims[1]      = {2};
constexpr std::int64_t kCuSeqlensCompDims[1]  = {2};
constexpr std::int64_t kGradCompressedDims[2] = {2, 32};
constexpr std::int64_t kGradKvDims[2]         = {8, 32};
constexpr std::int64_t kGradScoreDims[2]      = {8, 32};
constexpr std::int64_t kGradApeDims[2]        = {4, 32};

struct ProblemPorts {
    void* kv                                 = kDefaultKvMemory;
    const std::int64_t* kv_dims              = kKvDims;
    FTrainNumericType kv_type                = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType kv_index                 = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* score                              = kDefaultScoreMemory;
    const std::int64_t* score_dims           = kScoreDims;
    FTrainNumericType score_type             = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType score_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* ape                                = kDefaultApeMemory;
    const std::int64_t* ape_dims             = kApeDims;
    FTrainNumericType ape_type               = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType ape_index                = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* cu_seqlens                         = kDefaultCuSeqlensMemory;
    const std::int64_t* cu_seqlens_dims      = kCuSeqlensDims;
    FTrainNumericType cu_seqlens_type        = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType cu_seqlens_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* cu_seqlens_comp                    = kDefaultCuSeqlensCompMemory;
    const std::int64_t* cu_seqlens_comp_dims = kCuSeqlensCompDims;
    FTrainNumericType cu_seqlens_comp_type   = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType cu_seqlens_comp_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_compressed                    = kDefaultGradCompressedMemory;
    const std::int64_t* grad_compressed_dims = kGradCompressedDims;
    FTrainNumericType grad_compressed_type   = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_compressed_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_kv                            = kDefaultGradKvMemory;
    const std::int64_t* grad_kv_dims         = kGradKvDims;
    FTrainNumericType grad_kv_type           = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_kv_index            = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_score                         = kDefaultGradScoreMemory;
    const std::int64_t* grad_score_dims      = kGradScoreDims;
    FTrainNumericType grad_score_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_score_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_ape                           = kDefaultGradApeMemory;
    const std::int64_t* grad_ape_dims        = kGradApeDims;
    FTrainNumericType grad_ape_type          = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_ape_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* ratio                              = &kDefaultRatio;
    FTrainNumericType ratio_type             = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType ratio_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* head_dim                           = &kDefaultHeadDim;
    FTrainNumericType head_dim_type          = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType head_dim_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* coff                               = &kDefaultCoff;
    FTrainNumericType coff_type              = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType coff_index               = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type           = FTRAIN_NUMERIC_TYPE_BF16;
    alignas(16) static unsigned char kDefaultKvMemory[16];
    alignas(16) static unsigned char kDefaultScoreMemory[16];
    alignas(16) static unsigned char kDefaultApeMemory[16];
    alignas(16) static unsigned char kDefaultCuSeqlensMemory[16];
    alignas(16) static unsigned char kDefaultCuSeqlensCompMemory[16];
    alignas(16) static unsigned char kDefaultGradCompressedMemory[16];
    alignas(16) static unsigned char kDefaultGradKvMemory[16];
    alignas(16) static unsigned char kDefaultGradScoreMemory[16];
    alignas(16) static unsigned char kDefaultGradApeMemory[16];
    static std::int64_t kDefaultRatio;
    static std::int64_t kDefaultHeadDim;
    static std::int64_t kDefaultCoff;
};

alignas(16) unsigned char ProblemPorts::kDefaultKvMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultScoreMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultApeMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultCuSeqlensMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultCuSeqlensCompMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradCompressedMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradKvMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradScoreMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradApeMemory[16];

std::int64_t ProblemPorts::kDefaultRatio = 4;

std::int64_t ProblemPorts::kDefaultHeadDim = 32;

std::int64_t ProblemPorts::kDefaultCoff = 1;

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

CSACompressorBwdProblem makeProblem(const ProblemPorts& ports) {
    return CSACompressorBwdProblem{
        makeDeviceTensor(ports.kv, ports.kv_dims, 2, ports.kv_type, ports.kv_index),
        makeDeviceTensor(ports.score, ports.score_dims, 2, ports.score_type, ports.score_index),
        makeDeviceTensor(ports.ape, ports.ape_dims, 2, ports.ape_type, ports.ape_index),
        makeDeviceTensor(ports.cu_seqlens, ports.cu_seqlens_dims, 1, ports.cu_seqlens_type, ports.cu_seqlens_index),
        makeDeviceTensor(ports.cu_seqlens_comp, ports.cu_seqlens_comp_dims, 1, ports.cu_seqlens_comp_type,
                         ports.cu_seqlens_comp_index),
        makeDeviceTensor(ports.grad_compressed, ports.grad_compressed_dims, 2, ports.grad_compressed_type,
                         ports.grad_compressed_index),
        makeDeviceTensor(ports.grad_kv, ports.grad_kv_dims, 2, ports.grad_kv_type, ports.grad_kv_index),
        makeDeviceTensor(ports.grad_score, ports.grad_score_dims, 2, ports.grad_score_type, ports.grad_score_index),
        makeDeviceTensor(ports.grad_ape, ports.grad_ape_dims, 2, ports.grad_ape_type, ports.grad_ape_index),
        makeHostScalar(ports.ratio, ports.ratio_type, ports.ratio_index),
        makeHostScalar(ports.head_dim, ports.head_dim_type, ports.head_dim_index),
        makeHostScalar(ports.coff, ports.coff_type, ports.coff_index),
        CSACompressorBwdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

// StorageView copies dims into its own vector during construction; shape is not retained.
Tensor device(void* memory, std::initializer_list<std::int64_t> shape, FTrainNumericType dtype) {
    return Tensor{StorageView{FTrainStorageView{memory, shape.begin(), nullptr, static_cast<std::uint8_t>(shape.size()),
                                                dtype, FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
}

Tensor scalar(const std::int64_t& value) {
    return Tensor{StorageView{FTrainStorageView{const_cast<std::int64_t*>(&value), nullptr, nullptr, 0,
                                                FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true}}};
}

CSACompressorBwdProblem makeProblem(void* memory, FTrainNumericType compute_type) {
    static const std::int64_t ratio    = 4;
    static const std::int64_t head_dim = 32;
    static const std::int64_t coff     = 1;
    return {device(memory, {64, 32}, FTRAIN_NUMERIC_TYPE_BF16),
            device(memory, {64, 32}, FTRAIN_NUMERIC_TYPE_BF16),
            device(memory, {4, 32}, FTRAIN_NUMERIC_TYPE_FP32),
            device(memory, {3}, FTRAIN_NUMERIC_TYPE_INT32),
            device(memory, {3}, FTRAIN_NUMERIC_TYPE_INT32),
            device(memory, {16, 32}, FTRAIN_NUMERIC_TYPE_BF16),
            device(memory, {64, 32}, FTRAIN_NUMERIC_TYPE_BF16),
            device(memory, {64, 32}, FTRAIN_NUMERIC_TYPE_BF16),
            device(memory, {4, 32}, FTRAIN_NUMERIC_TYPE_FP32),
            scalar(ratio),
            scalar(head_dim),
            scalar(coff),
            CSACompressorBwdAttributes{compute_type}};
}

}  // namespace

TEST(Bf16CSACompressorBwdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_kv[16];
    rebound.kv = other_kv;
    alignas(16) static unsigned char other_score[16];
    rebound.score = other_score;
    alignas(16) static unsigned char other_ape[16];
    rebound.ape = other_ape;
    alignas(16) static unsigned char other_cu_seqlens[16];
    rebound.cu_seqlens = other_cu_seqlens;
    alignas(16) static unsigned char other_cu_seqlens_comp[16];
    rebound.cu_seqlens_comp = other_cu_seqlens_comp;
    alignas(16) static unsigned char other_grad_compressed[16];
    rebound.grad_compressed = other_grad_compressed;
    alignas(16) static unsigned char other_grad_kv[16];
    rebound.grad_kv = other_grad_kv;
    alignas(16) static unsigned char other_grad_score[16];
    rebound.grad_score = other_grad_score;
    alignas(16) static unsigned char other_grad_ape[16];
    rebound.grad_ape            = other_grad_ape;
    std::int64_t other_ratio    = 4;
    rebound.ratio               = &other_ratio;
    std::int64_t other_head_dim = 32;
    rebound.head_dim            = &other_head_dim;
    std::int64_t other_coff     = 1;
    rebound.coff                = &other_coff;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Bf16CSACompressorBwdTest, AcceptsContinuousAlignedOperands) {
    Bf16CSACompressorBwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "CSACompressorBwd");
}

TEST(Bf16CSACompressorBwdTest, RejectsUnsupportedComputeAndElementTypes) {
    Bf16CSACompressorBwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_kv;
    unsupported_kv.kv_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_kv), kConstraints).isSuccess());
    ProblemPorts unsupported_score;
    unsupported_score.score_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_score), kConstraints).isSuccess());
    ProblemPorts unsupported_ape;
    unsupported_ape.ape_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_ape), kConstraints).isSuccess());
    ProblemPorts unsupported_cu_seqlens;
    unsupported_cu_seqlens.cu_seqlens_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_cu_seqlens), kConstraints).isSuccess());
    ProblemPorts unsupported_cu_seqlens_comp;
    unsupported_cu_seqlens_comp.cu_seqlens_comp_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_cu_seqlens_comp), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_compressed;
    unsupported_grad_compressed.grad_compressed_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_compressed), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_kv;
    unsupported_grad_kv.grad_kv_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_kv), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_score;
    unsupported_grad_score.grad_score_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_score), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_ape;
    unsupported_grad_ape.grad_ape_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_ape), kConstraints).isSuccess());
    ProblemPorts unsupported_ratio;
    unsupported_ratio.ratio_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_head_dim;
    unsupported_head_dim.head_dim_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_head_dim), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_coff;
    unsupported_coff.coff_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_coff), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
}

TEST(Bf16CSACompressorBwdTest, RejectsNonContinuousIndexTypes) {
    Bf16CSACompressorBwd primitive;
    ProblemPorts strided_kv;
    strided_kv.kv_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_kv), kConstraints).isSuccess());
    ProblemPorts strided_score;
    strided_score.score_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_score), kConstraints).isSuccess());
    ProblemPorts strided_ape;
    strided_ape.ape_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_ape), kConstraints).isSuccess());
    ProblemPorts strided_cu_seqlens;
    strided_cu_seqlens.cu_seqlens_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_cu_seqlens), kConstraints).isSuccess());
    ProblemPorts strided_cu_seqlens_comp;
    strided_cu_seqlens_comp.cu_seqlens_comp_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_cu_seqlens_comp), kConstraints).isSuccess());
    ProblemPorts strided_grad_compressed;
    strided_grad_compressed.grad_compressed_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_compressed), kConstraints).isSuccess());
    ProblemPorts strided_grad_kv;
    strided_grad_kv.grad_kv_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_kv), kConstraints).isSuccess());
    ProblemPorts strided_grad_score;
    strided_grad_score.grad_score_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_score), kConstraints).isSuccess());
    ProblemPorts strided_grad_ape;
    strided_grad_ape.grad_ape_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_ape), kConstraints).isSuccess());
    ProblemPorts strided_ratio;
    strided_ratio.ratio_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_head_dim;
    strided_head_dim.head_dim_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_head_dim), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_coff;
    strided_coff.coff_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_coff), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16CSACompressorBwdTest, RejectsMisalignedAddresses) {
    Bf16CSACompressorBwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_kv;
    misaligned_kv.kv = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_kv), kConstraints).isSuccess());
    ProblemPorts misaligned_score;
    misaligned_score.score = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_score), kConstraints).isSuccess());
    ProblemPorts misaligned_ape;
    misaligned_ape.ape = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_ape), kConstraints).isSuccess());
    ProblemPorts misaligned_cu_seqlens;
    misaligned_cu_seqlens.cu_seqlens = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_cu_seqlens), kConstraints).isSuccess());
    ProblemPorts misaligned_cu_seqlens_comp;
    misaligned_cu_seqlens_comp.cu_seqlens_comp = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_cu_seqlens_comp), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_compressed;
    misaligned_grad_compressed.grad_compressed = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_compressed), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_kv;
    misaligned_grad_kv.grad_kv = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_kv), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_score;
    misaligned_grad_score.grad_score = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_score), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_ape;
    misaligned_grad_ape.grad_ape = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_ape), kConstraints).isSuccess());
    ProblemPorts misaligned_ratio;
    misaligned_ratio.ratio = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_head_dim;
    misaligned_head_dim.head_dim = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_head_dim), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_coff;
    misaligned_coff.coff = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_coff), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16CSACompressorBwdTest, ChecksRequiredAndOptionalAddresses) {
    Bf16CSACompressorBwd primitive;
    ProblemPorts null_kv;
    null_kv.kv = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_kv), kConstraints).isSuccess());
    ProblemPorts null_score;
    null_score.score = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_score), kConstraints).isSuccess());
    ProblemPorts null_ape;
    null_ape.ape = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_ape), kConstraints).isSuccess());
    ProblemPorts null_cu_seqlens;
    null_cu_seqlens.cu_seqlens = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_cu_seqlens), kConstraints).isSuccess());
    ProblemPorts null_cu_seqlens_comp;
    null_cu_seqlens_comp.cu_seqlens_comp = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_cu_seqlens_comp), kConstraints).isSuccess());
    ProblemPorts null_grad_compressed;
    null_grad_compressed.grad_compressed = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_compressed), kConstraints).isSuccess());
    ProblemPorts null_grad_kv;
    null_grad_kv.grad_kv = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_kv), kConstraints).isSuccess());
    ProblemPorts null_grad_score;
    null_grad_score.grad_score = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_score), kConstraints).isSuccess());
    ProblemPorts null_grad_ape;
    null_grad_ape.grad_ape = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_ape), kConstraints).isSuccess());
    ProblemPorts null_ratio;
    null_ratio.ratio = nullptr;
    try {
        makeProblem(null_ratio);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_head_dim;
    null_head_dim.head_dim = nullptr;
    try {
        makeProblem(null_head_dim);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_coff;
    null_coff.coff = nullptr;
    try {
        makeProblem(null_coff);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16CSACompressorBwdTest, ClonesIntoDistinctConfigurableCopies) {
    Bf16CSACompressorBwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "CSACompressorBwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<CSACompressorBwdProblem>&>(*first).configure(makeProblem(ports));
    alignas(16) unsigned char other_memory[16]{};
    {
        ProblemPorts other;
        other.kv = other_memory;
        static_cast<Primitive<CSACompressorBwdProblem>&>(*second).configure(makeProblem(other));
    }
    // Workspace checks alone do not prove pointer independence.
    // RebindsAddressesAndOutlivesOpsAndArgs covers execution with separate plans.
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Bf16CSACompressorBwdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original                  = makeProblem(ports).getProblemKey();
        const std::int64_t kv_dims[]         = {2147483648, 32};
        ports.kv_dims                        = kv_dims;
        const std::int64_t score_dims[]      = {2147483648, 32};
        ports.score_dims                     = score_dims;
        const std::int64_t grad_kv_dims[]    = {2147483648, 32};
        ports.grad_kv_dims                   = grad_kv_dims;
        const std::int64_t grad_score_dims[] = {2147483648, 32};
        ports.grad_score_dims                = grad_score_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Bf16CSACompressorBwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16CSACompressorBwdRegressionTest, RejectsUnsupportedComputeEvenWithBf16Storage) {
    alignas(16) unsigned char memory[16]{};
    const Bf16CSACompressorBwd kernel;
    EXPECT_TRUE(kernel.isApplicable(makeProblem(memory, FTRAIN_NUMERIC_TYPE_BF16), Constraints{0}).isSuccess());
    for (const auto type : {FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_INT32}) {
        EXPECT_EQ(kernel.isApplicable(makeProblem(memory, type), Constraints{0}).getStatus(),
                  FTRAIN_STATUS_UNSUPPORTED);
    }
}

TEST(CSACompressorBwdProblemTest, PreservesComputeTypeInTheSelectionKey) {
    alignas(16) unsigned char memory[16]{};
    const auto bf16 = makeProblem(memory, FTRAIN_NUMERIC_TYPE_BF16);
    const auto fp32 = makeProblem(memory, FTRAIN_NUMERIC_TYPE_FP32);
    EXPECT_EQ(bf16.getComputeType(), FTRAIN_NUMERIC_TYPE_BF16);
    EXPECT_EQ(fp32.getComputeType(), FTRAIN_NUMERIC_TYPE_FP32);
    EXPECT_NE(bf16.getProblemKey(), fp32.getProblemKey());
}

}  // namespace ftrain
