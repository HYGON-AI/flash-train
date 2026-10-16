// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/indexer_fwd/problem.hpp"
#include "flash_train/primitive/hygon/indexer_fwd/bf16_indexer_fwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kQDims[4] = {1, 4, 2, 32};
constexpr std::int64_t kKDims[4] = {1, 4, 1, 32};
constexpr std::int64_t kWDims[3] = {1, 4, 2};
constexpr std::int64_t kScoresDims[3] = {1, 4, 4};

struct ProblemPorts {
    void* q                         = kDefaultQMemory;
    const std::int64_t* q_dims      = kQDims;
    FTrainNumericType q_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType q_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* k                         = kDefaultKMemory;
    const std::int64_t* k_dims      = kKDims;
    FTrainNumericType k_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType k_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* w                         = kDefaultWMemory;
    const std::int64_t* w_dims      = kWDims;
    FTrainNumericType w_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType w_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* scores                    = kDefaultScoresMemory;
    const std::int64_t* scores_dims = kScoresDims;
    FTrainNumericType scores_type   = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType scores_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* ratio                     = &kDefaultRatio;
    FTrainNumericType ratio_type    = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType ratio_index     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* sm_scale                  = &kDefaultSmScale;
    FTrainNumericType sm_scale_type = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType sm_scale_index  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type  = FTRAIN_NUMERIC_TYPE_BF16;
    alignas(16) static unsigned char kDefaultQMemory[16];
    alignas(16) static unsigned char kDefaultKMemory[16];
    alignas(16) static unsigned char kDefaultWMemory[16];
    alignas(16) static unsigned char kDefaultScoresMemory[16];
    static std::int64_t kDefaultRatio;
    static float kDefaultSmScale;
};

alignas(16) unsigned char ProblemPorts::kDefaultQMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultKMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultWMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultScoresMemory[16];

std::int64_t ProblemPorts::kDefaultRatio = 1;

float ProblemPorts::kDefaultSmScale = 0.25F;

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

IndexerFwdProblem makeProblem(const ProblemPorts& ports) {
    return IndexerFwdProblem{
        makeDeviceTensor(ports.q, ports.q_dims, 4, ports.q_type, ports.q_index),
        makeDeviceTensor(ports.k, ports.k_dims, 4, ports.k_type, ports.k_index),
        makeDeviceTensor(ports.w, ports.w_dims, 3, ports.w_type, ports.w_index),
        makeDeviceTensor(ports.scores, ports.scores_dims, 3, ports.scores_type, ports.scores_index),
        makeHostScalar(ports.ratio, ports.ratio_type, ports.ratio_index),
        makeHostScalar(ports.sm_scale, ports.sm_scale_type, ports.sm_scale_index),
        IndexerFwdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

constexpr std::int64_t kBatch = 2;
constexpr std::int64_t kQueryLen = 4;
constexpr std::int64_t kKeyLen = 8;
constexpr std::int64_t kHeads = 2;
constexpr std::int64_t kHeadDim = 64;

const std::int64_t kBoundaryQDims[4] = {kBatch, kQueryLen, kHeads, kHeadDim};

const std::int64_t kBoundaryKDims[4] = {kBatch, kKeyLen, 1, kHeadDim};

const std::int64_t kBoundaryWDims[3] = {kBatch, kQueryLen, kHeads};

const std::int64_t kBoundaryScoresDims[3] = {kBatch, kQueryLen, kKeyLen};

struct Ports {
    alignas(8) unsigned char q[4096]{};
    alignas(8) unsigned char k[4096]{};
    alignas(8) unsigned char w[4096]{};
    alignas(8) unsigned char scores[4096]{};
    const std::int64_t* q_dims      = kBoundaryQDims;
    const std::int64_t* k_dims      = kBoundaryKDims;
    const std::int64_t* w_dims      = kBoundaryWDims;
    const std::int64_t* scores_dims = kBoundaryScoresDims;
    FTrainNumericType compute_type  = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainNumericType q_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainNumericType k_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainNumericType w_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType q_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    bool q_is_host                  = false;
    std::int64_t ratio              = 4;
    float sm_scale                  = 1.0F;
};

Tensor makeTensor(void* memory, const std::int64_t* dims, std::uint8_t rank, FTrainNumericType numeric_type,
                  FTrainIndexType index_type = FTRAIN_INDEX_TYPE_CONTINUOUS, bool is_host_memory = false) {
    return Tensor{
        StorageView{FTrainStorageView{memory, dims, nullptr, rank, numeric_type, index_type, is_host_memory}}};
}

Tensor hostScalar(const void* memory, FTrainNumericType dtype) {
    return Tensor{StorageView{
        FTrainStorageView{const_cast<void*>(memory), nullptr, nullptr, 0, dtype, FTRAIN_INDEX_TYPE_CONTINUOUS, true}}};
}

IndexerFwdProblem makeProblem(Ports& ports) {
    return IndexerFwdProblem{makeTensor(ports.q, ports.q_dims, 4, ports.q_type, ports.q_index, ports.q_is_host),
                             makeTensor(ports.k, ports.k_dims, 4, ports.k_type),
                             makeTensor(ports.w, ports.w_dims, 3, ports.w_type),
                             makeTensor(ports.scores, ports.scores_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
                             hostScalar(&ports.ratio, FTRAIN_NUMERIC_TYPE_INT64),
                             hostScalar(&ports.sm_scale, FTRAIN_NUMERIC_TYPE_FP32),
                             IndexerFwdAttributes{ports.compute_type}};
}

}  // namespace

TEST(Bf16IndexerFwdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_q[16];
    rebound.q = other_q;
    alignas(16) static unsigned char other_k[16];
    rebound.k = other_k;
    alignas(16) static unsigned char other_w[16];
    rebound.w = other_w;
    alignas(16) static unsigned char other_scores[16];
    rebound.scores           = other_scores;
    std::int64_t other_ratio = 1;
    rebound.ratio            = &other_ratio;
    float other_sm_scale     = 0.25F;
    rebound.sm_scale         = &other_sm_scale;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Bf16IndexerFwdTest, AcceptsContinuousAlignedOperands) {
    Bf16IndexerFwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Bf16IndexerFwd");
}

TEST(Bf16IndexerFwdTest, RejectsUnsupportedComputeAndElementTypes) {
    Bf16IndexerFwd primitive;
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
    ProblemPorts unsupported_scores;
    unsupported_scores.scores_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_scores), kConstraints).isSuccess());
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
}

TEST(Bf16IndexerFwdTest, RejectsNonContinuousIndexTypes) {
    Bf16IndexerFwd primitive;
    ProblemPorts strided_q;
    strided_q.q_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_q), kConstraints).isSuccess());
    ProblemPorts strided_k;
    strided_k.k_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_k), kConstraints).isSuccess());
    ProblemPorts strided_w;
    strided_w.w_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_w), kConstraints).isSuccess());
    ProblemPorts strided_scores;
    strided_scores.scores_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_scores), kConstraints).isSuccess());
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
}

TEST(Bf16IndexerFwdTest, RejectsMisalignedAddresses) {
    Bf16IndexerFwd primitive;
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
    ProblemPorts misaligned_scores;
    misaligned_scores.scores = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_scores), kConstraints).isSuccess());
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
}

TEST(Bf16IndexerFwdTest, ChecksRequiredAndOptionalAddresses) {
    Bf16IndexerFwd primitive;
    ProblemPorts null_q;
    null_q.q = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_q), kConstraints).isSuccess());
    ProblemPorts null_k;
    null_k.k = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_k), kConstraints).isSuccess());
    ProblemPorts null_w;
    null_w.w = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_w), kConstraints).isSuccess());
    ProblemPorts null_scores;
    null_scores.scores = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_scores), kConstraints).isSuccess());
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
}

TEST(Bf16IndexerFwdTest, ClonesIntoIndependentConfigurableCopies) {
    Bf16IndexerFwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Bf16IndexerFwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<IndexerFwdProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<IndexerFwdProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Bf16IndexerFwdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original         = makeProblem(ports).getProblemKey();
        const std::int64_t q_dims[] = {1, 4, 2, 33};
        ports.q_dims                = q_dims;
        const std::int64_t k_dims[] = {1, 4, 1, 33};
        ports.k_dims                = k_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Bf16IndexerFwd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16IndexerFwdTest, RejectsTooManyHeads) {
    {
        ProblemPorts ports;

        const std::int64_t q_dims[] = {ports.q_dims[0], ports.q_dims[1], 129, ports.q_dims[3]};
        ports.q_dims                = q_dims;

        const std::int64_t w_dims[] = {ports.w_dims[0], ports.w_dims[1], 129};
        ports.w_dims                = w_dims;

        auto problem = makeProblem(ports);
        Bf16IndexerFwd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16IndexerFwdTest, RejectsMissingBf16Instantiation) {
    {
        ProblemPorts ports;

        const std::int64_t q_dims[] = {ports.q_dims[0], ports.q_dims[1], ports.q_dims[2], 160};
        ports.q_dims                = q_dims;

        const std::int64_t k_dims[] = {ports.k_dims[0], ports.k_dims[1], ports.k_dims[2], 160};
        ports.k_dims                = k_dims;

        auto problem = makeProblem(ports);
        Bf16IndexerFwd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16IndexerFwdTest, RejectsWidthBeyondLauncherCoverage) {
    {
        ProblemPorts ports;

        const std::int64_t q_dims[] = {ports.q_dims[0], ports.q_dims[1], ports.q_dims[2], 576};
        ports.q_dims                = q_dims;

        const std::int64_t k_dims[] = {ports.k_dims[0], ports.k_dims[1], ports.k_dims[2], 576};
        ports.k_dims                = k_dims;

        auto problem = makeProblem(ports);
        Bf16IndexerFwd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16IndexerFwdTest, RejectsExcessiveLaunchSharedMemory) {
    {
        ProblemPorts ports;

        const std::int64_t k_dims[] = {ports.k_dims[0], 100000, ports.k_dims[2], ports.k_dims[3]};
        ports.k_dims                = k_dims;

        const std::int64_t scores_dims[] = {ports.scores_dims[0], ports.scores_dims[1], 100000};
        ports.scores_dims                = scores_dims;

        auto problem = makeProblem(ports);
        Bf16IndexerFwd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16IndexerFwdTest, PrecisionSpecificWidthCoverage) {
    {
        ProblemPorts ports;

        // BF16 has a D=256 specialization; FP16/FP32 use a runtime width.

        const std::int64_t q_dims[] = {ports.q_dims[0], ports.q_dims[1], ports.q_dims[2],
                                       ports.compute_type == FTRAIN_NUMERIC_TYPE_BF16 ? 256 : 160};
        ports.q_dims                = q_dims;

        const std::int64_t k_dims[] = {ports.k_dims[0], ports.k_dims[1], ports.k_dims[2],
                                       ports.compute_type == FTRAIN_NUMERIC_TYPE_BF16 ? 256 : 160};
        ports.k_dims                = k_dims;

        auto problem = makeProblem(ports);
        Bf16IndexerFwd primitive;
        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_SUCCESS)
                << primitive.getName();
        }
    }
}

TEST(Bf16IndexerFwdRegressionTest, UnsupportedScoreTypeIsASelectionDecision) {
    Ports ports;
    const IndexerFwdProblem problem(
        makeTensor(ports.q, ports.q_dims, 4, ports.q_type), makeTensor(ports.k, ports.k_dims, 4, ports.k_type),
        makeTensor(ports.w, ports.w_dims, 3, ports.w_type),
        makeTensor(ports.scores, ports.scores_dims, 3, FTRAIN_NUMERIC_TYPE_BF16),
        hostScalar(&ports.ratio, FTRAIN_NUMERIC_TYPE_INT64), hostScalar(&ports.sm_scale, FTRAIN_NUMERIC_TYPE_FP32),
        IndexerFwdAttributes{FTRAIN_NUMERIC_TYPE_BF16});
    {
        Bf16IndexerFwd primitive;
        EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    }
}

TEST(Bf16IndexerFwdRegressionTest, RejectsValuesBeyondKernelIntegerLimit) {
    Ports ports;
    ports.ratio            = std::int64_t{1} << 32;
    const auto unsupported = makeProblem(ports);
    {
        Bf16IndexerFwd primitive;
        EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    }
}

TEST(Bf16IndexerFwdRegressionTest, RejectsUnalignedDataAndSupportsIndependentClones) {
    Ports ports;
    Bf16IndexerFwd primitive;
    const auto valid = makeProblem(ports);
    ASSERT_TRUE(primitive.isApplicable(valid, Constraints{0}).isSuccess());
    auto first  = primitive.clone();
    auto second = primitive.clone();
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), primitive.getName());
    const IndexerFwdProblem unaligned(
        makeTensor(ports.q + 1, ports.q_dims, 4, ports.q_type), makeTensor(ports.k, ports.k_dims, 4, ports.k_type),
        makeTensor(ports.w, ports.w_dims, 3, ports.w_type),
        makeTensor(ports.scores, ports.scores_dims, 3, FTRAIN_NUMERIC_TYPE_FP32),
        hostScalar(&ports.ratio, FTRAIN_NUMERIC_TYPE_INT64), hostScalar(&ports.sm_scale, FTRAIN_NUMERIC_TYPE_FP32),
        IndexerFwdAttributes{FTRAIN_NUMERIC_TYPE_BF16});
    EXPECT_EQ(primitive.isApplicable(unaligned, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST(IndexerFwdProblemTest, ExposesTheValidatedShape) {
    Ports ports;
    const IndexerFwdProblem problem = makeProblem(ports);
    EXPECT_EQ(problem.getBatch(), static_cast<std::uint64_t>(kBatch));
    EXPECT_EQ(problem.getQueryLength(), static_cast<std::uint64_t>(kQueryLen));
    EXPECT_EQ(problem.getKeyLength(), static_cast<std::uint64_t>(kKeyLen));
    EXPECT_EQ(problem.getQueryHeads(), static_cast<std::uint64_t>(kHeads));
    EXPECT_EQ(problem.getHeadDim(), static_cast<std::uint64_t>(kHeadDim));
    EXPECT_EQ(problem.getComputeType(), FTRAIN_NUMERIC_TYPE_BF16);
    EXPECT_EQ(problem.getRatio(), 4);
}

TEST(IndexerFwdProblemTest, RejectsAMismatchedKeyLength) {
    Ports ports;
    const std::int64_t wrong_scores[3] = {kBatch, kQueryLen, kKeyLen - 1};
    ports.scores_dims                  = wrong_scores;
    EXPECT_THROW(makeProblem(ports), Exception);
}

TEST(IndexerFwdProblemTest, RejectsAHostOperand) {
    Ports ports;
    ports.q_is_host = true;
    EXPECT_THROW(makeProblem(ports), Exception);
}

TEST(IndexerFwdProblemTest, RejectsAStridedOperand) {
    Ports ports;
    ports.q_index = FTRAIN_INDEX_TYPE_INVALID;
    EXPECT_THROW(makeProblem(ports), Exception);
}

TEST(IndexerFwdProblemTest, ByteOverflowIsRejectedBeforeSelection) {
    Ports ports;
    const std::int64_t large[] = {std::int64_t{1} << 62, 4, 2, 64};
    ports.q_dims               = large;
    EXPECT_THROW(makeProblem(ports), Exception);
}

TEST(IndexerFwdProblemTest, CacheKeyIgnoresExecutionScalarValuesAndTracksStructuralValues) {
    Ports ports;
    const auto original = makeProblem(ports).getProblemKey();
    ports.sm_scale      = 0.75F;
    EXPECT_EQ(original, makeProblem(ports).getProblemKey());
    ports.ratio = 2;
    EXPECT_NE(original, makeProblem(ports).getProblemKey());
    ports.ratio = 0;
    EXPECT_THROW(makeProblem(ports), Exception);
}

TEST(Bf16IndexerFwdTest, LauncherRejectsUnsupportedHeadDimensions) {
    for (int width : {144, 160, 192, 240, 272, 432, 464, 528, 560}) {
        EXPECT_FALSE(launchBf16IndexerFwd(nullptr, nullptr, nullptr, nullptr, nullptr, 1, 4, 4, 2, 1, width, 1, 0.25F))
            << width;
    }
}

}  // namespace ftrain
