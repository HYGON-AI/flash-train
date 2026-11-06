// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <memory>

#include <set>

#include <string>

#include "flash_train/family/dense_attn_score/family.hpp"

#include "flash_train/family/dense_attn_score/finder.hpp"

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/dense_attn_score/problem.hpp"
#include "flash_train/primitive/hygon/dense_attn_score/bf16_dense_attn_score.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kQDims[4]     = {1, 4, 2, 32};
constexpr std::int64_t kKDims[4]     = {1, 4, 1, 32};
constexpr std::int64_t kLseDims[3]   = {1, 4, 2};
constexpr std::int64_t kOutDims[3]   = {1, 4, 4};
constexpr std::int64_t kDenomDims[2] = {1, 4};

struct ProblemPorts {
    void* q                              = kDefaultQMemory;
    const std::int64_t* q_dims           = kQDims;
    FTrainNumericType q_type             = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType q_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* k                              = kDefaultKMemory;
    const std::int64_t* k_dims           = kKDims;
    FTrainNumericType k_type             = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType k_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* lse                            = kDefaultLseMemory;
    const std::int64_t* lse_dims         = kLseDims;
    FTrainNumericType lse_type           = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType lse_index            = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* out                            = kDefaultOutMemory;
    const std::int64_t* out_dims         = kOutDims;
    FTrainNumericType out_type           = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType out_index            = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* denom                          = kDefaultDenomMemory;
    const std::int64_t* denom_dims       = kDenomDims;
    FTrainNumericType denom_type         = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType denom_index          = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* ratio                          = &kDefaultRatio;
    FTrainNumericType ratio_type         = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType ratio_index          = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* softmax_scale                  = &kDefaultSoftmaxScale;
    FTrainNumericType softmax_scale_type = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType softmax_scale_index  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type       = FTRAIN_NUMERIC_TYPE_BF16;
    alignas(16) static unsigned char kDefaultQMemory[16];
    alignas(16) static unsigned char kDefaultKMemory[16];
    alignas(16) static unsigned char kDefaultLseMemory[16];
    alignas(16) static unsigned char kDefaultOutMemory[16];
    alignas(16) static unsigned char kDefaultDenomMemory[16];
    static std::int64_t kDefaultRatio;
    static float kDefaultSoftmaxScale;
};

alignas(16) unsigned char ProblemPorts::kDefaultQMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultKMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultLseMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultOutMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDenomMemory[16];

std::int64_t ProblemPorts::kDefaultRatio = 1;

float ProblemPorts::kDefaultSoftmaxScale = 0.25F;

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

DenseAttnScoreProblem makeProblem(const ProblemPorts& ports) {
    return DenseAttnScoreProblem{
        makeDeviceTensor(ports.q, ports.q_dims, 4, ports.q_type, ports.q_index),
        makeDeviceTensor(ports.k, ports.k_dims, 4, ports.k_type, ports.k_index),
        makeDeviceTensor(ports.lse, ports.lse_dims, 3, ports.lse_type, ports.lse_index),
        makeDeviceTensor(ports.out, ports.out_dims, 3, ports.out_type, ports.out_index),
        makeDeviceTensor(ports.denom, ports.denom_dims, 2, ports.denom_type, ports.denom_index),
        makeHostScalar(ports.ratio, ports.ratio_type, ports.ratio_index),
        makeHostScalar(ports.softmax_scale, ports.softmax_scale_type, ports.softmax_scale_index),
        DenseAttnScoreAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Bf16DenseAttnScoreTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_q[16];
    rebound.q = other_q;
    alignas(16) static unsigned char other_k[16];
    rebound.k = other_k;
    alignas(16) static unsigned char other_lse[16];
    rebound.lse = other_lse;
    alignas(16) static unsigned char other_out[16];
    rebound.out = other_out;
    alignas(16) static unsigned char other_denom[16];
    rebound.denom             = other_denom;
    std::int64_t other_ratio  = 1;
    rebound.ratio             = &other_ratio;
    float other_softmax_scale = 0.25F;
    rebound.softmax_scale     = &other_softmax_scale;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Bf16DenseAttnScoreTest, AcceptsContinuousAlignedOperands) {
    Bf16DenseAttnScore primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Bf16DenseAttnScore");
}

TEST(Bf16DenseAttnScoreTest, RejectsUnsupportedComputeAndElementTypes) {
    Bf16DenseAttnScore primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    for (const auto& candidate : DenseAttnScoreFamily::makeRecords()) {
        SCOPED_TRACE(candidate->getName());
        EXPECT_FALSE(candidate->isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    }
    ProblemPorts unsupported_q;
    unsupported_q.q_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_q), kConstraints).isSuccess());
    ProblemPorts unsupported_k;
    unsupported_k.k_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_k), kConstraints).isSuccess());
    ProblemPorts unsupported_lse;
    unsupported_lse.lse_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_lse), kConstraints).isSuccess());
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
    ProblemPorts unsupported_softmax_scale;
    unsupported_softmax_scale.softmax_scale_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_softmax_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
}

TEST(Bf16DenseAttnScoreTest, RejectsNonContinuousIndexTypes) {
    Bf16DenseAttnScore primitive;
    ProblemPorts strided_q;
    strided_q.q_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_q), kConstraints).isSuccess());
    ProblemPorts strided_k;
    strided_k.k_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_k), kConstraints).isSuccess());
    ProblemPorts strided_lse;
    strided_lse.lse_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_lse), kConstraints).isSuccess());
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
    ProblemPorts strided_softmax_scale;
    strided_softmax_scale.softmax_scale_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_softmax_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16DenseAttnScoreTest, RejectsMisalignedAddresses) {
    Bf16DenseAttnScore primitive;
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
    ProblemPorts misaligned_softmax_scale;
    misaligned_softmax_scale.softmax_scale = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_softmax_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16DenseAttnScoreTest, ChecksRequiredAndOptionalAddresses) {
    Bf16DenseAttnScore primitive;
    ProblemPorts null_q;
    null_q.q = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_q), kConstraints).isSuccess());
    ProblemPorts null_k;
    null_k.k = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_k), kConstraints).isSuccess());
    ProblemPorts null_lse;
    null_lse.lse = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_lse), kConstraints).isSuccess());
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
    ProblemPorts null_softmax_scale;
    null_softmax_scale.softmax_scale = nullptr;
    try {
        makeProblem(null_softmax_scale);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16DenseAttnScoreTest, ClonesIntoDistinctConfigurableCopies) {
    Bf16DenseAttnScore prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Bf16DenseAttnScore");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<DenseAttnScoreProblem>&>(*first).configure(makeProblem(ports));
    alignas(16) unsigned char other_memory[16]{};
    {
        ProblemPorts other;
        other.q = other_memory;
        static_cast<Primitive<DenseAttnScoreProblem>&>(*second).configure(makeProblem(other));
    }
    // Workspace checks alone do not prove pointer independence.
    // RebindsAddressesAndOutlivesOpsAndArgs covers execution with separate plans.
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Bf16DenseAttnScoreTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original         = makeProblem(ports).getProblemKey();
        const std::int64_t q_dims[] = {1, 4, 2, 33};
        ports.q_dims                = q_dims;
        const std::int64_t k_dims[] = {1, 4, 1, 33};
        ports.k_dims                = k_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Bf16DenseAttnScore primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16DenseAttnScoreTest, RejectsTooManyHeads) {
    {
        ProblemPorts ports;

        const std::int64_t q_dims[] = {ports.q_dims[0], ports.q_dims[1], 129, ports.q_dims[3]};
        ports.q_dims                = q_dims;

        const std::int64_t lse_dims[] = {ports.lse_dims[0], ports.lse_dims[1], 129};
        ports.lse_dims                = lse_dims;

        auto problem = makeProblem(ports);
        Bf16DenseAttnScore primitive;

        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16DenseAttnScoreTest, RejectsMissingBf16Instantiation) {
    {
        ProblemPorts ports;

        const std::int64_t q_dims[] = {ports.q_dims[0], ports.q_dims[1], ports.q_dims[2], 160};
        ports.q_dims                = q_dims;

        const std::int64_t k_dims[] = {ports.k_dims[0], ports.k_dims[1], ports.k_dims[2], 160};
        ports.k_dims                = k_dims;

        auto problem = makeProblem(ports);
        Bf16DenseAttnScore primitive;

        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16DenseAttnScoreTest, RejectsExcessiveLaunchSharedMemory) {
    {
        ProblemPorts ports;

        const std::int64_t k_dims[] = {ports.k_dims[0], 100000, ports.k_dims[2], ports.k_dims[3]};
        ports.k_dims                = k_dims;

        const std::int64_t out_dims[] = {ports.out_dims[0], ports.out_dims[1], 100000};
        ports.out_dims                = out_dims;

        auto problem = makeProblem(ports);
        Bf16DenseAttnScore primitive;

        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Bf16DenseAttnScoreTest, PrecisionSpecificWidthCoverage) {
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
        Bf16DenseAttnScoreLargeD primitive;

        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_SUCCESS)
                << primitive.getName();
        }
    }
}

TEST(Bf16DenseAttnScoreTest, SelectionRejectsUnsupportedHeadDimensions) {
    const auto records = DenseAttnScoreFamily::makeRecords();
    for (std::int64_t width : {144, 160, 192, 240, 272, 432, 464, 528, 560}) {
        const std::int64_t q[] = {1, 4, 2, width}, k[] = {1, 4, 1, width};
        ProblemPorts ports;
        ports.q_dims       = q;
        ports.k_dims       = k;
        const auto problem = makeProblem(ports);
        for (const auto& record : records) {
            if (std::string(record->getName()).find("Bf16") == 0) {
                EXPECT_EQ(record->isApplicable(problem, kConstraints).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                    << record->getName() << " D=" << width;
            }
        }
    }
}

TEST(Bf16DenseAttnScoreTest, ChecksTheSelectedAlgorithmsSharedMemory) {
    // A full score row exceeds LDS; fixed-size ring/multi-query tiles still fit.
    EXPECT_FALSE(bf16DenseAttnScoreLaunchFits<Bf16DenseAttnScore>(32, 512, 32768));
    EXPECT_TRUE(bf16DenseAttnScoreLaunchFits<Bf16DenseAttnScoreRingTeacher>(32, 512, 32768));
    // With 16 heads the large-D tile exceeds the staged tile's footprint.
    // 12288 keys: staged uses 63232 bytes, large-D uses 67584 bytes.
    EXPECT_TRUE(bf16DenseAttnScoreLaunchFits<Bf16DenseAttnScore>(16, 128, 12288));
    EXPECT_FALSE(bf16DenseAttnScoreLaunchFits<Bf16DenseAttnScoreLargeD>(16, 128, 12288));
    EXPECT_TRUE(bf16DenseAttnScoreLaunchFits<Bf16DenseAttnScoreHoisted>(64, 128, 13312));
}

TEST(Bf16DenseAttnScoreTest, RejectsElementAlignedButMisalignedVectorOperand) {
    Bf16DenseAttnScore primitive;
    ProblemPorts ports;
    const auto aligned = makeProblem(ports);
    ASSERT_TRUE(primitive.isApplicable(aligned, kConstraints).isSuccess());
    void* const original = ports.q;
    for (const int offset : {2, 4}) {
        ports.q            = static_cast<unsigned char*>(original) + offset;
        const auto shifted = makeProblem(ports);
        EXPECT_NE(aligned.getProblemKey(), shifted.getProblemKey());
        EXPECT_EQ(FTRAIN_STATUS_UNSUPPORTED, primitive.isApplicable(shifted, kConstraints).getStatus());
    }
}

TEST(Bf16DenseAttnScoreTest, FinderReturnsOnlyTheRequestedPrecisionChain) {
    const auto problem    = makeProblem(ProblemPorts{});
    const auto candidates = DenseAttnScoreRegistrationOrderFinder::findCandidates(problem, Constraints{0});
    ASSERT_FALSE(candidates.empty());
    const std::set<std::string> names(candidates.begin(), candidates.end());
    EXPECT_EQ(names.size(), candidates.size());
    for (const auto& name : candidates) EXPECT_EQ(name.find("Bf16DenseAttnScore"), 0U) << name;
    for (const auto& record : DenseAttnScoreFamily::makeRecords()) {
        const std::string name = record->getName();
        EXPECT_EQ(names.count(name), name.find("Bf16DenseAttnScore") == 0 ? 1U : 0U) << name;
    }
}
TEST(Bf16DenseAttnScoreTest, FinderPrefersMeasuredBpermuteShapesOnly) {
    for (std::int64_t tokens : {1, 63, 64, 65, 255, 256, 257, 4096}) {
        for (std::int64_t heads : {32, 64, 128}) {
            for (std::int64_t width : {512, 576}) {
                const std::int64_t q[] = {1, tokens, heads, width}, k[] = {1, 129, 1, width};
                const std::int64_t lse[] = {1, tokens, heads}, out[] = {1, tokens, 129}, denom[] = {1, tokens};
                ProblemPorts ports;
                ports.q_dims     = q;
                ports.k_dims     = k;
                ports.lse_dims   = lse;
                ports.out_dims   = out;
                ports.denom_dims = denom;
                const auto candidates =
                    DenseAttnScoreRegistrationOrderFinder::findCandidates(makeProblem(ports), kConstraints);
                ASSERT_FALSE(candidates.empty());
                EXPECT_EQ(candidates.front(), heads == 64 && width == 512
                                                  ? "Bf16DenseAttnScoreRingTeacherBpermute"
                                                  : "Bf16DenseAttnScoreRingTeacher");
                EXPECT_EQ(std::set<std::string>(candidates.begin(), candidates.end()).size(), candidates.size());
                EXPECT_EQ(candidates.size(), 6U);
            }
        }
    }
}

TEST(Bf16DenseAttnScoreTest, RingHeadBindingMatchesKernelCoverage) {
    for (const std::int64_t heads : {0, 1, 8, 16, 32, 64, 128}) {
        SCOPED_TRACE(heads);
        const std::int64_t q[] = {1, 4, heads, 512}, k[] = {1, 4, 1, 512}, lse[] = {1, 4, heads};
        ProblemPorts ports;
        ports.q_dims       = q;
        ports.k_dims       = k;
        ports.lse_dims     = lse;
        const auto problem = makeProblem(ports);
        Bf16DenseAttnScoreRingTeacher primitive;
        const bool supported = heads == 16 || heads == 32 || heads == 64 || heads == 128;
        EXPECT_EQ(primitive.isApplicable(problem, kConstraints).isSuccess(), supported);
        if (supported) {
            EXPECT_NO_THROW(primitive.configure(problem));
        } else {
            EXPECT_THROW(primitive.configure(problem), Exception);
        }
    }
}

}  // namespace ftrain
