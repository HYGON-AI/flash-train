// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/compactify/problem.hpp"
#include "flash_train/primitive/hygon/compactify/int32_compactify.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kIndicesDims[2] = {2, 4};
constexpr std::int64_t kOutputDims[2] = {2, 4};
constexpr std::int64_t kTopkLengthDims[1] = {2};

struct ProblemPorts {
    void* indices                        = kDefaultIndicesMemory;
    const std::int64_t* indices_dims     = kIndicesDims;
    FTrainNumericType indices_type       = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType indices_index        = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* output                         = kDefaultOutputMemory;
    const std::int64_t* output_dims      = kOutputDims;
    FTrainNumericType output_type        = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType output_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* topk_length                    = kDefaultTopkLengthMemory;
    const std::int64_t* topk_length_dims = kTopkLengthDims;
    FTrainNumericType topk_length_type   = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType topk_length_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    alignas(16) static unsigned char kDefaultIndicesMemory[16];
    alignas(16) static unsigned char kDefaultOutputMemory[16];
    alignas(16) static unsigned char kDefaultTopkLengthMemory[16];
};

alignas(16) unsigned char ProblemPorts::kDefaultIndicesMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultOutputMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultTopkLengthMemory[16];

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

CompactifyProblem makeProblem(const ProblemPorts& ports) {
    return CompactifyProblem{
        makeDeviceTensor(ports.indices, ports.indices_dims, 2, ports.indices_type, ports.indices_index),
        makeDeviceTensor(ports.output, ports.output_dims, 2, ports.output_type, ports.output_index),
        makeDeviceTensor(ports.topk_length, ports.topk_length_dims, 1, ports.topk_length_type, ports.topk_length_index),
        CompactifyAttributes{}};
}

const Constraints kConstraints{0};

Tensor makeTensor(void* memory, const std::int64_t* dims, std::uint8_t rank, FTrainNumericType numeric_type) {
    return Tensor{
        StorageView{FTrainStorageView{memory, dims, nullptr, rank, numeric_type, FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
}

}  // namespace

TEST(Int32CompactifyTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_indices[16];
    rebound.indices = other_indices;
    alignas(16) static unsigned char other_output[16];
    rebound.output = other_output;
    alignas(16) static unsigned char other_topk_length[16];
    rebound.topk_length = other_topk_length;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Int32CompactifyTest, AcceptsContinuousAlignedOperands) {
    Int32Compactify primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Compactify");
}

TEST(Int32CompactifyTest, RejectsUnsupportedComputeAndElementTypes) {
    Int32Compactify primitive;
    ProblemPorts unsupported_indices;
    unsupported_indices.indices_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_indices), kConstraints).isSuccess());
    ProblemPorts unsupported_output;
    unsupported_output.output_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_output), kConstraints).isSuccess());
    ProblemPorts unsupported_topk_length;
    unsupported_topk_length.topk_length_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_topk_length), kConstraints).isSuccess());
}

TEST(Int32CompactifyTest, RejectsNonContinuousIndexTypes) {
    Int32Compactify primitive;
    ProblemPorts strided_indices;
    strided_indices.indices_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_indices), kConstraints).isSuccess());
    ProblemPorts strided_output;
    strided_output.output_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_output), kConstraints).isSuccess());
    ProblemPorts strided_topk_length;
    strided_topk_length.topk_length_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_topk_length), kConstraints).isSuccess());
}

TEST(Int32CompactifyTest, RejectsMisalignedAddresses) {
    Int32Compactify primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_indices;
    misaligned_indices.indices = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_indices), kConstraints).isSuccess());
    ProblemPorts misaligned_output;
    misaligned_output.output = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_output), kConstraints).isSuccess());
    ProblemPorts misaligned_topk_length;
    misaligned_topk_length.topk_length = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_topk_length), kConstraints).isSuccess());
}

TEST(Int32CompactifyTest, ChecksRequiredAndOptionalAddresses) {
    Int32Compactify primitive;
    ProblemPorts null_indices;
    null_indices.indices = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_indices), kConstraints).isSuccess());
    ProblemPorts null_output;
    null_output.output = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_output), kConstraints).isSuccess());
    ProblemPorts null_topk_length;
    null_topk_length.topk_length = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_topk_length), kConstraints).isSuccess());
}

TEST(Int32CompactifyTest, ClonesIntoIndependentConfigurableCopies) {
    Int32Compactify prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Compactify");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<CompactifyProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<CompactifyProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Int32CompactifyTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original                   = makeProblem(ports).getProblemKey();
        const std::int64_t indices_dims[]     = {2147483648, 4};
        ports.indices_dims                    = indices_dims;
        const std::int64_t output_dims[]      = {2147483648, 4};
        ports.output_dims                     = output_dims;
        const std::int64_t topk_length_dims[] = {2147483648};
        ports.topk_length_dims                = topk_length_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Int32Compactify primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Int32CompactifyRegressionTest, RejectsNonIntegerIndicesDuringSelection) {
    alignas(8) static unsigned char indices[64];
    alignas(8) static unsigned char output[64];
    alignas(8) static unsigned char lengths[64];
    const std::int64_t index_dims[2]  = {4, 4};
    const std::int64_t output_dims[2] = {4, 4};
    const std::int64_t length_dims[1] = {4};
    const CompactifyProblem problem(makeTensor(indices, index_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                                    makeTensor(output, output_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                                    makeTensor(lengths, length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                                    CompactifyAttributes{});
    const Int32Compactify kernel;
    EXPECT_EQ(kernel.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST(Int32CompactifyRegressionTest, RejectsZeroGridOrZeroBlockEvenWithNonNullAddresses) {
    alignas(8) int memory[16]{};
    for (const auto rows : {0, 4}) {
        for (const auto width : {0, 4}) {
            if (rows != 0 && width != 0) { continue; }
            const std::int64_t shape[] = {rows, width}, lengths[] = {rows};
            const auto tensor = [&](const std::int64_t* dims, std::uint8_t rank) {
                return Tensor{StorageView{FTrainStorageView{memory, dims, nullptr, rank, FTRAIN_NUMERIC_TYPE_INT32,
                                                            FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
            };
            const CompactifyProblem problem(tensor(shape, 2), tensor(shape, 2), tensor(lengths, 1),
                                            CompactifyAttributes{});
            EXPECT_EQ(Int32Compactify{}.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
        }
    }
}

TEST(CompactifyProblemTest, RejectsAMismatchedLengthVector) {
    alignas(8) static unsigned char indices[64];
    alignas(8) static unsigned char output[64];
    alignas(8) static unsigned char lengths[64];
    const std::int64_t index_dims[2]  = {4, 4};
    const std::int64_t output_dims[2] = {4, 4};
    const std::int64_t length_dims[1] = {3};
    EXPECT_THROW(
        CompactifyProblem(makeTensor(indices, index_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                          makeTensor(output, output_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                          makeTensor(lengths, length_dims, 1, FTRAIN_NUMERIC_TYPE_INT32), CompactifyAttributes{}),
        Exception);
}

}  // namespace ftrain
