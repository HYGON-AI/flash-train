// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/indexer_topk/problem.hpp"
#include "flash_train/primitive/hygon/indexer_topk/fp32_indexer_topk.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kInputValuesDims[2] = {4, 4};
constexpr std::int64_t kSeqLensDims[1]     = {4};
constexpr std::int64_t kOutIndicesDims[2]  = {4, 2};
constexpr std::int64_t kOutValuesDims[2]   = {4, 2};

struct ProblemPorts {
    void* input_values                    = kDefaultInputValuesMemory;
    const std::int64_t* input_values_dims = kInputValuesDims;
    FTrainNumericType input_values_type   = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType input_values_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* seq_lens                        = kDefaultSeqLensMemory;
    const std::int64_t* seq_lens_dims     = kSeqLensDims;
    FTrainNumericType seq_lens_type       = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType seq_lens_index        = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* out_indices                     = kDefaultOutIndicesMemory;
    const std::int64_t* out_indices_dims  = kOutIndicesDims;
    FTrainNumericType out_indices_type    = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType out_indices_index     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* out_values                      = kDefaultOutValuesMemory;
    const std::int64_t* out_values_dims   = kOutValuesDims;
    FTrainNumericType out_values_type     = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType out_values_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* top_k                           = &kDefaultTopK;
    FTrainNumericType top_k_type          = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType top_k_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* next_n                          = &kDefaultNextN;
    FTrainNumericType next_n_type         = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType next_n_index          = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* order_values                    = &kDefaultOrderValues;
    FTrainNumericType order_values_type   = FTRAIN_NUMERIC_TYPE_UINT8;
    FTrainIndexType order_values_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    alignas(16) static unsigned char kDefaultInputValuesMemory[16];
    alignas(16) static unsigned char kDefaultSeqLensMemory[16];
    alignas(16) static unsigned char kDefaultOutIndicesMemory[16];
    alignas(16) static unsigned char kDefaultOutValuesMemory[16];
    static std::int64_t kDefaultTopK;
    static std::int64_t kDefaultNextN;
    static std::uint8_t kDefaultOrderValues;
};

alignas(16) unsigned char ProblemPorts::kDefaultInputValuesMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultSeqLensMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultOutIndicesMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultOutValuesMemory[16];

std::int64_t ProblemPorts::kDefaultTopK = 2;

std::int64_t ProblemPorts::kDefaultNextN = 1;

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

IndexerTopKProblem makeProblem(const ProblemPorts& ports) {
    return IndexerTopKProblem{
        makeDeviceTensor(ports.input_values, ports.input_values_dims, 2, ports.input_values_type,
                         ports.input_values_index),
        makeDeviceTensor(ports.seq_lens, ports.seq_lens_dims, 1, ports.seq_lens_type, ports.seq_lens_index),
        makeDeviceTensor(ports.out_indices, ports.out_indices_dims, 2, ports.out_indices_type, ports.out_indices_index),
        makeDeviceTensor(ports.out_values, ports.out_values_dims, 2, ports.out_values_type, ports.out_values_index),
        makeHostScalar(ports.top_k, ports.top_k_type, ports.top_k_index),
        makeHostScalar(ports.next_n, ports.next_n_type, ports.next_n_index),
        makeHostScalar(ports.order_values, ports.order_values_type, ports.order_values_index),
        IndexerTopKAttributes{}};
}

const Constraints kConstraints{0};

Tensor makeTensor(void* memory, const std::int64_t* dims, std::uint8_t rank, FTrainNumericType numeric_type) {
    return Tensor{
        StorageView{FTrainStorageView{memory, dims, nullptr, rank, numeric_type, FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
}

Tensor hostInteger(const std::int64_t& value) {
    return Tensor{StorageView{FTrainStorageView{const_cast<std::int64_t*>(&value), nullptr, nullptr, 0,
                                                FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true}}};
}

Tensor hostBoolean(const std::uint8_t& value) {
    return Tensor{StorageView{FTrainStorageView{const_cast<std::uint8_t*>(&value), nullptr, nullptr, 0,
                                                FTRAIN_NUMERIC_TYPE_UINT8, FTRAIN_INDEX_TYPE_CONTINUOUS, true}}};
}

struct TopKPorts {
    alignas(8) unsigned char input[16384]{};
    alignas(8) unsigned char seq_lens[64]{};
    alignas(8) unsigned char out_indices[4096]{};
    alignas(8) unsigned char out_values[4096]{};
    static constexpr std::int64_t kInputDims[2]      = {8, 1024};
    static constexpr std::int64_t kSeqLensDims[1]    = {8};
    static constexpr std::int64_t kOutIndicesDims[2] = {8, 16};
    std::int64_t top_k                               = 16;
    std::int64_t next_n                              = 1;
    bool order_values                                = true;
    std::uint8_t order_values_storage                = 1;
    const void* out_values_addr                      = nullptr;
};

IndexerTopKProblem makeTopKProblem(TopKPorts& ports) {
    Tensor out_values = makeTensor(ports.order_values ? ports.out_values : const_cast<void*>(ports.out_values_addr),
                                   TopKPorts::kOutIndicesDims, 2, FTRAIN_NUMERIC_TYPE_FP32);
    if (!ports.order_values) {
        // An unset out_values still carries the shape the Problem validates.
        out_values =
            Tensor{StorageView{FTrainStorageView{nullptr, TopKPorts::kOutIndicesDims, nullptr, 2,
                                                 FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
    }
    return IndexerTopKProblem{makeTensor(ports.input, TopKPorts::kInputDims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                              makeTensor(ports.seq_lens, TopKPorts::kSeqLensDims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                              makeTensor(ports.out_indices, TopKPorts::kOutIndicesDims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                              std::move(out_values),
                              hostInteger(ports.top_k),
                              hostInteger(ports.next_n),
                              hostBoolean(ports.order_values_storage),
                              IndexerTopKAttributes{}};
}

}  // namespace

TEST(Fp32IndexerTopKTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_input_values[16];
    rebound.input_values = other_input_values;
    alignas(16) static unsigned char other_seq_lens[16];
    rebound.seq_lens = other_seq_lens;
    alignas(16) static unsigned char other_out_indices[16];
    rebound.out_indices = other_out_indices;
    alignas(16) static unsigned char other_out_values[16];
    rebound.out_values              = other_out_values;
    std::int64_t other_top_k        = 2;
    rebound.top_k                   = &other_top_k;
    std::int64_t other_next_n       = 1;
    rebound.next_n                  = &other_next_n;
    std::uint8_t other_order_values = 1;
    rebound.order_values            = &other_order_values;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp32IndexerTopKTest, AcceptsContinuousAlignedOperands) {
    Fp32IndexerTopK primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "IndexerTopK");
}

TEST(Fp32IndexerTopKTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp32IndexerTopK primitive;
    ProblemPorts unsupported_input_values;
    unsupported_input_values.input_values_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_input_values), kConstraints).isSuccess());
    ProblemPorts unsupported_seq_lens;
    unsupported_seq_lens.seq_lens_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_seq_lens), kConstraints).isSuccess());
    ProblemPorts unsupported_out_indices;
    unsupported_out_indices.out_indices_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_out_indices), kConstraints).isSuccess());
    ProblemPorts unsupported_out_values;
    unsupported_out_values.out_values_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_out_values), kConstraints).isSuccess());
    ProblemPorts unsupported_top_k;
    unsupported_top_k.top_k_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_top_k), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_next_n;
    unsupported_next_n.next_n_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_next_n), kConstraints).isSuccess());
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

TEST(Fp32IndexerTopKTest, RejectsNonContinuousIndexTypes) {
    Fp32IndexerTopK primitive;
    ProblemPorts strided_input_values;
    strided_input_values.input_values_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_input_values), kConstraints).isSuccess());
    ProblemPorts strided_seq_lens;
    strided_seq_lens.seq_lens_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_seq_lens), kConstraints).isSuccess());
    ProblemPorts strided_out_indices;
    strided_out_indices.out_indices_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_out_indices), kConstraints).isSuccess());
    ProblemPorts strided_out_values;
    strided_out_values.out_values_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_out_values), kConstraints).isSuccess());
    ProblemPorts strided_top_k;
    strided_top_k.top_k_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_top_k), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_next_n;
    strided_next_n.next_n_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_next_n), kConstraints).isSuccess());
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

TEST(Fp32IndexerTopKTest, RejectsMisalignedAddresses) {
    Fp32IndexerTopK primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_input_values;
    misaligned_input_values.input_values = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_input_values), kConstraints).isSuccess());
    ProblemPorts misaligned_seq_lens;
    misaligned_seq_lens.seq_lens = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_seq_lens), kConstraints).isSuccess());
    ProblemPorts misaligned_out_indices;
    misaligned_out_indices.out_indices = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_out_indices), kConstraints).isSuccess());
    ProblemPorts misaligned_out_values;
    misaligned_out_values.out_values = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_out_values), kConstraints).isSuccess());
    ProblemPorts misaligned_top_k;
    misaligned_top_k.top_k = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_top_k), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_next_n;
    misaligned_next_n.next_n = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_next_n), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32IndexerTopKTest, ChecksRequiredAndOptionalAddresses) {
    Fp32IndexerTopK primitive;
    ProblemPorts null_input_values;
    null_input_values.input_values = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_input_values), kConstraints).isSuccess());
    ProblemPorts null_seq_lens;
    null_seq_lens.seq_lens = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_seq_lens), kConstraints).isSuccess());
    ProblemPorts null_out_indices;
    null_out_indices.out_indices = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_out_indices), kConstraints).isSuccess());
    ProblemPorts null_out_values;
    null_out_values.out_values = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_out_values), kConstraints).isSuccess());
    // The values sink is optional only when value ordering is disabled.
    std::uint8_t omit_values     = 0;
    null_out_values.order_values = &omit_values;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(null_out_values), kConstraints).isSuccess());
    ProblemPorts null_top_k;
    null_top_k.top_k = nullptr;
    try {
        makeProblem(null_top_k);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_next_n;
    null_next_n.next_n = nullptr;
    try {
        makeProblem(null_next_n);
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

TEST(Fp32IndexerTopKTest, ClonesIntoDistinctConfigurableCopies) {
    Fp32IndexerTopK prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "IndexerTopK");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<IndexerTopKProblem>&>(*first).configure(makeProblem(ports));
    alignas(16) unsigned char other_memory[16]{};
    {
        ProblemPorts other;
        other.input_values = other_memory;
        static_cast<Primitive<IndexerTopKProblem>&>(*second).configure(makeProblem(other));
    }
    // Workspace checks alone do not prove pointer independence.
    // RebindsAddressesAndOutlivesOpsAndArgs covers execution with separate plans.
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp32IndexerTopKTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original                    = makeProblem(ports).getProblemKey();
        const std::int64_t input_values_dims[] = {4, 2147483648};
        ports.input_values_dims                = input_values_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp32IndexerTopK primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp32IndexerTopKRegressionTest, AcceptsAWellFormedCall) {
    TopKPorts ports;
    const IndexerTopKProblem problem = makeTopKProblem(ports);
    const Fp32IndexerTopK kernel;
    const Result result = kernel.isApplicable(problem, kConstraints);
    EXPECT_TRUE(result.isSuccess()) << result.getMessage();
}

TEST(Fp32IndexerTopKRegressionTest, RequiresTheValuesSinkOnlyWhenItIsAskedFor) {
    const Fp32IndexerTopK kernel;

    TopKPorts skipping;
    skipping.order_values            = false;
    skipping.order_values_storage    = 0;
    const IndexerTopKProblem skipped = makeTopKProblem(skipping);
    EXPECT_TRUE(kernel.isApplicable(skipped, kConstraints).isSuccess());

    TopKPorts asking;
    asking.order_values    = true;
    asking.out_values_addr = nullptr;
    asking.out_values[0]   = 0;
    Tensor absent_values{StorageView{FTrainStorageView{nullptr, TopKPorts::kOutIndicesDims, nullptr, 2,
                                                       FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, false}}};
    IndexerTopKProblem missing_sink{
        makeTensor(asking.input, TopKPorts::kInputDims, 2, FTRAIN_NUMERIC_TYPE_FP32),
        makeTensor(asking.seq_lens, TopKPorts::kSeqLensDims, 1, FTRAIN_NUMERIC_TYPE_INT32),
        makeTensor(asking.out_indices, TopKPorts::kOutIndicesDims, 2, FTRAIN_NUMERIC_TYPE_INT32),
        std::move(absent_values),
        hostInteger(asking.top_k),
        hostInteger(asking.next_n),
        hostBoolean(asking.order_values_storage),
        IndexerTopKAttributes{}};
    const Result result = kernel.isApplicable(missing_sink, kConstraints);
    EXPECT_FALSE(result.isSuccess());
    EXPECT_EQ(result.getStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST(Fp32IndexerTopKRegressionTest, RejectsShapesBeyondTheParallelCoverage) {
    std::int64_t wide_dims[2]           = {8, 32768};
    const std::int64_t seq_lens_dims[1] = {8};
    const std::int64_t out_dims[2]      = {8, 16};
    alignas(8) static unsigned char input[64];
    alignas(8) static unsigned char indices[64];
    alignas(8) static unsigned char values[64];
    static constexpr std::int64_t top_k = 16, next_n = 1;
    static constexpr std::uint8_t order_values = 1;
    IndexerTopKProblem problem{makeTensor(input, wide_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                               makeTensor(indices, seq_lens_dims, 1, FTRAIN_NUMERIC_TYPE_INT32),
                               makeTensor(indices, out_dims, 2, FTRAIN_NUMERIC_TYPE_INT32),
                               makeTensor(values, out_dims, 2, FTRAIN_NUMERIC_TYPE_FP32),
                               hostInteger(top_k),
                               hostInteger(next_n),
                               hostBoolean(order_values),
                               IndexerTopKAttributes{}};
    const Fp32IndexerTopK kernel;
    const Result result = kernel.isApplicable(problem, kConstraints);
    EXPECT_FALSE(result.isSuccess());
    EXPECT_EQ(result.getStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

TEST(IndexerTopKProblemTest, NonzeroUint8BooleanHasTheSameSelectionKey) {
    TopKPorts ports;
    const auto one             = makeTopKProblem(ports);
    ports.order_values_storage = 2;
    const auto two             = makeTopKProblem(ports);
    EXPECT_TRUE(two.getOrderValues());
    EXPECT_EQ(one.getProblemKey(), two.getProblemKey());
}

}  // namespace ftrain
