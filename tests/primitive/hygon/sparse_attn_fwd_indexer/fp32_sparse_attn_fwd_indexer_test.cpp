// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/sparse_attn_fwd_indexer/problem.hpp"
#include "flash_train/primitive/hygon/sparse_attn_fwd_indexer/fp32_sparse_attn_fwd_indexer.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kQDims[3] = {4, 2, 512};
constexpr std::int64_t kKvDims[2] = {4, 512};
constexpr std::int64_t kAttnSinkDims[1] = {2};
constexpr std::int64_t kTopkIndicesDims[2] = {4, 4};
constexpr std::int64_t kTopkLengthDims[1] = {4};
constexpr std::int64_t kOutDims[3] = {4, 2, 512};
constexpr std::int64_t kMaxLogitsDims[2] = {4, 2};
constexpr std::int64_t kLseDims[2] = {4, 2};
constexpr std::int64_t kLseIndexerDims[2] = {4, 2};

struct ProblemPorts {
    void* q                               = kDefaultQMemory;
    const std::int64_t* q_dims            = kQDims;
    FTrainNumericType q_type              = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType q_index               = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* kv                              = kDefaultKvMemory;
    const std::int64_t* kv_dims           = kKvDims;
    FTrainNumericType kv_type             = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType kv_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* attn_sink                       = kDefaultAttnSinkMemory;
    const std::int64_t* attn_sink_dims    = kAttnSinkDims;
    FTrainNumericType attn_sink_type      = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType attn_sink_index       = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* topk_indices                    = kDefaultTopkIndicesMemory;
    const std::int64_t* topk_indices_dims = kTopkIndicesDims;
    FTrainNumericType topk_indices_type   = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType topk_indices_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* topk_length                     = kDefaultTopkLengthMemory;
    const std::int64_t* topk_length_dims  = kTopkLengthDims;
    FTrainNumericType topk_length_type    = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType topk_length_index     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* out                             = kDefaultOutMemory;
    const std::int64_t* out_dims          = kOutDims;
    FTrainNumericType out_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType out_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* max_logits                      = kDefaultMaxLogitsMemory;
    const std::int64_t* max_logits_dims   = kMaxLogitsDims;
    FTrainNumericType max_logits_type     = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType max_logits_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* lse                             = kDefaultLseMemory;
    const std::int64_t* lse_dims          = kLseDims;
    FTrainNumericType lse_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType lse_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* lse_indexer                     = kDefaultLseIndexerMemory;
    const std::int64_t* lse_indexer_dims  = kLseIndexerDims;
    FTrainNumericType lse_indexer_type    = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType lse_indexer_index     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* softmax_scale                   = &kDefaultSoftmaxScale;
    FTrainNumericType softmax_scale_type  = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType softmax_scale_index   = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* indexer_topk                    = &kDefaultIndexerTopk;
    FTrainNumericType indexer_topk_type   = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType indexer_topk_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type        = FTRAIN_NUMERIC_TYPE_FP32;
    alignas(16) static unsigned char kDefaultQMemory[16];
    alignas(16) static unsigned char kDefaultKvMemory[16];
    alignas(16) static unsigned char kDefaultAttnSinkMemory[16];
    alignas(16) static unsigned char kDefaultTopkIndicesMemory[16];
    alignas(16) static unsigned char kDefaultTopkLengthMemory[16];
    alignas(16) static unsigned char kDefaultOutMemory[16];
    alignas(16) static unsigned char kDefaultMaxLogitsMemory[16];
    alignas(16) static unsigned char kDefaultLseMemory[16];
    alignas(16) static unsigned char kDefaultLseIndexerMemory[16];
    static float kDefaultSoftmaxScale;
    static std::int64_t kDefaultIndexerTopk;
};

alignas(16) unsigned char ProblemPorts::kDefaultQMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultKvMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultAttnSinkMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultTopkIndicesMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultTopkLengthMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultOutMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultMaxLogitsMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultLseMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultLseIndexerMemory[16];
float ProblemPorts::kDefaultSoftmaxScale = 0.125F;

std::int64_t ProblemPorts::kDefaultIndexerTopk = 2;

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

SparseAttnFwdIndexerProblem makeProblem(const ProblemPorts& ports) {
    return SparseAttnFwdIndexerProblem{
        makeDeviceTensor(ports.q, ports.q_dims, 3, ports.q_type, ports.q_index),
        makeDeviceTensor(ports.kv, ports.kv_dims, 2, ports.kv_type, ports.kv_index),
        makeDeviceTensor(ports.attn_sink, ports.attn_sink_dims, 1, ports.attn_sink_type, ports.attn_sink_index),
        makeDeviceTensor(ports.topk_indices, ports.topk_indices_dims, 2, ports.topk_indices_type,
                         ports.topk_indices_index),
        makeDeviceTensor(ports.topk_length, ports.topk_length_dims, 1, ports.topk_length_type, ports.topk_length_index),
        makeDeviceTensor(ports.out, ports.out_dims, 3, ports.out_type, ports.out_index),
        makeDeviceTensor(ports.max_logits, ports.max_logits_dims, 2, ports.max_logits_type, ports.max_logits_index),
        makeDeviceTensor(ports.lse, ports.lse_dims, 2, ports.lse_type, ports.lse_index),
        makeDeviceTensor(ports.lse_indexer, ports.lse_indexer_dims, 2, ports.lse_indexer_type, ports.lse_indexer_index),
        makeHostScalar(ports.softmax_scale, ports.softmax_scale_type, ports.softmax_scale_index),
        makeHostScalar(ports.indexer_topk, ports.indexer_topk_type, ports.indexer_topk_index),
        SparseAttnFwdIndexerAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Fp32SparseAttnFwdIndexerTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_q[16];
    rebound.q = other_q;
    alignas(16) static unsigned char other_kv[16];
    rebound.kv = other_kv;
    alignas(16) static unsigned char other_attn_sink[16];
    rebound.attn_sink = other_attn_sink;
    alignas(16) static unsigned char other_topk_indices[16];
    rebound.topk_indices = other_topk_indices;
    alignas(16) static unsigned char other_topk_length[16];
    rebound.topk_length = other_topk_length;
    alignas(16) static unsigned char other_out[16];
    rebound.out = other_out;
    alignas(16) static unsigned char other_max_logits[16];
    rebound.max_logits = other_max_logits;
    alignas(16) static unsigned char other_lse[16];
    rebound.lse = other_lse;
    alignas(16) static unsigned char other_lse_indexer[16];
    rebound.lse_indexer             = other_lse_indexer;
    float other_softmax_scale       = 0.125F;
    rebound.softmax_scale           = &other_softmax_scale;
    std::int64_t other_indexer_topk = 2;
    rebound.indexer_topk            = &other_indexer_topk;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp32SparseAttnFwdIndexerTest, AcceptsContinuousAlignedOperands) {
    Fp32SparseAttnFwdIndexer primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp32SparseAttnFwdIndexer");
}

TEST(Fp32SparseAttnFwdIndexerTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp32SparseAttnFwdIndexer primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_q;
    unsupported_q.q_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_q), kConstraints).isSuccess());
    ProblemPorts unsupported_kv;
    unsupported_kv.kv_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_kv), kConstraints).isSuccess());
    ProblemPorts unsupported_attn_sink;
    unsupported_attn_sink.attn_sink_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_attn_sink), kConstraints).isSuccess());
    ProblemPorts unsupported_topk_indices;
    unsupported_topk_indices.topk_indices_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_topk_indices), kConstraints).isSuccess());
    ProblemPorts unsupported_topk_length;
    unsupported_topk_length.topk_length_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_topk_length), kConstraints).isSuccess());
    ProblemPorts unsupported_out;
    unsupported_out.out_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_out), kConstraints).isSuccess());
    ProblemPorts unsupported_max_logits;
    unsupported_max_logits.max_logits_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_max_logits), kConstraints).isSuccess());
    ProblemPorts unsupported_lse;
    unsupported_lse.lse_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_lse), kConstraints).isSuccess());
    ProblemPorts unsupported_lse_indexer;
    unsupported_lse_indexer.lse_indexer_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_lse_indexer), kConstraints).isSuccess());
    ProblemPorts unsupported_softmax_scale;
    unsupported_softmax_scale.softmax_scale_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_softmax_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_indexer_topk;
    unsupported_indexer_topk.indexer_topk_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_indexer_topk), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
}

TEST(Fp32SparseAttnFwdIndexerTest, RejectsNonContinuousIndexTypes) {
    Fp32SparseAttnFwdIndexer primitive;
    ProblemPorts strided_q;
    strided_q.q_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_q), kConstraints).isSuccess());
    ProblemPorts strided_kv;
    strided_kv.kv_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_kv), kConstraints).isSuccess());
    ProblemPorts strided_attn_sink;
    strided_attn_sink.attn_sink_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_attn_sink), kConstraints).isSuccess());
    ProblemPorts strided_topk_indices;
    strided_topk_indices.topk_indices_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_topk_indices), kConstraints).isSuccess());
    ProblemPorts strided_topk_length;
    strided_topk_length.topk_length_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_topk_length), kConstraints).isSuccess());
    ProblemPorts strided_out;
    strided_out.out_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_out), kConstraints).isSuccess());
    ProblemPorts strided_max_logits;
    strided_max_logits.max_logits_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_max_logits), kConstraints).isSuccess());
    ProblemPorts strided_lse;
    strided_lse.lse_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_lse), kConstraints).isSuccess());
    ProblemPorts strided_lse_indexer;
    strided_lse_indexer.lse_indexer_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_lse_indexer), kConstraints).isSuccess());
    ProblemPorts strided_softmax_scale;
    strided_softmax_scale.softmax_scale_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_softmax_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_indexer_topk;
    strided_indexer_topk.indexer_topk_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_indexer_topk), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32SparseAttnFwdIndexerTest, RejectsMisalignedAddresses) {
    Fp32SparseAttnFwdIndexer primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_q;
    misaligned_q.q = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_q), kConstraints).isSuccess());
    ProblemPorts misaligned_kv;
    misaligned_kv.kv = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_kv), kConstraints).isSuccess());
    ProblemPorts misaligned_attn_sink;
    misaligned_attn_sink.attn_sink = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_attn_sink), kConstraints).isSuccess());
    ProblemPorts misaligned_topk_indices;
    misaligned_topk_indices.topk_indices = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_topk_indices), kConstraints).isSuccess());
    ProblemPorts misaligned_topk_length;
    misaligned_topk_length.topk_length = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_topk_length), kConstraints).isSuccess());
    ProblemPorts misaligned_out;
    misaligned_out.out = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_out), kConstraints).isSuccess());
    ProblemPorts misaligned_max_logits;
    misaligned_max_logits.max_logits = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_max_logits), kConstraints).isSuccess());
    ProblemPorts misaligned_lse;
    misaligned_lse.lse = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_lse), kConstraints).isSuccess());
    ProblemPorts misaligned_lse_indexer;
    misaligned_lse_indexer.lse_indexer = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_lse_indexer), kConstraints).isSuccess());
    ProblemPorts misaligned_softmax_scale;
    misaligned_softmax_scale.softmax_scale = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_softmax_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_indexer_topk;
    misaligned_indexer_topk.indexer_topk = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_indexer_topk), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32SparseAttnFwdIndexerTest, ChecksRequiredAndOptionalAddresses) {
    Fp32SparseAttnFwdIndexer primitive;
    ProblemPorts null_q;
    null_q.q = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_q), kConstraints).isSuccess());
    ProblemPorts null_kv;
    null_kv.kv = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_kv), kConstraints).isSuccess());
    ProblemPorts null_attn_sink;
    null_attn_sink.attn_sink = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_attn_sink), kConstraints).isSuccess());
    ProblemPorts null_topk_indices;
    null_topk_indices.topk_indices = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_topk_indices), kConstraints).isSuccess());
    ProblemPorts null_topk_length;
    null_topk_length.topk_length = nullptr;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(null_topk_length), kConstraints).isSuccess());
    ProblemPorts null_out;
    null_out.out = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_out), kConstraints).isSuccess());
    ProblemPorts null_max_logits;
    null_max_logits.max_logits = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_max_logits), kConstraints).isSuccess());
    ProblemPorts null_lse;
    null_lse.lse = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_lse), kConstraints).isSuccess());
    ProblemPorts null_lse_indexer;
    null_lse_indexer.lse_indexer = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_lse_indexer), kConstraints).isSuccess());
    ProblemPorts null_softmax_scale;
    null_softmax_scale.softmax_scale = nullptr;
    try {
        makeProblem(null_softmax_scale);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_indexer_topk;
    null_indexer_topk.indexer_topk = nullptr;
    try {
        makeProblem(null_indexer_topk);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32SparseAttnFwdIndexerTest, ClonesIntoIndependentConfigurableCopies) {
    Fp32SparseAttnFwdIndexer prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp32SparseAttnFwdIndexer");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<SparseAttnFwdIndexerProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<SparseAttnFwdIndexerProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp32SparseAttnFwdIndexerTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original          = makeProblem(ports).getProblemKey();
        const std::int64_t q_dims[]  = {4, 2, 513};
        ports.q_dims                 = q_dims;
        const std::int64_t kv_dims[] = {4, 513};
        ports.kv_dims                = kv_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp32SparseAttnFwdIndexer primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp32SparseAttnFwdIndexerTest, RejectsMissingHeadDimensionInstantiation) {
    {
        ProblemPorts ports;

        const std::int64_t q_dims[] = {ports.q_dims[0], ports.q_dims[1], 528};
        ports.q_dims                = q_dims;

        const std::int64_t kv_dims[] = {ports.kv_dims[0], 528};
        ports.kv_dims                = kv_dims;

        auto problem = makeProblem(ports);
        Fp32SparseAttnFwdIndexer primitive;

        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp32SparseAttnFwdIndexerTest, RejectsExcessiveLaunchSharedMemory) {
    {
        ProblemPorts ports;

        const std::int64_t topk_indices_dims[] = {ports.topk_indices_dims[0], 100000};
        ports.topk_indices_dims                = topk_indices_dims;

        auto problem = makeProblem(ports);
        Fp32SparseAttnFwdIndexer primitive;

        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

}  // namespace ftrain
