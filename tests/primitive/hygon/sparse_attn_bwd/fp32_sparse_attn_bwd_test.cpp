// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/sparse_attn_bwd/problem.hpp"
#include "flash_train/primitive/hygon/sparse_attn_bwd/fp32_sparse_attn_bwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kQDims[3] = {4, 2, 512};
constexpr std::int64_t kKvDims[2] = {4, 512};
constexpr std::int64_t kOutDims[3] = {4, 2, 512};
constexpr std::int64_t kDoutDims[3] = {4, 2, 512};
constexpr std::int64_t kLseDims[2] = {4, 2};
constexpr std::int64_t kAttnSinkDims[1] = {2};
constexpr std::int64_t kTopkIndicesDims[2] = {4, 4};
constexpr std::int64_t kTopkLengthDims[1] = {4};
constexpr std::int64_t kDqDims[3] = {4, 2, 512};
constexpr std::int64_t kDkvDims[2] = {4, 512};
constexpr std::int64_t kDsinkDims[1] = {2};

struct ProblemPorts {
    void* q                               = kDefaultQMemory;
    const std::int64_t* q_dims            = kQDims;
    FTrainNumericType q_type              = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType q_index               = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* kv                              = kDefaultKvMemory;
    const std::int64_t* kv_dims           = kKvDims;
    FTrainNumericType kv_type             = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType kv_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* out                             = kDefaultOutMemory;
    const std::int64_t* out_dims          = kOutDims;
    FTrainNumericType out_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType out_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* dout                            = kDefaultDoutMemory;
    const std::int64_t* dout_dims         = kDoutDims;
    FTrainNumericType dout_type           = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType dout_index            = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* lse                             = kDefaultLseMemory;
    const std::int64_t* lse_dims          = kLseDims;
    FTrainNumericType lse_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType lse_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
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
    void* dq                              = kDefaultDqMemory;
    const std::int64_t* dq_dims           = kDqDims;
    FTrainNumericType dq_type             = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType dq_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* dkv                             = kDefaultDkvMemory;
    const std::int64_t* dkv_dims          = kDkvDims;
    FTrainNumericType dkv_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType dkv_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* dsink                           = kDefaultDsinkMemory;
    const std::int64_t* dsink_dims        = kDsinkDims;
    FTrainNumericType dsink_type          = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType dsink_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* softmax_scale                   = &kDefaultSoftmaxScale;
    FTrainNumericType softmax_scale_type  = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType softmax_scale_index   = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type        = FTRAIN_NUMERIC_TYPE_FP32;
    alignas(16) static unsigned char kDefaultQMemory[16];
    alignas(16) static unsigned char kDefaultKvMemory[16];
    alignas(16) static unsigned char kDefaultOutMemory[16];
    alignas(16) static unsigned char kDefaultDoutMemory[16];
    alignas(16) static unsigned char kDefaultLseMemory[16];
    alignas(16) static unsigned char kDefaultAttnSinkMemory[16];
    alignas(16) static unsigned char kDefaultTopkIndicesMemory[16];
    alignas(16) static unsigned char kDefaultTopkLengthMemory[16];
    alignas(16) static unsigned char kDefaultDqMemory[16];
    alignas(16) static unsigned char kDefaultDkvMemory[16];
    alignas(16) static unsigned char kDefaultDsinkMemory[16];
    static float kDefaultSoftmaxScale;
};

alignas(16) unsigned char ProblemPorts::kDefaultQMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultKvMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultOutMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDoutMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultLseMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultAttnSinkMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultTopkIndicesMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultTopkLengthMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDqMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDkvMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDsinkMemory[16];
float ProblemPorts::kDefaultSoftmaxScale = 0.125F;

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

SparseAttnBwdProblem makeProblem(const ProblemPorts& ports) {
    return SparseAttnBwdProblem{
        makeDeviceTensor(ports.q, ports.q_dims, 3, ports.q_type, ports.q_index),
        makeDeviceTensor(ports.kv, ports.kv_dims, 2, ports.kv_type, ports.kv_index),
        makeDeviceTensor(ports.out, ports.out_dims, 3, ports.out_type, ports.out_index),
        makeDeviceTensor(ports.dout, ports.dout_dims, 3, ports.dout_type, ports.dout_index),
        makeDeviceTensor(ports.lse, ports.lse_dims, 2, ports.lse_type, ports.lse_index),
        makeDeviceTensor(ports.attn_sink, ports.attn_sink_dims, 1, ports.attn_sink_type, ports.attn_sink_index),
        makeDeviceTensor(ports.topk_indices, ports.topk_indices_dims, 2, ports.topk_indices_type,
                         ports.topk_indices_index),
        makeDeviceTensor(ports.topk_length, ports.topk_length_dims, 1, ports.topk_length_type, ports.topk_length_index),
        makeDeviceTensor(ports.dq, ports.dq_dims, 3, ports.dq_type, ports.dq_index),
        makeDeviceTensor(ports.dkv, ports.dkv_dims, 2, ports.dkv_type, ports.dkv_index),
        makeDeviceTensor(ports.dsink, ports.dsink_dims, 1, ports.dsink_type, ports.dsink_index),
        makeHostScalar(ports.softmax_scale, ports.softmax_scale_type, ports.softmax_scale_index),
        SparseAttnBwdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Fp32SparseAttnBwdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_q[16];
    rebound.q = other_q;
    alignas(16) static unsigned char other_kv[16];
    rebound.kv = other_kv;
    alignas(16) static unsigned char other_out[16];
    rebound.out = other_out;
    alignas(16) static unsigned char other_dout[16];
    rebound.dout = other_dout;
    alignas(16) static unsigned char other_lse[16];
    rebound.lse = other_lse;
    alignas(16) static unsigned char other_attn_sink[16];
    rebound.attn_sink = other_attn_sink;
    alignas(16) static unsigned char other_topk_indices[16];
    rebound.topk_indices = other_topk_indices;
    alignas(16) static unsigned char other_topk_length[16];
    rebound.topk_length = other_topk_length;
    alignas(16) static unsigned char other_dq[16];
    rebound.dq = other_dq;
    alignas(16) static unsigned char other_dkv[16];
    rebound.dkv = other_dkv;
    alignas(16) static unsigned char other_dsink[16];
    rebound.dsink             = other_dsink;
    float other_softmax_scale = 0.125F;
    rebound.softmax_scale     = &other_softmax_scale;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp32SparseAttnBwdTest, AcceptsContinuousAlignedOperands) {
    Fp32SparseAttnBwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp32SparseAttnBwd");
}

TEST(Fp32SparseAttnBwdTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp32SparseAttnBwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_q;
    unsupported_q.q_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_q), kConstraints).isSuccess());
    ProblemPorts unsupported_kv;
    unsupported_kv.kv_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_kv), kConstraints).isSuccess());
    ProblemPorts unsupported_out;
    unsupported_out.out_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_out), kConstraints).isSuccess());
    ProblemPorts unsupported_dout;
    unsupported_dout.dout_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_dout), kConstraints).isSuccess());
    ProblemPorts unsupported_lse;
    unsupported_lse.lse_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_lse), kConstraints).isSuccess());
    ProblemPorts unsupported_attn_sink;
    unsupported_attn_sink.attn_sink_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_attn_sink), kConstraints).isSuccess());
    ProblemPorts unsupported_topk_indices;
    unsupported_topk_indices.topk_indices_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_topk_indices), kConstraints).isSuccess());
    ProblemPorts unsupported_topk_length;
    unsupported_topk_length.topk_length_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_topk_length), kConstraints).isSuccess());
    ProblemPorts unsupported_dq;
    unsupported_dq.dq_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_dq), kConstraints).isSuccess());
    ProblemPorts unsupported_dkv;
    unsupported_dkv.dkv_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_dkv), kConstraints).isSuccess());
    ProblemPorts unsupported_dsink;
    unsupported_dsink.dsink_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_dsink), kConstraints).isSuccess());
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

TEST(Fp32SparseAttnBwdTest, RejectsNonContinuousIndexTypes) {
    Fp32SparseAttnBwd primitive;
    ProblemPorts strided_q;
    strided_q.q_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_q), kConstraints).isSuccess());
    ProblemPorts strided_kv;
    strided_kv.kv_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_kv), kConstraints).isSuccess());
    ProblemPorts strided_out;
    strided_out.out_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_out), kConstraints).isSuccess());
    ProblemPorts strided_dout;
    strided_dout.dout_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_dout), kConstraints).isSuccess());
    ProblemPorts strided_lse;
    strided_lse.lse_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_lse), kConstraints).isSuccess());
    ProblemPorts strided_attn_sink;
    strided_attn_sink.attn_sink_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_attn_sink), kConstraints).isSuccess());
    ProblemPorts strided_topk_indices;
    strided_topk_indices.topk_indices_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_topk_indices), kConstraints).isSuccess());
    ProblemPorts strided_topk_length;
    strided_topk_length.topk_length_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_topk_length), kConstraints).isSuccess());
    ProblemPorts strided_dq;
    strided_dq.dq_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_dq), kConstraints).isSuccess());
    ProblemPorts strided_dkv;
    strided_dkv.dkv_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_dkv), kConstraints).isSuccess());
    ProblemPorts strided_dsink;
    strided_dsink.dsink_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_dsink), kConstraints).isSuccess());
    ProblemPorts strided_softmax_scale;
    strided_softmax_scale.softmax_scale_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_softmax_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32SparseAttnBwdTest, RejectsMisalignedAddresses) {
    Fp32SparseAttnBwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_q;
    misaligned_q.q = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_q), kConstraints).isSuccess());
    ProblemPorts misaligned_kv;
    misaligned_kv.kv = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_kv), kConstraints).isSuccess());
    ProblemPorts misaligned_out;
    misaligned_out.out = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_out), kConstraints).isSuccess());
    ProblemPorts misaligned_dout;
    misaligned_dout.dout = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_dout), kConstraints).isSuccess());
    ProblemPorts misaligned_lse;
    misaligned_lse.lse = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_lse), kConstraints).isSuccess());
    ProblemPorts misaligned_attn_sink;
    misaligned_attn_sink.attn_sink = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_attn_sink), kConstraints).isSuccess());
    ProblemPorts misaligned_topk_indices;
    misaligned_topk_indices.topk_indices = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_topk_indices), kConstraints).isSuccess());
    ProblemPorts misaligned_topk_length;
    misaligned_topk_length.topk_length = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_topk_length), kConstraints).isSuccess());
    ProblemPorts misaligned_dq;
    misaligned_dq.dq = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_dq), kConstraints).isSuccess());
    ProblemPorts misaligned_dkv;
    misaligned_dkv.dkv = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_dkv), kConstraints).isSuccess());
    ProblemPorts misaligned_dsink;
    misaligned_dsink.dsink = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_dsink), kConstraints).isSuccess());
    ProblemPorts misaligned_softmax_scale;
    misaligned_softmax_scale.softmax_scale = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_softmax_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32SparseAttnBwdTest, ChecksRequiredAndOptionalAddresses) {
    Fp32SparseAttnBwd primitive;
    ProblemPorts null_q;
    null_q.q = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_q), kConstraints).isSuccess());
    ProblemPorts null_kv;
    null_kv.kv = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_kv), kConstraints).isSuccess());
    ProblemPorts null_out;
    null_out.out = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_out), kConstraints).isSuccess());
    ProblemPorts null_dout;
    null_dout.dout = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_dout), kConstraints).isSuccess());
    ProblemPorts null_lse;
    null_lse.lse = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_lse), kConstraints).isSuccess());
    ProblemPorts null_attn_sink;
    null_attn_sink.attn_sink = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_attn_sink), kConstraints).isSuccess());
    ProblemPorts null_topk_indices;
    null_topk_indices.topk_indices = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_topk_indices), kConstraints).isSuccess());
    ProblemPorts null_topk_length;
    null_topk_length.topk_length = nullptr;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(null_topk_length), kConstraints).isSuccess());
    ProblemPorts null_dq;
    null_dq.dq = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_dq), kConstraints).isSuccess());
    ProblemPorts null_dkv;
    null_dkv.dkv = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_dkv), kConstraints).isSuccess());
    ProblemPorts null_dsink;
    null_dsink.dsink = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_dsink), kConstraints).isSuccess());
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

TEST(Fp32SparseAttnBwdTest, ClonesIntoDistinctConfigurableCopies) {
    Fp32SparseAttnBwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp32SparseAttnBwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<SparseAttnBwdProblem>&>(*first).configure(makeProblem(ports));
    alignas(16) unsigned char other_memory[16]{};
    {
        ProblemPorts other;
        other.q = other_memory;
        static_cast<Primitive<SparseAttnBwdProblem>&>(*second).configure(makeProblem(other));
    }
    // Workspace checks alone do not prove pointer independence.
    // RebindsAddressesAndOutlivesOpsAndArgs covers execution with separate plans.
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp32SparseAttnBwdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original           = makeProblem(ports).getProblemKey();
        const std::int64_t q_dims[]   = {4, 2, 513};
        ports.q_dims                  = q_dims;
        const std::int64_t kv_dims[]  = {4, 513};
        ports.kv_dims                 = kv_dims;
        const std::int64_t dq_dims[]  = {4, 2, 513};
        ports.dq_dims                 = dq_dims;
        const std::int64_t dkv_dims[] = {4, 513};
        ports.dkv_dims                = dkv_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp32SparseAttnBwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp32SparseAttnBwdTest, RejectsMissingHeadDimensionInstantiation) {
    {
        ProblemPorts ports;

        const std::int64_t q_dims[] = {ports.q_dims[0], ports.q_dims[1], 528};
        ports.q_dims                = q_dims;

        const std::int64_t kv_dims[] = {ports.kv_dims[0], 528};
        ports.kv_dims                = kv_dims;

        const std::int64_t out_dims[] = {ports.out_dims[0], ports.out_dims[1], 528};
        ports.out_dims                = out_dims;

        const std::int64_t dout_dims[] = {ports.dout_dims[0], ports.dout_dims[1], 528};
        ports.dout_dims                = dout_dims;

        const std::int64_t dq_dims[] = {ports.dq_dims[0], ports.dq_dims[1], 528};
        ports.dq_dims                = dq_dims;

        const std::int64_t dkv_dims[] = {ports.dkv_dims[0], 528};
        ports.dkv_dims                = dkv_dims;

        auto problem = makeProblem(ports);
        Fp32SparseAttnBwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp32SparseAttnBwdTest, RejectsExcessiveLaunchSharedMemory) {
    {
        ProblemPorts ports;

        const std::int64_t topk_indices_dims[] = {ports.topk_indices_dims[0], 100000};
        ports.topk_indices_dims                = topk_indices_dims;

        auto problem = makeProblem(ports);
        Fp32SparseAttnBwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

}  // namespace ftrain
