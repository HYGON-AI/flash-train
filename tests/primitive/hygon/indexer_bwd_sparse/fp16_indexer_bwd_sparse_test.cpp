// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/indexer_bwd_sparse/problem.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_sparse/fp16_indexer_bwd_sparse.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kIndexQDims[4] = {1, 4, 2, 32};
constexpr std::int64_t kWeightsDims[3] = {1, 4, 2};
constexpr std::int64_t kIndexKDims[3] = {1, 4, 32};
constexpr std::int64_t kAttnScoreDims[3] = {1, 4, 4};
constexpr std::int64_t kIndexScoreDims[3] = {1, 4, 4};
constexpr std::int64_t kTopkIndicesDims[3] = {1, 4, 4};
constexpr std::int64_t kGradAttnScoreDims[3] = {1, 4, 4};
constexpr std::int64_t kDqDims[4] = {1, 4, 2, 32};
constexpr std::int64_t kDwDims[3] = {1, 4, 2};
constexpr std::int64_t kDkDims[3] = {1, 4, 32};

struct ProblemPorts {
    void* index_q                              = kDefaultIndexQMemory;
    const std::int64_t* index_q_dims           = kIndexQDims;
    FTrainNumericType index_q_type             = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType index_q_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* weights                              = kDefaultWeightsMemory;
    const std::int64_t* weights_dims           = kWeightsDims;
    FTrainNumericType weights_type             = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType weights_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* index_k                              = kDefaultIndexKMemory;
    const std::int64_t* index_k_dims           = kIndexKDims;
    FTrainNumericType index_k_type             = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType index_k_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* attn_score                           = kDefaultAttnScoreMemory;
    const std::int64_t* attn_score_dims        = kAttnScoreDims;
    FTrainNumericType attn_score_type          = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType attn_score_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* index_score                          = kDefaultIndexScoreMemory;
    const std::int64_t* index_score_dims       = kIndexScoreDims;
    FTrainNumericType index_score_type         = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType index_score_index          = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* topk_indices                         = kDefaultTopkIndicesMemory;
    const std::int64_t* topk_indices_dims      = kTopkIndicesDims;
    FTrainNumericType topk_indices_type        = FTRAIN_NUMERIC_TYPE_INT32;
    FTrainIndexType topk_indices_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_attn_score                      = kDefaultAttnScoreMemory;
    const std::int64_t* grad_attn_score_dims   = kGradAttnScoreDims;
    FTrainNumericType grad_attn_score_type     = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_attn_score_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* dq                                   = kDefaultDqMemory;
    const std::int64_t* dq_dims                = kDqDims;
    FTrainNumericType dq_type                  = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType dq_index                   = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* dw                                   = kDefaultDwMemory;
    const std::int64_t* dw_dims                = kDwDims;
    FTrainNumericType dw_type                  = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType dw_index                   = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* dk                                   = kDefaultDkMemory;
    const std::int64_t* dk_dims                = kDkDims;
    FTrainNumericType dk_type                  = FTRAIN_NUMERIC_TYPE_FP16;
    FTrainIndexType dk_index                   = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* sm_scale                             = &kDefaultSmScale;
    FTrainNumericType sm_scale_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType sm_scale_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* loss_coeff                           = &kDefaultLossCoeff;
    FTrainNumericType loss_coeff_type          = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType loss_coeff_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_loss                            = &kDefaultGradLoss;
    FTrainNumericType grad_loss_type           = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_loss_index            = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* topk_indices_global                  = &kDefaultTopkIndicesGlobal;
    FTrainNumericType topk_indices_global_type = FTRAIN_NUMERIC_TYPE_UINT8;
    FTrainIndexType topk_indices_global_index  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type             = FTRAIN_NUMERIC_TYPE_FP16;
    alignas(16) static unsigned char kDefaultIndexQMemory[16];
    alignas(16) static unsigned char kDefaultWeightsMemory[16];
    alignas(16) static unsigned char kDefaultIndexKMemory[16];
    alignas(16) static unsigned char kDefaultAttnScoreMemory[16];
    alignas(16) static unsigned char kDefaultIndexScoreMemory[16];
    alignas(16) static unsigned char kDefaultTopkIndicesMemory[16];
    alignas(16) static unsigned char kDefaultDqMemory[16];
    alignas(16) static unsigned char kDefaultDwMemory[16];
    alignas(16) static unsigned char kDefaultDkMemory[16];
    static float kDefaultSmScale;
    static float kDefaultLossCoeff;
    static float kDefaultGradLoss;
    static std::uint8_t kDefaultTopkIndicesGlobal;
};

alignas(16) unsigned char ProblemPorts::kDefaultIndexQMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultWeightsMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultIndexKMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultAttnScoreMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultIndexScoreMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultTopkIndicesMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDqMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDwMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDkMemory[16];
float ProblemPorts::kDefaultSmScale = 0.25F;

float ProblemPorts::kDefaultLossCoeff = 1;

float ProblemPorts::kDefaultGradLoss = 1;

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

IndexerBwdSparseProblem makeProblem(const ProblemPorts& ports, bool grad_loss_host = true) {
    return IndexerBwdSparseProblem{
        makeDeviceTensor(ports.index_q, ports.index_q_dims, 4, ports.index_q_type, ports.index_q_index),
        makeDeviceTensor(ports.weights, ports.weights_dims, 3, ports.weights_type, ports.weights_index),
        makeDeviceTensor(ports.index_k, ports.index_k_dims, 3, ports.index_k_type, ports.index_k_index),
        makeDeviceTensor(ports.attn_score, ports.attn_score_dims, 3, ports.attn_score_type, ports.attn_score_index),
        makeDeviceTensor(ports.index_score, ports.index_score_dims, 3, ports.index_score_type, ports.index_score_index),
        makeDeviceTensor(ports.topk_indices, ports.topk_indices_dims, 3, ports.topk_indices_type,
                         ports.topk_indices_index),
        makeDeviceTensor(ports.grad_attn_score, ports.grad_attn_score_dims, 3, ports.grad_attn_score_type,
                         ports.grad_attn_score_index),
        makeDeviceTensor(ports.dq, ports.dq_dims, 4, ports.dq_type, ports.dq_index),
        makeDeviceTensor(ports.dw, ports.dw_dims, 3, ports.dw_type, ports.dw_index),
        makeDeviceTensor(ports.dk, ports.dk_dims, 3, ports.dk_type, ports.dk_index),
        makeHostScalar(ports.sm_scale, ports.sm_scale_type, ports.sm_scale_index),
        makeHostScalar(ports.loss_coeff, ports.loss_coeff_type, ports.loss_coeff_index),
        grad_loss_host ? makeHostScalar(ports.grad_loss, ports.grad_loss_type, ports.grad_loss_index)
            : makeDeviceTensor(ports.grad_loss, nullptr, 0, ports.grad_loss_type, ports.grad_loss_index),
        makeHostScalar(ports.topk_indices_global, ports.topk_indices_global_type, ports.topk_indices_global_index),
        IndexerBwdSparseAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Fp16IndexerBwdSparseTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_index_q[16];
    rebound.index_q = other_index_q;
    alignas(16) static unsigned char other_weights[16];
    rebound.weights = other_weights;
    alignas(16) static unsigned char other_index_k[16];
    rebound.index_k = other_index_k;
    alignas(16) static unsigned char other_attn_score[16];
    rebound.attn_score = other_attn_score;
    alignas(16) static unsigned char other_index_score[16];
    rebound.index_score = other_index_score;
    alignas(16) static unsigned char other_topk_indices[16];
    rebound.topk_indices = other_topk_indices;
    alignas(16) static unsigned char other_dq[16];
    rebound.dq = other_dq;
    alignas(16) static unsigned char other_dw[16];
    rebound.dw = other_dw;
    alignas(16) static unsigned char other_dk[16];
    rebound.dk                             = other_dk;
    float other_sm_scale                   = 0.25F;
    rebound.sm_scale                       = &other_sm_scale;
    float other_loss_coeff                 = 1;
    rebound.loss_coeff                     = &other_loss_coeff;
    float other_grad_loss                  = 1;
    rebound.grad_loss                      = &other_grad_loss;
    std::uint8_t other_topk_indices_global = 0;
    rebound.topk_indices_global            = &other_topk_indices_global;
    rebound.grad_attn_score                = rebound.attn_score;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp16IndexerBwdSparseTest, AcceptsContinuousAlignedOperands) {
    Fp16IndexerBwdSparse primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp16IndexerBwdSparse");
}

TEST(Fp16IndexerBwdSparseTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp16IndexerBwdSparse primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_index_q;
    unsupported_index_q.index_q_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_index_q), kConstraints).isSuccess());
    ProblemPorts unsupported_weights;
    unsupported_weights.weights_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_weights), kConstraints).isSuccess());
    ProblemPorts unsupported_index_k;
    unsupported_index_k.index_k_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_index_k), kConstraints).isSuccess());
    ProblemPorts unsupported_attn_score;
    unsupported_attn_score.attn_score_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_attn_score), kConstraints).isSuccess());
    ProblemPorts unsupported_index_score;
    unsupported_index_score.index_score_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_index_score), kConstraints).isSuccess());
    ProblemPorts unsupported_topk_indices;
    unsupported_topk_indices.topk_indices_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_topk_indices), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_attn_score;
    unsupported_grad_attn_score.grad_attn_score_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_attn_score), kConstraints).isSuccess());
    ProblemPorts unsupported_dq;
    unsupported_dq.dq_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_dq), kConstraints).isSuccess());
    ProblemPorts unsupported_dw;
    unsupported_dw.dw_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_dw), kConstraints).isSuccess());
    ProblemPorts unsupported_dk;
    unsupported_dk.dk_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_dk), kConstraints).isSuccess());
    ProblemPorts unsupported_sm_scale;
    unsupported_sm_scale.sm_scale_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_sm_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_loss_coeff;
    unsupported_loss_coeff.loss_coeff_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_loss_coeff), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_grad_loss;
    unsupported_grad_loss.grad_loss_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_loss), kConstraints).isSuccess());
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

TEST(Fp16IndexerBwdSparseTest, RejectsNonContinuousIndexTypes) {
    Fp16IndexerBwdSparse primitive;
    ProblemPorts strided_index_q;
    strided_index_q.index_q_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_index_q), kConstraints).isSuccess());
    ProblemPorts strided_weights;
    strided_weights.weights_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_weights), kConstraints).isSuccess());
    ProblemPorts strided_index_k;
    strided_index_k.index_k_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_index_k), kConstraints).isSuccess());
    ProblemPorts strided_attn_score;
    strided_attn_score.attn_score_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_attn_score), kConstraints).isSuccess());
    ProblemPorts strided_index_score;
    strided_index_score.index_score_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_index_score), kConstraints).isSuccess());
    ProblemPorts strided_topk_indices;
    strided_topk_indices.topk_indices_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_topk_indices), kConstraints).isSuccess());
    ProblemPorts strided_grad_attn_score;
    strided_grad_attn_score.grad_attn_score_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_attn_score), kConstraints).isSuccess());
    ProblemPorts strided_dq;
    strided_dq.dq_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_dq), kConstraints).isSuccess());
    ProblemPorts strided_dw;
    strided_dw.dw_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_dw), kConstraints).isSuccess());
    ProblemPorts strided_dk;
    strided_dk.dk_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_dk), kConstraints).isSuccess());
    ProblemPorts strided_sm_scale;
    strided_sm_scale.sm_scale_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_sm_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_loss_coeff;
    strided_loss_coeff.loss_coeff_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_loss_coeff), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_grad_loss;
    strided_grad_loss.grad_loss_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_loss), kConstraints).isSuccess());
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

TEST(Fp16IndexerBwdSparseTest, RejectsMisalignedAddresses) {
    Fp16IndexerBwdSparse primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_index_q;
    misaligned_index_q.index_q = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_index_q), kConstraints).isSuccess());
    ProblemPorts misaligned_weights;
    misaligned_weights.weights = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_weights), kConstraints).isSuccess());
    ProblemPorts misaligned_index_k;
    misaligned_index_k.index_k = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_index_k), kConstraints).isSuccess());
    ProblemPorts misaligned_attn_score;
    misaligned_attn_score.attn_score = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_attn_score), kConstraints).isSuccess());
    ProblemPorts misaligned_index_score;
    misaligned_index_score.index_score = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_index_score), kConstraints).isSuccess());
    ProblemPorts misaligned_topk_indices;
    misaligned_topk_indices.topk_indices = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_topk_indices), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_attn_score;
    misaligned_grad_attn_score.grad_attn_score = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_attn_score), kConstraints).isSuccess());
    ProblemPorts misaligned_dq;
    misaligned_dq.dq = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_dq), kConstraints).isSuccess());
    ProblemPorts misaligned_dw;
    misaligned_dw.dw = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_dw), kConstraints).isSuccess());
    ProblemPorts misaligned_dk;
    misaligned_dk.dk = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_dk), kConstraints).isSuccess());
    ProblemPorts misaligned_sm_scale;
    misaligned_sm_scale.sm_scale = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_sm_scale), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_loss_coeff;
    misaligned_loss_coeff.loss_coeff = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_loss_coeff), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_grad_loss;
    misaligned_grad_loss.grad_loss = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_loss), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp16IndexerBwdSparseTest, ChecksRequiredAndOptionalAddresses) {
    Fp16IndexerBwdSparse primitive;
    ProblemPorts null_index_q;
    null_index_q.index_q = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_index_q), kConstraints).isSuccess());
    ProblemPorts null_weights;
    null_weights.weights = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_weights), kConstraints).isSuccess());
    ProblemPorts null_index_k;
    null_index_k.index_k = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_index_k), kConstraints).isSuccess());
    ProblemPorts null_attn_score;
    null_attn_score.attn_score = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_attn_score), kConstraints).isSuccess());
    ProblemPorts null_index_score;
    null_index_score.index_score = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_index_score), kConstraints).isSuccess());
    ProblemPorts null_topk_indices;
    null_topk_indices.topk_indices = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_topk_indices), kConstraints).isSuccess());
    ProblemPorts null_grad_attn_score;
    null_grad_attn_score.grad_attn_score = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_attn_score), kConstraints).isSuccess());
    ProblemPorts null_dq;
    null_dq.dq = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_dq), kConstraints).isSuccess());
    ProblemPorts null_dw;
    null_dw.dw = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_dw), kConstraints).isSuccess());
    ProblemPorts null_dk;
    null_dk.dk = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_dk), kConstraints).isSuccess());
    ProblemPorts null_sm_scale;
    null_sm_scale.sm_scale = nullptr;
    try {
        makeProblem(null_sm_scale);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_loss_coeff;
    null_loss_coeff.loss_coeff = nullptr;
    try {
        makeProblem(null_loss_coeff);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_grad_loss;
    null_grad_loss.grad_loss = nullptr;
    try {
        makeProblem(null_grad_loss);
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

TEST(Fp16IndexerBwdSparseTest, ClonesIntoIndependentConfigurableCopies) {
    Fp16IndexerBwdSparse prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp16IndexerBwdSparse");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<IndexerBwdSparseProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<IndexerBwdSparseProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp16IndexerBwdSparseTest, RequiresDeclaredAliasing) {
    ProblemPorts ports;

    auto problem = makeProblem(ports);
    Fp16IndexerBwdSparse primitive;
    const auto original_key = problem.getProblemKey();
    alignas(16) static unsigned char separate_memory[16];

    ProblemPorts separate    = ports;
    separate.grad_attn_score = separate_memory;
    const auto out_of_place  = makeProblem(separate);
    EXPECT_NE(original_key, out_of_place.getProblemKey());
    EXPECT_FALSE(primitive.isApplicable(out_of_place, Constraints{0}).isSuccess());
}

TEST(Fp16IndexerBwdSparseTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original               = makeProblem(ports).getProblemKey();
        const std::int64_t index_q_dims[] = {1, 4, 2, 33};
        ports.index_q_dims                = index_q_dims;
        const std::int64_t index_k_dims[] = {1, 4, 33};
        ports.index_k_dims                = index_k_dims;
        const std::int64_t dq_dims[]      = {1, 4, 2, 33};
        ports.dq_dims                     = dq_dims;
        const std::int64_t dk_dims[]      = {1, 4, 33};
        ports.dk_dims                     = dk_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Fp16IndexerBwdSparse primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp16IndexerBwdSparseTest, RejectsSelectionBeyondLauncherCoverage) {
    {
        ProblemPorts ports;

        const std::int64_t attn_score_dims[] = {ports.attn_score_dims[0], ports.attn_score_dims[1], 2049};
        ports.attn_score_dims                = attn_score_dims;

        const std::int64_t index_score_dims[] = {ports.index_score_dims[0], ports.index_score_dims[1], 2049};
        ports.index_score_dims                = index_score_dims;

        const std::int64_t topk_indices_dims[] = {ports.topk_indices_dims[0], ports.topk_indices_dims[1], 2049};
        ports.topk_indices_dims                = topk_indices_dims;

        const std::int64_t grad_attn_score_dims[] = {ports.grad_attn_score_dims[0], ports.grad_attn_score_dims[1],
                                                     2049};
        ports.grad_attn_score_dims                = grad_attn_score_dims;

        auto problem = makeProblem(ports);
        Fp16IndexerBwdSparse primitive;

        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp16IndexerBwdSparseTest, WorkspaceScalesWithKeysAndBatchOnly) {
    for (const std::int64_t queries : {4, 8}) {
        ProblemPorts ports;
        const std::int64_t q_dims[]       = {2, queries, 2, 32};
        const std::int64_t weights_dims[] = {2, queries, 2};
        const std::int64_t k_dims[]       = {2, 4, 32};
        const std::int64_t score_dims[]   = {2, queries, 4};
        ports.index_q_dims = ports.dq_dims = q_dims;
        ports.weights_dims = ports.dw_dims = weights_dims;
        ports.index_k_dims = ports.dk_dims = k_dims;
        ports.attn_score_dims = ports.grad_attn_score_dims = score_dims;
        ports.index_score_dims = ports.topk_indices_dims = score_dims;
        Fp16IndexerBwdSparse primitive;
        primitive.configure(makeProblem(ports));
        EXPECT_EQ(primitive.getRequiredWorkspaceBytes(), 2ULL * 4 * 32 * sizeof(float) + sizeof(int));
    }
}

TEST(Fp16IndexerBwdSparseTest, RejectsDeviceGradLossBeforeConfigure) {
    ProblemPorts ports;
    ports.grad_loss = reinterpret_cast<void*>(0x1000);
    const auto problem = makeProblem(ports, false);
    Fp16IndexerBwdSparse primitive;
    EXPECT_EQ(primitive.isApplicable(problem, kConstraints).getStatus(), FTRAIN_STATUS_UNSUPPORTED);
}

}  // namespace ftrain
