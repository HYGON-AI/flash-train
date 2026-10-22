// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/indexer_bwd_dense/problem.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_dense/fp32_indexer_bwd_dense.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kIndexQDims[4] = {1, 4, 2, 32};
constexpr std::int64_t kWeightsDims[3] = {1, 4, 2};
constexpr std::int64_t kIndexKDims[3] = {1, 4, 32};
constexpr std::int64_t kAttnScoreDims[3] = {1, 4, 4};
constexpr std::int64_t kAttnL1NormDims[2] = {1, 4};
constexpr std::int64_t kIndexScoreDims[3] = {1, 4, 4};
constexpr std::int64_t kIndexLseDims[2] = {1, 4};
constexpr std::int64_t kGradIndexScoreDims[3] = {1, 4, 4};
constexpr std::int64_t kDqDims[4] = {1, 4, 2, 32};
constexpr std::int64_t kDwDims[3] = {1, 4, 2};
constexpr std::int64_t kDkDims[3] = {1, 4, 32};

struct ProblemPorts {
    void* index_q                             = kDefaultIndexQMemory;
    const std::int64_t* index_q_dims          = kIndexQDims;
    FTrainNumericType index_q_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType index_q_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* weights                             = kDefaultWeightsMemory;
    const std::int64_t* weights_dims          = kWeightsDims;
    FTrainNumericType weights_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType weights_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* index_k                             = kDefaultIndexKMemory;
    const std::int64_t* index_k_dims          = kIndexKDims;
    FTrainNumericType index_k_type            = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType index_k_index             = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* attn_score                          = kDefaultAttnScoreMemory;
    const std::int64_t* attn_score_dims       = kAttnScoreDims;
    FTrainNumericType attn_score_type         = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType attn_score_index          = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* attn_l1norm                         = kDefaultAttnL1NormMemory;
    const std::int64_t* attn_l1norm_dims      = kAttnL1NormDims;
    FTrainNumericType attn_l1norm_type        = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType attn_l1norm_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* index_score                         = kDefaultIndexScoreMemory;
    const std::int64_t* index_score_dims      = kIndexScoreDims;
    FTrainNumericType index_score_type        = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType index_score_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* index_lse                           = kDefaultIndexLseMemory;
    const std::int64_t* index_lse_dims        = kIndexLseDims;
    FTrainNumericType index_lse_type          = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType index_lse_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_index_score                    = kDefaultIndexScoreMemory;
    const std::int64_t* grad_index_score_dims = kGradIndexScoreDims;
    FTrainNumericType grad_index_score_type   = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_index_score_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* dq                                  = kDefaultDqMemory;
    const std::int64_t* dq_dims               = kDqDims;
    FTrainNumericType dq_type                 = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType dq_index                  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* dw                                  = kDefaultDwMemory;
    const std::int64_t* dw_dims               = kDwDims;
    FTrainNumericType dw_type                 = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType dw_index                  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* dk                                  = kDefaultDkMemory;
    const std::int64_t* dk_dims               = kDkDims;
    FTrainNumericType dk_type                 = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType dk_index                  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* sm_scale                            = &kDefaultSmScale;
    FTrainNumericType sm_scale_type           = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType sm_scale_index            = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* loss_coeff                          = &kDefaultLossCoeff;
    FTrainNumericType loss_coeff_type         = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType loss_coeff_index          = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_loss                           = &kDefaultGradLoss;
    FTrainNumericType grad_loss_type          = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_loss_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* ratio                               = &kDefaultRatio;
    FTrainNumericType ratio_type              = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType ratio_index               = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type            = FTRAIN_NUMERIC_TYPE_FP32;
    alignas(16) static unsigned char kDefaultIndexQMemory[16];
    alignas(16) static unsigned char kDefaultWeightsMemory[16];
    alignas(16) static unsigned char kDefaultIndexKMemory[16];
    alignas(16) static unsigned char kDefaultAttnScoreMemory[16];
    alignas(16) static unsigned char kDefaultAttnL1NormMemory[16];
    alignas(16) static unsigned char kDefaultIndexScoreMemory[16];
    alignas(16) static unsigned char kDefaultIndexLseMemory[16];
    alignas(16) static unsigned char kDefaultDqMemory[16];
    alignas(16) static unsigned char kDefaultDwMemory[16];
    alignas(16) static unsigned char kDefaultDkMemory[16];
    static float kDefaultSmScale;
    static float kDefaultLossCoeff;
    static float kDefaultGradLoss;
    static std::int64_t kDefaultRatio;
};

alignas(16) unsigned char ProblemPorts::kDefaultIndexQMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultWeightsMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultIndexKMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultAttnScoreMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultAttnL1NormMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultIndexScoreMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultIndexLseMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDqMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDwMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultDkMemory[16];
float ProblemPorts::kDefaultSmScale = 0.25F;

float ProblemPorts::kDefaultLossCoeff = 1;

float ProblemPorts::kDefaultGradLoss = 1;

std::int64_t ProblemPorts::kDefaultRatio = 1;

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

IndexerBwdDenseProblem makeProblem(const ProblemPorts& ports) {
    return IndexerBwdDenseProblem{
        makeDeviceTensor(ports.index_q, ports.index_q_dims, 4, ports.index_q_type, ports.index_q_index),
        makeDeviceTensor(ports.weights, ports.weights_dims, 3, ports.weights_type, ports.weights_index),
        makeDeviceTensor(ports.index_k, ports.index_k_dims, 3, ports.index_k_type, ports.index_k_index),
        makeDeviceTensor(ports.attn_score, ports.attn_score_dims, 3, ports.attn_score_type, ports.attn_score_index),
        makeDeviceTensor(ports.attn_l1norm, ports.attn_l1norm_dims, 2, ports.attn_l1norm_type, ports.attn_l1norm_index),
        makeDeviceTensor(ports.index_score, ports.index_score_dims, 3, ports.index_score_type, ports.index_score_index),
        makeDeviceTensor(ports.index_lse, ports.index_lse_dims, 2, ports.index_lse_type, ports.index_lse_index),
        makeDeviceTensor(ports.grad_index_score, ports.grad_index_score_dims, 3, ports.grad_index_score_type,
                         ports.grad_index_score_index),
        makeDeviceTensor(ports.dq, ports.dq_dims, 4, ports.dq_type, ports.dq_index),
        makeDeviceTensor(ports.dw, ports.dw_dims, 3, ports.dw_type, ports.dw_index),
        makeDeviceTensor(ports.dk, ports.dk_dims, 3, ports.dk_type, ports.dk_index),
        makeHostScalar(ports.sm_scale, ports.sm_scale_type, ports.sm_scale_index),
        makeHostScalar(ports.loss_coeff, ports.loss_coeff_type, ports.loss_coeff_index),
        makeHostScalar(ports.grad_loss, ports.grad_loss_type, ports.grad_loss_index),
        makeHostScalar(ports.ratio, ports.ratio_type, ports.ratio_index),
        IndexerBwdDenseAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Fp32IndexerBwdDenseTest, KeyIgnoresAllocationAddresses) {
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
    alignas(16) static unsigned char other_attn_l1norm[16];
    rebound.attn_l1norm = other_attn_l1norm;
    alignas(16) static unsigned char other_index_score[16];
    rebound.index_score = other_index_score;
    alignas(16) static unsigned char other_index_lse[16];
    rebound.index_lse = other_index_lse;
    alignas(16) static unsigned char other_dq[16];
    rebound.dq = other_dq;
    alignas(16) static unsigned char other_dw[16];
    rebound.dw = other_dw;
    alignas(16) static unsigned char other_dk[16];
    rebound.dk               = other_dk;
    float other_sm_scale     = 0.25F;
    rebound.sm_scale         = &other_sm_scale;
    float other_loss_coeff   = 1;
    rebound.loss_coeff       = &other_loss_coeff;
    float other_grad_loss    = 1;
    rebound.grad_loss        = &other_grad_loss;
    std::int64_t other_ratio = 1;
    rebound.ratio            = &other_ratio;
    rebound.grad_index_score = rebound.index_score;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Fp32IndexerBwdDenseTest, AcceptsContinuousAlignedOperands) {
    Fp32IndexerBwdDense primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp32IndexerBwdDense");
}

TEST(Fp32IndexerBwdDenseTest, RejectsUnsupportedComputeAndElementTypes) {
    Fp32IndexerBwdDense primitive;
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
    ProblemPorts unsupported_attn_l1norm;
    unsupported_attn_l1norm.attn_l1norm_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_attn_l1norm), kConstraints).isSuccess());
    ProblemPorts unsupported_index_score;
    unsupported_index_score.index_score_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_index_score), kConstraints).isSuccess());
    ProblemPorts unsupported_index_lse;
    unsupported_index_lse.index_lse_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_index_lse), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_index_score;
    unsupported_grad_index_score.grad_index_score_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_index_score), kConstraints).isSuccess());
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
    ProblemPorts unsupported_ratio;
    unsupported_ratio.ratio_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
}

TEST(Fp32IndexerBwdDenseTest, RejectsNonContinuousIndexTypes) {
    Fp32IndexerBwdDense primitive;
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
    ProblemPorts strided_attn_l1norm;
    strided_attn_l1norm.attn_l1norm_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_attn_l1norm), kConstraints).isSuccess());
    ProblemPorts strided_index_score;
    strided_index_score.index_score_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_index_score), kConstraints).isSuccess());
    ProblemPorts strided_index_lse;
    strided_index_lse.index_lse_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_index_lse), kConstraints).isSuccess());
    ProblemPorts strided_grad_index_score;
    strided_grad_index_score.grad_index_score_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_index_score), kConstraints).isSuccess());
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
    ProblemPorts strided_ratio;
    strided_ratio.ratio_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32IndexerBwdDenseTest, RejectsMisalignedAddresses) {
    Fp32IndexerBwdDense primitive;
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
    ProblemPorts misaligned_attn_l1norm;
    misaligned_attn_l1norm.attn_l1norm = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_attn_l1norm), kConstraints).isSuccess());
    ProblemPorts misaligned_index_score;
    misaligned_index_score.index_score = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_index_score), kConstraints).isSuccess());
    ProblemPorts misaligned_index_lse;
    misaligned_index_lse.index_lse = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_index_lse), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_index_score;
    misaligned_grad_index_score.grad_index_score = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_index_score), kConstraints).isSuccess());
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
    ProblemPorts misaligned_ratio;
    misaligned_ratio.ratio = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_ratio), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32IndexerBwdDenseTest, ChecksRequiredAndOptionalAddresses) {
    Fp32IndexerBwdDense primitive;
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
    ProblemPorts null_attn_l1norm;
    null_attn_l1norm.attn_l1norm = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_attn_l1norm), kConstraints).isSuccess());
    ProblemPorts null_index_score;
    null_index_score.index_score = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_index_score), kConstraints).isSuccess());
    ProblemPorts null_index_lse;
    null_index_lse.index_lse = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_index_lse), kConstraints).isSuccess());
    ProblemPorts null_grad_index_score;
    null_grad_index_score.grad_index_score = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_index_score), kConstraints).isSuccess());
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
    ProblemPorts null_ratio;
    null_ratio.ratio = nullptr;
    try {
        makeProblem(null_ratio);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32IndexerBwdDenseTest, ClonesIntoDistinctConfigurableCopies) {
    Fp32IndexerBwdDense prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp32IndexerBwdDense");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<IndexerBwdDenseProblem>&>(*first).configure(makeProblem(ports));
    alignas(16) unsigned char other_memory[16]{};
    {
        ProblemPorts other;
        other.index_q = other_memory;
        static_cast<Primitive<IndexerBwdDenseProblem>&>(*second).configure(makeProblem(other));
    }
    // Workspace checks alone do not prove pointer independence.
    // RebindsAddressesAndOutlivesOpsAndArgs covers execution with separate plans.
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Fp32IndexerBwdDenseTest, RequiresDeclaredAliasing) {
    ProblemPorts ports;

    auto problem = makeProblem(ports);
    Fp32IndexerBwdDense primitive;
    const auto original_key = problem.getProblemKey();
    alignas(16) static unsigned char separate_memory[16];

    ProblemPorts separate     = ports;
    separate.grad_index_score = separate_memory;
    const auto out_of_place   = makeProblem(separate);
    EXPECT_NE(original_key, out_of_place.getProblemKey());
    EXPECT_FALSE(primitive.isApplicable(out_of_place, Constraints{0}).isSuccess());
}

TEST(Fp32IndexerBwdDenseTest, RejectsUnsupportedGeometryWithoutLaunching) {
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
        Fp32IndexerBwdDense primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

TEST(Fp32IndexerBwdDenseTest, RejectsExcessiveLaunchSharedMemory) {
    {
        ProblemPorts ports;

        const std::int64_t index_k_dims[] = {ports.index_k_dims[0], 100000, ports.index_k_dims[2]};
        ports.index_k_dims                = index_k_dims;

        const std::int64_t attn_score_dims[] = {ports.attn_score_dims[0], ports.attn_score_dims[1], 100000};
        ports.attn_score_dims                = attn_score_dims;

        const std::int64_t index_score_dims[] = {ports.index_score_dims[0], ports.index_score_dims[1], 100000};
        ports.index_score_dims                = index_score_dims;

        const std::int64_t grad_index_score_dims[] = {ports.grad_index_score_dims[0], ports.grad_index_score_dims[1],
                                                      100000};
        ports.grad_index_score_dims                = grad_index_score_dims;

        const std::int64_t dk_dims[] = {ports.dk_dims[0], 100000, ports.dk_dims[2]};
        ports.dk_dims                = dk_dims;

        auto problem = makeProblem(ports);
        Fp32IndexerBwdDense primitive;

        {
            EXPECT_EQ(primitive.isApplicable(problem, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

}  // namespace ftrain
