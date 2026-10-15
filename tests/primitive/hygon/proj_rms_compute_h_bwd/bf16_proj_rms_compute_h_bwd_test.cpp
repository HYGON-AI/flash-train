// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/family/proj_rms_compute_h_bwd/problem.hpp"
#include "flash_train/primitive/hygon/proj_rms_compute_h_bwd/bf16_proj_rms_compute_h_bwd.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kGradHPreDims[2] = {16, 2};
constexpr std::int64_t kGradHPostDims[2] = {16, 2};
constexpr std::int64_t kGradHResDims[2] = {16, 4};
constexpr std::int64_t kGradRDims[2] = {16, 1};
constexpr std::int64_t kHPreDims[2] = {16, 2};
constexpr std::int64_t kHPostDims[2] = {16, 2};
constexpr std::int64_t kProjDims[2] = {16, 8};
constexpr std::int64_t kRDims[2] = {16, 1};
constexpr std::int64_t kXDims[2] = {16, 32};
constexpr std::int64_t kWeightDims[2] = {8, 32};
constexpr std::int64_t kAlphaPreDims[1] = {1};
constexpr std::int64_t kAlphaPostDims[1] = {1};
constexpr std::int64_t kAlphaResDims[1] = {1};
constexpr std::int64_t kGradXDims[2] = {16, 32};
constexpr std::int64_t kGradWeightDims[2] = {8, 32};
constexpr std::int64_t kGradAlphaDims[1] = {3};
constexpr std::int64_t kGradBiasDims[1] = {8};

struct ProblemPorts {
    void* grad_h_pre                     = kDefaultGradHPreMemory;
    const std::int64_t* grad_h_pre_dims  = kGradHPreDims;
    FTrainNumericType grad_h_pre_type    = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_h_pre_index     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_h_post                    = kDefaultGradHPostMemory;
    const std::int64_t* grad_h_post_dims = kGradHPostDims;
    FTrainNumericType grad_h_post_type   = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_h_post_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_h_res                     = kDefaultGradHResMemory;
    const std::int64_t* grad_h_res_dims  = kGradHResDims;
    FTrainNumericType grad_h_res_type    = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_h_res_index     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_r                         = kDefaultGradRMemory;
    const std::int64_t* grad_r_dims      = kGradRDims;
    FTrainNumericType grad_r_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_r_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* h_pre                          = kDefaultHPreMemory;
    const std::int64_t* h_pre_dims       = kHPreDims;
    FTrainNumericType h_pre_type         = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType h_pre_index          = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* h_post                         = kDefaultHPostMemory;
    const std::int64_t* h_post_dims      = kHPostDims;
    FTrainNumericType h_post_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType h_post_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* proj                           = kDefaultProjMemory;
    const std::int64_t* proj_dims        = kProjDims;
    FTrainNumericType proj_type          = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType proj_index           = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* r                              = kDefaultRMemory;
    const std::int64_t* r_dims           = kRDims;
    FTrainNumericType r_type             = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType r_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* x                              = kDefaultXMemory;
    const std::int64_t* x_dims           = kXDims;
    FTrainNumericType x_type             = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType x_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* weight                         = kDefaultWeightMemory;
    const std::int64_t* weight_dims      = kWeightDims;
    FTrainNumericType weight_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType weight_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* alpha_pre                      = kDefaultAlphaPreMemory;
    const std::int64_t* alpha_pre_dims   = kAlphaPreDims;
    FTrainNumericType alpha_pre_type     = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType alpha_pre_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* alpha_post                     = kDefaultAlphaPostMemory;
    const std::int64_t* alpha_post_dims  = kAlphaPostDims;
    FTrainNumericType alpha_post_type    = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType alpha_post_index     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* alpha_res                      = kDefaultAlphaResMemory;
    const std::int64_t* alpha_res_dims   = kAlphaResDims;
    FTrainNumericType alpha_res_type     = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType alpha_res_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_x                         = kDefaultGradXMemory;
    const std::int64_t* grad_x_dims      = kGradXDims;
    FTrainNumericType grad_x_type        = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_x_index         = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_weight                    = kDefaultGradWeightMemory;
    const std::int64_t* grad_weight_dims = kGradWeightDims;
    FTrainNumericType grad_weight_type   = FTRAIN_NUMERIC_TYPE_BF16;
    FTrainIndexType grad_weight_index    = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_alpha                     = kDefaultGradAlphaMemory;
    const std::int64_t* grad_alpha_dims  = kGradAlphaDims;
    FTrainNumericType grad_alpha_type    = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_alpha_index     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* grad_bias                      = kDefaultGradBiasMemory;
    const std::int64_t* grad_bias_dims   = kGradBiasDims;
    FTrainNumericType grad_bias_type     = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType grad_bias_index      = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* n                              = &kDefaultN;
    FTrainNumericType n_type             = FTRAIN_NUMERIC_TYPE_INT64;
    FTrainIndexType n_index              = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* eps                            = &kDefaultEps;
    FTrainNumericType eps_type           = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType eps_index            = FTRAIN_INDEX_TYPE_CONTINUOUS;
    void* compute_h_eps                  = &kDefaultComputeHEps;
    FTrainNumericType compute_h_eps_type = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType compute_h_eps_index  = FTRAIN_INDEX_TYPE_CONTINUOUS;
    FTrainNumericType compute_type       = FTRAIN_NUMERIC_TYPE_BF16;
    alignas(16) static unsigned char kDefaultGradHPreMemory[16];
    alignas(16) static unsigned char kDefaultGradHPostMemory[16];
    alignas(16) static unsigned char kDefaultGradHResMemory[16];
    alignas(16) static unsigned char kDefaultGradRMemory[16];
    alignas(16) static unsigned char kDefaultHPreMemory[16];
    alignas(16) static unsigned char kDefaultHPostMemory[16];
    alignas(16) static unsigned char kDefaultProjMemory[16];
    alignas(16) static unsigned char kDefaultRMemory[16];
    alignas(16) static unsigned char kDefaultXMemory[16];
    alignas(16) static unsigned char kDefaultWeightMemory[16];
    alignas(16) static unsigned char kDefaultAlphaPreMemory[16];
    alignas(16) static unsigned char kDefaultAlphaPostMemory[16];
    alignas(16) static unsigned char kDefaultAlphaResMemory[16];
    alignas(16) static unsigned char kDefaultGradXMemory[16];
    alignas(16) static unsigned char kDefaultGradWeightMemory[16];
    alignas(16) static unsigned char kDefaultGradAlphaMemory[16];
    alignas(16) static unsigned char kDefaultGradBiasMemory[16];
    static std::int64_t kDefaultN;
    static float kDefaultEps;
    static float kDefaultComputeHEps;
};

alignas(16) unsigned char ProblemPorts::kDefaultGradHPreMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradHPostMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradHResMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradRMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultHPreMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultHPostMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultProjMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultRMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultXMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultWeightMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultAlphaPreMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultAlphaPostMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultAlphaResMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradXMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradWeightMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradAlphaMemory[16];
alignas(16) unsigned char ProblemPorts::kDefaultGradBiasMemory[16];

std::int64_t ProblemPorts::kDefaultN = 2;

float ProblemPorts::kDefaultEps = 1e-06;

float ProblemPorts::kDefaultComputeHEps = 1e-06;

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

ProjRmsComputeHBwdProblem makeProblem(const ProblemPorts& ports) {
    return ProjRmsComputeHBwdProblem{
        makeDeviceTensor(ports.grad_h_pre, ports.grad_h_pre_dims, 2, ports.grad_h_pre_type, ports.grad_h_pre_index),
        makeDeviceTensor(ports.grad_h_post, ports.grad_h_post_dims, 2, ports.grad_h_post_type, ports.grad_h_post_index),
        makeDeviceTensor(ports.grad_h_res, ports.grad_h_res_dims, 2, ports.grad_h_res_type, ports.grad_h_res_index),
        makeDeviceTensor(ports.grad_r, ports.grad_r_dims, 2, ports.grad_r_type, ports.grad_r_index),
        makeDeviceTensor(ports.h_pre, ports.h_pre_dims, 2, ports.h_pre_type, ports.h_pre_index),
        makeDeviceTensor(ports.h_post, ports.h_post_dims, 2, ports.h_post_type, ports.h_post_index),
        makeDeviceTensor(ports.proj, ports.proj_dims, 2, ports.proj_type, ports.proj_index),
        makeDeviceTensor(ports.r, ports.r_dims, 2, ports.r_type, ports.r_index),
        makeDeviceTensor(ports.x, ports.x_dims, 2, ports.x_type, ports.x_index),
        makeDeviceTensor(ports.weight, ports.weight_dims, 2, ports.weight_type, ports.weight_index),
        makeDeviceTensor(ports.alpha_pre, ports.alpha_pre_dims, 1, ports.alpha_pre_type, ports.alpha_pre_index),
        makeDeviceTensor(ports.alpha_post, ports.alpha_post_dims, 1, ports.alpha_post_type, ports.alpha_post_index),
        makeDeviceTensor(ports.alpha_res, ports.alpha_res_dims, 1, ports.alpha_res_type, ports.alpha_res_index),
        makeDeviceTensor(ports.grad_x, ports.grad_x_dims, 2, ports.grad_x_type, ports.grad_x_index),
        makeDeviceTensor(ports.grad_weight, ports.grad_weight_dims, 2, ports.grad_weight_type, ports.grad_weight_index),
        makeDeviceTensor(ports.grad_alpha, ports.grad_alpha_dims, 1, ports.grad_alpha_type, ports.grad_alpha_index),
        makeDeviceTensor(ports.grad_bias, ports.grad_bias_dims, 1, ports.grad_bias_type, ports.grad_bias_index),
        makeHostScalar(ports.n, ports.n_type, ports.n_index),
        makeHostScalar(ports.eps, ports.eps_type, ports.eps_index),
        makeHostScalar(ports.compute_h_eps, ports.compute_h_eps_type, ports.compute_h_eps_index),
        ProjRmsComputeHBwdAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Bf16ProjRmsComputeHBwdTest, KeyIgnoresAllocationAddresses) {
    const auto original_key = makeProblem(ProblemPorts{}).getProblemKey();
    ProblemPorts rebound;
    alignas(16) static unsigned char other_grad_h_pre[16];
    rebound.grad_h_pre = other_grad_h_pre;
    alignas(16) static unsigned char other_grad_h_post[16];
    rebound.grad_h_post = other_grad_h_post;
    alignas(16) static unsigned char other_grad_h_res[16];
    rebound.grad_h_res = other_grad_h_res;
    alignas(16) static unsigned char other_grad_r[16];
    rebound.grad_r = other_grad_r;
    alignas(16) static unsigned char other_h_pre[16];
    rebound.h_pre = other_h_pre;
    alignas(16) static unsigned char other_h_post[16];
    rebound.h_post = other_h_post;
    alignas(16) static unsigned char other_proj[16];
    rebound.proj = other_proj;
    alignas(16) static unsigned char other_r[16];
    rebound.r = other_r;
    alignas(16) static unsigned char other_x[16];
    rebound.x = other_x;
    alignas(16) static unsigned char other_weight[16];
    rebound.weight = other_weight;
    alignas(16) static unsigned char other_alpha_pre[16];
    rebound.alpha_pre = other_alpha_pre;
    alignas(16) static unsigned char other_alpha_post[16];
    rebound.alpha_post = other_alpha_post;
    alignas(16) static unsigned char other_alpha_res[16];
    rebound.alpha_res = other_alpha_res;
    alignas(16) static unsigned char other_grad_x[16];
    rebound.grad_x = other_grad_x;
    alignas(16) static unsigned char other_grad_weight[16];
    rebound.grad_weight = other_grad_weight;
    alignas(16) static unsigned char other_grad_alpha[16];
    rebound.grad_alpha = other_grad_alpha;
    alignas(16) static unsigned char other_grad_bias[16];
    rebound.grad_bias         = other_grad_bias;
    std::int64_t other_n      = 2;
    rebound.n                 = &other_n;
    float other_eps           = 1e-06;
    rebound.eps               = &other_eps;
    float other_compute_h_eps = 1e-06;
    rebound.compute_h_eps     = &other_compute_h_eps;
    EXPECT_EQ(original_key, makeProblem(rebound).getProblemKey());
}

TEST(Bf16ProjRmsComputeHBwdTest, AcceptsContinuousAlignedOperands) {
    Bf16ProjRmsComputeHBwd primitive;
    ProblemPorts ports;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ports), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Bf16ProjRmsComputeHBwd");
}

TEST(Bf16ProjRmsComputeHBwdTest, RejectsUnsupportedComputeAndElementTypes) {
    Bf16ProjRmsComputeHBwd primitive;
    ProblemPorts unsupported_compute;
    unsupported_compute.compute_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_h_pre;
    unsupported_grad_h_pre.grad_h_pre_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_h_pre), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_h_post;
    unsupported_grad_h_post.grad_h_post_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_h_post), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_h_res;
    unsupported_grad_h_res.grad_h_res_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_h_res), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_r;
    unsupported_grad_r.grad_r_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_r), kConstraints).isSuccess());
    ProblemPorts unsupported_h_pre;
    unsupported_h_pre.h_pre_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_h_pre), kConstraints).isSuccess());
    ProblemPorts unsupported_h_post;
    unsupported_h_post.h_post_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_h_post), kConstraints).isSuccess());
    ProblemPorts unsupported_proj;
    unsupported_proj.proj_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_proj), kConstraints).isSuccess());
    ProblemPorts unsupported_r;
    unsupported_r.r_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_r), kConstraints).isSuccess());
    ProblemPorts unsupported_x;
    unsupported_x.x_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_x), kConstraints).isSuccess());
    ProblemPorts unsupported_weight;
    unsupported_weight.weight_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_weight), kConstraints).isSuccess());
    ProblemPorts unsupported_alpha_pre;
    unsupported_alpha_pre.alpha_pre_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_alpha_pre), kConstraints).isSuccess());
    ProblemPorts unsupported_alpha_post;
    unsupported_alpha_post.alpha_post_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_alpha_post), kConstraints).isSuccess());
    ProblemPorts unsupported_alpha_res;
    unsupported_alpha_res.alpha_res_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_alpha_res), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_x;
    unsupported_grad_x.grad_x_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_x), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_weight;
    unsupported_grad_weight.grad_weight_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_weight), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_alpha;
    unsupported_grad_alpha.grad_alpha_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_alpha), kConstraints).isSuccess());
    ProblemPorts unsupported_grad_bias;
    unsupported_grad_bias.grad_bias_type = FTRAIN_NUMERIC_TYPE_FP64;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_grad_bias), kConstraints).isSuccess());
    ProblemPorts unsupported_n;
    unsupported_n.n_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_n), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_eps;
    unsupported_eps.eps_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
    ProblemPorts unsupported_compute_h_eps;
    unsupported_compute_h_eps.compute_h_eps_type = FTRAIN_NUMERIC_TYPE_FP64;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(unsupported_compute_h_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_TRUE(e.getResult().getStatus() == FTRAIN_STATUS_INVALID_ARGUMENT ||
                    e.getResult().getStatus() == FTRAIN_STATUS_OVERFLOW);
    }
}

TEST(Bf16ProjRmsComputeHBwdTest, RejectsNonContinuousIndexTypes) {
    Bf16ProjRmsComputeHBwd primitive;
    ProblemPorts strided_grad_h_pre;
    strided_grad_h_pre.grad_h_pre_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_h_pre), kConstraints).isSuccess());
    ProblemPorts strided_grad_h_post;
    strided_grad_h_post.grad_h_post_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_h_post), kConstraints).isSuccess());
    ProblemPorts strided_grad_h_res;
    strided_grad_h_res.grad_h_res_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_h_res), kConstraints).isSuccess());
    ProblemPorts strided_grad_r;
    strided_grad_r.grad_r_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_r), kConstraints).isSuccess());
    ProblemPorts strided_h_pre;
    strided_h_pre.h_pre_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_h_pre), kConstraints).isSuccess());
    ProblemPorts strided_h_post;
    strided_h_post.h_post_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_h_post), kConstraints).isSuccess());
    ProblemPorts strided_proj;
    strided_proj.proj_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_proj), kConstraints).isSuccess());
    ProblemPorts strided_r;
    strided_r.r_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_r), kConstraints).isSuccess());
    ProblemPorts strided_x;
    strided_x.x_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_x), kConstraints).isSuccess());
    ProblemPorts strided_weight;
    strided_weight.weight_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_weight), kConstraints).isSuccess());
    ProblemPorts strided_alpha_pre;
    strided_alpha_pre.alpha_pre_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_alpha_pre), kConstraints).isSuccess());
    ProblemPorts strided_alpha_post;
    strided_alpha_post.alpha_post_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_alpha_post), kConstraints).isSuccess());
    ProblemPorts strided_alpha_res;
    strided_alpha_res.alpha_res_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_alpha_res), kConstraints).isSuccess());
    ProblemPorts strided_grad_x;
    strided_grad_x.grad_x_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_x), kConstraints).isSuccess());
    ProblemPorts strided_grad_weight;
    strided_grad_weight.grad_weight_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_weight), kConstraints).isSuccess());
    ProblemPorts strided_grad_alpha;
    strided_grad_alpha.grad_alpha_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_alpha), kConstraints).isSuccess());
    ProblemPorts strided_grad_bias;
    strided_grad_bias.grad_bias_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_grad_bias), kConstraints).isSuccess());
    ProblemPorts strided_n;
    strided_n.n_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_n), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_eps;
    strided_eps.eps_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts strided_compute_h_eps;
    strided_compute_h_eps.compute_h_eps_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_compute_h_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16ProjRmsComputeHBwdTest, RejectsMisalignedAddresses) {
    Bf16ProjRmsComputeHBwd primitive;
    alignas(16) static unsigned char misaligned[16];
    ProblemPorts misaligned_grad_h_pre;
    misaligned_grad_h_pre.grad_h_pre = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_h_pre), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_h_post;
    misaligned_grad_h_post.grad_h_post = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_h_post), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_h_res;
    misaligned_grad_h_res.grad_h_res = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_h_res), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_r;
    misaligned_grad_r.grad_r = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_r), kConstraints).isSuccess());
    ProblemPorts misaligned_h_pre;
    misaligned_h_pre.h_pre = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_h_pre), kConstraints).isSuccess());
    ProblemPorts misaligned_h_post;
    misaligned_h_post.h_post = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_h_post), kConstraints).isSuccess());
    ProblemPorts misaligned_proj;
    misaligned_proj.proj = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_proj), kConstraints).isSuccess());
    ProblemPorts misaligned_r;
    misaligned_r.r = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_r), kConstraints).isSuccess());
    ProblemPorts misaligned_x;
    misaligned_x.x = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_x), kConstraints).isSuccess());
    ProblemPorts misaligned_weight;
    misaligned_weight.weight = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_weight), kConstraints).isSuccess());
    ProblemPorts misaligned_alpha_pre;
    misaligned_alpha_pre.alpha_pre = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_alpha_pre), kConstraints).isSuccess());
    ProblemPorts misaligned_alpha_post;
    misaligned_alpha_post.alpha_post = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_alpha_post), kConstraints).isSuccess());
    ProblemPorts misaligned_alpha_res;
    misaligned_alpha_res.alpha_res = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_alpha_res), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_x;
    misaligned_grad_x.grad_x = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_x), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_weight;
    misaligned_grad_weight.grad_weight = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_weight), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_alpha;
    misaligned_grad_alpha.grad_alpha = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_alpha), kConstraints).isSuccess());
    ProblemPorts misaligned_grad_bias;
    misaligned_grad_bias.grad_bias = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_grad_bias), kConstraints).isSuccess());
    ProblemPorts misaligned_n;
    misaligned_n.n = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_n), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_eps;
    misaligned_eps.eps = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts misaligned_compute_h_eps;
    misaligned_compute_h_eps.compute_h_eps = misaligned + 1;
    try {
        EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_compute_h_eps), kConstraints).isSuccess());
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16ProjRmsComputeHBwdTest, ChecksRequiredAndOptionalAddresses) {
    Bf16ProjRmsComputeHBwd primitive;
    ProblemPorts null_grad_h_pre;
    null_grad_h_pre.grad_h_pre = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_h_pre), kConstraints).isSuccess());
    ProblemPorts null_grad_h_post;
    null_grad_h_post.grad_h_post = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_h_post), kConstraints).isSuccess());
    ProblemPorts null_grad_h_res;
    null_grad_h_res.grad_h_res = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_h_res), kConstraints).isSuccess());
    ProblemPorts null_grad_r;
    null_grad_r.grad_r = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_r), kConstraints).isSuccess());
    ProblemPorts null_h_pre;
    null_h_pre.h_pre = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_h_pre), kConstraints).isSuccess());
    ProblemPorts null_h_post;
    null_h_post.h_post = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_h_post), kConstraints).isSuccess());
    ProblemPorts null_proj;
    null_proj.proj = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_proj), kConstraints).isSuccess());
    ProblemPorts null_r;
    null_r.r = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_r), kConstraints).isSuccess());
    ProblemPorts null_x;
    null_x.x = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_x), kConstraints).isSuccess());
    ProblemPorts null_weight;
    null_weight.weight = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_weight), kConstraints).isSuccess());
    ProblemPorts null_alpha_pre;
    null_alpha_pre.alpha_pre = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_alpha_pre), kConstraints).isSuccess());
    ProblemPorts null_alpha_post;
    null_alpha_post.alpha_post = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_alpha_post), kConstraints).isSuccess());
    ProblemPorts null_alpha_res;
    null_alpha_res.alpha_res = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_alpha_res), kConstraints).isSuccess());
    ProblemPorts null_grad_x;
    null_grad_x.grad_x = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_x), kConstraints).isSuccess());
    ProblemPorts null_grad_weight;
    null_grad_weight.grad_weight = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_weight), kConstraints).isSuccess());
    ProblemPorts null_grad_alpha;
    null_grad_alpha.grad_alpha = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_alpha), kConstraints).isSuccess());
    ProblemPorts null_grad_bias;
    null_grad_bias.grad_bias = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_grad_bias), kConstraints).isSuccess());
    ProblemPorts null_n;
    null_n.n = nullptr;
    try {
        makeProblem(null_n);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_eps;
    null_eps.eps = nullptr;
    try {
        makeProblem(null_eps);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
    ProblemPorts null_compute_h_eps;
    null_compute_h_eps.compute_h_eps = nullptr;
    try {
        makeProblem(null_compute_h_eps);
        FAIL() << "null host scalar";
    }
    catch (const Exception& e) {
        EXPECT_EQ(e.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Bf16ProjRmsComputeHBwdTest, ClonesIntoIndependentConfigurableCopies) {
    Bf16ProjRmsComputeHBwd prototype;
    ProblemPorts ports;
    prototype.configure(makeProblem(ports));
    const auto workspace_bytes            = prototype.getRequiredWorkspaceBytes();
    std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Bf16ProjRmsComputeHBwd");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    static_cast<Primitive<ProjRmsComputeHBwdProblem>&>(*first).configure(makeProblem(ports));
    {
        ProblemPorts other;
        static_cast<Primitive<ProjRmsComputeHBwdProblem>&>(*second).configure(makeProblem(other));
    }
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(second->getRequiredWorkspaceBytes(), workspace_bytes);
    EXPECT_EQ(prototype.getRequiredWorkspaceBytes(), workspace_bytes);
}

TEST(Bf16ProjRmsComputeHBwdTest, RejectsUnsupportedGeometryWithoutLaunching) {
    {
        ProblemPorts ports;

        const auto original                   = makeProblem(ports).getProblemKey();
        const std::int64_t x_dims[]           = {16, 33};
        ports.x_dims                          = x_dims;
        const std::int64_t weight_dims[]      = {8, 33};
        ports.weight_dims                     = weight_dims;
        const std::int64_t grad_x_dims[]      = {16, 33};
        ports.grad_x_dims                     = grad_x_dims;
        const std::int64_t grad_weight_dims[] = {8, 33};
        ports.grad_weight_dims                = grad_weight_dims;

        const auto unsupported = makeProblem(ports);
        EXPECT_NE(original, unsupported.getProblemKey());
        Bf16ProjRmsComputeHBwd primitive;

        {
            EXPECT_EQ(primitive.isApplicable(unsupported, Constraints{0}).getStatus(), FTRAIN_STATUS_UNSUPPORTED)
                << primitive.getName();
        }
    }
}

}  // namespace ftrain
