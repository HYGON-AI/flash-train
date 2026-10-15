// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_H_POST_BDA_BWD_FP16_H_POST_BDA_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_H_POST_BDA_BWD_FP16_H_POST_BDA_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/h_post_bda_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp16_h_post_bda_bwd.hip.
bool launchFp16HPostBdaBwd(FTrainStream stream, const void* grad_output, const void* residual, const void* x,
                           const float* bias, const void* h_res, const void* h_post, void* grad_residual, void* grad_x,
                           void* grad_h_res, void* grad_h_post, float* grad_bias, int batch_stride, int num_streams,
                           int channels, bool has_bias);

// One dtype chain of the HPostBdaBwd family: the ported backward, whose single
// main launch writes all four operand gradients and whose optional second
// launch reduces the bias gradient.
class Fp16HPostBdaBwd final : public Primitive<HPostBdaBwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const HPostBdaBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const HPostBdaBwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* grad_output_ = nullptr;
    const void* residual_    = nullptr;
    const void* x_           = nullptr;
    const float* bias_       = nullptr;
    const void* h_res_       = nullptr;
    const void* h_post_      = nullptr;
    void* grad_residual_     = nullptr;
    void* grad_x_            = nullptr;
    void* grad_h_res_        = nullptr;
    void* grad_h_post_       = nullptr;
    float* grad_bias_        = nullptr;
    int batch_stride_        = 0;
    int num_streams_         = 0;
    int channels_            = 0;
    bool has_bias_           = false;
    bool reduce_bias_        = false;
};

}  // namespace ftrain

#endif
