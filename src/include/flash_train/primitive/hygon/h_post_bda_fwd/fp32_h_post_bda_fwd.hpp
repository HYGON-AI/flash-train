// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_H_POST_BDA_FWD_FP32_H_POST_BDA_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_H_POST_BDA_FWD_FP32_H_POST_BDA_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/h_post_bda_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp32_h_post_bda_fwd.hip.
bool launchFp32HPostBdaFwd(FTrainStream stream, const void* h_res, const void* residual, const void* h_post,
                           const void* x, const float* bias, void* output, int batch_rows, int num_streams,
                           int channels, bool has_bias);

// One dtype chain of the HPostBdaFwd family: the ported residual-mixing and
// expansion forward, no workspace.
class Fp32HPostBdaFwd final : public Primitive<HPostBdaFwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const HPostBdaFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const HPostBdaFwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* h_res_    = nullptr;
    const void* residual_ = nullptr;
    const void* h_post_   = nullptr;
    const void* x_        = nullptr;
    const float* bias_    = nullptr;
    void* output_         = nullptr;
    int batch_stride_     = 0;
    int num_streams_      = 0;
    int channels_         = 0;
    bool has_bias_        = false;
};

}  // namespace ftrain

#endif
