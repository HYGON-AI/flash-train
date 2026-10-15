// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_COMPUTE_H_FWD_FP32_PROJ_RMS_COMPUTE_H_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_COMPUTE_H_FWD_FP32_PROJ_RMS_COMPUTE_H_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/proj_rms_compute_h_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp32_proj_rms_compute_h_fwd.hip.
std::uint64_t fp32ProjRmsComputeHFwdWorkspaceBytes(int rows, int outputs, int reduction, bool compute_h);
bool launchFp32ProjRmsComputeHFwd(FTrainStream stream, const void* x, const void* weight, void* proj, void* r,
                                  const float* alpha_pre, const float* alpha_post, const float* alpha_res,
                                  const float* bias, void* h_pre, void* h_post, void* h_res, int rows, int outputs,
                                  int num_streams, int reduction, float eps, float compute_h_eps, void* workspace);

// One dtype chain of the ProjRmsComputeHFwd family: the ported single launch
// that produces the projection, the RMS factor, and the activation split.
class Fp32ProjRmsComputeHFwd final : public Primitive<ProjRmsComputeHFwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const ProjRmsComputeHFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const ProjRmsComputeHFwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    std::uint64_t workspace_bytes_ = 0;
    const void* x_                 = nullptr;
    const void* weight_            = nullptr;
    const float* alpha_pre_        = nullptr;
    const float* alpha_post_       = nullptr;
    const float* alpha_res_        = nullptr;
    const float* bias_             = nullptr;
    void* h_pre_                   = nullptr;
    void* h_post_                  = nullptr;
    void* h_res_                   = nullptr;
    void* r_                       = nullptr;
    void* proj_                    = nullptr;
    int rows_                      = 0;
    int outputs_                   = 0;
    int num_streams_               = 0;
    int reduction_                 = 0;
    float eps_                     = 0.0F;
    float compute_h_eps_           = 0.0F;
};

}  // namespace ftrain

#endif
