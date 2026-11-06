// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_COMPUTE_H_BWD_FP32_PROJ_RMS_COMPUTE_H_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_COMPUTE_H_BWD_FP32_PROJ_RMS_COMPUTE_H_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/proj_rms_compute_h_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp32_proj_rms_compute_h_bwd.hip.
template<class Implementation>
std::uint64_t fp32ProjRmsComputeHBwdWorkspaceFloats(int rows, int outputs, int reduction);
void fp32ProjRmsComputeHBwdZeroMemory(FTrainStream stream, void* memory, std::uint64_t bytes);
template<class Implementation>
bool launchFp32ProjRmsComputeHBwd(FTrainStream stream, const void* grad_h_pre, const void* grad_h_post,
                                  const void* grad_h_res, const void* grad_r, const void* h_pre, const void* h_post,
                                  const void* proj, const void* r, const void* x, const void* weight,
                                  const float* alpha_pre, const float* alpha_post, const float* alpha_res, void* grad_x,
                                  void* grad_weight, float* grad_alpha, float* grad_bias, float* workspace, int rows,
                                  int outputs, int num_streams, int reduction, float eps, float compute_h_eps);

// One dtype chain of the ProjRmsComputeHBwd family: the ported backward, whose
// gradient preparation needs a caller-owned float scratch the kernel sizes
// from the problem, so the Primitive reserves it here.
template<class Implementation>
class Fp32ProjRmsComputeHBwdBase : public Primitive<ProjRmsComputeHBwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kPortable  = false;
    static constexpr bool kMmac      = false;
    static constexpr bool kMmacGradX = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const ProjRmsComputeHBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const ProjRmsComputeHBwdProblem& problem) override;

  protected:
    Fp32ProjRmsComputeHBwdBase()                                  = default;
    Fp32ProjRmsComputeHBwdBase(const Fp32ProjRmsComputeHBwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;

    const void* grad_h_pre_        = nullptr;
    const void* grad_h_post_       = nullptr;
    const void* grad_h_res_        = nullptr;
    const void* grad_r_            = nullptr;
    const void* h_pre_             = nullptr;
    const void* h_post_            = nullptr;
    const void* proj_              = nullptr;
    const void* r_                 = nullptr;
    const void* x_                 = nullptr;
    const void* weight_            = nullptr;
    const float* alpha_pre_        = nullptr;
    const float* alpha_post_       = nullptr;
    const float* alpha_res_        = nullptr;
    void* grad_x_                  = nullptr;
    void* grad_weight_             = nullptr;
    float* grad_alpha_             = nullptr;
    float* grad_bias_              = nullptr;
    int rows_                      = 0;
    int outputs_                   = 0;
    int num_streams_               = 0;
    int reduction_                 = 0;
    float eps_                     = 0.0F;
    float compute_h_eps_           = 0.0F;
    std::uint64_t workspace_bytes_ = 0;
};

class Fp32ProjRmsComputeHBwdMmacGradX final : public Fp32ProjRmsComputeHBwdBase<Fp32ProjRmsComputeHBwdMmacGradX> {
  public:
    static constexpr const char* kName = "Fp32ProjRmsComputeHBwdMmacGradX";
    static constexpr bool kMmacGradX   = true;
};

class Fp32ProjRmsComputeHBwd final : public Fp32ProjRmsComputeHBwdBase<Fp32ProjRmsComputeHBwd> {
  public:
    static constexpr const char* kName = "Fp32ProjRmsComputeHBwd";
    static constexpr bool kPortable    = true;
};

class Fp32ProjRmsComputeHBwdMmac final : public Fp32ProjRmsComputeHBwdBase<Fp32ProjRmsComputeHBwdMmac> {
  public:
    static constexpr const char* kName = "Fp32ProjRmsComputeHBwdMmac";
    static constexpr bool kMmac        = true;
};

}  // namespace ftrain

#endif
