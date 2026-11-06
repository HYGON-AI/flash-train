// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_COMPUTE_H_FWD_BF16_PROJ_RMS_COMPUTE_H_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_COMPUTE_H_FWD_BF16_PROJ_RMS_COMPUTE_H_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/proj_rms_compute_h_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
std::uint64_t bf16ProjRmsComputeHFwdWorkspaceBytes(int rows, int outputs, int reduction, bool compute_h);
template<class Implementation>
bool launchBf16ProjRmsComputeHFwd(FTrainStream stream, const void* x, const void* weight, void* proj, void* r,
                                  const float* alpha_pre, const float* alpha_post, const float* alpha_res,
                                  const float* bias, void* h_pre, void* h_post, void* h_res, int rows, int outputs,
                                  int num_streams, int reduction, float eps, float compute_h_eps, void* workspace);

// One dtype chain of the ProjRmsComputeHFwd family: the ported single launch
// that produces the projection, the RMS factor, and the activation split.
template<class Implementation>
class Bf16ProjRmsComputeHFwdBase : public Primitive<ProjRmsComputeHFwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr int kSplit                = 1;
    static constexpr bool kSplit1              = false;
    static constexpr bool kSplit2              = false;
    static constexpr bool kSplit4              = false;
    static constexpr bool kSplit8              = false;
    static constexpr bool kV2                  = false;
    static constexpr bool kV2ConflictFree      = false;
    static constexpr bool kV2FullK             = false;
    static constexpr bool kV2ConflictFreeFullK = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const ProjRmsComputeHFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const ProjRmsComputeHFwdProblem& problem) override;

  protected:
    Bf16ProjRmsComputeHFwdBase()                                  = default;
    Bf16ProjRmsComputeHFwdBase(const Bf16ProjRmsComputeHFwdBase&) = default;

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

class Bf16ProjRmsComputeHFwdV2FullK final : public Bf16ProjRmsComputeHFwdBase<Bf16ProjRmsComputeHFwdV2FullK> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsComputeHFwdV2FullK";
    static constexpr bool kV2FullK     = true;
};

class Bf16ProjRmsComputeHFwdV2ConflictFreeFullK final
    : public Bf16ProjRmsComputeHFwdBase<Bf16ProjRmsComputeHFwdV2ConflictFreeFullK> {
  public:
    static constexpr const char* kName         = "Bf16ProjRmsComputeHFwdV2ConflictFreeFullK";
    static constexpr bool kV2ConflictFreeFullK = true;
};

class Bf16ProjRmsComputeHFwd final : public Bf16ProjRmsComputeHFwdBase<Bf16ProjRmsComputeHFwd> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsComputeHFwd";
    static constexpr bool kSplit1      = true;
};

class Bf16ProjRmsComputeHFwdSplit2 final : public Bf16ProjRmsComputeHFwdBase<Bf16ProjRmsComputeHFwdSplit2> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsComputeHFwdSplit2";
    static constexpr bool kSplit2      = true;
    static constexpr int kSplit        = 2;
};

class Bf16ProjRmsComputeHFwdSplit4 final : public Bf16ProjRmsComputeHFwdBase<Bf16ProjRmsComputeHFwdSplit4> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsComputeHFwdSplit4";
    static constexpr bool kSplit4      = true;
    static constexpr int kSplit        = 4;
};

class Bf16ProjRmsComputeHFwdSplit8 final : public Bf16ProjRmsComputeHFwdBase<Bf16ProjRmsComputeHFwdSplit8> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsComputeHFwdSplit8";
    static constexpr bool kSplit8      = true;
    static constexpr int kSplit        = 8;
};

class Bf16ProjRmsComputeHFwdV2 final : public Bf16ProjRmsComputeHFwdBase<Bf16ProjRmsComputeHFwdV2> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsComputeHFwdV2";
    static constexpr bool kV2          = true;
};

class Bf16ProjRmsComputeHFwdV2ConflictFree final
    : public Bf16ProjRmsComputeHFwdBase<Bf16ProjRmsComputeHFwdV2ConflictFree> {
  public:
    static constexpr const char* kName    = "Bf16ProjRmsComputeHFwdV2ConflictFree";
    static constexpr bool kV2ConflictFree = true;
};

}  // namespace ftrain

#endif
