// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_BWD_BF16_PROJ_RMS_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_BWD_BF16_PROJ_RMS_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/proj_rms_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in bf16_proj_rms_bwd.hip.
template<class Implementation>
bool launchBf16ProjRmsBwd(FTrainStream stream, const void* grad_proj, const void* grad_r, const void* x,
                          const void* weight, const void* r, void* grad_x, void* grad_weight, int rows, int outputs,
                          int reduction, float eps);

// One dtype chain of the ProjRmsBwd family: the ported backward that produces
// both the input and the weight gradient in one launch, no workspace.
template<class Implementation>
class Bf16ProjRmsBwdBase : public Primitive<ProjRmsBwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kPortable = false;
    static constexpr bool kMmac     = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const ProjRmsBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const ProjRmsBwdProblem& problem) override;

  protected:
    Bf16ProjRmsBwdBase()                          = default;
    Bf16ProjRmsBwdBase(const Bf16ProjRmsBwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;

    const void* grad_proj_ = nullptr;
    const void* grad_r_    = nullptr;
    const void* x_         = nullptr;
    const void* weight_    = nullptr;
    const void* r_         = nullptr;
    void* grad_x_          = nullptr;
    void* grad_weight_     = nullptr;
    int rows_              = 0;
    int outputs_           = 0;
    int reduction_         = 0;
    float eps_             = 0.0F;
};

class Bf16ProjRmsBwd final : public Bf16ProjRmsBwdBase<Bf16ProjRmsBwd> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsBwd";
    static constexpr bool kPortable    = true;
};

class Bf16ProjRmsBwdMmac final : public Bf16ProjRmsBwdBase<Bf16ProjRmsBwdMmac> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsBwdMmac";
    static constexpr bool kMmac        = true;
};

}  // namespace ftrain

#endif
