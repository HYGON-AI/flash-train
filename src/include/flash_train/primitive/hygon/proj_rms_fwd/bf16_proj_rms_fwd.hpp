// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_FWD_BF16_PROJ_RMS_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_FWD_BF16_PROJ_RMS_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/proj_rms_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
std::uint64_t bf16ProjRmsFwdWorkspaceBytes(int rows, int outputs, int reduction, bool compute_h);
template<class Implementation>
bool launchBf16ProjRmsFwd(FTrainStream stream, const void* x, const void* weight, void* proj, void* r, int rows,
                          int outputs, int reduction, float eps, void* workspace);

// One dtype chain of the ProjRmsFwd family: the ported fused projection plus
// RMS factor, with caller-owned scratch for split-K.
template<class Implementation>
class Bf16ProjRmsFwdBase : public Primitive<ProjRmsFwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr int kSplit   = 1;
    static constexpr bool kSplit1 = false;
    static constexpr bool kSplit2 = false;
    static constexpr bool kSplit4 = false;
    static constexpr bool kSplit8 = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const ProjRmsFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const ProjRmsFwdProblem& problem) override;

  protected:
    Bf16ProjRmsFwdBase()                          = default;
    Bf16ProjRmsFwdBase(const Bf16ProjRmsFwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;

    std::uint64_t workspace_bytes_ = 0;
    const void* x_                 = nullptr;
    const void* weight_            = nullptr;
    void* proj_                    = nullptr;
    void* r_                       = nullptr;
    int rows_                      = 0;
    int outputs_                   = 0;
    int reduction_                 = 0;
    float eps_                     = 0.0F;
};

class Bf16ProjRmsFwd final : public Bf16ProjRmsFwdBase<Bf16ProjRmsFwd> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsFwd";
    static constexpr bool kSplit1      = true;
};

class Bf16ProjRmsFwdSplit2 final : public Bf16ProjRmsFwdBase<Bf16ProjRmsFwdSplit2> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsFwdSplit2";
    static constexpr bool kSplit2      = true;
    static constexpr int kSplit        = 2;
};

class Bf16ProjRmsFwdSplit4 final : public Bf16ProjRmsFwdBase<Bf16ProjRmsFwdSplit4> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsFwdSplit4";
    static constexpr bool kSplit4      = true;
    static constexpr int kSplit        = 4;
};

class Bf16ProjRmsFwdSplit8 final : public Bf16ProjRmsFwdBase<Bf16ProjRmsFwdSplit8> {
  public:
    static constexpr const char* kName = "Bf16ProjRmsFwdSplit8";
    static constexpr bool kSplit8      = true;
    static constexpr int kSplit        = 8;
};

}  // namespace ftrain

#endif
