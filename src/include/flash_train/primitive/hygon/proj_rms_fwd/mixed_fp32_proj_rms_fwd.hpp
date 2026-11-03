// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_FWD_MIXED_FP32_PROJ_RMS_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_FWD_MIXED_FP32_PROJ_RMS_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/proj_rms_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
std::uint64_t mixedFp32ProjRmsFwdWorkspaceBytes(int rows, int outputs, int reduction, bool compute_h);
using MixedFp32ProjRmsFwdLaunch = bool (*)(FTrainStream stream, const void* x, const void* weight, void* proj, void* r,
                                           int rows, int outputs, int reduction, float eps, void* workspace,
                                           FTrainNumericType x_type, FTrainNumericType weight_type);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, FTrainNumericType X_TYPE, FTrainNumericType W_TYPE>
bool launchMixedFp32ProjRmsFwd(FTrainStream stream, const void* x, const void* weight, void* proj, void* r, int rows,
                               int outputs, int reduction, float eps, void* workspace, FTrainNumericType x_type,
                               FTrainNumericType weight_type);

// One dtype chain of the ProjRmsFwd family: the ported fused projection plus
// RMS factor, with caller-owned scratch for split-K.
template<class Implementation>
class MixedFp32ProjRmsFwdBase : public Primitive<ProjRmsFwdProblem> {
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
    MixedFp32ProjRmsFwdBase()                               = default;
    MixedFp32ProjRmsFwdBase(const MixedFp32ProjRmsFwdBase&) = default;

  private:
    FTrainNumericType x_type_      = FTRAIN_NUMERIC_TYPE_INVALID;
    FTrainNumericType weight_type_ = FTRAIN_NUMERIC_TYPE_INVALID;

    void executeImpl(const Resources& resources) override;
    MixedFp32ProjRmsFwdLaunch launch_ = nullptr;

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

class MixedFp32ProjRmsFwd final : public MixedFp32ProjRmsFwdBase<MixedFp32ProjRmsFwd> {
  public:
    static constexpr const char* kName = "MixedFp32ProjRmsFwd";
    static constexpr bool kSplit1      = true;
};

class MixedFp32ProjRmsFwdSplit2 final : public MixedFp32ProjRmsFwdBase<MixedFp32ProjRmsFwdSplit2> {
  public:
    static constexpr const char* kName = "MixedFp32ProjRmsFwdSplit2";
    static constexpr bool kSplit2      = true;
    static constexpr int kSplit        = 2;
};

class MixedFp32ProjRmsFwdSplit4 final : public MixedFp32ProjRmsFwdBase<MixedFp32ProjRmsFwdSplit4> {
  public:
    static constexpr const char* kName = "MixedFp32ProjRmsFwdSplit4";
    static constexpr bool kSplit4      = true;
    static constexpr int kSplit        = 4;
};

class MixedFp32ProjRmsFwdSplit8 final : public MixedFp32ProjRmsFwdBase<MixedFp32ProjRmsFwdSplit8> {
  public:
    static constexpr const char* kName = "MixedFp32ProjRmsFwdSplit8";
    static constexpr bool kSplit8      = true;
    static constexpr int kSplit        = 8;
};

}  // namespace ftrain

#endif
