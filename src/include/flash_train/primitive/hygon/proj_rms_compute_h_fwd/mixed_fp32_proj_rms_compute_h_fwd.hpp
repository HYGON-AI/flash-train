// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_COMPUTE_H_FWD_MIXED_FP32_PROJ_RMS_COMPUTE_H_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_COMPUTE_H_FWD_MIXED_FP32_PROJ_RMS_COMPUTE_H_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/proj_rms_compute_h_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
std::uint64_t mixedFp32ProjRmsComputeHFwdWorkspaceBytes(int rows, int outputs, int reduction, bool compute_h);
using MixedFp32ProjRmsComputeHFwdLaunch = bool (*)(FTrainStream stream, const void* x, const void* weight, void* proj,
                                                   void* r, const float* alpha_pre, const float* alpha_post,
                                                   const float* alpha_res, const float* bias, void* h_pre, void* h_post,
                                                   void* h_res, int rows, int outputs, int num_streams, int reduction,
                                                   float eps, float compute_h_eps, void* workspace,
                                                   FTrainNumericType x_type, FTrainNumericType weight_type);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, FTrainNumericType X_TYPE, FTrainNumericType W_TYPE, bool FP32_STATE = false>
bool launchMixedFp32ProjRmsComputeHFwd(FTrainStream stream, const void* x, const void* weight, void* proj, void* r,
                                       const float* alpha_pre, const float* alpha_post, const float* alpha_res,
                                       const float* bias, void* h_pre, void* h_post, void* h_res, int rows, int outputs,
                                       int num_streams, int reduction, float eps, float compute_h_eps, void* workspace,
                                       FTrainNumericType x_type, FTrainNumericType weight_type);

// One dtype chain of the ProjRmsComputeHFwd family: the ported single launch
// that produces the projection, the RMS factor, and the activation split.
template<class Implementation>
class MixedFp32ProjRmsComputeHFwdBase : public Primitive<ProjRmsComputeHFwdProblem> {
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
    MixedFp32ProjRmsComputeHFwdBase()                                       = default;
    MixedFp32ProjRmsComputeHFwdBase(const MixedFp32ProjRmsComputeHFwdBase&) = default;

  private:
    FTrainNumericType x_type_      = FTRAIN_NUMERIC_TYPE_INVALID;
    FTrainNumericType weight_type_ = FTRAIN_NUMERIC_TYPE_INVALID;

    void executeImpl(const Resources& resources) override;
    MixedFp32ProjRmsComputeHFwdLaunch launch_ = nullptr;

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

class MixedFp32ProjRmsComputeHFwdV2FullK final
    : public MixedFp32ProjRmsComputeHFwdBase<MixedFp32ProjRmsComputeHFwdV2FullK> {
  public:
    static constexpr const char* kName = "MixedFp32ProjRmsComputeHFwdV2FullK";
    static constexpr bool kV2FullK     = true;
};

class MixedFp32ProjRmsComputeHFwdV2ConflictFreeFullK final
    : public MixedFp32ProjRmsComputeHFwdBase<MixedFp32ProjRmsComputeHFwdV2ConflictFreeFullK> {
  public:
    static constexpr const char* kName         = "MixedFp32ProjRmsComputeHFwdV2ConflictFreeFullK";
    static constexpr bool kV2ConflictFreeFullK = true;
};

class MixedFp32ProjRmsComputeHFwd final : public MixedFp32ProjRmsComputeHFwdBase<MixedFp32ProjRmsComputeHFwd> {
  public:
    static constexpr const char* kName = "MixedFp32ProjRmsComputeHFwd";
    static constexpr bool kSplit1      = true;
};

class MixedFp32ProjRmsComputeHFwdSplit2 final
    : public MixedFp32ProjRmsComputeHFwdBase<MixedFp32ProjRmsComputeHFwdSplit2> {
  public:
    static constexpr const char* kName = "MixedFp32ProjRmsComputeHFwdSplit2";
    static constexpr bool kSplit2      = true;
    static constexpr int kSplit        = 2;
};

class MixedFp32ProjRmsComputeHFwdSplit4 final
    : public MixedFp32ProjRmsComputeHFwdBase<MixedFp32ProjRmsComputeHFwdSplit4> {
  public:
    static constexpr const char* kName = "MixedFp32ProjRmsComputeHFwdSplit4";
    static constexpr bool kSplit4      = true;
    static constexpr int kSplit        = 4;
};

class MixedFp32ProjRmsComputeHFwdSplit8 final
    : public MixedFp32ProjRmsComputeHFwdBase<MixedFp32ProjRmsComputeHFwdSplit8> {
  public:
    static constexpr const char* kName = "MixedFp32ProjRmsComputeHFwdSplit8";
    static constexpr bool kSplit8      = true;
    static constexpr int kSplit        = 8;
};

class MixedFp32ProjRmsComputeHFwdV2 final : public MixedFp32ProjRmsComputeHFwdBase<MixedFp32ProjRmsComputeHFwdV2> {
  public:
    static constexpr const char* kName = "MixedFp32ProjRmsComputeHFwdV2";
    static constexpr bool kV2          = true;
};

class MixedFp32ProjRmsComputeHFwdV2ConflictFree final
    : public MixedFp32ProjRmsComputeHFwdBase<MixedFp32ProjRmsComputeHFwdV2ConflictFree> {
  public:
    static constexpr const char* kName    = "MixedFp32ProjRmsComputeHFwdV2ConflictFree";
    static constexpr bool kV2ConflictFree = true;
};

}  // namespace ftrain

#endif
