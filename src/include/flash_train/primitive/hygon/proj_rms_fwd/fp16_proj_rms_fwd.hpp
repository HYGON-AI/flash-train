// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_FWD_FP16_PROJ_RMS_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_PROJ_RMS_FWD_FP16_PROJ_RMS_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/proj_rms_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp16_proj_rms_fwd.hip.
std::uint64_t fp16ProjRmsFwdWorkspaceBytes(int rows, int outputs, int reduction, bool compute_h);
bool launchFp16ProjRmsFwd(FTrainStream stream, const void* x, const void* weight, void* proj, void* r, int rows,
                          int outputs, int reduction, float eps, void* workspace);

// One dtype chain of the ProjRmsFwd family: the ported fused projection plus
// RMS factor, with caller-owned scratch for split-K.
class Fp16ProjRmsFwd final : public Primitive<ProjRmsFwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const ProjRmsFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const ProjRmsFwdProblem& problem) override;

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

}  // namespace ftrain

#endif
