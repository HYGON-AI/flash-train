// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SINKHORN_BWD_FP32_SINKHORN_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SINKHORN_BWD_FP32_SINKHORN_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/sinkhorn_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp32_sinkhorn_bwd.hip.
bool launchFp32SinkhornBwd(FTrainStream stream, const void* grad_output, const float* m_init, void* grad_logits,
                           int batch, int size, int num_iterations, float eps);

// One dtype chain of the SinkhornBwd family. The backward kernel stages the
// whole iteration history in shared memory, so applicability also asks
// whether that footprint fits the device.
class Fp32SinkhornBwd final : public Primitive<SinkhornBwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SinkhornBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SinkhornBwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* grad_output_ = nullptr;
    const float* m_init_     = nullptr;
    void* grad_logits_       = nullptr;
    int batch_               = 0;
    int matrix_size_         = 0;
    int num_iterations_      = 0;
    float eps_               = 0.0F;
};

}  // namespace ftrain

#endif
