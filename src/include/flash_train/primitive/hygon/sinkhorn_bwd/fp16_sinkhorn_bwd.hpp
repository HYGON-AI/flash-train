// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SINKHORN_BWD_FP16_SINKHORN_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SINKHORN_BWD_FP16_SINKHORN_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/sinkhorn_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

using Fp16SinkhornBwdLaunch = bool (*)(FTrainStream stream, const void* grad_output, const float* m_init,
                                       void* grad_logits, int batch, int size, int num_iterations, float eps);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int SIZE, int ITERATIONS>
bool launchFp16SinkhornBwd(FTrainStream stream, const void* grad_output, const float* m_init, void* grad_logits,
                           int batch, int size, int num_iterations, float eps);

// One dtype chain of the SinkhornBwd family. The backward kernel stages the
// whole iteration history in shared memory, so applicability also asks
// whether that footprint fits the device.
template<class Implementation>
class Fp16SinkhornBwdBase : public Primitive<SinkhornBwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kCta  = false;
    static constexpr bool kWave = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SinkhornBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SinkhornBwdProblem& problem) override;

  protected:
    Fp16SinkhornBwdBase()                           = default;
    Fp16SinkhornBwdBase(const Fp16SinkhornBwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Fp16SinkhornBwdLaunch launch_ = nullptr;

    const void* grad_output_ = nullptr;
    const float* m_init_     = nullptr;
    void* grad_logits_       = nullptr;
    int batch_               = 0;
    int matrix_size_         = 0;
    int num_iterations_      = 0;
    float eps_               = 0.0F;
};

class Fp16SinkhornBwd final : public Fp16SinkhornBwdBase<Fp16SinkhornBwd> {
  public:
    static constexpr const char* kName = "Fp16SinkhornBwd";
    static constexpr bool kCta         = true;
};

class Fp16SinkhornBwdWave final : public Fp16SinkhornBwdBase<Fp16SinkhornBwdWave> {
  public:
    static constexpr const char* kName = "Fp16SinkhornBwdWave";
    static constexpr bool kWave        = true;
};

}  // namespace ftrain

#endif
