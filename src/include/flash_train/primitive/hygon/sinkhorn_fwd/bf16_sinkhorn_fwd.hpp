// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SINKHORN_FWD_BF16_SINKHORN_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SINKHORN_FWD_BF16_SINKHORN_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/sinkhorn_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

using Bf16SinkhornFwdLaunch = bool (*)(FTrainStream stream, const void* logits, void* output, float* m_init, int batch,
                                       int size, int num_iterations, float eps);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int SIZE, int ITERATIONS>
bool launchBf16SinkhornFwd(FTrainStream stream, const void* logits, void* output, float* m_init, int batch, int size,
                           int num_iterations, float eps);

// One dtype chain of the SinkhornFwd family: the ported Sinkhorn-Knopp
// projection, whose single launch also produces the FP32 backward state.
template<class Implementation>
class Bf16SinkhornFwdBase : public Primitive<SinkhornFwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kCta  = false;
    static constexpr bool kWave = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SinkhornFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SinkhornFwdProblem& problem) override;

  protected:
    Bf16SinkhornFwdBase()                           = default;
    Bf16SinkhornFwdBase(const Bf16SinkhornFwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16SinkhornFwdLaunch launch_ = nullptr;

    const void* logits_ = nullptr;
    void* output_       = nullptr;
    float* m_init_      = nullptr;
    int batch_          = 0;
    int matrix_size_    = 0;
    int num_iterations_ = 0;
    float eps_          = 0.0F;
};

class Bf16SinkhornFwd final : public Bf16SinkhornFwdBase<Bf16SinkhornFwd> {
  public:
    static constexpr const char* kName = "Bf16SinkhornFwd";
    static constexpr bool kCta         = true;
};

class Bf16SinkhornFwdWave final : public Bf16SinkhornFwdBase<Bf16SinkhornFwdWave> {
  public:
    static constexpr const char* kName = "Bf16SinkhornFwdWave";
    static constexpr bool kWave        = true;
};

}  // namespace ftrain

#endif
