// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SINKHORN_FWD_FP16_SINKHORN_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SINKHORN_FWD_FP16_SINKHORN_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/sinkhorn_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp16_sinkhorn_fwd.hip.
bool launchFp16SinkhornFwd(FTrainStream stream, const void* logits, void* output, float* m_init, int batch, int size,
                           int num_iterations, float eps);

// One dtype chain of the SinkhornFwd family: the ported Sinkhorn-Knopp
// projection, whose single launch also produces the FP32 backward state.
class Fp16SinkhornFwd final : public Primitive<SinkhornFwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SinkhornFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SinkhornFwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* logits_ = nullptr;
    void* output_       = nullptr;
    float* m_init_      = nullptr;
    int batch_          = 0;
    int matrix_size_    = 0;
    int num_iterations_ = 0;
    float eps_          = 0.0F;
};

}  // namespace ftrain

#endif
