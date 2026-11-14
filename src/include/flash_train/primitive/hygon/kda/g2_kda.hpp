// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_KDA_G2_KDA_HPP_
#define FTRAIN_PRIMITIVE_HYGON_KDA_G2_KDA_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/family/kda/problem.hpp"
#include "flash_train/primitive.hpp"
#include "flash_train/primitive/hygon/kda/g2_runner.hpp"

namespace ftrain {

// The G2 KDA Primitives: handwritten gfx936 kernels (chunk 64, K = V = 128), validated
// against an FP64 reference with cuDNN-frontend PR #556 tolerances. Supported values:
//   - device architecture gfx936;
//   - K = V = 128; bf16 q, k, v, o (d_o, dq, dk, dv); fp32 post-activation beta (dbeta);
//     g bf16 with gate_in_kernel (fp32 a_log [H], dt_bias [H*K], lower_bound in [-5, 0)),
//     fp32 per-token log decay without it -- the decay must then stay >= -5 per token
//     (the chunk algorithm's safe-gate range; caller precondition, not checked);
//   - int32 cu_seqlens (any lengths, zero-length sequences allowed); fp32 q_rstd/k_rstd,
//     bf16 aqk/akk;
//   - continuously indexed, 16-byte aligned ports; T * H * 128 < 2^31.
// Deterministic: fixed-order reductions only, no atomics.
class G2KdaForward final : public Primitive<KdaForwardProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const KdaForwardProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const KdaForwardProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    kda_g2::Shape shape_{};
    kda_g2::ForwardArgs args_{};
};

class G2KdaBackward final : public Primitive<KdaBackwardProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const KdaBackwardProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const KdaBackwardProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    kda_g2::Shape shape_{};
    kda_g2::BackwardArgs args_{};
};

}  // namespace ftrain

#endif
