// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_GDN2_CHUNK_BACKWARD_GFX938_HPP_
#define FTRAIN_PRIMITIVE_HYGON_GDN2_CHUNK_BACKWARD_GFX938_HPP_

#include "flash_train/family/gdn/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// GDN2 chunked backward for gfx938, migrated from the linear_atten GDN2
// archive 05_gfx938_aicc_port.  Unlike the gfx936 candidate it accepts any
// S >= 1 with K, V <= 128 (the device entry pads and crops internally) and
// queries the archive's own backward shape gate and workspace size.
class Gdn2ChunkBackwardGfx938 final : public Primitive<Gdn2BackwardProblem> {
  public:
    const char* getName() const noexcept override { return "Gdn2ChunkBackwardBf16Gfx938"; }
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const Gdn2BackwardProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return workspace_bytes_; }
    void configure(const Gdn2BackwardProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const std::uint16_t* q_ = nullptr;
    const std::uint16_t* k_ = nullptr;
    const std::uint16_t* v_ = nullptr;
    const float* g_ = nullptr;
    const float* b_gate_ = nullptr;
    const float* w_ = nullptr;
    const std::uint16_t* do_ = nullptr;
    std::uint16_t* dq_ = nullptr;
    std::uint16_t* dk_ = nullptr;
    std::uint16_t* dv_ = nullptr;
    float* dg_ = nullptr;
    std::uint16_t* db_ = nullptr;
    std::uint16_t* dw_ = nullptr;
    int b_ = 0, s_ = 0, h_ = 0, hv_ = 0, kdim_ = 0, vdim_ = 0;
    float scale_ = 0.0F;
    std::uint64_t workspace_bytes_ = 0;
};

}  // namespace ftrain

#endif
