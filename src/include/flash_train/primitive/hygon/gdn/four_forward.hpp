// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_GDN_FOUR_FORWARD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_GDN_FOUR_FORWARD_HPP_

#include "flash_train/family/gdn/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

class GdnFourForward final : public Primitive<GdnForwardProblem> {
  public:
    const char* getName() const noexcept override { return "GdnFourForwardBf16"; }
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const GdnForwardProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return workspace_bytes_; }
    void configure(const GdnForwardProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const std::uint16_t* q_ = nullptr;
    const std::uint16_t* k_ = nullptr;
    const std::uint16_t* v_ = nullptr;
    const float* g_ = nullptr;
    const float* beta_ = nullptr;
    std::uint16_t* out_ = nullptr;
    int b_ = 0, s_ = 0, h_ = 0, hv_ = 0, kdim_ = 0, vdim_ = 0;
    float scale_ = 0.0F;
    std::uint64_t workspace_bytes_ = 0;
};

}  // namespace ftrain

#endif
