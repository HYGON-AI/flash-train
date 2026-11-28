// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_GDN_FUSED_FORWARD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_GDN_FUSED_FORWARD_HPP_

#include "flash_train/family/gdn/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

class GdnFusedForward final : public Primitive<GdnForwardProblem> {
  public:
    explicit GdnFusedForward(FTrainNumericType type) : type_(type) {}

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const GdnForwardProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return 0; }
    void configure(const GdnForwardProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    FTrainNumericType type_;
    const void* q_ = nullptr;
    const void* k_ = nullptr;
    const void* v_ = nullptr;
    const float* g_ = nullptr;
    const float* beta_ = nullptr;
    const float* h0_ = nullptr;
    void* out_ = nullptr;
    float* ht_ = nullptr;
    int b_ = 0, s_ = 0, h_ = 0, hv_ = 0, kdim_ = 0, vdim_ = 0;
    float scale_ = 0.0F;
};

}  // namespace ftrain

#endif
