// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_PRIMITIVE_HYGON_DISTRIBUTED_AG_GROUP_GEMM_HPP_
#define FTRAIN_PRIMITIVE_HYGON_DISTRIBUTED_AG_GROUP_GEMM_HPP_

#include <flash_train/family/ag_group_gemm/family.hpp>
#include <flash_train/session/distributed.hpp>

#include <memory>
#include <optional>
#include <string>

namespace ftrain {

class HygonAGGroupGemmPrimitive final : public Primitive<AGGroupGemmProblem> {
  public:
    explicit HygonAGGroupGemmPrimitive(FTrainAGRoute route);

    const char* getName() const noexcept override { return name_.c_str(); }
    std::unique_ptr<PrimitiveBase> clone() const override {
        return std::make_unique<HygonAGGroupGemmPrimitive>(route_);
    }
    Result isApplicable(const AGGroupGemmProblem& problem, const Constraints& constraints) const override;
    void configure(const AGGroupGemmProblem& problem) override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return 0; }

  protected:
    void executeImpl(const Resources& resources) override;

  private:
    FTrainAGRoute route_;
    std::string name_;
    std::shared_ptr<AGGroupGemmSession> session_;
    std::optional<AGGroupGemmProblem> problem_;
};

}  // namespace ftrain

#endif
