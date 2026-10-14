// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_PRIMITIVE_HYGON_DISTRIBUTED_MOE_REDUCE_RS_HPP_
#define FTRAIN_PRIMITIVE_HYGON_DISTRIBUTED_MOE_REDUCE_RS_HPP_

#include <flash_train/family/moe_reduce_rs/family.hpp>
#include <flash_train/session/distributed.hpp>

#include <memory>
#include <optional>
#include <string>

namespace ftrain {

class HygonMoEReduceRSPrimitive final : public Primitive<MoEReduceRSProblem> {
  public:
    explicit HygonMoEReduceRSPrimitive(FTrainMoERSRoute route);

    const char* getName() const noexcept override { return name_.c_str(); }
    std::unique_ptr<PrimitiveBase> clone() const override {
        return std::make_unique<HygonMoEReduceRSPrimitive>(route_);
    }
    Result isApplicable(const MoEReduceRSProblem& problem, const Constraints& constraints) const override;
    void configure(const MoEReduceRSProblem& problem) override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return 0; }

  protected:
    void executeImpl(const Resources& resources) override;

  private:
    FTrainMoERSRoute route_;
    std::string name_;
    std::shared_ptr<MoEReduceRSSession> session_;
    std::optional<MoEReduceRSProblem> problem_;
};

}  // namespace ftrain

#endif
