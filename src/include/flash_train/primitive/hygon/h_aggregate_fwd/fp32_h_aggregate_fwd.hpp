// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_H_AGGREGATE_FWD_FP32_H_AGGREGATE_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_H_AGGREGATE_FWD_FP32_H_AGGREGATE_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/h_aggregate_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp32_h_aggregate_fwd.hip.
template<class Implementation>
bool launchFp32HAggregateFwd(FTrainStream stream, const void* x, const float* h_pre, void* aggregated, int batch_stride,
                             int num_streams, int channels);

// One dtype chain of the HAggregateFwd family: the ported weighted
// aggregation, no workspace.
template<class Implementation>
class Fp32HAggregateFwdBase : public Primitive<HAggregateFwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kTiled      = false;
    static constexpr bool kVectorized = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const HAggregateFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const HAggregateFwdProblem& problem) override;

  protected:
    Fp32HAggregateFwdBase()                             = default;
    Fp32HAggregateFwdBase(const Fp32HAggregateFwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;

    const void* x_      = nullptr;
    const float* h_pre_ = nullptr;
    void* aggregated_   = nullptr;
    int batch_stride_   = 0;
    int num_streams_    = 0;
    int channels_       = 0;
};

class Fp32HAggregateFwd final : public Fp32HAggregateFwdBase<Fp32HAggregateFwd> {
  public:
    static constexpr const char* kName = "Fp32HAggregateFwd";
    static constexpr bool kTiled       = true;
};

class Fp32HAggregateFwdVectorized final : public Fp32HAggregateFwdBase<Fp32HAggregateFwdVectorized> {
  public:
    static constexpr const char* kName = "Fp32HAggregateFwdVectorized";
    static constexpr bool kVectorized  = true;
};

}  // namespace ftrain

#endif
