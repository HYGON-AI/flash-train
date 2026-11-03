// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_H_AGGREGATE_FWD_MIXED_FP32_H_AGGREGATE_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_H_AGGREGATE_FWD_MIXED_FP32_H_AGGREGATE_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/h_aggregate_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

using MixedFp32HAggregateFwdLaunch = bool (*)(FTrainStream stream, const void* x, const float* h_pre, void* aggregated,
                                              int batch_stride, int num_streams, int channels,
                                              FTrainNumericType x_type);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, FTrainNumericType X_TYPE>
bool launchMixedFp32HAggregateFwd(FTrainStream stream, const void* x, const float* h_pre, void* aggregated,
                                  int batch_stride, int num_streams, int channels, FTrainNumericType x_type);

// One dtype chain of the HAggregateFwd family: the ported weighted
// aggregation, no workspace.
template<class Implementation>
class MixedFp32HAggregateFwdBase : public Primitive<HAggregateFwdProblem> {
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
    MixedFp32HAggregateFwdBase()                                  = default;
    MixedFp32HAggregateFwdBase(const MixedFp32HAggregateFwdBase&) = default;

  private:
    FTrainNumericType x_type_ = FTRAIN_NUMERIC_TYPE_INVALID;

    void executeImpl(const Resources& resources) override;
    MixedFp32HAggregateFwdLaunch launch_ = nullptr;

    const void* x_      = nullptr;
    const float* h_pre_ = nullptr;
    void* aggregated_   = nullptr;
    int batch_stride_   = 0;
    int num_streams_    = 0;
    int channels_       = 0;
};

class MixedFp32HAggregateFwd final : public MixedFp32HAggregateFwdBase<MixedFp32HAggregateFwd> {
  public:
    static constexpr const char* kName = "MixedFp32HAggregateFwd";
    static constexpr bool kTiled       = true;
};

class MixedFp32HAggregateFwdVectorized final : public MixedFp32HAggregateFwdBase<MixedFp32HAggregateFwdVectorized> {
  public:
    static constexpr const char* kName = "MixedFp32HAggregateFwdVectorized";
    static constexpr bool kVectorized  = true;
};

}  // namespace ftrain

#endif
