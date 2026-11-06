// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_H_AGGREGATE_BWD_MIXED_FP32_H_AGGREGATE_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_H_AGGREGATE_BWD_MIXED_FP32_H_AGGREGATE_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/h_aggregate_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

using MixedFp32HAggregateBwdLaunch = bool (*)(FTrainStream stream, const void* grad_aggregated, const void* x,
                                              const float* h_pre, void* grad_x, float* grad_h_pre, int batch_stride,
                                              int num_streams, int channels, FTrainNumericType x_type);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, FTrainNumericType X_TYPE>
bool launchMixedFp32HAggregateBwd(FTrainStream stream, const void* grad_aggregated, const void* x, const float* h_pre,
                                  void* grad_x, float* grad_h_pre, int batch_stride, int num_streams, int channels,
                                  FTrainNumericType x_type);

// One dtype chain of the HAggregateBwd family: the ported aggregation
// backward, whose weight reduction runs in FP32 shared memory, no workspace.
template<class Implementation>
class MixedFp32HAggregateBwdBase : public Primitive<HAggregateBwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kTiled      = false;
    static constexpr bool kVectorized = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const HAggregateBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const HAggregateBwdProblem& problem) override;

  protected:
    MixedFp32HAggregateBwdBase()                                  = default;
    MixedFp32HAggregateBwdBase(const MixedFp32HAggregateBwdBase&) = default;

  private:
    FTrainNumericType x_type_ = FTRAIN_NUMERIC_TYPE_INVALID;

    void executeImpl(const Resources& resources) override;
    MixedFp32HAggregateBwdLaunch launch_ = nullptr;

    const void* grad_aggregated_ = nullptr;
    const void* x_               = nullptr;
    const float* h_pre_          = nullptr;
    void* grad_x_                = nullptr;
    float* grad_h_pre_           = nullptr;
    int batch_stride_            = 0;
    int num_streams_             = 0;
    int channels_                = 0;
};

class MixedFp32HAggregateBwd final : public MixedFp32HAggregateBwdBase<MixedFp32HAggregateBwd> {
  public:
    static constexpr const char* kName = "MixedFp32HAggregateBwd";
    static constexpr bool kTiled       = true;
};

class MixedFp32HAggregateBwdVectorized final : public MixedFp32HAggregateBwdBase<MixedFp32HAggregateBwdVectorized> {
  public:
    static constexpr const char* kName = "MixedFp32HAggregateBwdVectorized";
    static constexpr bool kVectorized  = true;
};

}  // namespace ftrain

#endif
