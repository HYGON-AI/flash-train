// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_H_AGGREGATE_BWD_FP32_H_AGGREGATE_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_H_AGGREGATE_BWD_FP32_H_AGGREGATE_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/h_aggregate_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp32_h_aggregate_bwd.hip.
bool launchFp32HAggregateBwd(FTrainStream stream, const void* grad_aggregated, const void* x, const float* h_pre,
                             void* grad_x, float* grad_h_pre, int batch_stride, int num_streams, int channels);

// One dtype chain of the HAggregateBwd family: the ported aggregation
// backward, whose weight reduction runs in FP32 shared memory, no workspace.
class Fp32HAggregateBwd final : public Primitive<HAggregateBwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const HAggregateBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const HAggregateBwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* grad_aggregated_ = nullptr;
    const void* x_               = nullptr;
    const float* h_pre_          = nullptr;
    void* grad_x_                = nullptr;
    float* grad_h_pre_           = nullptr;
    int batch_stride_            = 0;
    int num_streams_             = 0;
    int channels_                = 0;
};

}  // namespace ftrain

#endif
