// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_H_POST_BDA_FWD_BF16_H_POST_BDA_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_H_POST_BDA_FWD_BF16_H_POST_BDA_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/h_post_bda_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

using Bf16HPostBdaFwdLaunch = bool (*)(FTrainStream stream, const void* h_res, const void* residual, const void* h_post,
                                       const void* x, const float* bias, void* output, int batch_rows, int num_streams,
                                       int channels, bool has_bias);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int STREAMS, bool FP32_MAPPING = false>
bool launchBf16HPostBdaFwd(FTrainStream stream, const void* h_res, const void* residual, const void* h_post,
                           const void* x, const float* bias, void* output, int batch_rows, int num_streams,
                           int channels, bool has_bias);

// One dtype chain of the HPostBdaFwd family: the ported residual-mixing and
// expansion forward, no workspace.
template<class Implementation>
class Bf16HPostBdaFwdBase : public Primitive<HPostBdaFwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kTiled      = false;
    static constexpr bool kGeneric    = false;
    static constexpr bool kVectorized = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const HPostBdaFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const HPostBdaFwdProblem& problem) override;

  protected:
    Bf16HPostBdaFwdBase()                           = default;
    Bf16HPostBdaFwdBase(const Bf16HPostBdaFwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16HPostBdaFwdLaunch launch_ = nullptr;

    const void* h_res_    = nullptr;
    const void* residual_ = nullptr;
    const void* h_post_   = nullptr;
    const void* x_        = nullptr;
    const float* bias_    = nullptr;
    void* output_         = nullptr;
    int batch_stride_     = 0;
    int num_streams_      = 0;
    int channels_         = 0;
    bool has_bias_        = false;
};

class Bf16HPostBdaFwd final : public Bf16HPostBdaFwdBase<Bf16HPostBdaFwd> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaFwd";
    static constexpr bool kTiled       = true;
};

class Bf16HPostBdaFwdVectorized final : public Bf16HPostBdaFwdBase<Bf16HPostBdaFwdVectorized> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaFwdVectorized";
    static constexpr bool kVectorized  = true;
};

class Bf16HPostBdaFwdGeneric final : public Bf16HPostBdaFwdBase<Bf16HPostBdaFwdGeneric> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaFwdGeneric";
    static constexpr bool kGeneric     = true;
};

}  // namespace ftrain

#endif
