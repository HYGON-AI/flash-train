// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_H_POST_BDA_BWD_FP32_H_POST_BDA_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_H_POST_BDA_BWD_FP32_H_POST_BDA_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/h_post_bda_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

using Fp32HPostBdaBwdLaunch = bool (*)(FTrainStream stream, const void* grad_output, const void* residual,
                                       const void* x, const float* bias, const void* h_res, const void* h_post,
                                       void* grad_residual, void* grad_x, void* grad_h_res, void* grad_h_post,
                                       float* grad_bias, int batch_stride, int num_streams, int channels,
                                       bool has_bias);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int STREAMS, int JCHUNK, bool REDUCE_BIAS>
bool launchFp32HPostBdaBwd(FTrainStream stream, const void* grad_output, const void* residual, const void* x,
                           const float* bias, const void* h_res, const void* h_post, void* grad_residual, void* grad_x,
                           void* grad_h_res, void* grad_h_post, float* grad_bias, int batch_stride, int num_streams,
                           int channels, bool has_bias);

// One dtype chain of the HPostBdaBwd family: the ported backward, whose single
// main launch writes all four operand gradients and whose optional second
// launch reduces the bias gradient.
template<class Implementation>
class Fp32HPostBdaBwdBase : public Primitive<HPostBdaBwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr int kTileB        = 64;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 1;
    static constexpr bool kGeneric     = false;
    static constexpr bool kTiled1      = false;
    static constexpr bool kTiled4      = false;
    static constexpr bool kVectorized1 = false;
    static constexpr bool kVectorized4 = false;
    static constexpr bool kCoalesced1  = false;
    static constexpr bool kCoalesced4  = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const HPostBdaBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const HPostBdaBwdProblem& problem) override;

  protected:
    Fp32HPostBdaBwdBase()                           = default;
    Fp32HPostBdaBwdBase(const Fp32HPostBdaBwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Fp32HPostBdaBwdLaunch launch_ = nullptr;

    const void* grad_output_ = nullptr;
    const void* residual_    = nullptr;
    const void* x_           = nullptr;
    const float* bias_       = nullptr;
    const void* h_res_       = nullptr;
    const void* h_post_      = nullptr;
    void* grad_residual_     = nullptr;
    void* grad_x_            = nullptr;
    void* grad_h_res_        = nullptr;
    void* grad_h_post_       = nullptr;
    float* grad_bias_        = nullptr;
    int batch_stride_        = 0;
    int num_streams_         = 0;
    int channels_            = 0;
    bool has_bias_           = false;
    bool reduce_bias_        = false;
};

class Fp32HPostBdaBwdCoalesced1 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced1> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced1";
    static constexpr bool kCoalesced1  = true;
};

class Fp32HPostBdaBwdCoalesced4 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced4> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced4";
    static constexpr bool kCoalesced4  = true;
};

class Fp32HPostBdaBwd final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwd> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwd";
    static constexpr bool kGeneric     = true;
};

class Fp32HPostBdaBwdTiled1 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled1> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled1";
    static constexpr bool kTiled1      = true;
};

class Fp32HPostBdaBwdTiled4 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled4> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled4";
    static constexpr bool kTiled4      = true;
};

class Fp32HPostBdaBwdVectorized1 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized1> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized1";
    static constexpr bool kVectorized1 = true;
};

class Fp32HPostBdaBwdVectorized4 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized4> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized4";
    static constexpr bool kVectorized4 = true;
};

class Fp32HPostBdaBwdBias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdBias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdBias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTiled1Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled1Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled1Bias16";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTiled1Chunk2 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled1Chunk2> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled1Chunk2";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTiled1Chunk2Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled1Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled1Chunk2Bias16";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTiled1Chunk4 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled1Chunk4> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled1Chunk4";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTiled1Chunk4Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled1Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled1Chunk4Bias16";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTiled1Chunk8 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled1Chunk8> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled1Chunk8";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTiled1Chunk8Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled1Chunk8Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled1Chunk8Bias16";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTiled4Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled4Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled4Bias16";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTiled4Chunk2 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled4Chunk2> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled4Chunk2";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTiled4Chunk2Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled4Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled4Chunk2Bias16";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTiled4Chunk4 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled4Chunk4> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled4Chunk4";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTiled4Chunk4Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled4Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled4Chunk4Bias16";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTiled4Chunk8 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled4Chunk8> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled4Chunk8";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTiled4Chunk8Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTiled4Chunk8Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTiled4Chunk8Bias16";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdVectorized1Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized1Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized1Bias16";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdVectorized1Chunk2 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized1Chunk2> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized1Chunk2";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdVectorized1Chunk2Bias16 final
    : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized1Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized1Chunk2Bias16";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdVectorized1Chunk4 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized1Chunk4> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized1Chunk4";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdVectorized1Chunk4Bias16 final
    : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized1Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized1Chunk4Bias16";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdVectorized1Chunk8 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized1Chunk8> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized1Chunk8";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdVectorized1Chunk8Bias16 final
    : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized1Chunk8Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized1Chunk8Bias16";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdVectorized4Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized4Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized4Bias16";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdVectorized4Chunk2 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized4Chunk2> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized4Chunk2";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdVectorized4Chunk2Bias16 final
    : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized4Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized4Chunk2Bias16";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdVectorized4Chunk4 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized4Chunk4> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized4Chunk4";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdVectorized4Chunk4Bias16 final
    : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized4Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized4Chunk4Bias16";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdVectorized4Chunk8 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized4Chunk8> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized4Chunk8";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdVectorized4Chunk8Bias16 final
    : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdVectorized4Chunk8Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdVectorized4Chunk8Bias16";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdCoalesced1Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced1Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced1Bias16";
    static constexpr bool kCoalesced1  = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdCoalesced1Chunk2 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced1Chunk2> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced1Chunk2";
    static constexpr bool kCoalesced1  = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdCoalesced1Chunk2Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced1Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced1Chunk2Bias16";
    static constexpr bool kCoalesced1  = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdCoalesced1Chunk4 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced1Chunk4> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced1Chunk4";
    static constexpr bool kCoalesced1  = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdCoalesced1Chunk4Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced1Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced1Chunk4Bias16";
    static constexpr bool kCoalesced1  = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdCoalesced4Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced4Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced4Bias16";
    static constexpr bool kCoalesced4  = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdCoalesced4Chunk2 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced4Chunk2> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced4Chunk2";
    static constexpr bool kCoalesced4  = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdCoalesced4Chunk2Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced4Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced4Chunk2Bias16";
    static constexpr bool kCoalesced4  = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdCoalesced4Chunk4 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced4Chunk4> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced4Chunk4";
    static constexpr bool kCoalesced4  = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdCoalesced4Chunk4Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdCoalesced4Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdCoalesced4Chunk4Bias16";
    static constexpr bool kCoalesced4  = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTileB1 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB1> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB1";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 1;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTileB1Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB1Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB1Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTileB2 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB2> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB2";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTileB2Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB2Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB2Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTileB4 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB4> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB4";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTileB4Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB4Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB4Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTileB8 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB8> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB8";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 8;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTileB8Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB8Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB8Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 8;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTileB16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 16;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTileB16Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB16Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB16Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 16;
    static constexpr int kBiasColumns  = 16;
};

class Fp32HPostBdaBwdTileB32 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB32> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB32";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 32;
    static constexpr int kBiasColumns  = 1;
};

class Fp32HPostBdaBwdTileB32Bias16 final : public Fp32HPostBdaBwdBase<Fp32HPostBdaBwdTileB32Bias16> {
  public:
    static constexpr const char* kName = "Fp32HPostBdaBwdTileB32Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 32;
    static constexpr int kBiasColumns  = 16;
};

}  // namespace ftrain

#endif
