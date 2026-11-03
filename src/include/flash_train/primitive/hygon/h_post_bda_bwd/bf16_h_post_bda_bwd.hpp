// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_H_POST_BDA_BWD_BF16_H_POST_BDA_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_H_POST_BDA_BWD_BF16_H_POST_BDA_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/h_post_bda_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

using Bf16HPostBdaBwdLaunch = bool (*)(FTrainStream stream, const void* grad_output, const void* residual,
                                       const void* x, const float* bias, const void* h_res, const void* h_post,
                                       void* grad_residual, void* grad_x, void* grad_h_res, void* grad_h_post,
                                       float* grad_bias, int batch_stride, int num_streams, int channels,
                                       bool has_bias);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int STREAMS, int JCHUNK, bool REDUCE_BIAS>
bool launchBf16HPostBdaBwd(FTrainStream stream, const void* grad_output, const void* residual, const void* x,
                           const float* bias, const void* h_res, const void* h_post, void* grad_residual, void* grad_x,
                           void* grad_h_res, void* grad_h_post, float* grad_bias, int batch_stride, int num_streams,
                           int channels, bool has_bias);

// One dtype chain of the HPostBdaBwd family: the ported backward, whose single
// main launch writes all four operand gradients and whose optional second
// launch reduces the bias gradient.
template<class Implementation>
class Bf16HPostBdaBwdBase : public Primitive<HPostBdaBwdProblem> {
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
    Bf16HPostBdaBwdBase()                           = default;
    Bf16HPostBdaBwdBase(const Bf16HPostBdaBwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16HPostBdaBwdLaunch launch_ = nullptr;

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

class Bf16HPostBdaBwdCoalesced1 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced1> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced1";
    static constexpr bool kCoalesced1  = true;
};

class Bf16HPostBdaBwdCoalesced4 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced4> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced4";
    static constexpr bool kCoalesced4  = true;
};

class Bf16HPostBdaBwd final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwd> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwd";
    static constexpr bool kGeneric     = true;
};

class Bf16HPostBdaBwdTiled1 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled1> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled1";
    static constexpr bool kTiled1      = true;
};

class Bf16HPostBdaBwdTiled4 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled4> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled4";
    static constexpr bool kTiled4      = true;
};

class Bf16HPostBdaBwdVectorized1 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized1> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized1";
    static constexpr bool kVectorized1 = true;
};

class Bf16HPostBdaBwdVectorized4 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized4> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized4";
    static constexpr bool kVectorized4 = true;
};

class Bf16HPostBdaBwdBias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdBias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdBias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTiled1Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled1Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled1Bias16";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTiled1Chunk2 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled1Chunk2> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled1Chunk2";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTiled1Chunk2Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled1Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled1Chunk2Bias16";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTiled1Chunk4 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled1Chunk4> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled1Chunk4";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTiled1Chunk4Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled1Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled1Chunk4Bias16";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTiled1Chunk8 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled1Chunk8> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled1Chunk8";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTiled1Chunk8Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled1Chunk8Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled1Chunk8Bias16";
    static constexpr bool kTiled1      = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTiled4Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled4Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled4Bias16";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTiled4Chunk2 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled4Chunk2> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled4Chunk2";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTiled4Chunk2Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled4Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled4Chunk2Bias16";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTiled4Chunk4 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled4Chunk4> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled4Chunk4";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTiled4Chunk4Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled4Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled4Chunk4Bias16";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTiled4Chunk8 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled4Chunk8> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled4Chunk8";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTiled4Chunk8Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTiled4Chunk8Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTiled4Chunk8Bias16";
    static constexpr bool kTiled4      = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdVectorized1Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized1Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized1Bias16";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdVectorized1Chunk2 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized1Chunk2> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized1Chunk2";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdVectorized1Chunk2Bias16 final
    : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized1Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized1Chunk2Bias16";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdVectorized1Chunk4 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized1Chunk4> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized1Chunk4";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdVectorized1Chunk4Bias16 final
    : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized1Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized1Chunk4Bias16";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdVectorized1Chunk8 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized1Chunk8> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized1Chunk8";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdVectorized1Chunk8Bias16 final
    : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized1Chunk8Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized1Chunk8Bias16";
    static constexpr bool kVectorized1 = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdVectorized4Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized4Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized4Bias16";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdVectorized4Chunk2 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized4Chunk2> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized4Chunk2";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdVectorized4Chunk2Bias16 final
    : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized4Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized4Chunk2Bias16";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdVectorized4Chunk4 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized4Chunk4> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized4Chunk4";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdVectorized4Chunk4Bias16 final
    : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized4Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized4Chunk4Bias16";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdVectorized4Chunk8 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized4Chunk8> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized4Chunk8";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdVectorized4Chunk8Bias16 final
    : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdVectorized4Chunk8Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdVectorized4Chunk8Bias16";
    static constexpr bool kVectorized4 = true;
    static constexpr int kChunk        = 8;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdCoalesced1Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced1Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced1Bias16";
    static constexpr bool kCoalesced1  = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdCoalesced1Chunk2 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced1Chunk2> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced1Chunk2";
    static constexpr bool kCoalesced1  = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdCoalesced1Chunk2Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced1Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced1Chunk2Bias16";
    static constexpr bool kCoalesced1  = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdCoalesced1Chunk4 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced1Chunk4> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced1Chunk4";
    static constexpr bool kCoalesced1  = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdCoalesced1Chunk4Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced1Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced1Chunk4Bias16";
    static constexpr bool kCoalesced1  = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdCoalesced4Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced4Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced4Bias16";
    static constexpr bool kCoalesced4  = true;
    static constexpr int kChunk        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdCoalesced4Chunk2 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced4Chunk2> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced4Chunk2";
    static constexpr bool kCoalesced4  = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdCoalesced4Chunk2Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced4Chunk2Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced4Chunk2Bias16";
    static constexpr bool kCoalesced4  = true;
    static constexpr int kChunk        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdCoalesced4Chunk4 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced4Chunk4> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced4Chunk4";
    static constexpr bool kCoalesced4  = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdCoalesced4Chunk4Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdCoalesced4Chunk4Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdCoalesced4Chunk4Bias16";
    static constexpr bool kCoalesced4  = true;
    static constexpr int kChunk        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTileB1 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB1> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB1";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 1;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTileB1Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB1Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB1Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 1;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTileB2 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB2> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB2";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 2;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTileB2Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB2Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB2Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 2;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTileB4 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB4> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB4";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 4;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTileB4Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB4Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB4Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 4;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTileB8 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB8> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB8";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 8;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTileB8Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB8Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB8Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 8;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTileB16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 16;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTileB16Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB16Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB16Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 16;
    static constexpr int kBiasColumns  = 16;
};

class Bf16HPostBdaBwdTileB32 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB32> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB32";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 32;
    static constexpr int kBiasColumns  = 1;
};

class Bf16HPostBdaBwdTileB32Bias16 final : public Bf16HPostBdaBwdBase<Bf16HPostBdaBwdTileB32Bias16> {
  public:
    static constexpr const char* kName = "Bf16HPostBdaBwdTileB32Bias16";
    static constexpr bool kGeneric     = true;
    static constexpr int kTileB        = 32;
    static constexpr int kBiasColumns  = 16;
};

}  // namespace ftrain

#endif
