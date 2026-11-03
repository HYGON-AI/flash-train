// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_CSA_COMPRESSOR_BWD_BF16_CSA_COMPRESSOR_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_CSA_COMPRESSOR_BWD_BF16_CSA_COMPRESSOR_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/csa_compressor_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

void bf16CSACompressorBwdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
using Bf16CSACompressorBwdLaunch = bool (*)(FTrainStream stream, const void* kv, const void* score, const float* ape,
                                            const int* cu_seqlens, const int* cu_seqlens_comp,
                                            const void* grad_compressed, void* grad_kv, void* grad_score,
                                            float* grad_ape, int total_comp, int num_sequences, int ratio, int head_dim,
                                            int coff, int total_tokens);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int CAPACITY, int DIM, bool OVERLAP>
bool launchBf16CSACompressorBwd(FTrainStream stream, const void* kv, const void* score, const float* ape,
                                const int* cu_seqlens, const int* cu_seqlens_comp, const void* grad_compressed,
                                void* grad_kv, void* grad_score, float* grad_ape, int total_comp, int num_sequences,
                                int ratio, int head_dim, int coff, int total_tokens);

// The CSACompressorBwd Primitive: the ported fused gated-pooling backward.
//
// The kernel accumulates its outputs with atomics, so the launch owns them:
// executeImpl clears the key, score, and position-table gradients on the
// stream before the launch, which is what makes "never-consumed positions
// stay exact zero" true for the caller.
template<class Implementation>
class Bf16CSACompressorBwdBase : public Primitive<CSACompressorBwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr int kCapacity    = 1;
    static constexpr int kDim         = 16;
    static constexpr bool kSerial     = false;
    static constexpr bool kCached     = false;
    static constexpr bool kTiled      = false;
    static constexpr bool kR128       = false;
    static constexpr bool kR128Wide   = false;
    static constexpr bool kShortRows1 = false;
    static constexpr bool kShortRows4 = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const CSACompressorBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const CSACompressorBwdProblem& problem) override;

  protected:
    Bf16CSACompressorBwdBase()                                = default;
    Bf16CSACompressorBwdBase(const Bf16CSACompressorBwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16CSACompressorBwdLaunch launch_ = nullptr;

    const void* kv_                 = nullptr;
    const void* score_              = nullptr;
    const float* ape_               = nullptr;
    const int* cu_seqlens_          = nullptr;
    const int* cu_seqlens_comp_     = nullptr;
    const void* grad_compressed_    = nullptr;
    void* grad_kv_                  = nullptr;
    void* grad_score_               = nullptr;
    float* grad_ape_                = nullptr;
    std::uint64_t grad_kv_bytes_    = 0;
    std::uint64_t grad_score_bytes_ = 0;
    std::uint64_t grad_ape_bytes_   = 0;
    int total_comp_                 = 0;
    int num_sequences_              = 0;
    int ratio_                      = 0;
    int head_dim_                   = 0;
    int coff_                       = 0;
    int total_tokens_               = 0;
};

class Bf16CSACompressorBwd final : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwd> {
  public:
    static constexpr const char* kName = "CSACompressorBwd";
    static constexpr bool kSerial      = true;
};

class Bf16CSACompressorBwdCached final : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdCached> {
  public:
    static constexpr const char* kName = "CSACompressorBwdCached";
    static constexpr bool kCached      = true;
    static constexpr int kCapacity     = 4;
};

class Bf16CSACompressorBwdTiled final : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdTiled> {
  public:
    static constexpr const char* kName = "CSACompressorBwdTiled";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 64;
    static constexpr int kDim          = 32;
};

class Bf16CSACompressorBwdR128 final : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdR128> {
  public:
    static constexpr const char* kName = "CSACompressorBwdR128";
    static constexpr bool kR128        = true;
};

class Bf16CSACompressorBwdR128Wide final : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdR128Wide> {
  public:
    static constexpr const char* kName = "CSACompressorBwdR128Wide";
    static constexpr bool kR128Wide    = true;
};

class Bf16CSACompressorBwdShortRows1 final : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdShortRows1> {
  public:
    static constexpr const char* kName = "CSACompressorBwdShortRows1";
    static constexpr bool kShortRows1  = true;
};

class Bf16CSACompressorBwdShortRows4 final : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdShortRows4> {
  public:
    static constexpr const char* kName = "CSACompressorBwdShortRows4";
    static constexpr bool kShortRows4  = true;
};

class Bf16CSACompressorBwdCachedCapacity8 final : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdCachedCapacity8> {
  public:
    static constexpr const char* kName = "CSACompressorBwdCachedCapacity8";
    static constexpr bool kCached      = true;
    static constexpr int kCapacity     = 8;
};

class Bf16CSACompressorBwdCachedCapacity16 final
    : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdCachedCapacity16> {
  public:
    static constexpr const char* kName = "CSACompressorBwdCachedCapacity16";
    static constexpr bool kCached      = true;
    static constexpr int kCapacity     = 16;
};

class Bf16CSACompressorBwdCachedCapacity32 final
    : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdCachedCapacity32> {
  public:
    static constexpr const char* kName = "CSACompressorBwdCachedCapacity32";
    static constexpr bool kCached      = true;
    static constexpr int kCapacity     = 32;
};

class Bf16CSACompressorBwdCachedCapacity64 final
    : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdCachedCapacity64> {
  public:
    static constexpr const char* kName = "CSACompressorBwdCachedCapacity64";
    static constexpr bool kCached      = true;
    static constexpr int kCapacity     = 64;
};

class Bf16CSACompressorBwdTiledCapacity128 final
    : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdTiledCapacity128> {
  public:
    static constexpr const char* kName = "CSACompressorBwdTiledCapacity128";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 128;
    static constexpr int kDim          = 32;
};

class Bf16CSACompressorBwdTiledCapacity160 final
    : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdTiledCapacity160> {
  public:
    static constexpr const char* kName = "CSACompressorBwdTiledCapacity160";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 160;
    static constexpr int kDim          = 32;
};

class Bf16CSACompressorBwdTiledCapacity256 final
    : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdTiledCapacity256> {
  public:
    static constexpr const char* kName = "CSACompressorBwdTiledCapacity256";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 256;
    static constexpr int kDim          = 32;
};

class Bf16CSACompressorBwdTiledCapacity256Dim16 final
    : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdTiledCapacity256Dim16> {
  public:
    static constexpr const char* kName = "CSACompressorBwdTiledCapacity256Dim16";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 256;
    static constexpr int kDim          = 16;
};

class Bf16CSACompressorBwdTiledCapacity256Dim8 final
    : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdTiledCapacity256Dim8> {
  public:
    static constexpr const char* kName = "CSACompressorBwdTiledCapacity256Dim8";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 256;
    static constexpr int kDim          = 8;
};

class Bf16CSACompressorBwdTiledCapacity256Dim4 final
    : public Bf16CSACompressorBwdBase<Bf16CSACompressorBwdTiledCapacity256Dim4> {
  public:
    static constexpr const char* kName = "CSACompressorBwdTiledCapacity256Dim4";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 256;
    static constexpr int kDim          = 4;
};

}  // namespace ftrain

#endif
