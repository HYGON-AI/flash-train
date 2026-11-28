// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_CSA_COMPRESSOR_FWD_BF16_CSA_COMPRESSOR_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_CSA_COMPRESSOR_FWD_BF16_CSA_COMPRESSOR_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/csa_compressor_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

using Bf16CSACompressorFwdLaunch = bool (*)(FTrainStream stream, const void* kv, const void* score, const float* ape,
                                            const int* cu_seqlens, const int* cu_seqlens_comp, void* compressed,
                                            int total_comp, int num_sequences, int ratio, int head_dim, int coff,
                                            int total_tokens);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int CAPACITY, int DIM, bool OVERLAP>
bool launchBf16CSACompressorFwd(FTrainStream stream, const void* kv, const void* score, const float* ape,
                                const int* cu_seqlens, const int* cu_seqlens_comp, void* compressed, int total_comp,
                                int num_sequences, int ratio, int head_dim, int coff, int total_tokens);

// The CSACompressorFwd Primitive: the ported fused gated-pooling forward, no
// workspace and no dtype chain.
template<class Implementation>
class Bf16CSACompressorFwdBase : public Primitive<CSACompressorFwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr int kCapacity         = 1;
    static constexpr int kDim              = 16;
    static constexpr bool kSerial          = false;
    static constexpr bool kCached          = false;
    static constexpr bool kTiled           = false;
    static constexpr bool kR128            = false;
    static constexpr bool kR128WindowMajor = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const CSACompressorFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const CSACompressorFwdProblem& problem) override;

  protected:
    Bf16CSACompressorFwdBase()                                = default;
    Bf16CSACompressorFwdBase(const Bf16CSACompressorFwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16CSACompressorFwdLaunch launch_ = nullptr;

    const void* kv_             = nullptr;
    const void* score_          = nullptr;
    const float* ape_           = nullptr;
    const int* cu_seqlens_      = nullptr;
    const int* cu_seqlens_comp_ = nullptr;
    void* compressed_           = nullptr;
    int total_comp_             = 0;
    int num_sequences_          = 0;
    int ratio_                  = 0;
    int head_dim_               = 0;
    int coff_                   = 0;
    int total_tokens_           = 0;
};

class Bf16CSACompressorFwd final : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwd> {
  public:
    static constexpr const char* kName = "CSACompressorFwd";
    static constexpr bool kSerial      = true;
};

class Bf16CSACompressorFwdCached final : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdCached> {
  public:
    static constexpr const char* kName = "CSACompressorFwdCached";
    static constexpr bool kCached      = true;
    static constexpr int kCapacity     = 4;
};

class Bf16CSACompressorFwdTiled final : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiled> {
  public:
    static constexpr const char* kName = "CSACompressorFwdTiled";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 64;
    static constexpr int kDim          = 32;
};

class Bf16CSACompressorFwdR128 final : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdR128> {
  public:
    static constexpr const char* kName = "CSACompressorFwdR128";
    static constexpr bool kR128        = true;
};

class Bf16CSACompressorFwdR128WindowMajor final : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdR128WindowMajor> {
  public:
    static constexpr const char* kName     = "CSACompressorFwdR128WindowMajor";
    static constexpr bool kR128WindowMajor = true;
};

class Bf16CSACompressorFwdCachedCapacity8 final : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdCachedCapacity8> {
  public:
    static constexpr const char* kName = "CSACompressorFwdCachedCapacity8";
    static constexpr bool kCached      = true;
    static constexpr int kCapacity     = 8;
};

class Bf16CSACompressorFwdCachedCapacity16 final
    : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdCachedCapacity16> {
  public:
    static constexpr const char* kName = "CSACompressorFwdCachedCapacity16";
    static constexpr bool kCached      = true;
    static constexpr int kCapacity     = 16;
};

class Bf16CSACompressorFwdCachedCapacity32 final
    : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdCachedCapacity32> {
  public:
    static constexpr const char* kName = "CSACompressorFwdCachedCapacity32";
    static constexpr bool kCached      = true;
    static constexpr int kCapacity     = 32;
};

class Bf16CSACompressorFwdCachedCapacity64 final
    : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdCachedCapacity64> {
  public:
    static constexpr const char* kName = "CSACompressorFwdCachedCapacity64";
    static constexpr bool kCached      = true;
    static constexpr int kCapacity     = 64;
};

class Bf16CSACompressorFwdTiledCapacity128 final
    : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity128> {
  public:
    static constexpr const char* kName = "CSACompressorFwdTiledCapacity128";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 128;
    static constexpr int kDim          = 32;
};

class Bf16CSACompressorFwdTiledCapacity160 final
    : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity160> {
  public:
    static constexpr const char* kName = "CSACompressorFwdTiledCapacity160";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 160;
    static constexpr int kDim          = 32;
};

class Bf16CSACompressorFwdTiledCapacity256 final
    : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity256> {
  public:
    static constexpr const char* kName = "CSACompressorFwdTiledCapacity256";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 256;
    static constexpr int kDim          = 32;
};

class Bf16CSACompressorFwdTiledCapacity256Dim16 final
    : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity256Dim16> {
  public:
    static constexpr const char* kName = "CSACompressorFwdTiledCapacity256Dim16";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 256;
    static constexpr int kDim          = 16;
};

class Bf16CSACompressorFwdTiledCapacity256Dim8 final
    : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity256Dim8> {
  public:
    static constexpr const char* kName = "CSACompressorFwdTiledCapacity256Dim8";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 256;
    static constexpr int kDim          = 8;
};

class Bf16CSACompressorFwdTiledCapacity256Dim4 final
    : public Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity256Dim4> {
  public:
    static constexpr const char* kName = "CSACompressorFwdTiledCapacity256Dim4";
    static constexpr bool kTiled       = true;
    static constexpr int kCapacity     = 256;
    static constexpr int kDim          = 4;
};

}  // namespace ftrain

#endif
