// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_FWD_BF16_INDEXER_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_FWD_BF16_INDEXER_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/indexer_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
bool bf16IndexerFwdLaunchFits(int heads, int width, int keys);

bool bf16IndexerFwdDenseMmacSupported();
using Bf16IndexerFwdLaunch = bool (*)(FTrainStream stream, const void* q, const void* k, const void* w, float* scores,
                                      int batch, int query_length, int key_length, int query_heads, int kv_heads,
                                      int head_dim, int ratio, float sm_scale);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int WIDTH, int HEADS>
bool launchBf16IndexerFwd(FTrainStream stream, const void* q, const void* k, const void* w, float* scores, int batch,
                          int query_length, int key_length, int query_heads, int kv_heads, int head_dim, int ratio,
                          float sm_scale);

// One dtype chain of the IndexerFwd family: the ported MMAC indexer-forward
// kernel launched for ports of one storage type, no workspace.
//
// The chain fixes both the compute type and the ports' storage type, so a
// BF16 call never reaches the FP32 kernel and vice versa; a mixed call is
// rejected by both and therefore needs a third implementation, exactly as
// docs/development.md's "add an implementation" section describes.
//
// isApplicable checks the implementation limits. Problem keys encode every
// operand property used by selection, including address alignment.
template<class Implementation>
class Bf16IndexerFwdBase : public Primitive<IndexerFwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kStaged     = false;
    static constexpr bool kLargeD     = false;
    static constexpr bool kHoisted    = false;
    static constexpr bool kPrefetch   = false;
    static constexpr bool kScoreRing  = false;
    static constexpr bool kMultiQuery = false;
    static constexpr bool kTransposed = false;

    const char* getName() const noexcept override;

    std::unique_ptr<PrimitiveBase> clone() const override;

    Result isApplicable(const IndexerFwdProblem& problem, const Constraints& constraints) const override;

    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;

    void configure(const IndexerFwdProblem& problem) override;

  protected:
    Bf16IndexerFwdBase()                          = default;
    Bf16IndexerFwdBase(const Bf16IndexerFwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16IndexerFwdLaunch launch_ = nullptr;

    const void* q_    = nullptr;
    const void* k_    = nullptr;
    const void* w_    = nullptr;
    float* scores_    = nullptr;
    int batch_        = 0;
    int query_length_ = 0;
    int key_length_   = 0;
    int query_heads_  = 0;
    int head_dim_     = 0;
    int ratio_        = 0;
    float sm_scale_   = 0.0F;
};

class Bf16IndexerFwdMultiQuery final : public Bf16IndexerFwdBase<Bf16IndexerFwdMultiQuery> {
  public:
    static constexpr const char* kName = "Bf16IndexerFwdMultiQuery";
    static constexpr bool kMultiQuery  = true;
};

class Bf16IndexerFwdTransposed final : public Bf16IndexerFwdBase<Bf16IndexerFwdTransposed> {
  public:
    static constexpr const char* kName = "Bf16IndexerFwdTransposed";
    static constexpr bool kTransposed  = true;
};

class Bf16IndexerFwd final : public Bf16IndexerFwdBase<Bf16IndexerFwd> {
  public:
    static constexpr const char* kName = "Bf16IndexerFwd";
    static constexpr bool kStaged      = true;
};

class Bf16IndexerFwdLargeD final : public Bf16IndexerFwdBase<Bf16IndexerFwdLargeD> {
  public:
    static constexpr const char* kName = "Bf16IndexerFwdLargeD";
    static constexpr bool kLargeD      = true;
};

class Bf16IndexerFwdHoisted final : public Bf16IndexerFwdBase<Bf16IndexerFwdHoisted> {
  public:
    static constexpr const char* kName = "Bf16IndexerFwdHoisted";
    static constexpr bool kHoisted     = true;
};

class Bf16IndexerFwdPrefetch final : public Bf16IndexerFwdBase<Bf16IndexerFwdPrefetch> {
  public:
    static constexpr const char* kName = "Bf16IndexerFwdPrefetch";
    static constexpr bool kPrefetch    = true;
};

class Bf16IndexerFwdScoreRing final : public Bf16IndexerFwdBase<Bf16IndexerFwdScoreRing> {
  public:
    static constexpr const char* kName = "Bf16IndexerFwdScoreRing";
    static constexpr bool kScoreRing   = true;
};

}  // namespace ftrain

#endif
