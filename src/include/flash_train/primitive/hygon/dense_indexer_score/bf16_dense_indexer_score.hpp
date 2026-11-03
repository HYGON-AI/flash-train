// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_DENSE_INDEXER_SCORE_BF16_DENSE_INDEXER_SCORE_HPP_
#define FTRAIN_PRIMITIVE_HYGON_DENSE_INDEXER_SCORE_BF16_DENSE_INDEXER_SCORE_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/dense_indexer_score/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
bool bf16DenseIndexerScoreLaunchFits(int heads, int width, int keys);

bool bf16DenseIndexerScoreDenseMmacSupported();
using Bf16DenseIndexerScoreLaunch = bool (*)(FTrainStream stream, const void* q, const void* k, const void* w,
                                             float* out, float* denom, int batch, int query_length, int key_length,
                                             int query_heads, int kv_heads, int head_dim, int ratio, float sm_scale);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int WIDTH, int HEADS>
bool launchBf16DenseIndexerScore(FTrainStream stream, const void* q, const void* k, const void* w, float* out,
                                 float* denom, int batch, int query_length, int key_length, int query_heads,
                                 int kv_heads, int head_dim, int ratio, float sm_scale);

// One dtype chain of the DenseIndexerScore family: the ported MMAC dense
// indexer-score kernel, no workspace.
template<class Implementation>
class Bf16DenseIndexerScoreBase : public Primitive<DenseIndexerScoreProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kStaged     = false;
    static constexpr bool kLargeD     = false;
    static constexpr bool kHoisted    = false;
    static constexpr bool kMultiQuery = false;
    static constexpr bool kTransposed = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const DenseIndexerScoreProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const DenseIndexerScoreProblem& problem) override;

  protected:
    Bf16DenseIndexerScoreBase()                                 = default;
    Bf16DenseIndexerScoreBase(const Bf16DenseIndexerScoreBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16DenseIndexerScoreLaunch launch_ = nullptr;

    const void* q_    = nullptr;
    const void* k_    = nullptr;
    const void* w_    = nullptr;
    float* out_       = nullptr;
    float* denom_     = nullptr;
    int batch_        = 0;
    int query_length_ = 0;
    int key_length_   = 0;
    int query_heads_  = 0;
    int head_dim_     = 0;
    int ratio_        = 0;
    float sm_scale_   = 0.0F;
};

class Bf16DenseIndexerScoreMultiQuery final : public Bf16DenseIndexerScoreBase<Bf16DenseIndexerScoreMultiQuery> {
  public:
    static constexpr const char* kName = "Bf16DenseIndexerScoreMultiQuery";
    static constexpr bool kMultiQuery  = true;
};

class Bf16DenseIndexerScoreTransposed final : public Bf16DenseIndexerScoreBase<Bf16DenseIndexerScoreTransposed> {
  public:
    static constexpr const char* kName = "Bf16DenseIndexerScoreTransposed";
    static constexpr bool kTransposed  = true;
};

class Bf16DenseIndexerScore final : public Bf16DenseIndexerScoreBase<Bf16DenseIndexerScore> {
  public:
    static constexpr const char* kName = "Bf16DenseIndexerScore";
    static constexpr bool kStaged      = true;
};

class Bf16DenseIndexerScoreLargeD final : public Bf16DenseIndexerScoreBase<Bf16DenseIndexerScoreLargeD> {
  public:
    static constexpr const char* kName = "Bf16DenseIndexerScoreLargeD";
    static constexpr bool kLargeD      = true;
};

class Bf16DenseIndexerScoreHoisted final : public Bf16DenseIndexerScoreBase<Bf16DenseIndexerScoreHoisted> {
  public:
    static constexpr const char* kName = "Bf16DenseIndexerScoreHoisted";
    static constexpr bool kHoisted     = true;
};

}  // namespace ftrain

#endif
