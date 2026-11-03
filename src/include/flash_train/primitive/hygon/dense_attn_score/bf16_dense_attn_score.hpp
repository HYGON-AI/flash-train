// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_DENSE_ATTN_SCORE_BF16_DENSE_ATTN_SCORE_HPP_
#define FTRAIN_PRIMITIVE_HYGON_DENSE_ATTN_SCORE_BF16_DENSE_ATTN_SCORE_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/dense_attn_score/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
bool bf16DenseAttnScoreLaunchFits(int heads, int width, int keys);

bool bf16DenseAttnScoreDenseMmacSupported();
using Bf16DenseAttnScoreLaunch = bool (*)(FTrainStream stream, const void* q, const void* k, const float* lse,
                                          float* out, float* denom, int batch, int query_length, int key_length,
                                          int query_heads, int kv_heads, int head_dim, int ratio, float softmax_scale);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int WIDTH, int HEADS>
bool launchBf16DenseAttnScore(FTrainStream stream, const void* q, const void* k, const float* lse, float* out,
                              float* denom, int batch, int query_length, int key_length, int query_heads, int kv_heads,
                              int head_dim, int ratio, float softmax_scale);

// One dtype chain of the DenseAttnScore family: the ported MMAC dense
// attention-score kernel, no workspace.
template<class Implementation>
class Bf16DenseAttnScoreBase : public Primitive<DenseAttnScoreProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kBpermuteReduce = false;
    static constexpr bool kPadding        = false;
    static constexpr bool kStaged         = false;
    static constexpr bool kLargeD         = false;
    static constexpr bool kHoisted        = false;
    static constexpr bool kRingTeacher    = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const DenseAttnScoreProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const DenseAttnScoreProblem& problem) override;

  protected:
    Bf16DenseAttnScoreBase()                              = default;
    Bf16DenseAttnScoreBase(const Bf16DenseAttnScoreBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16DenseAttnScoreLaunch launch_ = nullptr;

    std::uint64_t workspace_bytes_ = 0;
    int groups_                    = 0;
    const void* q_                 = nullptr;
    const void* k_                 = nullptr;
    const float* lse_              = nullptr;
    float* out_                    = nullptr;
    float* denom_                  = nullptr;
    int batch_                     = 0;
    int query_length_              = 0;
    int key_length_                = 0;
    int query_heads_               = 0;
    int head_dim_                  = 0;
    int ratio_                     = 0;
    float softmax_scale_           = 0.0F;
};

class Bf16DenseAttnScoreRingTeacher final : public Bf16DenseAttnScoreBase<Bf16DenseAttnScoreRingTeacher> {
  public:
    static constexpr const char* kName = "Bf16DenseAttnScoreRingTeacher";
    static constexpr bool kRingTeacher = true;
};

class Bf16DenseAttnScore final : public Bf16DenseAttnScoreBase<Bf16DenseAttnScore> {
  public:
    static constexpr const char* kName = "Bf16DenseAttnScore";
    static constexpr bool kStaged      = true;
};

class Bf16DenseAttnScoreLargeD final : public Bf16DenseAttnScoreBase<Bf16DenseAttnScoreLargeD> {
  public:
    static constexpr const char* kName = "Bf16DenseAttnScoreLargeD";
    static constexpr bool kLargeD      = true;
};

class Bf16DenseAttnScoreHoisted final : public Bf16DenseAttnScoreBase<Bf16DenseAttnScoreHoisted> {
  public:
    static constexpr const char* kName = "Bf16DenseAttnScoreHoisted";
    static constexpr bool kHoisted     = true;
};

// Full-FP32 cross-lane reduction from the updated original implementation.
class Bf16DenseAttnScoreRingTeacherBpermute final
    : public Bf16DenseAttnScoreBase<Bf16DenseAttnScoreRingTeacherBpermute> {
  public:
    static constexpr const char* kName    = "Bf16DenseAttnScoreRingTeacherBpermute";
    static constexpr bool kRingTeacher    = true;
    static constexpr bool kBpermuteReduce = true;
};

// Classify padded upper heads and launch mutually exclusive compact/full paths.
class Bf16DenseAttnScorePadding final : public Bf16DenseAttnScoreBase<Bf16DenseAttnScorePadding> {
  public:
    static constexpr const char* kName = "Bf16DenseAttnScorePadding";
    static constexpr bool kRingTeacher = true;
    static constexpr bool kPadding     = true;
};
bool launchBf16DenseAttnScorePadding(FTrainStream stream, const void* q, const void* k, const float* lse, float* out,
                                     float* denom, int batch, int queries, int keys, float scale, int* flags,
                                     int groups);

}  // namespace ftrain

#endif
