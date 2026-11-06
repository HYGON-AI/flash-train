// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_DENSE_ATTN_SCORE_THD_BF16_DENSE_ATTN_SCORE_THD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_DENSE_ATTN_SCORE_THD_BF16_DENSE_ATTN_SCORE_THD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/dense_attn_score_thd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
bool bf16DenseAttnScoreThdLaunchFits(int heads, int width, int keys);

// Called by isApplicable in .cpp; the HIP translation unit owns the device capability query.
bool bf16DenseAttnScoreThdDenseMmacSupported();
using Bf16DenseAttnScoreThdLaunch = bool (*)(FTrainStream stream, const void* q, const void* k, const float* lse,
                                             float* out, float* denom, const int* cu_seqlens_q,
                                             const int* cu_seqlens_kv, const int* q_causal_offsets, int total_q,
                                             int total_k, int max_seqlen_kv, int query_heads, int kv_heads,
                                             int head_dim, int num_sequences, int ratio, float softmax_scale);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int WIDTH, int HEADS>
bool launchBf16DenseAttnScoreThd(FTrainStream stream, const void* q, const void* k, const float* lse, float* out,
                                 float* denom, const int* cu_seqlens_q, const int* cu_seqlens_kv,
                                 const int* q_causal_offsets, int total_q, int total_k, int max_seqlen_kv,
                                 int query_heads, int kv_heads, int head_dim, int num_sequences, int ratio,
                                 float softmax_scale);

// One dtype chain of the DenseAttnScoreThd family: the ported packed-batch
// MMAC dense attention-score kernel, no workspace.
template<class Implementation>
class Bf16DenseAttnScoreThdBase : public Primitive<DenseAttnScoreThdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kBpermuteReduce = false;
    static constexpr bool kStaged         = false;
    static constexpr bool kRingTeacher    = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const DenseAttnScoreThdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const DenseAttnScoreThdProblem& problem) override;

  protected:
    Bf16DenseAttnScoreThdBase()                                 = default;
    Bf16DenseAttnScoreThdBase(const Bf16DenseAttnScoreThdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16DenseAttnScoreThdLaunch launch_ = nullptr;

    const void* q_             = nullptr;
    const void* k_             = nullptr;
    const float* lse_          = nullptr;
    const int* cu_seqlens_q_   = nullptr;
    const int* cu_seqlens_kv_  = nullptr;
    const int* causal_offsets_ = nullptr;
    float* out_                = nullptr;
    float* denom_              = nullptr;
    int total_q_               = 0;
    int total_k_               = 0;
    int max_seqlen_kv_         = 0;
    int query_heads_           = 0;
    int head_dim_              = 0;
    int num_sequences_         = 0;
    int ratio_                 = 0;
    float softmax_scale_       = 0.0F;
};

class Bf16DenseAttnScoreThd final : public Bf16DenseAttnScoreThdBase<Bf16DenseAttnScoreThd> {
  public:
    static constexpr const char* kName = "Bf16DenseAttnScoreThd";
    static constexpr bool kStaged      = true;
};

class Bf16DenseAttnScoreThdRingTeacher final : public Bf16DenseAttnScoreThdBase<Bf16DenseAttnScoreThdRingTeacher> {
  public:
    static constexpr const char* kName = "Bf16DenseAttnScoreThdRingTeacher";
    static constexpr bool kRingTeacher = true;
};

// Full-FP32 cross-lane reduction from the updated original implementation.
class Bf16DenseAttnScoreThdRingTeacherBpermute final
    : public Bf16DenseAttnScoreThdBase<Bf16DenseAttnScoreThdRingTeacherBpermute> {
  public:
    static constexpr const char* kName    = "Bf16DenseAttnScoreThdRingTeacherBpermute";
    static constexpr bool kRingTeacher    = true;
    static constexpr bool kBpermuteReduce = true;
};

}  // namespace ftrain

#endif
