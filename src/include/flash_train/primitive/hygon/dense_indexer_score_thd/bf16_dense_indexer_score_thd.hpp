// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_DENSE_INDEXER_SCORE_THD_BF16_DENSE_INDEXER_SCORE_THD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_DENSE_INDEXER_SCORE_THD_BF16_DENSE_INDEXER_SCORE_THD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/dense_indexer_score_thd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
bool bf16DenseIndexerScoreThdLaunchFits(int heads, int width, int keys);

bool bf16DenseIndexerScoreThdDenseMmacSupported();
using Bf16DenseIndexerScoreThdLaunch = bool (*)(FTrainStream stream, const void* q, const void* k, const void* w,
                                                float* out, float* denom, const int* cu_seqlens_q,
                                                const int* cu_seqlens_kv, const int* q_causal_offsets, int total_q,
                                                int total_k, int max_seqlen_kv, int query_heads, int kv_heads,
                                                int head_dim, int num_sequences, int ratio, float sm_scale);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int WIDTH, int HEADS>
bool launchBf16DenseIndexerScoreThd(FTrainStream stream, const void* q, const void* k, const void* w, float* out,
                                    float* denom, const int* cu_seqlens_q, const int* cu_seqlens_kv,
                                    const int* q_causal_offsets, int total_q, int total_k, int max_seqlen_kv,
                                    int query_heads, int kv_heads, int head_dim, int num_sequences, int ratio,
                                    float sm_scale);

// One dtype chain of the DenseIndexerScoreThd family: the ported packed-batch
// MMAC dense indexer-score kernel, no workspace.
template<class Implementation>
class Bf16DenseIndexerScoreThdBase : public Primitive<DenseIndexerScoreThdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kStaged     = false;
    static constexpr bool kMultiQuery = false;
    static constexpr bool kTransposed = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const DenseIndexerScoreThdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const DenseIndexerScoreThdProblem& problem) override;

  protected:
    Bf16DenseIndexerScoreThdBase()                                    = default;
    Bf16DenseIndexerScoreThdBase(const Bf16DenseIndexerScoreThdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16DenseIndexerScoreThdLaunch launch_ = nullptr;

    const void* q_             = nullptr;
    const void* k_             = nullptr;
    const void* w_             = nullptr;
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
    float sm_scale_            = 0.0F;
};

class Bf16DenseIndexerScoreThd final : public Bf16DenseIndexerScoreThdBase<Bf16DenseIndexerScoreThd> {
  public:
    static constexpr const char* kName = "Bf16DenseIndexerScoreThd";
    static constexpr bool kStaged      = true;
};

class Bf16DenseIndexerScoreThdMultiQuery final
    : public Bf16DenseIndexerScoreThdBase<Bf16DenseIndexerScoreThdMultiQuery> {
  public:
    static constexpr const char* kName = "Bf16DenseIndexerScoreThdMultiQuery";
    static constexpr bool kMultiQuery  = true;
};

class Bf16DenseIndexerScoreThdTransposed final
    : public Bf16DenseIndexerScoreThdBase<Bf16DenseIndexerScoreThdTransposed> {
  public:
    static constexpr const char* kName = "Bf16DenseIndexerScoreThdTransposed";
    static constexpr bool kTransposed  = true;
};

}  // namespace ftrain

#endif
