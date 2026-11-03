// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_FWD_THD_BF16_INDEXER_FWD_THD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_FWD_THD_BF16_INDEXER_FWD_THD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/indexer_fwd_thd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
bool bf16IndexerFwdThdLaunchFits(int heads, int width, int keys);

bool bf16IndexerFwdThdDenseMmacSupported();
using Bf16IndexerFwdThdLaunch = bool (*)(FTrainStream stream, const void* q, const void* k, const void* w,
                                         float* scores, const int* cu_seqlens_q, const int* cu_seqlens_kv,
                                         const int* q_causal_offsets, int total_q, int total_k, int max_seqlen_kv,
                                         int query_heads, int kv_heads, int head_dim, int num_sequences, int ratio,
                                         float sm_scale);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int WIDTH, int HEADS>
bool launchBf16IndexerFwdThd(FTrainStream stream, const void* q, const void* k, const void* w, float* scores,
                             const int* cu_seqlens_q, const int* cu_seqlens_kv, const int* q_causal_offsets,
                             int total_q, int total_k, int max_seqlen_kv, int query_heads, int kv_heads, int head_dim,
                             int num_sequences, int ratio, float sm_scale);

// One dtype chain of the IndexerFwdThd family: the ported packed-batch MMAC
// indexer forward, no workspace.
template<class Implementation>
class Bf16IndexerFwdThdBase : public Primitive<IndexerFwdThdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kStaged     = false;
    static constexpr bool kMultiQuery = false;
    static constexpr bool kTransposed = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const IndexerFwdThdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const IndexerFwdThdProblem& problem) override;

  protected:
    Bf16IndexerFwdThdBase()                             = default;
    Bf16IndexerFwdThdBase(const Bf16IndexerFwdThdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16IndexerFwdThdLaunch launch_ = nullptr;

    const void* q_             = nullptr;
    const void* k_             = nullptr;
    const void* w_             = nullptr;
    const int* cu_seqlens_q_   = nullptr;
    const int* cu_seqlens_kv_  = nullptr;
    const int* causal_offsets_ = nullptr;
    float* scores_             = nullptr;
    int total_q_               = 0;
    int total_k_               = 0;
    int max_seqlen_kv_         = 0;
    int query_heads_           = 0;
    int head_dim_              = 0;
    int num_sequences_         = 0;
    int ratio_                 = 0;
    float sm_scale_            = 0.0F;
};

class Bf16IndexerFwdThd final : public Bf16IndexerFwdThdBase<Bf16IndexerFwdThd> {
  public:
    static constexpr const char* kName = "Bf16IndexerFwdThd";
    static constexpr bool kStaged      = true;
};

class Bf16IndexerFwdThdMultiQuery final : public Bf16IndexerFwdThdBase<Bf16IndexerFwdThdMultiQuery> {
  public:
    static constexpr const char* kName = "Bf16IndexerFwdThdMultiQuery";
    static constexpr bool kMultiQuery  = true;
};

class Bf16IndexerFwdThdTransposed final : public Bf16IndexerFwdThdBase<Bf16IndexerFwdThdTransposed> {
  public:
    static constexpr const char* kName = "Bf16IndexerFwdThdTransposed";
    static constexpr bool kTransposed  = true;
};

}  // namespace ftrain

#endif
