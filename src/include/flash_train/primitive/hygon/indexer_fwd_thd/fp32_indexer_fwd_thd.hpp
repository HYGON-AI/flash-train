// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_FWD_THD_FP32_INDEXER_FWD_THD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_FWD_THD_FP32_INDEXER_FWD_THD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/indexer_fwd_thd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool fp32IndexerFwdThdLaunchFits(int heads, int width, int keys);

// Defined in fp32_indexer_fwd_thd.hip.
bool fp32IndexerFwdThdDenseMmacSupported();
bool launchFp32IndexerFwdThd(FTrainStream stream, const void* q, const void* k, const void* w, float* scores,
                             const int* cu_seqlens_q, const int* cu_seqlens_kv, const int* q_causal_offsets,
                             int total_q, int total_k, int max_seqlen_kv, int query_heads, int kv_heads, int head_dim,
                             int num_sequences, int ratio, float sm_scale);

// One dtype chain of the IndexerFwdThd family: the ported packed-batch MMAC
// indexer forward, no workspace.
class Fp32IndexerFwdThd final : public Primitive<IndexerFwdThdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const IndexerFwdThdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const IndexerFwdThdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

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

}  // namespace ftrain

#endif
