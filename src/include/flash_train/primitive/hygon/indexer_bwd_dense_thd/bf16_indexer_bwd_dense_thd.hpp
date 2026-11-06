// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_BWD_DENSE_THD_BF16_INDEXER_BWD_DENSE_THD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_BWD_DENSE_THD_BF16_INDEXER_BWD_DENSE_THD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/indexer_bwd_dense_thd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool bf16IndexerBwdDenseThdLaunchFits(int heads, int width, int columns, int queries);

bool bf16IndexerBwdDenseThdDenseMmacSupported();
void bf16IndexerBwdDenseThdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
using Bf16IndexerBwdDenseThdLaunch = bool (*)(FTrainStream stream, const void* index_q, const void* weights,
                                              const void* index_k, const float* attn_score, const float* attn_l1norm,
                                              float* index_score, const float* index_lse, void* dq, void* dw, void* dk,
                                              float* dk_float, int* done_count, const int* cu_seqlens_q,
                                              const int* cu_seqlens_k, const int* q_causal_offsets, int total_q,
                                              int total_k, int max_seqlen_kv, int heads, int head_dim,
                                              int num_sequences, int ratio, float sm_scale, float loss_coeff,
                                              float grad_loss);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int WIDTH, bool HAS_PAIR, bool HAS_PROOF>
bool launchBf16IndexerBwdDenseThd(FTrainStream stream, const void* index_q, const void* weights, const void* index_k,
                                  const float* attn_score, const float* attn_l1norm, float* index_score,
                                  const float* index_lse, void* dq, void* dw, void* dk, float* dk_float,
                                  int* done_count, const int* cu_seqlens_q, const int* cu_seqlens_k,
                                  const int* q_causal_offsets, int total_q, int total_k, int max_seqlen_kv, int heads,
                                  int head_dim, int num_sequences, int ratio, float sm_scale, float loss_coeff,
                                  float grad_loss);

// One dtype chain of the IndexerBwdDenseThd family: the ported packed-batch
// dense indexer backward with its FP32 dk accumulator and completion counter.
template<class Implementation>
class Bf16IndexerBwdDenseThdBase : public Primitive<IndexerBwdDenseThdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kFused   = false;
    static constexpr bool kPair    = false;
    static constexpr bool kProof   = false;
    static constexpr bool kTwoWave = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const IndexerBwdDenseThdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const IndexerBwdDenseThdProblem& problem) override;

  protected:
    Bf16IndexerBwdDenseThdBase()                                  = default;
    Bf16IndexerBwdDenseThdBase(const Bf16IndexerBwdDenseThdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16IndexerBwdDenseThdLaunch launch_ = nullptr;

    const void* index_q_             = nullptr;
    const void* weights_             = nullptr;
    const void* index_k_             = nullptr;
    const float* attn_score_         = nullptr;
    const float* attn_l1norm_        = nullptr;
    const float* index_lse_          = nullptr;
    const int* cu_seqlens_q_         = nullptr;
    const int* cu_seqlens_k_         = nullptr;
    const int* causal_offsets_       = nullptr;
    float* score_gradient_           = nullptr;
    void* dq_                        = nullptr;
    void* dw_                        = nullptr;
    void* dk_                        = nullptr;
    int total_q_                     = 0;
    int total_k_                     = 0;
    int max_seqlen_kv_               = 0;
    int heads_                       = 0;
    int head_dim_                    = 0;
    int num_sequences_               = 0;
    int ratio_                       = 0;
    float sm_scale_                  = 0.0F;
    float loss_coeff_                = 0.0F;
    float grad_loss_                 = 0.0F;
    std::uint64_t workspace_bytes_   = 0;
    std::uint64_t dk_float_offset_   = 0;
    std::uint64_t done_count_offset_ = 0;
};

class Bf16IndexerBwdDenseThd final : public Bf16IndexerBwdDenseThdBase<Bf16IndexerBwdDenseThd> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdDenseThd";
    static constexpr bool kFused       = true;
};

class Bf16IndexerBwdDenseThdTwoWave final : public Bf16IndexerBwdDenseThdBase<Bf16IndexerBwdDenseThdTwoWave> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdDenseThdTwoWave";
    static constexpr bool kTwoWave     = true;
};

class Bf16IndexerBwdDenseThdTwoWavePair final : public Bf16IndexerBwdDenseThdBase<Bf16IndexerBwdDenseThdTwoWavePair> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdDenseThdTwoWavePair";
    static constexpr bool kTwoWave     = true;
    static constexpr bool kPair        = true;
};

}  // namespace ftrain

#endif
