// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_BWD_DENSE_BF16_INDEXER_BWD_DENSE_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_BWD_DENSE_BF16_INDEXER_BWD_DENSE_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/indexer_bwd_dense/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool bf16IndexerBwdDenseLaunchFits(int heads, int width, int columns, int queries);

bool bf16IndexerBwdDenseDenseMmacSupported();
void bf16IndexerBwdDenseZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
using Bf16IndexerBwdDenseLaunch = bool (*)(FTrainStream stream, const void* index_q, const void* weights,
                                           const void* index_k, const float* attn_score, const float* attn_l1norm,
                                           float* index_score, const float* index_lse, void* dq, void* dw, void* dk,
                                           float* dk_float, int* done_count, int batch, int query_length,
                                           int key_length, int heads, int head_dim, int ratio, float sm_scale,
                                           float loss_coeff, float grad_loss);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int WIDTH, bool HAS_PAIR, bool HAS_PROOF>
bool launchBf16IndexerBwdDense(FTrainStream stream, const void* index_q, const void* weights, const void* index_k,
                               const float* attn_score, const float* attn_l1norm, float* index_score,
                               const float* index_lse, void* dq, void* dw, void* dk, float* dk_float, int* done_count,
                               int batch, int query_length, int key_length, int heads, int head_dim, int ratio,
                               float sm_scale, float loss_coeff, float grad_loss);

// One dtype chain of the IndexerBwdDense family: the ported dense indexer
// backward with its FP32 dk accumulator and completion counter.
template<class Implementation>
class Bf16IndexerBwdDenseBase : public Primitive<IndexerBwdDenseProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kFused   = false;
    static constexpr bool kPair    = false;
    static constexpr bool kProof   = false;
    static constexpr bool kTwoWave = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const IndexerBwdDenseProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const IndexerBwdDenseProblem& problem) override;

  protected:
    Bf16IndexerBwdDenseBase()                               = default;
    Bf16IndexerBwdDenseBase(const Bf16IndexerBwdDenseBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16IndexerBwdDenseLaunch launch_ = nullptr;

    const void* index_q_             = nullptr;
    const void* weights_             = nullptr;
    const void* index_k_             = nullptr;
    const float* attn_score_         = nullptr;
    const float* attn_l1norm_        = nullptr;
    const float* index_lse_          = nullptr;
    float* score_gradient_           = nullptr;
    void* dq_                        = nullptr;
    void* dw_                        = nullptr;
    void* dk_                        = nullptr;
    int batch_                       = 0;
    int query_length_                = 0;
    int key_length_                  = 0;
    int heads_                       = 0;
    int head_dim_                    = 0;
    int ratio_                       = 0;
    float sm_scale_                  = 0.0F;
    float loss_coeff_                = 0.0F;
    float grad_loss_                 = 0.0F;
    std::uint64_t workspace_bytes_   = 0;
    std::uint64_t dk_float_offset_   = 0;
    std::uint64_t done_count_offset_ = 0;
};

class Bf16IndexerBwdDense final : public Bf16IndexerBwdDenseBase<Bf16IndexerBwdDense> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdDense";
    static constexpr bool kFused       = true;
};

class Bf16IndexerBwdDenseTwoWave final : public Bf16IndexerBwdDenseBase<Bf16IndexerBwdDenseTwoWave> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdDenseTwoWave";
    static constexpr bool kTwoWave     = true;
};

class Bf16IndexerBwdDenseTwoWavePair final : public Bf16IndexerBwdDenseBase<Bf16IndexerBwdDenseTwoWavePair> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdDenseTwoWavePair";
    static constexpr bool kTwoWave     = true;
    static constexpr bool kPair        = true;
};

class Bf16IndexerBwdDenseTwoWaveProof final : public Bf16IndexerBwdDenseBase<Bf16IndexerBwdDenseTwoWaveProof> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdDenseTwoWaveProof";
    static constexpr bool kTwoWave     = true;
    static constexpr bool kProof       = true;
};

}  // namespace ftrain

#endif
