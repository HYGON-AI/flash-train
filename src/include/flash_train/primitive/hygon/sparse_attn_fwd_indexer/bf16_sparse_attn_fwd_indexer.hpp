// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_FWD_INDEXER_BF16_SPARSE_ATTN_FWD_INDEXER_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_FWD_INDEXER_BF16_SPARSE_ATTN_FWD_INDEXER_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/sparse_attn_fwd_indexer/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
bool bf16SparseAttnFwdIndexerLaunchFits(int width, int columns);

bool bf16SparseAttnFwdIndexerSparseAttnFusedSupported();
using Bf16SparseAttnFwdIndexerLaunch = bool (*)(FTrainStream stream, const void* q, const void* kv,
                                                const float* attn_sink, const int* topk_indices, void* out,
                                                float* max_logits, float* lse, float* lse_indexer,
                                                const int* topk_length, int indexer_topk, int tokens, int heads,
                                                int head_dim, int value_dim, int kv_tokens, int selection_width,
                                                float softmax_scale);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int ND, int TOPK>
bool launchBf16SparseAttnFwdIndexer(FTrainStream stream, const void* q, const void* kv, const float* attn_sink,
                                    const int* topk_indices, void* out, float* max_logits, float* lse,
                                    float* lse_indexer, const int* topk_length, int indexer_topk, int tokens, int heads,
                                    int head_dim, int value_dim, int kv_tokens, int selection_width,
                                    float softmax_scale);

// One dtype chain of the SparseAttnFwdIndexer family: the ported fused MMAC
// sparse attention forward with the indexer statistics, no workspace.
template<class Implementation>
class Bf16SparseAttnFwdIndexerBase : public Primitive<SparseAttnFwdIndexerProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kFused       = false;
    static constexpr bool kTileIndex   = false;
    static constexpr bool kWindow640   = false;
    static constexpr bool kMla64       = false;
    static constexpr bool kMlaMasked   = false;
    static constexpr bool kMlaTail160  = false;
    static constexpr bool kMlaTail256  = false;
    static constexpr bool kMlaPartial2 = false;
    static constexpr bool kMlaPartial4 = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseAttnFwdIndexerProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseAttnFwdIndexerProblem& problem) override;

  protected:
    Bf16SparseAttnFwdIndexerBase()                                    = default;
    Bf16SparseAttnFwdIndexerBase(const Bf16SparseAttnFwdIndexerBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16SparseAttnFwdIndexerLaunch launch_ = nullptr;

    const void* q_           = nullptr;
    const void* kv_          = nullptr;
    const float* attn_sink_  = nullptr;
    const int* topk_indices_ = nullptr;
    const int* topk_length_  = nullptr;
    void* out_               = nullptr;
    float* max_logits_       = nullptr;
    float* lse_              = nullptr;
    float* lse_indexer_      = nullptr;
    int indexer_topk_        = 0;
    int tokens_              = 0;
    int heads_               = 0;
    int head_dim_            = 0;
    int value_dim_           = 0;
    int kv_tokens_           = 0;
    int selection_width_     = 0;
    float softmax_scale_     = 0.0F;
};

class Bf16SparseAttnFwdIndexer final : public Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexer> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdIndexer";
    static constexpr bool kFused       = true;
};

class Bf16SparseAttnFwdIndexerMla64 final : public Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMla64> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdIndexerMla64";
    static constexpr bool kMla64       = true;
};

class Bf16SparseAttnFwdIndexerMlaMasked final : public Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMlaMasked> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdIndexerMlaMasked";
    static constexpr bool kMlaMasked   = true;
};

class Bf16SparseAttnFwdIndexerMlaTail160 final
    : public Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMlaTail160> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdIndexerMlaTail160";
    static constexpr bool kMlaTail160  = true;
};

class Bf16SparseAttnFwdIndexerMlaTail256 final
    : public Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMlaTail256> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdIndexerMlaTail256";
    static constexpr bool kMlaTail256  = true;
};

class Bf16SparseAttnFwdIndexerMlaPartial2 final
    : public Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMlaPartial2> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdIndexerMlaPartial2";
    static constexpr bool kMlaPartial2 = true;
};

class Bf16SparseAttnFwdIndexerMlaPartial4 final
    : public Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMlaPartial4> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdIndexerMlaPartial4";
    static constexpr bool kMlaPartial4 = true;
};

class Bf16SparseAttnFwdIndexerTileIndex final : public Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerTileIndex> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdIndexerTileIndex";
    static constexpr bool kFused = true;
    static constexpr bool kTileIndex = true;
};

class Bf16SparseAttnFwdIndexerWindow640 final : public Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerWindow640> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdIndexerWindow640";
    static constexpr bool kFused = true;
    static constexpr bool kWindow640 = true;
};

}  // namespace ftrain

#endif
