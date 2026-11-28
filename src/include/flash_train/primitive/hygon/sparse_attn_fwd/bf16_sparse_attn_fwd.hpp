// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_FWD_BF16_SPARSE_ATTN_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_FWD_BF16_SPARSE_ATTN_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/sparse_attn_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool bf16SparseAttnFwdLaunchFits(int width, int columns);

bool bf16SparseAttnFwdSparseAttnFusedSupported();
using Bf16SparseAttnFwdLaunch = bool (*)(FTrainStream stream, const void* q, const void* kv, const float* attn_sink,
                                         const int* topk_indices, void* out, float* lse, const int* topk_length,
                                         int tokens, int heads, int head_dim, int value_dim, int kv_tokens,
                                         int selection_width, float softmax_scale);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int ND, int TOPK>
bool launchBf16SparseAttnFwd(FTrainStream stream, const void* q, const void* kv, const float* attn_sink,
                             const int* topk_indices, void* out, float* lse, const int* topk_length, int tokens,
                             int heads, int head_dim, int value_dim, int kv_tokens, int selection_width,
                             float softmax_scale);

// One dtype chain of the SparseAttnFwd family: the ported fused MMAC sparse
// attention forward, no workspace.
template<class Implementation>
class Bf16SparseAttnFwdBase : public Primitive<SparseAttnFwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kFused        = false;
    static constexpr bool kDuplicate576 = false;
    static constexpr bool kMla64        = false;
    static constexpr bool kMlaMasked    = false;
    static constexpr bool kMlaTail160   = false;
    static constexpr bool kMlaTail256   = false;
    static constexpr bool kMlaPartial2  = false;
    static constexpr bool kMlaPartial4  = false;
    static constexpr bool kShortTwoPass = false;
    static constexpr bool kCsaPadded    = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseAttnFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseAttnFwdProblem& problem) override;

  protected:
    Bf16SparseAttnFwdBase()                             = default;
    Bf16SparseAttnFwdBase(const Bf16SparseAttnFwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16SparseAttnFwdLaunch launch_ = nullptr;

    const void* q_           = nullptr;
    const void* kv_          = nullptr;
    const float* attn_sink_  = nullptr;
    const int* topk_indices_ = nullptr;
    const int* topk_length_  = nullptr;
    void* out_               = nullptr;
    float* lse_              = nullptr;
    int tokens_              = 0;
    int heads_               = 0;
    int head_dim_            = 0;
    int value_dim_           = 0;
    int kv_tokens_           = 0;
    int selection_width_     = 0;
    float softmax_scale_     = 0.0F;
};

class Bf16SparseAttnFwd final : public Bf16SparseAttnFwdBase<Bf16SparseAttnFwd> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwd";
    static constexpr bool kFused       = true;
};

class Bf16SparseAttnFwdMla64 final : public Bf16SparseAttnFwdBase<Bf16SparseAttnFwdMla64> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdMla64";
    static constexpr bool kMla64       = true;
};

class Bf16SparseAttnFwdMlaMasked final : public Bf16SparseAttnFwdBase<Bf16SparseAttnFwdMlaMasked> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdMlaMasked";
    static constexpr bool kMlaMasked   = true;
};

class Bf16SparseAttnFwdMlaTail160 final : public Bf16SparseAttnFwdBase<Bf16SparseAttnFwdMlaTail160> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdMlaTail160";
    static constexpr bool kMlaTail160  = true;
};

class Bf16SparseAttnFwdMlaTail256 final : public Bf16SparseAttnFwdBase<Bf16SparseAttnFwdMlaTail256> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdMlaTail256";
    static constexpr bool kMlaTail256  = true;
};

class Bf16SparseAttnFwdMlaPartial2 final : public Bf16SparseAttnFwdBase<Bf16SparseAttnFwdMlaPartial2> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdMlaPartial2";
    static constexpr bool kMlaPartial2 = true;
};

class Bf16SparseAttnFwdMlaPartial4 final : public Bf16SparseAttnFwdBase<Bf16SparseAttnFwdMlaPartial4> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdMlaPartial4";
    static constexpr bool kMlaPartial4 = true;
};

class Bf16SparseAttnFwdShortTwoPass final : public Bf16SparseAttnFwdBase<Bf16SparseAttnFwdShortTwoPass> {
  public:
    static constexpr const char* kName  = "Bf16SparseAttnFwdShortTwoPass";
    static constexpr bool kShortTwoPass = true;
};

class Bf16SparseAttnFwdCsaPadded final : public Bf16SparseAttnFwdBase<Bf16SparseAttnFwdCsaPadded> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnFwdCsaPadded";
    static constexpr bool kCsaPadded   = true;
};

// D576 finite-input classifier plus guarded per-query reuse. Architecture-specific
// V prefetch/rescale scheduling is retained in the device implementation.
class Bf16SparseAttnFwdDuplicate576 final : public Bf16SparseAttnFwdBase<Bf16SparseAttnFwdDuplicate576> {
  public:
    static constexpr const char* kName  = "Bf16SparseAttnFwdDuplicate576";
    static constexpr bool kFused        = true;
    static constexpr bool kDuplicate576 = true;
};

bool launchBf16SparseAttnFwdDuplicate576(FTrainStream stream, const void* q, const void* kv, const float* sink,
                                         const int* indices, const int* lengths, void* out, float* lse, int tokens,
                                         int kv_tokens, int width, float scale, int* flags);

}  // namespace ftrain

#endif
