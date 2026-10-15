// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_FWD_FP16_SPARSE_ATTN_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_FWD_FP16_SPARSE_ATTN_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/sparse_attn_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool fp16SparseAttnFwdLaunchFits(int width, int columns);

// Defined in fp16_sparse_attn_fwd.hip.
bool fp16SparseAttnFwdSparseAttnFusedSupported();
bool launchFp16SparseAttnFwd(FTrainStream stream, const void* q, const void* kv, const float* attn_sink,
                             const int* topk_indices, void* out, float* lse, const int* topk_length, int tokens,
                             int heads, int head_dim, int value_dim, int kv_tokens, int selection_width,
                             float softmax_scale);

// One dtype chain of the SparseAttnFwd family: the ported fused MMAC sparse
// attention forward, no workspace.
class Fp16SparseAttnFwd final : public Primitive<SparseAttnFwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseAttnFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseAttnFwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

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

}  // namespace ftrain

#endif
