// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_TARGET_FP16_SPARSE_ATTN_TARGET_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_TARGET_FP16_SPARSE_ATTN_TARGET_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/sparse_attn_target/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in fp16_sparse_attn_target.hip.
bool fp16SparseAttnTargetSparseMmacSupported();
bool fp16SparseAttnTargetSparseMmacShmFits(int query_heads, int head_dim, int selection_width);
void fp16SparseAttnTargetZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
bool launchFp16SparseAttnTarget(FTrainStream stream, const void* q, const void* k, const float* lse,
                                const int* topk_indices, float* target, int* row_count, int batch, int query_length,
                                int key_length, int query_heads, int head_dim, int selection_width, float softmax_scale,
                                bool topk_indices_global);

// One dtype chain of the SparseAttnTarget family. Like the other sparse
// MMAC gather kernels this one owns a per-row counter it uses as a
// last-block sync, so the Primitive reserves and clears that counter.
class Fp16SparseAttnTarget final : public Primitive<SparseAttnTargetProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseAttnTargetProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseAttnTargetProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* q_                  = nullptr;
    const void* k_                  = nullptr;
    const float* lse_               = nullptr;
    const int* topk_indices_        = nullptr;
    float* target_                  = nullptr;
    int batch_                      = 0;
    int query_length_               = 0;
    int key_length_                 = 0;
    int query_heads_                = 0;
    int head_dim_                   = 0;
    int selection_width_            = 0;
    float softmax_scale_            = 0.0F;
    bool topk_indices_global_       = false;
    std::uint64_t workspace_bytes_  = 0;
    std::uint64_t row_count_offset_ = 0;
};

}  // namespace ftrain

#endif
