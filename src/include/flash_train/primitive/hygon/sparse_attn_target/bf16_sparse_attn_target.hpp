// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_TARGET_BF16_SPARSE_ATTN_TARGET_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_TARGET_BF16_SPARSE_ATTN_TARGET_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/sparse_attn_target/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
bool bf16SparseAttnTargetAlgorithmFits(int heads, int width);

bool bf16SparseAttnTargetSparseMmacSupported();
bool bf16SparseAttnTargetSparseMmacShmFits(int query_heads, int head_dim, int selection_width);
void bf16SparseAttnTargetZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
using Bf16SparseAttnTargetLaunch = bool (*)(FTrainStream stream, const void* q, const void* k, const float* lse,
                                            const int* topk_indices, float* target, int* row_count, int batch,
                                            int query_length, int key_length, int query_heads, int head_dim,
                                            int selection_width, float softmax_scale, bool topk_indices_global);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int WIDTH, int HEADS>
bool launchBf16SparseAttnTarget(FTrainStream stream, const void* q, const void* k, const float* lse,
                                const int* topk_indices, float* target, int* row_count, int batch, int query_length,
                                int key_length, int query_heads, int head_dim, int selection_width, float softmax_scale,
                                bool topk_indices_global);

// One dtype chain of the SparseAttnTarget family. Like the other sparse
// MMAC gather kernels this one owns a per-row counter it uses as a
// last-block sync, so the Primitive reserves and clears that counter.
template<class Implementation>
class Bf16SparseAttnTargetBase : public Primitive<SparseAttnTargetProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kLdsQ      = false;
    static constexpr bool kGlobalQ   = false;
    static constexpr bool kTarget512 = false;
    static constexpr bool kTarget576 = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseAttnTargetProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseAttnTargetProblem& problem) override;

  protected:
    Bf16SparseAttnTargetBase()                                = default;
    Bf16SparseAttnTargetBase(const Bf16SparseAttnTargetBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16SparseAttnTargetLaunch launch_ = nullptr;

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

class Bf16SparseAttnTarget final : public Bf16SparseAttnTargetBase<Bf16SparseAttnTarget> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnTarget";
    static constexpr bool kLdsQ        = true;
};

class Bf16SparseAttnTargetGlobalQ final : public Bf16SparseAttnTargetBase<Bf16SparseAttnTargetGlobalQ> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnTargetGlobalQ";
    static constexpr bool kGlobalQ     = true;
};

class Bf16SparseAttnTargetTarget512 final : public Bf16SparseAttnTargetBase<Bf16SparseAttnTargetTarget512> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnTargetTarget512";
    static constexpr bool kTarget512   = true;
};

class Bf16SparseAttnTargetTarget576 final : public Bf16SparseAttnTargetBase<Bf16SparseAttnTargetTarget576> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnTargetTarget576";
    static constexpr bool kTarget576   = true;
};

}  // namespace ftrain

#endif
