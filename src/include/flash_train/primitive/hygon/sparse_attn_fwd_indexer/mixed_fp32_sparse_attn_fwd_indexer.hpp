// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_FWD_INDEXER_MIXED_FP32_SPARSE_ATTN_FWD_INDEXER_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_FWD_INDEXER_MIXED_FP32_SPARSE_ATTN_FWD_INDEXER_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/sparse_attn_fwd_indexer/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool mixedFp32SparseAttnFwdIndexerLaunchFits(int width, int columns);

// Fixed conversion launches; conversion direction is selected in the C++ translation unit.
void launchMixedFp32SparseAttnFwdIndexerCastBf16ToFp32(const void* source, void* destination, std::uint64_t elements,
                                                       FTrainStream stream);
void launchMixedFp32SparseAttnFwdIndexerCastFp16ToFp32(const void* source, void* destination, std::uint64_t elements,
                                                       FTrainStream stream);
void launchMixedFp32SparseAttnFwdIndexerCastFp32ToBf16(const void* source, void* destination, std::uint64_t elements,
                                                       FTrainStream stream);
void launchMixedFp32SparseAttnFwdIndexerCastFp32ToFp16(const void* source, void* destination, std::uint64_t elements,
                                                       FTrainStream stream);

void castMixedFp32SparseAttnFwdIndexer(const void* source, void* destination, FTrainNumericType source_type,
                                       FTrainNumericType destination_type, std::uint64_t elements, FTrainStream stream);

bool mixedFp32SparseAttnFwdIndexerSparseAttnFusedSupported();
using MixedFp32SparseAttnFwdIndexerLaunch = bool (*)(FTrainStream stream, const void* q, const void* kv,
                                                     const float* attn_sink, const int* topk_indices, void* out,
                                                     float* max_logits, float* lse, float* lse_indexer,
                                                     const int* topk_length, int indexer_topk, int tokens, int heads,
                                                     int head_dim, int value_dim, int kv_tokens, int selection_width,
                                                     float softmax_scale);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<int WIDTH>
bool launchMixedFp32SparseAttnFwdIndexer(FTrainStream stream, const void* q, const void* kv, const float* attn_sink,
                                         const int* topk_indices, void* out, float* max_logits, float* lse,
                                         float* lse_indexer, const int* topk_length, int indexer_topk, int tokens,
                                         int heads, int head_dim, int value_dim, int kv_tokens, int selection_width,
                                         float softmax_scale);

// Mixed input storage follows upstream FP32 promotion in caller-owned workspace.
// Results retain the original port dtypes; selection checks remain in isApplicable.
class MixedFp32SparseAttnFwdIndexer final : public Primitive<SparseAttnFwdIndexerProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseAttnFwdIndexerProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseAttnFwdIndexerProblem& problem) override;

  private:
    FTrainNumericType q_type_            = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t q_elements_            = 0;
    std::uint64_t q_conversion_offset_   = 0;
    FTrainNumericType kv_type_           = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t kv_elements_           = 0;
    std::uint64_t kv_conversion_offset_  = 0;
    FTrainNumericType out_type_          = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t out_elements_          = 0;
    std::uint64_t out_conversion_offset_ = 0;
    std::uint64_t workspace_bytes_       = 0;

    void executeImpl(const Resources& resources) override;
    MixedFp32SparseAttnFwdIndexerLaunch launch_ = nullptr;

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

}  // namespace ftrain

#endif
