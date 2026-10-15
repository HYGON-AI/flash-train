// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_BWD_DENSE_THD_MIXED_FP32_INDEXER_BWD_DENSE_THD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_BWD_DENSE_THD_MIXED_FP32_INDEXER_BWD_DENSE_THD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/indexer_bwd_dense_thd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool mixedFp32IndexerBwdDenseThdLaunchFits(int heads, int width, int columns, int queries);

void castMixedFp32IndexerBwdDenseThd(const void* source, void* destination, FTrainNumericType source_type,
                                     FTrainNumericType destination_type, std::uint64_t elements, FTrainStream stream);

// Defined in mixed_fp32_indexer_bwd_dense_thd.hip.
bool mixedFp32IndexerBwdDenseThdDenseMmacSupported();
void mixedFp32IndexerBwdDenseThdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
bool launchMixedFp32IndexerBwdDenseThd(FTrainStream stream, const void* index_q, const void* weights,
                                       const void* index_k, const float* attn_score, const float* attn_l1norm,
                                       float* index_score, const float* index_lse, void* dq, void* dw, void* dk,
                                       float* dk_float, int* done_count, const int* cu_seqlens_q,
                                       const int* cu_seqlens_k, const int* q_causal_offsets, int total_q, int total_k,
                                       int max_seqlen_kv, int heads, int head_dim, int num_sequences, int ratio,
                                       float sm_scale, float loss_coeff, float grad_loss);

// Mixed input storage follows upstream FP32 promotion in caller-owned workspace.
// Results retain the original port dtypes; selection checks remain in isApplicable.
class MixedFp32IndexerBwdDenseThd final : public Primitive<IndexerBwdDenseThdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const IndexerBwdDenseThdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const IndexerBwdDenseThdProblem& problem) override;

  private:
    FTrainNumericType index_q_type_          = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t index_q_elements_          = 0;
    std::uint64_t index_q_conversion_offset_ = 0;
    FTrainNumericType weights_type_          = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t weights_elements_          = 0;
    std::uint64_t weights_conversion_offset_ = 0;
    FTrainNumericType index_k_type_          = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t index_k_elements_          = 0;
    std::uint64_t index_k_conversion_offset_ = 0;
    FTrainNumericType dq_type_               = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t dq_elements_               = 0;
    std::uint64_t dq_conversion_offset_      = 0;
    FTrainNumericType dw_type_               = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t dw_elements_               = 0;
    std::uint64_t dw_conversion_offset_      = 0;
    FTrainNumericType dk_type_               = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t dk_elements_               = 0;
    std::uint64_t dk_conversion_offset_      = 0;
    void executeImpl(const Resources& resources) override;

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

}  // namespace ftrain

#endif
