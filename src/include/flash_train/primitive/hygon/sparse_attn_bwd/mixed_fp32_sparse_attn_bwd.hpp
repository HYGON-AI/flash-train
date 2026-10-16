// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_BWD_MIXED_FP32_SPARSE_ATTN_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_BWD_MIXED_FP32_SPARSE_ATTN_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/sparse_attn_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool mixedFp32SparseAttnBwdLaunchFits(int width, int columns);

void castMixedFp32SparseAttnBwd(const void* source, void* destination, FTrainNumericType source_type,
                                FTrainNumericType destination_type, std::uint64_t elements, FTrainStream stream);

// Defined in mixed_fp32_sparse_attn_bwd.hip.
bool mixedFp32SparseAttnBwdSparseAttnFusedSupported();
void mixedFp32SparseAttnBwdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
bool launchMixedFp32SparseAttnBwdFused(FTrainStream stream, const void* q, const void* kv, const void* out,
                                       const void* dout, const float* lse, const float* attn_sink,
                                       const int* topk_indices, const int* topk_length, void* dq, float* dkv_float,
                                       float* dsink, void* dkv, int* done_count, int tokens, int heads, int head_dim,
                                       int backward_width, int kv_tokens, int selection_width, float softmax_scale);

// Mixed input storage follows upstream FP32 promotion in caller-owned workspace.
// Results retain the original port dtypes; selection checks remain in isApplicable.
class MixedFp32SparseAttnBwd final : public Primitive<SparseAttnBwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseAttnBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseAttnBwdProblem& problem) override;

  private:
    FTrainNumericType q_type_             = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t q_elements_             = 0;
    std::uint64_t q_conversion_offset_    = 0;
    FTrainNumericType kv_type_            = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t kv_elements_            = 0;
    std::uint64_t kv_conversion_offset_   = 0;
    FTrainNumericType out_type_           = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t out_elements_           = 0;
    std::uint64_t out_conversion_offset_  = 0;
    FTrainNumericType dout_type_          = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t dout_elements_          = 0;
    std::uint64_t dout_conversion_offset_ = 0;
    FTrainNumericType dq_type_            = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t dq_elements_            = 0;
    std::uint64_t dq_conversion_offset_   = 0;
    FTrainNumericType dkv_type_           = FTRAIN_NUMERIC_TYPE_INVALID;
    std::uint64_t dkv_elements_           = 0;
    std::uint64_t dkv_conversion_offset_  = 0;
    void executeImpl(const Resources& resources) override;

    const void* q_                   = nullptr;
    const void* kv_                  = nullptr;
    const void* out_                 = nullptr;
    const void* dout_                = nullptr;
    const float* lse_                = nullptr;
    const float* attn_sink_          = nullptr;
    const int* topk_indices_         = nullptr;
    const int* topk_length_          = nullptr;
    void* dq_                        = nullptr;
    void* dkv_                       = nullptr;
    float* dsink_                    = nullptr;
    int tokens_                      = 0;
    int heads_                       = 0;
    int head_dim_                    = 0;
    int backward_width_              = 0;
    int kv_tokens_                   = 0;
    int selection_width_             = 0;
    float softmax_scale_             = 0.0F;
    std::uint64_t workspace_bytes_   = 0;
    std::uint64_t dkv_float_offset_  = 0;
    std::uint64_t done_count_offset_ = 0;
};

}  // namespace ftrain

#endif
