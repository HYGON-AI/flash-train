// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_BWD_BF16_SPARSE_ATTN_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_BWD_BF16_SPARSE_ATTN_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/sparse_attn_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool bf16SparseAttnBwdLaunchFits(int width, int columns);

bool bf16SparseAttnBwdSparseAttnFusedSupported();
void bf16SparseAttnBwdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
using Bf16SparseAttnBwdLaunch = bool (*)(FTrainStream stream, const void* q, const void* kv, const void* out,
                                         const void* dout, const float* lse, const float* attn_sink,
                                         const int* topk_indices, const int* topk_length, void* dq, float* dkv_float,
                                         float* dsink, void* dkv, int* done_count, int tokens, int heads, int head_dim,
                                         int backward_width, int kv_tokens, int selection_width, float softmax_scale);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<int WIDTH, bool COMPACT>
bool launchBf16SparseAttnBwd(FTrainStream stream, const void* q, const void* kv, const void* out, const void* dout,
                             const float* lse, const float* attn_sink, const int* topk_indices, const int* topk_length,
                             void* dq, float* dkv_float, float* dsink, void* dkv, int* done_count, int tokens,
                             int heads, int head_dim, int backward_width, int kv_tokens, int selection_width,
                             float softmax_scale);

// Basic fused sparse-attention backward for one storage precision.
// Workspace holds the FP32 dKV accumulator and its completion counter.
// All initialization and execution use the caller's stream asynchronously.
// Packed/cached/split and shape-specialized implementations are deferred.
// Add each future implementation in its own <precision>_sparse_attn_bwd_<name>
// .hpp/.cpp/.hip files and register it in SparseAttnBwdFamily::makeRecords().
// Rank named candidates statically in the Finder from Problem metadata;
// isApplicable owns each candidate's support boundaries. Do not hide several
// algorithms behind this baseline's executeImpl() or HIP launcher.
class Bf16SparseAttnBwd final : public Primitive<SparseAttnBwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseAttnBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseAttnBwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16SparseAttnBwdLaunch launch_ = nullptr;

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
