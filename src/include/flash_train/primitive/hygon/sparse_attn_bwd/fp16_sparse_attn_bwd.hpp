// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_BWD_FP16_SPARSE_ATTN_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_ATTN_BWD_FP16_SPARSE_ATTN_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/sparse_attn_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool fp16SparseAttnBwdLaunchFits(int width, int columns);

// Defined in fp16_sparse_attn_bwd.hip.
bool fp16SparseAttnBwdSparseAttnFusedSupported();
void fp16SparseAttnBwdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
bool launchFp16SparseAttnBwdFused2(FTrainStream stream, const void* q, const void* kv, const void* out,
                                   const void* dout, const float* lse, const float* attn_sink, const int* topk_indices,
                                   const int* topk_length, void* dq, float* dkv_float, float* dsink, void* dkv,
                                   int* done_count, void* pdl_ws, int tokens, int heads, int head_dim,
                                   int backward_width, int kv_tokens, int selection_width, float softmax_scale);
bool launchFp16SparseAttnBwdFused(FTrainStream stream, const void* q, const void* kv, const void* out, const void* dout,
                                  const float* lse, const float* attn_sink, const int* topk_indices,
                                  const int* topk_length, void* dq, float* dkv_float, float* dsink, void* dkv,
                                  int* done_count, int tokens, int heads, int head_dim, int backward_width,
                                  int kv_tokens, int selection_width, float softmax_scale);

// One dtype chain of the SparseAttnBwd family: the ported fused MMAC sparse
// attention backward.
//
// The kernel accumulates dkv in FP32 and lets the last arriving blocks cast
// the accumulator into dkv once a completion counter reaches the total block
// count, so the Primitive owns three workspace regions -- the FP32
// accumulator, that counter plus its two internal dispatch counters, and the
// packed-stats scratch the large-D variant reads back. The upstream launcher
// clears its accumulator; the Primitive clears counters and the sink gradient
// on the stream before launch.
class Fp16SparseAttnBwd final : public Primitive<SparseAttnBwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseAttnBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseAttnBwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* q_                     = nullptr;
    const void* kv_                    = nullptr;
    const void* out_                   = nullptr;
    const void* dout_                  = nullptr;
    const float* lse_                  = nullptr;
    const float* attn_sink_            = nullptr;
    const int* topk_indices_           = nullptr;
    const int* topk_length_            = nullptr;
    void* dq_                          = nullptr;
    void* dkv_                         = nullptr;
    float* dsink_                      = nullptr;
    int tokens_                        = 0;
    int heads_                         = 0;
    int head_dim_                      = 0;
    int backward_width_                = 0;
    int kv_tokens_                     = 0;
    int selection_width_               = 0;
    float softmax_scale_               = 0.0F;
    bool use_packed_stats_             = false;
    std::uint64_t workspace_bytes_     = 0;
    std::uint64_t dkv_float_offset_    = 0;
    std::uint64_t done_count_offset_   = 0;
    std::uint64_t packed_stats_offset_ = 0;
};

}  // namespace ftrain

#endif
