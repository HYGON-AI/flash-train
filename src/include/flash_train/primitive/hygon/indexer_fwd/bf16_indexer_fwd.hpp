// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_FWD_BF16_INDEXER_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_FWD_BF16_INDEXER_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/indexer_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

bool bf16IndexerFwdLaunchFits(int heads, int width, int keys);

// Defined in bf16_indexer_fwd.hip.
bool bf16IndexerFwdDenseMmacSupported();
bool launchBf16IndexerFwd(FTrainStream stream, const void* q, const void* k, const void* w, float* scores, int batch,
                          int query_length, int key_length, int query_heads, int kv_heads, int head_dim, int ratio,
                          float sm_scale);

// One dtype chain of the IndexerFwd family: the ported MMAC indexer-forward
// kernel launched for ports of one storage type, no workspace.
//
// The chain fixes both the compute type and the ports' storage type, so a
// BF16 call never reaches the FP32 kernel and vice versa; a mixed call is
// rejected by both and therefore needs a third implementation, exactly as
// docs/development.md's "add an implementation" section describes.
//
// isApplicable checks the implementation limits. Problem keys encode every
// operand property used by selection, including address alignment.
class Bf16IndexerFwd final : public Primitive<IndexerFwdProblem> {
  public:
    const char* getName() const noexcept override;

    std::unique_ptr<PrimitiveBase> clone() const override;

    Result isApplicable(const IndexerFwdProblem& problem, const Constraints& constraints) const override;

    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;

    void configure(const IndexerFwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* q_    = nullptr;
    const void* k_    = nullptr;
    const void* w_    = nullptr;
    float* scores_    = nullptr;
    int batch_        = 0;
    int query_length_ = 0;
    int key_length_   = 0;
    int query_heads_  = 0;
    int head_dim_     = 0;
    int ratio_        = 0;
    float sm_scale_   = 0.0F;
};

}  // namespace ftrain

#endif
