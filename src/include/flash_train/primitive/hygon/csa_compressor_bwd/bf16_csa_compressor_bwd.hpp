// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_CSA_COMPRESSOR_BWD_BF16_CSA_COMPRESSOR_BWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_CSA_COMPRESSOR_BWD_BF16_CSA_COMPRESSOR_BWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/csa_compressor_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in bf16_csa_compressor_bwd.hip.
void bf16CSACompressorBwdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
bool launchBf16CSACompressorBwd(FTrainStream stream, const void* kv, const void* score, const float* ape,
                                const int* cu_seqlens, const int* cu_seqlens_comp, const void* grad_compressed,
                                void* grad_kv, void* grad_score, float* grad_ape, int total_comp, int num_sequences,
                                int ratio, int head_dim, int coff, int total_tokens);

// The CSACompressorBwd Primitive: the ported fused gated-pooling backward.
//
// The kernel accumulates its outputs with atomics, so the launch owns them:
// executeImpl clears the key, score, and position-table gradients on the
// stream before the launch, which is what makes "never-consumed positions
// stay exact zero" true for the caller.
class Bf16CSACompressorBwd final : public Primitive<CSACompressorBwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const CSACompressorBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const CSACompressorBwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* kv_                 = nullptr;
    const void* score_              = nullptr;
    const float* ape_               = nullptr;
    const int* cu_seqlens_          = nullptr;
    const int* cu_seqlens_comp_     = nullptr;
    const void* grad_compressed_    = nullptr;
    void* grad_kv_                  = nullptr;
    void* grad_score_               = nullptr;
    float* grad_ape_                = nullptr;
    std::uint64_t grad_kv_bytes_    = 0;
    std::uint64_t grad_score_bytes_ = 0;
    std::uint64_t grad_ape_bytes_   = 0;
    int total_comp_                 = 0;
    int num_sequences_              = 0;
    int ratio_                      = 0;
    int head_dim_                   = 0;
    int coff_                       = 0;
    int total_tokens_               = 0;
};

}  // namespace ftrain

#endif
