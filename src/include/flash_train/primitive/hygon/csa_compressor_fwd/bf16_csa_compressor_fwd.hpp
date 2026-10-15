// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_CSA_COMPRESSOR_FWD_BF16_CSA_COMPRESSOR_FWD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_CSA_COMPRESSOR_FWD_BF16_CSA_COMPRESSOR_FWD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/csa_compressor_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Defined in bf16_csa_compressor_fwd.hip.
bool launchBf16CSACompressorFwd(FTrainStream stream, const void* kv, const void* score, const float* ape,
                                const int* cu_seqlens, const int* cu_seqlens_comp, void* compressed, int total_comp,
                                int num_sequences, int ratio, int head_dim, int coff, int total_tokens);

// The CSACompressorFwd Primitive: the ported fused gated-pooling forward, no
// workspace and no dtype chain.
class Bf16CSACompressorFwd final : public Primitive<CSACompressorFwdProblem> {
  public:
    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const CSACompressorFwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const CSACompressorFwdProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const void* kv_             = nullptr;
    const void* score_          = nullptr;
    const float* ape_           = nullptr;
    const int* cu_seqlens_      = nullptr;
    const int* cu_seqlens_comp_ = nullptr;
    void* compressed_           = nullptr;
    int total_comp_             = 0;
    int num_sequences_          = 0;
    int ratio_                  = 0;
    int head_dim_               = 0;
    int coff_                   = 0;
    int total_tokens_           = 0;
};

}  // namespace ftrain

#endif
