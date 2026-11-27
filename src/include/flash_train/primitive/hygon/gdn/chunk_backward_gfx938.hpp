// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_GDN_CHUNK_BACKWARD_GFX938_HPP_
#define FTRAIN_PRIMITIVE_HYGON_GDN_CHUNK_BACKWARD_GFX938_HPP_

#include "flash_train/family/gdn/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// GDN chunked backward for gfx938, migrated from the linear_atten GDN archive
// 13_gfx938_aicc_port.  Unlike the gfx936 candidate the archive's backward
// entry keeps its workspace in a process-level cache (single-stream use), so
// the FlashTrain workspace only carries the optional state-layout conversion
// buffers; h0 is passed K-first (the entry transposes internally), dht/dh0
// are converted to/from the V-first layout by the wrapper.
class GdnChunkBackwardGfx938 final : public Primitive<GdnBackwardProblem> {
  public:
    const char* getName() const noexcept override { return "GdnChunkBackwardBf16Gfx938"; }
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const GdnBackwardProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return workspace_bytes_; }
    void configure(const GdnBackwardProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const std::uint16_t* q_ = nullptr;
    const std::uint16_t* k_ = nullptr;
    const std::uint16_t* v_ = nullptr;
    const float* g_ = nullptr;
    const float* beta_ = nullptr;
    const std::uint16_t* do_ = nullptr;
    const float* h0_ = nullptr;
    const float* dht_ = nullptr;
    std::uint16_t* dq_ = nullptr;
    std::uint16_t* dk_ = nullptr;
    std::uint16_t* dv_ = nullptr;
    float* dg_ = nullptr;
    float* dbeta_ = nullptr;
    float* dh0_ = nullptr;
    int b_ = 0, s_ = 0, h_ = 0, hv_ = 0, kdim_ = 0, vdim_ = 0;
    float scale_ = 0.0F;
    std::uint64_t core_bytes_ = 0, dht_offset_ = 0, dh0_offset_ = 0, workspace_bytes_ = 0;
};

}  // namespace ftrain

#endif
