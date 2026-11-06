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

bool bf16SparseAttnBwdLaunchFits(int width, int columns, int heads, bool fused);

bool bf16SparseAttnBwdSparseAttnFusedSupported();
void bf16SparseAttnBwdZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
// Resolve concrete template symbols once while configuring the selected primitive.
// The returned function submits a fixed sequence and never selects kernels by shape.
using Bf16SparseAttnBwdLaunch = bool (*)(FTrainStream, const void*, const void*, const void*, const void*, const float*,
                                         const float*, const int*, const int*, void*, float*, float*, void*, int*,
                                         void*, int, int, int, int, int, int, float);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int ND, int HEADS, int TOPK, bool COMPACT, bool RAW, bool RAW_K>
bool launchBf16SparseAttnBwd(FTrainStream stream, const void* q, const void* kv, const void* out, const void* dout,
                             const float* lse, const float* sink, const int* indices, const int* lengths, void* dq,
                             float* dkv_float, float* dsink, void* dkv, int* counters, void* packed, int T, int H,
                             int D, int V, int S, int K, float scale);

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
template<class Implementation>
class Bf16SparseAttnBwdBase : public Primitive<SparseAttnBwdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kFused           = false;
    static constexpr bool kPacked          = false;
    static constexpr bool kCached          = false;
    static constexpr bool kCachedLowMemory = false;
    static constexpr bool kRegisterFused   = false;
    static constexpr bool kSplit = false;
    static constexpr bool kNoPrefetch = false;
    static constexpr bool kLocalFusion = false;
    static constexpr bool kKeyPacked = false;
    static constexpr bool kGrouped = false;
    static constexpr bool kCompact = false;
    static constexpr bool kGlmPair = false;
    static constexpr bool kGlmLocal = false;
    static constexpr bool kPadded = false;
    static constexpr bool kCompactValues   = false;
    static constexpr bool kGlobalIndices   = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseAttnBwdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseAttnBwdProblem& problem) override;

  protected:
    Bf16SparseAttnBwdBase()                             = default;
    Bf16SparseAttnBwdBase(const Bf16SparseAttnBwdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16SparseAttnBwdLaunch launch_ = nullptr;

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
    std::uint64_t workspace_bytes_     = 0;
    std::uint64_t dkv_float_offset_    = 0;
    std::uint64_t done_count_offset_   = 0;
    std::uint64_t packed_stats_offset_ = 0;
};

class Bf16SparseAttnBwd final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwd> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwd";
    static constexpr bool kFused       = true;
};

class Bf16SparseAttnBwdPacked final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdPacked> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwdPacked";
    static constexpr bool kPacked      = true;
};

class Bf16SparseAttnBwdCached final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCached> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwdCached";
    static constexpr bool kCached      = true;
};

class Bf16SparseAttnBwdCachedLowMemory final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedLowMemory> {
  public:
    static constexpr const char* kName     = "Bf16SparseAttnBwdCachedLowMemory";
    static constexpr bool kCachedLowMemory = true;
    static constexpr bool kPadded          = true;
    static constexpr bool kCompactValues   = true;
};

class Bf16SparseAttnBwdRegisterFused final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdRegisterFused> {
  public:
    static constexpr const char* kName   = "Bf16SparseAttnBwdRegisterFused";
    static constexpr bool kRegisterFused = true;
};


// Each named record fixes one execution schedule; shape eligibility belongs to isApplicable.
class Bf16SparseAttnBwdPackedSplit final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdPackedSplit> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwdPackedSplit";
    static constexpr bool kPacked = true;
    static constexpr bool kSplit = true;
};

class Bf16SparseAttnBwdCachedNoPrefetch final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedNoPrefetch> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwdCachedNoPrefetch";
    static constexpr bool kCached = true;
    static constexpr bool kNoPrefetch = true;
};

class Bf16SparseAttnBwdCachedLocal final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedLocal> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwdCachedLocal";
    static constexpr bool kCached = true;
    static constexpr bool kLocalFusion = true;
};

class Bf16SparseAttnBwdCachedLocalAligned final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedLocalAligned> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwdCachedLocalAligned";
    static constexpr bool kCached = true;
    static constexpr bool kLocalFusion = true;
    static constexpr bool kKeyPacked = true;
};

class Bf16SparseAttnBwdCachedGrouped final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGrouped> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwdCachedGrouped";
    static constexpr bool kCached = true;
    static constexpr bool kLocalFusion = true;
    static constexpr bool kKeyPacked = true;
    static constexpr bool kGrouped = true;
};

class Bf16SparseAttnBwdCachedCompact final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedCompact> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwdCachedCompact";
    static constexpr bool kCached = true;
    static constexpr bool kCompact = true;
};

class Bf16SparseAttnBwdCachedGlmPair final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGlmPair> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwdCachedGlmPair";
    static constexpr bool kCached = true;
    static constexpr bool kGlmPair = true;
    static constexpr bool kCompactValues = true;
};

class Bf16SparseAttnBwdCachedGlmLocal final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGlmLocal> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwdCachedGlmLocal";
    static constexpr bool kCached = true;
    static constexpr bool kGlmLocal = true;
    static constexpr bool kCompactValues = true;
};

class Bf16SparseAttnBwdCachedPadded final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedPadded> {
  public:
    static constexpr const char* kName = "Bf16SparseAttnBwdCachedPadded";
    static constexpr bool kCached = true;
    static constexpr bool kPadded = true;
    static constexpr bool kCompactValues = true;
};

class Bf16SparseAttnBwdCachedGlobalIndices final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGlobalIndices> {
  public:
    static constexpr const char* kName   = "Bf16SparseAttnBwdCachedGlobalIndices";
    static constexpr bool kCached        = true;
    static constexpr bool kGlobalIndices = true;
};

class Bf16SparseAttnBwdCachedNoPrefetchGlobalIndices final
    : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedNoPrefetchGlobalIndices> {
  public:
    static constexpr const char* kName   = "Bf16SparseAttnBwdCachedNoPrefetchGlobalIndices";
    static constexpr bool kCached        = true;
    static constexpr bool kNoPrefetch    = true;
    static constexpr bool kGlobalIndices = true;
};

class Bf16SparseAttnBwdCachedGlm final : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGlm> {
  public:
    static constexpr const char* kName   = "Bf16SparseAttnBwdCachedGlm";
    static constexpr bool kCached        = true;
    static constexpr bool kCompactValues = true;
};

class Bf16SparseAttnBwdCachedGlmGlobalIndices final
    : public Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGlmGlobalIndices> {
  public:
    static constexpr const char* kName   = "Bf16SparseAttnBwdCachedGlmGlobalIndices";
    static constexpr bool kCached        = true;
    static constexpr bool kCompactValues = true;
    static constexpr bool kGlobalIndices = true;
};

}  // namespace ftrain

#endif
