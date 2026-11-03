// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_BWD_SPARSE_BF16_INDEXER_BWD_SPARSE_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_BWD_SPARSE_BF16_INDEXER_BWD_SPARSE_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/indexer_bwd_sparse/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {
template<class Implementation>
bool bf16IndexerBwdSparseAlgorithmFits(int heads, int width, int columns);

bool bf16IndexerBwdSparseDenseMmacSupported();
void bf16IndexerBwdSparseZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
using Bf16IndexerBwdSparseLaunch = bool (*)(FTrainStream stream, const void* index_q, const void* weights,
                                            const void* index_k, float* attn_score, const float* index_score,
                                            const int* topk_indices, void* dq, void* dw, void* dk, float* dk_float,
                                            int* done_count, int batch, int query_length, int key_length, int heads,
                                            int head_dim, int selection_width, float sm_scale, float loss_coeff,
                                            float grad_loss, bool topk_indices_global, const float* grad_loss_device);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int WIDTH, int NW, bool DEVICE_SCALE>
bool launchBf16IndexerBwdSparse(FTrainStream stream, const void* index_q, const void* weights, const void* index_k,
                                float* attn_score, const float* index_score, const int* topk_indices, void* dq,
                                void* dw, void* dk, float* dk_float, int* done_count, int batch, int query_length,
                                int key_length, int heads, int head_dim, int selection_width, float sm_scale,
                                float loss_coeff, float grad_loss, bool topk_indices_global,
                                const float* grad_loss_device);

// One dtype chain of the IndexerBwdSparse family. dk accumulates in FP32 and
// is cast by the last arriving block, so the Primitive owns the accumulator
// and the completion counter and clears both on the stream.
template<class Implementation>
class Bf16IndexerBwdSparseBase : public Primitive<IndexerBwdSparseProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kGeneric          = false;
    static constexpr bool kGenericCast      = false;
    static constexpr bool kAligned          = false;
    static constexpr bool kAlignedCast      = false;
    static constexpr bool kCachedScores     = false;
    static constexpr bool kCachedScoresCast = false;
    static constexpr bool kTwoWave          = false;
    static constexpr bool kTwoWaveCast      = false;
    static constexpr bool kPaddedBufferCast = false;
    static constexpr bool kPaddedFlatCast   = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const IndexerBwdSparseProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const IndexerBwdSparseProblem& problem) override;

  protected:
    Bf16IndexerBwdSparseBase()                                = default;
    Bf16IndexerBwdSparseBase(const Bf16IndexerBwdSparseBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16IndexerBwdSparseLaunch launch_ = nullptr;

    const void* index_q_             = nullptr;
    const void* weights_             = nullptr;
    const void* index_k_             = nullptr;
    const void* index_score_         = nullptr;
    const int* topk_indices_         = nullptr;
    float* score_gradient_           = nullptr;
    void* dq_                        = nullptr;
    void* dw_                        = nullptr;
    void* dk_                        = nullptr;
    int batch_                       = 0;
    int query_length_                = 0;
    int key_length_                  = 0;
    int heads_                       = 0;
    int head_dim_                    = 0;
    int selection_width_             = 0;
    float sm_scale_                  = 0.0F;
    float loss_coeff_                = 0.0F;
    const float* grad_loss_device_   = nullptr;
    float grad_loss_                 = 0.0F;
    bool topk_indices_global_        = false;
    std::uint64_t workspace_bytes_   = 0;
    std::uint64_t dk_float_offset_   = 0;
    std::uint64_t done_count_offset_ = 0;
};

class Bf16IndexerBwdSparsePaddedBufferCast final
    : public Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparsePaddedBufferCast> {
  public:
    static constexpr const char* kName      = "Bf16IndexerBwdSparsePaddedBufferCast";
    static constexpr bool kPaddedBufferCast = true;
};

class Bf16IndexerBwdSparsePaddedFlatCast final : public Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparsePaddedFlatCast> {
  public:
    static constexpr const char* kName    = "Bf16IndexerBwdSparsePaddedFlatCast";
    static constexpr bool kPaddedFlatCast = true;
};

class Bf16IndexerBwdSparse final : public Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparse> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdSparse";
    static constexpr bool kGeneric     = true;
};

class Bf16IndexerBwdSparseGenericCast final : public Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseGenericCast> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdSparseGenericCast";
    static constexpr bool kGenericCast = true;
};

class Bf16IndexerBwdSparseAligned final : public Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseAligned> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdSparseAligned";
    static constexpr bool kAligned     = true;
};

class Bf16IndexerBwdSparseAlignedCast final : public Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseAlignedCast> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdSparseAlignedCast";
    static constexpr bool kAlignedCast = true;
};

class Bf16IndexerBwdSparseCachedScores final : public Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseCachedScores> {
  public:
    static constexpr const char* kName  = "Bf16IndexerBwdSparseCachedScores";
    static constexpr bool kCachedScores = true;
};

class Bf16IndexerBwdSparseCachedScoresCast final
    : public Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseCachedScoresCast> {
  public:
    static constexpr const char* kName      = "Bf16IndexerBwdSparseCachedScoresCast";
    static constexpr bool kCachedScoresCast = true;
};

class Bf16IndexerBwdSparseTwoWave final : public Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseTwoWave> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdSparseTwoWave";
    static constexpr bool kTwoWave     = true;
};

class Bf16IndexerBwdSparseTwoWaveCast final : public Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseTwoWaveCast> {
  public:
    static constexpr const char* kName = "Bf16IndexerBwdSparseTwoWaveCast";
    static constexpr bool kTwoWaveCast = true;
};

}  // namespace ftrain

#endif
