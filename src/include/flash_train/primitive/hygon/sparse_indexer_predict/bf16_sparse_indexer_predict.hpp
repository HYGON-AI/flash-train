// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_SPARSE_INDEXER_PREDICT_BF16_SPARSE_INDEXER_PREDICT_HPP_
#define FTRAIN_PRIMITIVE_HYGON_SPARSE_INDEXER_PREDICT_BF16_SPARSE_INDEXER_PREDICT_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/sparse_indexer_predict/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

template<class Implementation>
bool bf16SparseIndexerPredictAlgorithmFits(int heads, int width);

bool bf16SparseIndexerPredictSparseMmacSupported();
bool bf16SparseIndexerPredictSparseMmacShmFits(int query_heads, int head_dim, int selection_width);
void bf16SparseIndexerPredictZeroWorkspace(FTrainStream stream, void* memory, std::uint64_t bytes);
using Bf16SparseIndexerPredictLaunch = bool (*)(FTrainStream stream, const void* q, const void* k, const void* w,
                                                const int* topk_indices, float* predict, int* row_count, int batch,
                                                int query_length, int key_length, int query_heads, int head_dim,
                                                int selection_width, float sm_scale, bool topk_indices_global);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int WIDTH, int HEADS>
bool launchBf16SparseIndexerPredict(FTrainStream stream, const void* q, const void* k, const void* w,
                                    const int* topk_indices, float* predict, int* row_count, int batch,
                                    int query_length, int key_length, int query_heads, int head_dim,
                                    int selection_width, float sm_scale, bool topk_indices_global);

// One dtype chain of the SparseIndexerPredict family. The kernel reports the
// number of rows it wrote through a per-row counter array, which this
// Primitive owns as workspace: the counter region is reserved, zeroed on the
// stream, and handed to the launch.
template<class Implementation>
class Bf16SparseIndexerPredictBase : public Primitive<SparseIndexerPredictProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr bool kLdsQ            = false;
    static constexpr bool kGlobalQ         = false;
    static constexpr bool kMmacReduce      = false;
    static constexpr bool kSingle          = false;
    static constexpr bool kShared          = false;
    static constexpr bool kSharedNormCache = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const SparseIndexerPredictProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const SparseIndexerPredictProblem& problem) override;

  protected:
    Bf16SparseIndexerPredictBase()                                    = default;
    Bf16SparseIndexerPredictBase(const Bf16SparseIndexerPredictBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Bf16SparseIndexerPredictLaunch launch_ = nullptr;

    const void* q_                  = nullptr;
    const void* k_                  = nullptr;
    const void* w_                  = nullptr;
    const int* topk_indices_        = nullptr;
    float* predict_                 = nullptr;
    int batch_                      = 0;
    int query_length_               = 0;
    int key_length_                 = 0;
    int query_heads_                = 0;
    int head_dim_                   = 0;
    int selection_width_            = 0;
    float sm_scale_                 = 0.0F;
    bool topk_indices_global_       = false;
    std::uint64_t workspace_bytes_  = 0;
    std::uint64_t row_count_offset_ = 0;
};

class Bf16SparseIndexerPredictSingle final : public Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredictSingle> {
  public:
    static constexpr const char* kName = "Bf16SparseIndexerPredictSingle";
    static constexpr bool kSingle      = true;
};

class Bf16SparseIndexerPredictShared final : public Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredictShared> {
  public:
    static constexpr const char* kName = "Bf16SparseIndexerPredictShared";
    static constexpr bool kShared      = true;
};

class Bf16SparseIndexerPredictSharedNormCache final
    : public Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredictSharedNormCache> {
  public:
    static constexpr const char* kName     = "Bf16SparseIndexerPredictSharedNormCache";
    static constexpr bool kSharedNormCache = true;
};

class Bf16SparseIndexerPredict final : public Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredict> {
  public:
    static constexpr const char* kName = "Bf16SparseIndexerPredict";
    static constexpr bool kLdsQ        = true;
};

class Bf16SparseIndexerPredictGlobalQ final : public Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredictGlobalQ> {
  public:
    static constexpr const char* kName = "Bf16SparseIndexerPredictGlobalQ";
    static constexpr bool kGlobalQ     = true;
};

// Small-D global-Q implementation with full-FP32 MMAC row reduction.
class Bf16SparseIndexerPredictGlobalQMmacReduce final
    : public Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredictGlobalQMmacReduce> {
  public:
    static constexpr const char* kName = "Bf16SparseIndexerPredictGlobalQMmacReduce";
    static constexpr bool kGlobalQ     = true;
    static constexpr bool kMmacReduce  = true;
};

}  // namespace ftrain

#endif
