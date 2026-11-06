// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_TOPK_FP32_INDEXER_TOPK_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_TOPK_FP32_INDEXER_TOPK_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/indexer_topk/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

using Fp32IndexerTopKLaunch = bool (*)(FTrainStream stream, const float* input_values, const int* seq_lens,
                                       int* out_indices, float* out_values, int num_rows, int num_columns, int top_k,
                                       int batch_size, int next_n, bool order_values);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int SORT_SIZE, int CAPACITY, bool ORDER>
bool launchFp32IndexerTopK(FTrainStream stream, const float* input_values, const int* seq_lens, int* out_indices,
                           float* out_values, int num_rows, int num_columns, int top_k, int batch_size, int next_n,
                           bool order_values);

// The IndexerTopK Primitive: the ported resource-bounded parallel top-K
// kernel, no workspace. Each fixed sort strategy has its own registered record.
template<class Implementation>
class Fp32IndexerTopKBase : public Primitive<IndexerTopKProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr int kSortSize      = 1024;
    static constexpr int kCapacity      = 8;
    static constexpr bool kBitonic      = false;
    static constexpr bool kWideRadix    = false;
    static constexpr bool kRadix1024    = false;
    static constexpr bool kOrderedUInt  = false;
    static constexpr bool kGroupedAll   = false;
    static constexpr bool kGroupedRadix = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const IndexerTopKProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const IndexerTopKProblem& problem) override;

  protected:
    Fp32IndexerTopKBase()                           = default;
    Fp32IndexerTopKBase(const Fp32IndexerTopKBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Fp32IndexerTopKLaunch launch_ = nullptr;

    const float* input_values_ = nullptr;
    const int* seq_lens_       = nullptr;
    int* out_indices_          = nullptr;
    float* out_values_         = nullptr;
    int num_rows_              = 0;
    int num_columns_           = 0;
    int top_k_                 = 0;
    int num_sequences_         = 0;
    int next_n_                = 0;
    bool order_values_         = false;
};

class Fp32IndexerTopKOrderedUInt final : public Fp32IndexerTopKBase<Fp32IndexerTopKOrderedUInt> {
  public:
    static constexpr const char* kName = "IndexerTopKOrderedUInt";
    static constexpr bool kOrderedUInt = true;
};

class Fp32IndexerTopKGroupedAll final : public Fp32IndexerTopKBase<Fp32IndexerTopKGroupedAll> {
  public:
    static constexpr const char* kName = "IndexerTopKGroupedAll";
    static constexpr bool kGroupedAll  = true;
};

class Fp32IndexerTopKGroupedRadix final : public Fp32IndexerTopKBase<Fp32IndexerTopKGroupedRadix> {
  public:
    static constexpr const char* kName  = "IndexerTopKGroupedRadix";
    static constexpr bool kGroupedRadix = true;
};

class Fp32IndexerTopK final : public Fp32IndexerTopKBase<Fp32IndexerTopK> {
  public:
    static constexpr const char* kName = "IndexerTopK";
    static constexpr bool kBitonic     = true;
};

class Fp32IndexerTopKWideRadix final : public Fp32IndexerTopKBase<Fp32IndexerTopKWideRadix> {
  public:
    static constexpr const char* kName = "IndexerTopKWideRadix";
    static constexpr bool kWideRadix   = true;
};

class Fp32IndexerTopKRadix1024 final : public Fp32IndexerTopKBase<Fp32IndexerTopKRadix1024> {
  public:
    static constexpr const char* kName = "IndexerTopKRadix1024";
    static constexpr bool kRadix1024   = true;
};

class Fp32IndexerTopKSize4 final : public Fp32IndexerTopKBase<Fp32IndexerTopKSize4> {
  public:
    static constexpr const char* kName = "IndexerTopKSize4";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 4;
};

class Fp32IndexerTopKSize8 final : public Fp32IndexerTopKBase<Fp32IndexerTopKSize8> {
  public:
    static constexpr const char* kName = "IndexerTopKSize8";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 8;
};

class Fp32IndexerTopKSize16 final : public Fp32IndexerTopKBase<Fp32IndexerTopKSize16> {
  public:
    static constexpr const char* kName = "IndexerTopKSize16";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 16;
};

class Fp32IndexerTopKSize32 final : public Fp32IndexerTopKBase<Fp32IndexerTopKSize32> {
  public:
    static constexpr const char* kName = "IndexerTopKSize32";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 32;
};

class Fp32IndexerTopKSize64 final : public Fp32IndexerTopKBase<Fp32IndexerTopKSize64> {
  public:
    static constexpr const char* kName = "IndexerTopKSize64";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 64;
};

class Fp32IndexerTopKSize128 final : public Fp32IndexerTopKBase<Fp32IndexerTopKSize128> {
  public:
    static constexpr const char* kName = "IndexerTopKSize128";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 128;
};

class Fp32IndexerTopKSize256 final : public Fp32IndexerTopKBase<Fp32IndexerTopKSize256> {
  public:
    static constexpr const char* kName = "IndexerTopKSize256";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 256;
};

class Fp32IndexerTopKSize512 final : public Fp32IndexerTopKBase<Fp32IndexerTopKSize512> {
  public:
    static constexpr const char* kName = "IndexerTopKSize512";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 512;
};

class Fp32IndexerTopKWideRadixCapacity16 final : public Fp32IndexerTopKBase<Fp32IndexerTopKWideRadixCapacity16> {
  public:
    static constexpr const char* kName = "IndexerTopKWideRadixCapacity16";
    static constexpr bool kWideRadix   = true;
    static constexpr int kCapacity     = 16;
};

class Fp32IndexerTopKWideRadixCapacity32 final : public Fp32IndexerTopKBase<Fp32IndexerTopKWideRadixCapacity32> {
  public:
    static constexpr const char* kName = "IndexerTopKWideRadixCapacity32";
    static constexpr bool kWideRadix   = true;
    static constexpr int kCapacity     = 32;
};

class Fp32IndexerTopKWideRadixCapacity64 final : public Fp32IndexerTopKBase<Fp32IndexerTopKWideRadixCapacity64> {
  public:
    static constexpr const char* kName = "IndexerTopKWideRadixCapacity64";
    static constexpr bool kWideRadix   = true;
    static constexpr int kCapacity     = 64;
};

}  // namespace ftrain

#endif
