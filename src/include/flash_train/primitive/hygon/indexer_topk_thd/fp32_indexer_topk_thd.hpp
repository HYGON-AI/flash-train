// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_INDEXER_TOPK_THD_FP32_INDEXER_TOPK_THD_HPP_
#define FTRAIN_PRIMITIVE_HYGON_INDEXER_TOPK_THD_FP32_INDEXER_TOPK_THD_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"

#include "flash_train/family/indexer_topk_thd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

using Fp32IndexerTopKThdLaunch = bool (*)(FTrainStream stream, const float* input_values, const int* cu_seqlens_q,
                                          const int* cu_seqlens_kv, const int* q_causal_offsets, int* out_indices,
                                          float* out_values, int num_rows, int num_columns, int top_k,
                                          int num_sequences, int ratio, bool order_values);
// Fixed launch implementations are defined and instantiated in the HIP translation unit.
template<class Implementation, int SORT_SIZE, int CAPACITY, bool ORDER>
bool launchFp32IndexerTopKThd(FTrainStream stream, const float* input_values, const int* cu_seqlens_q,
                              const int* cu_seqlens_kv, const int* q_causal_offsets, int* out_indices,
                              float* out_values, int num_rows, int num_columns, int top_k, int num_sequences, int ratio,
                              bool order_values);

// The IndexerTopKThd Primitive: the ported packed-batch top-K kernel, no
// workspace and no dtype chain.
template<class Implementation>
class Fp32IndexerTopKThdBase : public Primitive<IndexerTopKThdProblem> {
  public:
    // Fixed primitive types select code at compile time; no runtime mode.
    static constexpr int kSortSize   = 1024;
    static constexpr int kCapacity   = 8;
    static constexpr bool kBitonic   = false;
    static constexpr bool kWideRadix = false;

    const char* getName() const noexcept override;
    std::unique_ptr<PrimitiveBase> clone() const override;
    Result isApplicable(const IndexerTopKThdProblem& problem, const Constraints& constraints) const override;
    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;
    void configure(const IndexerTopKThdProblem& problem) override;

  protected:
    Fp32IndexerTopKThdBase()                              = default;
    Fp32IndexerTopKThdBase(const Fp32IndexerTopKThdBase&) = default;

  private:
    void executeImpl(const Resources& resources) override;
    Fp32IndexerTopKThdLaunch launch_ = nullptr;

    const float* input_values_ = nullptr;
    const int* cu_seqlens_q_   = nullptr;
    const int* cu_seqlens_kv_  = nullptr;
    const int* causal_offsets_ = nullptr;
    int* out_indices_          = nullptr;
    float* out_values_         = nullptr;
    int num_rows_              = 0;
    int num_columns_           = 0;
    int top_k_                 = 0;
    int num_sequences_         = 0;
    int ratio_                 = 0;
    bool order_values_         = false;
};

class Fp32IndexerTopKThd final : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThd> {
  public:
    static constexpr const char* kName = "IndexerTopKThd";
    static constexpr bool kBitonic     = true;
};

class Fp32IndexerTopKThdWideRadix final : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdWideRadix> {
  public:
    static constexpr const char* kName = "IndexerTopKThdWideRadix";
    static constexpr bool kWideRadix   = true;
};

class Fp32IndexerTopKThdSize4 final : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdSize4> {
  public:
    static constexpr const char* kName = "IndexerTopKThdSize4";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 4;
};

class Fp32IndexerTopKThdSize8 final : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdSize8> {
  public:
    static constexpr const char* kName = "IndexerTopKThdSize8";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 8;
};

class Fp32IndexerTopKThdSize16 final : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdSize16> {
  public:
    static constexpr const char* kName = "IndexerTopKThdSize16";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 16;
};

class Fp32IndexerTopKThdSize32 final : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdSize32> {
  public:
    static constexpr const char* kName = "IndexerTopKThdSize32";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 32;
};

class Fp32IndexerTopKThdSize64 final : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdSize64> {
  public:
    static constexpr const char* kName = "IndexerTopKThdSize64";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 64;
};

class Fp32IndexerTopKThdSize128 final : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdSize128> {
  public:
    static constexpr const char* kName = "IndexerTopKThdSize128";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 128;
};

class Fp32IndexerTopKThdSize256 final : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdSize256> {
  public:
    static constexpr const char* kName = "IndexerTopKThdSize256";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 256;
};

class Fp32IndexerTopKThdSize512 final : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdSize512> {
  public:
    static constexpr const char* kName = "IndexerTopKThdSize512";
    static constexpr bool kBitonic     = true;
    static constexpr int kSortSize     = 512;
};

class Fp32IndexerTopKThdWideRadixCapacity16 final
    : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdWideRadixCapacity16> {
  public:
    static constexpr const char* kName = "IndexerTopKThdWideRadixCapacity16";
    static constexpr bool kWideRadix   = true;
    static constexpr int kCapacity     = 16;
};

class Fp32IndexerTopKThdWideRadixCapacity32 final
    : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdWideRadixCapacity32> {
  public:
    static constexpr const char* kName = "IndexerTopKThdWideRadixCapacity32";
    static constexpr bool kWideRadix   = true;
    static constexpr int kCapacity     = 32;
};

class Fp32IndexerTopKThdWideRadixCapacity64 final
    : public Fp32IndexerTopKThdBase<Fp32IndexerTopKThdWideRadixCapacity64> {
  public:
    static constexpr const char* kName = "IndexerTopKThdWideRadixCapacity64";
    static constexpr bool kWideRadix   = true;
    static constexpr int kCapacity     = 64;
};

}  // namespace ftrain

#endif
