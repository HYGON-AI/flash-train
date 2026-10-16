// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_INDEXER_SCORE_PROBLEM_HPP_
#define FTRAIN_FAMILY_DENSE_INDEXER_SCORE_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/dense_indexer_score.hpp"

namespace ftrain {

// One DenseIndexerScore call's parameters as a support checklist: the
// full-KV indexer score and its log-sum-exp denominator, computed under the
// ratio-causal mask.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class DenseIndexerScoreProblem final {
  public:
    DenseIndexerScoreProblem(Tensor q, Tensor k, Tensor w, Tensor out, Tensor denom, Tensor ratio, Tensor sm_scale,
                             DenseIndexerScoreAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getQDims() const noexcept { return q_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getKDims() const noexcept { return k_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getWDims() const noexcept { return w_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getOutDims() const noexcept { return out_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getDenomDims() const noexcept { return denom_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getRatioDims() const noexcept { return ratio_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getSmScaleDims() const noexcept { return sm_scale_.getStorageView().getDims(); }

    std::uint64_t getBatch() const noexcept { return dimOf(q_, 0); }
    std::uint64_t getQueryLength() const noexcept { return dimOf(q_, 1); }
    std::uint64_t getKeyLength() const noexcept { return dimOf(k_, 1); }
    std::uint64_t getQueryHeads() const noexcept { return dimOf(q_, 2); }
    std::uint64_t getHeadDim() const noexcept { return dimOf(q_, 3); }

    bool hasSingleKvHead() const noexcept { return dimOf(k_, 2) == 1; }

    // --- attributes ---

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }
    std::int64_t getRatio() const noexcept { return ratio_value_; }

    // --- element types ---

    FTrainNumericType getQNumericType() const noexcept { return numericTypeOf(q_); }
    FTrainNumericType getKNumericType() const noexcept { return numericTypeOf(k_); }
    FTrainNumericType getWNumericType() const noexcept { return numericTypeOf(w_); }
    FTrainNumericType getOutNumericType() const noexcept { return numericTypeOf(out_); }
    FTrainNumericType getDenomNumericType() const noexcept { return numericTypeOf(denom_); }

    // --- layouts ---

    FTrainIndexType getQIndexType() const noexcept { return indexTypeOf(q_); }
    FTrainIndexType getKIndexType() const noexcept { return indexTypeOf(k_); }
    FTrainIndexType getWIndexType() const noexcept { return indexTypeOf(w_); }
    FTrainIndexType getOutIndexType() const noexcept { return indexTypeOf(out_); }
    FTrainIndexType getDenomIndexType() const noexcept { return indexTypeOf(denom_); }

    // --- addresses; every port is required for this operator ---

    void* getQAddress() const noexcept { return q_.getStorageView().getMemory(); }
    void* getKAddress() const noexcept { return k_.getStorageView().getMemory(); }
    void* getWAddress() const noexcept { return w_.getStorageView().getMemory(); }
    void* getOutAddress() const noexcept { return out_.getStorageView().getMemory(); }
    void* getDenomAddress() const noexcept { return denom_.getStorageView().getMemory(); }

    // --- selection cache ---

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getRatioNumericType() const noexcept { return ratio_.getStorageView().getNumericType(); }
    FTrainIndexType getRatioIndexType() const noexcept { return ratio_.getStorageView().getIndexType(); }
    void* getRatioAddress() const noexcept { return ratio_.getStorageView().getMemory(); }
    FTrainNumericType getSmScaleNumericType() const noexcept { return sm_scale_.getStorageView().getNumericType(); }
    FTrainIndexType getSmScaleIndexType() const noexcept { return sm_scale_.getStorageView().getIndexType(); }
    void* getSmScaleAddress() const noexcept { return sm_scale_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dimOf(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }
    static FTrainNumericType numericTypeOf(const Tensor& port) noexcept {
        return port.getStorageView().getNumericType();
    }
    static FTrainIndexType indexTypeOf(const Tensor& port) noexcept { return port.getStorageView().getIndexType(); }

    Tensor q_;
    Tensor k_;
    Tensor w_;
    Tensor out_;
    Tensor denom_;
    Tensor ratio_;
    Tensor sm_scale_;
    std::int64_t ratio_value_{};
    DenseIndexerScoreAttributes attributes_;
};

}  // namespace ftrain

#endif
