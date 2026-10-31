// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_ATTN_SCORE_PROBLEM_HPP_
#define FTRAIN_FAMILY_DENSE_ATTN_SCORE_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/dense_attn_score.hpp"

namespace ftrain {

// One DenseAttnScore call's parameters as a support checklist: the full-KV
// attention scores and their row total, both derived from the forward's
// per-head log-sum-exp.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class DenseAttnScoreProblem final {
  public:
    DenseAttnScoreProblem(Tensor q, Tensor k, Tensor lse, Tensor out, Tensor denom, Tensor ratio, Tensor softmax_scale,
                          DenseAttnScoreAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getQDims() const noexcept { return q_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getKDims() const noexcept { return k_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getLseDims() const noexcept { return lse_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getOutDims() const noexcept { return out_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getDenomDims() const noexcept { return denom_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getRatioDims() const noexcept { return ratio_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getSoftmaxScaleDims() const noexcept {
        return softmax_scale_.getStorageView().getDims();
    }

    std::uint64_t getBatch() const noexcept { return dim(q_, 0); }
    std::uint64_t getQueryLength() const noexcept { return dim(q_, 1); }
    std::uint64_t getKeyLength() const noexcept { return dim(k_, 1); }
    std::uint64_t getQueryHeads() const noexcept { return dim(q_, 2); }
    std::uint64_t getHeadDim() const noexcept { return dim(q_, 3); }

    bool hasSingleKvHead() const noexcept { return dim(k_, 2) == 1; }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }
    std::int64_t getRatio() const noexcept { return ratio_value_; }

    FTrainNumericType getQNumericType() const noexcept { return q_.getStorageView().getNumericType(); }
    FTrainNumericType getKNumericType() const noexcept { return k_.getStorageView().getNumericType(); }
    FTrainNumericType getLseNumericType() const noexcept { return lse_.getStorageView().getNumericType(); }
    FTrainNumericType getOutNumericType() const noexcept { return out_.getStorageView().getNumericType(); }
    FTrainNumericType getDenomNumericType() const noexcept { return denom_.getStorageView().getNumericType(); }

    FTrainIndexType getQIndexType() const noexcept { return q_.getStorageView().getIndexType(); }
    FTrainIndexType getKIndexType() const noexcept { return k_.getStorageView().getIndexType(); }
    FTrainIndexType getLseIndexType() const noexcept { return lse_.getStorageView().getIndexType(); }
    FTrainIndexType getOutIndexType() const noexcept { return out_.getStorageView().getIndexType(); }
    FTrainIndexType getDenomIndexType() const noexcept { return denom_.getStorageView().getIndexType(); }

    void* getQAddress() const noexcept { return q_.getStorageView().getMemory(); }
    void* getKAddress() const noexcept { return k_.getStorageView().getMemory(); }
    void* getLseAddress() const noexcept { return lse_.getStorageView().getMemory(); }
    void* getOutAddress() const noexcept { return out_.getStorageView().getMemory(); }
    void* getDenomAddress() const noexcept { return denom_.getStorageView().getMemory(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getRatioNumericType() const noexcept { return ratio_.getStorageView().getNumericType(); }
    FTrainIndexType getRatioIndexType() const noexcept { return ratio_.getStorageView().getIndexType(); }
    void* getRatioAddress() const noexcept { return ratio_.getStorageView().getMemory(); }
    FTrainNumericType getSoftmaxScaleNumericType() const noexcept {
        return softmax_scale_.getStorageView().getNumericType();
    }
    FTrainIndexType getSoftmaxScaleIndexType() const noexcept { return softmax_scale_.getStorageView().getIndexType(); }
    void* getSoftmaxScaleAddress() const noexcept { return softmax_scale_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dim(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }

    Tensor q_;
    Tensor k_;
    Tensor lse_;
    Tensor out_;
    Tensor denom_;
    Tensor ratio_;
    Tensor softmax_scale_;
    std::int64_t ratio_value_{};
    DenseAttnScoreAttributes attributes_;
};

}  // namespace ftrain

#endif
