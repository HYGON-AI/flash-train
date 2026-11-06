// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_ATTN_SCORE_THD_PROBLEM_HPP_
#define FTRAIN_FAMILY_DENSE_ATTN_SCORE_THD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/dense_attn_score_thd.hpp"

namespace ftrain {

// One DenseAttnScoreThd call's parameters as a support checklist: the
// packed-batch dense attention score and its row total.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class DenseAttnScoreThdProblem final {
  public:
    DenseAttnScoreThdProblem(Tensor q, Tensor k, Tensor lse, Tensor cu_seqlens_q, Tensor cu_seqlens_kv,
                             Tensor q_causal_offsets, Tensor out, Tensor denom, Tensor ratio, Tensor softmax_scale,
                             Tensor max_seqlen_kv, DenseAttnScoreThdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getQDims() const noexcept { return q_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getKDims() const noexcept { return k_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getLseDims() const noexcept { return lse_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getCuSeqlensQDims() const noexcept {
        return cu_seqlens_q_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getCuSeqlensKvDims() const noexcept {
        return cu_seqlens_kv_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getQCausalOffsetsDims() const noexcept {
        return q_causal_offsets_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getOutDims() const noexcept { return out_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getDenomDims() const noexcept { return denom_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getRatioDims() const noexcept { return ratio_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getSoftmaxScaleDims() const noexcept {
        return softmax_scale_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getMaxSeqlenKvDims() const noexcept {
        return max_seqlen_kv_.getStorageView().getDims();
    }

    std::uint64_t getTotalQ() const noexcept { return dim(q_, 0); }
    std::uint64_t getTotalK() const noexcept { return dim(k_, 0); }
    std::uint64_t getMaxSeqlenKv() const noexcept { return static_cast<std::uint64_t>(max_seqlen_kv_value_); }
    std::uint64_t getNumSequences() const noexcept { return dim(cu_seqlens_q_, 0) - 1; }
    std::uint64_t getQueryHeads() const noexcept { return dim(q_, 1); }
    std::uint64_t getHeadDim() const noexcept { return dim(q_, 2); }

    bool hasSingleKvHead() const noexcept { return dim(k_, 1) == 1; }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }
    std::int64_t getRatio() const noexcept { return ratio_value_; }

    FTrainNumericType getQNumericType() const noexcept { return q_.getStorageView().getNumericType(); }
    FTrainNumericType getKNumericType() const noexcept { return k_.getStorageView().getNumericType(); }
    FTrainNumericType getLseNumericType() const noexcept { return lse_.getStorageView().getNumericType(); }
    FTrainNumericType getOutNumericType() const noexcept { return out_.getStorageView().getNumericType(); }
    FTrainNumericType getDenomNumericType() const noexcept { return denom_.getStorageView().getNumericType(); }
    FTrainNumericType getCuSeqLensQNumericType() const noexcept {
        return cu_seqlens_q_.getStorageView().getNumericType();
    }
    FTrainNumericType getCuSeqLensKvNumericType() const noexcept {
        return cu_seqlens_kv_.getStorageView().getNumericType();
    }
    FTrainNumericType getCausalOffsetsNumericType() const noexcept {
        return q_causal_offsets_.getStorageView().getNumericType();
    }

    FTrainIndexType getQIndexType() const noexcept { return q_.getStorageView().getIndexType(); }
    FTrainIndexType getKIndexType() const noexcept { return k_.getStorageView().getIndexType(); }
    FTrainIndexType getLseIndexType() const noexcept { return lse_.getStorageView().getIndexType(); }
    FTrainIndexType getOutIndexType() const noexcept { return out_.getStorageView().getIndexType(); }
    FTrainIndexType getDenomIndexType() const noexcept { return denom_.getStorageView().getIndexType(); }
    FTrainIndexType getCuSeqLensQIndexType() const noexcept { return cu_seqlens_q_.getStorageView().getIndexType(); }
    FTrainIndexType getCuSeqLensKvIndexType() const noexcept { return cu_seqlens_kv_.getStorageView().getIndexType(); }
    FTrainIndexType getCausalOffsetsIndexType() const noexcept {
        return q_causal_offsets_.getStorageView().getIndexType();
    }

    // getCausalOffsetsAddress may be null: the offsets are optional metadata.
    void* getQAddress() const noexcept { return q_.getStorageView().getMemory(); }
    void* getKAddress() const noexcept { return k_.getStorageView().getMemory(); }
    void* getLseAddress() const noexcept { return lse_.getStorageView().getMemory(); }
    void* getOutAddress() const noexcept { return out_.getStorageView().getMemory(); }
    void* getDenomAddress() const noexcept { return denom_.getStorageView().getMemory(); }
    void* getCuSeqLensQAddress() const noexcept { return cu_seqlens_q_.getStorageView().getMemory(); }
    void* getCuSeqLensKvAddress() const noexcept { return cu_seqlens_kv_.getStorageView().getMemory(); }
    void* getCausalOffsetsAddress() const noexcept { return q_causal_offsets_.getStorageView().getMemory(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getRatioNumericType() const noexcept { return ratio_.getStorageView().getNumericType(); }
    FTrainIndexType getRatioIndexType() const noexcept { return ratio_.getStorageView().getIndexType(); }
    void* getRatioAddress() const noexcept { return ratio_.getStorageView().getMemory(); }
    FTrainNumericType getSoftmaxScaleNumericType() const noexcept {
        return softmax_scale_.getStorageView().getNumericType();
    }
    FTrainIndexType getSoftmaxScaleIndexType() const noexcept { return softmax_scale_.getStorageView().getIndexType(); }
    void* getSoftmaxScaleAddress() const noexcept { return softmax_scale_.getStorageView().getMemory(); }
    FTrainNumericType getMaxSeqlenKvNumericType() const noexcept {
        return max_seqlen_kv_.getStorageView().getNumericType();
    }
    FTrainIndexType getMaxSeqlenKvIndexType() const noexcept { return max_seqlen_kv_.getStorageView().getIndexType(); }
    void* getMaxSeqlenKvAddress() const noexcept { return max_seqlen_kv_.getStorageView().getMemory(); }

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
    Tensor cu_seqlens_q_;
    Tensor cu_seqlens_kv_;
    Tensor q_causal_offsets_;
    Tensor out_;
    Tensor denom_;
    Tensor ratio_;
    Tensor softmax_scale_;
    Tensor max_seqlen_kv_;
    std::int64_t ratio_value_{};
    std::int64_t max_seqlen_kv_value_{};
    DenseAttnScoreThdAttributes attributes_;
};

}  // namespace ftrain

#endif
