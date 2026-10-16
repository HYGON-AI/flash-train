// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_BWD_DENSE_THD_PROBLEM_HPP_
#define FTRAIN_FAMILY_INDEXER_BWD_DENSE_THD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/indexer_bwd_dense_thd.hpp"

namespace ftrain {

// One IndexerBwdDenseThd call's parameters as a support checklist: the dense
// indexer backward over one packed batch.
//
// In-place contract: grad_index_score must alias index_score. The constructor
// validates the packed q/w/k ranks and their shared dimensions, the
// cumulative-length metadata, the FP32 score (total_q,max_sk), normalizer
// (total_q), and log-sum-exp (total_q) ports, the gradient shapes, and
// 32-bit-indexable operands.
class IndexerBwdDenseThdProblem final {
  public:
    IndexerBwdDenseThdProblem(Tensor index_q, Tensor weights, Tensor index_k, Tensor attn_score, Tensor attn_l1norm,
                              Tensor index_score, Tensor index_lse, Tensor cu_seqlens_q, Tensor cu_seqlens_k,
                              Tensor q_causal_offsets, Tensor grad_index_score, Tensor dq, Tensor dw, Tensor dk,
                              Tensor sm_scale, Tensor loss_coeff, Tensor grad_loss, Tensor ratio,
                              IndexerBwdDenseThdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getIndexQDims() const noexcept { return index_q_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getWeightsDims() const noexcept { return weights_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getIndexKDims() const noexcept { return index_k_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getAttnScoreDims() const noexcept {
        return attn_score_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getAttnL1normDims() const noexcept {
        return attn_l1norm_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getIndexScoreDims() const noexcept {
        return index_score_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getIndexLseDims() const noexcept { return index_lse_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getCuSeqlensQDims() const noexcept {
        return cu_seqlens_q_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getCuSeqlensKDims() const noexcept {
        return cu_seqlens_k_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getQCausalOffsetsDims() const noexcept {
        return q_causal_offsets_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getGradIndexScoreDims() const noexcept {
        return grad_index_score_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getDqDims() const noexcept { return dq_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getDwDims() const noexcept { return dw_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getDkDims() const noexcept { return dk_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getSmScaleDims() const noexcept { return sm_scale_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getLossCoeffDims() const noexcept {
        return loss_coeff_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getGradLossDims() const noexcept { return grad_loss_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getRatioDims() const noexcept { return ratio_.getStorageView().getDims(); }

    std::uint64_t getTotalQ() const noexcept { return dim(index_q_, 0); }
    std::uint64_t getTotalK() const noexcept { return dim(index_k_, 0); }
    std::uint64_t getMaxSeqlenKv() const noexcept { return dim(attn_score_, 1); }
    std::uint64_t getNumSequences() const noexcept { return dim(cu_seqlens_q_, 0) - 1; }
    std::uint64_t getHeads() const noexcept { return dim(index_q_, 1); }
    std::uint64_t getHeadDim() const noexcept { return dim(index_q_, 2); }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }
    std::int64_t getRatio() const noexcept { return ratio_value_; }

    FTrainNumericType getIndexQNumericType() const noexcept { return index_q_.getStorageView().getNumericType(); }
    FTrainNumericType getWeightsNumericType() const noexcept { return weights_.getStorageView().getNumericType(); }
    FTrainNumericType getIndexKNumericType() const noexcept { return index_k_.getStorageView().getNumericType(); }
    FTrainNumericType getAttnScoreNumericType() const noexcept { return attn_score_.getStorageView().getNumericType(); }
    FTrainNumericType getAttnL1NormNumericType() const noexcept {
        return attn_l1norm_.getStorageView().getNumericType();
    }
    FTrainNumericType getIndexScoreNumericType() const noexcept {
        return index_score_.getStorageView().getNumericType();
    }
    FTrainNumericType getIndexLseNumericType() const noexcept { return index_lse_.getStorageView().getNumericType(); }
    FTrainNumericType getCuSeqLensQNumericType() const noexcept {
        return cu_seqlens_q_.getStorageView().getNumericType();
    }
    FTrainNumericType getCuSeqLensKNumericType() const noexcept {
        return cu_seqlens_k_.getStorageView().getNumericType();
    }
    FTrainNumericType getCausalOffsetsNumericType() const noexcept {
        return q_causal_offsets_.getStorageView().getNumericType();
    }
    FTrainNumericType getGradIndexScoreNumericType() const noexcept {
        return grad_index_score_.getStorageView().getNumericType();
    }
    FTrainNumericType getDqNumericType() const noexcept { return dq_.getStorageView().getNumericType(); }
    FTrainNumericType getDwNumericType() const noexcept { return dw_.getStorageView().getNumericType(); }
    FTrainNumericType getDkNumericType() const noexcept { return dk_.getStorageView().getNumericType(); }

    FTrainIndexType getIndexQIndexType() const noexcept { return index_q_.getStorageView().getIndexType(); }
    FTrainIndexType getWeightsIndexType() const noexcept { return weights_.getStorageView().getIndexType(); }
    FTrainIndexType getIndexKIndexType() const noexcept { return index_k_.getStorageView().getIndexType(); }
    FTrainIndexType getAttnScoreIndexType() const noexcept { return attn_score_.getStorageView().getIndexType(); }
    FTrainIndexType getAttnL1NormIndexType() const noexcept { return attn_l1norm_.getStorageView().getIndexType(); }
    FTrainIndexType getIndexScoreIndexType() const noexcept { return index_score_.getStorageView().getIndexType(); }
    FTrainIndexType getIndexLseIndexType() const noexcept { return index_lse_.getStorageView().getIndexType(); }
    FTrainIndexType getCuSeqLensQIndexType() const noexcept { return cu_seqlens_q_.getStorageView().getIndexType(); }
    FTrainIndexType getCuSeqLensKIndexType() const noexcept { return cu_seqlens_k_.getStorageView().getIndexType(); }
    FTrainIndexType getCausalOffsetsIndexType() const noexcept {
        return q_causal_offsets_.getStorageView().getIndexType();
    }
    FTrainIndexType getGradIndexScoreIndexType() const noexcept {
        return grad_index_score_.getStorageView().getIndexType();
    }
    FTrainIndexType getDqIndexType() const noexcept { return dq_.getStorageView().getIndexType(); }
    FTrainIndexType getDwIndexType() const noexcept { return dw_.getStorageView().getIndexType(); }
    FTrainIndexType getDkIndexType() const noexcept { return dk_.getStorageView().getIndexType(); }

    // getCausalOffsetsAddress may be null: the offsets are optional metadata.
    void* getIndexQAddress() const noexcept { return index_q_.getStorageView().getMemory(); }
    void* getWeightsAddress() const noexcept { return weights_.getStorageView().getMemory(); }
    void* getIndexKAddress() const noexcept { return index_k_.getStorageView().getMemory(); }
    void* getAttnScoreAddress() const noexcept { return attn_score_.getStorageView().getMemory(); }
    void* getAttnL1NormAddress() const noexcept { return attn_l1norm_.getStorageView().getMemory(); }
    void* getIndexScoreAddress() const noexcept { return index_score_.getStorageView().getMemory(); }
    void* getIndexLseAddress() const noexcept { return index_lse_.getStorageView().getMemory(); }
    void* getCuSeqLensQAddress() const noexcept { return cu_seqlens_q_.getStorageView().getMemory(); }
    void* getCuSeqLensKAddress() const noexcept { return cu_seqlens_k_.getStorageView().getMemory(); }
    void* getCausalOffsetsAddress() const noexcept { return q_causal_offsets_.getStorageView().getMemory(); }
    void* getGradIndexScoreAddress() const noexcept { return grad_index_score_.getStorageView().getMemory(); }
    void* getDqAddress() const noexcept { return dq_.getStorageView().getMemory(); }
    void* getDwAddress() const noexcept { return dw_.getStorageView().getMemory(); }
    void* getDkAddress() const noexcept { return dk_.getStorageView().getMemory(); }

    bool isInPlace() const noexcept { return getIndexScoreAddress() == getGradIndexScoreAddress(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getSmScaleNumericType() const noexcept { return sm_scale_.getStorageView().getNumericType(); }
    FTrainIndexType getSmScaleIndexType() const noexcept { return sm_scale_.getStorageView().getIndexType(); }
    void* getSmScaleAddress() const noexcept { return sm_scale_.getStorageView().getMemory(); }
    FTrainNumericType getLossCoeffNumericType() const noexcept { return loss_coeff_.getStorageView().getNumericType(); }
    FTrainIndexType getLossCoeffIndexType() const noexcept { return loss_coeff_.getStorageView().getIndexType(); }
    void* getLossCoeffAddress() const noexcept { return loss_coeff_.getStorageView().getMemory(); }
    FTrainNumericType getGradLossNumericType() const noexcept { return grad_loss_.getStorageView().getNumericType(); }
    FTrainIndexType getGradLossIndexType() const noexcept { return grad_loss_.getStorageView().getIndexType(); }
    void* getGradLossAddress() const noexcept { return grad_loss_.getStorageView().getMemory(); }
    FTrainNumericType getRatioNumericType() const noexcept { return ratio_.getStorageView().getNumericType(); }
    FTrainIndexType getRatioIndexType() const noexcept { return ratio_.getStorageView().getIndexType(); }
    void* getRatioAddress() const noexcept { return ratio_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dim(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }

    Tensor index_q_;
    Tensor weights_;
    Tensor index_k_;
    Tensor attn_score_;
    Tensor attn_l1norm_;
    Tensor index_score_;
    Tensor index_lse_;
    Tensor cu_seqlens_q_;
    Tensor cu_seqlens_k_;
    Tensor q_causal_offsets_;
    Tensor grad_index_score_;
    Tensor dq_;
    Tensor dw_;
    Tensor dk_;
    Tensor sm_scale_;
    Tensor loss_coeff_;
    Tensor grad_loss_;
    Tensor ratio_;
    std::int64_t ratio_value_{};
    IndexerBwdDenseThdAttributes attributes_;
};

}  // namespace ftrain

#endif
