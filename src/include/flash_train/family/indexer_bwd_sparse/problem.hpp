// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_BWD_SPARSE_PROBLEM_HPP_
#define FTRAIN_FAMILY_INDEXER_BWD_SPARSE_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/indexer_bwd_sparse.hpp"

namespace ftrain {

// One IndexerBwdSparse call's parameters as a support checklist: the KL
// gradient of the sparse indexer over the selected KV entries.
//
// In-place contract: the kernel reads the per-selection target and writes the
// selection gradient into the same buffer, so grad_attn_score must alias
// attn_score (the constructor rejects a call where they do not). The
// constructor also validates q/weights/index_k ranks and their shared
// dimensions, the FP32 score ports, the INT32 selection, the gradient shapes,
// and 32-bit-indexable operands.
class IndexerBwdSparseProblem final {
  public:
    IndexerBwdSparseProblem(Tensor index_q, Tensor weights, Tensor index_k, Tensor attn_score, Tensor index_score,
                            Tensor topk_indices, Tensor grad_attn_score, Tensor dq, Tensor dw, Tensor dk,
                            Tensor sm_scale, Tensor loss_coeff, Tensor grad_loss, Tensor topk_indices_global,
                            IndexerBwdSparseAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getIndexQDims() const noexcept { return index_q_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getWeightsDims() const noexcept { return weights_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getIndexKDims() const noexcept { return index_k_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getAttnScoreDims() const noexcept {
        return attn_score_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getIndexScoreDims() const noexcept {
        return index_score_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getTopkIndicesDims() const noexcept {
        return topk_indices_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getGradAttnScoreDims() const noexcept {
        return grad_attn_score_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getDqDims() const noexcept { return dq_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getDwDims() const noexcept { return dw_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getDkDims() const noexcept { return dk_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getSmScaleDims() const noexcept { return sm_scale_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getLossCoeffDims() const noexcept {
        return loss_coeff_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getGradLossDims() const noexcept { return grad_loss_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getTopkIndicesGlobalDims() const noexcept {
        return topk_indices_global_.getStorageView().getDims();
    }

    std::uint64_t getBatch() const noexcept { return dim(index_q_, 0); }
    std::uint64_t getQueryLength() const noexcept { return dim(index_q_, 1); }
    std::uint64_t getKeyLength() const noexcept { return dim(index_k_, 1); }
    std::uint64_t getHeads() const noexcept { return dim(index_q_, 2); }
    std::uint64_t getHeadDim() const noexcept { return dim(index_q_, 3); }
    std::uint64_t getSelectionWidth() const noexcept { return dim(topk_indices_, 2); }
    std::uint64_t getNumSequences() const noexcept { return getBatch() * getQueryLength(); }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }

    FTrainNumericType getIndexQNumericType() const noexcept { return index_q_.getStorageView().getNumericType(); }
    FTrainNumericType getWeightsNumericType() const noexcept { return weights_.getStorageView().getNumericType(); }
    FTrainNumericType getIndexKNumericType() const noexcept { return index_k_.getStorageView().getNumericType(); }
    FTrainNumericType getAttnScoreNumericType() const noexcept { return attn_score_.getStorageView().getNumericType(); }
    FTrainNumericType getIndexScoreNumericType() const noexcept {
        return index_score_.getStorageView().getNumericType();
    }
    FTrainNumericType getTopkIndicesNumericType() const noexcept {
        return topk_indices_.getStorageView().getNumericType();
    }
    FTrainNumericType getGradAttnScoreNumericType() const noexcept {
        return grad_attn_score_.getStorageView().getNumericType();
    }
    FTrainNumericType getDqNumericType() const noexcept { return dq_.getStorageView().getNumericType(); }
    FTrainNumericType getDwNumericType() const noexcept { return dw_.getStorageView().getNumericType(); }
    FTrainNumericType getDkNumericType() const noexcept { return dk_.getStorageView().getNumericType(); }

    FTrainIndexType getIndexQIndexType() const noexcept { return index_q_.getStorageView().getIndexType(); }
    FTrainIndexType getWeightsIndexType() const noexcept { return weights_.getStorageView().getIndexType(); }
    FTrainIndexType getIndexKIndexType() const noexcept { return index_k_.getStorageView().getIndexType(); }
    FTrainIndexType getAttnScoreIndexType() const noexcept { return attn_score_.getStorageView().getIndexType(); }
    FTrainIndexType getIndexScoreIndexType() const noexcept { return index_score_.getStorageView().getIndexType(); }
    FTrainIndexType getTopkIndicesIndexType() const noexcept { return topk_indices_.getStorageView().getIndexType(); }
    FTrainIndexType getGradAttnScoreIndexType() const noexcept {
        return grad_attn_score_.getStorageView().getIndexType();
    }
    FTrainIndexType getDqIndexType() const noexcept { return dq_.getStorageView().getIndexType(); }
    FTrainIndexType getDwIndexType() const noexcept { return dw_.getStorageView().getIndexType(); }
    FTrainIndexType getDkIndexType() const noexcept { return dk_.getStorageView().getIndexType(); }

    void* getIndexQAddress() const noexcept { return index_q_.getStorageView().getMemory(); }
    void* getWeightsAddress() const noexcept { return weights_.getStorageView().getMemory(); }
    void* getIndexKAddress() const noexcept { return index_k_.getStorageView().getMemory(); }
    void* getAttnScoreAddress() const noexcept { return attn_score_.getStorageView().getMemory(); }
    void* getIndexScoreAddress() const noexcept { return index_score_.getStorageView().getMemory(); }
    void* getTopkIndicesAddress() const noexcept { return topk_indices_.getStorageView().getMemory(); }
    void* getGradAttnScoreAddress() const noexcept { return grad_attn_score_.getStorageView().getMemory(); }
    void* getDqAddress() const noexcept { return dq_.getStorageView().getMemory(); }
    void* getDwAddress() const noexcept { return dw_.getStorageView().getMemory(); }
    void* getDkAddress() const noexcept { return dk_.getStorageView().getMemory(); }

    bool isInPlace() const noexcept { return getAttnScoreAddress() == getGradAttnScoreAddress(); }

    // Scalar ports, following Gemm alpha/beta; BF16 also accepts device grad_loss.
    FTrainNumericType getSmScaleNumericType() const noexcept { return sm_scale_.getStorageView().getNumericType(); }
    FTrainIndexType getSmScaleIndexType() const noexcept { return sm_scale_.getStorageView().getIndexType(); }
    void* getSmScaleAddress() const noexcept { return sm_scale_.getStorageView().getMemory(); }
    FTrainNumericType getLossCoeffNumericType() const noexcept { return loss_coeff_.getStorageView().getNumericType(); }
    FTrainIndexType getLossCoeffIndexType() const noexcept { return loss_coeff_.getStorageView().getIndexType(); }
    void* getLossCoeffAddress() const noexcept { return loss_coeff_.getStorageView().getMemory(); }
    FTrainNumericType getGradLossNumericType() const noexcept { return grad_loss_.getStorageView().getNumericType(); }
    FTrainIndexType getGradLossIndexType() const noexcept { return grad_loss_.getStorageView().getIndexType(); }
    bool isGradLossHostMemory() const noexcept { return grad_loss_.getStorageView().isHostMemory(); }
    void* getGradLossAddress() const noexcept { return grad_loss_.getStorageView().getMemory(); }
    FTrainNumericType getTopkIndicesGlobalNumericType() const noexcept {
        return topk_indices_global_.getStorageView().getNumericType();
    }
    FTrainIndexType getTopkIndicesGlobalIndexType() const noexcept {
        return topk_indices_global_.getStorageView().getIndexType();
    }
    void* getTopkIndicesGlobalAddress() const noexcept { return topk_indices_global_.getStorageView().getMemory(); }

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
    Tensor index_score_;
    Tensor topk_indices_;
    Tensor grad_attn_score_;
    Tensor dq_;
    Tensor dw_;
    Tensor dk_;
    Tensor sm_scale_;
    Tensor loss_coeff_;
    Tensor grad_loss_;
    Tensor topk_indices_global_;
    IndexerBwdSparseAttributes attributes_;
};

}  // namespace ftrain

#endif
