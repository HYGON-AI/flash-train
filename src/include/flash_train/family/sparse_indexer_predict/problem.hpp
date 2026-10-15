// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_INDEXER_PREDICT_PROBLEM_HPP_
#define FTRAIN_FAMILY_SPARSE_INDEXER_PREDICT_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/sparse_indexer_predict.hpp"

namespace ftrain {

// One SparseIndexerPredict call's parameters as a support checklist: the
// softmax over the selected KV entries.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class SparseIndexerPredictProblem final {
  public:
    SparseIndexerPredictProblem(Tensor q, Tensor k, Tensor w, Tensor topk_indices, Tensor predict, Tensor sm_scale,
                                Tensor topk_indices_global, SparseIndexerPredictAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getQDims() const noexcept { return q_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getKDims() const noexcept { return k_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getWDims() const noexcept { return w_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getTopkIndicesDims() const noexcept {
        return topk_indices_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getPredictDims() const noexcept { return predict_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getSmScaleDims() const noexcept { return sm_scale_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getTopkIndicesGlobalDims() const noexcept {
        return topk_indices_global_.getStorageView().getDims();
    }

    std::uint64_t getBatch() const noexcept { return dim(q_, 0); }
    std::uint64_t getQueryLength() const noexcept { return dim(q_, 1); }
    std::uint64_t getKeyLength() const noexcept { return dim(k_, 1); }
    std::uint64_t getQueryHeads() const noexcept { return dim(q_, 2); }
    std::uint64_t getHeadDim() const noexcept { return dim(q_, 3); }
    std::uint64_t getSelectionWidth() const noexcept { return dim(topk_indices_, 2); }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }

    FTrainNumericType getQNumericType() const noexcept { return q_.getStorageView().getNumericType(); }
    FTrainNumericType getKNumericType() const noexcept { return k_.getStorageView().getNumericType(); }
    FTrainNumericType getWNumericType() const noexcept { return w_.getStorageView().getNumericType(); }
    FTrainNumericType getTopkIndicesNumericType() const noexcept {
        return topk_indices_.getStorageView().getNumericType();
    }
    FTrainNumericType getPredictNumericType() const noexcept { return predict_.getStorageView().getNumericType(); }

    FTrainIndexType getQIndexType() const noexcept { return q_.getStorageView().getIndexType(); }
    FTrainIndexType getKIndexType() const noexcept { return k_.getStorageView().getIndexType(); }
    FTrainIndexType getWIndexType() const noexcept { return w_.getStorageView().getIndexType(); }
    FTrainIndexType getTopkIndicesIndexType() const noexcept { return topk_indices_.getStorageView().getIndexType(); }
    FTrainIndexType getPredictIndexType() const noexcept { return predict_.getStorageView().getIndexType(); }

    void* getQAddress() const noexcept { return q_.getStorageView().getMemory(); }
    void* getKAddress() const noexcept { return k_.getStorageView().getMemory(); }
    void* getWAddress() const noexcept { return w_.getStorageView().getMemory(); }
    void* getTopkIndicesAddress() const noexcept { return topk_indices_.getStorageView().getMemory(); }
    void* getPredictAddress() const noexcept { return predict_.getStorageView().getMemory(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getSmScaleNumericType() const noexcept { return sm_scale_.getStorageView().getNumericType(); }
    FTrainIndexType getSmScaleIndexType() const noexcept { return sm_scale_.getStorageView().getIndexType(); }
    void* getSmScaleAddress() const noexcept { return sm_scale_.getStorageView().getMemory(); }
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

    Tensor q_;
    Tensor k_;
    Tensor w_;
    Tensor topk_indices_;
    Tensor predict_;
    Tensor sm_scale_;
    Tensor topk_indices_global_;
    SparseIndexerPredictAttributes attributes_;
};

}  // namespace ftrain

#endif
