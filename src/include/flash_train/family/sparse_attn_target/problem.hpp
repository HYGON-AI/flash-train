// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_TARGET_PROBLEM_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_TARGET_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/sparse_attn_target.hpp"

namespace ftrain {

// One SparseAttnTarget call's parameters as a support checklist: the
// normalized teacher scores on the selected KV entries, recomputed from the
// query, the keys, and the forward's log-sum-exp.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class SparseAttnTargetProblem final {
  public:
    SparseAttnTargetProblem(Tensor q, Tensor k, Tensor lse, Tensor topk_indices, Tensor target, Tensor softmax_scale,
                            Tensor topk_indices_global, SparseAttnTargetAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getQDims() const noexcept { return q_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getKDims() const noexcept { return k_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getLseDims() const noexcept { return lse_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getTopkIndicesDims() const noexcept {
        return topk_indices_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getTargetDims() const noexcept { return target_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getSoftmaxScaleDims() const noexcept {
        return softmax_scale_.getStorageView().getDims();
    }
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
    FTrainNumericType getLseNumericType() const noexcept { return lse_.getStorageView().getNumericType(); }
    FTrainNumericType getTopkIndicesNumericType() const noexcept {
        return topk_indices_.getStorageView().getNumericType();
    }
    FTrainNumericType getTargetNumericType() const noexcept { return target_.getStorageView().getNumericType(); }

    FTrainIndexType getQIndexType() const noexcept { return q_.getStorageView().getIndexType(); }
    FTrainIndexType getKIndexType() const noexcept { return k_.getStorageView().getIndexType(); }
    FTrainIndexType getLseIndexType() const noexcept { return lse_.getStorageView().getIndexType(); }
    FTrainIndexType getTopkIndicesIndexType() const noexcept { return topk_indices_.getStorageView().getIndexType(); }
    FTrainIndexType getTargetIndexType() const noexcept { return target_.getStorageView().getIndexType(); }

    void* getQAddress() const noexcept { return q_.getStorageView().getMemory(); }
    void* getKAddress() const noexcept { return k_.getStorageView().getMemory(); }
    void* getLseAddress() const noexcept { return lse_.getStorageView().getMemory(); }
    void* getTopkIndicesAddress() const noexcept { return topk_indices_.getStorageView().getMemory(); }
    void* getTargetAddress() const noexcept { return target_.getStorageView().getMemory(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getSoftmaxScaleNumericType() const noexcept {
        return softmax_scale_.getStorageView().getNumericType();
    }
    FTrainIndexType getSoftmaxScaleIndexType() const noexcept { return softmax_scale_.getStorageView().getIndexType(); }
    void* getSoftmaxScaleAddress() const noexcept { return softmax_scale_.getStorageView().getMemory(); }
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
    Tensor lse_;
    Tensor topk_indices_;
    Tensor target_;
    Tensor softmax_scale_;
    Tensor topk_indices_global_;
    SparseAttnTargetAttributes attributes_;
};

}  // namespace ftrain

#endif
