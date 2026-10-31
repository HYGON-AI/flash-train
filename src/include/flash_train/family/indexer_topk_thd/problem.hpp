// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_TOPK_THD_PROBLEM_HPP_
#define FTRAIN_FAMILY_INDEXER_TOPK_THD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/indexer_topk_thd.hpp"

namespace ftrain {

// One IndexerTopKThd call's parameters as a support checklist: per-row
// selection over one packed batch, whose valid prefix the kernel derives from
// cu_seqlens_q / cu_seqlens_kv and the ratio.
//
// No compute-type chain: FP32 scores in, INT32 indices out. The constructor
// validates rank-two FP32 input, rank-one INT32 cumulative lengths, the
// optional INT32 q_causal_offsets metadata, the (n_rows, top_k) INT32 output,
// the optional FP32 value output, and the positive scalars.
class IndexerTopKThdProblem final {
  public:
    IndexerTopKThdProblem(Tensor input_values, Tensor cu_seqlens_q, Tensor cu_seqlens_kv, Tensor q_causal_offsets,
                          Tensor out_indices, Tensor out_values, Tensor ratio, Tensor top_k, Tensor order_values,
                          IndexerTopKThdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getInputValuesDims() const noexcept {
        return input_values_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getCuSeqlensQDims() const noexcept {
        return cu_seqlens_q_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getCuSeqlensKvDims() const noexcept {
        return cu_seqlens_kv_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getQCausalOffsetsDims() const noexcept {
        return q_causal_offsets_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getOutIndicesDims() const noexcept {
        return out_indices_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getOutValuesDims() const noexcept {
        return out_values_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getRatioDims() const noexcept { return ratio_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getTopKDims() const noexcept { return top_k_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getOrderValuesDims() const noexcept {
        return order_values_.getStorageView().getDims();
    }

    std::uint64_t getNumRows() const noexcept { return dim(input_values_, 0); }
    std::uint64_t getNumColumns() const noexcept { return dim(input_values_, 1); }
    std::uint64_t getNumSequences() const noexcept { return dim(cu_seqlens_q_, 0) - 1; }

    std::int64_t getRatio() const noexcept { return ratio_value_; }
    std::int64_t getTopK() const noexcept { return top_k_value_; }
    bool getOrderValues() const noexcept { return order_values_value_; }

    FTrainNumericType getInputValuesNumericType() const noexcept {
        return input_values_.getStorageView().getNumericType();
    }
    FTrainNumericType getCuSeqLensQNumericType() const noexcept {
        return cu_seqlens_q_.getStorageView().getNumericType();
    }
    FTrainNumericType getCuSeqLensKvNumericType() const noexcept {
        return cu_seqlens_kv_.getStorageView().getNumericType();
    }
    FTrainNumericType getCausalOffsetsNumericType() const noexcept {
        return q_causal_offsets_.getStorageView().getNumericType();
    }
    FTrainNumericType getOutIndicesNumericType() const noexcept {
        return out_indices_.getStorageView().getNumericType();
    }
    FTrainNumericType getOutValuesNumericType() const noexcept { return out_values_.getStorageView().getNumericType(); }

    FTrainIndexType getInputValuesIndexType() const noexcept { return input_values_.getStorageView().getIndexType(); }
    FTrainIndexType getCuSeqLensQIndexType() const noexcept { return cu_seqlens_q_.getStorageView().getIndexType(); }
    FTrainIndexType getCuSeqLensKvIndexType() const noexcept { return cu_seqlens_kv_.getStorageView().getIndexType(); }
    FTrainIndexType getCausalOffsetsIndexType() const noexcept {
        return q_causal_offsets_.getStorageView().getIndexType();
    }
    FTrainIndexType getOutIndicesIndexType() const noexcept { return out_indices_.getStorageView().getIndexType(); }
    FTrainIndexType getOutValuesIndexType() const noexcept { return out_values_.getStorageView().getIndexType(); }

    // getCausalOffsetsAddress and getOutValuesAddress may be null.
    void* getInputValuesAddress() const noexcept { return input_values_.getStorageView().getMemory(); }
    void* getCuSeqLensQAddress() const noexcept { return cu_seqlens_q_.getStorageView().getMemory(); }
    void* getCuSeqLensKvAddress() const noexcept { return cu_seqlens_kv_.getStorageView().getMemory(); }
    void* getCausalOffsetsAddress() const noexcept { return q_causal_offsets_.getStorageView().getMemory(); }
    void* getOutIndicesAddress() const noexcept { return out_indices_.getStorageView().getMemory(); }
    void* getOutValuesAddress() const noexcept { return out_values_.getStorageView().getMemory(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getRatioNumericType() const noexcept { return ratio_.getStorageView().getNumericType(); }
    FTrainIndexType getRatioIndexType() const noexcept { return ratio_.getStorageView().getIndexType(); }
    void* getRatioAddress() const noexcept { return ratio_.getStorageView().getMemory(); }
    FTrainNumericType getTopKNumericType() const noexcept { return top_k_.getStorageView().getNumericType(); }
    FTrainIndexType getTopKIndexType() const noexcept { return top_k_.getStorageView().getIndexType(); }
    void* getTopKAddress() const noexcept { return top_k_.getStorageView().getMemory(); }
    FTrainNumericType getOrderValuesNumericType() const noexcept {
        return order_values_.getStorageView().getNumericType();
    }
    FTrainIndexType getOrderValuesIndexType() const noexcept { return order_values_.getStorageView().getIndexType(); }
    void* getOrderValuesAddress() const noexcept { return order_values_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dim(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }

    Tensor input_values_;
    Tensor cu_seqlens_q_;
    Tensor cu_seqlens_kv_;
    Tensor q_causal_offsets_;
    Tensor out_indices_;
    Tensor out_values_;
    Tensor ratio_;
    Tensor top_k_;
    Tensor order_values_;
    std::int64_t ratio_value_{};
    std::int64_t top_k_value_{};
    std::uint8_t order_values_value_{};
    IndexerTopKThdAttributes attributes_;
};

}  // namespace ftrain

#endif
