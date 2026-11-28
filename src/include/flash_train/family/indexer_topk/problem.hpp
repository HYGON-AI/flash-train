// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_TOPK_PROBLEM_HPP_
#define FTRAIN_FAMILY_INDEXER_TOPK_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/indexer_topk.hpp"

namespace ftrain {

// One IndexerTopK call's parameters as a support checklist: per-row selection
// over an FP32 score matrix, with the row's valid prefix given by seq_lens.
//
// This family has no compute-type chain: the kernel reads and writes fixed
// element types (FP32 scores in, INT32 indices out), so a single
// implementation serves every call and `ftrainArgsSetIndexerTopK` carries no
// compute type. The constructor validates ranks, the FP32/INT32 element
// types, the row-count agreement with seq_lens, the (n_rows, top_k) output
// shape, and that seq_lens covers every row of the request.
class IndexerTopKProblem final {
  public:
    IndexerTopKProblem(Tensor input_values, Tensor seq_lens, Tensor out_indices, Tensor out_values, Tensor top_k,
                       Tensor next_n, Tensor order_values, IndexerTopKAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getInputValuesDims() const noexcept {
        return input_values_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getSeqLensDims() const noexcept { return seq_lens_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getOutIndicesDims() const noexcept {
        return out_indices_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getOutValuesDims() const noexcept {
        return out_values_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getTopKDims() const noexcept { return top_k_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getNextNDims() const noexcept { return next_n_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getOrderValuesDims() const noexcept {
        return order_values_.getStorageView().getDims();
    }

    std::uint64_t getNumRows() const noexcept { return dim(input_values_, 0); }
    std::uint64_t getNumColumns() const noexcept { return dim(input_values_, 1); }
    std::uint64_t getNumSequences() const noexcept { return dim(seq_lens_, 0); }

    std::int64_t getTopK() const noexcept { return top_k_value_; }
    std::int64_t getNextN() const noexcept { return next_n_value_; }
    bool getOrderValues() const noexcept { return order_values_value_; }

    FTrainNumericType getInputValuesNumericType() const noexcept {
        return input_values_.getStorageView().getNumericType();
    }
    FTrainNumericType getSeqLensNumericType() const noexcept { return seq_lens_.getStorageView().getNumericType(); }
    FTrainNumericType getOutIndicesNumericType() const noexcept {
        return out_indices_.getStorageView().getNumericType();
    }
    FTrainNumericType getOutValuesNumericType() const noexcept { return out_values_.getStorageView().getNumericType(); }

    FTrainIndexType getInputValuesIndexType() const noexcept { return input_values_.getStorageView().getIndexType(); }
    FTrainIndexType getSeqLensIndexType() const noexcept { return seq_lens_.getStorageView().getIndexType(); }
    FTrainIndexType getOutIndicesIndexType() const noexcept { return out_indices_.getStorageView().getIndexType(); }
    FTrainIndexType getOutValuesIndexType() const noexcept { return out_values_.getStorageView().getIndexType(); }

    // getOutValuesAddress may be null when order_values is false.
    void* getInputValuesAddress() const noexcept { return input_values_.getStorageView().getMemory(); }
    void* getSeqLensAddress() const noexcept { return seq_lens_.getStorageView().getMemory(); }
    void* getOutIndicesAddress() const noexcept { return out_indices_.getStorageView().getMemory(); }
    void* getOutValuesAddress() const noexcept { return out_values_.getStorageView().getMemory(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getTopKNumericType() const noexcept { return top_k_.getStorageView().getNumericType(); }
    FTrainIndexType getTopKIndexType() const noexcept { return top_k_.getStorageView().getIndexType(); }
    void* getTopKAddress() const noexcept { return top_k_.getStorageView().getMemory(); }
    FTrainNumericType getNextNNumericType() const noexcept { return next_n_.getStorageView().getNumericType(); }
    FTrainIndexType getNextNIndexType() const noexcept { return next_n_.getStorageView().getIndexType(); }
    void* getNextNAddress() const noexcept { return next_n_.getStorageView().getMemory(); }
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
    Tensor seq_lens_;
    Tensor out_indices_;
    Tensor out_values_;
    Tensor top_k_;
    Tensor next_n_;
    Tensor order_values_;
    std::int64_t top_k_value_{};
    std::int64_t next_n_value_{};
    std::uint8_t order_values_value_{};
    IndexerTopKAttributes attributes_;
};

}  // namespace ftrain

#endif
