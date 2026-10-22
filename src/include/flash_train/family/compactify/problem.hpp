// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_COMPACTIFY_PROBLEM_HPP_
#define FTRAIN_FAMILY_COMPACTIFY_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/compactify.hpp"

namespace ftrain {

// One Compactify call's parameters as a support checklist: the valid entries
// of every INT32 index row are packed to the front and the row's valid prefix
// length is reported.
//
// No compute-type chain: the kernel reads and writes INT32 only. The
// constructor validates the rank-two INT32 operands and that the output and
// length tensors agree with the input's shape; the selection width comes from
// the input itself.
class CompactifyProblem final {
  public:
    CompactifyProblem(Tensor indices, Tensor output, Tensor topk_length, CompactifyAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getIndicesDims() const noexcept { return indices_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getOutputDims() const noexcept { return output_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getTopkLengthDims() const noexcept {
        return topk_length_.getStorageView().getDims();
    }

    std::uint64_t getNumRows() const noexcept { return dim(indices_, 0); }
    std::uint64_t getSelectionWidth() const noexcept { return dim(indices_, 1); }

    FTrainNumericType getIndicesNumericType() const noexcept { return indices_.getStorageView().getNumericType(); }
    FTrainNumericType getOutputNumericType() const noexcept { return output_.getStorageView().getNumericType(); }
    FTrainNumericType getTopkLengthNumericType() const noexcept {
        return topk_length_.getStorageView().getNumericType();
    }

    FTrainIndexType getIndicesIndexType() const noexcept { return indices_.getStorageView().getIndexType(); }
    FTrainIndexType getOutputIndexType() const noexcept { return output_.getStorageView().getIndexType(); }
    FTrainIndexType getTopkLengthIndexType() const noexcept { return topk_length_.getStorageView().getIndexType(); }

    void* getIndicesAddress() const noexcept { return indices_.getStorageView().getMemory(); }
    void* getOutputAddress() const noexcept { return output_.getStorageView().getMemory(); }
    void* getTopkLengthAddress() const noexcept { return topk_length_.getStorageView().getMemory(); }

    // The parallel scatter requires disjoint continuous storage, including lengths.
    bool hasOverlappingBuffers() const noexcept;

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence, alignment and overlap used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dim(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }

    Tensor indices_;
    Tensor output_;
    Tensor topk_length_;
    CompactifyAttributes attributes_;
};

}  // namespace ftrain

#endif
