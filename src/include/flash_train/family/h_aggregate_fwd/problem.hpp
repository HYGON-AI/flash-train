// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_AGGREGATE_FWD_PROBLEM_HPP_
#define FTRAIN_FAMILY_H_AGGREGATE_FWD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/h_aggregate_fwd.hpp"

namespace ftrain {

// One HAggregateFwd call's parameters as a support checklist: the weighted
// n-stream to 1-stream aggregation.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class HAggregateFwdProblem final {
  public:
    HAggregateFwdProblem(Tensor x, Tensor h_pre, Tensor aggregated, HAggregateFwdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getXDims() const noexcept { return x_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getHPreDims() const noexcept { return h_pre_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getAggregatedDims() const noexcept {
        return aggregated_.getStorageView().getDims();
    }

    std::uint64_t getStreams() const noexcept { return dim(x_, 2); }
    std::uint64_t getChannels() const noexcept { return dim(x_, 3); }
    std::uint64_t getBatchStride() const noexcept { return dim(x_, 0) * dim(x_, 1); }

    bool hasEmptyOutput() const noexcept { return getBatchStride() == 0 || getChannels() == 0; }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }

    FTrainNumericType getXNumericType() const noexcept { return x_.getStorageView().getNumericType(); }
    FTrainNumericType getHPreNumericType() const noexcept { return h_pre_.getStorageView().getNumericType(); }
    FTrainNumericType getAggregatedNumericType() const noexcept {
        return aggregated_.getStorageView().getNumericType();
    }

    FTrainIndexType getXIndexType() const noexcept { return x_.getStorageView().getIndexType(); }
    FTrainIndexType getHPreIndexType() const noexcept { return h_pre_.getStorageView().getIndexType(); }
    FTrainIndexType getAggregatedIndexType() const noexcept { return aggregated_.getStorageView().getIndexType(); }

    void* getXAddress() const noexcept { return x_.getStorageView().getMemory(); }
    void* getHPreAddress() const noexcept { return h_pre_.getStorageView().getMemory(); }
    void* getAggregatedAddress() const noexcept { return aggregated_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dim(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }

    Tensor x_;
    Tensor h_pre_;
    Tensor aggregated_;
    HAggregateFwdAttributes attributes_;
};

}  // namespace ftrain

#endif
