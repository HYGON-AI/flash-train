// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_PROJ_RMS_FWD_PROBLEM_HPP_
#define FTRAIN_FAMILY_PROJ_RMS_FWD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/proj_rms_fwd.hpp"

namespace ftrain {

// One ProjRmsFwd call's parameters as a support checklist: the projection
// proj = x @ weightᵀ together with the per-row RMS factor
// r = 1 / (|x| / sqrt(K) + eps), both produced by one launch.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class ProjRmsFwdProblem final {
  public:
    ProjRmsFwdProblem(Tensor x, Tensor weight, Tensor proj, Tensor r, Tensor eps, ProjRmsFwdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getXDims() const noexcept { return x_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getWeightDims() const noexcept { return weight_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getProjDims() const noexcept { return proj_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getRDims() const noexcept { return r_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getEpsDims() const noexcept { return eps_.getStorageView().getDims(); }

    std::uint64_t getRows() const noexcept { return dim(x_, 0); }
    std::uint64_t getOutputs() const noexcept { return dim(weight_, 0); }
    std::uint64_t getReduction() const noexcept { return dim(x_, 1); }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }
    float getEps() const noexcept { return eps_value_; }

    FTrainNumericType getXNumericType() const noexcept { return x_.getStorageView().getNumericType(); }
    FTrainNumericType getWeightNumericType() const noexcept { return weight_.getStorageView().getNumericType(); }
    FTrainNumericType getProjNumericType() const noexcept { return proj_.getStorageView().getNumericType(); }
    FTrainNumericType getRNumericType() const noexcept { return r_.getStorageView().getNumericType(); }

    FTrainIndexType getXIndexType() const noexcept { return x_.getStorageView().getIndexType(); }
    FTrainIndexType getWeightIndexType() const noexcept { return weight_.getStorageView().getIndexType(); }
    FTrainIndexType getProjIndexType() const noexcept { return proj_.getStorageView().getIndexType(); }
    FTrainIndexType getRIndexType() const noexcept { return r_.getStorageView().getIndexType(); }

    void* getXAddress() const noexcept { return x_.getStorageView().getMemory(); }
    void* getWeightAddress() const noexcept { return weight_.getStorageView().getMemory(); }
    void* getProjAddress() const noexcept { return proj_.getStorageView().getMemory(); }
    void* getRAddress() const noexcept { return r_.getStorageView().getMemory(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getEpsNumericType() const noexcept { return eps_.getStorageView().getNumericType(); }
    FTrainIndexType getEpsIndexType() const noexcept { return eps_.getStorageView().getIndexType(); }
    void* getEpsAddress() const noexcept { return eps_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dim(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }

    Tensor x_;
    Tensor weight_;
    Tensor proj_;
    Tensor r_;
    Tensor eps_;
    float eps_value_{};
    ProjRmsFwdAttributes attributes_;
};

}  // namespace ftrain

#endif
