// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_POST_BDA_FWD_PROBLEM_HPP_
#define FTRAIN_FAMILY_H_POST_BDA_FWD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/h_post_bda_fwd.hpp"

namespace ftrain {

// One HPostBdaFwd call's parameters as a support checklist: h_resᵀ @ residual
// + h_post * (x + bias).
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class HPostBdaFwdProblem final {
  public:
    HPostBdaFwdProblem(Tensor h_res, Tensor residual, Tensor h_post, Tensor x, Tensor bias, Tensor output,
                       HPostBdaFwdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getHResDims() const noexcept { return h_res_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getResidualDims() const noexcept { return residual_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getHPostDims() const noexcept { return h_post_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getXDims() const noexcept { return x_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getBiasDims() const noexcept { return bias_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getOutputDims() const noexcept { return output_.getStorageView().getDims(); }

    std::uint64_t getStreams() const noexcept { return dim(h_res_, 2); }
    std::uint64_t getChannels() const noexcept { return dim(residual_, 3); }
    // Historical name: returns the flattened S*B row count, not a memory stride.
    std::uint64_t getBatchStride() const noexcept { return dim(residual_, 0) * dim(residual_, 1); }
    bool hasBias() const noexcept { return bias_.getStorageView().getMemory() != nullptr; }

    bool hasEmptyOutput() const noexcept { return getBatchStride() == 0 || getChannels() == 0; }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }

    FTrainNumericType getHResNumericType() const noexcept { return h_res_.getStorageView().getNumericType(); }
    FTrainNumericType getResidualNumericType() const noexcept { return residual_.getStorageView().getNumericType(); }
    FTrainNumericType getHPostNumericType() const noexcept { return h_post_.getStorageView().getNumericType(); }
    FTrainNumericType getXNumericType() const noexcept { return x_.getStorageView().getNumericType(); }
    FTrainNumericType getBiasNumericType() const noexcept { return bias_.getStorageView().getNumericType(); }
    FTrainNumericType getOutputNumericType() const noexcept { return output_.getStorageView().getNumericType(); }

    FTrainIndexType getHResIndexType() const noexcept { return h_res_.getStorageView().getIndexType(); }
    FTrainIndexType getResidualIndexType() const noexcept { return residual_.getStorageView().getIndexType(); }
    FTrainIndexType getHPostIndexType() const noexcept { return h_post_.getStorageView().getIndexType(); }
    FTrainIndexType getXIndexType() const noexcept { return x_.getStorageView().getIndexType(); }
    FTrainIndexType getBiasIndexType() const noexcept { return bias_.getStorageView().getIndexType(); }
    FTrainIndexType getOutputIndexType() const noexcept { return output_.getStorageView().getIndexType(); }

    // getBiasAddress may be null: the bias is optional.
    void* getHResAddress() const noexcept { return h_res_.getStorageView().getMemory(); }
    void* getResidualAddress() const noexcept { return residual_.getStorageView().getMemory(); }
    void* getHPostAddress() const noexcept { return h_post_.getStorageView().getMemory(); }
    void* getXAddress() const noexcept { return x_.getStorageView().getMemory(); }
    void* getBiasAddress() const noexcept { return bias_.getStorageView().getMemory(); }
    void* getOutputAddress() const noexcept { return output_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dim(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }

    Tensor h_res_;
    Tensor residual_;
    Tensor h_post_;
    Tensor x_;
    Tensor bias_;
    Tensor output_;
    HPostBdaFwdAttributes attributes_;
};

}  // namespace ftrain

#endif
