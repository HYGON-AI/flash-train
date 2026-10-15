// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_POST_BDA_BWD_PROBLEM_HPP_
#define FTRAIN_FAMILY_H_POST_BDA_BWD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/h_post_bda_bwd.hpp"

namespace ftrain {

// One HPostBdaBwd call's parameters as a support checklist: the four operand
// gradients of the residual mixing and expansion, plus the bias gradient.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class HPostBdaBwdProblem final {
  public:
    HPostBdaBwdProblem(Tensor grad_output, Tensor residual, Tensor x, Tensor bias, Tensor h_res, Tensor h_post,
                       Tensor grad_residual, Tensor grad_x, Tensor grad_h_res, Tensor grad_h_post, Tensor grad_bias,
                       HPostBdaBwdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getGradOutputDims() const noexcept {
        return grad_output_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getResidualDims() const noexcept { return residual_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getXDims() const noexcept { return x_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getBiasDims() const noexcept { return bias_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getHResDims() const noexcept { return h_res_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getHPostDims() const noexcept { return h_post_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getGradResidualDims() const noexcept {
        return grad_residual_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getGradXDims() const noexcept { return grad_x_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getGradHResDims() const noexcept { return grad_h_res_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getGradHPostDims() const noexcept {
        return grad_h_post_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getGradBiasDims() const noexcept { return grad_bias_.getStorageView().getDims(); }

    std::uint64_t getStreams() const noexcept { return dim(h_res_, 2); }
    std::uint64_t getChannels() const noexcept { return dim(residual_, 3); }
    std::uint64_t getBatchStride() const noexcept { return dim(residual_, 0) * dim(residual_, 1); }
    bool hasBias() const noexcept { return bias_.getStorageView().getMemory() != nullptr; }
    bool hasBiasGradient() const noexcept { return grad_bias_.getStorageView().getMemory() != nullptr; }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }

    FTrainNumericType getGradOutputNumericType() const noexcept {
        return grad_output_.getStorageView().getNumericType();
    }
    FTrainNumericType getResidualNumericType() const noexcept { return residual_.getStorageView().getNumericType(); }
    FTrainNumericType getXNumericType() const noexcept { return x_.getStorageView().getNumericType(); }
    FTrainNumericType getBiasNumericType() const noexcept { return bias_.getStorageView().getNumericType(); }
    FTrainNumericType getHResNumericType() const noexcept { return h_res_.getStorageView().getNumericType(); }
    FTrainNumericType getHPostNumericType() const noexcept { return h_post_.getStorageView().getNumericType(); }
    FTrainNumericType getGradResidualNumericType() const noexcept {
        return grad_residual_.getStorageView().getNumericType();
    }
    FTrainNumericType getGradXNumericType() const noexcept { return grad_x_.getStorageView().getNumericType(); }
    FTrainNumericType getGradHResNumericType() const noexcept { return grad_h_res_.getStorageView().getNumericType(); }
    FTrainNumericType getGradHPostNumericType() const noexcept {
        return grad_h_post_.getStorageView().getNumericType();
    }
    FTrainNumericType getGradBiasNumericType() const noexcept { return grad_bias_.getStorageView().getNumericType(); }

    FTrainIndexType getGradOutputIndexType() const noexcept { return grad_output_.getStorageView().getIndexType(); }
    FTrainIndexType getResidualIndexType() const noexcept { return residual_.getStorageView().getIndexType(); }
    FTrainIndexType getXIndexType() const noexcept { return x_.getStorageView().getIndexType(); }
    FTrainIndexType getBiasIndexType() const noexcept { return bias_.getStorageView().getIndexType(); }
    FTrainIndexType getHResIndexType() const noexcept { return h_res_.getStorageView().getIndexType(); }
    FTrainIndexType getHPostIndexType() const noexcept { return h_post_.getStorageView().getIndexType(); }
    FTrainIndexType getGradResidualIndexType() const noexcept { return grad_residual_.getStorageView().getIndexType(); }
    FTrainIndexType getGradXIndexType() const noexcept { return grad_x_.getStorageView().getIndexType(); }
    FTrainIndexType getGradHResIndexType() const noexcept { return grad_h_res_.getStorageView().getIndexType(); }
    FTrainIndexType getGradHPostIndexType() const noexcept { return grad_h_post_.getStorageView().getIndexType(); }
    FTrainIndexType getGradBiasIndexType() const noexcept { return grad_bias_.getStorageView().getIndexType(); }

    // bias' and grad_bias' addresses may be null.
    void* getGradOutputAddress() const noexcept { return grad_output_.getStorageView().getMemory(); }
    void* getResidualAddress() const noexcept { return residual_.getStorageView().getMemory(); }
    void* getXAddress() const noexcept { return x_.getStorageView().getMemory(); }
    void* getBiasAddress() const noexcept { return bias_.getStorageView().getMemory(); }
    void* getHResAddress() const noexcept { return h_res_.getStorageView().getMemory(); }
    void* getHPostAddress() const noexcept { return h_post_.getStorageView().getMemory(); }
    void* getGradResidualAddress() const noexcept { return grad_residual_.getStorageView().getMemory(); }
    void* getGradXAddress() const noexcept { return grad_x_.getStorageView().getMemory(); }
    void* getGradHResAddress() const noexcept { return grad_h_res_.getStorageView().getMemory(); }
    void* getGradHPostAddress() const noexcept { return grad_h_post_.getStorageView().getMemory(); }
    void* getGradBiasAddress() const noexcept { return grad_bias_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dim(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }

    Tensor grad_output_;
    Tensor residual_;
    Tensor x_;
    Tensor bias_;
    Tensor h_res_;
    Tensor h_post_;
    Tensor grad_residual_;
    Tensor grad_x_;
    Tensor grad_h_res_;
    Tensor grad_h_post_;
    Tensor grad_bias_;
    HPostBdaBwdAttributes attributes_;
};

}  // namespace ftrain

#endif
