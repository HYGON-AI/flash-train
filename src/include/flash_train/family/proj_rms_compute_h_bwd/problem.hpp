// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_BWD_PROBLEM_HPP_
#define FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_BWD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/proj_rms_compute_h_bwd.hpp"

namespace ftrain {

// One ProjRmsComputeHBwd call's parameters as a support checklist: the
// gradients of the projection, RMS factor, and activation split, including the
// scalar and bias gradients the mHC mapping parameters need.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class ProjRmsComputeHBwdProblem final {
  public:
    ProjRmsComputeHBwdProblem(Tensor grad_h_pre, Tensor grad_h_post, Tensor grad_h_res, Tensor grad_r, Tensor h_pre,
                              Tensor h_post, Tensor proj, Tensor r, Tensor x, Tensor weight, Tensor alpha_pre,
                              Tensor alpha_post, Tensor alpha_res, Tensor grad_x, Tensor grad_weight, Tensor grad_alpha,
                              Tensor grad_bias, Tensor n, Tensor eps, Tensor compute_h_eps,
                              ProjRmsComputeHBwdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getGradHPreDims() const noexcept { return grad_h_pre_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getGradHPostDims() const noexcept {
        return grad_h_post_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getGradHResDims() const noexcept { return grad_h_res_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getGradRDims() const noexcept { return grad_r_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getHPreDims() const noexcept { return h_pre_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getHPostDims() const noexcept { return h_post_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getProjDims() const noexcept { return proj_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getRDims() const noexcept { return r_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getXDims() const noexcept { return x_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getWeightDims() const noexcept { return weight_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getAlphaPreDims() const noexcept { return alpha_pre_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getAlphaPostDims() const noexcept {
        return alpha_post_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getAlphaResDims() const noexcept { return alpha_res_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getGradXDims() const noexcept { return grad_x_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getGradWeightDims() const noexcept {
        return grad_weight_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getGradAlphaDims() const noexcept {
        return grad_alpha_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getGradBiasDims() const noexcept { return grad_bias_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getNDims() const noexcept { return n_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getEpsDims() const noexcept { return eps_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getComputeHEpsDims() const noexcept {
        return compute_h_eps_.getStorageView().getDims();
    }

    std::uint64_t getRows() const noexcept { return dim(x_, 0); }
    std::uint64_t getOutputs() const noexcept { return dim(weight_, 0); }
    std::uint64_t getReduction() const noexcept { return dim(x_, 1); }
    std::uint64_t getNumStreams() const noexcept { return static_cast<std::uint64_t>(n_value_); }
    std::uint64_t getBatchStride() const noexcept { return getRows(); }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }
    std::int64_t getStreamCount() const noexcept { return n_value_; }
    float getEps() const noexcept { return eps_value_; }
    float getComputeHEps() const noexcept { return compute_h_eps_value_; }

    FTrainNumericType getGradHPreNumericType() const noexcept { return grad_h_pre_.getStorageView().getNumericType(); }
    FTrainNumericType getGradHPostNumericType() const noexcept {
        return grad_h_post_.getStorageView().getNumericType();
    }
    FTrainNumericType getGradHResNumericType() const noexcept { return grad_h_res_.getStorageView().getNumericType(); }
    FTrainNumericType getGradRNumericType() const noexcept { return grad_r_.getStorageView().getNumericType(); }
    FTrainNumericType getHPreNumericType() const noexcept { return h_pre_.getStorageView().getNumericType(); }
    FTrainNumericType getHPostNumericType() const noexcept { return h_post_.getStorageView().getNumericType(); }
    FTrainNumericType getProjNumericType() const noexcept { return proj_.getStorageView().getNumericType(); }
    FTrainNumericType getRNumericType() const noexcept { return r_.getStorageView().getNumericType(); }
    FTrainNumericType getXNumericType() const noexcept { return x_.getStorageView().getNumericType(); }
    FTrainNumericType getWeightNumericType() const noexcept { return weight_.getStorageView().getNumericType(); }
    FTrainNumericType getAlphaPreNumericType() const noexcept { return alpha_pre_.getStorageView().getNumericType(); }
    FTrainNumericType getAlphaPostNumericType() const noexcept { return alpha_post_.getStorageView().getNumericType(); }
    FTrainNumericType getAlphaResNumericType() const noexcept { return alpha_res_.getStorageView().getNumericType(); }
    FTrainNumericType getGradXNumericType() const noexcept { return grad_x_.getStorageView().getNumericType(); }
    FTrainNumericType getGradWeightNumericType() const noexcept {
        return grad_weight_.getStorageView().getNumericType();
    }
    FTrainNumericType getGradAlphaNumericType() const noexcept { return grad_alpha_.getStorageView().getNumericType(); }
    FTrainNumericType getGradBiasNumericType() const noexcept { return grad_bias_.getStorageView().getNumericType(); }

    FTrainIndexType getGradHPreIndexType() const noexcept { return grad_h_pre_.getStorageView().getIndexType(); }
    FTrainIndexType getGradHPostIndexType() const noexcept { return grad_h_post_.getStorageView().getIndexType(); }
    FTrainIndexType getGradHResIndexType() const noexcept { return grad_h_res_.getStorageView().getIndexType(); }
    FTrainIndexType getGradRIndexType() const noexcept { return grad_r_.getStorageView().getIndexType(); }
    FTrainIndexType getHPreIndexType() const noexcept { return h_pre_.getStorageView().getIndexType(); }
    FTrainIndexType getHPostIndexType() const noexcept { return h_post_.getStorageView().getIndexType(); }
    FTrainIndexType getProjIndexType() const noexcept { return proj_.getStorageView().getIndexType(); }
    FTrainIndexType getRIndexType() const noexcept { return r_.getStorageView().getIndexType(); }
    FTrainIndexType getXIndexType() const noexcept { return x_.getStorageView().getIndexType(); }
    FTrainIndexType getWeightIndexType() const noexcept { return weight_.getStorageView().getIndexType(); }
    FTrainIndexType getAlphaPreIndexType() const noexcept { return alpha_pre_.getStorageView().getIndexType(); }
    FTrainIndexType getAlphaPostIndexType() const noexcept { return alpha_post_.getStorageView().getIndexType(); }
    FTrainIndexType getAlphaResIndexType() const noexcept { return alpha_res_.getStorageView().getIndexType(); }
    FTrainIndexType getGradXIndexType() const noexcept { return grad_x_.getStorageView().getIndexType(); }
    FTrainIndexType getGradWeightIndexType() const noexcept { return grad_weight_.getStorageView().getIndexType(); }
    FTrainIndexType getGradAlphaIndexType() const noexcept { return grad_alpha_.getStorageView().getIndexType(); }
    FTrainIndexType getGradBiasIndexType() const noexcept { return grad_bias_.getStorageView().getIndexType(); }

    void* getGradHPreAddress() const noexcept { return grad_h_pre_.getStorageView().getMemory(); }
    void* getGradHPostAddress() const noexcept { return grad_h_post_.getStorageView().getMemory(); }
    void* getGradHResAddress() const noexcept { return grad_h_res_.getStorageView().getMemory(); }
    void* getGradRAddress() const noexcept { return grad_r_.getStorageView().getMemory(); }
    void* getHPreAddress() const noexcept { return h_pre_.getStorageView().getMemory(); }
    void* getHPostAddress() const noexcept { return h_post_.getStorageView().getMemory(); }
    void* getProjAddress() const noexcept { return proj_.getStorageView().getMemory(); }
    void* getRAddress() const noexcept { return r_.getStorageView().getMemory(); }
    void* getXAddress() const noexcept { return x_.getStorageView().getMemory(); }
    void* getWeightAddress() const noexcept { return weight_.getStorageView().getMemory(); }
    void* getAlphaPreAddress() const noexcept { return alpha_pre_.getStorageView().getMemory(); }
    void* getAlphaPostAddress() const noexcept { return alpha_post_.getStorageView().getMemory(); }
    void* getAlphaResAddress() const noexcept { return alpha_res_.getStorageView().getMemory(); }
    void* getGradXAddress() const noexcept { return grad_x_.getStorageView().getMemory(); }
    void* getGradWeightAddress() const noexcept { return grad_weight_.getStorageView().getMemory(); }
    void* getGradAlphaAddress() const noexcept { return grad_alpha_.getStorageView().getMemory(); }
    void* getGradBiasAddress() const noexcept { return grad_bias_.getStorageView().getMemory(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getNNumericType() const noexcept { return n_.getStorageView().getNumericType(); }
    FTrainIndexType getNIndexType() const noexcept { return n_.getStorageView().getIndexType(); }
    void* getNAddress() const noexcept { return n_.getStorageView().getMemory(); }
    FTrainNumericType getEpsNumericType() const noexcept { return eps_.getStorageView().getNumericType(); }
    FTrainIndexType getEpsIndexType() const noexcept { return eps_.getStorageView().getIndexType(); }
    void* getEpsAddress() const noexcept { return eps_.getStorageView().getMemory(); }
    FTrainNumericType getComputeHEpsNumericType() const noexcept {
        return compute_h_eps_.getStorageView().getNumericType();
    }
    FTrainIndexType getComputeHEpsIndexType() const noexcept { return compute_h_eps_.getStorageView().getIndexType(); }
    void* getComputeHEpsAddress() const noexcept { return compute_h_eps_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dim(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }

    Tensor grad_h_pre_;
    Tensor grad_h_post_;
    Tensor grad_h_res_;
    Tensor grad_r_;
    Tensor h_pre_;
    Tensor h_post_;
    Tensor proj_;
    Tensor r_;
    Tensor x_;
    Tensor weight_;
    Tensor alpha_pre_;
    Tensor alpha_post_;
    Tensor alpha_res_;
    Tensor grad_x_;
    Tensor grad_weight_;
    Tensor grad_alpha_;
    Tensor grad_bias_;
    Tensor n_;
    Tensor eps_;
    Tensor compute_h_eps_;
    std::int64_t n_value_{};
    float eps_value_{};
    float compute_h_eps_value_{};
    ProjRmsComputeHBwdAttributes attributes_;
};

}  // namespace ftrain

#endif
