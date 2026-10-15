// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SINKHORN_FWD_PROBLEM_HPP_
#define FTRAIN_FAMILY_SINKHORN_FWD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/sinkhorn_fwd.hpp"

namespace ftrain {

// One SinkhornFwd call's parameters as a support checklist: the Sinkhorn-Knopp
// projection of a batch of square logit matrices, plus the FP32 softmax+eps
// state its backward consumes.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class SinkhornFwdProblem final {
  public:
    SinkhornFwdProblem(Tensor logits, Tensor output, Tensor m_init, Tensor num_iterations, Tensor eps,
                       SinkhornFwdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getLogitsDims() const noexcept { return logits_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getOutputDims() const noexcept { return output_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getMInitDims() const noexcept { return m_init_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getNumIterationsDims() const noexcept {
        return num_iterations_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getEpsDims() const noexcept { return eps_.getStorageView().getDims(); }

    std::uint64_t getBatch() const noexcept { return numElements(logits_) / getMatrixSize() / getMatrixSize(); }
    std::uint64_t getMatrixSize() const noexcept {
        return static_cast<std::uint64_t>(logits_.getStorageView().getDims().back());
    }

    bool hasEmptyOutput() const noexcept { return getBatch() == 0; }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }
    std::int64_t getNumIterations() const noexcept { return num_iterations_value_; }
    float getEps() const noexcept { return eps_value_; }

    FTrainNumericType getLogitsNumericType() const noexcept { return logits_.getStorageView().getNumericType(); }
    FTrainNumericType getOutputNumericType() const noexcept { return output_.getStorageView().getNumericType(); }
    FTrainNumericType getMInitNumericType() const noexcept { return m_init_.getStorageView().getNumericType(); }

    FTrainIndexType getLogitsIndexType() const noexcept { return logits_.getStorageView().getIndexType(); }
    FTrainIndexType getOutputIndexType() const noexcept { return output_.getStorageView().getIndexType(); }
    FTrainIndexType getMInitIndexType() const noexcept { return m_init_.getStorageView().getIndexType(); }

    void* getLogitsAddress() const noexcept { return logits_.getStorageView().getMemory(); }
    void* getOutputAddress() const noexcept { return output_.getStorageView().getMemory(); }
    void* getMInitAddress() const noexcept { return m_init_.getStorageView().getMemory(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getNumIterationsNumericType() const noexcept {
        return num_iterations_.getStorageView().getNumericType();
    }
    FTrainIndexType getNumIterationsIndexType() const noexcept {
        return num_iterations_.getStorageView().getIndexType();
    }
    void* getNumIterationsAddress() const noexcept { return num_iterations_.getStorageView().getMemory(); }
    FTrainNumericType getEpsNumericType() const noexcept { return eps_.getStorageView().getNumericType(); }
    FTrainIndexType getEpsIndexType() const noexcept { return eps_.getStorageView().getIndexType(); }
    void* getEpsAddress() const noexcept { return eps_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t numElements(const Tensor& port) noexcept {
        std::uint64_t count = 1;
        for (const std::int64_t dim : port.getStorageView().getDims()) { count *= static_cast<std::uint64_t>(dim); }
        return count;
    }

    Tensor logits_;
    Tensor output_;
    Tensor m_init_;
    Tensor num_iterations_;
    Tensor eps_;
    std::int64_t num_iterations_value_{};
    float eps_value_{};
    SinkhornFwdAttributes attributes_;
};

}  // namespace ftrain

#endif
