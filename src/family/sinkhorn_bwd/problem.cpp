// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/sinkhorn_bwd/problem.hpp"

namespace ftrain {
namespace {

std::uint64_t elementBytes(FTrainNumericType dtype) noexcept {
    switch (dtype) {
        case FTRAIN_NUMERIC_TYPE_FP64:
        case FTRAIN_NUMERIC_TYPE_INT64:
        case FTRAIN_NUMERIC_TYPE_UINT64: return 8;
        case FTRAIN_NUMERIC_TYPE_FP32:
        case FTRAIN_NUMERIC_TYPE_INT32:
        case FTRAIN_NUMERIC_TYPE_UINT32: return 4;
        case FTRAIN_NUMERIC_TYPE_FP16:
        case FTRAIN_NUMERIC_TYPE_BF16:
        case FTRAIN_NUMERIC_TYPE_INT16:
        case FTRAIN_NUMERIC_TYPE_UINT16: return 2;
        default: return 1;
    }
}

constexpr const char* kOpName = "SinkhornBwd";

}  // namespace

SinkhornBwdProblem::SinkhornBwdProblem(Tensor grad_output, Tensor m_init, Tensor grad_logits, Tensor num_iterations,
                                       Tensor eps, SinkhornBwdAttributes attributes)
    : grad_output_(std::move(grad_output)), m_init_(std::move(m_init)), grad_logits_(std::move(grad_logits)),
      num_iterations_(std::move(num_iterations)), eps_(std::move(eps)), attributes_(attributes) {
    const StorageView& num_iterations_view = num_iterations_.getStorageView();
    const StorageView& eps_view            = eps_.getStorageView();
    const StorageView& grad_output_view    = grad_output_.getStorageView();
    const StorageView& grad_logits_view    = grad_logits_.getStorageView();
    const StorageView& m_init_view         = m_init_.getStorageView();

    if (!num_iterations_view.getDims().empty() || !eps_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!num_iterations_view.isHostMemory() || !eps_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port : {&grad_output_, &m_init_, &grad_logits_, &num_iterations_, &eps_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    // num_iterations participates in problem validation or implementation selection.
    if (num_iterations_view.getNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: num_iterations has the wrong scalar type", kOpName);
    }
    std::memcpy(&num_iterations_value_, num_iterations_view.getMemory(), sizeof(num_iterations_value_));
    // eps participates in problem validation or implementation selection.
    if (eps_view.getNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: eps has the wrong scalar type", kOpName);
    }
    std::memcpy(&eps_value_, eps_view.getMemory(), sizeof(eps_value_));

    if (grad_output_view.isHostMemory() || m_init_view.isHostMemory() || grad_logits_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }
    if (grad_output_view.getDims().size() < 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_output");
    }
    if (grad_logits_view.getDims() != grad_output_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_logits");
    }
    if (m_init_view.getDims().size() < 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "m_init");
    }

    const std::vector<std::int64_t>& dims = grad_output_view.getDims();
    const std::int64_t size               = dims.back();
    if (size <= 0 || dims[dims.size() - 2] != size) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: grad_output must be nonempty and square", kOpName);
    }
    const auto& state_dims = m_init_view.getDims();
    if (state_dims.back() != size || state_dims[state_dims.size() - 2] != size ||
        numElements(m_init_) != numElements(grad_output_)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "%s: m_init must hold one FP32 HC x HC matrix per gradient matrix", kOpName);
    }
    if (getNumIterations() < 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: num_iterations must be non-negative", kOpName);
    }
    if (!(getEps() > 0.0F) || !std::isfinite(getEps())) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: eps must be finite and positive", kOpName);
    }
}

std::vector<std::uint64_t> SinkhornBwdProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(64);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    key.push_back(static_cast<std::uint64_t>(getNumIterations()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&grad_output_, &m_init_, &grad_logits_, &num_iterations_, &eps_}) {
        const auto& view = port->getStorageView();
        key.push_back(static_cast<std::uint64_t>(view.getNumericType()));
        key.push_back(static_cast<std::uint64_t>(view.getIndexType()));
        key.push_back(view.getMemory() != nullptr);
        key.push_back(reinterpret_cast<std::uintptr_t>(view.getMemory()) % elementBytes(view.getNumericType()));
        key.push_back(view.getDims().size());
        for (const auto dim : view.getDims()) { key.push_back(static_cast<std::uint64_t>(dim)); }
        key.push_back(view.getStrides().size());
        for (const auto stride : view.getStrides()) { key.push_back(static_cast<std::uint64_t>(stride)); }
    }
    return key;
}

}  // namespace ftrain
