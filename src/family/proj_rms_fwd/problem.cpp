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
#include "flash_train/family/proj_rms_fwd/problem.hpp"

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

constexpr const char* kOpName = "ProjRmsFwd";

}  // namespace

ProjRmsFwdProblem::ProjRmsFwdProblem(Tensor x, Tensor weight, Tensor proj, Tensor r, Tensor eps,
                                     ProjRmsFwdAttributes attributes)
    : x_(std::move(x)), weight_(std::move(weight)), proj_(std::move(proj)), r_(std::move(r)), eps_(std::move(eps)),
      attributes_(attributes) {
    const StorageView& eps_view    = eps_.getStorageView();
    const StorageView& x_view      = x_.getStorageView();
    const StorageView& weight_view = weight_.getStorageView();
    const StorageView& proj_view   = proj_.getStorageView();
    const StorageView& r_view      = r_.getStorageView();

    if (!eps_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!eps_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port : {&x_, &weight_, &proj_, &r_, &eps_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    // eps participates in problem validation or implementation selection.
    if (eps_view.getNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: eps has the wrong scalar type", kOpName);
    }
    std::memcpy(&eps_value_, eps_view.getMemory(), sizeof(eps_value_));

    if (x_view.isHostMemory() || weight_view.isHostMemory() || proj_view.isHostMemory() || r_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }
    if (x_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "x");
    }
    if (weight_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "weight");
    }
    if (proj_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "proj");
    }
    if (r_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "r");
    }

    if (weight_view.getDims()[1] != x_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "weight");
    }
    if (proj_view.getDims()[0] != x_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "proj");
    }
    if (proj_view.getDims()[1] != weight_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "proj");
    }
    if (r_view.getDims()[0] != dim(x_, 0)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "r");
    }
    if (r_view.getDims()[1] != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "r");
    }
    if (getReduction() == 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: the RMS reduction depth must be positive", kOpName);
    }
    if (!(getEps() > 0.0F) || !std::isfinite(getEps())) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: eps must be finite and positive", kOpName);
    }
}

std::vector<std::uint64_t> ProjRmsFwdProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(96);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&x_, &weight_, &proj_, &r_, &eps_}) {
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
