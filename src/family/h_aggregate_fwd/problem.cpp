// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/h_aggregate_fwd/problem.hpp"

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

constexpr const char* kOpName = "HAggregateFwd";

}  // namespace

HAggregateFwdProblem::HAggregateFwdProblem(Tensor x, Tensor h_pre, Tensor aggregated,
                                           HAggregateFwdAttributes attributes)
    : x_(std::move(x)), h_pre_(std::move(h_pre)), aggregated_(std::move(aggregated)), attributes_(attributes) {
    const StorageView& x_view          = x_.getStorageView();
    const StorageView& h_pre_view      = h_pre_.getStorageView();
    const StorageView& aggregated_view = aggregated_.getStorageView();

    for (const Tensor* port : {&x_, &h_pre_, &aggregated_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    if (x_view.isHostMemory() || h_pre_view.isHostMemory() || aggregated_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }
    if (x_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "x");
    }
    if (h_pre_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "h_pre");
    }
    if (aggregated_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "aggregated");
    }

    if (h_pre_view.getDims()[0] != x_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "h_pre");
    }
    if (h_pre_view.getDims()[1] != x_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "h_pre");
    }
    if (h_pre_view.getDims()[2] != x_view.getDims()[2]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "h_pre");
    }
    if (aggregated_view.getDims()[0] != x_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "aggregated");
    }
    if (aggregated_view.getDims()[1] != x_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "aggregated");
    }
    if (aggregated_view.getDims()[2] != x_view.getDims()[3]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "aggregated");
    }
}

std::vector<std::uint64_t> HAggregateFwdProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(64);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&x_, &h_pre_, &aggregated_}) {
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
