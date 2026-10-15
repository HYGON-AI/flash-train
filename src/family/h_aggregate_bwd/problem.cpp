// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/h_aggregate_bwd/problem.hpp"

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

constexpr const char* kOpName = "HAggregateBwd";

}  // namespace

HAggregateBwdProblem::HAggregateBwdProblem(Tensor grad_aggregated, Tensor x, Tensor h_pre, Tensor grad_x,
                                           Tensor grad_h_pre, HAggregateBwdAttributes attributes)
    : grad_aggregated_(std::move(grad_aggregated)), x_(std::move(x)), h_pre_(std::move(h_pre)),
      grad_x_(std::move(grad_x)), grad_h_pre_(std::move(grad_h_pre)), attributes_(attributes) {
    const StorageView& grad_aggregated_view = grad_aggregated_.getStorageView();
    const StorageView& x_view               = x_.getStorageView();
    const StorageView& h_pre_view           = h_pre_.getStorageView();
    const StorageView& grad_x_view          = grad_x_.getStorageView();
    const StorageView& grad_h_pre_view      = grad_h_pre_.getStorageView();

    for (const Tensor* port : {&grad_aggregated_, &x_, &h_pre_, &grad_x_, &grad_h_pre_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    if (grad_aggregated_view.isHostMemory() || x_view.isHostMemory() || h_pre_view.isHostMemory() ||
        grad_x_view.isHostMemory() || grad_h_pre_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }
    if (grad_aggregated_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_aggregated");
    }
    if (x_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "x");
    }
    if (h_pre_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "h_pre");
    }
    if (grad_x_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_x");
    }
    if (grad_h_pre_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_h_pre");
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
    if (grad_aggregated_view.getDims()[0] != x_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                        "grad_aggregated");
    }
    if (grad_aggregated_view.getDims()[1] != x_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                        "grad_aggregated");
    }
    if (grad_aggregated_view.getDims()[2] != x_view.getDims()[3]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                        "grad_aggregated");
    }
    if (grad_x_view.getDims() != x_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_x");
    }
    if (grad_h_pre_view.getDims() != h_pre_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_h_pre");
    }
}

std::vector<std::uint64_t> HAggregateBwdProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(96);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&grad_aggregated_, &x_, &h_pre_, &grad_x_, &grad_h_pre_}) {
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
