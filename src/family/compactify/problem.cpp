// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/compactify/problem.hpp"

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

constexpr const char* kOpName = "Compactify";

}  // namespace

CompactifyProblem::CompactifyProblem(Tensor indices, Tensor output, Tensor topk_length, CompactifyAttributes attributes)
    : indices_(std::move(indices)), output_(std::move(output)), topk_length_(std::move(topk_length)),
      attributes_(attributes) {
    const StorageView& indices_view     = indices_.getStorageView();
    const StorageView& output_view      = output_.getStorageView();
    const StorageView& topk_length_view = topk_length_.getStorageView();

    for (const Tensor* port : {&indices_, &output_, &topk_length_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    if (indices_view.isHostMemory() || output_view.isHostMemory() || topk_length_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }
    if (indices_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "indices");
    }
    if (output_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "output");
    }
    if (topk_length_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "topk_length");
    }
    if (output_view.getDims() != indices_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "output");
    }
    if (topk_length_view.getDims()[0] != dim(indices_, 0)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "topk_length");
    }
}

bool CompactifyProblem::hasOverlappingBuffers() const noexcept {
    const auto overlaps = [](const Tensor& a, const Tensor& b) {
        const auto& av = a.getStorageView();
        const auto& bv = b.getStorageView();
        // Other layouts are rejected by the primitive; do not invent their byte spans.
        if (av.getIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS || bv.getIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
            av.getMemory() == nullptr || bv.getMemory() == nullptr) {
            return false;
        }
        std::uint64_t a_bytes = elementBytes(av.getNumericType());
        std::uint64_t b_bytes = elementBytes(bv.getNumericType());
        // The constructor has already checked these products for overflow.
        for (const auto dim : av.getDims()) { a_bytes *= static_cast<std::uint64_t>(dim); }
        for (const auto dim : bv.getDims()) { b_bytes *= static_cast<std::uint64_t>(dim); }
        if (a_bytes == 0 || b_bytes == 0) { return false; }
        const auto a_begin = reinterpret_cast<std::uintptr_t>(av.getMemory());
        const auto b_begin = reinterpret_cast<std::uintptr_t>(bv.getMemory());
        // Subtract the lower address instead of forming potentially overflowing end pointers.
        return a_begin <= b_begin ? b_begin - a_begin < a_bytes : a_begin - b_begin < b_bytes;
    };
    return overlaps(indices_, output_) || overlaps(indices_, topk_length_) || overlaps(output_, topk_length_);
}

std::vector<std::uint64_t> CompactifyProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(64);
    key.push_back(hasOverlappingBuffers());
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&indices_, &output_, &topk_length_}) {
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
