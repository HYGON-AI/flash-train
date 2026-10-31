// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/indexer_topk/problem.hpp"

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

constexpr const char* kOpName = "IndexerTopK";

}  // namespace

IndexerTopKProblem::IndexerTopKProblem(Tensor input_values, Tensor seq_lens, Tensor out_indices, Tensor out_values,
                                       Tensor top_k, Tensor next_n, Tensor order_values,
                                       IndexerTopKAttributes attributes)
    : input_values_(std::move(input_values)), seq_lens_(std::move(seq_lens)), out_indices_(std::move(out_indices)),
      out_values_(std::move(out_values)), top_k_(std::move(top_k)), next_n_(std::move(next_n)),
      order_values_(std::move(order_values)), attributes_(attributes) {
    const StorageView& top_k_view        = top_k_.getStorageView();
    const StorageView& next_n_view       = next_n_.getStorageView();
    const StorageView& order_values_view = order_values_.getStorageView();
    const StorageView& input_values_view = input_values_.getStorageView();
    const StorageView& seq_lens_view     = seq_lens_.getStorageView();
    const StorageView& out_indices_view  = out_indices_.getStorageView();
    const StorageView& out_values_view   = out_values_.getStorageView();

    if (!top_k_view.getDims().empty() || !next_n_view.getDims().empty() || !order_values_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!top_k_view.isHostMemory() || !next_n_view.isHostMemory() || !order_values_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port :
         {&input_values_, &seq_lens_, &out_indices_, &out_values_, &top_k_, &next_n_, &order_values_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    // top_k participates in problem validation or implementation selection.
    if (top_k_view.getNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: top_k has the wrong scalar type", kOpName);
    }
    std::memcpy(&top_k_value_, top_k_view.getMemory(), sizeof(top_k_value_));
    // next_n participates in problem validation or implementation selection.
    if (next_n_view.getNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: next_n has the wrong scalar type", kOpName);
    }
    std::memcpy(&next_n_value_, next_n_view.getMemory(), sizeof(next_n_value_));
    // order_values participates in problem validation or implementation selection.
    if (order_values_view.getNumericType() != FTRAIN_NUMERIC_TYPE_UINT8) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: order_values has the wrong scalar type", kOpName);
    }
    std::memcpy(&order_values_value_, order_values_view.getMemory(), sizeof(order_values_value_));

    if (input_values_view.isHostMemory() || seq_lens_view.isHostMemory() || out_indices_view.isHostMemory() ||
        out_values_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }

    if (input_values_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "input_values");
    }
    if (seq_lens_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "seq_lens");
    }
    if (out_indices_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "out_indices");
    }
    if (out_values_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "out_values");
    }

    if (getTopK() <= 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s must be positive", kOpName, "top_k");
    }
    if (getNextN() <= 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s must be positive", kOpName, "next_n");
    }
    if (out_indices_view.getDims()[0] != dim(input_values_, 0)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "out_indices");
    }
    if (out_indices_view.getDims()[1] != getTopK()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "out_indices");
    }
    if (out_values_view.getDims() != out_indices_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "out_values");
    }

    // Every request's row budget must be covered by seq_lens, exactly as the
    // kernel's grid derivation assumes.
    const std::int64_t request_stride = getNextN();
    const std::int64_t needed         = static_cast<std::int64_t>(dim(input_values_, 0)) / request_stride +
                                        (dim(input_values_, 0) % request_stride != 0);
    if (dim(seq_lens_, 0) < needed) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: seq_lens holds %llu entries but %lld requests need %lld",
                        kOpName, static_cast<unsigned long long>(dim(seq_lens_, 0)),
                        static_cast<long long>(dim(input_values_, 0)), static_cast<long long>(needed));
    }
    if (dim(input_values_, 1) == 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: input_values must hold at least one column", kOpName);
    }
}

std::vector<std::uint64_t> IndexerTopKProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(96);
    key.push_back(static_cast<std::uint64_t>(getTopK()));
    key.push_back(static_cast<std::uint64_t>(getNextN()));
    key.push_back(static_cast<std::uint64_t>(getOrderValues()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port :
         {&input_values_, &seq_lens_, &out_indices_, &out_values_, &top_k_, &next_n_, &order_values_}) {
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
