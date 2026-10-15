// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/indexer_topk_thd/problem.hpp"

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

constexpr const char* kOpName = "IndexerTopKThd";

}  // namespace

IndexerTopKThdProblem::IndexerTopKThdProblem(Tensor input_values, Tensor cu_seqlens_q, Tensor cu_seqlens_kv,
                                             Tensor q_causal_offsets, Tensor out_indices, Tensor out_values,
                                             Tensor ratio, Tensor top_k, Tensor order_values,
                                             IndexerTopKThdAttributes attributes)
    : input_values_(std::move(input_values)), cu_seqlens_q_(std::move(cu_seqlens_q)),
      cu_seqlens_kv_(std::move(cu_seqlens_kv)), q_causal_offsets_(std::move(q_causal_offsets)),
      out_indices_(std::move(out_indices)), out_values_(std::move(out_values)), ratio_(std::move(ratio)),
      top_k_(std::move(top_k)), order_values_(std::move(order_values)), attributes_(attributes) {
    const StorageView& ratio_view            = ratio_.getStorageView();
    const StorageView& top_k_view            = top_k_.getStorageView();
    const StorageView& order_values_view     = order_values_.getStorageView();
    const StorageView& input_values_view     = input_values_.getStorageView();
    const StorageView& cu_seqlens_q_view     = cu_seqlens_q_.getStorageView();
    const StorageView& cu_seqlens_kv_view    = cu_seqlens_kv_.getStorageView();
    const StorageView& q_causal_offsets_view = q_causal_offsets_.getStorageView();
    const StorageView& out_indices_view      = out_indices_.getStorageView();
    const StorageView& out_values_view       = out_values_.getStorageView();

    if (!ratio_view.getDims().empty() || !top_k_view.getDims().empty() || !order_values_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!ratio_view.isHostMemory() || !top_k_view.isHostMemory() || !order_values_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port : {&input_values_, &cu_seqlens_q_, &cu_seqlens_kv_, &q_causal_offsets_, &out_indices_,
                               &out_values_, &ratio_, &top_k_, &order_values_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    // ratio participates in problem validation or implementation selection.
    if (ratio_view.getNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: ratio has the wrong scalar type", kOpName);
    }
    std::memcpy(&ratio_value_, ratio_view.getMemory(), sizeof(ratio_value_));
    // top_k participates in problem validation or implementation selection.
    if (top_k_view.getNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: top_k has the wrong scalar type", kOpName);
    }
    std::memcpy(&top_k_value_, top_k_view.getMemory(), sizeof(top_k_value_));
    // order_values participates in problem validation or implementation selection.
    if (order_values_view.getNumericType() != FTRAIN_NUMERIC_TYPE_UINT8) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: order_values has the wrong scalar type", kOpName);
    }
    std::memcpy(&order_values_value_, order_values_view.getMemory(), sizeof(order_values_value_));

    if (getRatio() <= 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s must be positive", kOpName, "ratio");
    }
    if (input_values_view.isHostMemory() || cu_seqlens_q_view.isHostMemory() || cu_seqlens_kv_view.isHostMemory() ||
        q_causal_offsets_view.isHostMemory() || out_indices_view.isHostMemory() || out_values_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }

    if (input_values_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "input_values");
    }
    if (cu_seqlens_q_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "cu_seqlens_q");
    }
    if (cu_seqlens_kv_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "cu_seqlens_kv");
    }
    if (q_causal_offsets_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "q_causal_offsets");
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
    if (cu_seqlens_q_view.getDims()[0] != cu_seqlens_kv_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                        "cu_seqlens_q");
    }
    if (cu_seqlens_q_view.getDims()[0] < 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: cu_seqlens_q must hold at least two entries", kOpName);
    }
    if (q_causal_offsets_view.getDims()[0] != cu_seqlens_q_view.getDims()[0] - 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                        "q_causal_offsets");
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
    if (dim(input_values_, 1) == 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: input_values must hold at least one column", kOpName);
    }
}

std::vector<std::uint64_t> IndexerTopKThdProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(128);
    key.push_back(static_cast<std::uint64_t>(getRatio()));
    key.push_back(static_cast<std::uint64_t>(getTopK()));
    key.push_back(static_cast<std::uint64_t>(getOrderValues()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&input_values_, &cu_seqlens_q_, &cu_seqlens_kv_, &q_causal_offsets_, &out_indices_,
                               &out_values_, &ratio_, &top_k_, &order_values_}) {
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
