// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/sparse_attn_fwd_indexer/problem.hpp"

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

constexpr const char* kOpName    = "SparseAttnFwdIndexer";
constexpr std::int64_t kValueDim = 512;

}  // namespace

SparseAttnFwdIndexerProblem::SparseAttnFwdIndexerProblem(Tensor q, Tensor kv, Tensor attn_sink, Tensor topk_indices,
                                                         Tensor topk_length, Tensor out, Tensor max_logits, Tensor lse,
                                                         Tensor lse_indexer, Tensor softmax_scale, Tensor indexer_topk,
                                                         SparseAttnFwdIndexerAttributes attributes)
    : q_(std::move(q)), kv_(std::move(kv)), attn_sink_(std::move(attn_sink)), topk_indices_(std::move(topk_indices)),
      topk_length_(std::move(topk_length)), out_(std::move(out)), max_logits_(std::move(max_logits)),
      lse_(std::move(lse)), lse_indexer_(std::move(lse_indexer)), softmax_scale_(std::move(softmax_scale)),
      indexer_topk_(std::move(indexer_topk)), attributes_(attributes) {
    const StorageView& softmax_scale_view = softmax_scale_.getStorageView();
    const StorageView& indexer_topk_view  = indexer_topk_.getStorageView();
    const StorageView& q_view             = q_.getStorageView();
    const StorageView& kv_view            = kv_.getStorageView();
    const StorageView& attn_sink_view     = attn_sink_.getStorageView();
    const StorageView& topk_indices_view  = topk_indices_.getStorageView();
    const StorageView& topk_length_view   = topk_length_.getStorageView();
    const StorageView& out_view           = out_.getStorageView();
    const StorageView& max_logits_view    = max_logits_.getStorageView();
    const StorageView& lse_view           = lse_.getStorageView();
    const StorageView& lse_indexer_view   = lse_indexer_.getStorageView();

    if (!softmax_scale_view.getDims().empty() || !indexer_topk_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!softmax_scale_view.isHostMemory() || !indexer_topk_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port : {&q_, &kv_, &attn_sink_, &topk_indices_, &topk_length_, &out_, &max_logits_, &lse_,
                               &lse_indexer_, &softmax_scale_, &indexer_topk_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    // indexer_topk participates in problem validation or implementation selection.
    if (indexer_topk_view.getNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: indexer_topk has the wrong scalar type", kOpName);
    }
    std::memcpy(&indexer_topk_value_, indexer_topk_view.getMemory(), sizeof(indexer_topk_value_));

    if (q_view.isHostMemory() || kv_view.isHostMemory() || attn_sink_view.isHostMemory() ||
        topk_indices_view.isHostMemory() || topk_length_view.isHostMemory() || out_view.isHostMemory() ||
        max_logits_view.isHostMemory() || lse_view.isHostMemory() || lse_indexer_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }

    if (q_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "q");
    }
    if (kv_view.getDims().size() != 2 && kv_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "kv");
    }
    if (attn_sink_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "attn_sink");
    }
    if (topk_indices_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "topk_indices");
    }
    if (topk_length_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "topk_length");
    }
    if (out_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "out");
    }
    if (max_logits_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "max_logits");
    }
    if (lse_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "lse");
    }
    if (lse_indexer_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "lse_indexer");
    }

    if (kv_view.getDims()[kv_view.getDims().size() - 1] != q_view.getDims()[2]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "kv");
    }
    if (out_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "out");
    }
    if (out_view.getDims()[1] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "out");
    }
    if (max_logits_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "max_logits");
    }
    if (max_logits_view.getDims()[1] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "max_logits");
    }
    if (lse_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "lse");
    }
    if (lse_view.getDims()[1] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "lse");
    }
    if (lse_indexer_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "lse_indexer");
    }
    if (lse_indexer_view.getDims()[1] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "lse_indexer");
    }
    if (attn_sink_view.getDims()[0] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "attn_sink");
    }
    if (topk_indices_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                        "topk_indices");
    }
    if (topk_length_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "topk_length");
    }
}

std::vector<std::uint64_t> SparseAttnFwdIndexerProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(192);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    key.push_back(static_cast<std::uint64_t>(getIndexerTopk()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&q_, &kv_, &attn_sink_, &topk_indices_, &topk_length_, &out_, &max_logits_, &lse_,
                               &lse_indexer_, &softmax_scale_, &indexer_topk_}) {
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
