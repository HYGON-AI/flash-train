// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/env.h"
#include "flash_train/error.hpp"
#include "flash_train/family/sparse_attn_bwd/problem.hpp"

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

constexpr const char* kOpName = "SparseAttnBwd";

}  // namespace

SparseAttnBwdProblem::SparseAttnBwdProblem(Tensor q, Tensor kv, Tensor out, Tensor dout, Tensor lse, Tensor attn_sink,
                                           Tensor topk_indices, Tensor topk_length, Tensor dq, Tensor dkv, Tensor dsink,
                                           Tensor softmax_scale, SparseAttnBwdAttributes attributes)
    : q_(std::move(q)), kv_(std::move(kv)), out_(std::move(out)), dout_(std::move(dout)), lse_(std::move(lse)),
      attn_sink_(std::move(attn_sink)), topk_indices_(std::move(topk_indices)), topk_length_(std::move(topk_length)),
      dq_(std::move(dq)), dkv_(std::move(dkv)), dsink_(std::move(dsink)), softmax_scale_(std::move(softmax_scale)),
      attributes_(attributes) {
    // Capture once per problem, before selection; execute never rereads the environment.
    if (attributes_.getComputeType() == FTRAIN_NUMERIC_TYPE_BF16) {
        const char* cached = std::getenv(DSA_SAB2_CACHED);
        cached_backward_   = cached == nullptr || std::atoi(cached) != 0;
        const char* mode   = std::getenv(DSA_BWD_WORKSPACE);
        if (mode && *mode && std::strcmp(mode, "0") != 0) {
            if (std::strcmp(mode, "1") != 0) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "DSA_BWD_WORKSPACE must be 0 (high performance) or 1 (low memory)");
            }
            low_memory_workspace_ = true;
        }
    }
    const StorageView& softmax_scale_view = softmax_scale_.getStorageView();
    const StorageView& q_view             = q_.getStorageView();
    const StorageView& kv_view            = kv_.getStorageView();
    const StorageView& out_view           = out_.getStorageView();
    const StorageView& dout_view          = dout_.getStorageView();
    const StorageView& lse_view           = lse_.getStorageView();
    const StorageView& attn_sink_view     = attn_sink_.getStorageView();
    const StorageView& topk_indices_view  = topk_indices_.getStorageView();
    const StorageView& topk_length_view   = topk_length_.getStorageView();
    const StorageView& dq_view            = dq_.getStorageView();
    const StorageView& dkv_view           = dkv_.getStorageView();
    const StorageView& dsink_view         = dsink_.getStorageView();

    if (!softmax_scale_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!softmax_scale_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port : {&q_, &kv_, &out_, &dout_, &lse_, &attn_sink_, &topk_indices_, &topk_length_, &dq_, &dkv_,
                               &dsink_, &softmax_scale_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    if (q_view.isHostMemory() || kv_view.isHostMemory() || out_view.isHostMemory() || dout_view.isHostMemory() ||
        lse_view.isHostMemory() || attn_sink_view.isHostMemory() || topk_indices_view.isHostMemory() ||
        topk_length_view.isHostMemory() || dq_view.isHostMemory() || dkv_view.isHostMemory() ||
        dsink_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }

    if (q_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "q");
    }
    if (kv_view.getDims().size() != 2 && kv_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "kv");
    }
    if (out_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "out");
    }
    if (dout_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "dout");
    }
    if (lse_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "lse");
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
    if (dq_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "dq");
    }
    if (dkv_view.getDims().size() != 2 && dkv_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "dkv");
    }
    if (dsink_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "dsink");
    }

    const std::size_t kv_rank = kv_view.getDims().size();
    if (kv_view.getDims()[0] == 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: kv must hold at least one entry", kOpName);
    }
    if (kv_rank == 3) {
        if (kv_view.getDims().size() != 3 || kv_view.getDims()[1] != 1) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "kv");
        }
    }
    if (kv_view.getDims()[kv_rank - 1] != q_view.getDims()[2]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "kv");
    }
    if (dkv_view.getDims() != kv_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "dkv");
    }

    if (out_view.getDims() != dout_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "out");
    }
    if (out_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "out");
    }
    if (out_view.getDims()[1] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "out");
    }
    const std::int64_t value_dim = out_view.getDims()[2];
    if (value_dim <= 0 || value_dim > q_view.getDims()[2]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: the value width must lie in 1..head_dim", kOpName);
    }
    if (dq_view.getDims() != q_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "dq");
    }
    if (lse_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "lse");
    }
    if (lse_view.getDims()[1] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "lse");
    }
    if (attn_sink_view.getDims()[0] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "attn_sink");
    }
    if (dsink_view.getDims()[0] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "dsink");
    }
    if (topk_indices_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                        "topk_indices");
    }
    if (topk_length_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "topk_length");
    }
    if (topk_indices_view.getDims()[1] <= 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s must be positive", kOpName, "selection width");
    }
}

std::vector<std::uint64_t> SparseAttnBwdProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(256);
    key.push_back(low_memory_workspace_);
    key.push_back(cached_backward_);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&q_, &kv_, &out_, &dout_, &lse_, &attn_sink_, &topk_indices_, &topk_length_, &dq_, &dkv_,
                               &dsink_, &softmax_scale_}) {
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
