// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/sparse_attn_target/problem.hpp"

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

constexpr const char* kOpName = "SparseAttnTarget";

}  // namespace

SparseAttnTargetProblem::SparseAttnTargetProblem(Tensor q, Tensor k, Tensor lse, Tensor topk_indices, Tensor target,
                                                 Tensor softmax_scale, Tensor topk_indices_global,
                                                 SparseAttnTargetAttributes attributes)
    : q_(std::move(q)), k_(std::move(k)), lse_(std::move(lse)), topk_indices_(std::move(topk_indices)),
      target_(std::move(target)), softmax_scale_(std::move(softmax_scale)),
      topk_indices_global_(std::move(topk_indices_global)), attributes_(attributes) {
    const StorageView& softmax_scale_view       = softmax_scale_.getStorageView();
    const StorageView& topk_indices_global_view = topk_indices_global_.getStorageView();
    const StorageView& q_view                   = q_.getStorageView();
    const StorageView& k_view                   = k_.getStorageView();
    const StorageView& lse_view                 = lse_.getStorageView();
    const StorageView& topk_indices_view        = topk_indices_.getStorageView();
    const StorageView& target_view              = target_.getStorageView();

    if (!softmax_scale_view.getDims().empty() || !topk_indices_global_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!softmax_scale_view.isHostMemory() || !topk_indices_global_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port : {&q_, &k_, &lse_, &topk_indices_, &target_, &softmax_scale_, &topk_indices_global_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    if (q_view.isHostMemory() || k_view.isHostMemory() || lse_view.isHostMemory() || topk_indices_view.isHostMemory() ||
        target_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }

    if (q_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "q");
    }
    if (k_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "k");
    }
    if (lse_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "lse");
    }
    if (topk_indices_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "topk_indices");
    }
    if (target_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "target");
    }

    if (k_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "k");
    }
    if (k_view.getDims()[2] != q_view.getDims()[3]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "k");
    }
    if (lse_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "lse");
    }
    if (lse_view.getDims()[1] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "lse");
    }
    if (lse_view.getDims()[2] != q_view.getDims()[2]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "lse");
    }
    if (topk_indices_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                        "topk_indices");
    }
    if (topk_indices_view.getDims()[1] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                        "topk_indices");
    }
    if (target_view.getDims() != topk_indices_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "target");
    }
    if (topk_indices_view.getDims()[2] <= 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s must be positive", kOpName, "selection width");
    }
}

std::vector<std::uint64_t> SparseAttnTargetProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(128);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&q_, &k_, &lse_, &topk_indices_, &target_, &softmax_scale_, &topk_indices_global_}) {
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
