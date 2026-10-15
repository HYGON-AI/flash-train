// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/dense_attn_score/problem.hpp"

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

constexpr const char* kOpName = "DenseAttnScore";

}  // namespace

DenseAttnScoreProblem::DenseAttnScoreProblem(Tensor q, Tensor k, Tensor lse, Tensor out, Tensor denom, Tensor ratio,
                                             Tensor softmax_scale, DenseAttnScoreAttributes attributes)
    : q_(std::move(q)), k_(std::move(k)), lse_(std::move(lse)), out_(std::move(out)), denom_(std::move(denom)),
      ratio_(std::move(ratio)), softmax_scale_(std::move(softmax_scale)), attributes_(attributes) {
    const StorageView& ratio_view         = ratio_.getStorageView();
    const StorageView& softmax_scale_view = softmax_scale_.getStorageView();
    const StorageView& q_view             = q_.getStorageView();
    const StorageView& k_view             = k_.getStorageView();
    const StorageView& lse_view           = lse_.getStorageView();
    const StorageView& out_view           = out_.getStorageView();
    const StorageView& denom_view         = denom_.getStorageView();

    if (!ratio_view.getDims().empty() || !softmax_scale_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!ratio_view.isHostMemory() || !softmax_scale_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port : {&q_, &k_, &lse_, &out_, &denom_, &ratio_, &softmax_scale_}) {
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

    if (getRatio() <= 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s must be positive", kOpName, "ratio");
    }
    if (q_view.isHostMemory() || k_view.isHostMemory() || lse_view.isHostMemory() || out_view.isHostMemory() ||
        denom_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }

    if (q_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "q");
    }
    if (k_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "k");
    }
    if (lse_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "lse");
    }
    if (out_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "out");
    }
    if (denom_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "denom");
    }

    if (k_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "k");
    }
    if (k_view.getDims()[3] != q_view.getDims()[3]) {
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
    if (out_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "out");
    }
    if (out_view.getDims()[1] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "out");
    }
    if (out_view.getDims()[2] != k_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "out");
    }
    if (denom_view.getDims()[0] != q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "denom");
    }
    if (denom_view.getDims()[1] != q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "denom");
    }
}

std::vector<std::uint64_t> DenseAttnScoreProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(96);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    key.push_back(static_cast<std::uint64_t>(getRatio()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&q_, &k_, &lse_, &out_, &denom_, &ratio_, &softmax_scale_}) {
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
