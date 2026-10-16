// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/indexer_bwd_dense/problem.hpp"

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

constexpr const char* kOpName = "IndexerBwdDense";

}  // namespace

IndexerBwdDenseProblem::IndexerBwdDenseProblem(Tensor index_q, Tensor weights, Tensor index_k, Tensor attn_score,
                                               Tensor attn_l1norm, Tensor index_score, Tensor index_lse,
                                               Tensor grad_index_score, Tensor dq, Tensor dw, Tensor dk,
                                               Tensor sm_scale, Tensor loss_coeff, Tensor grad_loss, Tensor ratio,
                                               IndexerBwdDenseAttributes attributes)
    : index_q_(std::move(index_q)), weights_(std::move(weights)), index_k_(std::move(index_k)),
      attn_score_(std::move(attn_score)), attn_l1norm_(std::move(attn_l1norm)), index_score_(std::move(index_score)),
      index_lse_(std::move(index_lse)), grad_index_score_(std::move(grad_index_score)), dq_(std::move(dq)),
      dw_(std::move(dw)), dk_(std::move(dk)), sm_scale_(std::move(sm_scale)), loss_coeff_(std::move(loss_coeff)),
      grad_loss_(std::move(grad_loss)), ratio_(std::move(ratio)), attributes_(attributes) {
    const StorageView& sm_scale_view         = sm_scale_.getStorageView();
    const StorageView& loss_coeff_view       = loss_coeff_.getStorageView();
    const StorageView& grad_loss_view        = grad_loss_.getStorageView();
    const StorageView& ratio_view            = ratio_.getStorageView();
    const StorageView& index_q_view          = index_q_.getStorageView();
    const StorageView& weights_view          = weights_.getStorageView();
    const StorageView& index_k_view          = index_k_.getStorageView();
    const StorageView& attn_score_view       = attn_score_.getStorageView();
    const StorageView& attn_l1norm_view      = attn_l1norm_.getStorageView();
    const StorageView& index_score_view      = index_score_.getStorageView();
    const StorageView& index_lse_view        = index_lse_.getStorageView();
    const StorageView& grad_index_score_view = grad_index_score_.getStorageView();
    const StorageView& dq_view               = dq_.getStorageView();
    const StorageView& dw_view               = dw_.getStorageView();
    const StorageView& dk_view               = dk_.getStorageView();

    if (!sm_scale_view.getDims().empty() || !loss_coeff_view.getDims().empty() || !grad_loss_view.getDims().empty() ||
        !ratio_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!sm_scale_view.isHostMemory() || !loss_coeff_view.isHostMemory() || !grad_loss_view.isHostMemory() ||
        !ratio_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port : {&index_q_, &weights_, &index_k_, &attn_score_, &attn_l1norm_, &index_score_, &index_lse_,
                               &grad_index_score_, &dq_, &dw_, &dk_, &sm_scale_, &loss_coeff_, &grad_loss_, &ratio_}) {
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
    if (index_q_view.isHostMemory() || weights_view.isHostMemory() || index_k_view.isHostMemory() ||
        attn_score_view.isHostMemory() || attn_l1norm_view.isHostMemory() || index_score_view.isHostMemory() ||
        index_lse_view.isHostMemory() || grad_index_score_view.isHostMemory() || dq_view.isHostMemory() ||
        dw_view.isHostMemory() || dk_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }

    if (index_q_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "index_q");
    }
    if (weights_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "weights");
    }
    if (index_k_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "index_k");
    }
    if (attn_score_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "attn_score");
    }
    if (attn_l1norm_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "attn_l1norm");
    }
    if (index_score_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "index_score");
    }
    if (index_lse_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "index_lse");
    }
    if (grad_index_score_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_index_score");
    }
    if (dq_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "dq");
    }
    if (dw_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "dw");
    }
    if (dk_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "dk");
    }

    if (grad_index_score_view.getDims() != index_score_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName,
                        "grad_index_score");
    }
    if (attn_score_view.getDims() != index_score_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "attn_score");
    }
    if (index_k_view.getDims()[0] != index_q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "index_k");
    }
    if (index_k_view.getDims()[2] != index_q_view.getDims()[3]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "index_k");
    }
    if (index_score_view.getDims()[0] != index_q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "index_score");
    }
    if (index_score_view.getDims()[1] != index_q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "index_score");
    }
    if (index_score_view.getDims()[2] != index_k_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "index_score");
    }
    if (attn_l1norm_view.getDims()[0] != index_q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "attn_l1norm");
    }
    if (attn_l1norm_view.getDims()[1] != index_q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "attn_l1norm");
    }
    if (index_lse_view.getDims()[0] != index_q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "index_lse");
    }
    if (index_lse_view.getDims()[1] != index_q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "index_lse");
    }
    if (weights_view.getDims()[0] != index_q_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "weights");
    }
    if (weights_view.getDims()[1] != index_q_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "weights");
    }
    if (weights_view.getDims()[2] != index_q_view.getDims()[2]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "weights");
    }
    if (dq_view.getDims() != index_q_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "dq");
    }
    if (dw_view.getDims() != weights_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "dw");
    }
    if (dk_view.getDims() != index_k_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "dk");
    }
    if (getRatio() <= 0) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: ratio must be positive", kOpName); }
}

std::vector<std::uint64_t> IndexerBwdDenseProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(256);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    key.push_back(static_cast<std::uint64_t>(getRatio()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&index_q_, &weights_, &index_k_, &attn_score_, &attn_l1norm_, &index_score_, &index_lse_,
                               &grad_index_score_, &dq_, &dw_, &dk_, &sm_scale_, &loss_coeff_, &grad_loss_, &ratio_}) {
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
    key.push_back(static_cast<std::uint64_t>(isInPlace()));
    return key;
}

}  // namespace ftrain
