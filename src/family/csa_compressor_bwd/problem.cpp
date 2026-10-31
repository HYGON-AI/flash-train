// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/csa_compressor_bwd/problem.hpp"

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

constexpr const char* kOpName = "CSACompressorBwd";

}  // namespace

CSACompressorBwdProblem::CSACompressorBwdProblem(Tensor kv, Tensor score, Tensor ape, Tensor cu_seqlens,
                                                 Tensor cu_seqlens_comp, Tensor grad_compressed, Tensor grad_kv,
                                                 Tensor grad_score, Tensor grad_ape, Tensor ratio, Tensor head_dim,
                                                 Tensor coff, CSACompressorBwdAttributes attributes)
    : kv_(std::move(kv)), score_(std::move(score)), ape_(std::move(ape)), cu_seqlens_(std::move(cu_seqlens)),
      cu_seqlens_comp_(std::move(cu_seqlens_comp)), grad_compressed_(std::move(grad_compressed)),
      grad_kv_(std::move(grad_kv)), grad_score_(std::move(grad_score)), grad_ape_(std::move(grad_ape)),
      ratio_(std::move(ratio)), head_dim_(std::move(head_dim)), coff_(std::move(coff)), attributes_(attributes) {
    const StorageView& ratio_view           = ratio_.getStorageView();
    const StorageView& head_dim_view        = head_dim_.getStorageView();
    const StorageView& coff_view            = coff_.getStorageView();
    const StorageView& kv_view              = kv_.getStorageView();
    const StorageView& score_view           = score_.getStorageView();
    const StorageView& ape_view             = ape_.getStorageView();
    const StorageView& cu_seqlens_view      = cu_seqlens_.getStorageView();
    const StorageView& cu_seqlens_comp_view = cu_seqlens_comp_.getStorageView();
    const StorageView& grad_compressed_view = grad_compressed_.getStorageView();
    const StorageView& grad_kv_view         = grad_kv_.getStorageView();
    const StorageView& grad_score_view      = grad_score_.getStorageView();
    const StorageView& grad_ape_view        = grad_ape_.getStorageView();

    if (!ratio_view.getDims().empty() || !head_dim_view.getDims().empty() || !coff_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!ratio_view.isHostMemory() || !head_dim_view.isHostMemory() || !coff_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port : {&kv_, &score_, &ape_, &cu_seqlens_, &cu_seqlens_comp_, &grad_compressed_, &grad_kv_,
                               &grad_score_, &grad_ape_, &ratio_, &head_dim_, &coff_}) {
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
    // head_dim participates in problem validation or implementation selection.
    if (head_dim_view.getNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: head_dim has the wrong scalar type", kOpName);
    }
    std::memcpy(&head_dim_value_, head_dim_view.getMemory(), sizeof(head_dim_value_));
    // coff participates in problem validation or implementation selection.
    if (coff_view.getNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: coff has the wrong scalar type", kOpName);
    }
    std::memcpy(&coff_value_, coff_view.getMemory(), sizeof(coff_value_));

    if (getRatio() <= 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s must be positive", kOpName, "ratio");
    }
    if (kv_view.isHostMemory() || score_view.isHostMemory() || ape_view.isHostMemory() ||
        cu_seqlens_view.isHostMemory() || cu_seqlens_comp_view.isHostMemory() || grad_compressed_view.isHostMemory() ||
        grad_kv_view.isHostMemory() || grad_score_view.isHostMemory() || grad_ape_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }

    if (kv_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "kv");
    }
    if (score_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "score");
    }
    if (ape_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "ape");
    }
    if (cu_seqlens_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "cu_seqlens");
    }
    if (cu_seqlens_comp_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "cu_seqlens_comp");
    }
    if (grad_compressed_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_compressed");
    }
    if (grad_kv_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_kv");
    }
    if (grad_score_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_score");
    }
    if (grad_ape_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_ape");
    }

    if (getHeadDim() <= 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: head_dim must be positive", kOpName);
    }
    if (getCoff() <= 0) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: coff must be positive", kOpName); }
    if (getHeadDim() > std::numeric_limits<std::int64_t>::max() / getCoff()) {
        throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: coff * head_dim overflows int64_t", kOpName);
    }
    const std::int64_t width = getCoff() * getHeadDim();
    if (kv_view.getDims()[1] != width) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "kv");
    }
    if (score_view.getDims() != kv_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "score");
    }
    if (grad_kv_view.getDims() != kv_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_kv");
    }
    if (grad_score_view.getDims() != kv_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_score");
    }
    if (ape_view.getDims()[0] != getRatio()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "ape");
    }
    if (ape_view.getDims()[1] != width) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "ape");
    }
    if (grad_ape_view.getDims() != ape_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_ape");
    }
    if (grad_compressed_view.getDims()[1] != getHeadDim()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                        "grad_compressed");
    }
    if (cu_seqlens_view.getDims()[0] != cu_seqlens_comp_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "cu_seqlens");
    }
    if (cu_seqlens_view.getDims()[0] < 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: cu_seqlens must hold at least two entries", kOpName);
    }
}

std::vector<std::uint64_t> CSACompressorBwdProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(224);
    key.push_back(static_cast<std::uint64_t>(getComputeType()));
    key.push_back(static_cast<std::uint64_t>(getRatio()));
    key.push_back(static_cast<std::uint64_t>(getHeadDim()));
    key.push_back(static_cast<std::uint64_t>(getCoff()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&kv_, &score_, &ape_, &cu_seqlens_, &cu_seqlens_comp_, &grad_compressed_, &grad_kv_,
                               &grad_score_, &grad_ape_, &ratio_, &head_dim_, &coff_}) {
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
