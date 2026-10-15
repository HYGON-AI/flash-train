// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/indexer_fwd/problem.hpp"

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

constexpr const char* kOpName = "IndexerFwd";

void validate(const Tensor& q, const Tensor& k, const Tensor& w, const Tensor& scores) {
    const Tensor* const ports[4] = {&q, &k, &w, &scores};
    const char* const names[4]   = {"q", "k", "w", "scores"};
    for (std::size_t index = 0; index < 4; ++index) {
        if ((*ports[index]).getStorageView().isHostMemory()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s must be in device memory", kOpName, names[index]);
        }
    }

    if ((q).getStorageView().getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "q");
    }
    if ((k).getStorageView().getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "k");
    }
    if ((w).getStorageView().getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "w");
    }
    if ((scores).getStorageView().getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "scores");
    }

    // One KV head: the kernel's score tile is MQA (Hkv == 1).
    if ((k).getStorageView().getDims().size() != 4 || (k).getStorageView().getDims()[2] != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "k");
    }
    if ((k).getStorageView().getDims()[0] != (q).getStorageView().getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "k");
    }
    if ((k).getStorageView().getDims()[3] != (q).getStorageView().getDims()[3]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "k");
    }
    if ((w).getStorageView().getDims()[0] != (q).getStorageView().getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "w");
    }
    if ((w).getStorageView().getDims()[1] != (q).getStorageView().getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "w");
    }
    if ((w).getStorageView().getDims()[2] != (q).getStorageView().getDims()[2]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "w");
    }
    if ((scores).getStorageView().getDims()[0] != (q).getStorageView().getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "scores");
    }
    if ((scores).getStorageView().getDims()[1] != (q).getStorageView().getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "scores");
    }
    if ((scores).getStorageView().getDims()[2] != (k).getStorageView().getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "scores");
    }
}

}  // namespace

IndexerFwdProblem::IndexerFwdProblem(Tensor q, Tensor k, Tensor w, Tensor scores, Tensor ratio, Tensor sm_scale,
                                     IndexerFwdAttributes attributes)
    : q_(std::move(q)), k_(std::move(k)), w_(std::move(w)), scores_(std::move(scores)), ratio_(std::move(ratio)),
      sm_scale_(std::move(sm_scale)), attributes_(attributes) {
    const StorageView& ratio_view    = ratio_.getStorageView();
    const StorageView& sm_scale_view = sm_scale_.getStorageView();
    const StorageView& q_view        = q_.getStorageView();
    const StorageView& k_view        = k_.getStorageView();
    const StorageView& w_view        = w_.getStorageView();
    const StorageView& scores_view   = scores_.getStorageView();

    if (!ratio_view.getDims().empty() || !sm_scale_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!ratio_view.isHostMemory() || !sm_scale_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port : {&q_, &k_, &w_, &scores_, &ratio_, &sm_scale_}) {
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
    validate(q_, k_, w_, scores_);
}

std::uint64_t IndexerFwdProblem::getBatch() const noexcept {
    return static_cast<std::uint64_t>(q_.getStorageView().getDims()[0]);
}

std::uint64_t IndexerFwdProblem::getQueryLength() const noexcept {
    return static_cast<std::uint64_t>(q_.getStorageView().getDims()[1]);
}

std::uint64_t IndexerFwdProblem::getKeyLength() const noexcept {
    return static_cast<std::uint64_t>(k_.getStorageView().getDims()[1]);
}

std::uint64_t IndexerFwdProblem::getQueryHeads() const noexcept {
    return static_cast<std::uint64_t>(q_.getStorageView().getDims()[2]);
}

std::uint64_t IndexerFwdProblem::getHeadDim() const noexcept {
    return static_cast<std::uint64_t>(q_.getStorageView().getDims()[3]);
}

FTrainNumericType IndexerFwdProblem::getQNumericType() const noexcept { return q_.getStorageView().getNumericType(); }

FTrainNumericType IndexerFwdProblem::getKNumericType() const noexcept { return k_.getStorageView().getNumericType(); }

FTrainNumericType IndexerFwdProblem::getWNumericType() const noexcept { return w_.getStorageView().getNumericType(); }

FTrainNumericType IndexerFwdProblem::getScoresNumericType() const noexcept {
    return scores_.getStorageView().getNumericType();
}

FTrainIndexType IndexerFwdProblem::getQIndexType() const noexcept { return q_.getStorageView().getIndexType(); }

FTrainIndexType IndexerFwdProblem::getKIndexType() const noexcept { return k_.getStorageView().getIndexType(); }

FTrainIndexType IndexerFwdProblem::getWIndexType() const noexcept { return w_.getStorageView().getIndexType(); }

FTrainIndexType IndexerFwdProblem::getScoresIndexType() const noexcept {
    return scores_.getStorageView().getIndexType();
}

void* IndexerFwdProblem::getQAddress() const noexcept { return q_.getStorageView().getMemory(); }

void* IndexerFwdProblem::getKAddress() const noexcept { return k_.getStorageView().getMemory(); }

void* IndexerFwdProblem::getWAddress() const noexcept { return w_.getStorageView().getMemory(); }

void* IndexerFwdProblem::getScoresAddress() const noexcept { return scores_.getStorageView().getMemory(); }

std::vector<std::uint64_t> IndexerFwdProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(64);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    key.push_back(static_cast<std::uint64_t>(getRatio()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&q_, &k_, &w_, &scores_, &ratio_, &sm_scale_}) {
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
