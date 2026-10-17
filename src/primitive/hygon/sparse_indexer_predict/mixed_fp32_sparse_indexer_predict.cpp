// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_indexer_predict/mixed_fp32_sparse_indexer_predict.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const SparseIndexerPredictProblem& problem) noexcept {
    for (const auto* dims :
         {&problem.getQDims(), &problem.getKDims(), &problem.getWDims(), &problem.getTopkIndicesDims(),
          &problem.getPredictDims(), &problem.getSmScaleDims(), &problem.getTopkIndicesGlobalDims()}) {
        std::uint64_t count = 1;
        for (const std::int64_t dim : *dims) {
            if (dim > std::numeric_limits<std::int32_t>::max()) { return false; }
            if (dim != 0 && count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) /
                                        static_cast<std::uint64_t>(dim)) {
                return false;
            }
            count *= static_cast<std::uint64_t>(dim);
        }
    }
    return true;
}

// The head-dimension bound the indexer predict kernel's tile covers.
constexpr std::uint64_t kMaxHeadDim = 128;

}  // namespace

const char* MixedFp32SparseIndexerPredict::getName() const noexcept { return "MixedFp32SparseIndexerPredict"; }

std::unique_ptr<PrimitiveBase> MixedFp32SparseIndexerPredict::clone() const {
    return std::make_unique<MixedFp32SparseIndexerPredict>(*this);
}

Result MixedFp32SparseIndexerPredict::isApplicable(const SparseIndexerPredictProblem& problem,
                                                   const Constraints&) const {
    if (problem.getBatch() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getKNumericType() == problem.getQNumericType() &&
        problem.getWNumericType() == problem.getQNumericType()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires mixed input storage types");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if ((problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_FP32 &&
         problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_FP16 &&
         problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_BF16) ||
        (problem.getKNumericType() != FTRAIN_NUMERIC_TYPE_FP32 &&
         problem.getKNumericType() != FTRAIN_NUMERIC_TYPE_FP16 &&
         problem.getKNumericType() != FTRAIN_NUMERIC_TYPE_BF16) ||
        (problem.getWNumericType() != FTRAIN_NUMERIC_TYPE_FP32 &&
         problem.getWNumericType() != FTRAIN_NUMERIC_TYPE_FP16 &&
         problem.getWNumericType() != FTRAIN_NUMERIC_TYPE_BF16) ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getPredictNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getSmScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesGlobalNumericType() != FTRAIN_NUMERIC_TYPE_UINT8) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getPredictIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSmScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesGlobalIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) %
                (problem.getQNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) %
                (problem.getKNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWAddress()) %
                (problem.getWNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getPredictAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSmScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getWAddress() == nullptr ||
        problem.getTopkIndicesAddress() == nullptr || problem.getPredictAddress() == nullptr ||
        problem.getSmScaleAddress() == nullptr || problem.getTopkIndicesGlobalAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (problem.getKeyLength() == 0 || problem.getQueryLength() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > kMaxHeadDim || problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension exceeds the kernel's tile coverage");
    }
    if (problem.getQueryHeads() == 0 || problem.getQueryHeads() > 128) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "query heads must lie in 1..128");
    }
    if (!mixedFp32SparseIndexerPredictSparseMmacSupported() ||
        !mixedFp32SparseIndexerPredictSparseMmacShmFits(static_cast<int>(problem.getQueryHeads()),
                                                        static_cast<int>(problem.getHeadDim()),
                                                        static_cast<int>(problem.getSelectionWidth()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    return Result{};
}

std::uint64_t MixedFp32SparseIndexerPredict::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void MixedFp32SparseIndexerPredict::configure(const SparseIndexerPredictProblem& problem) {
    q_                   = problem.getQAddress();
    k_                   = problem.getKAddress();
    w_                   = problem.getWAddress();
    topk_indices_        = static_cast<const int*>(problem.getTopkIndicesAddress());
    predict_             = static_cast<float*>(problem.getPredictAddress());
    batch_               = static_cast<int>(problem.getBatch());
    query_length_        = static_cast<int>(problem.getQueryLength());
    key_length_          = static_cast<int>(problem.getKeyLength());
    query_heads_         = static_cast<int>(problem.getQueryHeads());
    head_dim_            = static_cast<int>(problem.getHeadDim());
    selection_width_     = static_cast<int>(problem.getSelectionWidth());
    sm_scale_            = *static_cast<const float*>(problem.getSmScaleAddress());
    topk_indices_global_ = *static_cast<const std::uint8_t*>(problem.getTopkIndicesGlobalAddress());
    workspace_bytes_     = 0;
    row_count_offset_    = workspace_bytes_;
    workspace_bytes_ += static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(query_length_) * sizeof(int);

    // Conversion buffers follow this implementation's FP32 kernel scratch.
    q_type_     = problem.getQNumericType();
    q_elements_ = 1;
    for (const auto dim : problem.getQDims()) { q_elements_ *= static_cast<std::uint64_t>(dim); }
    q_conversion_offset_ = workspace_bytes_;
    if (q_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += q_elements_ * sizeof(float); }
    k_type_     = problem.getKNumericType();
    k_elements_ = 1;
    for (const auto dim : problem.getKDims()) { k_elements_ *= static_cast<std::uint64_t>(dim); }
    k_conversion_offset_ = workspace_bytes_;
    if (k_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += k_elements_ * sizeof(float); }
    w_type_     = problem.getWNumericType();
    w_elements_ = 1;
    for (const auto dim : problem.getWDims()) { w_elements_ *= static_cast<std::uint64_t>(dim); }
    w_conversion_offset_ = workspace_bytes_;
    if (w_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += w_elements_ * sizeof(float); }
}

void MixedFp32SparseIndexerPredict::executeImpl(const Resources& resources) {
    const void* q_compute = q_;
    if (q_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + q_conversion_offset_;
        castMixedFp32SparseIndexerPredict(q_, converted, q_type_, FTRAIN_NUMERIC_TYPE_FP32, q_elements_,
                                          resources.getStream());
        q_compute = converted;
    }
    const void* k_compute = k_;
    if (k_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + k_conversion_offset_;
        castMixedFp32SparseIndexerPredict(k_, converted, k_type_, FTRAIN_NUMERIC_TYPE_FP32, k_elements_,
                                          resources.getStream());
        k_compute = converted;
    }
    const void* w_compute = w_;
    if (w_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + w_conversion_offset_;
        castMixedFp32SparseIndexerPredict(w_, converted, w_type_, FTRAIN_NUMERIC_TYPE_FP32, w_elements_,
                                          resources.getStream());
        w_compute = converted;
    }

    int* const row_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + row_count_offset_);
    mixedFp32SparseIndexerPredictZeroWorkspace(
        resources.getStream(), row_count,
        static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(query_length_) * sizeof(int));
    const bool launched = launchMixedFp32SparseIndexerPredict(
        resources.getStream(), q_compute, k_compute, w_compute, topk_indices_, predict_, row_count, batch_,
        query_length_, key_length_, query_heads_, head_dim_, selection_width_, sm_scale_, topk_indices_global_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SparseIndexerPredict kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
