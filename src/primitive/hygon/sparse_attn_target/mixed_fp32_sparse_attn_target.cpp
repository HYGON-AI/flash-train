// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_attn_target/mixed_fp32_sparse_attn_target.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const SparseAttnTargetProblem& problem) noexcept {
    for (const auto* dims :
         {&problem.getQDims(), &problem.getKDims(), &problem.getLseDims(), &problem.getTopkIndicesDims(),
          &problem.getTargetDims(), &problem.getSoftmaxScaleDims(), &problem.getTopkIndicesGlobalDims()}) {
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

// The kernel's tile covers at most this many query heads and reads whole
// 16-element head fragments.
constexpr std::uint64_t kMaxQueryHeads    = 128;
constexpr std::uint64_t kHeadDimAlignment = 16;

}  // namespace

const char* MixedFp32SparseAttnTarget::getName() const noexcept { return "MixedFp32SparseAttnTarget"; }

std::unique_ptr<PrimitiveBase> MixedFp32SparseAttnTarget::clone() const {
    return std::make_unique<MixedFp32SparseAttnTarget>(*this);
}

Result MixedFp32SparseAttnTarget::isApplicable(const SparseAttnTargetProblem& problem, const Constraints&) const {
    if (problem.getBatch() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getKNumericType() == problem.getQNumericType()) {
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
        problem.getLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getTargetNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getSoftmaxScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesGlobalNumericType() != FTRAIN_NUMERIC_TYPE_UINT8) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTargetIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSoftmaxScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesGlobalIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) %
                (problem.getQNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) %
                (problem.getKNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTargetAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getLseAddress() == nullptr ||
        problem.getTopkIndicesAddress() == nullptr || problem.getTargetAddress() == nullptr ||
        problem.getSoftmaxScaleAddress() == nullptr || problem.getTopkIndicesGlobalAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (problem.getBatch() == 0 || problem.getQueryLength() == 0 || problem.getKeyLength() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty batch, query, or key length has no kernel work");
    }
    if (problem.getQueryHeads() == 0 || problem.getQueryHeads() > kMaxQueryHeads) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the query head count is outside the kernel's tile coverage");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() % kHeadDimAlignment != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (!mixedFp32SparseAttnTargetSparseMmacSupported() ||
        !mixedFp32SparseAttnTargetSparseMmacShmFits(static_cast<int>(problem.getQueryHeads()),
                                                    static_cast<int>(problem.getHeadDim()),
                                                    static_cast<int>(problem.getSelectionWidth()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "the device or this shape cannot run the sparse MMAC gather path this kernel needs");
    }
    return Result{};
}

std::uint64_t MixedFp32SparseAttnTarget::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void MixedFp32SparseAttnTarget::configure(const SparseAttnTargetProblem& problem) {
    q_                   = problem.getQAddress();
    k_                   = problem.getKAddress();
    lse_                 = static_cast<const float*>(problem.getLseAddress());
    topk_indices_        = static_cast<const int*>(problem.getTopkIndicesAddress());
    target_              = static_cast<float*>(problem.getTargetAddress());
    batch_               = static_cast<int>(problem.getBatch());
    query_length_        = static_cast<int>(problem.getQueryLength());
    key_length_          = static_cast<int>(problem.getKeyLength());
    query_heads_         = static_cast<int>(problem.getQueryHeads());
    head_dim_            = static_cast<int>(problem.getHeadDim());
    selection_width_     = static_cast<int>(problem.getSelectionWidth());
    softmax_scale_       = *static_cast<const float*>(problem.getSoftmaxScaleAddress());
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
}

void MixedFp32SparseAttnTarget::executeImpl(const Resources& resources) {
    const void* q_compute = q_;
    if (q_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + q_conversion_offset_;
        castMixedFp32SparseAttnTarget(q_, converted, q_type_, FTRAIN_NUMERIC_TYPE_FP32, q_elements_,
                                      resources.getStream());
        q_compute = converted;
    }
    const void* k_compute = k_;
    if (k_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + k_conversion_offset_;
        castMixedFp32SparseAttnTarget(k_, converted, k_type_, FTRAIN_NUMERIC_TYPE_FP32, k_elements_,
                                      resources.getStream());
        k_compute = converted;
    }

    int* const row_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + row_count_offset_);
    mixedFp32SparseAttnTargetZeroWorkspace(
        resources.getStream(), row_count,
        static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(query_length_) * sizeof(int));
    const bool launched = launchMixedFp32SparseAttnTarget(
        resources.getStream(), q_compute, k_compute, lse_, topk_indices_, target_, row_count, batch_, query_length_,
        key_length_, query_heads_, head_dim_, selection_width_, softmax_scale_, topk_indices_global_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SparseAttnTarget kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
