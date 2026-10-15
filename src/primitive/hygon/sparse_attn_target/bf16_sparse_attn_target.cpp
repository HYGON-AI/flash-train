// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_attn_target/bf16_sparse_attn_target.hpp"

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

const char* Bf16SparseAttnTarget::getName() const noexcept { return "Bf16SparseAttnTarget"; }

std::unique_ptr<PrimitiveBase> Bf16SparseAttnTarget::clone() const {
    return std::make_unique<Bf16SparseAttnTarget>(*this);
}

Result Bf16SparseAttnTarget::isApplicable(const SparseAttnTargetProblem& problem, const Constraints&) const {
    if (problem.getBatch() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getKNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
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
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTargetAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getLseAddress() == nullptr ||
        problem.getTopkIndicesAddress() == nullptr || problem.getTargetAddress() == nullptr) {
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
    if (!bf16SparseAttnTargetSparseMmacSupported() ||
        !bf16SparseAttnTargetSparseMmacShmFits(static_cast<int>(problem.getQueryHeads()),
                                               static_cast<int>(problem.getHeadDim()),
                                               static_cast<int>(problem.getSelectionWidth()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "the device or this shape cannot run the sparse MMAC gather path this kernel needs");
    }
    return Result{};
}

std::uint64_t Bf16SparseAttnTarget::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void Bf16SparseAttnTarget::configure(const SparseAttnTargetProblem& problem) {
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
}

void Bf16SparseAttnTarget::executeImpl(const Resources& resources) {
    int* const row_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + row_count_offset_);
    bf16SparseAttnTargetZeroWorkspace(
        resources.getStream(), row_count,
        static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(query_length_) * sizeof(int));
    const bool launched = launchBf16SparseAttnTarget(resources.getStream(), q_, k_, lse_, topk_indices_, target_,
                                                     row_count, batch_, query_length_, key_length_, query_heads_,
                                                     head_dim_, selection_width_, softmax_scale_, topk_indices_global_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SparseAttnTarget kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
