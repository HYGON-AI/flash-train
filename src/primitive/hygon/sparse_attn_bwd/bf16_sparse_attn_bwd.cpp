// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_attn_bwd/bf16_sparse_attn_bwd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const SparseAttnBwdProblem& problem) noexcept {
    for (const auto* dims : {&problem.getQDims(), &problem.getKvDims(), &problem.getOutDims(), &problem.getDoutDims(),
                             &problem.getLseDims(), &problem.getAttnSinkDims(), &problem.getTopkIndicesDims(),
                             &problem.getTopkLengthDims(), &problem.getDqDims(), &problem.getDkvDims(),
                             &problem.getDsinkDims(), &problem.getSoftmaxScaleDims()}) {
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

bool supportsShape(const SparseAttnBwdProblem& problem) noexcept {
    return problem.getValueDim() == problem.getHeadDim() || problem.getValueDim() == 512;
}

constexpr std::uint64_t kMaxHeadDim      = 576;
constexpr std::uint64_t kCompactValueDim = 512;
// Completion, locality rejection, valid-key count, and padded-head rejection.
constexpr std::uint64_t kCounterCount = 4;

}  // namespace

const char* Bf16SparseAttnBwd::getName() const noexcept { return "Bf16SparseAttnBwd"; }

std::unique_ptr<PrimitiveBase> Bf16SparseAttnBwd::clone() const { return std::make_unique<Bf16SparseAttnBwd>(*this); }

Result Bf16SparseAttnBwd::isApplicable(const SparseAttnBwdProblem& problem, const Constraints&) const {
    if (problem.getHeads() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires full or 512-wide value channels");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getKvNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getOutNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getDoutNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAttnSinkNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getTopkLengthNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getDqNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getDkvNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getDsinkNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getSoftmaxScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDoutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAttnSinkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkLengthIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDqIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDkvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDsinkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSoftmaxScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKvAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDoutAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnSinkAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkLengthAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDqAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDkvAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDsinkAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKvAddress() == nullptr || problem.getOutAddress() == nullptr ||
        problem.getDoutAddress() == nullptr || problem.getLseAddress() == nullptr ||
        problem.getAttnSinkAddress() == nullptr || problem.getTopkIndicesAddress() == nullptr ||
        problem.getDqAddress() == nullptr || problem.getDkvAddress() == nullptr ||
        problem.getDsinkAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (problem.getTokens() == 0) { return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty query has no kernel work"); }
    if (problem.getHeadDim() < kCompactValueDim || problem.getHeadDim() > kMaxHeadDim ||
        problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "Q/K head dimension is outside the kernel's staged width");
    }
    if (!bf16SparseAttnBwdSparseAttnFusedSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "the device does not expose the MMAC sparse-attention path this kernel needs");
    }
    if (!bf16SparseAttnBwdLaunchFits(static_cast<int>(problem.getHeadDim()),
                                     static_cast<int>(problem.getSelectionWidth()),
                                     static_cast<int>(problem.getHeads()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t Bf16SparseAttnBwd::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void Bf16SparseAttnBwd::configure(const SparseAttnBwdProblem& problem) {
    q_                = problem.getQAddress();
    kv_               = problem.getKvAddress();
    out_              = problem.getOutAddress();
    dout_             = problem.getDoutAddress();
    lse_              = static_cast<const float*>(problem.getLseAddress());
    attn_sink_        = static_cast<const float*>(problem.getAttnSinkAddress());
    topk_indices_     = static_cast<const int*>(problem.getTopkIndicesAddress());
    topk_length_      = static_cast<const int*>(problem.getTopkLengthAddress());
    dq_               = problem.getDqAddress();
    dkv_              = problem.getDkvAddress();
    dsink_            = static_cast<float*>(problem.getDsinkAddress());
    tokens_           = static_cast<int>(problem.getTokens());
    heads_            = static_cast<int>(problem.getHeads());
    head_dim_         = static_cast<int>(problem.getHeadDim());
    backward_width_   = static_cast<int>(problem.getValueDim());
    kv_tokens_        = static_cast<int>(problem.getKvTokens());
    selection_width_  = static_cast<int>(problem.getSelectionWidth());
    softmax_scale_    = *static_cast<const float*>(problem.getSoftmaxScaleAddress());
    full_workspace_   = !problem.usesLowMemoryWorkspace();
    cached_backward_  = problem.usesCachedBackward();
    use_packed_stats_ = (head_dim_ == 512 || head_dim_ == static_cast<int>(kMaxHeadDim)) &&
                        (backward_width_ == head_dim_ || bf16SparseAttnBwdSparseAttnBwdCompactValuesSupported(
                                                             heads_, selection_width_, cached_backward_));

    // Preserve the cache-line alignment of the original standalone packed-stat allocation.
    workspace_bytes_   = 0;
    dkv_float_offset_  = workspace_bytes_;
    workspace_bytes_  += static_cast<std::uint64_t>(kv_tokens_) * static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    done_count_offset_  = workspace_bytes_;
    workspace_bytes_   += kCounterCount * sizeof(int);
    if (use_packed_stats_) {
        workspace_bytes_              = (workspace_bytes_ + 255) & ~std::uint64_t{255};
        packed_stats_offset_          = workspace_bytes_;
        const bool inverse_workspace  = backward_width_ == 512 && head_dim_ == 576 && heads_ == 64 &&
                                        (tokens_ == 4096 || tokens_ == 4097) && kv_tokens_ == 4096 &&
                                        selection_width_ == 2048;
        const std::uint64_t rows      = inverse_workspace && !full_workspace_ ? (tokens_ + 1) / 2 : tokens_;
        const std::uint64_t columns   = static_cast<std::uint64_t>(selection_width_) * heads_ +
                                        (inverse_workspace ? kv_tokens_ + 36ULL * 4 * 64 * 4 : 0);
        workspace_bytes_             += rows * columns * sizeof(std::int32_t);
    }
}

void Bf16SparseAttnBwd::executeImpl(const Resources& resources) {
    float* const dkv_float =
        reinterpret_cast<float*>(static_cast<unsigned char*>(resources.getWorkspace()) + dkv_float_offset_);
    int* const done_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + done_count_offset_);
    // The upstream launchers initialize dkv_float themselves (fused with
    // classification for fused2); do not add a redundant memset here.
    bf16SparseAttnBwdZeroWorkspace(resources.getStream(), done_count, kCounterCount * sizeof(int));
    bf16SparseAttnBwdZeroWorkspace(resources.getStream(), dsink_, static_cast<std::uint64_t>(heads_) * sizeof(float));

    if (use_packed_stats_) {
        void* const packed_stats = (static_cast<unsigned char*>(resources.getWorkspace()) + packed_stats_offset_);
        const bool launched      = launchBf16SparseAttnBwdFused2(
            resources.getStream(), q_, kv_, out_, dout_, lse_, attn_sink_, topk_indices_, topk_length_, dq_, dkv_float,
            dsink_, dkv_, done_count, packed_stats, tokens_, heads_, head_dim_, backward_width_, kv_tokens_,
            selection_width_, softmax_scale_, full_workspace_, cached_backward_);
        if (launched) { return; }
    }

    const bool launched =
        launchBf16SparseAttnBwdFused(resources.getStream(), q_, kv_, out_, dout_, lse_, attn_sink_, topk_indices_,
                                     topk_length_, dq_, dkv_float, dsink_, dkv_, done_count, tokens_, heads_, head_dim_,
                                     backward_width_, kv_tokens_, selection_width_, softmax_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SparseAttnBwd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
