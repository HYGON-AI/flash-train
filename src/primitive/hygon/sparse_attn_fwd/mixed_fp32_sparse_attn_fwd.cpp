// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_attn_fwd/mixed_fp32_sparse_attn_fwd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const SparseAttnFwdProblem& problem) noexcept {
    for (const auto* dims :
         {&problem.getQDims(), &problem.getKvDims(), &problem.getAttnSinkDims(), &problem.getTopkIndicesDims(),
          &problem.getTopkLengthDims(), &problem.getOutDims(), &problem.getLseDims(), &problem.getSoftmaxScaleDims()}) {
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

bool supportsValueWidth(const SparseAttnFwdProblem& problem) noexcept { return problem.getValueDim() == 512; }

// The fused forward stages Q/K head rows up to this width, and the value
// width is fixed by the kernel's d_v specialization.
constexpr std::uint64_t kMaxHeadDim = 576;
constexpr std::uint64_t kValueDim   = 512;

}  // namespace

const char* MixedFp32SparseAttnFwd::getName() const noexcept { return "MixedFp32SparseAttnFwd"; }

std::unique_ptr<PrimitiveBase> MixedFp32SparseAttnFwd::clone() const {
    return std::make_unique<MixedFp32SparseAttnFwd>(*this);
}

Result MixedFp32SparseAttnFwd::isApplicable(const SparseAttnFwdProblem& problem, const Constraints&) const {
    if (problem.getHeads() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getKvNumericType() == problem.getQNumericType()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires mixed input storage types");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if ((problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_FP32 &&
         problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_FP16 &&
         problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_BF16) ||
        (problem.getKvNumericType() != FTRAIN_NUMERIC_TYPE_FP32 &&
         problem.getKvNumericType() != FTRAIN_NUMERIC_TYPE_FP16 &&
         problem.getKvNumericType() != FTRAIN_NUMERIC_TYPE_BF16) ||
        problem.getAttnSinkNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getTopkLengthNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getOutNumericType() != problem.getQNumericType() ||
        problem.getLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getSoftmaxScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAttnSinkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkLengthIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSoftmaxScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) %
                (problem.getQNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKvAddress()) %
                (problem.getKvNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnSinkAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) %
                (problem.getOutNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // nullptr selects the full top-k extent; only supplied lengths need address validation.
    if (problem.getTopkLengthAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getTopkLengthAddress()) % alignof(std::int32_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKvAddress() == nullptr ||
        problem.getAttnSinkAddress() == nullptr || problem.getTopkIndicesAddress() == nullptr ||
        problem.getOutAddress() == nullptr || problem.getLseAddress() == nullptr ||
        problem.getSoftmaxScaleAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!problem.hasNonEmptyInputs()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the request has no kernel work"); }
    if (!problem.hasSingleKvHead()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the kernel supports one KV head"); }
    if (problem.getHeadDim() < kValueDim || problem.getHeadDim() > kMaxHeadDim || problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "Q/K head dimension is outside the kernel's staged width");
    }
    if (!supportsValueWidth(problem)) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the value width must be 512"); }
    if (!problem.hasNonEmptySelection()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "selection width must be positive");
    }
    if (!mixedFp32SparseAttnFwdSparseAttnFusedSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "the device does not expose the MMAC sparse-attention path this kernel needs");
    }
    if (!mixedFp32SparseAttnFwdLaunchFits(static_cast<int>(problem.getHeadDim()),
                                          static_cast<int>(problem.getSelectionWidth()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t MixedFp32SparseAttnFwd::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void MixedFp32SparseAttnFwd::configure(const SparseAttnFwdProblem& problem) {
    q_               = problem.getQAddress();
    kv_              = problem.getKvAddress();
    attn_sink_       = static_cast<const float*>(problem.getAttnSinkAddress());
    topk_indices_    = static_cast<const int*>(problem.getTopkIndicesAddress());
    topk_length_     = static_cast<const int*>(problem.getTopkLengthAddress());
    out_             = problem.getOutAddress();
    lse_             = static_cast<float*>(problem.getLseAddress());
    tokens_          = static_cast<int>(problem.getTokens());
    heads_           = static_cast<int>(problem.getHeads());
    head_dim_        = static_cast<int>(problem.getHeadDim());
    value_dim_       = static_cast<int>(problem.getValueDim());
    kv_tokens_       = static_cast<int>(problem.getKvTokens());
    selection_width_ = static_cast<int>(problem.getSelectionWidth());
    softmax_scale_   = *static_cast<const float*>(problem.getSoftmaxScaleAddress());

    // Conversion buffers follow this implementation's FP32 kernel scratch.
    workspace_bytes_ = 0;
    q_type_          = problem.getQNumericType();
    q_elements_      = 1;
    for (const auto dim : problem.getQDims()) { q_elements_ *= static_cast<std::uint64_t>(dim); }
    q_conversion_offset_ = workspace_bytes_;
    if (q_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += q_elements_ * sizeof(float); }
    kv_type_     = problem.getKvNumericType();
    kv_elements_ = 1;
    for (const auto dim : problem.getKvDims()) { kv_elements_ *= static_cast<std::uint64_t>(dim); }
    kv_conversion_offset_ = workspace_bytes_;
    if (kv_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += kv_elements_ * sizeof(float); }
    out_type_     = problem.getOutNumericType();
    out_elements_ = 1;
    for (const auto dim : problem.getOutDims()) { out_elements_ *= static_cast<std::uint64_t>(dim); }
    out_conversion_offset_ = workspace_bytes_;
    if (out_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += out_elements_ * sizeof(float); }
}

void MixedFp32SparseAttnFwd::executeImpl(const Resources& resources) {
    const void* q_compute = q_;
    if (q_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + q_conversion_offset_;
        castMixedFp32SparseAttnFwd(q_, converted, q_type_, FTRAIN_NUMERIC_TYPE_FP32, q_elements_,
                                   resources.getStream());
        q_compute = converted;
    }
    const void* kv_compute = kv_;
    if (kv_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + kv_conversion_offset_;
        castMixedFp32SparseAttnFwd(kv_, converted, kv_type_, FTRAIN_NUMERIC_TYPE_FP32, kv_elements_,
                                   resources.getStream());
        kv_compute = converted;
    }
    void* out_compute = out_;
    if (out_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + out_conversion_offset_;
        out_compute     = converted;
    }

    const bool launched = launchMixedFp32SparseAttnFwd(
        resources.getStream(), q_compute, kv_compute, attn_sink_, topk_indices_, out_compute, lse_, topk_length_,
        tokens_, heads_, head_dim_, value_dim_, kv_tokens_, selection_width_, softmax_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SparseAttnFwd kernel rejected a shape its applicability check accepted");
    }

    if (out_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        castMixedFp32SparseAttnFwd(out_compute, out_, FTRAIN_NUMERIC_TYPE_FP32, out_type_, out_elements_,
                                   resources.getStream());
    }
}

}  // namespace ftrain
