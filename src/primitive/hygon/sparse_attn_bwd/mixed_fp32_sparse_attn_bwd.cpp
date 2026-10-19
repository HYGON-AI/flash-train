// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_attn_bwd/mixed_fp32_sparse_attn_bwd.hpp"

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
// The completion counter plus the two internal dispatch counters the large-D
// variant reads and writes.
constexpr std::uint64_t kCounterCount = 3;

}  // namespace

const char* MixedFp32SparseAttnBwd::getName() const noexcept { return "MixedFp32SparseAttnBwd"; }

std::unique_ptr<PrimitiveBase> MixedFp32SparseAttnBwd::clone() const {
    return std::make_unique<MixedFp32SparseAttnBwd>(*this);
}

Result MixedFp32SparseAttnBwd::isApplicable(const SparseAttnBwdProblem& problem, const Constraints&) const {
    if (problem.getHeads() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires full or 512-wide value channels");
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
        problem.getOutNumericType() != problem.getQNumericType() ||
        problem.getDoutNumericType() != problem.getQNumericType() ||
        problem.getLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAttnSinkNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getTopkLengthNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getDqNumericType() != problem.getQNumericType() ||
        problem.getDkvNumericType() != problem.getKvNumericType() ||
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
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) %
                (problem.getQNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKvAddress()) %
                (problem.getKvNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) %
                (problem.getOutNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDoutAddress()) %
                (problem.getDoutNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnSinkAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDqAddress()) %
                (problem.getDqNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDkvAddress()) %
                (problem.getDkvNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDsinkAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // nullptr selects the full top-k extent; only supplied lengths need address validation.
    if (problem.getTopkLengthAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getTopkLengthAddress()) % alignof(std::int32_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKvAddress() == nullptr || problem.getOutAddress() == nullptr ||
        problem.getDoutAddress() == nullptr || problem.getLseAddress() == nullptr ||
        problem.getAttnSinkAddress() == nullptr || problem.getTopkIndicesAddress() == nullptr ||
        problem.getDqAddress() == nullptr || problem.getDkvAddress() == nullptr ||
        problem.getDsinkAddress() == nullptr || problem.getSoftmaxScaleAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (problem.getTokens() == 0) { return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty query has no kernel work"); }
    if (problem.getHeadDim() < kCompactValueDim || problem.getHeadDim() > kMaxHeadDim ||
        problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "Q/K head dimension is outside the kernel's staged width");
    }
    if (!mixedFp32SparseAttnBwdSparseAttnFusedSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "the device does not expose the MMAC sparse-attention path this kernel needs");
    }
    if (!mixedFp32SparseAttnBwdLaunchFits(static_cast<int>(problem.getHeadDim()),
                                          static_cast<int>(problem.getSelectionWidth()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t MixedFp32SparseAttnBwd::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void MixedFp32SparseAttnBwd::configure(const SparseAttnBwdProblem& problem) {
    q_               = problem.getQAddress();
    kv_              = problem.getKvAddress();
    out_             = problem.getOutAddress();
    dout_            = problem.getDoutAddress();
    lse_             = static_cast<const float*>(problem.getLseAddress());
    attn_sink_       = static_cast<const float*>(problem.getAttnSinkAddress());
    topk_indices_    = static_cast<const int*>(problem.getTopkIndicesAddress());
    topk_length_     = static_cast<const int*>(problem.getTopkLengthAddress());
    dq_              = problem.getDqAddress();
    dkv_             = problem.getDkvAddress();
    dsink_           = static_cast<float*>(problem.getDsinkAddress());
    tokens_          = static_cast<int>(problem.getTokens());
    heads_           = static_cast<int>(problem.getHeads());
    head_dim_        = static_cast<int>(problem.getHeadDim());
    backward_width_  = static_cast<int>(problem.getValueDim());
    kv_tokens_       = static_cast<int>(problem.getKvTokens());
    selection_width_ = static_cast<int>(problem.getSelectionWidth());
    softmax_scale_   = *static_cast<const float*>(problem.getSoftmaxScaleAddress());

    // Each scratch region contains four-byte elements, so adjacent regions
    // retain the required alignment without padding.
    workspace_bytes_   = 0;
    dkv_float_offset_  = workspace_bytes_;
    workspace_bytes_  += static_cast<std::uint64_t>(kv_tokens_) * static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    done_count_offset_  = workspace_bytes_;
    workspace_bytes_   += kCounterCount * sizeof(int);

    // Conversion buffers follow this implementation's FP32 kernel scratch.
    q_type_     = problem.getQNumericType();
    q_elements_ = 1;
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
    dout_type_     = problem.getDoutNumericType();
    dout_elements_ = 1;
    for (const auto dim : problem.getDoutDims()) { dout_elements_ *= static_cast<std::uint64_t>(dim); }
    dout_conversion_offset_ = workspace_bytes_;
    if (dout_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += dout_elements_ * sizeof(float); }
    dq_type_     = problem.getDqNumericType();
    dq_elements_ = 1;
    for (const auto dim : problem.getDqDims()) { dq_elements_ *= static_cast<std::uint64_t>(dim); }
    dq_conversion_offset_ = workspace_bytes_;
    if (dq_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += dq_elements_ * sizeof(float); }
    dkv_type_     = problem.getDkvNumericType();
    dkv_elements_ = 1;
    for (const auto dim : problem.getDkvDims()) { dkv_elements_ *= static_cast<std::uint64_t>(dim); }
    dkv_conversion_offset_ = workspace_bytes_;
    if (dkv_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += dkv_elements_ * sizeof(float); }
}

void MixedFp32SparseAttnBwd::executeImpl(const Resources& resources) {
    const void* q_compute = q_;
    if (q_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + q_conversion_offset_;
        castMixedFp32SparseAttnBwd(q_, converted, q_type_, FTRAIN_NUMERIC_TYPE_FP32, q_elements_,
                                   resources.getStream());
        q_compute = converted;
    }
    const void* kv_compute = kv_;
    if (kv_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + kv_conversion_offset_;
        castMixedFp32SparseAttnBwd(kv_, converted, kv_type_, FTRAIN_NUMERIC_TYPE_FP32, kv_elements_,
                                   resources.getStream());
        kv_compute = converted;
    }
    const void* out_compute = out_;
    if (out_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + out_conversion_offset_;
        castMixedFp32SparseAttnBwd(out_, converted, out_type_, FTRAIN_NUMERIC_TYPE_FP32, out_elements_,
                                   resources.getStream());
        out_compute = converted;
    }
    const void* dout_compute = dout_;
    if (dout_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + dout_conversion_offset_;
        castMixedFp32SparseAttnBwd(dout_, converted, dout_type_, FTRAIN_NUMERIC_TYPE_FP32, dout_elements_,
                                   resources.getStream());
        dout_compute = converted;
    }
    void* dq_compute = dq_;
    if (dq_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + dq_conversion_offset_;
        dq_compute      = converted;
    }
    void* dkv_compute = dkv_;
    if (dkv_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + dkv_conversion_offset_;
        dkv_compute     = converted;
    }

    float* const dkv_float =
        reinterpret_cast<float*>(static_cast<unsigned char*>(resources.getWorkspace()) + dkv_float_offset_);
    int* const done_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + done_count_offset_);
    // The upstream launchers initialize dkv_float themselves (fused with
    // classification for fused2); do not add a redundant memset here.
    mixedFp32SparseAttnBwdZeroWorkspace(resources.getStream(), done_count, kCounterCount * sizeof(int));
    mixedFp32SparseAttnBwdZeroWorkspace(resources.getStream(), dsink_,
                                        static_cast<std::uint64_t>(heads_) * sizeof(float));

    const bool launched = launchMixedFp32SparseAttnBwdFused(
        resources.getStream(), q_compute, kv_compute, out_compute, dout_compute, lse_, attn_sink_, topk_indices_,
        topk_length_, dq_compute, dkv_float, dsink_, dkv_compute, done_count, tokens_, heads_, head_dim_,
        backward_width_, kv_tokens_, selection_width_, softmax_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SparseAttnBwd kernel rejected a shape its applicability check accepted");
    }

    if (dq_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        castMixedFp32SparseAttnBwd(dq_compute, dq_, FTRAIN_NUMERIC_TYPE_FP32, dq_type_, dq_elements_,
                                   resources.getStream());
    }
    if (dkv_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        castMixedFp32SparseAttnBwd(dkv_compute, dkv_, FTRAIN_NUMERIC_TYPE_FP32, dkv_type_, dkv_elements_,
                                   resources.getStream());
    }
}

}  // namespace ftrain
