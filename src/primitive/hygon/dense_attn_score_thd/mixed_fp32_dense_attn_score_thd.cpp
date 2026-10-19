// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/dense_attn_score_thd/mixed_fp32_dense_attn_score_thd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const DenseAttnScoreThdProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    if (problem.getMaxSeqlenKv() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims : {&problem.getQDims(), &problem.getKDims(), &problem.getLseDims(),
                             &problem.getCuSeqlensQDims(), &problem.getCuSeqlensKvDims(),
                             &problem.getQCausalOffsetsDims(), &problem.getOutDims(), &problem.getDenomDims(),
                             &problem.getRatioDims(), &problem.getSoftmaxScaleDims(), &problem.getMaxSeqlenKvDims()}) {
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

constexpr std::uint64_t kMaxHeadDim       = 576;
constexpr std::uint64_t kHeadDimAlignment = 16;

}  // namespace

const char* MixedFp32DenseAttnScoreThd::getName() const noexcept { return "MixedFp32DenseAttnScoreThd"; }

std::unique_ptr<PrimitiveBase> MixedFp32DenseAttnScoreThd::clone() const {
    return std::make_unique<MixedFp32DenseAttnScoreThd>(*this);
}

Result MixedFp32DenseAttnScoreThd::isApplicable(const DenseAttnScoreThdProblem& problem, const Constraints&) const {
    if (problem.getQueryHeads() == 0) {
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
        problem.getCuSeqLensQNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getCuSeqLensKvNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getCausalOffsetsNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getOutNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getDenomNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getSoftmaxScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getMaxSeqlenKvNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCausalOffsetsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDenomIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSoftmaxScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getMaxSeqlenKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) %
                (problem.getQNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) %
                (problem.getKNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensQAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensKvAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDenomAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getMaxSeqlenKvAddress()) % alignof(std::int64_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // nullptr selects zero causal offsets; only supplied offsets need address validation.
    if (problem.getCausalOffsetsAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getCausalOffsetsAddress()) % alignof(std::int32_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getLseAddress() == nullptr ||
        problem.getCuSeqLensQAddress() == nullptr || problem.getCuSeqLensKvAddress() == nullptr ||
        problem.getOutAddress() == nullptr || problem.getDenomAddress() == nullptr ||
        problem.getSoftmaxScaleAddress() == nullptr || problem.getRatioAddress() == nullptr ||
        problem.getMaxSeqlenKvAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!problem.hasSingleKvHead()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the kernel supports one KV head"); }
    if (problem.getTotalK() == 0 || problem.getTotalQ() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty packed query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > kMaxHeadDim ||
        problem.getHeadDim() % kHeadDimAlignment != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (!mixedFp32DenseAttnScoreThdDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!mixedFp32DenseAttnScoreThdLaunchFits(static_cast<int>(problem.getQueryHeads()),
                                              static_cast<int>(problem.getHeadDim()),
                                              static_cast<int>(problem.getMaxSeqlenKv()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t MixedFp32DenseAttnScoreThd::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void MixedFp32DenseAttnScoreThd::configure(const DenseAttnScoreThdProblem& problem) {
    q_              = problem.getQAddress();
    k_              = problem.getKAddress();
    lse_            = static_cast<const float*>(problem.getLseAddress());
    cu_seqlens_q_   = static_cast<const int*>(problem.getCuSeqLensQAddress());
    cu_seqlens_kv_  = static_cast<const int*>(problem.getCuSeqLensKvAddress());
    causal_offsets_ = problem.getCausalOffsetsAddress() == nullptr
                          ? nullptr
                          : static_cast<const int*>(problem.getCausalOffsetsAddress());
    out_            = static_cast<float*>(problem.getOutAddress());
    denom_          = static_cast<float*>(problem.getDenomAddress());
    total_q_        = static_cast<int>(problem.getTotalQ());
    total_k_        = static_cast<int>(problem.getTotalK());
    max_seqlen_kv_  = static_cast<int>(problem.getMaxSeqlenKv());
    query_heads_    = static_cast<int>(problem.getQueryHeads());
    head_dim_       = static_cast<int>(problem.getHeadDim());
    num_sequences_  = static_cast<int>(problem.getNumSequences());
    ratio_          = static_cast<int>(problem.getRatio());
    softmax_scale_  = *static_cast<const float*>(problem.getSoftmaxScaleAddress());

    // Conversion buffers follow this implementation's FP32 kernel scratch.
    workspace_bytes_ = 0;
    q_type_          = problem.getQNumericType();
    q_elements_      = 1;
    for (const auto dim : problem.getQDims()) { q_elements_ *= static_cast<std::uint64_t>(dim); }
    q_conversion_offset_ = workspace_bytes_;
    if (q_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += q_elements_ * sizeof(float); }
    k_type_     = problem.getKNumericType();
    k_elements_ = 1;
    for (const auto dim : problem.getKDims()) { k_elements_ *= static_cast<std::uint64_t>(dim); }
    k_conversion_offset_ = workspace_bytes_;
    if (k_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += k_elements_ * sizeof(float); }
}

void MixedFp32DenseAttnScoreThd::executeImpl(const Resources& resources) {
    const void* q_compute = q_;
    if (q_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + q_conversion_offset_;
        castMixedFp32DenseAttnScoreThd(q_, converted, q_type_, FTRAIN_NUMERIC_TYPE_FP32, q_elements_,
                                       resources.getStream());
        q_compute = converted;
    }
    const void* k_compute = k_;
    if (k_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + k_conversion_offset_;
        castMixedFp32DenseAttnScoreThd(k_, converted, k_type_, FTRAIN_NUMERIC_TYPE_FP32, k_elements_,
                                       resources.getStream());
        k_compute = converted;
    }

    const bool launched = launchMixedFp32DenseAttnScoreThd(
        resources.getStream(), q_compute, k_compute, lse_, out_, denom_, cu_seqlens_q_, cu_seqlens_kv_, causal_offsets_,
        total_q_, total_k_, max_seqlen_kv_, query_heads_, /*kv_heads=*/1, head_dim_, num_sequences_, ratio_,
        softmax_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "DenseAttnScoreThd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
