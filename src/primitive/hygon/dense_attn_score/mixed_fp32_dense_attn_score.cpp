// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/dense_attn_score/mixed_fp32_dense_attn_score.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const DenseAttnScoreProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims : {&problem.getQDims(), &problem.getKDims(), &problem.getLseDims(), &problem.getOutDims(),
                             &problem.getDenomDims(), &problem.getRatioDims(), &problem.getSoftmaxScaleDims()}) {
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

const char* MixedFp32DenseAttnScore::getName() const noexcept { return "MixedFp32DenseAttnScore"; }

std::unique_ptr<PrimitiveBase> MixedFp32DenseAttnScore::clone() const {
    return std::make_unique<MixedFp32DenseAttnScore>(*this);
}

Result MixedFp32DenseAttnScore::isApplicable(const DenseAttnScoreProblem& problem, const Constraints&) const {
    if (problem.getBatch() == 0 || problem.getQueryHeads() == 0) {
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
        problem.getOutNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getDenomNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getSoftmaxScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDenomIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSoftmaxScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) %
                (problem.getQNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) %
                (problem.getKNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDenomAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // StorageView rejects null rank-zero scalars before Problem construction.
    // DenseAttnScoreProblem requires softmax_scale to be a host scalar, so it is non-null here.
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getLseAddress() == nullptr ||
        problem.getOutAddress() == nullptr || problem.getDenomAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!problem.hasSingleKvHead()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the kernel supports one KV head"); }
    if (problem.getKeyLength() == 0 || problem.getQueryLength() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > kMaxHeadDim ||
        problem.getHeadDim() % kHeadDimAlignment != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (!mixedFp32DenseAttnScoreDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!mixedFp32DenseAttnScoreLaunchFits(static_cast<int>(problem.getQueryHeads()),
                                           static_cast<int>(problem.getHeadDim()),
                                           static_cast<int>(problem.getKeyLength()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t MixedFp32DenseAttnScore::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void MixedFp32DenseAttnScore::configure(const DenseAttnScoreProblem& problem) {
    q_             = problem.getQAddress();
    k_             = problem.getKAddress();
    lse_           = static_cast<const float*>(problem.getLseAddress());
    out_           = static_cast<float*>(problem.getOutAddress());
    denom_         = static_cast<float*>(problem.getDenomAddress());
    batch_         = static_cast<int>(problem.getBatch());
    query_length_  = static_cast<int>(problem.getQueryLength());
    key_length_    = static_cast<int>(problem.getKeyLength());
    query_heads_   = static_cast<int>(problem.getQueryHeads());
    head_dim_      = static_cast<int>(problem.getHeadDim());
    ratio_         = static_cast<int>(problem.getRatio());
    softmax_scale_ = *static_cast<const float*>(problem.getSoftmaxScaleAddress());

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

void MixedFp32DenseAttnScore::executeImpl(const Resources& resources) {
    const void* q_compute = q_;
    if (q_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + q_conversion_offset_;
        castMixedFp32DenseAttnScore(q_, converted, q_type_, FTRAIN_NUMERIC_TYPE_FP32, q_elements_,
                                    resources.getStream());
        q_compute = converted;
    }
    const void* k_compute = k_;
    if (k_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + k_conversion_offset_;
        castMixedFp32DenseAttnScore(k_, converted, k_type_, FTRAIN_NUMERIC_TYPE_FP32, k_elements_,
                                    resources.getStream());
        k_compute = converted;
    }

    const bool launched = launchMixedFp32DenseAttnScore(resources.getStream(), q_compute, k_compute, lse_, out_, denom_,
                                                        batch_, query_length_, key_length_, query_heads_,
                                                        /*kv_heads=*/1, head_dim_, ratio_, softmax_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "DenseAttnScore kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
