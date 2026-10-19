// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_dense/mixed_fp32_indexer_bwd_dense.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const IndexerBwdDenseProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims : {&problem.getIndexQDims(), &problem.getWeightsDims(), &problem.getIndexKDims(),
                             &problem.getAttnScoreDims(), &problem.getAttnL1normDims(), &problem.getIndexScoreDims(),
                             &problem.getIndexLseDims(), &problem.getGradIndexScoreDims(), &problem.getDqDims(),
                             &problem.getDwDims(), &problem.getDkDims(), &problem.getSmScaleDims(),
                             &problem.getLossCoeffDims(), &problem.getGradLossDims(), &problem.getRatioDims()}) {
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

constexpr std::uint64_t kCounterCount = 1;

}  // namespace

const char* MixedFp32IndexerBwdDense::getName() const noexcept { return "MixedFp32IndexerBwdDense"; }

std::unique_ptr<PrimitiveBase> MixedFp32IndexerBwdDense::clone() const {
    return std::make_unique<MixedFp32IndexerBwdDense>(*this);
}

Result MixedFp32IndexerBwdDense::isApplicable(const IndexerBwdDenseProblem& problem, const Constraints&) const {
    if (problem.getBatch() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getWeightsNumericType() == problem.getIndexQNumericType() &&
        problem.getIndexKNumericType() == problem.getIndexQNumericType()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires mixed input storage types");
    }

    // An in-place output still has an independent view whose dtype/layout must agree.
    if (problem.getGradIndexScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradIndexScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "in-place score gradient requires continuous FP32 storage");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if ((problem.getIndexQNumericType() != FTRAIN_NUMERIC_TYPE_FP32 &&
         problem.getIndexQNumericType() != FTRAIN_NUMERIC_TYPE_FP16 &&
         problem.getIndexQNumericType() != FTRAIN_NUMERIC_TYPE_BF16) ||
        (problem.getWeightsNumericType() != FTRAIN_NUMERIC_TYPE_FP32 &&
         problem.getWeightsNumericType() != FTRAIN_NUMERIC_TYPE_FP16 &&
         problem.getWeightsNumericType() != FTRAIN_NUMERIC_TYPE_BF16) ||
        (problem.getIndexKNumericType() != FTRAIN_NUMERIC_TYPE_FP32 &&
         problem.getIndexKNumericType() != FTRAIN_NUMERIC_TYPE_FP16 &&
         problem.getIndexKNumericType() != FTRAIN_NUMERIC_TYPE_BF16) ||
        problem.getAttnScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAttnL1NormNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getIndexScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getIndexLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getDqNumericType() != problem.getIndexQNumericType() ||
        problem.getDwNumericType() != problem.getWeightsNumericType() ||
        problem.getDkNumericType() != problem.getIndexKNumericType() ||
        problem.getSmScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getLossCoeffNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradLossNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getIndexQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWeightsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getIndexKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAttnScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAttnL1NormIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getIndexScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getIndexLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDqIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDwIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSmScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLossCoeffIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradLossIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getIndexQAddress()) %
                (problem.getIndexQNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float)
                                                                            : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWeightsAddress()) %
                (problem.getWeightsNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float)
                                                                             : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getIndexKAddress()) %
                (problem.getIndexKNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float)
                                                                            : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnScoreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnL1NormAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getIndexScoreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getIndexLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDqAddress()) %
                (problem.getDqNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDwAddress()) %
                (problem.getDwNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDkAddress()) %
                (problem.getDkNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSmScaleAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLossCoeffAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradLossAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getIndexQAddress() == nullptr || problem.getWeightsAddress() == nullptr ||
        problem.getIndexKAddress() == nullptr || problem.getAttnScoreAddress() == nullptr ||
        problem.getAttnL1NormAddress() == nullptr || problem.getIndexScoreAddress() == nullptr ||
        problem.getIndexLseAddress() == nullptr || problem.getDqAddress() == nullptr ||
        problem.getDwAddress() == nullptr || problem.getDkAddress() == nullptr ||
        problem.getSmScaleAddress() == nullptr || problem.getRatioAddress() == nullptr ||
        problem.getLossCoeffAddress() == nullptr || problem.getGradLossAddress() == nullptr ||
        problem.getGradIndexScoreAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!problem.isInPlace()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "grad_index_score must alias index_score"); }
    if (problem.getKeyLength() == 0 || problem.getQueryLength() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > 128 || problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (problem.getHeads() == 0 || problem.getHeads() > 64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the head count is outside the kernel's tile coverage");
    }
    if (!mixedFp32IndexerBwdDenseDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!mixedFp32IndexerBwdDenseLaunchFits(
            static_cast<int>(problem.getHeads()), static_cast<int>(problem.getHeadDim()),
            static_cast<int>(problem.getKeyLength()), static_cast<int>(problem.getQueryLength()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t MixedFp32IndexerBwdDense::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void MixedFp32IndexerBwdDense::configure(const IndexerBwdDenseProblem& problem) {
    index_q_     = problem.getIndexQAddress();
    weights_     = problem.getWeightsAddress();
    index_k_     = problem.getIndexKAddress();
    attn_score_  = static_cast<const float*>(problem.getAttnScoreAddress());
    attn_l1norm_ = static_cast<const float*>(problem.getAttnL1NormAddress());
    index_lse_   = static_cast<const float*>(problem.getIndexLseAddress());
    // The kernel reads the predictions and writes their gradient through one
    // buffer, which the Problem guarantees to be the same address.
    score_gradient_ = static_cast<float*>(problem.getGradIndexScoreAddress());
    dq_             = problem.getDqAddress();
    dw_             = problem.getDwAddress();
    dk_             = problem.getDkAddress();
    batch_          = static_cast<int>(problem.getBatch());
    query_length_   = static_cast<int>(problem.getQueryLength());
    key_length_     = static_cast<int>(problem.getKeyLength());
    heads_          = static_cast<int>(problem.getHeads());
    head_dim_       = static_cast<int>(problem.getHeadDim());
    ratio_          = static_cast<int>(problem.getRatio());
    sm_scale_       = *static_cast<const float*>(problem.getSmScaleAddress());
    loss_coeff_     = *static_cast<const float*>(problem.getLossCoeffAddress());
    grad_loss_      = *static_cast<const float*>(problem.getGradLossAddress());

    workspace_bytes_    = 0;
    dk_float_offset_    = workspace_bytes_;
    workspace_bytes_   += static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(key_length_) *
                          static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    done_count_offset_  = workspace_bytes_;
    workspace_bytes_   += kCounterCount * sizeof(int);

    // Conversion buffers follow this implementation's FP32 kernel scratch.
    index_q_type_     = problem.getIndexQNumericType();
    index_q_elements_ = 1;
    for (const auto dim : problem.getIndexQDims()) { index_q_elements_ *= static_cast<std::uint64_t>(dim); }
    index_q_conversion_offset_ = workspace_bytes_;
    if (index_q_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += index_q_elements_ * sizeof(float); }
    weights_type_     = problem.getWeightsNumericType();
    weights_elements_ = 1;
    for (const auto dim : problem.getWeightsDims()) { weights_elements_ *= static_cast<std::uint64_t>(dim); }
    weights_conversion_offset_ = workspace_bytes_;
    if (weights_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += weights_elements_ * sizeof(float); }
    index_k_type_     = problem.getIndexKNumericType();
    index_k_elements_ = 1;
    for (const auto dim : problem.getIndexKDims()) { index_k_elements_ *= static_cast<std::uint64_t>(dim); }
    index_k_conversion_offset_ = workspace_bytes_;
    if (index_k_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += index_k_elements_ * sizeof(float); }
    dq_type_     = problem.getDqNumericType();
    dq_elements_ = 1;
    for (const auto dim : problem.getDqDims()) { dq_elements_ *= static_cast<std::uint64_t>(dim); }
    dq_conversion_offset_ = workspace_bytes_;
    if (dq_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += dq_elements_ * sizeof(float); }
    dw_type_     = problem.getDwNumericType();
    dw_elements_ = 1;
    for (const auto dim : problem.getDwDims()) { dw_elements_ *= static_cast<std::uint64_t>(dim); }
    dw_conversion_offset_ = workspace_bytes_;
    if (dw_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += dw_elements_ * sizeof(float); }
    dk_type_     = problem.getDkNumericType();
    dk_elements_ = 1;
    for (const auto dim : problem.getDkDims()) { dk_elements_ *= static_cast<std::uint64_t>(dim); }
    dk_conversion_offset_ = workspace_bytes_;
    if (dk_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += dk_elements_ * sizeof(float); }
}

void MixedFp32IndexerBwdDense::executeImpl(const Resources& resources) {
    const void* index_q_compute = index_q_;
    if (index_q_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + index_q_conversion_offset_;
        castMixedFp32IndexerBwdDense(index_q_, converted, index_q_type_, FTRAIN_NUMERIC_TYPE_FP32, index_q_elements_,
                                     resources.getStream());
        index_q_compute = converted;
    }
    const void* weights_compute = weights_;
    if (weights_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + weights_conversion_offset_;
        castMixedFp32IndexerBwdDense(weights_, converted, weights_type_, FTRAIN_NUMERIC_TYPE_FP32, weights_elements_,
                                     resources.getStream());
        weights_compute = converted;
    }
    const void* index_k_compute = index_k_;
    if (index_k_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + index_k_conversion_offset_;
        castMixedFp32IndexerBwdDense(index_k_, converted, index_k_type_, FTRAIN_NUMERIC_TYPE_FP32, index_k_elements_,
                                     resources.getStream());
        index_k_compute = converted;
    }
    void* dq_compute = dq_;
    if (dq_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + dq_conversion_offset_;
        dq_compute      = converted;
    }
    void* dw_compute = dw_;
    if (dw_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + dw_conversion_offset_;
        dw_compute      = converted;
    }
    void* dk_compute = dk_;
    if (dk_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + dk_conversion_offset_;
        dk_compute      = converted;
    }

    float* const dk_float =
        reinterpret_cast<float*>(static_cast<unsigned char*>(resources.getWorkspace()) + dk_float_offset_);
    int* const done_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + done_count_offset_);
    const std::uint64_t dk_float_bytes = static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(key_length_) *
                                         static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    mixedFp32IndexerBwdDenseZeroWorkspace(resources.getStream(), dk_float, dk_float_bytes);
    mixedFp32IndexerBwdDenseZeroWorkspace(resources.getStream(), done_count, kCounterCount * sizeof(int));

    const bool launched = launchMixedFp32IndexerBwdDense(
        resources.getStream(), index_q_compute, weights_compute, index_k_compute, attn_score_, attn_l1norm_,
        score_gradient_, index_lse_, dq_compute, dw_compute, dk_compute, dk_float, done_count, batch_, query_length_,
        key_length_, heads_, head_dim_, ratio_, sm_scale_, loss_coeff_, grad_loss_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "IndexerBwdDense kernel rejected a shape its applicability check accepted");
    }

    if (dq_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        castMixedFp32IndexerBwdDense(dq_compute, dq_, FTRAIN_NUMERIC_TYPE_FP32, dq_type_, dq_elements_,
                                     resources.getStream());
    }
    if (dw_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        castMixedFp32IndexerBwdDense(dw_compute, dw_, FTRAIN_NUMERIC_TYPE_FP32, dw_type_, dw_elements_,
                                     resources.getStream());
    }
    if (dk_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        castMixedFp32IndexerBwdDense(dk_compute, dk_, FTRAIN_NUMERIC_TYPE_FP32, dk_type_, dk_elements_,
                                     resources.getStream());
    }
}

}  // namespace ftrain
