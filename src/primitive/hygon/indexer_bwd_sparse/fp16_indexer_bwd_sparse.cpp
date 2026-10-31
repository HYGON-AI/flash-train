// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_sparse/fp16_indexer_bwd_sparse.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const IndexerBwdSparseProblem& problem) noexcept {
    for (const auto* dims :
         {&problem.getIndexQDims(), &problem.getWeightsDims(), &problem.getIndexKDims(), &problem.getAttnScoreDims(),
          &problem.getIndexScoreDims(), &problem.getTopkIndicesDims(), &problem.getGradAttnScoreDims(),
          &problem.getDqDims(), &problem.getDwDims(), &problem.getDkDims(), &problem.getSmScaleDims(),
          &problem.getLossCoeffDims(), &problem.getGradLossDims(), &problem.getTopkIndicesGlobalDims()}) {
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

// One completion counter guards the dk cast.
constexpr std::uint64_t kCounterCount = 1;

}  // namespace

const char* Fp16IndexerBwdSparse::getName() const noexcept { return "Fp16IndexerBwdSparse"; }

std::unique_ptr<PrimitiveBase> Fp16IndexerBwdSparse::clone() const {
    return std::make_unique<Fp16IndexerBwdSparse>(*this);
}

Result Fp16IndexerBwdSparse::isApplicable(const IndexerBwdSparseProblem& problem, const Constraints&) const {
    if (problem.getBatch() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    // Validate type, layout and alignment even when the output aliases its input.
    if (problem.getGradAttnScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradAttnScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "in-place score gradient requires continuous FP32 storage");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getIndexQNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getWeightsNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getIndexKNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getAttnScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getIndexScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getDqNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getDwNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getDkNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getSmScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getLossCoeffNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradLossNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesGlobalNumericType() != FTRAIN_NUMERIC_TYPE_UINT8) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getIndexQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWeightsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getIndexKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAttnScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getIndexScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDqIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDwIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSmScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLossCoeffIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradLossIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesGlobalIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getIndexQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWeightsAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getIndexKAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnScoreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradAttnScoreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getIndexScoreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDqAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDwAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDkAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSmScaleAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLossCoeffAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradLossAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getIndexQAddress() == nullptr || problem.getWeightsAddress() == nullptr ||
        problem.getIndexKAddress() == nullptr || problem.getAttnScoreAddress() == nullptr ||
        problem.getIndexScoreAddress() == nullptr || problem.getTopkIndicesAddress() == nullptr ||
        problem.getDqAddress() == nullptr || problem.getDwAddress() == nullptr || problem.getDkAddress() == nullptr ||
        problem.getSmScaleAddress() == nullptr || problem.getLossCoeffAddress() == nullptr ||
        problem.getGradLossAddress() == nullptr || problem.getGradAttnScoreAddress() == nullptr ||
        problem.getTopkIndicesGlobalAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!problem.isGradLossHostMemory()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires grad_loss in host memory");
    }
    if (!problem.isInPlace()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "grad_attn_score must alias attn_score"); }
    if (problem.getKeyLength() == 0 || problem.getQueryLength() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > 128 || problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (problem.getHeads() == 0 || problem.getHeads() > 64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the head count is outside the kernel's tile coverage");
    }
    if (!fp16IndexerBwdSparseDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!fp16IndexerBwdSparseLaunchFits(static_cast<int>(problem.getHeads()), static_cast<int>(problem.getHeadDim()),
                                        static_cast<int>(problem.getSelectionWidth()),
                                        static_cast<int>(problem.getQueryLength()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t Fp16IndexerBwdSparse::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void Fp16IndexerBwdSparse::configure(const IndexerBwdSparseProblem& problem) {
    index_q_      = problem.getIndexQAddress();
    weights_      = problem.getWeightsAddress();
    index_k_      = problem.getIndexKAddress();
    index_score_  = problem.getIndexScoreAddress();
    topk_indices_ = static_cast<const int*>(problem.getTopkIndicesAddress());
    // The kernel reads the target and writes its gradient through one buffer,
    // which the Problem guarantees to be the same address.
    score_gradient_      = static_cast<float*>(problem.getGradAttnScoreAddress());
    dq_                  = problem.getDqAddress();
    dw_                  = problem.getDwAddress();
    dk_                  = problem.getDkAddress();
    batch_               = static_cast<int>(problem.getBatch());
    query_length_        = static_cast<int>(problem.getQueryLength());
    key_length_          = static_cast<int>(problem.getKeyLength());
    heads_               = static_cast<int>(problem.getHeads());
    head_dim_            = static_cast<int>(problem.getHeadDim());
    selection_width_     = static_cast<int>(problem.getSelectionWidth());
    sm_scale_            = *static_cast<const float*>(problem.getSmScaleAddress());
    loss_coeff_          = *static_cast<const float*>(problem.getLossCoeffAddress());
    grad_loss_           = *static_cast<const float*>(problem.getGradLossAddress());
    topk_indices_global_ = *static_cast<const std::uint8_t*>(problem.getTopkIndicesGlobalAddress());

    workspace_bytes_  = 0;
    dk_float_offset_  = workspace_bytes_;
    workspace_bytes_    += static_cast<std::uint64_t>(problem.getBatch()) * static_cast<std::uint64_t>(key_length_) *
                           static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    done_count_offset_  = workspace_bytes_;
    workspace_bytes_   += kCounterCount * sizeof(int);
}

void Fp16IndexerBwdSparse::executeImpl(const Resources& resources) {
    float* const dk_float =
        reinterpret_cast<float*>(static_cast<unsigned char*>(resources.getWorkspace()) + dk_float_offset_);
    int* const done_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + done_count_offset_);
    const std::uint64_t dk_float_bytes = static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(key_length_) *
                                         static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    fp16IndexerBwdSparseZeroWorkspace(resources.getStream(), dk_float, dk_float_bytes);
    fp16IndexerBwdSparseZeroWorkspace(resources.getStream(), done_count, kCounterCount * sizeof(int));

    const bool launched = launchFp16IndexerBwdSparse(
        resources.getStream(), index_q_, weights_, index_k_, score_gradient_, static_cast<const float*>(index_score_),
        topk_indices_, dq_, dw_, dk_, dk_float, done_count, batch_, query_length_, key_length_, heads_, head_dim_,
        selection_width_, sm_scale_, loss_coeff_, grad_loss_, topk_indices_global_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "IndexerBwdSparse kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
