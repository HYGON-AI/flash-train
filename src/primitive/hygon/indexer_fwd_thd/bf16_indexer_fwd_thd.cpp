// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/indexer_fwd_thd/bf16_indexer_fwd_thd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const IndexerFwdThdProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    if (problem.getMaxSeqlenKv() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims :
         {&problem.getQDims(), &problem.getKDims(), &problem.getWDims(), &problem.getCuSeqlensQDims(),
          &problem.getCuSeqlensKvDims(), &problem.getQCausalOffsetsDims(), &problem.getScoresDims(),
          &problem.getRatioDims(), &problem.getSmScaleDims(), &problem.getMaxSeqlenKvDims()}) {
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

const char* Bf16IndexerFwdThd::getName() const noexcept { return "Bf16IndexerFwdThd"; }

std::unique_ptr<PrimitiveBase> Bf16IndexerFwdThd::clone() const { return std::make_unique<Bf16IndexerFwdThd>(*this); }

Result Bf16IndexerFwdThd::isApplicable(const IndexerFwdThdProblem& problem, const Constraints&) const {
    if (problem.getQueryHeads() == 0) {
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
        problem.getWNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getCuSeqLensQNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getCuSeqLensKvNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getCausalOffsetsNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getScoresNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getSmScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getMaxSeqlenKvNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCausalOffsetsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getScoresIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSmScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getMaxSeqlenKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensQAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensKvAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCausalOffsetsAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getScoresAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSmScaleAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getMaxSeqlenKvAddress()) % alignof(std::int64_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getWAddress() == nullptr ||
        problem.getCuSeqLensQAddress() == nullptr || problem.getCuSeqLensKvAddress() == nullptr ||
        problem.getScoresAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (problem.getTotalK() == 0 || problem.getTotalQ() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty packed query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > kMaxHeadDim ||
        problem.getHeadDim() % kHeadDimAlignment != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (!bf16IndexerFwdThdDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!bf16IndexerFwdThdLaunchFits(static_cast<int>(problem.getQueryHeads()), static_cast<int>(problem.getHeadDim()),
                                     static_cast<int>(problem.getMaxSeqlenKv()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t Bf16IndexerFwdThd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Bf16IndexerFwdThd::configure(const IndexerFwdThdProblem& problem) {
    q_              = problem.getQAddress();
    k_              = problem.getKAddress();
    w_              = problem.getWAddress();
    cu_seqlens_q_   = static_cast<const int*>(problem.getCuSeqLensQAddress());
    cu_seqlens_kv_  = static_cast<const int*>(problem.getCuSeqLensKvAddress());
    causal_offsets_ = problem.getCausalOffsetsAddress() == nullptr
                          ? nullptr
                          : static_cast<const int*>(problem.getCausalOffsetsAddress());
    scores_         = static_cast<float*>(problem.getScoresAddress());
    total_q_        = static_cast<int>(problem.getTotalQ());
    total_k_        = static_cast<int>(problem.getTotalK());
    max_seqlen_kv_  = static_cast<int>(problem.getMaxSeqlenKv());
    query_heads_    = static_cast<int>(problem.getQueryHeads());
    head_dim_       = static_cast<int>(problem.getHeadDim());
    num_sequences_  = static_cast<int>(problem.getNumSequences());
    ratio_          = static_cast<int>(problem.getRatio());
    sm_scale_       = *static_cast<const float*>(problem.getSmScaleAddress());
}

void Bf16IndexerFwdThd::executeImpl(const Resources& resources) {
    const bool launched = launchBf16IndexerFwdThd(
        resources.getStream(), q_, k_, w_, scores_, cu_seqlens_q_, cu_seqlens_kv_, causal_offsets_, total_q_, total_k_,
        max_seqlen_kv_, query_heads_, /*kv_heads=*/1, head_dim_, num_sequences_, ratio_, sm_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "IndexerFwdThd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
