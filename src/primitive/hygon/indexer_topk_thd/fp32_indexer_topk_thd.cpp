// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/indexer_topk_thd/fp32_indexer_topk_thd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const IndexerTopKThdProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    if (problem.getTopK() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims :
         {&problem.getInputValuesDims(), &problem.getCuSeqlensQDims(), &problem.getCuSeqlensKvDims(),
          &problem.getQCausalOffsetsDims(), &problem.getOutIndicesDims(), &problem.getOutValuesDims(),
          &problem.getRatioDims(), &problem.getTopKDims(), &problem.getOrderValuesDims()}) {
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

constexpr std::uint64_t kMaxColumns = 16384;
constexpr std::uint64_t kMaxTopK    = 4096;

}  // namespace

const char* Fp32IndexerTopKThd::getName() const noexcept { return "IndexerTopKThd"; }

std::unique_ptr<PrimitiveBase> Fp32IndexerTopKThd::clone() const { return std::make_unique<Fp32IndexerTopKThd>(*this); }

Result Fp32IndexerTopKThd::isApplicable(const IndexerTopKThdProblem& problem, const Constraints&) const {
    if (problem.getNumRows() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getInputValuesNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getCuSeqLensQNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getCuSeqLensKvNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getCausalOffsetsNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getOutIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getOutValuesNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getTopKNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getOrderValuesNumericType() != FTRAIN_NUMERIC_TYPE_UINT8) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getInputValuesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCausalOffsetsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutValuesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOrderValuesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getInputValuesAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensQAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensKvAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCausalOffsetsAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutValuesAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopKAddress()) % alignof(std::int64_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getInputValuesAddress() == nullptr || problem.getCuSeqLensQAddress() == nullptr ||
        problem.getCuSeqLensKvAddress() == nullptr || problem.getOutIndicesAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (problem.getOrderValues() && problem.getOutValuesAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "order_values asks for out_values but its address is null");
    }
    if (problem.getNumColumns() > kMaxColumns) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the score matrix has more columns than the kernel covers");
    }
    if (static_cast<std::uint64_t>(problem.getTopK()) > kMaxTopK) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "top_k exceeds the kernel's selection bound");
    }
    return Result{};
}

std::uint64_t Fp32IndexerTopKThd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Fp32IndexerTopKThd::configure(const IndexerTopKThdProblem& problem) {
    input_values_   = static_cast<const float*>(problem.getInputValuesAddress());
    cu_seqlens_q_   = static_cast<const int*>(problem.getCuSeqLensQAddress());
    cu_seqlens_kv_  = static_cast<const int*>(problem.getCuSeqLensKvAddress());
    causal_offsets_ = problem.getCausalOffsetsAddress() == nullptr
                          ? nullptr
                          : static_cast<const int*>(problem.getCausalOffsetsAddress());
    out_indices_    = static_cast<int*>(problem.getOutIndicesAddress());
    out_values_     = static_cast<float*>(problem.getOutValuesAddress());
    num_rows_       = static_cast<int>(problem.getNumRows());
    num_columns_    = static_cast<int>(problem.getNumColumns());
    top_k_          = static_cast<int>(problem.getTopK());
    num_sequences_  = static_cast<int>(problem.getNumSequences());
    ratio_          = static_cast<int>(problem.getRatio());
    order_values_   = problem.getOrderValues();
}

void Fp32IndexerTopKThd::executeImpl(const Resources& resources) {
    const bool launched = launchFp32IndexerTopKThd(
        resources.getStream(), input_values_, cu_seqlens_q_, cu_seqlens_kv_, causal_offsets_, out_indices_,
        order_values_ ? out_values_ : nullptr, num_rows_, num_columns_, top_k_, num_sequences_, ratio_, order_values_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "IndexerTopKThd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
