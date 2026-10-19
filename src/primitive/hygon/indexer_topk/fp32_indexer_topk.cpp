// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/indexer_topk/fp32_indexer_topk.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const IndexerTopKProblem& problem) noexcept {
    if (problem.getTopK() > std::numeric_limits<std::int32_t>::max()) { return false; }
    if (problem.getNextN() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims : {&problem.getInputValuesDims(), &problem.getSeqLensDims(), &problem.getOutIndicesDims(),
                             &problem.getOutValuesDims(), &problem.getTopKDims(), &problem.getNextNDims(),
                             &problem.getOrderValuesDims()}) {
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

// The resource-bound the parallel kernel was built for; there is no serial
// fallback upstream, so a larger shape is simply unsupported.
constexpr std::uint64_t kMaxColumns = 16384;
constexpr std::uint64_t kMaxTopK    = 4096;

}  // namespace

const char* Fp32IndexerTopK::getName() const noexcept { return "IndexerTopK"; }

std::unique_ptr<PrimitiveBase> Fp32IndexerTopK::clone() const { return std::make_unique<Fp32IndexerTopK>(*this); }

Result Fp32IndexerTopK::isApplicable(const IndexerTopKProblem& problem, const Constraints&) const {
    if (problem.getNumRows() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getInputValuesNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getSeqLensNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getOutIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getOutValuesNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopKNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getNextNNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getOrderValuesNumericType() != FTRAIN_NUMERIC_TYPE_UINT8) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getInputValuesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSeqLensIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutValuesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getNextNIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOrderValuesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getInputValuesAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSeqLensAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopKAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getNextNAddress()) % alignof(std::int64_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // nullptr omits output values when order_values is false; otherwise it is rejected below.
    if (problem.getOutValuesAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getOutValuesAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    if (problem.getInputValuesAddress() == nullptr || problem.getSeqLensAddress() == nullptr ||
        problem.getOutIndicesAddress() == nullptr || problem.getNextNAddress() == nullptr ||
        problem.getTopKAddress() == nullptr) {
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

std::uint64_t Fp32IndexerTopK::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Fp32IndexerTopK::configure(const IndexerTopKProblem& problem) {
    input_values_  = static_cast<const float*>(problem.getInputValuesAddress());
    seq_lens_      = static_cast<const int*>(problem.getSeqLensAddress());
    out_indices_   = static_cast<int*>(problem.getOutIndicesAddress());
    out_values_    = static_cast<float*>(problem.getOutValuesAddress());
    num_rows_      = static_cast<int>(problem.getNumRows());
    num_columns_   = static_cast<int>(problem.getNumColumns());
    top_k_         = static_cast<int>(problem.getTopK());
    num_sequences_ = static_cast<int>(problem.getNumSequences());
    next_n_        = static_cast<int>(problem.getNextN());
    order_values_  = problem.getOrderValues();
}

void Fp32IndexerTopK::executeImpl(const Resources& resources) {
    const bool launched = launchFp32IndexerTopKK(resources.getStream(), input_values_, seq_lens_, out_indices_,
                                                 order_values_ ? out_values_ : nullptr, num_rows_, num_columns_, top_k_,
                                                 num_sequences_, next_n_, order_values_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "IndexerTopK kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
