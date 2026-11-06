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

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation>
Fp32IndexerTopKLaunch bindFp32IndexerTopK(bool order) {
    constexpr int size     = Implementation::kSortSize;
    constexpr int capacity = Implementation::kCapacity;
    return order ? &launchFp32IndexerTopK<Implementation, size, capacity, true>
                 : &launchFp32IndexerTopK<Implementation, size, capacity, false>;
}

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

template<class Implementation>
const char* Fp32IndexerTopKBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Fp32IndexerTopKBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Fp32IndexerTopKBase<Implementation>::isApplicable(const IndexerTopKProblem& problem, const Constraints&) const {
    if (!((Implementation::kWideRadix && problem.getNumColumns() <= 256 * Implementation::kCapacity) ||
          (Implementation::kOrderedUInt && problem.getOrderValues() && problem.getNumColumns() == 4096 &&
           problem.getTopK() > 1024 && problem.getTopK() <= 2048) ||
          (Implementation::kGroupedAll && !problem.getOrderValues() && problem.getOutValuesAddress() == nullptr &&
           problem.getNumColumns() >= 1024 && problem.getTopK() >= static_cast<std::int64_t>(problem.getNumColumns()) &&
           problem.getTopK() % 4 == 0 && reinterpret_cast<std::uintptr_t>(problem.getOutIndicesAddress()) % 16 == 0) ||
          (Implementation::kGroupedRadix && !problem.getOrderValues() && problem.getNumColumns() == 4096 &&
           problem.getTopK() == 2048 && problem.getNextN() == 1) ||
          ((Implementation::kBitonic || Implementation::kRadix1024) &&
           problem.getNumColumns() <= Implementation::kSortSize &&
           (!Implementation::kRadix1024 || problem.getNumColumns() > 512)))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

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

template<class Implementation>
std::uint64_t Fp32IndexerTopKBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return 0;
}

template<class Implementation>
void Fp32IndexerTopKBase<Implementation>::configure(const IndexerTopKProblem& problem) {
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
    launch_        = bindFp32IndexerTopK<Implementation>(order_values_);
}

template<class Implementation>
void Fp32IndexerTopKBase<Implementation>::executeImpl(const Resources& resources) {
    const bool launched =
        launch_(resources.getStream(), input_values_, seq_lens_, out_indices_, order_values_ ? out_values_ : nullptr,
                num_rows_, num_columns_, top_k_, num_sequences_, next_n_, order_values_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "IndexerTopK kernel rejected a shape its applicability check accepted");
    }
}

template class Fp32IndexerTopKBase<Fp32IndexerTopKOrderedUInt>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKGroupedAll>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKGroupedRadix>;
template class Fp32IndexerTopKBase<Fp32IndexerTopK>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKWideRadix>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKRadix1024>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKSize4>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKSize8>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKSize16>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKSize32>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKSize64>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKSize128>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKSize256>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKSize512>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKWideRadixCapacity16>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKWideRadixCapacity32>;
template class Fp32IndexerTopKBase<Fp32IndexerTopKWideRadixCapacity64>;

}  // namespace ftrain
