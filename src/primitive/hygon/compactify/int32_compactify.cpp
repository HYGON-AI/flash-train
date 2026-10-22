// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/primitive/hygon/compactify/int32_compactify.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const CompactifyProblem& problem) noexcept {
    for (const auto* dims : {&problem.getIndicesDims(), &problem.getOutputDims(), &problem.getTopkLengthDims()}) {
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

}  // namespace

const char* Int32Compactify::getName() const noexcept { return "Compactify"; }

std::unique_ptr<PrimitiveBase> Int32Compactify::clone() const { return std::make_unique<Int32Compactify>(*this); }

Result Int32Compactify::isApplicable(const CompactifyProblem& problem, const Constraints&) const {
    if (problem.getNumRows() == 0 || problem.getSelectionWidth() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getOutputNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getTopkLengthNumericType() != FTRAIN_NUMERIC_TYPE_INT32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutputIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkLengthIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutputAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkLengthAddress()) % alignof(std::int32_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getIndicesAddress() == nullptr || problem.getOutputAddress() == nullptr ||
        problem.getTopkLengthAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    // Scatter can overwrite indices still being read by another wave; the length
    // write can also clobber either data buffer. The original Torch binding owns
    // separate allocations, whereas C callers can supply overlapping subviews.
    if (problem.hasOverlappingBuffers()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compactify requires non-overlapping buffers");
    }
    return Result{};
}

std::uint64_t Int32Compactify::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Int32Compactify::configure(const CompactifyProblem& problem) {
    indices_         = static_cast<const int*>(problem.getIndicesAddress());
    output_          = static_cast<int*>(problem.getOutputAddress());
    topk_length_     = static_cast<int*>(problem.getTopkLengthAddress());
    num_rows_        = static_cast<int>(problem.getNumRows());
    selection_width_ = static_cast<int>(problem.getSelectionWidth());
}

void Int32Compactify::executeImpl(const Resources& resources) {
    launchInt32Compactify(resources.getStream(), indices_, output_, topk_length_, num_rows_, selection_width_);
}

}  // namespace ftrain
