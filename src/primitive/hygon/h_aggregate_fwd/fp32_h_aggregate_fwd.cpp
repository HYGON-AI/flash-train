// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/h_aggregate_fwd/fp32_h_aggregate_fwd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const HAggregateFwdProblem& problem) noexcept {
    for (const auto* dims : {&problem.getXDims(), &problem.getHPreDims(), &problem.getAggregatedDims()}) {
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

bool supportsShape(const HAggregateFwdProblem& problem) noexcept {
    return problem.getStreams() > 0 && problem.getStreams() <= 32;
}

}  // namespace

const char* Fp32HAggregateFwd::getName() const noexcept { return "Fp32HAggregateFwd"; }

std::unique_ptr<PrimitiveBase> Fp32HAggregateFwd::clone() const { return std::make_unique<Fp32HAggregateFwd>(*this); }

Result Fp32HAggregateFwd::isApplicable(const HAggregateFwdProblem& problem, const Constraints&) const {
    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }
    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds this kernel implementation's tile limits");
    }

    // channel launch rounding adds 255 before division by 256; the expression must fit int.
    if (problem.getChannels() > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) - 255) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "channel tile arithmetic exceeds signed 32-bit indexing");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getHPreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAggregatedNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHPreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAggregatedIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getXAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAggregatedAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.hasEmptyOutput()) { return Result{}; }
    if (problem.getXAddress() == nullptr || problem.getHPreAddress() == nullptr ||
        problem.getAggregatedAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

std::uint64_t Fp32HAggregateFwd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Fp32HAggregateFwd::configure(const HAggregateFwdProblem& problem) {
    x_            = problem.getXAddress();
    h_pre_        = static_cast<const float*>(problem.getHPreAddress());
    aggregated_   = problem.getAggregatedAddress();
    batch_stride_ = static_cast<int>(problem.getBatchStride());
    num_streams_  = static_cast<int>(problem.getStreams());
    channels_     = static_cast<int>(problem.getChannels());
}

void Fp32HAggregateFwd::executeImpl(const Resources& resources) {
    if (batch_stride_ == 0 || channels_ == 0) { return; }
    const bool launched =
        launchFp32HAggregateFwd(resources.getStream(), x_, h_pre_, aggregated_, batch_stride_, num_streams_, channels_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "HAggregateFwd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
