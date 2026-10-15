// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/h_aggregate_bwd/fp32_h_aggregate_bwd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const HAggregateBwdProblem& problem) noexcept {
    for (const auto* dims : {&problem.getGradAggregatedDims(), &problem.getXDims(), &problem.getHPreDims(),
                             &problem.getGradXDims(), &problem.getGradHPreDims()}) {
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

bool supportsShape(const HAggregateBwdProblem& problem) noexcept {
    return problem.getStreams() > 0 && problem.getStreams() <= 32;
}

}  // namespace

const char* Fp32HAggregateBwd::getName() const noexcept { return "Fp32HAggregateBwd"; }

std::unique_ptr<PrimitiveBase> Fp32HAggregateBwd::clone() const { return std::make_unique<Fp32HAggregateBwd>(*this); }

Result Fp32HAggregateBwd::isApplicable(const HAggregateBwdProblem& problem, const Constraints&) const {
    if (problem.getChannels() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this backward implementation requires nonzero channels");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }
    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds this kernel implementation's tile limits");
    }

    // the channel loop rounds up to a 64-element tile; the expression must fit int.
    if (problem.getChannels() > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) - 63) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "channel tile arithmetic exceeds signed 32-bit indexing");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getGradAggregatedNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getHPreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradXNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradHPreNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getGradAggregatedIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHPreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradHPreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getGradAggregatedAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getXAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradXAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradHPreAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.hasEmptyOutput()) { return Result{}; }
    if (problem.getGradAggregatedAddress() == nullptr || problem.getXAddress() == nullptr ||
        problem.getHPreAddress() == nullptr || problem.getGradXAddress() == nullptr ||
        problem.getGradHPreAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

std::uint64_t Fp32HAggregateBwd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Fp32HAggregateBwd::configure(const HAggregateBwdProblem& problem) {
    grad_aggregated_ = problem.getGradAggregatedAddress();
    x_               = problem.getXAddress();
    h_pre_           = static_cast<const float*>(problem.getHPreAddress());
    grad_x_          = problem.getGradXAddress();
    grad_h_pre_      = static_cast<float*>(problem.getGradHPreAddress());
    batch_stride_    = static_cast<int>(problem.getBatchStride());
    num_streams_     = static_cast<int>(problem.getStreams());
    channels_        = static_cast<int>(problem.getChannels());
}

void Fp32HAggregateBwd::executeImpl(const Resources& resources) {
    if (batch_stride_ == 0) { return; }
    const bool launched = launchFp32HAggregateBwd(resources.getStream(), grad_aggregated_, x_, h_pre_, grad_x_,
                                                  grad_h_pre_, batch_stride_, num_streams_, channels_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "HAggregateBwd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
