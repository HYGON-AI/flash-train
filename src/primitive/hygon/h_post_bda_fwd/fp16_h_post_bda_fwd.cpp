// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/h_post_bda_fwd/fp16_h_post_bda_fwd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const HPostBdaFwdProblem& problem) noexcept {
    for (const auto* dims : {&problem.getHResDims(), &problem.getResidualDims(), &problem.getHPostDims(),
                             &problem.getXDims(), &problem.getBiasDims(), &problem.getOutputDims()}) {
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

bool supportsShape(const HPostBdaFwdProblem& problem) noexcept {
    return problem.getStreams() > 0 && problem.getStreams() <= 32;
}

}  // namespace

const char* Fp16HPostBdaFwd::getName() const noexcept { return "Fp16HPostBdaFwd"; }

std::unique_ptr<PrimitiveBase> Fp16HPostBdaFwd::clone() const { return std::make_unique<Fp16HPostBdaFwd>(*this); }

Result Fp16HPostBdaFwd::isApplicable(const HPostBdaFwdProblem& problem, const Constraints&) const {
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

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getHResNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getResidualNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getHPostNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getBiasNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getOutputNumericType() != FTRAIN_NUMERIC_TYPE_FP16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getHResIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getResidualIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHPostIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getBiasIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutputIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getHResAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getResidualAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPostAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getXAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutputAddress()) % alignof(std::uint16_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // nullptr omits bias; hasBias() disables kernel bias reads.
    if (problem.getBiasAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getBiasAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    if (problem.hasEmptyOutput()) { return Result{}; }
    if (problem.getHResAddress() == nullptr || problem.getResidualAddress() == nullptr ||
        problem.getHPostAddress() == nullptr || problem.getXAddress() == nullptr ||
        problem.getOutputAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

std::uint64_t Fp16HPostBdaFwd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Fp16HPostBdaFwd::configure(const HPostBdaFwdProblem& problem) {
    h_res_        = problem.getHResAddress();
    residual_     = problem.getResidualAddress();
    h_post_       = problem.getHPostAddress();
    x_            = problem.getXAddress();
    bias_         = static_cast<const float*>(problem.getBiasAddress());
    output_       = problem.getOutputAddress();
    batch_stride_ = static_cast<int>(problem.getBatchStride());
    num_streams_  = static_cast<int>(problem.getStreams());
    channels_     = static_cast<int>(problem.getChannels());
    has_bias_     = problem.hasBias();
}

void Fp16HPostBdaFwd::executeImpl(const Resources& resources) {
    if (batch_stride_ == 0 || channels_ == 0) { return; }
    const bool launched = launchFp16HPostBdaFwd(resources.getStream(), h_res_, residual_, h_post_, x_, bias_, output_,
                                                batch_stride_, num_streams_, channels_, has_bias_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "HPostBdaFwd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
