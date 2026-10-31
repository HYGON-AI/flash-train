// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/h_post_bda_bwd/fp32_h_post_bda_bwd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const HPostBdaBwdProblem& problem) noexcept {
    for (const auto* dims :
         {&problem.getGradOutputDims(), &problem.getResidualDims(), &problem.getXDims(), &problem.getBiasDims(),
          &problem.getHResDims(), &problem.getHPostDims(), &problem.getGradResidualDims(), &problem.getGradXDims(),
          &problem.getGradHResDims(), &problem.getGradHPostDims(), &problem.getGradBiasDims()}) {
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

bool supportsShape(const HPostBdaBwdProblem& problem) noexcept {
    return problem.getStreams() > 0 && problem.getStreams() <= 32;
}

}  // namespace

const char* Fp32HPostBdaBwd::getName() const noexcept { return "Fp32HPostBdaBwd"; }

std::unique_ptr<PrimitiveBase> Fp32HPostBdaBwd::clone() const { return std::make_unique<Fp32HPostBdaBwd>(*this); }

Result Fp32HPostBdaBwd::isApplicable(const HPostBdaBwdProblem& problem, const Constraints&) const {
    if (problem.getChannels() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this backward implementation requires nonzero channels");
    }

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
    const std::uint64_t channel_tiles = (problem.getChannels() + 255) / 256;
    // The launcher and fused kernel both use int for batch * channel_tiles + batch.
    if (problem.getBatchStride() > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / (channel_tiles + 1)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "combined backward grid exceeds signed 32-bit indexing");
    }
    // grad_bias remains nonempty for an empty batch; this implementation cannot skip that reduction.
    if (problem.getBatchStride() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this backward implementation requires a nonempty batch");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getGradOutputNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getResidualNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getBiasNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getHResNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getHPostNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradResidualNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradXNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradHResNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradHPostNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradBiasNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getGradOutputIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getResidualIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getBiasIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHResIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHPostIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradResidualIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradHResIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradHPostIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradBiasIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getGradOutputAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getResidualAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getXAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHResAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPostAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradResidualAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradXAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradHResAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradHPostAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // nullptr omits the separate bias-gradient output and its reduction.
    if (problem.getGradBiasAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getGradBiasAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    // nullptr omits bias; hasBias() disables kernel bias reads.
    if (problem.getBiasAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getBiasAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    if (problem.getGradOutputAddress() == nullptr || problem.getResidualAddress() == nullptr ||
        problem.getXAddress() == nullptr || problem.getHResAddress() == nullptr ||
        problem.getHPostAddress() == nullptr || problem.getGradResidualAddress() == nullptr ||
        problem.getGradXAddress() == nullptr || problem.getGradHResAddress() == nullptr ||
        problem.getGradHPostAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

std::uint64_t Fp32HPostBdaBwd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Fp32HPostBdaBwd::configure(const HPostBdaBwdProblem& problem) {
    grad_output_   = problem.getGradOutputAddress();
    residual_      = problem.getResidualAddress();
    x_             = problem.getXAddress();
    bias_          = static_cast<const float*>(problem.getBiasAddress());
    h_res_         = problem.getHResAddress();
    h_post_        = problem.getHPostAddress();
    grad_residual_ = problem.getGradResidualAddress();
    grad_x_        = problem.getGradXAddress();
    grad_h_res_    = problem.getGradHResAddress();
    grad_h_post_   = problem.getGradHPostAddress();
    grad_bias_     = static_cast<float*>(problem.getGradBiasAddress());
    batch_stride_  = static_cast<int>(problem.getBatchStride());
    num_streams_   = static_cast<int>(problem.getStreams());
    channels_      = static_cast<int>(problem.getChannels());
    has_bias_      = problem.hasBias();
    reduce_bias_   = problem.hasBiasGradient();
}

void Fp32HPostBdaBwd::executeImpl(const Resources& resources) {
    const bool launched =
        launchFp32HPostBdaBwd(resources.getStream(), grad_output_, residual_, x_, bias_, h_res_, h_post_,
                              grad_residual_, grad_x_, grad_h_res_, grad_h_post_, reduce_bias_ ? grad_bias_ : nullptr,
                              batch_stride_, num_streams_, channels_, has_bias_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "HPostBdaBwd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
