// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/h_post_bda_fwd/fp32_h_post_bda_fwd.hpp"

namespace ftrain {
namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation>
Fp32HPostBdaFwdLaunch bindFp32HPostBdaFwd(int streams) {
    if constexpr (Implementation::kGeneric) {
        return &launchFp32HPostBdaFwd<Implementation, 0>;
    } else
        switch (streams) {
            case 2: return &launchFp32HPostBdaFwd<Implementation, 2>;
            case 4: return &launchFp32HPostBdaFwd<Implementation, 4>;
            case 8: return &launchFp32HPostBdaFwd<Implementation, 8>;
            case 16: return &launchFp32HPostBdaFwd<Implementation, 16>;
            default:
                throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured HPostBdaFwd streams are not instantiated");
        }
}

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

template<class Implementation>
const char* Fp32HPostBdaFwdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Fp32HPostBdaFwdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Fp32HPostBdaFwdBase<Implementation>::isApplicable(const HPostBdaFwdProblem& problem, const Constraints&) const {
    if constexpr (!Implementation::kGeneric) {
        if (problem.getStreams() != 2 && problem.getStreams() != 4 && problem.getStreams() != 8 &&
            problem.getStreams() != 16) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "specialized forward requires 2, 4, 8 or 16 streams");
        }
    }

    if (!(!Implementation::kVectorized ||
          (problem.getChannels() % 8 == 0 && (problem.getStreams() == 2 || problem.getStreams() == 4 ||
                                              problem.getStreams() == 8 || problem.getStreams() == 16)))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
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

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getHResNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getResidualNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getHPostNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getBiasNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getOutputNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
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
    if (reinterpret_cast<std::uintptr_t>(problem.getHResAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getResidualAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPostAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getXAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutputAddress()) % alignof(float) != 0) {
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

template<class Implementation>
std::uint64_t Fp32HPostBdaFwdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return 0;
}

template<class Implementation>
void Fp32HPostBdaFwdBase<Implementation>::configure(const HPostBdaFwdProblem& problem) {
    h_res_        = problem.getHResAddress();
    residual_     = problem.getResidualAddress();
    h_post_       = problem.getHPostAddress();
    x_            = problem.getXAddress();
    bias_         = static_cast<const float*>(problem.getBiasAddress());
    output_       = problem.getOutputAddress();
    batch_stride_ = static_cast<int>(problem.getBatchStride());
    num_streams_  = static_cast<int>(problem.getStreams());
    launch_       = bindFp32HPostBdaFwd<Implementation>(num_streams_);
    channels_     = static_cast<int>(problem.getChannels());
    has_bias_     = problem.hasBias();
}

template<class Implementation>
void Fp32HPostBdaFwdBase<Implementation>::executeImpl(const Resources& resources) {
    if (batch_stride_ == 0 || channels_ == 0) { return; }
    const bool launched = launch_(resources.getStream(), h_res_, residual_, h_post_, x_, bias_, output_, batch_stride_,
                                  num_streams_, channels_, has_bias_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "HPostBdaFwd kernel rejected a shape its applicability check accepted");
    }
}

template class Fp32HPostBdaFwdBase<Fp32HPostBdaFwd>;
template class Fp32HPostBdaFwdBase<Fp32HPostBdaFwdVectorized>;
template class Fp32HPostBdaFwdBase<Fp32HPostBdaFwdGeneric>;

}  // namespace ftrain
