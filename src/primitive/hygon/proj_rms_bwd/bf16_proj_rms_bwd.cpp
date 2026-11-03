// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include <cstring>
#include <hip/hip_runtime_api.h>
#include "flash_train/primitive/hygon/proj_rms_bwd/bf16_proj_rms_bwd.hpp"

namespace ftrain {
namespace {

bool isGfx936Or938(FTrainDeviceId device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, static_cast<int>(device)) != hipSuccess) { return false; }
    return (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 || std::strncmp(props.gcnArchName, "gfx938", 6) == 0) &&
           (props.gcnArchName[6] == '\0' || props.gcnArchName[6] == ':');
}

bool hasInt32Indexing(const ProjRmsBwdProblem& problem) noexcept {
    for (const auto* dims :
         {&problem.getGradProjDims(), &problem.getGradRDims(), &problem.getXDims(), &problem.getWeightDims(),
          &problem.getRDims(), &problem.getGradXDims(), &problem.getGradWeightDims(), &problem.getEpsDims()}) {
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

bool supportsShape(const ProjRmsBwdProblem& problem) noexcept {
    constexpr std::uint64_t fragment = 16;
    return problem.getOutputs() > 0 && problem.getOutputs() <= 32 && problem.getReduction() >= fragment &&
           problem.getReduction() % fragment == 0;
}

}  // namespace

template<class Implementation>
const char* Bf16ProjRmsBwdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Bf16ProjRmsBwdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Bf16ProjRmsBwdBase<Implementation>::isApplicable(const ProjRmsBwdProblem& problem,
                                                        const Constraints& constraints) const {
    // The MMAC staging and grad-X epilogue use vector memory accesses.
    if (!Implementation::kPortable && (((reinterpret_cast<std::uintptr_t>(problem.getXAddress()) |
                                         reinterpret_cast<std::uintptr_t>(problem.getWeightAddress()) |
                                         reinterpret_cast<std::uintptr_t>(problem.getGradXAddress()) |
                                         reinterpret_cast<std::uintptr_t>(problem.getGradProjAddress())) &
                                        15) != 0)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC algorithm requires 16-byte aligned vector operands");
    }
    if (Implementation::kMmac && problem.getGradProjNumericType() == problem.getComputeType() &&
        problem.getOutputs() % (problem.getGradProjNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? 4 : 8) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC grad-projection rows must preserve vector alignment");
    }
    if (!Implementation::kPortable && !isGfx936Or938(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC primitive requires gfx936 or gfx938");
    }
    if (!(!Implementation::kMmac || (problem.getOutputs() > 0 && problem.getOutputs() <= 32 &&
                                     problem.getReduction() % ((problem.getComputeType() == FTRAIN_NUMERIC_TYPE_BF16 ||
                                                                problem.getComputeType() == FTRAIN_NUMERIC_TYPE_FP16)
                                                                   ? 512
                                                                   : 256) ==
                                         0))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

    if (problem.getRows() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty rows");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }
    // The portable path rounds K up to a 16-column tile using signed int arithmetic.
    if (problem.getReduction() > std::numeric_limits<std::int32_t>::max() - 15) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "padded reduction exceeds signed 32-bit indexing");
    }
    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds this kernel implementation's tile limits");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getGradProjNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getGradRNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getWeightNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getRNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getGradXNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getGradWeightNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getEpsNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getGradProjIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradRIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWeightIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradWeightIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getEpsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getGradProjAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradRAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getXAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWeightAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradXAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradWeightAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getEpsAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getGradProjAddress() == nullptr || problem.getGradRAddress() == nullptr ||
        problem.getXAddress() == nullptr || problem.getWeightAddress() == nullptr || problem.getRAddress() == nullptr ||
        problem.getGradXAddress() == nullptr || problem.getGradWeightAddress() == nullptr ||
        problem.getEpsAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Bf16ProjRmsBwdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return 0;
}

template<class Implementation>
void Bf16ProjRmsBwdBase<Implementation>::configure(const ProjRmsBwdProblem& problem) {
    grad_proj_   = problem.getGradProjAddress();
    grad_r_      = problem.getGradRAddress();
    x_           = problem.getXAddress();
    weight_      = problem.getWeightAddress();
    r_           = problem.getRAddress();
    grad_x_      = problem.getGradXAddress();
    grad_weight_ = problem.getGradWeightAddress();
    rows_        = static_cast<int>(problem.getRows());
    outputs_     = static_cast<int>(problem.getOutputs());
    reduction_   = static_cast<int>(problem.getReduction());
    eps_         = problem.getEps();
}

template<class Implementation>
void Bf16ProjRmsBwdBase<Implementation>::executeImpl(const Resources& resources) {
    const bool launched =
        launchBf16ProjRmsBwd<Implementation>(resources.getStream(), grad_proj_, grad_r_, x_, weight_, r_, grad_x_,
                                             grad_weight_, rows_, outputs_, reduction_, eps_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "ProjRmsBackward kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain

namespace ftrain {
template class Bf16ProjRmsBwdBase<Bf16ProjRmsBwd>;
template class Bf16ProjRmsBwdBase<Bf16ProjRmsBwdMmac>;
}  // namespace ftrain
