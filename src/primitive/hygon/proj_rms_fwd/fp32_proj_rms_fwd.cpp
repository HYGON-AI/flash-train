// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>

#include <hip/hip_runtime_api.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/proj_rms_fwd/fp32_proj_rms_fwd.hpp"

// Device capability and shape applicability belong to host-side configuration.
namespace ftrain {
namespace {
template<class Implementation>
static size_t proj_rms_fwd_workspace_bytes(int M, int N, int K, bool compute_h) {
    size_t bytes = 0;
    {
        const int n_tiles = (N + 15) / 16;
        const int split   = Implementation::kSplit;
        if (split > 1) {
            bytes = (size_t)split * M * (N + 1) * sizeof(float) + (size_t)((M + (64) - 1) / (64)) * sizeof(int);
        }
    }
    return bytes;
}
}  // namespace

template<class Implementation>
std::uint64_t fp32ProjRmsFwdWorkspaceBytes(int rows, int outputs, int reduction, bool compute_h) {
    return proj_rms_fwd_workspace_bytes<Implementation>(rows, outputs, reduction, compute_h);
}

namespace {

bool isGfx936Or938(FTrainDeviceId device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, static_cast<int>(device)) != hipSuccess) { return false; }
    return (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 || std::strncmp(props.gcnArchName, "gfx938", 6) == 0) &&
           (props.gcnArchName[6] == '\0' || props.gcnArchName[6] == ':');
}

bool hasInt32Indexing(const ProjRmsFwdProblem& problem) noexcept {
    for (const auto* dims : {&problem.getXDims(), &problem.getWeightDims(), &problem.getProjDims(), &problem.getRDims(),
                             &problem.getEpsDims()}) {
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

bool supportsShape(const ProjRmsFwdProblem& problem) noexcept {
    constexpr std::uint64_t fragment = 8;
    return problem.getOutputs() > 0 && problem.getOutputs() <= 32 && problem.getReduction() >= fragment &&
           problem.getReduction() % fragment == 0;
}

}  // namespace

template<class Implementation>
const char* Fp32ProjRmsFwdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Fp32ProjRmsFwdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Fp32ProjRmsFwdBase<Implementation>::isApplicable(const ProjRmsFwdProblem& problem,
                                                        const Constraints& constraints) const {
    // The MMAC staging uses vector memory accesses.
    if (((reinterpret_cast<std::uintptr_t>(problem.getXAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getWeightAddress())) &
         15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC algorithm requires 16-byte aligned vector operands");
    }
    if (!isGfx936Or938(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC primitive requires gfx936 or gfx938");
    }
    if (!(problem.getReduction() % Implementation::kSplit == 0 &&
          (problem.getReduction() / Implementation::kSplit) % ((problem.getComputeType() == FTRAIN_NUMERIC_TYPE_BF16 ||
                                                                problem.getComputeType() == FTRAIN_NUMERIC_TYPE_FP16)
                                                                   ? 16
                                                                   : 8) ==
              0)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

    if (problem.getRows() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty rows");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }
    // MMAC chunks round K up and form signed int indices within the padded tile.
    constexpr std::uint64_t chunk = 64;
    if (problem.getReduction() > std::numeric_limits<std::int32_t>::max() - (chunk - 1)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "padded reduction exceeds signed 32-bit indexing");
    }
    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds this kernel implementation's tile limits");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getWeightNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getProjNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getEpsNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWeightIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getProjIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getEpsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getXAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWeightAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getProjAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getEpsAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getXAddress() == nullptr || problem.getWeightAddress() == nullptr ||
        problem.getProjAddress() == nullptr || problem.getRAddress() == nullptr || problem.getEpsAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Fp32ProjRmsFwdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return workspace_bytes_;
}

template<class Implementation>
void Fp32ProjRmsFwdBase<Implementation>::configure(const ProjRmsFwdProblem& problem) {
    x_               = problem.getXAddress();
    weight_          = problem.getWeightAddress();
    proj_            = problem.getProjAddress();
    r_               = problem.getRAddress();
    rows_            = static_cast<int>(problem.getRows());
    outputs_         = static_cast<int>(problem.getOutputs());
    reduction_       = static_cast<int>(problem.getReduction());
    eps_             = problem.getEps();
    workspace_bytes_ = fp32ProjRmsFwdWorkspaceBytes<Implementation>(rows_, outputs_, reduction_, false);
}

template<class Implementation>
void Fp32ProjRmsFwdBase<Implementation>::executeImpl(const Resources& resources) {
    const bool launched = launchFp32ProjRmsFwd<Implementation>(resources.getStream(), x_, weight_, proj_, r_, rows_,
                                                               outputs_, reduction_, eps_, resources.getWorkspace());
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "ProjRmsForward kernel rejected a shape its applicability check accepted");
    }
}

template class Fp32ProjRmsFwdBase<Fp32ProjRmsFwd>;
template class Fp32ProjRmsFwdBase<Fp32ProjRmsFwdSplit2>;
template class Fp32ProjRmsFwdBase<Fp32ProjRmsFwdSplit4>;
template class Fp32ProjRmsFwdBase<Fp32ProjRmsFwdSplit8>;
}  // namespace ftrain
