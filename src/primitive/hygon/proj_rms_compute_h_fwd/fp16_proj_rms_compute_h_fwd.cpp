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
#include "flash_train/primitive/hygon/proj_rms_compute_h_fwd/fp16_proj_rms_compute_h_fwd.hpp"

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
std::uint64_t fp16ProjRmsComputeHFwdWorkspaceBytes(int rows, int outputs, int reduction, bool compute_h) {
    return proj_rms_fwd_workspace_bytes<Implementation>(rows, outputs, reduction, compute_h);
}

namespace {

bool isGfx936Or938(FTrainDeviceId device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, static_cast<int>(device)) != hipSuccess) { return false; }
    return (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 || std::strncmp(props.gcnArchName, "gfx938", 6) == 0) &&
           (props.gcnArchName[6] == '\0' || props.gcnArchName[6] == ':');
}

bool hasInt32Indexing(const ProjRmsComputeHFwdProblem& problem) noexcept {
    if (problem.getStreamCount() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims :
         {&problem.getXDims(), &problem.getWeightDims(), &problem.getAlphaPreDims(), &problem.getAlphaPostDims(),
          &problem.getAlphaResDims(), &problem.getBiasDims(), &problem.getHPreDims(), &problem.getHPostDims(),
          &problem.getHResDims(), &problem.getRDims(), &problem.getProjDims(), &problem.getNDims(),
          &problem.getEpsDims(), &problem.getComputeHEpsDims()}) {
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

bool supportsShape(const ProjRmsComputeHFwdProblem& problem) noexcept {
    constexpr std::uint64_t fragment = 16;
    return problem.getOutputs() > 0 && problem.getOutputs() <= 32 && problem.getReduction() >= fragment &&
           problem.getReduction() % fragment == 0;
}

}  // namespace

template<class Implementation>
const char* Fp16ProjRmsComputeHFwdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Fp16ProjRmsComputeHFwdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Fp16ProjRmsComputeHFwdBase<Implementation>::isApplicable(const ProjRmsComputeHFwdProblem& problem,
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
    if ((Implementation::kV2FullK || Implementation::kV2ConflictFreeFullK) && problem.getReduction() % 64 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "full K requires complete staging tiles");
    }
    if (!(problem.getReduction() % Implementation::kSplit == 0 &&
          (problem.getReduction() / Implementation::kSplit) % 8 == 0)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

    if (problem.getRows() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty rows");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }
    // MMAC chunks round K up and form signed int indices within the padded tile.
    constexpr std::uint64_t chunk = 128;
    if (problem.getReduction() > std::numeric_limits<std::int32_t>::max() - (chunk - 1)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "padded reduction exceeds signed 32-bit indexing");
    }
    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds this kernel implementation's tile limits");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getWeightNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getAlphaPreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAlphaPostNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAlphaResNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getBiasNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getHPreNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getHPostNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getHResNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getRNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getProjNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getNNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getEpsNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getComputeHEpsNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWeightIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAlphaPreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAlphaPostIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAlphaResIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getBiasIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHPreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHPostIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHResIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getProjIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getNIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getEpsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getComputeHEpsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getXAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWeightAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAlphaPreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAlphaPostAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAlphaResAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getBiasAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPreAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPostAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHResAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getProjAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getNAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getEpsAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getComputeHEpsAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getXAddress() == nullptr || problem.getWeightAddress() == nullptr ||
        problem.getAlphaPreAddress() == nullptr || problem.getAlphaPostAddress() == nullptr ||
        problem.getAlphaResAddress() == nullptr || problem.getBiasAddress() == nullptr ||
        problem.getHPreAddress() == nullptr || problem.getHPostAddress() == nullptr ||
        problem.getHResAddress() == nullptr || problem.getRAddress() == nullptr ||
        problem.getProjAddress() == nullptr || problem.getEpsAddress() == nullptr ||
        problem.getComputeHEpsAddress() == nullptr || problem.getNAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Fp16ProjRmsComputeHFwdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return workspace_bytes_;
}

template<class Implementation>
void Fp16ProjRmsComputeHFwdBase<Implementation>::configure(const ProjRmsComputeHFwdProblem& problem) {
    x_               = problem.getXAddress();
    weight_          = problem.getWeightAddress();
    alpha_pre_       = static_cast<const float*>(problem.getAlphaPreAddress());
    alpha_post_      = static_cast<const float*>(problem.getAlphaPostAddress());
    alpha_res_       = static_cast<const float*>(problem.getAlphaResAddress());
    bias_            = static_cast<const float*>(problem.getBiasAddress());
    h_pre_           = problem.getHPreAddress();
    h_post_          = problem.getHPostAddress();
    h_res_           = problem.getHResAddress();
    r_               = problem.getRAddress();
    proj_            = problem.getProjAddress();
    rows_            = static_cast<int>(problem.getRows());
    outputs_         = static_cast<int>(problem.getOutputs());
    num_streams_     = static_cast<int>(problem.getStreamCount());
    reduction_       = static_cast<int>(problem.getReduction());
    eps_             = problem.getEps();
    compute_h_eps_   = problem.getComputeHEps();
    workspace_bytes_ = fp16ProjRmsComputeHFwdWorkspaceBytes<Implementation>(rows_, outputs_, reduction_, true);
}

template<class Implementation>
void Fp16ProjRmsComputeHFwdBase<Implementation>::executeImpl(const Resources& resources) {
    const bool launched = launchFp16ProjRmsComputeHFwd<Implementation>(
        resources.getStream(), x_, weight_, proj_, r_, alpha_pre_, alpha_post_, alpha_res_, bias_, h_pre_, h_post_,
        h_res_, rows_, outputs_, num_streams_, reduction_, eps_, compute_h_eps_, resources.getWorkspace());
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "ProjRmsComputeHForward kernel rejected a shape its applicability check accepted");
    }
}

template class Fp16ProjRmsComputeHFwdBase<Fp16ProjRmsComputeHFwdV2FullK>;
template class Fp16ProjRmsComputeHFwdBase<Fp16ProjRmsComputeHFwdV2ConflictFreeFullK>;
template class Fp16ProjRmsComputeHFwdBase<Fp16ProjRmsComputeHFwd>;
template class Fp16ProjRmsComputeHFwdBase<Fp16ProjRmsComputeHFwdSplit2>;
template class Fp16ProjRmsComputeHFwdBase<Fp16ProjRmsComputeHFwdSplit4>;
template class Fp16ProjRmsComputeHFwdBase<Fp16ProjRmsComputeHFwdSplit8>;
template class Fp16ProjRmsComputeHFwdBase<Fp16ProjRmsComputeHFwdV2>;
template class Fp16ProjRmsComputeHFwdBase<Fp16ProjRmsComputeHFwdV2ConflictFree>;
}  // namespace ftrain
