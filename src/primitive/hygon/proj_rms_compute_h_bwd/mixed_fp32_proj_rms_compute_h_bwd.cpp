// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>

#include <hip/hip_runtime_api.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/proj_rms_compute_h_bwd/mixed_fp32_proj_rms_compute_h_bwd.hpp"

namespace ftrain {
namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, bool FP32_STATE>
MixedFp32ProjRmsComputeHBwdLaunch bindMixedFp32ProjRmsComputeHBwd(FTrainNumericType x_type,
                                                                  FTrainNumericType weight_type) {
    if (x_type == FTRAIN_NUMERIC_TYPE_BF16 && weight_type == FTRAIN_NUMERIC_TYPE_FP16)
        return &launchMixedFp32ProjRmsComputeHBwd<Implementation, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP16,
                                                  FP32_STATE>;
    if (x_type == FTRAIN_NUMERIC_TYPE_BF16 && weight_type == FTRAIN_NUMERIC_TYPE_FP32)
        return &launchMixedFp32ProjRmsComputeHBwd<Implementation, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP32,
                                                  FP32_STATE>;
    if (x_type == FTRAIN_NUMERIC_TYPE_FP16 && weight_type == FTRAIN_NUMERIC_TYPE_BF16)
        return &launchMixedFp32ProjRmsComputeHBwd<Implementation, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_BF16,
                                                  FP32_STATE>;
    if (x_type == FTRAIN_NUMERIC_TYPE_FP16 && weight_type == FTRAIN_NUMERIC_TYPE_FP32)
        return &launchMixedFp32ProjRmsComputeHBwd<Implementation, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP32,
                                                  FP32_STATE>;
    if (x_type == FTRAIN_NUMERIC_TYPE_FP32 && weight_type == FTRAIN_NUMERIC_TYPE_BF16)
        return &launchMixedFp32ProjRmsComputeHBwd<Implementation, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_BF16,
                                                  FP32_STATE>;
    if (x_type == FTRAIN_NUMERIC_TYPE_FP32 && weight_type == FTRAIN_NUMERIC_TYPE_FP16)
        return &launchMixedFp32ProjRmsComputeHBwd<Implementation, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_NUMERIC_TYPE_FP16,
                                                  FP32_STATE>;
    throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured mixed storage types are not instantiated");
}

bool isGfx936Or938(FTrainDeviceId device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, static_cast<int>(device)) != hipSuccess) { return false; }
    return (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 || std::strncmp(props.gcnArchName, "gfx938", 6) == 0) &&
           (props.gcnArchName[6] == '\0' || props.gcnArchName[6] == ':');
}

bool hasInt32Indexing(const ProjRmsComputeHBwdProblem& problem) noexcept {
    if (problem.getStreamCount() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims : {&problem.getGradHPreDims(),  &problem.getGradHPostDims(),  &problem.getGradHResDims(),
                             &problem.getGradRDims(),     &problem.getHPreDims(),       &problem.getHPostDims(),
                             &problem.getProjDims(),      &problem.getRDims(),          &problem.getXDims(),
                             &problem.getWeightDims(),    &problem.getAlphaPreDims(),   &problem.getAlphaPostDims(),
                             &problem.getAlphaResDims(),  &problem.getGradXDims(),      &problem.getGradWeightDims(),
                             &problem.getGradAlphaDims(), &problem.getGradBiasDims(),   &problem.getNDims(),
                             &problem.getEpsDims(),       &problem.getComputeHEpsDims()}) {
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

bool supportsShape(const ProjRmsComputeHBwdProblem& problem) noexcept {
    const std::uint64_t fragment = problem.getXNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? 8 : 16;
    return problem.getOutputs() > 0 && problem.getOutputs() <= 32 && problem.getReduction() >= fragment &&
           problem.getReduction() % fragment == 0;
}

// The three mHC scalars (alpha_pre, alpha_post, alpha_res) accumulate into one
// FP32 gradient vector of this size.
constexpr std::uint64_t kAlphaGradientBytes = 3 * sizeof(float);

}  // namespace

template<class Implementation>
const char* MixedFp32ProjRmsComputeHBwdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> MixedFp32ProjRmsComputeHBwdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result MixedFp32ProjRmsComputeHBwdBase<Implementation>::isApplicable(const ProjRmsComputeHBwdProblem& problem,
                                                                     const Constraints& constraints) const {
    // The MMAC staging and grad-X epilogue use vector memory accesses.
    if (!Implementation::kPortable && (((reinterpret_cast<std::uintptr_t>(problem.getXAddress()) |
                                         reinterpret_cast<std::uintptr_t>(problem.getWeightAddress()) |
                                         reinterpret_cast<std::uintptr_t>(problem.getGradXAddress())) &
                                        15) != 0)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC algorithm requires 16-byte aligned vector operands");
    }
    if (!Implementation::kPortable && !isGfx936Or938(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC primitive requires gfx936 or gfx938");
    }
    if (!(Implementation::kPortable ||
          (problem.getOutputs() > 0 && problem.getOutputs() <= 32 && problem.getReduction() % 256 == 0))) {
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

    const auto x_type      = problem.getXNumericType();
    const auto weight_type = problem.getWeightNumericType();
    if (x_type == weight_type ||
        (x_type != FTRAIN_NUMERIC_TYPE_BF16 && x_type != FTRAIN_NUMERIC_TYPE_FP16 &&
         x_type != FTRAIN_NUMERIC_TYPE_FP32) ||
        (weight_type != FTRAIN_NUMERIC_TYPE_BF16 && weight_type != FTRAIN_NUMERIC_TYPE_FP16 &&
         weight_type != FTRAIN_NUMERIC_TYPE_FP32)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "requires different supported x and weight storage types");
    }

    // Mapping state may retain activation storage or use FP32 for mixed training.
    const auto state_type = problem.getHPreNumericType();
    if (state_type != x_type && state_type != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "mapping state must use x storage or FP32");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getGradHPreNumericType() != state_type || problem.getGradHPostNumericType() != state_type ||
        problem.getGradHResNumericType() != state_type || problem.getGradRNumericType() != state_type ||
        problem.getHPostNumericType() != state_type || problem.getProjNumericType() != state_type ||
        problem.getRNumericType() != state_type ||

        problem.getAlphaPreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAlphaPostNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAlphaResNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradXNumericType() != problem.getXNumericType() ||
        problem.getGradWeightNumericType() != problem.getWeightNumericType() ||
        problem.getGradAlphaNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradBiasNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getNNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getEpsNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getComputeHEpsNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getGradHPreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradHPostIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradHResIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradRIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHPreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHPostIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getProjIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWeightIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAlphaPreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAlphaPostIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAlphaResIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradWeightIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradAlphaIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradBiasIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getNIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getEpsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getComputeHEpsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getGradHPreAddress()) %
                (problem.getGradHPreNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float)
                                                                              : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradHPostAddress()) %
                (problem.getGradHPostNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float)
                                                                               : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradHResAddress()) %
                (problem.getGradHResNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float)
                                                                              : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradRAddress()) %
                (problem.getGradRNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPreAddress()) %
                (problem.getHPreNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPostAddress()) %
                (problem.getHPostNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getProjAddress()) %
                (problem.getProjNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRAddress()) %
                (problem.getRNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getXAddress()) %
                (problem.getXNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWeightAddress()) %
                (problem.getWeightNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float)
                                                                            : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAlphaPreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAlphaPostAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAlphaResAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradXAddress()) %
                (problem.getGradXNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradWeightAddress()) %
                (problem.getGradWeightNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float)
                                                                                : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradAlphaAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradBiasAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getNAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getEpsAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getComputeHEpsAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getGradHPreAddress() == nullptr || problem.getGradHPostAddress() == nullptr ||
        problem.getGradHResAddress() == nullptr || problem.getGradRAddress() == nullptr ||
        problem.getHPreAddress() == nullptr || problem.getHPostAddress() == nullptr ||
        problem.getProjAddress() == nullptr || problem.getRAddress() == nullptr || problem.getXAddress() == nullptr ||
        problem.getWeightAddress() == nullptr || problem.getAlphaPreAddress() == nullptr ||
        problem.getAlphaPostAddress() == nullptr || problem.getAlphaResAddress() == nullptr ||
        problem.getGradXAddress() == nullptr || problem.getGradWeightAddress() == nullptr ||
        problem.getGradAlphaAddress() == nullptr || problem.getGradBiasAddress() == nullptr ||
        problem.getEpsAddress() == nullptr || problem.getComputeHEpsAddress() == nullptr ||
        problem.getNAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t MixedFp32ProjRmsComputeHBwdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return workspace_bytes_;
}

template<class Implementation>
void MixedFp32ProjRmsComputeHBwdBase<Implementation>::configure(const ProjRmsComputeHBwdProblem& problem) {
    x_type_      = problem.getXNumericType();
    weight_type_ = problem.getWeightNumericType();
    launch_      = problem.getHPreNumericType() == FTRAIN_NUMERIC_TYPE_FP32 && x_type_ != FTRAIN_NUMERIC_TYPE_FP32
                       ? bindMixedFp32ProjRmsComputeHBwd<Implementation, true>(x_type_, weight_type_)
                       : bindMixedFp32ProjRmsComputeHBwd<Implementation, false>(x_type_, weight_type_);

    grad_h_pre_    = problem.getGradHPreAddress();
    grad_h_post_   = problem.getGradHPostAddress();
    grad_h_res_    = problem.getGradHResAddress();
    grad_r_        = problem.getGradRAddress();
    h_pre_         = problem.getHPreAddress();
    h_post_        = problem.getHPostAddress();
    proj_          = problem.getProjAddress();
    r_             = problem.getRAddress();
    x_             = problem.getXAddress();
    weight_        = problem.getWeightAddress();
    alpha_pre_     = static_cast<const float*>(problem.getAlphaPreAddress());
    alpha_post_    = static_cast<const float*>(problem.getAlphaPostAddress());
    alpha_res_     = static_cast<const float*>(problem.getAlphaResAddress());
    grad_x_        = problem.getGradXAddress();
    grad_weight_   = problem.getGradWeightAddress();
    grad_alpha_    = static_cast<float*>(problem.getGradAlphaAddress());
    grad_bias_     = static_cast<float*>(problem.getGradBiasAddress());
    rows_          = static_cast<int>(problem.getRows());
    outputs_       = static_cast<int>(problem.getOutputs());
    num_streams_   = static_cast<int>(problem.getStreamCount());
    reduction_     = static_cast<int>(problem.getReduction());
    eps_           = problem.getEps();
    compute_h_eps_ = problem.getComputeHEps();
    // The kernel's float scratch scales with the problem; the Primitive owns
    // it so the launch stays free of allocator calls (and graph-capturable).
    workspace_bytes_ = 0;
    workspace_bytes_ +=
        mixedFp32ProjRmsComputeHBwdWorkspaceFloats<Implementation>(rows_, outputs_, reduction_) * sizeof(float);
}

template<class Implementation>
void MixedFp32ProjRmsComputeHBwdBase<Implementation>::executeImpl(const Resources& resources) {
    // The scalar and bias gradients are atomic accumulators: the launch owns
    // them, so they start at exact zero.
    mixedFp32ProjRmsComputeHBwdZeroMemory(resources.getStream(), grad_alpha_, kAlphaGradientBytes);
    mixedFp32ProjRmsComputeHBwdZeroMemory(resources.getStream(), grad_bias_,
                                          static_cast<std::uint64_t>(outputs_) * sizeof(float));
    float* const workspace = static_cast<float*>(resources.getWorkspace());
    const bool launched =
        launch_(resources.getStream(), grad_h_pre_, grad_h_post_, grad_h_res_, grad_r_, h_pre_, h_post_, proj_, r_, x_,
                weight_, alpha_pre_, alpha_post_, alpha_res_, grad_x_, grad_weight_, grad_alpha_, grad_bias_, workspace,
                rows_, outputs_, num_streams_, reduction_, eps_, compute_h_eps_, x_type_, weight_type_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "ProjRmsComputeHBackward kernel rejected a shape its applicability check accepted");
    }
}

template class MixedFp32ProjRmsComputeHBwdBase<MixedFp32ProjRmsComputeHBwdMmacGradX>;
template class MixedFp32ProjRmsComputeHBwdBase<MixedFp32ProjRmsComputeHBwd>;
template class MixedFp32ProjRmsComputeHBwdBase<MixedFp32ProjRmsComputeHBwdMmac>;
}  // namespace ftrain
