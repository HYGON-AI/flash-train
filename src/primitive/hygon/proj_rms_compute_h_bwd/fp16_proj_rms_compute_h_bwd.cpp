// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/proj_rms_compute_h_bwd/fp16_proj_rms_compute_h_bwd.hpp"

namespace ftrain {
namespace {

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
    constexpr std::uint64_t fragment = 16;
    return problem.getOutputs() > 0 && problem.getOutputs() <= 32 && problem.getReduction() >= fragment &&
           problem.getReduction() % fragment == 0;
}

// The three mHC scalars (alpha_pre, alpha_post, alpha_res) accumulate into one
// FP32 gradient vector of this size.
constexpr std::uint64_t kAlphaGradientBytes = 3 * sizeof(float);

}  // namespace

const char* Fp16ProjRmsComputeHBwd::getName() const noexcept { return "Fp16ProjRmsComputeHBwd"; }

std::unique_ptr<PrimitiveBase> Fp16ProjRmsComputeHBwd::clone() const {
    return std::make_unique<Fp16ProjRmsComputeHBwd>(*this);
}

Result Fp16ProjRmsComputeHBwd::isApplicable(const ProjRmsComputeHBwdProblem& problem, const Constraints&) const {
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

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getGradHPreNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getGradHPostNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getGradHResNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getGradRNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getHPreNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getHPostNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getProjNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getRNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getWeightNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getAlphaPreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAlphaPostNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAlphaResNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradXNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getGradWeightNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
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
    if (reinterpret_cast<std::uintptr_t>(problem.getGradHPreAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradHPostAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradHResAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradRAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPreAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPostAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getProjAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getXAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWeightAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAlphaPreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAlphaPostAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAlphaResAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradXAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradWeightAddress()) % alignof(std::uint16_t) != 0 ||
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

std::uint64_t Fp16ProjRmsComputeHBwd::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void Fp16ProjRmsComputeHBwd::configure(const ProjRmsComputeHBwdProblem& problem) {
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
    workspace_bytes_  = 0;
    workspace_bytes_ += fp16ProjRmsComputeHBwdWorkspaceFloats(rows_, outputs_, reduction_) * sizeof(float);
}

void Fp16ProjRmsComputeHBwd::executeImpl(const Resources& resources) {
    // The scalar and bias gradients are atomic accumulators: the launch owns
    // them, so they start at exact zero.
    fp16ProjRmsComputeHBwdZeroMemory(resources.getStream(), grad_alpha_, kAlphaGradientBytes);
    fp16ProjRmsComputeHBwdZeroMemory(resources.getStream(), grad_bias_,
                                     static_cast<std::uint64_t>(outputs_) * sizeof(float));
    float* const workspace = static_cast<float*>(resources.getWorkspace());
    const bool launched    = launchFp16ProjRmsComputeHBwd(
        resources.getStream(), grad_h_pre_, grad_h_post_, grad_h_res_, grad_r_, h_pre_, h_post_, proj_, r_, x_, weight_,
        alpha_pre_, alpha_post_, alpha_res_, grad_x_, grad_weight_, grad_alpha_, grad_bias_, workspace, rows_, outputs_,
        num_streams_, reduction_, eps_, compute_h_eps_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "ProjRmsComputeHBackward kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
