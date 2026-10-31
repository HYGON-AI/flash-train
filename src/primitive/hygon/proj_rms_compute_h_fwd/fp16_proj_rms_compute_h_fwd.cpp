// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/proj_rms_compute_h_fwd/fp16_proj_rms_compute_h_fwd.hpp"

namespace ftrain {
namespace {

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

const char* Fp16ProjRmsComputeHFwd::getName() const noexcept { return "Fp16ProjRmsComputeHFwd"; }

std::unique_ptr<PrimitiveBase> Fp16ProjRmsComputeHFwd::clone() const {
    return std::make_unique<Fp16ProjRmsComputeHFwd>(*this);
}

Result Fp16ProjRmsComputeHFwd::isApplicable(const ProjRmsComputeHFwdProblem& problem, const Constraints&) const {
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

std::uint64_t Fp16ProjRmsComputeHFwd::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void Fp16ProjRmsComputeHFwd::configure(const ProjRmsComputeHFwdProblem& problem) {
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
    workspace_bytes_ = fp16ProjRmsComputeHFwdWorkspaceBytes(rows_, outputs_, reduction_, true);
}

void Fp16ProjRmsComputeHFwd::executeImpl(const Resources& resources) {
    const bool launched = launchFp16ProjRmsComputeHFwd(
        resources.getStream(), x_, weight_, proj_, r_, alpha_pre_, alpha_post_, alpha_res_, bias_, h_pre_, h_post_,
        h_res_, rows_, outputs_, num_streams_, reduction_, eps_, compute_h_eps_, resources.getWorkspace());
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "ProjRmsComputeHForward kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
