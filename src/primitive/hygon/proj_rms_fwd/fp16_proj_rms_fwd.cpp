// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/proj_rms_fwd/fp16_proj_rms_fwd.hpp"

namespace ftrain {
namespace {

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
    constexpr std::uint64_t fragment = 16;
    return problem.getOutputs() > 0 && problem.getOutputs() <= 32 && problem.getReduction() >= fragment &&
           problem.getReduction() % fragment == 0;
}

}  // namespace

const char* Fp16ProjRmsFwd::getName() const noexcept { return "Fp16ProjRmsFwd"; }

std::unique_ptr<PrimitiveBase> Fp16ProjRmsFwd::clone() const { return std::make_unique<Fp16ProjRmsFwd>(*this); }

Result Fp16ProjRmsFwd::isApplicable(const ProjRmsFwdProblem& problem, const Constraints&) const {
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
        problem.getProjNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getRNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
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
    if (reinterpret_cast<std::uintptr_t>(problem.getXAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWeightAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getProjAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getEpsAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getXAddress() == nullptr || problem.getWeightAddress() == nullptr ||
        problem.getProjAddress() == nullptr || problem.getRAddress() == nullptr || problem.getEpsAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

std::uint64_t Fp16ProjRmsFwd::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void Fp16ProjRmsFwd::configure(const ProjRmsFwdProblem& problem) {
    x_               = problem.getXAddress();
    weight_          = problem.getWeightAddress();
    proj_            = problem.getProjAddress();
    r_               = problem.getRAddress();
    rows_            = static_cast<int>(problem.getRows());
    outputs_         = static_cast<int>(problem.getOutputs());
    reduction_       = static_cast<int>(problem.getReduction());
    eps_             = problem.getEps();
    workspace_bytes_ = fp16ProjRmsFwdWorkspaceBytes(rows_, outputs_, reduction_, false);
}

void Fp16ProjRmsFwd::executeImpl(const Resources& resources) {
    const bool launched = launchFp16ProjRmsFwd(resources.getStream(), x_, weight_, proj_, r_, rows_, outputs_,
                                               reduction_, eps_, resources.getWorkspace());
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "ProjRmsForward kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
