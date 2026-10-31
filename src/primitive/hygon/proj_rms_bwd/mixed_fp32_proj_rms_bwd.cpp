// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/proj_rms_bwd/mixed_fp32_proj_rms_bwd.hpp"

namespace ftrain {
namespace {

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
    const std::uint64_t fragment = problem.getXNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? 8 : 16;
    return problem.getOutputs() > 0 && problem.getOutputs() <= 32 && problem.getReduction() >= fragment &&
           problem.getReduction() % fragment == 0;
}

}  // namespace

const char* MixedFp32ProjRmsBwd::getName() const noexcept { return "MixedFp32ProjRmsBwd"; }

std::unique_ptr<PrimitiveBase> MixedFp32ProjRmsBwd::clone() const {
    return std::make_unique<MixedFp32ProjRmsBwd>(*this);
}

Result MixedFp32ProjRmsBwd::isApplicable(const ProjRmsBwdProblem& problem, const Constraints&) const {
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

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getGradProjNumericType() != problem.getXNumericType() ||
        problem.getGradRNumericType() != problem.getXNumericType() ||

        problem.getRNumericType() != problem.getXNumericType() ||
        problem.getGradXNumericType() != problem.getXNumericType() ||
        problem.getGradWeightNumericType() != problem.getWeightNumericType() ||
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
    if (reinterpret_cast<std::uintptr_t>(problem.getGradProjAddress()) %
                (problem.getGradProjNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float)
                                                                              : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradRAddress()) %
                (problem.getGradRNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getXAddress()) %
                (problem.getXNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWeightAddress()) %
                (problem.getWeightNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float)
                                                                            : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRAddress()) %
                (problem.getRNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradXAddress()) %
                (problem.getGradXNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradWeightAddress()) %
                (problem.getGradWeightNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float)
                                                                                : alignof(std::uint16_t)) !=
            0 ||
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

std::uint64_t MixedFp32ProjRmsBwd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void MixedFp32ProjRmsBwd::configure(const ProjRmsBwdProblem& problem) {
    x_type_      = problem.getXNumericType();
    weight_type_ = problem.getWeightNumericType();

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

void MixedFp32ProjRmsBwd::executeImpl(const Resources& resources) {
    const bool launched =
        launchMixedFp32ProjRmsBwd(resources.getStream(), grad_proj_, grad_r_, x_, weight_, r_, grad_x_, grad_weight_,
                                  rows_, outputs_, reduction_, eps_, x_type_, weight_type_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "ProjRmsBackward kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
