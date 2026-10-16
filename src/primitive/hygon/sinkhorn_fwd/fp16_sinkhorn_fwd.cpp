// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sinkhorn_fwd/fp16_sinkhorn_fwd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const SinkhornFwdProblem& problem) noexcept {
    if (problem.getNumIterations() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims : {&problem.getLogitsDims(), &problem.getOutputDims(), &problem.getMInitDims(),
                             &problem.getNumIterationsDims(), &problem.getEpsDims()}) {
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

bool supportsShape(const SinkhornFwdProblem& problem) noexcept { return problem.getMatrixSize() <= 32; }

}  // namespace

const char* Fp16SinkhornFwd::getName() const noexcept { return "Fp16SinkhornFwd"; }

std::unique_ptr<PrimitiveBase> Fp16SinkhornFwd::clone() const { return std::make_unique<Fp16SinkhornFwd>(*this); }

Result Fp16SinkhornFwd::isApplicable(const SinkhornFwdProblem& problem, const Constraints&) const {
    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }
    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds this kernel implementation's tile limits");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getLogitsNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getOutputNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getMInitNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getNumIterationsNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getEpsNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getLogitsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutputIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getMInitIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getNumIterationsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getEpsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getLogitsAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutputAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getMInitAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getNumIterationsAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getEpsAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.hasEmptyOutput()) { return Result{}; }
    if (problem.getLogitsAddress() == nullptr || problem.getOutputAddress() == nullptr ||
        problem.getMInitAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

std::uint64_t Fp16SinkhornFwd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Fp16SinkhornFwd::configure(const SinkhornFwdProblem& problem) {
    logits_         = problem.getLogitsAddress();
    output_         = problem.getOutputAddress();
    m_init_         = static_cast<float*>(problem.getMInitAddress());
    batch_          = static_cast<int>(problem.getBatch());
    matrix_size_    = static_cast<int>(problem.getMatrixSize());
    num_iterations_ = static_cast<int>(problem.getNumIterations());
    eps_            = problem.getEps();
}

void Fp16SinkhornFwd::executeImpl(const Resources& resources) {
    if (batch_ == 0) { return; }
    const bool launched = launchFp16SinkhornFwd(resources.getStream(), logits_, output_, m_init_, batch_, matrix_size_,
                                                num_iterations_, eps_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SinkhornFwd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
