// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sinkhorn_bwd/bf16_sinkhorn_bwd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const SinkhornBwdProblem& problem) noexcept {
    if (problem.getNumIterations() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims : {&problem.getGradOutputDims(), &problem.getMInitDims(), &problem.getGradLogitsDims(),
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

bool supportsShape(const SinkhornBwdProblem& problem) noexcept {
    const auto size = problem.getMatrixSize();
    return size <= 32 && (size * size * 3 + size * 2 +
                          static_cast<std::uint64_t>(problem.getNumIterations()) * (2 * size + 2 * size * size)) *
                                 sizeof(float) <=
                             65536;
}

}  // namespace

const char* Bf16SinkhornBwd::getName() const noexcept { return "Bf16SinkhornBwd"; }

std::unique_ptr<PrimitiveBase> Bf16SinkhornBwd::clone() const { return std::make_unique<Bf16SinkhornBwd>(*this); }

Result Bf16SinkhornBwd::isApplicable(const SinkhornBwdProblem& problem, const Constraints&) const {
    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }
    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "shape or iteration count exceeds this implementation's tile or shared-memory limits");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getGradOutputNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getMInitNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradLogitsNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getNumIterationsNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getEpsNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getGradOutputIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getMInitIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradLogitsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getNumIterationsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getEpsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getGradOutputAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getMInitAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradLogitsAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getNumIterationsAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getEpsAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.hasEmptyOutput()) { return Result{}; }
    if (problem.getGradOutputAddress() == nullptr || problem.getMInitAddress() == nullptr ||
        problem.getGradLogitsAddress() == nullptr || problem.getEpsAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

std::uint64_t Bf16SinkhornBwd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Bf16SinkhornBwd::configure(const SinkhornBwdProblem& problem) {
    grad_output_    = problem.getGradOutputAddress();
    m_init_         = static_cast<const float*>(problem.getMInitAddress());
    grad_logits_    = problem.getGradLogitsAddress();
    batch_          = static_cast<int>(problem.getBatch());
    matrix_size_    = static_cast<int>(problem.getMatrixSize());
    num_iterations_ = static_cast<int>(problem.getNumIterations());
    eps_            = problem.getEps();
}

void Bf16SinkhornBwd::executeImpl(const Resources& resources) {
    if (batch_ == 0) { return; }
    const bool launched = launchBf16SinkhornBwd(resources.getStream(), grad_output_, m_init_, grad_logits_, batch_,
                                                matrix_size_, num_iterations_, eps_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SinkhornBwd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
