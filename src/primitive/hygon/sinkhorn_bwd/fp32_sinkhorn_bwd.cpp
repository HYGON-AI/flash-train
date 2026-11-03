// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sinkhorn_bwd/fp32_sinkhorn_bwd.hpp"

namespace ftrain {
namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, int SIZE>
Fp32SinkhornBwdLaunch bindIterations(int iterations) {
    switch (iterations) {
        case 3: return &launchFp32SinkhornBwd<Implementation, SIZE, 3>;
        case 5: return &launchFp32SinkhornBwd<Implementation, SIZE, 5>;
        case 7: return &launchFp32SinkhornBwd<Implementation, SIZE, 7>;
        case 10: return &launchFp32SinkhornBwd<Implementation, SIZE, 10>;
        case 15: return &launchFp32SinkhornBwd<Implementation, SIZE, 15>;
        case 20: return &launchFp32SinkhornBwd<Implementation, SIZE, 20>;
        default:
            throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured Sinkhorn iteration count is not instantiated");
    }
}

template<class Implementation>
Fp32SinkhornBwdLaunch bindFp32SinkhornBwd(int size, int iterations) {
    if constexpr (Implementation::kWave) {
        if (size == 4) return bindIterations<Implementation, 4>(iterations);
        return bindIterations<Implementation, 8>(iterations);
    } else {
        return &launchFp32SinkhornBwd<Implementation, 0, 0>;
    }
}

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

template<class Implementation>
const char* Fp32SinkhornBwdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Fp32SinkhornBwdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Fp32SinkhornBwdBase<Implementation>::isApplicable(const SinkhornBwdProblem& problem, const Constraints&) const {
    if (!(!Implementation::kWave || ((problem.getMatrixSize() == 4 || problem.getMatrixSize() == 8) &&
                                     (problem.getNumIterations() == 3 || problem.getNumIterations() == 5 ||
                                      problem.getNumIterations() == 7 || problem.getNumIterations() == 10 ||
                                      problem.getNumIterations() == 15 || problem.getNumIterations() == 20)))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }
    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "shape or iteration count exceeds this implementation's tile or shared-memory limits");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getGradOutputNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getMInitNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradLogitsNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
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
    if (reinterpret_cast<std::uintptr_t>(problem.getGradOutputAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getMInitAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradLogitsAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getNumIterationsAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getEpsAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.hasEmptyOutput()) { return Result{}; }
    if (problem.getGradOutputAddress() == nullptr || problem.getMInitAddress() == nullptr ||
        problem.getGradLogitsAddress() == nullptr || problem.getEpsAddress() == nullptr ||
        problem.getNumIterationsAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Fp32SinkhornBwdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return 0;
}

template<class Implementation>
void Fp32SinkhornBwdBase<Implementation>::configure(const SinkhornBwdProblem& problem) {
    grad_output_    = problem.getGradOutputAddress();
    m_init_         = static_cast<const float*>(problem.getMInitAddress());
    grad_logits_    = problem.getGradLogitsAddress();
    batch_          = static_cast<int>(problem.getBatch());
    matrix_size_    = static_cast<int>(problem.getMatrixSize());
    num_iterations_ = static_cast<int>(problem.getNumIterations());
    eps_            = problem.getEps();
    launch_         = bindFp32SinkhornBwd<Implementation>(matrix_size_, num_iterations_);
}

template<class Implementation>
void Fp32SinkhornBwdBase<Implementation>::executeImpl(const Resources& resources) {
    if (batch_ == 0) { return; }
    const bool launched = launch_(resources.getStream(), grad_output_, m_init_, grad_logits_, batch_, matrix_size_,
                                  num_iterations_, eps_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SinkhornBwd kernel rejected a shape its applicability check accepted");
    }
}

template class Fp32SinkhornBwdBase<Fp32SinkhornBwd>;
template class Fp32SinkhornBwdBase<Fp32SinkhornBwdWave>;
}  // namespace ftrain
