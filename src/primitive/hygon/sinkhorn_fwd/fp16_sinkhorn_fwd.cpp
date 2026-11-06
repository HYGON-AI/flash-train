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

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, int SIZE>
Fp16SinkhornFwdLaunch bindIterations(int iterations) {
    return &launchFp16SinkhornFwd<Implementation, SIZE, 0>;
}

template<class Implementation>
Fp16SinkhornFwdLaunch bindFp16SinkhornFwd(int size, int iterations) {
    if constexpr (Implementation::kWave) {
        if (size == 4) return bindIterations<Implementation, 4>(iterations);
        return bindIterations<Implementation, 8>(iterations);
    } else {
        return &launchFp16SinkhornFwd<Implementation, 0, 0>;
    }
}

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

template<class Implementation>
const char* Fp16SinkhornFwdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Fp16SinkhornFwdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Fp16SinkhornFwdBase<Implementation>::isApplicable(const SinkhornFwdProblem& problem, const Constraints&) const {
    if (!(!Implementation::kWave || ((problem.getMatrixSize() == 4 || problem.getMatrixSize() == 8)))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

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
        problem.getMInitAddress() == nullptr || problem.getEpsAddress() == nullptr ||
        problem.getNumIterationsAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Fp16SinkhornFwdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return 0;
}

template<class Implementation>
void Fp16SinkhornFwdBase<Implementation>::configure(const SinkhornFwdProblem& problem) {
    logits_         = problem.getLogitsAddress();
    output_         = problem.getOutputAddress();
    m_init_         = static_cast<float*>(problem.getMInitAddress());
    batch_          = static_cast<int>(problem.getBatch());
    matrix_size_    = static_cast<int>(problem.getMatrixSize());
    num_iterations_ = static_cast<int>(problem.getNumIterations());
    eps_            = problem.getEps();
    launch_         = bindFp16SinkhornFwd<Implementation>(matrix_size_, num_iterations_);
}

template<class Implementation>
void Fp16SinkhornFwdBase<Implementation>::executeImpl(const Resources& resources) {
    if (batch_ == 0) { return; }
    const bool launched =
        launch_(resources.getStream(), logits_, output_, m_init_, batch_, matrix_size_, num_iterations_, eps_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SinkhornFwd kernel rejected a shape its applicability check accepted");
    }
}

template class Fp16SinkhornFwdBase<Fp16SinkhornFwd>;
template class Fp16SinkhornFwdBase<Fp16SinkhornFwdWave>;
}  // namespace ftrain
