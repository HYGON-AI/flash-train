#include "flash_train/primitive/gemm/fp32_gemm.hpp"

#include <cstdint>
#include <memory>

#include "flash_train/error.hpp"

namespace ftrain {
namespace {

constexpr std::uint64_t kRequiredWorkspaceBytes = 0;

bool isAlignedForFloat(const void* memory) noexcept {
    return memory == nullptr || reinterpret_cast<std::uintptr_t>(memory) % alignof(float) == 0;
}

}  // namespace

const char* Fp32Gemm::getName() const noexcept { return "Fp32Gemm"; }

std::unique_ptr<PrimitiveBase> Fp32Gemm::clone() const { return std::make_unique<Fp32Gemm>(*this); }

// The support checklist, one GemmProblem getter group at a time: compute
// and element types, layouts, addresses, and shape. Memory placement is
// constructor-guaranteed (device a/b/c/d, host alpha/beta).
Result Fp32Gemm::isApplicable(const GemmProblem& problem, const Constraints&) const {
    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type is not fp32");
    }
    if (problem.getANumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getBNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getCNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getDNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAlphaNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getBetaNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not fp32");
    }
    if (problem.getAIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getBIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAlphaIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getBetaIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (!isAlignedForFloat(problem.getAAddress()) || !isAlignedForFloat(problem.getBAddress()) ||
        !isAlignedForFloat(problem.getCAddress()) || !isAlignedForFloat(problem.getDAddress()) ||
        !isAlignedForFloat(problem.getAlphaAddress()) || !isAlignedForFloat(problem.getBetaAddress())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not float-aligned");
    }

    if (problem.hasEmptyOutput()) { return Result{}; }

    const bool has_required_addresses =
        problem.getAlphaAddress() != nullptr && problem.getBetaAddress() != nullptr &&
        problem.getDAddress() != nullptr &&
        (problem.getShape().k == 0 || (problem.getAAddress() != nullptr && problem.getBAddress() != nullptr));
    if (!has_required_addresses) { return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null"); }
    return Result{};
}

std::uint64_t Fp32Gemm::getRequiredWorkspaceBytes() const noexcept { return kRequiredWorkspaceBytes; }

void Fp32Gemm::configure(const GemmProblem& problem) {
    a_ = static_cast<const float*>(problem.getAAddress());
    b_ = static_cast<const float*>(problem.getBAddress());
    c_ = static_cast<const float*>(problem.getCAddress());
    // alpha and beta live in host memory: read their values here.
    alpha_ = *static_cast<const float*>(problem.getAlphaAddress());
    beta_  = *static_cast<const float*>(problem.getBetaAddress());
    d_     = static_cast<float*>(problem.getDAddress());
    m_     = problem.getShape().m;
    n_     = problem.getShape().n;
    k_     = problem.getShape().k;
}

void Fp32Gemm::executeImpl(const Resources& resources) {
    launchFp32Gemm(a_, b_, c_, alpha_, beta_, d_, m_, n_, k_, resources.getStream());
}

}  // namespace ftrain
