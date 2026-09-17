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

// The support checklist, one GemmProblem getter at a time: element types,
// layout, memory placement, compute type, shape, addresses, aliasing.
Result Fp32Gemm::isApplicable(const GemmProblem& problem, const Constraints&) const {
    for (const GemmPort port : kAllGemmPorts) {
        if (problem.getNumericType(port) != FTRAIN_NUMERIC_TYPE_FP32) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not fp32");
        }
        if (problem.getIndexType(port) != FTRAIN_INDEX_TYPE_CONTINUOUS) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
        }
        if (problem.isHostMemory(port)) { return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is host memory"); }
        if (!isAlignedForFloat(problem.getAddress(port))) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not float-aligned");
        }
    }
    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type is not fp32");
    }

    if (problem.hasEmptyOutput()) { return Result{}; }

    const bool has_required_addresses =
        problem.getAddress(GemmPort::kC) != nullptr && problem.getAddress(GemmPort::kAlpha) != nullptr &&
        problem.getAddress(GemmPort::kBeta) != nullptr && problem.getAddress(GemmPort::kD) != nullptr &&
        (problem.getShape().k == 0 ||
         (problem.getAddress(GemmPort::kA) != nullptr && problem.getAddress(GemmPort::kB) != nullptr));
    if (!has_required_addresses) { return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null"); }

    const MemoryRelations relations = problem.getMemoryRelations();
    if (relations.has_range_overflow) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address range overflows");
    }
    if (relations.output_input_overlap) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the output overlaps an input"); }
    return Result{};
}

std::uint64_t Fp32Gemm::getRequiredWorkspaceBytes() const noexcept { return kRequiredWorkspaceBytes; }

void Fp32Gemm::configure(const GemmProblem& problem) {
    a_     = static_cast<const float*>(problem.getAddress(GemmPort::kA));
    b_     = static_cast<const float*>(problem.getAddress(GemmPort::kB));
    c_     = static_cast<const float*>(problem.getAddress(GemmPort::kC));
    alpha_ = static_cast<const float*>(problem.getAddress(GemmPort::kAlpha));
    beta_  = static_cast<const float*>(problem.getAddress(GemmPort::kBeta));
    d_     = static_cast<float*>(problem.getAddress(GemmPort::kD));
    m_     = problem.getShape().m;
    n_     = problem.getShape().n;
    k_     = problem.getShape().k;
}

void Fp32Gemm::executeImpl(void*, std::uint64_t, FTrainStream stream) {
    launchFp32Gemm(a_, b_, c_, alpha_, beta_, d_, m_, n_, k_, stream);
}

}  // namespace ftrain
