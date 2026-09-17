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

Result Fp32Gemm::isApplicable(const GemmProblem& problem, const SelectionContext&) const {
    // --- data type / layout / memory ---
    const StorageView* const views[6] = {&problem.a, &problem.b, &problem.c, &problem.d, &problem.alpha, &problem.beta};
    for (const StorageView* const view : views) {
        if (view->getNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not fp32");
        }
        if (view->getIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
        }
        if (view->isHostMemory()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is host memory"); }
        if (!isAlignedForFloat(view->getMemory())) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not float-aligned");
        }
    }

    // --- shape semantics / attributes ---
    if (!hasGemmRanks(problem)) { return Result(FTRAIN_STATUS_UNSUPPORTED, "matrix operands are not rank 2"); }
    if (problem.attributes.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type is not fp32");
    }

    // --- degenerate shape fast path ---
    const GemmShape shape = problem.getShape();
    if (shape.m * shape.n == 0) { return Result{}; }

    // --- address rules ---
    const bool has_required_addresses =
        problem.c.getMemory() != nullptr && problem.alpha.getMemory() != nullptr &&
        problem.beta.getMemory() != nullptr && problem.d.getMemory() != nullptr &&
        (shape.k == 0 || (problem.a.getMemory() != nullptr && problem.b.getMemory() != nullptr));
    if (!has_required_addresses) { return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null"); }

    // --- memory relations ---
    const MemoryRelations relations = getMemoryRelations(problem);
    if (relations.has_range_overflow) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address range overflows");
    }
    if (relations.output_input_overlap) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the output overlaps an input"); }
    return Result{};
}

std::uint64_t Fp32Gemm::getRequiredWorkspaceBytes() const noexcept { return kRequiredWorkspaceBytes; }

void Fp32Gemm::configure(const GemmProblem& problem) {
    a_     = static_cast<const float*>(problem.a.getMemory());
    b_     = static_cast<const float*>(problem.b.getMemory());
    c_     = static_cast<const float*>(problem.c.getMemory());
    alpha_ = static_cast<const float*>(problem.alpha.getMemory());
    beta_  = static_cast<const float*>(problem.beta.getMemory());
    d_     = static_cast<float*>(problem.d.getMemory());
    m_     = problem.getShape().m;
    n_     = problem.getShape().n;
    k_     = problem.getShape().k;
}

void Fp32Gemm::executeImpl(void*, std::uint64_t, FTrainStream stream) {
    launchFp32Gemm(a_, b_, c_, alpha_, beta_, d_, m_, n_, k_, stream);
}

}  // namespace ftrain
