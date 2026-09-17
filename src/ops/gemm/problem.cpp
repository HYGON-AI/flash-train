#include "flash_train/ops/gemm/problem.hpp"

#include <cstdint>
#include <limits>

namespace ftrain {
namespace {

bool hasRangeOverflow(const void* memory, std::uint64_t bytes) noexcept {
    if (memory == nullptr || bytes == 0) { return false; }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(memory);
    return bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

bool rangesOverlap(const void* first_memory, std::uint64_t first_bytes, const void* second_memory,
                   std::uint64_t second_bytes) noexcept {
    if (first_memory == nullptr || second_memory == nullptr || first_bytes == 0 || second_bytes == 0) { return false; }
    if (hasRangeOverflow(first_memory, first_bytes) || hasRangeOverflow(second_memory, second_bytes)) { return true; }

    const std::uintptr_t first_begin  = reinterpret_cast<std::uintptr_t>(first_memory);
    const std::uintptr_t second_begin = reinterpret_cast<std::uintptr_t>(second_memory);
    const std::uintptr_t first_end    = first_begin + static_cast<std::uintptr_t>(first_bytes);
    const std::uintptr_t second_end   = second_begin + static_cast<std::uintptr_t>(second_bytes);
    return first_begin < second_end && second_begin < first_end;
}

}  // namespace

bool gemmMatrixBytesOverflow(std::uint64_t rows, std::uint64_t columns, std::uint64_t& bytes) noexcept {
    if (rows != 0 && columns > std::numeric_limits<std::uint64_t>::max() / rows) { return true; }
    const std::uint64_t elements = rows * columns;
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) { return true; }
    bytes = elements * sizeof(float);
    return false;
}

bool hasGemmRanks(const GemmProblem& problem) {
    return problem.a.getDims().size() == 2 && problem.b.getDims().size() == 2 && problem.c.getDims().size() == 2 &&
           problem.d.getDims().size() == 2 && problem.alpha.getDims().empty() && problem.beta.getDims().empty();
}

MemoryRelations getMemoryRelations(const GemmProblem& problem) {
    if (!hasGemmRanks(problem)) { return MemoryRelations{false, false}; }

    const GemmShape shape = problem.getShape();
    std::uint64_t a_bytes = 0;
    std::uint64_t b_bytes = 0;
    std::uint64_t c_bytes = 0;
    std::uint64_t d_bytes = 0;
    if (gemmMatrixBytesOverflow(shape.m, shape.k, a_bytes) || gemmMatrixBytesOverflow(shape.k, shape.n, b_bytes) ||
        gemmMatrixBytesOverflow(shape.m, shape.n, c_bytes) || gemmMatrixBytesOverflow(shape.m, shape.n, d_bytes)) {
        return MemoryRelations{true, false};
    }

    const void* a_memory     = problem.a.getMemory();
    const void* b_memory     = problem.b.getMemory();
    const void* c_memory     = problem.c.getMemory();
    const void* alpha_memory = problem.alpha.getMemory();
    const void* beta_memory  = problem.beta.getMemory();
    void* d_memory           = problem.d.getMemory();

    const bool has_range_overflow = hasRangeOverflow(a_memory, a_bytes) || hasRangeOverflow(b_memory, b_bytes) ||
                                    hasRangeOverflow(c_memory, c_bytes) ||
                                    hasRangeOverflow(alpha_memory, sizeof(float)) ||
                                    hasRangeOverflow(beta_memory, sizeof(float)) || hasRangeOverflow(d_memory, d_bytes);
    const bool output_input_overlap = rangesOverlap(d_memory, d_bytes, a_memory, a_bytes) ||
                                      rangesOverlap(d_memory, d_bytes, b_memory, b_bytes) ||
                                      rangesOverlap(d_memory, d_bytes, c_memory, c_bytes) ||
                                      rangesOverlap(d_memory, d_bytes, alpha_memory, sizeof(float)) ||
                                      rangesOverlap(d_memory, d_bytes, beta_memory, sizeof(float));
    return MemoryRelations{has_range_overflow, output_input_overlap};
}

}  // namespace ftrain
