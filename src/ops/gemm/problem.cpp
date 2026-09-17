#include "flash_train/ops/gemm/problem.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

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

enum class GemmSelectionTokenTag : std::uint64_t {
    kFormatVersion = 1,
    kA,
    kB,
    kC,
    kAlpha,
    kBeta,
    kD,
    kDimensions,
    kStrides,
    kComputeType,
    kMemoryRelations,
};

void appendSignedVector(GemmSelectionTokenTag tag, const std::vector<std::int64_t>& operands,
                        std::vector<std::uint64_t>& tokens) {
    tokens.push_back(static_cast<std::uint64_t>(tag));
    tokens.push_back(static_cast<std::uint64_t>(operands.size()));
    for (const std::int64_t operand : operands) { tokens.push_back(static_cast<std::uint64_t>(operand)); }
}

void appendStorageViewTokens(GemmSelectionTokenTag tag, const StorageView& view, std::size_t required_alignment,
                             std::vector<std::uint64_t>& tokens) {
    const bool has_memory = view.getMemory() != nullptr;
    const std::uintptr_t address_class =
        has_memory ? reinterpret_cast<std::uintptr_t>(view.getMemory()) % required_alignment : 0;

    tokens.push_back(static_cast<std::uint64_t>(tag));
    tokens.push_back(static_cast<std::uint64_t>(view.getNumericType()));
    tokens.push_back(static_cast<std::uint64_t>(view.getIndexType()));
    tokens.push_back(static_cast<std::uint64_t>(view.isHostMemory()));
    tokens.push_back(static_cast<std::uint64_t>(has_memory));
    tokens.push_back(static_cast<std::uint64_t>(address_class));
    appendSignedVector(GemmSelectionTokenTag::kDimensions, view.getDims(), tokens);
    appendSignedVector(GemmSelectionTokenTag::kStrides, view.getStrides(), tokens);
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

std::vector<std::uint64_t> GemmProblem::getSelectionTokens() const {
    std::vector<std::uint64_t> tokens;
    tokens.reserve(96);
    tokens.push_back(static_cast<std::uint64_t>(GemmSelectionTokenTag::kFormatVersion));
    tokens.push_back(1);
    appendStorageViewTokens(GemmSelectionTokenTag::kA, a, alignof(float), tokens);
    appendStorageViewTokens(GemmSelectionTokenTag::kB, b, alignof(float), tokens);
    appendStorageViewTokens(GemmSelectionTokenTag::kC, c, alignof(float), tokens);
    appendStorageViewTokens(GemmSelectionTokenTag::kAlpha, alpha, alignof(float), tokens);
    appendStorageViewTokens(GemmSelectionTokenTag::kBeta, beta, alignof(float), tokens);
    appendStorageViewTokens(GemmSelectionTokenTag::kD, d, alignof(float), tokens);
    tokens.push_back(static_cast<std::uint64_t>(GemmSelectionTokenTag::kComputeType));
    tokens.push_back(static_cast<std::uint64_t>(attributes.getComputeType()));

    const MemoryRelations relations = getMemoryRelations(*this);
    tokens.push_back(static_cast<std::uint64_t>(GemmSelectionTokenTag::kMemoryRelations));
    tokens.push_back(static_cast<std::uint64_t>(relations.has_range_overflow));
    tokens.push_back(static_cast<std::uint64_t>(relations.output_input_overlap));
    return tokens;
}

}  // namespace ftrain
