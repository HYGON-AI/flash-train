#ifndef FTRAIN_OPS_GEMM_PROBLEM_HPP_
#define FTRAIN_OPS_GEMM_PROBLEM_HPP_

#include <cstdint>

#include "flash_train/common.h"
#include "flash_train/operation/gemm.hpp"
#include "flash_train/storage_view.hpp"

namespace ftrain {

// GEMM extents: m rows, n columns, k reduction depth.
struct GemmShape {
    std::uint64_t m;
    std::uint64_t n;
    std::uint64_t k;
};

// One Gemm call's parameters, describing d = alpha * (a @ b) + beta * c.
// Assembled by the engine from Args; the views are non-owning and stay valid
// for the call.
struct GemmProblem {
    StorageView a;
    StorageView b;
    StorageView c;
    StorageView d;
    StorageView alpha;
    StorageView beta;
    GemmAttributes attributes;

    // Returns {m, n, k}, where m = a's rows, k = a's columns (= b's rows),
    // and n = b's columns.
    GemmShape getShape() const noexcept {
        return GemmShape{static_cast<std::uint64_t>(a.getDims().empty() ? 0 : a.getDims()[0]),
                         static_cast<std::uint64_t>(b.getDims().empty()      ? 0
                                                    : b.getDims().size() > 1 ? b.getDims()[1]
                                                                             : 0),
                         static_cast<std::uint64_t>(a.getDims().size() > 1 ? a.getDims()[1] : 0)};
    }
};

// Rank predicate shared by the Gemm primitives and engine: a, b, c, and d
// are rank two and alpha and beta are scalars.
bool hasGemmRanks(const GemmProblem& problem);

// Computes the byte size of a rows-by-columns element block. Returns false
// and leaves bytes untouched when the size overflows std::uint64_t; shared
// by the engine's problem validation and getMemoryRelations.
bool gemmMatrixBytesOverflow(std::uint64_t rows, std::uint64_t columns, std::uint64_t& bytes) noexcept;

// Memory-relation summary of one GemmProblem, shared by the Gemm
// primitives' applicability checks and the engine's selection tokens.
struct MemoryRelations {
    bool has_range_overflow;
    bool output_input_overlap;
};

MemoryRelations getMemoryRelations(const GemmProblem& problem);

}  // namespace ftrain

#endif
