#ifndef FTRAIN_PROBLEM_GEMM_PROBLEM_HPP_
#define FTRAIN_PROBLEM_GEMM_PROBLEM_HPP_

#include <array>
#include <cstdint>
#include <vector>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/operation/gemm.hpp"
#include "flash_train/storage_view.hpp"

namespace ftrain {

// GEMM extents: m rows, n columns, k reduction depth.
struct GemmShape {
    std::uint64_t m;
    std::uint64_t n;
    std::uint64_t k;
};

// The six semantic ports of d = alpha * (a @ b) + beta * c, addressed by
// GemmProblem's per-port property getters.
enum class GemmPort : std::uint8_t {
    kA,
    kB,
    kC,
    kD,
    kAlpha,
    kBeta,
};

// Every port, in enumeration order.
inline constexpr GemmPort kAllGemmPorts[]{
    GemmPort::kA, GemmPort::kB, GemmPort::kC, GemmPort::kD, GemmPort::kAlpha, GemmPort::kBeta,
};

// Memory-relation summary of one GemmProblem, shared by the Gemm
// primitives' applicability checks and the problem's selection tokens.
struct MemoryRelations {
    bool has_range_overflow;
    bool output_input_overlap;
};

// One Gemm call's parameters as a support checklist: each public getter
// names one property a Gemm Primitive must decide whether it supports, so
// walking the list function by function is the entire checklist. The
// constructor validates the cross-field structure -- rank-two a, b, c, and
// d; scalar alpha and beta; a's column count equal to b's row count; c and
// d shaped m-by-n; matrix byte sizes within std::uint64_t -- throwing
// Exception with FTRAIN_STATUS_INVALID_ARGUMENT or FTRAIN_STATUS_OVERFLOW,
// so only a well-formed GemmProblem reaches a Primitive and isApplicable
// answers support questions only.
class GemmProblem final {
  public:
    // Takes the six ports' views by value together with the call's
    // attributes. See the class comment for the validation this performs;
    // allocation failure throws std::bad_alloc.
    GemmProblem(StorageView a, StorageView b, StorageView c, StorageView d, StorageView alpha, StorageView beta,
                GemmAttributes attributes);

    // --- shape ---

    // Returns {m, n, k}: m = a's rows, k = a's columns (= b's rows), and
    // n = b's columns.
    GemmShape getShape() const noexcept;

    // Returns whether m * n == 0: the kernel has no work, and no operand
    // address is required.
    bool hasEmptyOutput() const noexcept;

    // --- element and compute types ---

    // Returns the call's compute type.
    FTrainNumericType getComputeType() const noexcept;

    // Returns the port's element type.
    FTrainNumericType getNumericType(GemmPort port) const noexcept;

    // --- layout ---

    // Returns the port's predefined layout, or FTRAIN_INDEX_TYPE_INVALID
    // when the port is indexed by getStrides(port) instead.
    FTrainIndexType getIndexType(GemmPort port) const noexcept;

    // Returns the port's explicit strides; empty for a predefined layout.
    const std::vector<std::int64_t>& getStrides(GemmPort port) const noexcept;

    // --- memory placement and addresses ---

    // Returns whether the port's memory is host memory.
    bool isHostMemory(GemmPort port) const noexcept;

    // Returns the port's base address; null only where the kernel skips
    // the port (see hasEmptyOutput and the reduction-depth rule).
    void* getAddress(GemmPort port) const noexcept;

    // --- aliasing ---

    // Returns the memory-relation summary across the ports: whether any
    // port's byte range overflows the address space, and whether d overlaps
    // an input.
    MemoryRelations getMemoryRelations() const;

    // --- selection cache ---

    // Encodes every property that can change which Primitive applies into
    // integer tokens; problems with equal tokens must select the same
    // Primitive, because a cache hit skips the applicability check. Encodes
    // memory addresses only through properties that affect applicability
    // (alignment classes and overlap here), never as raw values, and gives
    // every variable-length field its element count so different encodings
    // cannot alias. Device and workspace limit are added by the engine.
    std::vector<std::uint64_t> getSelectionTokens() const;

  private:
    const StorageView& view(GemmPort port) const noexcept { return ports_[static_cast<std::size_t>(port)]; }

    std::array<StorageView, 6> ports_;
    GemmAttributes attributes_;
};

}  // namespace ftrain

#endif
