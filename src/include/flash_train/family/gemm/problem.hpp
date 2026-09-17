#ifndef FTRAIN_FAMILY_GEMM_PROBLEM_HPP_
#define FTRAIN_FAMILY_GEMM_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"

#include "flash_train/operation/gemm.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {

// GEMM extents: m rows, n columns, k reduction depth.
struct GemmShape {
    std::uint64_t m;
    std::uint64_t n;
    std::uint64_t k;
};

// One Gemm call's parameters as a support checklist: each public getter
// names one property of one port that a Gemm Primitive must decide
// whether it supports, so walking the list function by function is the
// entire checklist. The constructor validates the cross-field structure
// -- rank-two a, b, c, and d; scalar alpha and beta; a's column count
// equal to b's row count; c and d shaped m-by-n; matrix byte sizes
// within std::uint64_t; a, b, c, and d in device memory; alpha and beta
// in host memory -- throwing Exception with FTRAIN_STATUS_INVALID_ARGUMENT
// or FTRAIN_STATUS_OVERFLOW, so only a well-formed GemmProblem reaches a
// Primitive and isApplicable answers support questions only. c's
// address may be null: the call then computes d = alpha * (a @ b).
class GemmProblem final {
  public:
    // Takes the six ports' Tensors by value together with the call's
    // attributes. See the class comment for the validation this performs;
    // allocation failure throws std::bad_alloc.
    GemmProblem(Tensor a, Tensor b, Tensor c, Tensor d, Tensor alpha, Tensor beta, GemmAttributes attributes);

    // --- shape ---

    // Returns {m, n, k}: m = a's rows, k = a's columns (= b's rows), and
    // n = b's columns.
    GemmShape getShape() const noexcept {
        const StorageView& a = a_.getStorageView();
        const StorageView& b = b_.getStorageView();
        return GemmShape{static_cast<std::uint64_t>(a.getDims()[0]), static_cast<std::uint64_t>(b.getDims()[1]),
                         static_cast<std::uint64_t>(a.getDims()[1])};
    }

    // Returns whether m * n == 0: the kernel has no work, and no operand
    // address is required.
    bool hasEmptyOutput() const noexcept { return getShape().m * getShape().n == 0; }

    // --- element and compute types ---

    // Returns the call's compute type.
    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }

    FTrainNumericType getANumericType() const noexcept { return a_.getStorageView().getNumericType(); }
    FTrainNumericType getBNumericType() const noexcept { return b_.getStorageView().getNumericType(); }
    FTrainNumericType getCNumericType() const noexcept { return c_.getStorageView().getNumericType(); }
    FTrainNumericType getDNumericType() const noexcept { return d_.getStorageView().getNumericType(); }
    FTrainNumericType getAlphaNumericType() const noexcept { return alpha_.getStorageView().getNumericType(); }
    FTrainNumericType getBetaNumericType() const noexcept { return beta_.getStorageView().getNumericType(); }

    // --- layouts ---

    FTrainIndexType getAIndexType() const noexcept { return a_.getStorageView().getIndexType(); }
    FTrainIndexType getBIndexType() const noexcept { return b_.getStorageView().getIndexType(); }
    FTrainIndexType getCIndexType() const noexcept { return c_.getStorageView().getIndexType(); }
    FTrainIndexType getDIndexType() const noexcept { return d_.getStorageView().getIndexType(); }
    FTrainIndexType getAlphaIndexType() const noexcept { return alpha_.getStorageView().getIndexType(); }
    FTrainIndexType getBetaIndexType() const noexcept { return beta_.getStorageView().getIndexType(); }

    // --- addresses; null only where the kernel skips the port (see
    // hasEmptyOutput and the reduction-depth rule) ---

    void* getAAddress() const noexcept { return a_.getStorageView().getMemory(); }
    void* getBAddress() const noexcept { return b_.getStorageView().getMemory(); }
    void* getCAddress() const noexcept { return c_.getStorageView().getMemory(); }
    void* getDAddress() const noexcept { return d_.getStorageView().getMemory(); }
    void* getAlphaAddress() const noexcept { return alpha_.getStorageView().getMemory(); }
    void* getBetaAddress() const noexcept { return beta_.getStorageView().getMemory(); }

    // --- aliasing ---

    // Returns whether c and d are one tensor (one address): the in-place
    // accumulation d = alpha * (a @ b) + beta * c. This is the only
    // aliasing mode Gemm defines; any other overlap between ports is the
    // caller's responsibility.
    bool isInPlace() const noexcept { return getCAddress() == getDAddress(); }

    // --- selection cache ---

    // Encodes every property that can change which Primitive applies into
    // this problem's key; problems with equal keys must select the same
    // Primitive, because a cache hit skips the applicability check. Each
    // field opens with the next sequential index and every variable-length
    // field carries its element count, so different encodings cannot alias
    // however the values interleave. Memory addresses enter only through
    // properties that affect applicability (alignment classes and in-place
    // aliasing here), never as raw values. The device is added by the
    // engine.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    // One Tensor per semantic port, expanded as named members.
    Tensor a_;
    Tensor b_;
    Tensor c_;
    Tensor d_;
    Tensor alpha_;
    Tensor beta_;
    GemmAttributes attributes_;
};

}  // namespace ftrain

#endif
