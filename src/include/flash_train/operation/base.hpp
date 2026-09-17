#ifndef FTRAIN_OPERATION_BASE_HPP_
#define FTRAIN_OPERATION_BASE_HPP_

#include <cstdint>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"

namespace ftrain {

// The storage family of a PatternBuilder operand: one Tensor, an ordered
// TensorList, or a GroupedTensor.
enum class OperandKind : std::uint8_t {
    kTensor,
    kTensorList,
    kGroupedTensor,
};

// The normalized internal operation kinds.
enum class OperationKind : std::uint16_t {
    kGemm,
    kGroupedABCDGemm,
    kGroupedBCDGemm,
    kGroupedABGemm,
};

// Identifies an operand within its owning PatternBuilder.
class PatternOperandId {
  public:
    explicit constexpr PatternOperandId(std::uint64_t index) noexcept : index_(index) {}

    constexpr std::uint64_t getIndex() const noexcept { return index_; }

    constexpr bool operator==(PatternOperandId other) const noexcept { return index_ == other.index_; }
    constexpr bool operator!=(PatternOperandId other) const noexcept { return !(*this == other); }

  private:
    std::uint64_t index_;
};

// Identifies an operation within its owning PatternBuilder.
class PatternOperationId {
  public:
    explicit constexpr PatternOperationId(std::uint64_t index) noexcept : index_(index) {}

    constexpr std::uint64_t getIndex() const noexcept { return index_; }

    constexpr bool operator==(PatternOperationId other) const noexcept { return index_ == other.index_; }
    constexpr bool operator!=(PatternOperationId other) const noexcept { return !(*this == other); }

  private:
    std::uint64_t index_;
};

// Typed operand helper. Specializations for every OperandKind are defined
// in this header. Type names the filled argument slot that
// Args::setOperand<Kind> accepts for this storage family.
template<OperandKind Kind>
class Operand;

// Typed operation helper. Specializations live in
// flash_train/operation/gemm.hpp. Type names the per-invocation attributes
// that Args::setOperation<Kind> accepts for this operation kind.
template<OperationKind Kind>
class Operation;

template<>
class Operand<OperandKind::kTensor> {
  public:
    using Id   = FTrainTensorId;
    using Type = Tensor;

    // Converts one Tensor role ID into its PatternBuilder operand ID.
    static constexpr PatternOperandId getPatternOperandId(Id id) noexcept { return PatternOperandId{id.opaque}; }
};

template<>
class Operand<OperandKind::kTensorList> {
  public:
    using Id   = FTrainTensorListId;
    using Type = TensorList;

    // Converts one TensorList role ID into its PatternBuilder operand ID.
    static constexpr PatternOperandId getPatternOperandId(Id id) noexcept { return PatternOperandId{id.opaque}; }
};

template<>
class Operand<OperandKind::kGroupedTensor> {
  public:
    using Id   = FTrainGroupedTensorId;
    using Type = GroupedTensor;

    // Converts one GroupedTensor role ID into its PatternBuilder operand ID.
    static constexpr PatternOperandId getPatternOperandId(Id id) noexcept { return PatternOperandId{id.opaque}; }
};

}  // namespace ftrain

#endif
