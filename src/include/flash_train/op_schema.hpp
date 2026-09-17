#ifndef FTRAIN_OP_SCHEMA_HPP_
#define FTRAIN_OP_SCHEMA_HPP_

#include <cstdint>
#include <variant>

#include "flash_train/common.h"

namespace ftrain {

// The storage family of a Pattern operand: one Tensor, an ordered
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

// Per-operation parameters. The attribute types below store
// already-validated values and perform no validation.
class GemmAttributes final {
  public:
    explicit GemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

class GroupedABCDGemmAttributes final {
  public:
    explicit GroupedABCDGemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

class GroupedBCDGemmAttributes final {
  public:
    explicit GroupedBCDGemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

class GroupedABGemmAttributes final {
  public:
    explicit GroupedABGemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

using OperationAttributes =
    std::variant<GemmAttributes, GroupedABCDGemmAttributes, GroupedBCDGemmAttributes, GroupedABGemmAttributes>;

// Typed operand construction helper. Specializations live in
// flash_train/op_definitions.hpp and add operands to a Pattern through
// addToPattern().
template<OperandKind Kind>
class Operand;

// Typed operation construction helper. Specializations live in
// flash_train/op_definitions.hpp and add operations to a Pattern through
// addToPattern().
template<OperationKind Kind>
class Operation;

}  // namespace ftrain

#endif
