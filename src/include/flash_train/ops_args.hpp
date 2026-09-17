#ifndef FTRAIN_OPS_ARGS_HPP_
#define FTRAIN_OPS_ARGS_HPP_

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/operation/operation.hpp"

namespace ftrain {

class Args;

// The immutable result of a successful match: the PatternKey and the complete
// user-to-supported role mapping. Constructing from the user Pattern and the
// supported Pattern runs the match and throws Exception with
// FTRAIN_STATUS_UNSUPPORTED when the structures do not match. Retains
// neither source Pattern, so both may be destroyed after construction.
// Allocation failure throws std::bad_alloc.
class Ops final {
  public:
    // Matches user_pattern against supported_pattern: operand, operation, and
    // consumer insertion order is ignored, input and output port order is
    // significant. Throws Exception with FTRAIN_STATUS_UNSUPPORTED when the
    // structures do not match.
    Ops(const Pattern& user_pattern, const Pattern& supported_pattern);

    const PatternKey& getPatternKey() const noexcept { return pattern_key_; }

    // Returns Args for this Ops with every operand and operation slot unset;
    // slot kinds follow this Ops' supported schema. Allocation failure
    // throws std::bad_alloc.
    Args makeArgs() const;

  private:
    PatternKey pattern_key_;
    std::vector<PatternOperandId> supported_operand_ids_by_user_operand_;
    std::vector<PatternOperationId> supported_op_ids_by_user_op_;
    std::vector<OperandKind> operand_kinds_;
    std::vector<OperationKind> operation_kinds_;
};

// One invocation's concrete parameters, addressed by user PatternBuilder IDs.
// Created by Ops::makeArgs() with every slot unset; setters accept any order
// and replace previous values. Tensor memory stays non-owning (see
// StorageView). Concurrent access to one Args requires external
// synchronization when any access modifies it.
class Args final {
  public:
    const PatternKey& getPatternKey() const noexcept { return pattern_key_; }

    std::uint64_t getNumOperands() const noexcept { return operands_.size(); }

    std::uint64_t getNumOps() const noexcept { return op_arguments_.size(); }

    // Returns true once every operand and operation slot has been set.
    bool isComplete() const noexcept;

    // Returns the operand slot stored at supported_operand_id (a supported-role
    // ID, not a user PatternBuilder ID); an unset slot holds no value. An
    // out-of-range ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    const std::optional<OperandValue>& getOperand(PatternOperandId supported_operand_id) const;

    // Returns the attributes stored at supported_op_id (a supported-role ID,
    // not a user PatternBuilder ID); an unset slot holds no value. An
    // out-of-range ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    const std::optional<OperationValue>& getOpArgument(PatternOperationId supported_op_id) const;

    // Setters take user PatternBuilder IDs and move the value into the mapped
    // slot, replacing any previous value. setOperand<Kind> accepts the
    // argument value named by OperandTraits<Kind>::Type; setOperation<Kind>
    // accepts the attributes named by OperationTraits<Kind>::Type. Throws
    // Exception with FTRAIN_STATUS_INVALID_ARGUMENT when the ID is out of
    // range or the mapped slot's kind does not match Kind. This Args is
    // unchanged on failure.
    template<OperandKind Kind>
    void setOperand(PatternOperandId user_operand_id, typename OperandTraits<Kind>::Type&& operand) {
        const std::size_t supported_operand_index = mapUserOperand(user_operand_id, Kind);
        operands_[supported_operand_index]        = std::move(operand);
    }

    template<OperationKind Kind>
    void setOperation(PatternOperationId user_op_id, typename OperationTraits<Kind>::Type&& attributes) {
        setOp(user_op_id, Kind, OperationValue{std::move(attributes)});
    }

  private:
    friend class Ops;

    // Copies one matched schema's snapshot into unset slots; invoked by
    // Ops::makeArgs().
    Args(const PatternKey& pattern_key, const std::vector<PatternOperandId>& supported_operand_ids_by_user_operand,
         const std::vector<PatternOperationId>& supported_op_ids_by_user_op,
         const std::vector<OperandKind>& operand_kinds, const std::vector<OperationKind>& operation_kinds);

    std::size_t mapUserOperand(PatternOperandId user_operand_id, OperandKind expected_kind) const;
    std::size_t mapUserOp(PatternOperationId user_op_id, OperationKind expected_kind) const;
    void setOp(PatternOperationId user_op_id, OperationKind expected_kind, OperationValue&& attributes);

    PatternKey pattern_key_;
    std::vector<PatternOperandId> supported_operand_ids_by_user_operand_;
    std::vector<PatternOperationId> supported_op_ids_by_user_op_;
    std::vector<OperandKind> operand_kinds_;
    std::vector<OperationKind> operation_kinds_;
    std::vector<std::optional<OperandValue>> operands_;
    std::vector<std::optional<OperationValue>> op_arguments_;
};

}  // namespace ftrain

#endif
