#ifndef FTRAIN_OPS_ARGS_HPP_
#define FTRAIN_OPS_ARGS_HPP_

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/operation/operation.hpp"

namespace ftrain {

// One whole-structure match: for each user PatternBuilder operand (operation) ID,
// the corresponding supported PatternBuilder operand (operation) ID. Meaningful only
// with the Patterns passed to the producing call; the object retains neither.
// Out-of-range IDs throw Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
class MatchResult final {
  public:
    std::size_t getNumMappedOperands() const noexcept { return supported_operand_ids_by_user_operand_.size(); }

    std::size_t getNumMappedOps() const noexcept { return supported_op_ids_by_user_op_.size(); }

    // Returns the supported PatternBuilder operand assigned to user_operand_id. An
    // out-of-range ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    PatternOperandId getSupportedOperandId(PatternOperandId user_operand_id) const;

    // Returns the supported PatternBuilder operation assigned to user_op_id. An
    // out-of-range ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    PatternOperationId getSupportedOpId(PatternOperationId user_op_id) const;

    // Moves the mapped IDs out; this MatchResult holds none afterwards.
    std::vector<PatternOperandId> takeSupportedOperandIdsByUserOperand() noexcept {
        return std::move(supported_operand_ids_by_user_operand_);
    }

    // Moves the mapped IDs out; this MatchResult holds none afterwards.
    std::vector<PatternOperationId> takeSupportedOpIdsByUserOp() noexcept {
        return std::move(supported_op_ids_by_user_op_);
    }

  private:
    friend class Matcher;

    MatchResult(std::vector<PatternOperandId>&& supported_operand_ids_by_user_operand,
                std::vector<PatternOperationId>&& supported_op_ids_by_user_op) noexcept
        : supported_operand_ids_by_user_operand_(std::move(supported_operand_ids_by_user_operand)),
          supported_op_ids_by_user_op_(std::move(supported_op_ids_by_user_op)) {}

    std::vector<PatternOperandId> supported_operand_ids_by_user_operand_;
    std::vector<PatternOperationId> supported_op_ids_by_user_op_;
};

class Matcher final {
  public:
    Matcher() = delete;

    // Returns a complete one-to-one mapping from user_pattern's operands and
    // operations onto supported_pattern's, or std::nullopt when the structures
    // are not exactly equivalent. Insertion order and consumer-list order are
    // ignored; input and output port order is significant. Allocation failure
    // throws std::bad_alloc.
    // Fast correspondence for structures with equal keys: uniquely-colored
    // roles pair directly, equal-color classes pair in stable index order,
    // and the induced bijection is verified against both Patterns' wiring.
    // Returns std::nullopt when verification fails; the caller falls back to
    // match(). Equal keys alone do not guarantee isomorphic structures.
    static std::optional<MatchResult> matchBySignature(const Pattern& user_pattern, const Pattern& supported_pattern);

    static std::optional<MatchResult> match(const Pattern& user_pattern, const Pattern& supported_pattern);
};

// A one-to-one mapping between one user PatternBuilder's roles and the corresponding
// supported PatternBuilder's roles. Out-of-range IDs throw Exception with
// FTRAIN_STATUS_INVALID_ARGUMENT.
class RoleMapping final {
  public:
    // Adopts the user-to-supported correspondence produced by Matcher::match()
    // or matchBySignature(); the source MatchResult is left empty. Allocation
    // failure throws std::bad_alloc.
    static RoleMapping fromMatchResult(MatchResult&& result);

    std::uint64_t getNumOperands() const noexcept { return supported_operand_ids_by_user_operand_.size(); }

    std::uint64_t getNumOps() const noexcept { return supported_op_ids_by_user_op_.size(); }

    // Maps a user operand ID to its supported operand ID.
    PatternOperandId getSupportedOperandId(PatternOperandId user_operand_id) const;

    // Maps a user operation ID to its supported operation ID.
    PatternOperationId getSupportedOpId(PatternOperationId user_op_id) const;

  private:
    RoleMapping(std::vector<PatternOperandId>&& supported_operand_ids_by_user_operand,
                std::vector<PatternOperationId>&& supported_op_ids_by_user_op) noexcept
        : supported_operand_ids_by_user_operand_(std::move(supported_operand_ids_by_user_operand)),
          supported_op_ids_by_user_op_(std::move(supported_op_ids_by_user_op)) {}

    std::vector<PatternOperandId> supported_operand_ids_by_user_operand_;
    std::vector<PatternOperationId> supported_op_ids_by_user_op_;
};

// The immutable result of a successful match: the PatternKey and the complete
// user-to-supported role mapping. Constructing from the user Pattern and the
// supported Pattern runs the match and throws Exception with
// FTRAIN_STATUS_UNSUPPORTED when the structures do not match. Retains
// neither source Pattern, so both may be destroyed after construction.
// Allocation failure throws std::bad_alloc.
class Args;

class Ops final {
  public:
    Ops(const Pattern& user_pattern, const Pattern& supported_pattern);

    const PatternKey& getPatternKey() const noexcept { return pattern_key_; }

    const RoleMapping& getRoleMapping() const noexcept { return role_mapping_; }

    // Returns Args for this Ops with every operand and operation slot unset;
    // slot kinds follow this Ops' supported schema. Allocation failure
    // throws std::bad_alloc.
    Args makeArgs() const;

  private:
    Ops(PatternKey&& pattern_key, RoleMapping&& role_mapping) noexcept
        : pattern_key_(std::move(pattern_key)), role_mapping_(std::move(role_mapping)) {}

    PatternKey pattern_key_;
    RoleMapping role_mapping_;
    std::vector<OperandKind> supported_operand_kinds_;
    std::vector<OperationKind> supported_op_kinds_;
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
    const std::optional<Operand>& getOperand(PatternOperandId supported_operand_id) const;

    // Returns the attributes stored at supported_op_id (a supported-role ID,
    // not a user PatternBuilder ID); an unset slot holds no value. An
    // out-of-range ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    const std::optional<OperationAttributes>& getOpArgument(PatternOperationId supported_op_id) const;

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
        setOp(user_op_id, Kind, OperationAttributes{std::move(attributes)});
    }

  private:
    friend class Ops;

    // Fills one unset slot per supported role; invoked by Ops::makeArgs().
    Args(PatternKey&& pattern_key, RoleMapping&& role_mapping, const std::vector<OperandKind>& supported_operand_kinds,
         const std::vector<OperationKind>& supported_op_kinds);

    std::size_t mapUserOperand(PatternOperandId user_operand_id, OperandKind expected_kind) const;
    std::size_t mapUserOp(PatternOperationId user_op_id, OperationKind expected_kind) const;
    void setOp(PatternOperationId user_op_id, OperationKind expected_kind, OperationAttributes&& attributes);

    PatternKey pattern_key_;
    RoleMapping role_mapping_;
    std::vector<OperandKind> supported_operand_kinds_;
    std::vector<OperationKind> supported_op_kinds_;
    std::vector<std::optional<Operand>> operands_;
    std::vector<std::optional<OperationAttributes>> op_arguments_;
};

}  // namespace ftrain

#endif
