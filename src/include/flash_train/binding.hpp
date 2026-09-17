#ifndef FTRAIN_BINDING_HPP_
#define FTRAIN_BINDING_HPP_

#include <cstddef>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/matcher.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {

// A one-to-one mapping between one user PatternBuilder's roles and the corresponding
// supported PatternBuilder's roles. The four accessors are paired inverses;
// out-of-range IDs throw Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
class RoleMapping final {
  public:
    // Builds the bidirectional mapping from one complete user-to-supported
    // correspondence produced by Matcher::match() or matchBySignature().
    // Allocation failure throws std::bad_alloc.
    static RoleMapping fromMatchResult(const MatchResult& result);

    std::uint64_t getNumOperands() const noexcept { return supported_operand_indices_by_user_operand_.size(); }

    std::uint64_t getNumOps() const noexcept { return supported_op_indices_by_user_op_.size(); }

    // Maps a user operand ID to its supported operand ID.
    PatternOperandId getSupportedOperandId(PatternOperandId user_operand_id) const;

    // Maps a supported operand ID back to its user operand ID.
    PatternOperandId getUserOperandId(PatternOperandId supported_operand_id) const;

    // Maps a user operation ID to its supported operation ID.
    PatternOperationId getSupportedOpId(PatternOperationId user_op_id) const;

    // Maps a supported operation ID back to its user operation ID.
    PatternOperationId getUserOpId(PatternOperationId supported_op_id) const;

  private:
    RoleMapping(std::vector<std::size_t>&& supported_operand_indices_by_user_operand,
                std::vector<std::size_t>&& user_operand_indices_by_supported_operand,
                std::vector<std::size_t>&& supported_op_indices_by_user_op,
                std::vector<std::size_t>&& user_op_indices_by_supported_op) noexcept
        : supported_operand_indices_by_user_operand_(std::move(supported_operand_indices_by_user_operand)),
          user_operand_indices_by_supported_operand_(std::move(user_operand_indices_by_supported_operand)),
          supported_op_indices_by_user_op_(std::move(supported_op_indices_by_user_op)),
          user_op_indices_by_supported_op_(std::move(user_op_indices_by_supported_op)) {}

    std::vector<std::size_t> supported_operand_indices_by_user_operand_;
    std::vector<std::size_t> user_operand_indices_by_supported_operand_;
    std::vector<std::size_t> supported_op_indices_by_user_op_;
    std::vector<std::size_t> user_op_indices_by_supported_op_;
};

// The immutable result of a successful match: the PatternKey and the complete
// user-to-supported role mapping. Retains neither source PatternBuilder, so both may
// be destroyed after construction.
class Ops final {
  public:
    Ops(PatternKey pattern_key, RoleMapping role_mapping) noexcept
        : pattern_key_(std::move(pattern_key)), role_mapping_(std::move(role_mapping)) {}

    const PatternKey& getPatternKey() const noexcept { return pattern_key_; }

    const RoleMapping& getRoleMapping() const noexcept { return role_mapping_; }

  private:
    PatternKey pattern_key_;
    RoleMapping role_mapping_;
};

using OpsOperand = std::variant<Tensor, TensorList, GroupedTensor>;

// Maps each OpsOperand alternative to its OperandKind. An alternative without
// a mapping fails to compile in generic runtime code.
template<typename OperandType>
struct OpsOperandTraits;

template<>
struct OpsOperandTraits<Tensor> {
    static constexpr OperandKind kKind = OperandKind::kTensor;
};

template<>
struct OpsOperandTraits<TensorList> {
    static constexpr OperandKind kKind = OperandKind::kTensorList;
};

template<>
struct OpsOperandTraits<GroupedTensor> {
    static constexpr OperandKind kKind = OperandKind::kGroupedTensor;
};

// One operation argument slot: the operation's kind plus its attributes once
// set. getAttributes() throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT
// while unset.
class OpArgument final {
  public:
    explicit OpArgument(OperationKind kind) noexcept : kind_(kind) {}

    OperationKind getKind() const noexcept { return kind_; }

    bool isSet() const noexcept { return attributes_.has_value(); }

    // Returns the concrete attributes. An unset role throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT.
    const OperationAttributes& getAttributes() const;

  private:
    friend class Args;

    void setAttributes(OperationAttributes&& attributes) { attributes_.emplace(std::move(attributes)); }

    OperationKind kind_;
    std::optional<OperationAttributes> attributes_;
};

// One invocation's concrete parameters, addressed by user PatternBuilder IDs. Every
// operand and operation slot starts unset; setters accept any order and
// replace previous values. Tensor memory stays non-owning (see StorageView).
// Concurrent access to one Args requires external synchronization when any
// access modifies it.
class Args final {
  public:
    // Creates Args with every slot unset. supported_pattern must be the
    // supported PatternBuilder of the engine that produced ops. Throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT when its role counts differ from the Ops
    // mapping's counts. Allocation failure throws std::bad_alloc.
    Args(const Ops& ops, const Pattern& supported_pattern);

    const PatternKey& getPatternKey() const noexcept { return pattern_key_; }

    std::uint64_t getNumOperands() const noexcept { return operands_.size(); }

    std::uint64_t getNumOps() const noexcept { return op_arguments_.size(); }

    // Returns true once every operand and operation slot has been set.
    bool isComplete() const noexcept;

    // Returns the operand slot stored at supported_operand_id (a supported-role
    // ID, not a user PatternBuilder ID). An out-of-range ID throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT.
    const OpsOperand& getOperand(PatternOperandId supported_operand_id) const;

    // Returns the operation slot stored at supported_op_id (a supported-role
    // ID, not a user PatternBuilder ID). An out-of-range ID throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT.
    const OpArgument& getOpArgument(PatternOperationId supported_op_id) const;

    // Setters take user PatternBuilder IDs and move the value into the mapped
    // slot, replacing any previous value. setOperand<Kind> accepts the filled
    // argument slot named by OperandTraits<Kind>::Type; setOperation<Kind> accepts
    // the attributes named by OperationTraits<Kind>::Type. Throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT when the ID is out of range, the mapped
    // slot's kind does not match Kind, or the operand slot is unset. This
    // Args is unchanged on failure.
    template<OperandKind Kind>
    void setOperand(PatternOperandId user_operand_id, typename OperandTraits<Kind>::Type&& operand) {
        const std::size_t supported_operand_index = mapUserOperand(user_operand_id, Kind);
        if (!operand.isSet()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User Operand role %zu argument has not been set",
                            user_operand_id.getIndex());
        }
        operands_[supported_operand_index] = std::move(operand);
    }

    template<OperationKind Kind>
    void setOperation(PatternOperationId user_op_id, typename OperationTraits<Kind>::Type&& attributes) {
        setOp(user_op_id, Kind, OperationAttributes{std::move(attributes)});
    }

  private:
    std::size_t mapUserOperand(PatternOperandId user_operand_id, OperandKind expected_kind) const;
    std::size_t mapUserOp(PatternOperationId user_op_id, OperationKind expected_kind) const;
    void setOp(PatternOperationId user_op_id, OperationKind expected_kind, OperationAttributes&& attributes);

    PatternKey pattern_key_;
    std::vector<std::size_t> supported_operand_indices_by_user_operand_;
    std::vector<std::size_t> supported_op_indices_by_user_op_;
    std::vector<OpsOperand> operands_;
    std::vector<OpArgument> op_arguments_;
};

}  // namespace ftrain

#endif
