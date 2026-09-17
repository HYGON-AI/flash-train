#ifndef FTRAIN_RUNTIME_HPP_
#define FTRAIN_RUNTIME_HPP_

#include <cstddef>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "flash_train/matcher.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {

// A one-to-one mapping between one user Pattern's roles and the corresponding
// supported Pattern's roles. The four accessors are paired inverses;
// out-of-range IDs throw Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
class RoleMapping final {
  public:
    // Builds the mapping from two canonicalizations of the same structure.
    // Throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT when the two
    // PatternKeys differ or the role counts differ. Allocation failure throws
    // std::bad_alloc.
    static RoleMapping compose(const PatternCanonicalization& user_canonicalization,
                               const PatternCanonicalization& supported_canonicalization);

    std::uint64_t getNumOperands() const noexcept { return supported_operand_indices_by_user_operand_.size(); }

    std::uint64_t getNumOps() const noexcept { return supported_op_indices_by_user_op_.size(); }

    // Maps a user operand ID to its supported operand ID.
    OperandId getSupportedOperandId(OperandId user_operand_id) const;

    // Maps a supported operand ID back to its user operand ID.
    OperandId getUserOperandId(OperandId supported_operand_id) const;

    // Maps a user operation ID to its supported operation ID.
    OperationId getSupportedOpId(OperationId user_op_id) const;

    // Maps a supported operation ID back to its user operation ID.
    OperationId getUserOpId(OperationId supported_op_id) const;

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

// The supported Pattern's operand and operation kinds, copied in construction
// order. Out-of-range IDs throw Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
class SupportedSchema final {
  public:
    // Copies the role kinds from supported_pattern. Allocation failure throws
    // std::bad_alloc.
    explicit SupportedSchema(const Pattern& supported_pattern);

    std::uint64_t getNumOperands() const noexcept { return operand_kinds_.size(); }

    std::uint64_t getNumOps() const noexcept { return op_kinds_.size(); }

    // Returns the kind of the supported operand role.
    OperandKind getOperandKind(OperandId supported_operand_id) const;

    // Returns the kind of the supported operation role.
    OperationKind getOperationKind(OperationId supported_op_id) const;

  private:
    std::vector<OperandKind> operand_kinds_;
    std::vector<OperationKind> op_kinds_;
};

// The immutable result of a successful match: the PatternKey and the complete
// user-to-supported role mapping. Retains neither source Pattern, so both may
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

// One invocation's concrete parameters, addressed by user Pattern IDs. Every
// operand and operation slot starts unset; setters accept any order and
// replace previous values. Tensor memory stays non-owning (see StorageView).
// Concurrent access to one Args requires external synchronization when any
// access modifies it.
class Args final {
  public:
    // Creates Args with every slot unset. Throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT when the schema's role counts differ from
    // the Ops mapping's counts. Allocation failure throws std::bad_alloc.
    Args(const Ops& ops, const SupportedSchema& supported_schema);

    const PatternKey& getPatternKey() const noexcept { return pattern_key_; }

    std::uint64_t getNumOperands() const noexcept { return operands_.size(); }

    std::uint64_t getNumOps() const noexcept { return op_arguments_.size(); }

    // Returns true once every operand and operation slot has been set.
    bool isComplete() const noexcept;

    // Returns the operand slot stored at supported_operand_id (a supported-role
    // ID, not a user Pattern ID). An out-of-range ID throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT.
    const OpsOperand& getOperand(OperandId supported_operand_id) const;

    // Returns the operation slot stored at supported_op_id (a supported-role
    // ID, not a user Pattern ID). An out-of-range ID throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT.
    const OpArgument& getOpArgument(OperationId supported_op_id) const;

    // Setters take user Pattern IDs and move the value into the mapped slot,
    // replacing any previous value. Throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT when the ID is out of range, the operand
    // slot's storage family does not match the setter, or the operation slot's
    // kind does not match the setter. This Args is unchanged on failure.
    void setTensor(OperandId user_operand_id, TensorStorage&& storage);
    void setTensorList(OperandId user_operand_id, TensorListStorage&& storage);
    void setGroupedTensor(OperandId user_operand_id, GroupedTensorStorage&& storage);

    void setGroupedABCDGemm(OperationId user_op_id, GroupedABCDGemmAttributes&& attributes);
    void setGemm(OperationId user_op_id, GemmAttributes&& attributes);
    void setGroupedBCDGemm(OperationId user_op_id, GroupedBCDGemmAttributes&& attributes);
    void setGroupedABGemm(OperationId user_op_id, GroupedABGemmAttributes&& attributes);

  private:
    std::size_t mapUserOperand(OperandId user_operand_id, OperandKind expected_kind) const;
    std::size_t mapUserOp(OperationId user_op_id, OperationKind expected_kind) const;
    void setOp(OperationId user_op_id, OperationKind expected_kind, OperationAttributes&& attributes);

    PatternKey pattern_key_;
    std::vector<std::size_t> supported_operand_indices_by_user_operand_;
    std::vector<std::size_t> supported_op_indices_by_user_op_;
    std::vector<OpsOperand> operands_;
    std::vector<OpArgument> op_arguments_;
};

}  // namespace ftrain

#endif
