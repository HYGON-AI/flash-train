#ifndef FTRAIN_PATTERN_HPP_
#define FTRAIN_PATTERN_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/operation/base.hpp"

namespace ftrain {

class Pattern;

// Identifies one input port of one operation within its owning PatternBuilder.
class PatternOperationInputPortId {
  public:
    constexpr PatternOperationInputPortId(PatternOperationId op_id, std::size_t port_index) noexcept
        : op_id_(op_id), port_index_(port_index) {}

    constexpr PatternOperationId getOpId() const noexcept { return op_id_; }

    constexpr std::size_t getPortIndex() const noexcept { return port_index_; }

    constexpr bool operator==(PatternOperationInputPortId other) const noexcept {
        return op_id_ == other.op_id_ && port_index_ == other.port_index_;
    }
    constexpr bool operator!=(PatternOperationInputPortId other) const noexcept { return !(*this == other); }

  private:
    PatternOperationId op_id_;
    std::size_t port_index_;
};

// Identifies one output port of one operation within its owning PatternBuilder.
class PatternOperationOutputPortId {
  public:
    constexpr PatternOperationOutputPortId(PatternOperationId op_id, std::size_t port_index) noexcept
        : op_id_(op_id), port_index_(port_index) {}

    constexpr PatternOperationId getOpId() const noexcept { return op_id_; }

    constexpr std::size_t getPortIndex() const noexcept { return port_index_; }

    constexpr bool operator==(PatternOperationOutputPortId other) const noexcept {
        return op_id_ == other.op_id_ && port_index_ == other.port_index_;
    }
    constexpr bool operator!=(PatternOperationOutputPortId other) const noexcept { return !(*this == other); }

  private:
    PatternOperationId op_id_;
    std::size_t port_index_;
};

// One operand's topology entry on a Pattern: its kind, the output
// port producing it (if any), and the input ports consuming it. The
// consumers list has no defined order.
class PatternOperandNode final {
  public:
    PatternOperandNode(OperandKind kind, std::optional<PatternOperationOutputPortId>&& producer,
                       std::vector<PatternOperationInputPortId>&& consumers) noexcept
        : kind_(kind), producer_(std::move(producer)), consumers_(std::move(consumers)) {}

    OperandKind getKind() const noexcept { return kind_; }

    const std::optional<PatternOperationOutputPortId>& getProducer() const noexcept { return producer_; }

    const std::vector<PatternOperationInputPortId>& getConsumers() const noexcept { return consumers_; }

  private:
    OperandKind kind_;
    std::optional<PatternOperationOutputPortId> producer_;
    std::vector<PatternOperationInputPortId> consumers_;
};

// A structural fingerprint of a PatternBuilder. Patterns built from the same
// operands and operations produce equal keys regardless of the order in which
// the pieces were added. The key does not reference the source PatternBuilder.
class PatternKey final {
  public:
    // Builds the key from one structural token stream, hashing it once;
    // the key never references the stream afterwards.
    explicit PatternKey(std::vector<std::uint64_t>&& tokens) noexcept;

    PatternKey(const PatternKey&)            = default;
    PatternKey& operator=(const PatternKey&) = default;
    PatternKey(PatternKey&& other) noexcept;
    PatternKey& operator=(PatternKey&& other) noexcept;

    bool operator==(const PatternKey& other) const { return tokens_ == other.tokens_; }

    bool operator!=(const PatternKey& other) const { return !(*this == other); }

    std::size_t getHash() const noexcept { return hash_; }

  private:
    std::vector<std::uint64_t> tokens_;
    std::size_t hash_;
};

class PatternKeyHasher final {
  public:
    std::size_t operator()(const PatternKey& key) const noexcept { return key.getHash(); }
};

// A mutable builder for a compute-graph structure. addOperand and
// addOperation are plain appends: apart from allocation failure they never
// reject, and the accumulated structure is validated once by
// buildPattern(). The builder can be discarded after
// preparation.
class PatternBuilder {
  public:
    // Appends one operand of the storage family named by Kind and returns
    // its typed role ID. Allocation failure throws std::bad_alloc.
    template<OperandKind Kind>
    typename OperandTraits<Kind>::Id addOperand() {
        const std::size_t index = operand_kinds_.size();
        operand_kinds_.push_back(Kind);
        return typename OperandTraits<Kind>::Id{index};
    }

    // Appends one operation of the kind named by Kind, wiring the semantic
    // ports from role_ids, and returns its typed operation ID. The stored
    // IDs are validated when the Pattern is constructed. Allocation
    // failure throws std::bad_alloc.
    template<OperationKind Kind, typename... RoleIds>
    typename OperationTraits<Kind>::Id addOperation(RoleIds&&... role_ids) {
        const std::size_t op_index = op_nodes_.size();
        op_nodes_.push_back(OperationTraits<Kind>::createPatternOperationNode(std::forward<RoleIds>(role_ids)...));
        return typename OperationTraits<Kind>::Id{op_index};
    }

    // Validates the accumulated structure, derives the reverse topology and
    // structural coloring, and returns the verified Pattern. Throws
    // Exception with FTRAIN_STATUS_INVALID_ARGUMENT naming the offending
    // operation and port on violation; this builder is unchanged.
    Pattern buildPattern() const;

  private:
    std::vector<OperandKind> operand_kinds_;
    std::vector<PatternOperationNode> op_nodes_;
};

// The verified, self-describing form of a PatternBuilder: the frozen operand kinds
// and operation nodes plus the derived reverse topology (each operand's
// producer and consumers), the structural coloring, and the PatternKey.
// PatternBuilder::buildPattern() performs one validation pass over the whole
// builder (operand IDs in range; IDs distinct within an operation's inputs
// and outputs; no output that is also an input; at most one producer per
// operand; no dependency cycle among operations) and throws Exception with
// FTRAIN_STATUS_INVALID_ARGUMENT naming the offending operation and port on
// violation. The object never references the builder.
class Pattern final {
  public:
    const PatternKey& getKey() const noexcept { return key_; }

    std::uint64_t getNumOperands() const noexcept { return operand_nodes_.size(); }

    std::uint64_t getNumOps() const noexcept { return op_nodes_.size(); }

    // Returns the operand node identified by operand_id. An out-of-range ID
    // throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    const PatternOperandNode& getOperandNode(PatternOperandId operand_id) const;

    // Returns the operation node identified by op_id. An out-of-range ID
    // throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    const PatternOperationNode& getOpNode(PatternOperationId op_id) const;

    // Returns the structural colors of the operands, indexed by operand
    // index. Structurally identical roles share one color.
    const std::vector<std::uint64_t>& getOperandColors() const noexcept { return operand_colors_; }

    // Returns the structural colors of the operations, indexed by op index.
    const std::vector<std::uint64_t>& getOpColors() const noexcept { return op_colors_; }

  private:
    friend class PatternBuilder;

    Pattern(std::vector<PatternOperandNode>&& operand_nodes, std::vector<PatternOperationNode>&& op_nodes,
            PatternKey&& key, std::vector<std::uint64_t>&& operand_colors,
            std::vector<std::uint64_t>&& op_colors) noexcept;

    std::vector<PatternOperandNode> operand_nodes_;
    std::vector<PatternOperationNode> op_nodes_;
    PatternKey key_;
    std::vector<std::uint64_t> operand_colors_;
    std::vector<std::uint64_t> op_colors_;
};

}  // namespace ftrain

#endif
