#ifndef FTRAIN_PATTERN_HPP_
#define FTRAIN_PATTERN_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/operation/base.hpp"

namespace ftrain {

class PatternBuilder;
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
    explicit PatternOperandNode(OperandKind kind) noexcept : kind_(kind) {}

    OperandKind getKind() const noexcept { return kind_; }

    const std::optional<PatternOperationOutputPortId>& getProducer() const noexcept { return producer_; }

    const std::vector<PatternOperationInputPortId>& getConsumers() const noexcept { return consumers_; }

  private:
    friend class Pattern;

    OperandKind kind_;
    std::optional<PatternOperationOutputPortId> producer_;
    std::vector<PatternOperationInputPortId> consumers_;
};

// One operation's topology entry on a Pattern: its kind plus
// ordered input and output operand IDs. Input and output port order is
// significant; the order in which operations were added is not.
class PatternOperationNode final {
  public:
    PatternOperationNode(OperationKind kind, std::vector<PatternOperandId>&& inputs,
                         std::vector<PatternOperandId>&& outputs) noexcept
        : kind_(kind), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {}

    OperationKind getKind() const noexcept { return kind_; }

    const std::vector<PatternOperandId>& getInputs() const noexcept { return inputs_; }

    const std::vector<PatternOperandId>& getOutputs() const noexcept { return outputs_; }

  private:
    OperationKind kind_;
    std::vector<PatternOperandId> inputs_;
    std::vector<PatternOperandId> outputs_;
};

// A structural fingerprint of a PatternBuilder. Patterns built from the same
// operands and operations produce equal keys regardless of the order in which
// the pieces were added. The key does not reference the source PatternBuilder.
class PatternKey final {
  public:
    PatternKey(const PatternKey&)            = default;
    PatternKey& operator=(const PatternKey&) = default;
    PatternKey(PatternKey&& other) noexcept;
    PatternKey& operator=(PatternKey&& other) noexcept;

    bool operator==(const PatternKey& other) const { return tokens_ == other.tokens_; }

    bool operator!=(const PatternKey& other) const { return !(*this == other); }

    std::size_t getHash() const noexcept { return hash_; }

  private:
    friend class Pattern;

    explicit PatternKey(std::vector<std::uint64_t>&& tokens) noexcept;

    std::vector<std::uint64_t> tokens_;
    std::size_t hash_;
};

class PatternKeyHasher final {
  public:
    std::size_t operator()(const PatternKey& key) const noexcept { return key.getHash(); }
};

// A mutable builder for a compute-graph structure. addOperand and
// addOperation are plain appends: apart from allocation failure they never
// reject, and the accumulated structure is validated once by the
// Pattern constructor. The builder can be discarded after
// preparation.
class PatternBuilder {
  public:
    // Appends one operand of the storage family named by Kind and returns
    // its typed role ID. Allocation failure throws std::bad_alloc.
    template<OperandKind Kind>
    typename Operand<Kind>::Id addOperand() {
        return typename Operand<Kind>::Id{appendOperand(Kind).getIndex()};
    }

    std::uint64_t getNumOperands() const noexcept { return operand_kinds_.size(); }

    // Returns the storage family of the operand identified by operand_id.
    // An out-of-range ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    OperandKind getOperandKind(PatternOperandId operand_id) const;

    // Appends one operation of the kind named by Kind with the semantic
    // ports described by topology and returns its typed operation ID. The
    // stored IDs are validated when the Pattern is constructed.
    // Allocation failure throws std::bad_alloc.
    template<OperationKind Kind>
    typename Operation<Kind>::Id addOperation(const typename Operation<Kind>::Topology& topology) {
        std::vector<PatternOperandId> inputs  = Operation<Kind>::getInputPatternIds(topology);
        std::vector<PatternOperandId> outputs = Operation<Kind>::getOutputPatternIds(topology);
        return typename Operation<Kind>::Id{appendOperation(Kind, std::move(inputs), std::move(outputs)).getIndex()};
    }

    std::uint64_t getNumOps() const noexcept { return op_nodes_.size(); }

    // Validates the accumulated structure, derives the reverse topology and
    // structural coloring, and returns the verified Pattern. Throws
    // Exception with FTRAIN_STATUS_INVALID_ARGUMENT naming the offending
    // operation and port on violation; this builder is unchanged.
    Pattern buildPattern() const;

  private:
    friend class Pattern;

    PatternOperandId appendOperand(OperandKind kind);

    PatternOperationId appendOperation(OperationKind kind, std::vector<PatternOperandId>&& inputs,
                                       std::vector<PatternOperandId>&& outputs);

    std::vector<OperandKind> operand_kinds_;
    std::vector<PatternOperationNode> op_nodes_;
};

// The verified, self-describing form of a PatternBuilder: the frozen operand kinds
// and operation nodes plus the derived reverse topology (each operand's
// producer and consumers), the structural coloring, and the PatternKey.
// Construction performs one validation pass over the whole builder (operand
// IDs in range; IDs distinct within an operation's inputs and outputs; no
// output that is also an input; at most one producer per operand) and
// throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT naming the offending
// operation and port on violation. The object never references the builder.
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

    struct Parts {
        std::vector<PatternOperandNode> operand_nodes;
        std::vector<PatternOperationNode> op_nodes;
        PatternKey key;
        std::vector<std::uint64_t> operand_colors;
        std::vector<std::uint64_t> op_colors;
    };

    // Validates the builder's stored operand IDs once and derives each
    // operand's producer and consumers.
    static void validateAndDeriveTopology(const PatternBuilder& pattern, std::vector<PatternOperandNode>& operand_nodes,
                                          const std::vector<PatternOperationNode>& op_nodes);

    static Parts makeParts(const PatternBuilder& pattern);

    Pattern(Parts&& parts) noexcept;

    std::vector<PatternOperandNode> operand_nodes_;
    std::vector<PatternOperationNode> op_nodes_;
    PatternKey key_;
    std::vector<std::uint64_t> operand_colors_;
    std::vector<std::uint64_t> op_colors_;
};

}  // namespace ftrain

#endif
