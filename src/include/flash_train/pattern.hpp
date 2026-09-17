#ifndef FTRAIN_PATTERN_HPP_
#define FTRAIN_PATTERN_HPP_

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/op_schema.hpp"

namespace ftrain {

class Pattern;

// Identifies an operand within its owning Pattern.
class OperandId {
  public:
    explicit constexpr OperandId(std::uint64_t index) noexcept : index_(index) {}

    constexpr std::uint64_t getIndex() const noexcept { return index_; }

    constexpr bool operator==(OperandId other) const noexcept { return index_ == other.index_; }
    constexpr bool operator!=(OperandId other) const noexcept { return !(*this == other); }

  private:
    std::uint64_t index_;
};

// Identifies an operation within its owning Pattern.
class OperationId {
  public:
    explicit constexpr OperationId(std::uint64_t index) noexcept : index_(index) {}

    constexpr std::uint64_t getIndex() const noexcept { return index_; }

    constexpr bool operator==(OperationId other) const noexcept { return index_ == other.index_; }
    constexpr bool operator!=(OperationId other) const noexcept { return !(*this == other); }

  private:
    std::uint64_t index_;
};

// Identifies one input port of one operation within its owning Pattern.
class OperationInputPortId {
  public:
    constexpr OperationInputPortId(OperationId op_id, std::size_t port_index) noexcept
        : op_id_(op_id), port_index_(port_index) {}

    constexpr OperationId getOpId() const noexcept { return op_id_; }

    constexpr std::size_t getPortIndex() const noexcept { return port_index_; }

    constexpr bool operator==(OperationInputPortId other) const noexcept {
        return op_id_ == other.op_id_ && port_index_ == other.port_index_;
    }
    constexpr bool operator!=(OperationInputPortId other) const noexcept { return !(*this == other); }

  private:
    OperationId op_id_;
    std::size_t port_index_;
};

// Identifies one output port of one operation within its owning Pattern.
class OperationOutputPortId {
  public:
    constexpr OperationOutputPortId(OperationId op_id, std::size_t port_index) noexcept
        : op_id_(op_id), port_index_(port_index) {}

    constexpr OperationId getOpId() const noexcept { return op_id_; }

    constexpr std::size_t getPortIndex() const noexcept { return port_index_; }

    constexpr bool operator==(OperationOutputPortId other) const noexcept {
        return op_id_ == other.op_id_ && port_index_ == other.port_index_;
    }
    constexpr bool operator!=(OperationOutputPortId other) const noexcept { return !(*this == other); }

  private:
    OperationId op_id_;
    std::size_t port_index_;
};

// One operand's topology entry: its kind, the output port producing it (if
// any), and the input ports consuming it. The consumers list has no defined
// order.
class PatternOperandNode final {
  public:
    explicit PatternOperandNode(OperandKind kind) noexcept : kind_(kind) {}

    OperandKind getKind() const noexcept { return kind_; }

    const std::optional<OperationOutputPortId>& getProducer() const noexcept { return producer_; }

    const std::vector<OperationInputPortId>& getConsumers() const noexcept { return consumers_; }

  private:
    friend class Pattern;

    OperandKind kind_;
    std::optional<OperationOutputPortId> producer_;
    std::vector<OperationInputPortId> consumers_;
};

// One operation's topology entry: its kind plus ordered input and output
// operand IDs. Input and output port order is significant; the order in which
// operations were added to the Pattern is not.
class PatternOpNode final {
  public:
    PatternOpNode(OperationKind kind, std::vector<OperandId>&& inputs, std::vector<OperandId>&& outputs) noexcept
        : kind_(kind), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {}

    OperationKind getKind() const noexcept { return kind_; }

    const std::vector<OperandId>& getInputs() const noexcept { return inputs_; }

    const std::vector<OperandId>& getOutputs() const noexcept { return outputs_; }

  private:
    OperationKind kind_;
    std::vector<OperandId> inputs_;
    std::vector<OperandId> outputs_;
};

// A mutable compute-graph structure of operands and operations, each
// identified by an index. Build it through the typed helpers in
// flash_train/op_definitions.hpp. References returned by getters stay valid
// until the next addOperand/addOperation call or destruction.
class Pattern {
  public:
    std::uint64_t getNumOperands() const noexcept { return operand_nodes_.size(); }

    // Returns the operand node identified by operand_id. An out-of-range ID throws
    // Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    const PatternOperandNode& getOperandNode(OperandId operand_id) const;

    std::uint64_t getNumOps() const noexcept { return op_nodes_.size(); }

    // Returns the operation node identified by op_id. An out-of-range ID throws
    // Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    const PatternOpNode& getOpNode(OperationId op_id) const;

  private:
    template<OperandKind Kind>
    friend class Operand;
    template<OperationKind Kind>
    friend class Operation;

    // Adds one disconnected operand of kind and returns its ID. The Pattern is
    // unchanged on failure. Throws Exception with FTRAIN_STATUS_OVERFLOW at
    // capacity and std::bad_alloc on allocation failure.
    OperandId addOperand(OperandKind kind);

    // Adds one operation of kind with ordered input and output operand IDs
    // and returns its ID; operations may be added in any order. Inputs must be
    // distinct and may be operands without a producer; outputs must not
    // already have a producer. Throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT when an ID is out of range, an ID repeats
    // within the inputs or the outputs, or an output ID also appears among the
    // inputs. The Pattern is unchanged on failure. Throws Exception with
    // FTRAIN_STATUS_OVERFLOW at capacity and std::bad_alloc on allocation
    // failure.
    OperationId addOperation(OperationKind kind, std::vector<OperandId>&& inputs, std::vector<OperandId>&& outputs);

    std::vector<PatternOperandNode> operand_nodes_;
    std::vector<PatternOpNode> op_nodes_;
};

}  // namespace ftrain

#endif
