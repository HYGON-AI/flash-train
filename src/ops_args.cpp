// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <utility>

#include "flash_train/error.hpp"
#include "flash_train/ops_args.hpp"

namespace ftrain {

// -------------------------------------------------------------------- Matcher

namespace {

// One whole-structure match: for each user PatternBuilder operand (operation) ID,
// the corresponding supported PatternBuilder operand (operation) ID. Meaningful only
// with the Patterns passed to the producing call; the object retains neither.
struct MatchResult final {
    std::vector<PatternOperandId> supported_operand_ids_by_user_operand;
    std::vector<PatternOperationId> supported_op_ids_by_user_op;
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

    // Exact backtracking fallback, entered only when the signature pairing
    // above failed to verify.
    static std::optional<MatchResult> match(const Pattern& user_pattern, const Pattern& supported_pattern);
};

constexpr std::size_t kUnmatchedIndex = std::numeric_limits<std::size_t>::max();
constexpr PatternOperandId kUnmatchedOperandId{kUnmatchedIndex};
constexpr PatternOperationId kUnmatchedOpId{kUnmatchedIndex};

// Backtracking search for a complete bijection: pick the unmatched supported
// operation with the fewest compatible user operations, try each candidate,
// and undo the bindings a failed branch added. Binding an operation pair also
// binds its ports' operands, so operations anchor the search and the operands
// follow; operands no operation touches are paired only after every operation
// is bound. The two *_ids_by_* vector pairs record the mapping in both
// directions, so compatibility checks can enforce "already mapped implies
// equal" on both sides.
class MatchState final {
  private:
    friend class ::ftrain::Matcher;

    const Pattern& user_pattern_;
    const Pattern& supported_pattern_;

    std::vector<PatternOperandId> user_operand_ids_by_supported_operand_;
    std::vector<PatternOperandId> supported_operand_ids_by_user_operand_;

    std::vector<PatternOperationId> user_op_ids_by_supported_op_;
    std::vector<PatternOperationId> supported_op_ids_by_user_op_;

    std::vector<PatternOperandId> bound_supported_operand_ids_;
    std::vector<PatternOperationId> bound_supported_op_ids_;

    MatchState(const Pattern& user_pattern, const Pattern& supported_pattern);

    std::vector<PatternOperandId> takeSupportedOperandIdsByUserOperand() noexcept {
        return std::move(supported_operand_ids_by_user_operand_);
    }

    std::vector<PatternOperationId> takeSupportedOpIdsByUserOp() noexcept {
        return std::move(supported_op_ids_by_user_op_);
    }

    bool isOperandPairCompatible(PatternOperandId supported_operand_id, PatternOperandId user_operand_id) const;
    bool haveEqualConsumerSignatures(const PatternOperandNode& supported_operand_node,
                                     const PatternOperandNode& user_operand_node) const;
    bool isOpPairCompatible(PatternOperationId supported_op_id, PatternOperationId user_op_id) const;
    bool bindOperand(PatternOperandId supported_operand_id, PatternOperandId user_operand_id);
    bool bindOp(PatternOperationId supported_op_id, PatternOperationId user_op_id);
    void rollback(std::size_t operand_checkpoint, std::size_t op_checkpoint) noexcept;
    std::optional<PatternOperationId> selectNextSupportedOp() const;
    bool searchOps();
    bool bindRemainingOperands();
    bool validateCompleteMapping() const;
};

MatchState::MatchState(const Pattern& user_pattern, const Pattern& supported_pattern)
    : user_pattern_(user_pattern), supported_pattern_(supported_pattern),
      user_operand_ids_by_supported_operand_(supported_pattern.getNumOperands(), kUnmatchedOperandId),
      supported_operand_ids_by_user_operand_(user_pattern.getNumOperands(), kUnmatchedOperandId),
      user_op_ids_by_supported_op_(supported_pattern.getNumOps(), kUnmatchedOpId),
      supported_op_ids_by_user_op_(user_pattern.getNumOps(), kUnmatchedOpId) {
    bound_supported_operand_ids_.reserve(supported_pattern.getNumOperands());
    bound_supported_op_ids_.reserve(supported_pattern.getNumOps());
}

// Two consumer lists match as (input port, consumer kind) multisets: equal
// multisets are a necessary condition for a compatible operand pair and stay
// cheap to check before any operation is bound.
bool MatchState::haveEqualConsumerSignatures(const PatternOperandNode& supported_operand_node,
                                             const PatternOperandNode& user_operand_node) const {
    const auto& supported_consumers = supported_operand_node.getConsumers();
    const auto& user_consumers      = user_operand_node.getConsumers();
    if (supported_consumers.size() != user_consumers.size()) { return false; }

    const auto signature = [](const Pattern& pattern, const auto& consumers) {
        std::vector<std::pair<std::size_t, OperationKind>> consumer_signature;
        consumer_signature.reserve(consumers.size());
        for (const PatternOperationInputPortId consumer : consumers) {
            consumer_signature.emplace_back(consumer.getPortIndex(), pattern.getOpNode(consumer.getOpId()).getKind());
        }
        std::sort(consumer_signature.begin(), consumer_signature.end());
        return consumer_signature;
    };
    return signature(supported_pattern_, supported_consumers) == signature(user_pattern_, user_consumers);
}

// Necessary conditions for pairing these two operands: kinds equal, producer
// ports and kinds equal, consumer multisets equal, and every already-mapped
// neighbor agreeing with the pairing. Unmapped neighbors stay unconstrained
// here; they are checked when they themselves bind.
bool MatchState::isOperandPairCompatible(PatternOperandId supported_operand_id,
                                         PatternOperandId user_operand_id) const {
    const std::size_t supported_operand_index = supported_operand_id.getIndex();
    const std::size_t user_operand_index      = user_operand_id.getIndex();

    const PatternOperandId mapped_user_operand = user_operand_ids_by_supported_operand_[supported_operand_index];
    if (mapped_user_operand.getIndex() != kUnmatchedIndex) { return mapped_user_operand == user_operand_id; }
    if (supported_operand_ids_by_user_operand_[user_operand_index].getIndex() != kUnmatchedIndex) { return false; }

    const PatternOperandNode& supported_operand_node = supported_pattern_.getOperandNode(supported_operand_id);
    const PatternOperandNode& user_operand_node      = user_pattern_.getOperandNode(user_operand_id);
    if (supported_operand_node.getKind() != user_operand_node.getKind()) { return false; }

    const auto& supported_producer = supported_operand_node.getProducer();
    const auto& user_producer      = user_operand_node.getProducer();
    if (supported_producer.has_value() != user_producer.has_value()) { return false; }
    if (supported_producer.has_value()) {
        if (supported_producer->getPortIndex() != user_producer->getPortIndex()) { return false; }
        if (supported_pattern_.getOpNode(supported_producer->getOpId()).getKind() !=
            user_pattern_.getOpNode(user_producer->getOpId()).getKind()) {
            return false;
        }

        const PatternOperationId mapped_user_producer =
            user_op_ids_by_supported_op_[supported_producer->getOpId().getIndex()];
        if (mapped_user_producer.getIndex() != kUnmatchedIndex && mapped_user_producer != user_producer->getOpId()) {
            return false;
        }
        const PatternOperationId mapped_supported_producer =
            supported_op_ids_by_user_op_[user_producer->getOpId().getIndex()];
        if (mapped_supported_producer.getIndex() != kUnmatchedIndex &&
            mapped_supported_producer != supported_producer->getOpId()) {
            return false;
        }
    }

    if (!haveEqualConsumerSignatures(supported_operand_node, user_operand_node)) { return false; }

    for (const PatternOperationInputPortId supported_consumer : supported_operand_node.getConsumers()) {
        const PatternOperationId mapped_user_consumer =
            user_op_ids_by_supported_op_[supported_consumer.getOpId().getIndex()];
        if (mapped_user_consumer.getIndex() == kUnmatchedIndex) { continue; }
        const auto& user_inputs = user_pattern_.getOpNode(mapped_user_consumer).getInputs();
        if (user_inputs[supported_consumer.getPortIndex()] != user_operand_id) { return false; }
    }

    for (const PatternOperationInputPortId user_consumer : user_operand_node.getConsumers()) {
        const PatternOperationId mapped_supported_consumer =
            supported_op_ids_by_user_op_[user_consumer.getOpId().getIndex()];
        if (mapped_supported_consumer.getIndex() == kUnmatchedIndex) { continue; }
        const auto& supported_inputs = supported_pattern_.getOpNode(mapped_supported_consumer).getInputs();
        if (supported_inputs[user_consumer.getPortIndex()] != supported_operand_id) { return false; }
    }

    return true;
}

// Necessary conditions for pairing these two operations: kinds and port
// counts equal, and every port's operands pairwise compatible.
bool MatchState::isOpPairCompatible(PatternOperationId supported_op_id, PatternOperationId user_op_id) const {
    const std::size_t supported_op_index = supported_op_id.getIndex();
    const std::size_t user_op_index      = user_op_id.getIndex();

    const PatternOperationId mapped_user_op = user_op_ids_by_supported_op_[supported_op_index];
    if (mapped_user_op.getIndex() != kUnmatchedIndex) { return mapped_user_op == user_op_id; }
    if (supported_op_ids_by_user_op_[user_op_index].getIndex() != kUnmatchedIndex) { return false; }

    const PatternOperationNode& supported_op_node = supported_pattern_.getOpNode(supported_op_id);
    const PatternOperationNode& user_op_node      = user_pattern_.getOpNode(user_op_id);
    if (supported_op_node.getKind() != user_op_node.getKind()) { return false; }
    if (supported_op_node.getInputs().size() != user_op_node.getInputs().size() ||
        supported_op_node.getOutputs().size() != user_op_node.getOutputs().size()) {
        return false;
    }

    for (std::size_t port_index = 0; port_index < supported_op_node.getInputs().size(); ++port_index) {
        if (!isOperandPairCompatible(supported_op_node.getInputs()[port_index], user_op_node.getInputs()[port_index])) {
            return false;
        }
    }
    for (std::size_t port_index = 0; port_index < supported_op_node.getOutputs().size(); ++port_index) {
        if (!isOperandPairCompatible(supported_op_node.getOutputs()[port_index],
                                     user_op_node.getOutputs()[port_index])) {
            return false;
        }
    }

    return true;
}

bool MatchState::bindOperand(PatternOperandId supported_operand_id, PatternOperandId user_operand_id) {
    const std::size_t supported_operand_index = supported_operand_id.getIndex();
    const std::size_t user_operand_index      = user_operand_id.getIndex();

    const PatternOperandId mapped_user_operand = user_operand_ids_by_supported_operand_[supported_operand_index];
    if (mapped_user_operand.getIndex() != kUnmatchedIndex) { return mapped_user_operand == user_operand_id; }
    if (supported_operand_ids_by_user_operand_[user_operand_index].getIndex() != kUnmatchedIndex ||
        !isOperandPairCompatible(supported_operand_id, user_operand_id)) {
        return false;
    }

    user_operand_ids_by_supported_operand_[supported_operand_index] = user_operand_id;
    supported_operand_ids_by_user_operand_[user_operand_index]      = supported_operand_id;
    bound_supported_operand_ids_.push_back(supported_operand_id);
    return true;
}

bool MatchState::bindOp(PatternOperationId supported_op_id, PatternOperationId user_op_id) {
    const std::size_t supported_op_index = supported_op_id.getIndex();
    const std::size_t user_op_index      = user_op_id.getIndex();

    const PatternOperationId mapped_user_op = user_op_ids_by_supported_op_[supported_op_index];
    if (mapped_user_op.getIndex() != kUnmatchedIndex) { return mapped_user_op == user_op_id; }
    if (supported_op_ids_by_user_op_[user_op_index].getIndex() != kUnmatchedIndex ||
        !isOpPairCompatible(supported_op_id, user_op_id)) {
        return false;
    }

    user_op_ids_by_supported_op_[supported_op_index] = user_op_id;
    supported_op_ids_by_user_op_[user_op_index]      = supported_op_id;
    bound_supported_op_ids_.push_back(supported_op_id);

    const PatternOperationNode& supported_op_node = supported_pattern_.getOpNode(supported_op_id);
    const PatternOperationNode& user_op_node      = user_pattern_.getOpNode(user_op_id);
    for (std::size_t port_index = 0; port_index < supported_op_node.getInputs().size(); ++port_index) {
        if (!bindOperand(supported_op_node.getInputs()[port_index], user_op_node.getInputs()[port_index])) {
            return false;
        }
    }
    for (std::size_t port_index = 0; port_index < supported_op_node.getOutputs().size(); ++port_index) {
        if (!bindOperand(supported_op_node.getOutputs()[port_index], user_op_node.getOutputs()[port_index])) {
            return false;
        }
    }

    return true;
}

void MatchState::rollback(std::size_t operand_checkpoint, std::size_t op_checkpoint) noexcept {
    while (bound_supported_operand_ids_.size() > operand_checkpoint) {
        const PatternOperandId supported_operand_id = bound_supported_operand_ids_.back();
        const std::size_t supported_operand_index   = supported_operand_id.getIndex();
        const PatternOperandId user_operand_id      = user_operand_ids_by_supported_operand_[supported_operand_index];

        supported_operand_ids_by_user_operand_[user_operand_id.getIndex()] = kUnmatchedOperandId;
        user_operand_ids_by_supported_operand_[supported_operand_index]    = kUnmatchedOperandId;
        bound_supported_operand_ids_.pop_back();
    }

    while (bound_supported_op_ids_.size() > op_checkpoint) {
        const PatternOperationId supported_op_id = bound_supported_op_ids_.back();
        const std::size_t supported_op_index     = supported_op_id.getIndex();
        const PatternOperationId user_op_id      = user_op_ids_by_supported_op_[supported_op_index];

        supported_op_ids_by_user_op_[user_op_id.getIndex()] = kUnmatchedOpId;
        user_op_ids_by_supported_op_[supported_op_index]    = kUnmatchedOpId;
        bound_supported_op_ids_.pop_back();
    }
}

// Returns the unmatched supported operation with the fewest compatible user
// operations: binding the most constrained operation first surfaces dead ends
// with the least branching.
std::optional<PatternOperationId> MatchState::selectNextSupportedOp() const {
    std::optional<PatternOperationId> selected_supported_op;
    std::size_t fewest_candidates = std::numeric_limits<std::size_t>::max();

    for (std::size_t supported_op_index = 0; supported_op_index < supported_pattern_.getNumOps();
         ++supported_op_index) {
        if (user_op_ids_by_supported_op_[supported_op_index].getIndex() != kUnmatchedIndex) { continue; }

        const PatternOperationId supported_op_id{supported_op_index};
        std::size_t num_candidates = 0;
        for (std::size_t user_op_index = 0; user_op_index < user_pattern_.getNumOps(); ++user_op_index) {
            if (supported_op_ids_by_user_op_[user_op_index].getIndex() != kUnmatchedIndex) { continue; }
            if (isOpPairCompatible(supported_op_id, PatternOperationId{user_op_index})) { ++num_candidates; }
        }

        if (num_candidates < fewest_candidates) {
            selected_supported_op = supported_op_id;
            fewest_candidates     = num_candidates;
            if (fewest_candidates == 0) { break; }
        }
    }

    return selected_supported_op;
}

// Pairs the operands no operation touched: with every operation bound, the
// unmatched operands on both sides have neither producer nor consumers, so
// any compatible pairing completes the bijection.
bool MatchState::bindRemainingOperands() {
    for (std::size_t supported_operand_index = 0; supported_operand_index < supported_pattern_.getNumOperands();
         ++supported_operand_index) {
        if (user_operand_ids_by_supported_operand_[supported_operand_index].getIndex() != kUnmatchedIndex) { continue; }

        const PatternOperandId supported_operand_id{supported_operand_index};
        const PatternOperandNode& supported_operand_node = supported_pattern_.getOperandNode(supported_operand_id);
        if (supported_operand_node.getProducer().has_value() || !supported_operand_node.getConsumers().empty()) {
            return false;
        }

        bool found_user_operand = false;
        for (std::size_t user_operand_index = 0; user_operand_index < user_pattern_.getNumOperands();
             ++user_operand_index) {
            if (supported_operand_ids_by_user_operand_[user_operand_index].getIndex() != kUnmatchedIndex) { continue; }
            const PatternOperandId user_operand_id{user_operand_index};
            const PatternOperandNode& user_operand_node = user_pattern_.getOperandNode(user_operand_id);
            if (user_operand_node.getProducer().has_value() || !user_operand_node.getConsumers().empty()) { continue; }
            if (!isOperandPairCompatible(supported_operand_id, user_operand_id)) { continue; }

            if (!bindOperand(supported_operand_id, user_operand_id)) { continue; }
            found_user_operand = true;
            break;
        }

        if (!found_user_operand) { return false; }
    }

    return true;
}

// Rejects unless the finished mapping is a bijection that preserves kinds
// and every operation's input and output wiring in both directions.
bool MatchState::validateCompleteMapping() const {
    for (std::size_t supported_operand_index = 0; supported_operand_index < supported_pattern_.getNumOperands();
         ++supported_operand_index) {
        const PatternOperandId user_operand_id = user_operand_ids_by_supported_operand_[supported_operand_index];
        if (user_operand_id.getIndex() == kUnmatchedIndex ||
            supported_operand_ids_by_user_operand_[user_operand_id.getIndex()] !=
                PatternOperandId{supported_operand_index} ||
            supported_pattern_.getOperandNode(PatternOperandId{supported_operand_index}).getKind() !=
                user_pattern_.getOperandNode(user_operand_id).getKind()) {
            return false;
        }
    }

    for (std::size_t supported_op_index = 0; supported_op_index < supported_pattern_.getNumOps();
         ++supported_op_index) {
        const PatternOperationId user_op_id = user_op_ids_by_supported_op_[supported_op_index];
        if (user_op_id.getIndex() == kUnmatchedIndex ||
            supported_op_ids_by_user_op_[user_op_id.getIndex()] != PatternOperationId{supported_op_index}) {
            return false;
        }

        const PatternOperationNode& supported_op_node =
            supported_pattern_.getOpNode(PatternOperationId{supported_op_index});
        const PatternOperationNode& user_op_node = user_pattern_.getOpNode(user_op_id);
        if (supported_op_node.getKind() != user_op_node.getKind() ||
            supported_op_node.getInputs().size() != user_op_node.getInputs().size() ||
            supported_op_node.getOutputs().size() != user_op_node.getOutputs().size()) {
            return false;
        }

        for (std::size_t port_index = 0; port_index < supported_op_node.getInputs().size(); ++port_index) {
            if (user_operand_ids_by_supported_operand_[supported_op_node.getInputs()[port_index].getIndex()] !=
                user_op_node.getInputs()[port_index]) {
                return false;
            }
        }
        for (std::size_t port_index = 0; port_index < supported_op_node.getOutputs().size(); ++port_index) {
            if (user_operand_ids_by_supported_operand_[supported_op_node.getOutputs()[port_index].getIndex()] !=
                user_op_node.getOutputs()[port_index]) {
                return false;
            }
        }
    }

    return true;
}

// One search level: branch over the candidates of the most constrained
// supported operation; once every operation is bound, finish with the
// untouched operands and verify the complete mapping.
bool MatchState::searchOps() {
    const std::optional<PatternOperationId> supported_op_id = selectNextSupportedOp();
    if (!supported_op_id.has_value()) {
        const std::size_t operand_checkpoint = bound_supported_operand_ids_.size();
        const std::size_t op_checkpoint      = bound_supported_op_ids_.size();
        if (bindRemainingOperands() && validateCompleteMapping()) { return true; }
        rollback(operand_checkpoint, op_checkpoint);
        return false;
    }

    for (std::size_t user_op_index = 0; user_op_index < user_pattern_.getNumOps(); ++user_op_index) {
        if (supported_op_ids_by_user_op_[user_op_index].getIndex() != kUnmatchedIndex) { continue; }
        const PatternOperationId user_op_id{user_op_index};
        if (!isOpPairCompatible(*supported_op_id, user_op_id)) { continue; }

        const std::size_t operand_checkpoint = bound_supported_operand_ids_.size();
        const std::size_t op_checkpoint      = bound_supported_op_ids_.size();
        if (bindOp(*supported_op_id, user_op_id) && searchOps()) { return true; }
        rollback(operand_checkpoint, op_checkpoint);
    }

    return false;
}

std::optional<MatchResult> Matcher::matchBySignature(const Pattern& user_pattern, const Pattern& supported_pattern) {
    const std::size_t num_operands = user_pattern.getNumOperands();
    const std::size_t num_ops      = user_pattern.getNumOps();
    if (supported_pattern.getNumOperands() != num_operands || supported_pattern.getNumOps() != num_ops) {
        return std::nullopt;
    }

    // Equal colors are only a necessary condition, so pair the members of
    // each equal-color class in stable index order and verify the induced
    // bijection against both Patterns' wiring below.
    std::vector<std::size_t> paired_supported_operand_by_user(num_operands, kUnmatchedIndex);
    std::vector<std::size_t> paired_supported_op_by_user(num_ops, kUnmatchedIndex);
    const auto pair_by_color = [](const std::vector<std::uint64_t>& user_colors,
                                  const std::vector<std::uint64_t>& supported_colors,
                                  std::vector<std::size_t>& paired_supported_by_user) {
        std::unordered_map<std::uint64_t, std::vector<std::size_t>> user_classes;
        std::unordered_map<std::uint64_t, std::vector<std::size_t>> supported_classes;
        for (std::size_t index = 0; index < user_colors.size(); ++index) {
            user_classes[user_colors[index]].push_back(index);
            supported_classes[supported_colors[index]].push_back(index);
        }
        for (const auto& entry : user_classes) {
            const auto supported_entry = supported_classes.find(entry.first);
            if (supported_entry == supported_classes.end() || supported_entry->second.size() != entry.second.size()) {
                return false;
            }
            for (std::size_t member = 0; member < entry.second.size(); ++member) {
                paired_supported_by_user[entry.second[member]] = supported_entry->second[member];
            }
        }
        return true;
    };
    if (!pair_by_color(user_pattern.getOperandColors(), supported_pattern.getOperandColors(),
                       paired_supported_operand_by_user) ||
        !pair_by_color(user_pattern.getOpColors(), supported_pattern.getOpColors(), paired_supported_op_by_user)) {
        return std::nullopt;
    }

    // Verify the induced operation correspondence: kinds, port counts, and
    // every port's operand pairing must agree.
    for (std::size_t user_op_index = 0; user_op_index < num_ops; ++user_op_index) {
        const PatternOperationNode& user_node = user_pattern.getOpNode(PatternOperationId{user_op_index});
        const PatternOperationNode& supported_node =
            supported_pattern.getOpNode(PatternOperationId{paired_supported_op_by_user[user_op_index]});
        if (user_node.getKind() != supported_node.getKind() ||
            user_node.getInputs().size() != supported_node.getInputs().size() ||
            user_node.getOutputs().size() != supported_node.getOutputs().size()) {
            return std::nullopt;
        }
        for (std::size_t port = 0; port < user_node.getInputs().size(); ++port) {
            if (paired_supported_operand_by_user[user_node.getInputs()[port].getIndex()] !=
                supported_node.getInputs()[port].getIndex()) {
                return std::nullopt;
            }
        }
        for (std::size_t port = 0; port < user_node.getOutputs().size(); ++port) {
            if (paired_supported_operand_by_user[user_node.getOutputs()[port].getIndex()] !=
                supported_node.getOutputs()[port].getIndex()) {
                return std::nullopt;
            }
        }
    }

    // Verify the induced operand correspondence: kinds, producers, and
    // consumer sets must agree under the pairing.
    for (std::size_t user_operand_index = 0; user_operand_index < num_operands; ++user_operand_index) {
        const PatternOperandNode& user_node = user_pattern.getOperandNode(PatternOperandId{user_operand_index});
        const PatternOperandNode& supported_node =
            supported_pattern.getOperandNode(PatternOperandId{paired_supported_operand_by_user[user_operand_index]});
        if (user_node.getKind() != supported_node.getKind()) { return std::nullopt; }

        const auto& user_producer      = user_node.getProducer();
        const auto& supported_producer = supported_node.getProducer();
        if (user_producer.has_value() != supported_producer.has_value()) { return std::nullopt; }
        if (user_producer.has_value()) {
            if (user_producer->getPortIndex() != supported_producer->getPortIndex() ||
                paired_supported_op_by_user[user_producer->getOpId().getIndex()] !=
                    supported_producer->getOpId().getIndex()) {
                return std::nullopt;
            }
        }

        if (user_node.getConsumers().size() != supported_node.getConsumers().size()) { return std::nullopt; }
        std::vector<std::pair<std::size_t, std::uint64_t>> user_consumers;
        for (const PatternOperationInputPortId consumer : user_node.getConsumers()) {
            user_consumers.emplace_back(paired_supported_op_by_user[consumer.getOpId().getIndex()],
                                        static_cast<std::uint64_t>(consumer.getPortIndex()));
        }
        std::sort(user_consumers.begin(), user_consumers.end());
        std::vector<std::pair<std::size_t, std::uint64_t>> supported_consumers;
        for (const PatternOperationInputPortId consumer : supported_node.getConsumers()) {
            supported_consumers.emplace_back(consumer.getOpId().getIndex(),
                                             static_cast<std::uint64_t>(consumer.getPortIndex()));
        }
        std::sort(supported_consumers.begin(), supported_consumers.end());
        if (user_consumers != supported_consumers) { return std::nullopt; }
    }

    // Materialize the verified pairing as ID vectors for the caller.
    std::vector<PatternOperandId> supported_operand_ids_by_user;
    supported_operand_ids_by_user.reserve(num_operands);
    for (std::size_t index = 0; index < num_operands; ++index) {
        supported_operand_ids_by_user.push_back(PatternOperandId{paired_supported_operand_by_user[index]});
    }
    std::vector<PatternOperationId> supported_op_ids_by_user;
    supported_op_ids_by_user.reserve(num_ops);
    for (std::size_t index = 0; index < num_ops; ++index) {
        supported_op_ids_by_user.push_back(PatternOperationId{paired_supported_op_by_user[index]});
    }
    return MatchResult{std::move(supported_operand_ids_by_user), std::move(supported_op_ids_by_user)};
}

std::optional<MatchResult> Matcher::match(const Pattern& user_pattern, const Pattern& supported_pattern) {
    if (user_pattern.getNumOperands() != supported_pattern.getNumOperands() ||
        user_pattern.getNumOps() != supported_pattern.getNumOps()) {
        return std::nullopt;
    }

    MatchState state(user_pattern, supported_pattern);
    if (!state.searchOps()) { return std::nullopt; }

    return MatchResult{state.takeSupportedOperandIdsByUserOperand(), state.takeSupportedOpIdsByUserOp()};
}

}  // namespace

// ----------------------------------------------------------------- Ops/Args

Ops::Ops(const Pattern& user_pattern, const Pattern& supported_pattern) : pattern_key_(user_pattern.getKey()) {
    if (user_pattern.getKey() != supported_pattern.getKey()) {
        throw Exception(FTRAIN_STATUS_UNSUPPORTED, "Ops: user Pattern does not match the supported Pattern");
    }
    std::optional<MatchResult> matched = Matcher::matchBySignature(user_pattern, supported_pattern);
    if (!matched.has_value()) { matched = Matcher::match(user_pattern, supported_pattern); }
    if (!matched.has_value()) {
        throw Exception(FTRAIN_STATUS_UNSUPPORTED, "Ops: user Pattern does not match the supported Pattern");
    }
    supported_operand_ids_by_user_operand_ = std::move(matched->supported_operand_ids_by_user_operand);
    supported_op_ids_by_user_op_           = std::move(matched->supported_op_ids_by_user_op);

    operand_kinds_.reserve(supported_pattern.getNumOperands());
    for (std::size_t supported_operand_index = 0; supported_operand_index < supported_pattern.getNumOperands();
         ++supported_operand_index) {
        operand_kinds_.push_back(supported_pattern.getOperandNode(PatternOperandId{supported_operand_index}).getKind());
    }
    operation_kinds_.reserve(supported_pattern.getNumOps());
    for (std::size_t supported_op_index = 0; supported_op_index < supported_pattern.getNumOps(); ++supported_op_index) {
        operation_kinds_.push_back(supported_pattern.getOpNode(PatternOperationId{supported_op_index}).getKind());
    }
}

Args Ops::makeArgs() const {
    return Args(pattern_key_, supported_operand_ids_by_user_operand_, supported_op_ids_by_user_op_, operand_kinds_,
                operation_kinds_);
}

Args::Args(const PatternKey& pattern_key, const std::vector<PatternOperandId>& supported_operand_ids_by_user_operand,
           const std::vector<PatternOperationId>& supported_op_ids_by_user_op,
           const std::vector<OperandKind>& operand_kinds, const std::vector<OperationKind>& operation_kinds)
    : pattern_key_(pattern_key), supported_operand_ids_by_user_operand_(supported_operand_ids_by_user_operand),
      supported_op_ids_by_user_op_(supported_op_ids_by_user_op), operand_kinds_(operand_kinds),
      operation_kinds_(operation_kinds), operands_(operand_kinds.size()), op_arguments_(operation_kinds.size()) {}

bool Args::isComplete() const noexcept {
    return std::all_of(operands_.begin(), operands_.end(),
                       [](const std::optional<OperandValue>& operand) noexcept { return operand.has_value(); }) &&
           std::all_of(op_arguments_.begin(), op_arguments_.end(),
                       [](const std::optional<OperationValue>& attributes) noexcept { return attributes.has_value(); });
}

const std::optional<OperandValue>& Args::getOperand(PatternOperandId supported_operand_id) const {
    if (supported_operand_id.getIndex() >= operands_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "Supported OperandValue role index %zu is out of range [0, %zu)",
                        supported_operand_id.getIndex(), operands_.size());
    }
    return operands_[supported_operand_id.getIndex()];
}

const std::optional<OperationValue>& Args::getOpArgument(PatternOperationId supported_op_id) const {
    if (supported_op_id.getIndex() >= op_arguments_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Supported Op role index %zu is out of range [0, %zu)",
                        supported_op_id.getIndex(), op_arguments_.size());
    }
    return op_arguments_[supported_op_id.getIndex()];
}

std::size_t Args::mapUserOperand(PatternOperandId user_operand_id, OperandKind expected_kind) const {
    const std::size_t user_operand_index = user_operand_id.getIndex();
    if (user_operand_index >= supported_operand_ids_by_user_operand_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User OperandValue role index %zu is out of range [0, %zu)",
                        user_operand_index, supported_operand_ids_by_user_operand_.size());
    }
    const std::size_t supported_operand_index = supported_operand_ids_by_user_operand_[user_operand_index].getIndex();
    if (supported_operand_index >= operands_.size()) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Ops operand role mapping contains out-of-range slot %zu",
                        supported_operand_index);
    }
    if (operand_kinds_[supported_operand_index] != expected_kind) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User OperandValue role %zu has a different storage family",
                        user_operand_index);
    }
    return supported_operand_index;
}

std::size_t Args::mapUserOp(PatternOperationId user_op_id, OperationKind expected_kind) const {
    const std::size_t user_op_index = user_op_id.getIndex();
    if (user_op_index >= supported_op_ids_by_user_op_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User Op role index %zu is out of range [0, %zu)",
                        user_op_index, supported_op_ids_by_user_op_.size());
    }
    const std::size_t supported_op_index = supported_op_ids_by_user_op_[user_op_index].getIndex();
    if (supported_op_index >= op_arguments_.size()) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Ops operation role mapping contains out-of-range slot %zu",
                        supported_op_index);
    }
    if (operation_kinds_[supported_op_index] != expected_kind) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "User Op role %zu has a different operation kind",
                        user_op_index);
    }
    return supported_op_index;
}

void Args::setOp(PatternOperationId user_op_id, OperationKind expected_kind, OperationValue&& attributes) {
    const std::size_t supported_op_index = mapUserOp(user_op_id, expected_kind);
    op_arguments_[supported_op_index]    = std::move(attributes);
}

}  // namespace ftrain
