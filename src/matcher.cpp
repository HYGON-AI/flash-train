#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/matcher.hpp"

namespace ftrain {
namespace {

constexpr std::size_t kUnmatchedIndex = std::numeric_limits<std::size_t>::max();
constexpr PatternOperandId kUnmatchedOperandId{kUnmatchedIndex};
constexpr PatternOperationId kUnmatchedOpId{kUnmatchedIndex};

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

    bool search();

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

bool MatchState::haveEqualConsumerSignatures(const PatternOperandNode& supported_operand_node,
                                             const PatternOperandNode& user_operand_node) const {
    const auto& supported_consumers = supported_operand_node.getConsumers();
    const auto& user_consumers      = user_operand_node.getConsumers();
    if (supported_consumers.size() != user_consumers.size()) { return false; }

    for (const PatternOperationInputPortId supported_consumer : supported_consumers) {
        const OperationKind consumer_kind = supported_pattern_.getOpNode(supported_consumer.getOpId()).getKind();
        const std::size_t port_index      = supported_consumer.getPortIndex();

        std::size_t supported_occurrences = 0;
        for (const PatternOperationInputPortId other_supported_consumer : supported_consumers) {
            if (other_supported_consumer.getPortIndex() == port_index &&
                supported_pattern_.getOpNode(other_supported_consumer.getOpId()).getKind() == consumer_kind) {
                ++supported_occurrences;
            }
        }

        std::size_t user_occurrences = 0;
        for (const PatternOperationInputPortId user_consumer : user_consumers) {
            if (user_consumer.getPortIndex() == port_index &&
                user_pattern_.getOpNode(user_consumer.getOpId()).getKind() == consumer_kind) {
                ++user_occurrences;
            }
        }

        if (supported_occurrences != user_occurrences) { return false; }
    }

    return true;
}

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

bool MatchState::search() { return searchOps(); }

}  // namespace

std::optional<MatchResult> Matcher::matchBySignature(const Pattern& user_pattern, const Pattern& supported_pattern) {
    const std::size_t num_operands = user_pattern.getNumOperands();
    const std::size_t num_ops      = user_pattern.getNumOps();
    if (supported_pattern.getNumOperands() != num_operands || supported_pattern.getNumOps() != num_ops) {
        return std::nullopt;
    }

    // Pair the members of each equal-color class in stable index order.
    std::vector<std::size_t> paired_supported_operand_by_user(num_operands, kUnmatchedIndex);
    std::vector<std::size_t> paired_supported_op_by_user(num_ops, kUnmatchedIndex);
    {
        std::unordered_map<std::uint64_t, std::vector<std::size_t>> user_operand_classes;
        std::unordered_map<std::uint64_t, std::vector<std::size_t>> supported_operand_classes;
        for (std::size_t index = 0; index < num_operands; ++index) {
            user_operand_classes[user_pattern.getOperandColors()[index]].push_back(index);
            supported_operand_classes[supported_pattern.getOperandColors()[index]].push_back(index);
        }
        for (const auto& entry : user_operand_classes) {
            const auto supported_entry = supported_operand_classes.find(entry.first);
            if (supported_entry == supported_operand_classes.end() ||
                supported_entry->second.size() != entry.second.size()) {
                return std::nullopt;
            }
            for (std::size_t member = 0; member < entry.second.size(); ++member) {
                paired_supported_operand_by_user[entry.second[member]] = supported_entry->second[member];
            }
        }

        std::unordered_map<std::uint64_t, std::vector<std::size_t>> user_op_classes;
        std::unordered_map<std::uint64_t, std::vector<std::size_t>> supported_op_classes;
        for (std::size_t index = 0; index < num_ops; ++index) {
            user_op_classes[user_pattern.getOpColors()[index]].push_back(index);
            supported_op_classes[supported_pattern.getOpColors()[index]].push_back(index);
        }
        for (const auto& entry : user_op_classes) {
            const auto supported_entry = supported_op_classes.find(entry.first);
            if (supported_entry == supported_op_classes.end() ||
                supported_entry->second.size() != entry.second.size()) {
                return std::nullopt;
            }
            for (std::size_t member = 0; member < entry.second.size(); ++member) {
                paired_supported_op_by_user[entry.second[member]] = supported_entry->second[member];
            }
        }
    }

    std::vector<PatternOperandId> supported_operand_ids_by_user(num_operands, PatternOperandId{0});
    for (std::size_t index = 0; index < num_operands; ++index) {
        supported_operand_ids_by_user[index] = PatternOperandId{paired_supported_operand_by_user[index]};
    }
    std::vector<PatternOperationId> supported_op_ids_by_user(num_ops, PatternOperationId{0});
    for (std::size_t index = 0; index < num_ops; ++index) {
        supported_op_ids_by_user[index] = PatternOperationId{paired_supported_op_by_user[index]};
    }

    // Verify the induced op correspondence against both Patterns' wiring.
    for (std::size_t user_op_index = 0; user_op_index < num_ops; ++user_op_index) {
        const PatternOperationNode& user_node = user_pattern.getOpNode(PatternOperationId{user_op_index});
        const PatternOperationNode& supported_node =
            supported_pattern.getOpNode(supported_op_ids_by_user[user_op_index]);
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
            supported_pattern.getOperandNode(supported_operand_ids_by_user[user_operand_index]);
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

    return MatchResult(std::move(supported_operand_ids_by_user), std::move(supported_op_ids_by_user));
}

PatternOperandId MatchResult::getSupportedOperandId(PatternOperandId user_operand_id) const {
    const std::size_t index = user_operand_id.getIndex();
    if (index >= supported_operand_ids_by_user_operand_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "PatternOperandId index %zu is out of range for %zu matched operands", index,
                        supported_operand_ids_by_user_operand_.size());
    }
    return supported_operand_ids_by_user_operand_[index];
}

PatternOperationId MatchResult::getSupportedOpId(PatternOperationId user_op_id) const {
    const std::size_t index = user_op_id.getIndex();
    if (index >= supported_op_ids_by_user_op_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "PatternOperationId index %zu is out of range for %zu matched operations", index,
                        supported_op_ids_by_user_op_.size());
    }
    return supported_op_ids_by_user_op_[index];
}

std::optional<MatchResult> Matcher::match(const Pattern& user_pattern, const Pattern& supported_pattern) {
    if (user_pattern.getNumOperands() != supported_pattern.getNumOperands() ||
        user_pattern.getNumOps() != supported_pattern.getNumOps()) {
        return std::nullopt;
    }

    MatchState state(user_pattern, supported_pattern);
    if (!state.search()) { return std::nullopt; }

    return MatchResult(state.takeSupportedOperandIdsByUserOperand(), state.takeSupportedOpIdsByUserOp());
}

}  // namespace ftrain
