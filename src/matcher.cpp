#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/matcher.hpp"

namespace ftrain {
namespace {

constexpr std::uint64_t kPatternKeyMagic         = 0x46545241494E504BULL;
constexpr std::uint64_t kPatternKeyFormatVersion = 1;
constexpr std::uint64_t kPatternKeyHashOffset    = 14695981039346656037ULL;
constexpr std::uint64_t kPatternKeyHashPrime     = 1099511628211ULL;

static_assert(std::numeric_limits<std::size_t>::digits <= std::numeric_limits<std::uint64_t>::digits);

enum class PatternKeyTokenTag : std::uint64_t {
    kOperandSection = 1,
    kOpSection,
    kOpRecord,
    kInputSection,
    kOutputSection,
};

enum class CanonicalSignatureTag : std::uint64_t {
    kOperand = 1,
    kOp,
    kProducer,
    kConsumers,
    kInputs,
    kOutputs,
};

struct CanonicalVertexRef {
    enum class Kind : std::uint8_t {
        kOperand,
        kOp,
    };

    Kind kind;
    std::size_t index;
};

struct CanonicalColoring {
    std::vector<std::size_t> operand_colors;
    std::vector<std::size_t> op_colors;
};

struct CanonicalCandidate {
    std::vector<std::uint64_t> tokens;
    std::vector<OperandId> pattern_operand_ids_by_canonical_operand;
    std::vector<OperationId> pattern_op_ids_by_canonical_op;
};

struct CanonicalVertexSignature {
    CanonicalVertexRef vertex;
    std::vector<std::uint64_t> tokens;
};

class CanonicalSearchState final {
  public:
    explicit CanonicalSearchState(const Pattern& pattern) noexcept : pattern_(pattern) {}

    // Exhaustively resolves structural symmetries and returns the
    // lexicographically least complete topology encoding with its mappings.
    CanonicalCandidate run();

  private:
    static std::size_t getColor(const CanonicalColoring& coloring, CanonicalVertexRef vertex) noexcept;
    static void setColor(CanonicalColoring& coloring, CanonicalVertexRef vertex, std::size_t color) noexcept;
    static std::size_t getNumColors(const CanonicalColoring& coloring) noexcept;

    CanonicalColoring makeInitialColoring() const;
    std::vector<std::uint64_t> makeSignature(CanonicalVertexRef vertex, const CanonicalColoring& coloring) const;
    static std::size_t assignColors(std::vector<CanonicalVertexSignature>& signatures, CanonicalColoring& coloring);
    std::size_t refine(CanonicalColoring& coloring) const;
    std::vector<CanonicalVertexRef> selectAmbiguousClass(const CanonicalColoring& coloring,
                                                         std::size_t num_colors) const;
    CanonicalCandidate serializeCandidate(const CanonicalColoring& coloring) const;
    void search(CanonicalColoring coloring);

    const Pattern& pattern_;
    std::optional<CanonicalCandidate> best_candidate_;
};

std::size_t CanonicalSearchState::getColor(const CanonicalColoring& coloring, CanonicalVertexRef vertex) noexcept {
    if (vertex.kind == CanonicalVertexRef::Kind::kOperand) { return coloring.operand_colors[vertex.index]; }
    return coloring.op_colors[vertex.index];
}

void CanonicalSearchState::setColor(CanonicalColoring& coloring, CanonicalVertexRef vertex,
                                    std::size_t color) noexcept {
    if (vertex.kind == CanonicalVertexRef::Kind::kOperand) {
        coloring.operand_colors[vertex.index] = color;
        return;
    }
    coloring.op_colors[vertex.index] = color;
}

std::size_t CanonicalSearchState::getNumColors(const CanonicalColoring& coloring) noexcept {
    std::size_t largest_color = 0;
    bool has_vertex           = false;
    for (const std::size_t color : coloring.operand_colors) {
        largest_color = std::max(largest_color, color);
        has_vertex    = true;
    }
    for (const std::size_t color : coloring.op_colors) {
        largest_color = std::max(largest_color, color);
        has_vertex    = true;
    }
    return has_vertex ? largest_color + 1 : 0;
}

std::size_t CanonicalSearchState::assignColors(std::vector<CanonicalVertexSignature>& signatures,
                                               CanonicalColoring& coloring) {
    std::sort(signatures.begin(), signatures.end(),
              [](const CanonicalVertexSignature& left, const CanonicalVertexSignature& right) {
                  return left.tokens < right.tokens;
              });

    std::size_t color = 0;
    for (std::size_t index = 0; index < signatures.size(); ++index) {
        if (index != 0 && signatures[index].tokens != signatures[index - 1].tokens) { ++color; }
        setColor(coloring, signatures[index].vertex, color);
    }
    return signatures.empty() ? 0 : color + 1;
}

CanonicalColoring CanonicalSearchState::makeInitialColoring() const {
    CanonicalColoring coloring{std::vector<std::size_t>(pattern_.getNumOperands()),
                               std::vector<std::size_t>(pattern_.getNumOps())};
    std::vector<CanonicalVertexSignature> signatures;
    signatures.reserve(pattern_.getNumOperands() + pattern_.getNumOps());

    for (std::size_t operand_index = 0; operand_index < pattern_.getNumOperands(); ++operand_index) {
        const PatternOperandNode& operand_node = pattern_.getOperandNode(OperandId{operand_index});
        std::vector<std::uint64_t> tokens{static_cast<std::uint64_t>(CanonicalSignatureTag::kOperand),
                                          static_cast<std::uint64_t>(operand_node.getKind())};
        signatures.push_back(CanonicalVertexSignature{
            CanonicalVertexRef{CanonicalVertexRef::Kind::kOperand, operand_index},
            std::move(tokens)
        });
    }
    for (std::size_t op_index = 0; op_index < pattern_.getNumOps(); ++op_index) {
        const PatternOpNode& op_node = pattern_.getOpNode(OperationId{op_index});
        std::vector<std::uint64_t> tokens{static_cast<std::uint64_t>(CanonicalSignatureTag::kOp),
                                          static_cast<std::uint64_t>(op_node.getKind())};
        signatures.push_back(CanonicalVertexSignature{
            CanonicalVertexRef{CanonicalVertexRef::Kind::kOp, op_index},
            std::move(tokens)
        });
    }

    static_cast<void>(assignColors(signatures, coloring));
    return coloring;
}

std::vector<std::uint64_t> CanonicalSearchState::makeSignature(CanonicalVertexRef vertex,
                                                               const CanonicalColoring& coloring) const {
    std::vector<std::uint64_t> tokens;

    if (vertex.kind == CanonicalVertexRef::Kind::kOperand) {
        const PatternOperandNode& operand_node = pattern_.getOperandNode(OperandId{vertex.index});
        tokens.reserve(8 + operand_node.getConsumers().size() * 2);
        tokens.push_back(static_cast<std::uint64_t>(CanonicalSignatureTag::kOperand));
        tokens.push_back(static_cast<std::uint64_t>(getColor(coloring, vertex)));
        tokens.push_back(static_cast<std::uint64_t>(operand_node.getKind()));
        tokens.push_back(static_cast<std::uint64_t>(CanonicalSignatureTag::kProducer));

        const std::optional<OperationOutputPortId>& producer = operand_node.getProducer();
        tokens.push_back(producer.has_value() ? 1 : 0);
        if (producer.has_value()) {
            tokens.push_back(static_cast<std::uint64_t>(producer->getPortIndex()));
            tokens.push_back(static_cast<std::uint64_t>(coloring.op_colors[producer->getOpId().getIndex()]));
        }

        std::vector<std::pair<std::size_t, std::size_t>> consumers;
        consumers.reserve(operand_node.getConsumers().size());
        for (const OperationInputPortId consumer : operand_node.getConsumers()) {
            consumers.emplace_back(consumer.getPortIndex(), coloring.op_colors[consumer.getOpId().getIndex()]);
        }
        std::sort(consumers.begin(), consumers.end());

        tokens.push_back(static_cast<std::uint64_t>(CanonicalSignatureTag::kConsumers));
        tokens.push_back(static_cast<std::uint64_t>(consumers.size()));
        for (const auto& consumer : consumers) {
            tokens.push_back(static_cast<std::uint64_t>(consumer.first));
            tokens.push_back(static_cast<std::uint64_t>(consumer.second));
        }
        return tokens;
    }

    const PatternOpNode& op_node = pattern_.getOpNode(OperationId{vertex.index});
    tokens.reserve(7 + op_node.getInputs().size() + op_node.getOutputs().size());
    tokens.push_back(static_cast<std::uint64_t>(CanonicalSignatureTag::kOp));
    tokens.push_back(static_cast<std::uint64_t>(getColor(coloring, vertex)));
    tokens.push_back(static_cast<std::uint64_t>(op_node.getKind()));
    tokens.push_back(static_cast<std::uint64_t>(CanonicalSignatureTag::kInputs));
    tokens.push_back(static_cast<std::uint64_t>(op_node.getInputs().size()));
    for (const OperandId input : op_node.getInputs()) {
        tokens.push_back(static_cast<std::uint64_t>(coloring.operand_colors[input.getIndex()]));
    }
    tokens.push_back(static_cast<std::uint64_t>(CanonicalSignatureTag::kOutputs));
    tokens.push_back(static_cast<std::uint64_t>(op_node.getOutputs().size()));
    for (const OperandId output : op_node.getOutputs()) {
        tokens.push_back(static_cast<std::uint64_t>(coloring.operand_colors[output.getIndex()]));
    }
    return tokens;
}

std::size_t CanonicalSearchState::refine(CanonicalColoring& coloring) const {
    std::size_t num_colors = getNumColors(coloring);
    while (true) {
        std::vector<CanonicalVertexSignature> signatures;
        signatures.reserve(pattern_.getNumOperands() + pattern_.getNumOps());
        for (std::size_t operand_index = 0; operand_index < pattern_.getNumOperands(); ++operand_index) {
            const CanonicalVertexRef vertex{CanonicalVertexRef::Kind::kOperand, operand_index};
            signatures.push_back(CanonicalVertexSignature{vertex, makeSignature(vertex, coloring)});
        }
        for (std::size_t op_index = 0; op_index < pattern_.getNumOps(); ++op_index) {
            const CanonicalVertexRef vertex{CanonicalVertexRef::Kind::kOp, op_index};
            signatures.push_back(CanonicalVertexSignature{vertex, makeSignature(vertex, coloring)});
        }

        CanonicalColoring refined{std::vector<std::size_t>(pattern_.getNumOperands()),
                                  std::vector<std::size_t>(pattern_.getNumOps())};
        const std::size_t refined_num_colors = assignColors(signatures, refined);
        coloring                             = std::move(refined);
        if (refined_num_colors == num_colors) { return refined_num_colors; }
        num_colors = refined_num_colors;
    }
}

std::vector<CanonicalVertexRef> CanonicalSearchState::selectAmbiguousClass(const CanonicalColoring& coloring,
                                                                           std::size_t num_colors) const {
    std::vector<std::vector<CanonicalVertexRef>> color_classes(num_colors);
    for (std::size_t operand_index = 0; operand_index < coloring.operand_colors.size(); ++operand_index) {
        color_classes[coloring.operand_colors[operand_index]].push_back(
            CanonicalVertexRef{CanonicalVertexRef::Kind::kOperand, operand_index});
    }
    for (std::size_t op_index = 0; op_index < coloring.op_colors.size(); ++op_index) {
        color_classes[coloring.op_colors[op_index]].push_back(
            CanonicalVertexRef{CanonicalVertexRef::Kind::kOp, op_index});
    }

    std::size_t selected_color = num_colors;
    std::size_t selected_size  = std::numeric_limits<std::size_t>::max();
    for (std::size_t color = 0; color < color_classes.size(); ++color) {
        const std::size_t class_size = color_classes[color].size();
        if (class_size > 1 && class_size < selected_size) {
            selected_color = color;
            selected_size  = class_size;
        }
    }

    if (selected_color == num_colors) { return {}; }
    return std::move(color_classes[selected_color]);
}

CanonicalCandidate CanonicalSearchState::serializeCandidate(const CanonicalColoring& coloring) const {
    CanonicalCandidate candidate;
    candidate.pattern_operand_ids_by_canonical_operand.reserve(pattern_.getNumOperands());
    candidate.pattern_op_ids_by_canonical_op.reserve(pattern_.getNumOps());

    for (std::size_t operand_index = 0; operand_index < pattern_.getNumOperands(); ++operand_index) {
        candidate.pattern_operand_ids_by_canonical_operand.push_back(OperandId{operand_index});
    }
    std::sort(candidate.pattern_operand_ids_by_canonical_operand.begin(),
              candidate.pattern_operand_ids_by_canonical_operand.end(), [&coloring](OperandId left, OperandId right) {
                  return coloring.operand_colors[left.getIndex()] < coloring.operand_colors[right.getIndex()];
              });

    for (std::size_t op_index = 0; op_index < pattern_.getNumOps(); ++op_index) {
        candidate.pattern_op_ids_by_canonical_op.push_back(OperationId{op_index});
    }
    std::sort(candidate.pattern_op_ids_by_canonical_op.begin(), candidate.pattern_op_ids_by_canonical_op.end(),
              [&coloring](OperationId left, OperationId right) {
                  return coloring.op_colors[left.getIndex()] < coloring.op_colors[right.getIndex()];
              });

    std::vector<std::size_t> canonical_operand_indices_by_pattern_operand(pattern_.getNumOperands());
    for (std::size_t canonical_index = 0; canonical_index < candidate.pattern_operand_ids_by_canonical_operand.size();
         ++canonical_index) {
        canonical_operand_indices_by_pattern_operand[candidate.pattern_operand_ids_by_canonical_operand[canonical_index]
                                                         .getIndex()] = canonical_index;
    }

    std::size_t token_capacity = 6 + pattern_.getNumOperands();
    for (const OperationId op_id : candidate.pattern_op_ids_by_canonical_op) {
        const PatternOpNode& op_node  = pattern_.getOpNode(op_id);
        token_capacity               += 6 + op_node.getInputs().size() + op_node.getOutputs().size();
    }
    candidate.tokens.reserve(token_capacity);

    candidate.tokens.push_back(kPatternKeyMagic);
    candidate.tokens.push_back(kPatternKeyFormatVersion);
    candidate.tokens.push_back(static_cast<std::uint64_t>(PatternKeyTokenTag::kOperandSection));
    candidate.tokens.push_back(static_cast<std::uint64_t>(pattern_.getNumOperands()));
    for (const OperandId operand_id : candidate.pattern_operand_ids_by_canonical_operand) {
        candidate.tokens.push_back(static_cast<std::uint64_t>(pattern_.getOperandNode(operand_id).getKind()));
    }

    candidate.tokens.push_back(static_cast<std::uint64_t>(PatternKeyTokenTag::kOpSection));
    candidate.tokens.push_back(static_cast<std::uint64_t>(pattern_.getNumOps()));
    for (const OperationId op_id : candidate.pattern_op_ids_by_canonical_op) {
        const PatternOpNode& op_node = pattern_.getOpNode(op_id);
        candidate.tokens.push_back(static_cast<std::uint64_t>(PatternKeyTokenTag::kOpRecord));
        candidate.tokens.push_back(static_cast<std::uint64_t>(op_node.getKind()));
        candidate.tokens.push_back(static_cast<std::uint64_t>(PatternKeyTokenTag::kInputSection));
        candidate.tokens.push_back(static_cast<std::uint64_t>(op_node.getInputs().size()));
        for (const OperandId input : op_node.getInputs()) {
            candidate.tokens.push_back(
                static_cast<std::uint64_t>(canonical_operand_indices_by_pattern_operand[input.getIndex()]));
        }
        candidate.tokens.push_back(static_cast<std::uint64_t>(PatternKeyTokenTag::kOutputSection));
        candidate.tokens.push_back(static_cast<std::uint64_t>(op_node.getOutputs().size()));
        for (const OperandId output : op_node.getOutputs()) {
            candidate.tokens.push_back(
                static_cast<std::uint64_t>(canonical_operand_indices_by_pattern_operand[output.getIndex()]));
        }
    }

    return candidate;
}

void CanonicalSearchState::search(CanonicalColoring coloring) {
    const std::size_t num_colors                          = refine(coloring);
    const std::vector<CanonicalVertexRef> ambiguous_class = selectAmbiguousClass(coloring, num_colors);
    if (ambiguous_class.empty()) {
        CanonicalCandidate candidate = serializeCandidate(coloring);
        if (!best_candidate_.has_value() || candidate.tokens < best_candidate_->tokens) {
            best_candidate_.emplace(std::move(candidate));
        }
        return;
    }

    const std::size_t individualized_color = num_colors;
    for (const CanonicalVertexRef vertex : ambiguous_class) {
        CanonicalColoring branch = coloring;
        setColor(branch, vertex, individualized_color);
        search(std::move(branch));
    }
}

CanonicalCandidate CanonicalSearchState::run() {
    search(makeInitialColoring());
    return std::move(*best_candidate_);
}

std::size_t hashPatternKeyTokens(const std::vector<std::uint64_t>& tokens) noexcept {
    std::uint64_t hash = kPatternKeyHashOffset;
    for (const std::uint64_t token : tokens) {
        hash ^= token;
        hash *= kPatternKeyHashPrime;
        hash ^= token >> 32;
        hash *= kPatternKeyHashPrime;
    }
    return static_cast<std::size_t>(hash);
}

constexpr std::size_t kUnmatchedIndex = std::numeric_limits<std::size_t>::max();
constexpr OperandId kUnmatchedOperandId{kUnmatchedIndex};
constexpr OperationId kUnmatchedOpId{kUnmatchedIndex};

class MatchState final {
  private:
    friend class ::ftrain::Matcher;

    const Pattern& user_pattern_;
    const Pattern& supported_pattern_;

    std::vector<OperandId> user_operand_ids_by_supported_operand_;
    std::vector<OperandId> supported_operand_ids_by_user_operand_;

    std::vector<OperationId> user_op_ids_by_supported_op_;
    std::vector<OperationId> supported_op_ids_by_user_op_;

    std::vector<OperandId> bound_supported_operand_ids_;
    std::vector<OperationId> bound_supported_op_ids_;

    MatchState(const Pattern& user_pattern, const Pattern& supported_pattern);

    bool search();

    std::vector<OperandId> takeSupportedOperandIdsByUserOperand() noexcept {
        return std::move(supported_operand_ids_by_user_operand_);
    }

    std::vector<OperationId> takeSupportedOpIdsByUserOp() noexcept { return std::move(supported_op_ids_by_user_op_); }

    bool isOperandPairCompatible(OperandId supported_operand_id, OperandId user_operand_id) const;
    bool haveEqualConsumerSignatures(const PatternOperandNode& supported_operand_node,
                                     const PatternOperandNode& user_operand_node) const;
    bool isOpPairCompatible(OperationId supported_op_id, OperationId user_op_id) const;
    bool bindOperand(OperandId supported_operand_id, OperandId user_operand_id);
    bool bindOp(OperationId supported_op_id, OperationId user_op_id);
    void rollback(std::size_t operand_checkpoint, std::size_t op_checkpoint) noexcept;
    std::optional<OperationId> selectNextSupportedOp() const;
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

    for (const OperationInputPortId supported_consumer : supported_consumers) {
        const OperationKind consumer_kind = supported_pattern_.getOpNode(supported_consumer.getOpId()).getKind();
        const std::size_t port_index      = supported_consumer.getPortIndex();

        std::size_t supported_occurrences = 0;
        for (const OperationInputPortId other_supported_consumer : supported_consumers) {
            if (other_supported_consumer.getPortIndex() == port_index &&
                supported_pattern_.getOpNode(other_supported_consumer.getOpId()).getKind() == consumer_kind) {
                ++supported_occurrences;
            }
        }

        std::size_t user_occurrences = 0;
        for (const OperationInputPortId user_consumer : user_consumers) {
            if (user_consumer.getPortIndex() == port_index &&
                user_pattern_.getOpNode(user_consumer.getOpId()).getKind() == consumer_kind) {
                ++user_occurrences;
            }
        }

        if (supported_occurrences != user_occurrences) { return false; }
    }

    return true;
}

bool MatchState::isOperandPairCompatible(OperandId supported_operand_id, OperandId user_operand_id) const {
    const std::size_t supported_operand_index = supported_operand_id.getIndex();
    const std::size_t user_operand_index      = user_operand_id.getIndex();

    const OperandId mapped_user_operand = user_operand_ids_by_supported_operand_[supported_operand_index];
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

        const OperationId mapped_user_producer = user_op_ids_by_supported_op_[supported_producer->getOpId().getIndex()];
        if (mapped_user_producer.getIndex() != kUnmatchedIndex && mapped_user_producer != user_producer->getOpId()) {
            return false;
        }
        const OperationId mapped_supported_producer = supported_op_ids_by_user_op_[user_producer->getOpId().getIndex()];
        if (mapped_supported_producer.getIndex() != kUnmatchedIndex &&
            mapped_supported_producer != supported_producer->getOpId()) {
            return false;
        }
    }

    if (!haveEqualConsumerSignatures(supported_operand_node, user_operand_node)) { return false; }

    for (const OperationInputPortId supported_consumer : supported_operand_node.getConsumers()) {
        const OperationId mapped_user_consumer = user_op_ids_by_supported_op_[supported_consumer.getOpId().getIndex()];
        if (mapped_user_consumer.getIndex() == kUnmatchedIndex) { continue; }
        const auto& user_inputs = user_pattern_.getOpNode(mapped_user_consumer).getInputs();
        if (user_inputs[supported_consumer.getPortIndex()] != user_operand_id) { return false; }
    }

    for (const OperationInputPortId user_consumer : user_operand_node.getConsumers()) {
        const OperationId mapped_supported_consumer = supported_op_ids_by_user_op_[user_consumer.getOpId().getIndex()];
        if (mapped_supported_consumer.getIndex() == kUnmatchedIndex) { continue; }
        const auto& supported_inputs = supported_pattern_.getOpNode(mapped_supported_consumer).getInputs();
        if (supported_inputs[user_consumer.getPortIndex()] != supported_operand_id) { return false; }
    }

    return true;
}

bool MatchState::isOpPairCompatible(OperationId supported_op_id, OperationId user_op_id) const {
    const std::size_t supported_op_index = supported_op_id.getIndex();
    const std::size_t user_op_index      = user_op_id.getIndex();

    const OperationId mapped_user_op = user_op_ids_by_supported_op_[supported_op_index];
    if (mapped_user_op.getIndex() != kUnmatchedIndex) { return mapped_user_op == user_op_id; }
    if (supported_op_ids_by_user_op_[user_op_index].getIndex() != kUnmatchedIndex) { return false; }

    const PatternOpNode& supported_op_node = supported_pattern_.getOpNode(supported_op_id);
    const PatternOpNode& user_op_node      = user_pattern_.getOpNode(user_op_id);
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

bool MatchState::bindOperand(OperandId supported_operand_id, OperandId user_operand_id) {
    const std::size_t supported_operand_index = supported_operand_id.getIndex();
    const std::size_t user_operand_index      = user_operand_id.getIndex();

    const OperandId mapped_user_operand = user_operand_ids_by_supported_operand_[supported_operand_index];
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

bool MatchState::bindOp(OperationId supported_op_id, OperationId user_op_id) {
    const std::size_t supported_op_index = supported_op_id.getIndex();
    const std::size_t user_op_index      = user_op_id.getIndex();

    const OperationId mapped_user_op = user_op_ids_by_supported_op_[supported_op_index];
    if (mapped_user_op.getIndex() != kUnmatchedIndex) { return mapped_user_op == user_op_id; }
    if (supported_op_ids_by_user_op_[user_op_index].getIndex() != kUnmatchedIndex ||
        !isOpPairCompatible(supported_op_id, user_op_id)) {
        return false;
    }

    user_op_ids_by_supported_op_[supported_op_index] = user_op_id;
    supported_op_ids_by_user_op_[user_op_index]      = supported_op_id;
    bound_supported_op_ids_.push_back(supported_op_id);

    const PatternOpNode& supported_op_node = supported_pattern_.getOpNode(supported_op_id);
    const PatternOpNode& user_op_node      = user_pattern_.getOpNode(user_op_id);
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
        const OperandId supported_operand_id      = bound_supported_operand_ids_.back();
        const std::size_t supported_operand_index = supported_operand_id.getIndex();
        const OperandId user_operand_id           = user_operand_ids_by_supported_operand_[supported_operand_index];

        supported_operand_ids_by_user_operand_[user_operand_id.getIndex()] = kUnmatchedOperandId;
        user_operand_ids_by_supported_operand_[supported_operand_index]    = kUnmatchedOperandId;
        bound_supported_operand_ids_.pop_back();
    }

    while (bound_supported_op_ids_.size() > op_checkpoint) {
        const OperationId supported_op_id    = bound_supported_op_ids_.back();
        const std::size_t supported_op_index = supported_op_id.getIndex();
        const OperationId user_op_id         = user_op_ids_by_supported_op_[supported_op_index];

        supported_op_ids_by_user_op_[user_op_id.getIndex()] = kUnmatchedOpId;
        user_op_ids_by_supported_op_[supported_op_index]    = kUnmatchedOpId;
        bound_supported_op_ids_.pop_back();
    }
}

std::optional<OperationId> MatchState::selectNextSupportedOp() const {
    std::optional<OperationId> selected_supported_op;
    std::size_t fewest_candidates = std::numeric_limits<std::size_t>::max();

    for (std::size_t supported_op_index = 0; supported_op_index < supported_pattern_.getNumOps();
         ++supported_op_index) {
        if (user_op_ids_by_supported_op_[supported_op_index].getIndex() != kUnmatchedIndex) { continue; }

        const OperationId supported_op_id{supported_op_index};
        std::size_t num_candidates = 0;
        for (std::size_t user_op_index = 0; user_op_index < user_pattern_.getNumOps(); ++user_op_index) {
            if (supported_op_ids_by_user_op_[user_op_index].getIndex() != kUnmatchedIndex) { continue; }
            if (isOpPairCompatible(supported_op_id, OperationId{user_op_index})) { ++num_candidates; }
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

        const OperandId supported_operand_id{supported_operand_index};
        const PatternOperandNode& supported_operand_node = supported_pattern_.getOperandNode(supported_operand_id);
        if (supported_operand_node.getProducer().has_value() || !supported_operand_node.getConsumers().empty()) {
            return false;
        }

        bool found_user_operand = false;
        for (std::size_t user_operand_index = 0; user_operand_index < user_pattern_.getNumOperands();
             ++user_operand_index) {
            if (supported_operand_ids_by_user_operand_[user_operand_index].getIndex() != kUnmatchedIndex) { continue; }
            const OperandId user_operand_id{user_operand_index};
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
        const OperandId user_operand_id = user_operand_ids_by_supported_operand_[supported_operand_index];
        if (user_operand_id.getIndex() == kUnmatchedIndex ||
            supported_operand_ids_by_user_operand_[user_operand_id.getIndex()] != OperandId{supported_operand_index} ||
            supported_pattern_.getOperandNode(OperandId{supported_operand_index}).getKind() !=
                user_pattern_.getOperandNode(user_operand_id).getKind()) {
            return false;
        }
    }

    for (std::size_t supported_op_index = 0; supported_op_index < supported_pattern_.getNumOps();
         ++supported_op_index) {
        const OperationId user_op_id = user_op_ids_by_supported_op_[supported_op_index];
        if (user_op_id.getIndex() == kUnmatchedIndex ||
            supported_op_ids_by_user_op_[user_op_id.getIndex()] != OperationId{supported_op_index}) {
            return false;
        }

        const PatternOpNode& supported_op_node = supported_pattern_.getOpNode(OperationId{supported_op_index});
        const PatternOpNode& user_op_node      = user_pattern_.getOpNode(user_op_id);
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
    const std::optional<OperationId> supported_op_id = selectNextSupportedOp();
    if (!supported_op_id.has_value()) {
        const std::size_t operand_checkpoint = bound_supported_operand_ids_.size();
        const std::size_t op_checkpoint      = bound_supported_op_ids_.size();
        if (bindRemainingOperands() && validateCompleteMapping()) { return true; }
        rollback(operand_checkpoint, op_checkpoint);
        return false;
    }

    for (std::size_t user_op_index = 0; user_op_index < user_pattern_.getNumOps(); ++user_op_index) {
        if (supported_op_ids_by_user_op_[user_op_index].getIndex() != kUnmatchedIndex) { continue; }
        const OperationId user_op_id{user_op_index};
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

PatternKey::PatternKey(std::vector<std::uint64_t>&& tokens) noexcept
    : tokens_(std::move(tokens)), hash_(hashPatternKeyTokens(tokens_)) {}

PatternKey::PatternKey(PatternKey&& other) noexcept
    : tokens_(std::move(other.tokens_)), hash_(hashPatternKeyTokens(tokens_)) {
    other.hash_ = hashPatternKeyTokens(other.tokens_);
}

PatternKey& PatternKey::operator=(PatternKey&& other) noexcept {
    if (this == &other) { return *this; }

    tokens_     = std::move(other.tokens_);
    hash_       = hashPatternKeyTokens(tokens_);
    other.hash_ = hashPatternKeyTokens(other.tokens_);
    return *this;
}

std::size_t PatternCanonicalization::getCanonicalOperandIndex(OperandId pattern_operand_id) const {
    const std::size_t index = pattern_operand_id.getIndex();
    if (index >= canonical_operand_indices_by_pattern_operand_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "OperandId index %zu is out of range for %zu canonical operands", index,
                        canonical_operand_indices_by_pattern_operand_.size());
    }
    return canonical_operand_indices_by_pattern_operand_[index];
}

OperandId PatternCanonicalization::getOperandId(std::size_t canonical_operand_index) const {
    if (canonical_operand_index >= pattern_operand_ids_by_canonical_operand_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Canonical Operand index %zu is out of range for %zu operands",
                        canonical_operand_index, pattern_operand_ids_by_canonical_operand_.size());
    }
    return pattern_operand_ids_by_canonical_operand_[canonical_operand_index];
}

std::size_t PatternCanonicalization::getCanonicalOpIndex(OperationId pattern_op_id) const {
    const std::size_t index = pattern_op_id.getIndex();
    if (index >= canonical_op_indices_by_pattern_op_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "OperationId index %zu is out of range for %zu canonical operations", index,
                        canonical_op_indices_by_pattern_op_.size());
    }
    return canonical_op_indices_by_pattern_op_[index];
}

OperationId PatternCanonicalization::getOperationId(std::size_t canonical_op_index) const {
    if (canonical_op_index >= pattern_op_ids_by_canonical_op_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Canonical Op index %zu is out of range for %zu operations",
                        canonical_op_index, pattern_op_ids_by_canonical_op_.size());
    }
    return pattern_op_ids_by_canonical_op_[canonical_op_index];
}

OperandId MatchResult::getSupportedOperandId(OperandId user_operand_id) const {
    const std::size_t index = user_operand_id.getIndex();
    if (index >= supported_operand_ids_by_user_operand_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "OperandId index %zu is out of range for %zu matched operands",
                        index, supported_operand_ids_by_user_operand_.size());
    }
    return supported_operand_ids_by_user_operand_[index];
}

OperationId MatchResult::getSupportedOpId(OperationId user_op_id) const {
    const std::size_t index = user_op_id.getIndex();
    if (index >= supported_op_ids_by_user_op_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "OperationId index %zu is out of range for %zu matched operations", index,
                        supported_op_ids_by_user_op_.size());
    }
    return supported_op_ids_by_user_op_[index];
}

PatternCanonicalization Matcher::canonicalize(const Pattern& pattern) {
    CanonicalCandidate candidate = CanonicalSearchState(pattern).run();

    std::vector<std::size_t> canonical_operand_indices_by_pattern_operand(pattern.getNumOperands());
    for (std::size_t canonical_index = 0; canonical_index < candidate.pattern_operand_ids_by_canonical_operand.size();
         ++canonical_index) {
        canonical_operand_indices_by_pattern_operand[candidate.pattern_operand_ids_by_canonical_operand[canonical_index]
                                                         .getIndex()] = canonical_index;
    }

    std::vector<std::size_t> canonical_op_indices_by_pattern_op(pattern.getNumOps());
    for (std::size_t canonical_index = 0; canonical_index < candidate.pattern_op_ids_by_canonical_op.size();
         ++canonical_index) {
        canonical_op_indices_by_pattern_op[candidate.pattern_op_ids_by_canonical_op[canonical_index].getIndex()] =
            canonical_index;
    }

    return PatternCanonicalization(
        PatternKey(std::move(candidate.tokens)), std::move(canonical_operand_indices_by_pattern_operand),
        std::move(candidate.pattern_operand_ids_by_canonical_operand), std::move(canonical_op_indices_by_pattern_op),
        std::move(candidate.pattern_op_ids_by_canonical_op));
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
