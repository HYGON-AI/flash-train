#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <utility>

#include "flash_train/error.hpp"
#include "flash_train/pattern.hpp"

namespace ftrain {

// ------------------------------------------------------------------ Signature

namespace {

constexpr std::uint64_t kPatternKeyHashOffset = 14695981039346656037ULL;
constexpr std::uint64_t kPatternKeyHashPrime  = 1099511628211ULL;

void mixSignatureHash(std::uint64_t operand, std::uint64_t& hash) noexcept {
    hash ^= operand;
    hash *= kPatternKeyHashPrime;
    hash ^= operand >> 32;
    hash *= kPatternKeyHashPrime;
}

std::size_t hashPatternKeyTokens(const std::vector<std::uint64_t>& tokens) noexcept {
    std::uint64_t hash = kPatternKeyHashOffset;
    for (const std::uint64_t token : tokens) { mixSignatureHash(token, hash); }
    return static_cast<std::size_t>(hash);
}

// Vertex tags for structural signature tokens.
enum class SignatureTag : std::uint64_t {
    kOperand = 1,
    kOp,
    kProducer,
    kConsumers,
    kInputs,
    kOutputs,
};

// One vertex's round signature: its current color plus its wiring
// neighborhood, encoded so structurally identical vertices produce equal
// token vectors.
struct VertexSignature {
    bool is_operand;
    std::size_t vertex_index;
    std::vector<std::uint64_t> tokens;
};

// Assigns each vertex an absolute color by hashing its round signature, so
// colors are comparable across Patterns: structurally identical vertices
// hash equally wherever they live, and different kinds or port orders hash
// differently.
void assignSignatureColors(const std::vector<VertexSignature>& signatures, std::vector<std::uint64_t>& operand_colors,
                           std::vector<std::uint64_t>& op_colors) {
    for (const VertexSignature& signature : signatures) {
        const std::uint64_t color = static_cast<std::uint64_t>(hashPatternKeyTokens(signature.tokens));
        if (signature.is_operand) {
            operand_colors[signature.vertex_index] = color;
        } else {
            op_colors[signature.vertex_index] = color;
        }
    }
}

// Counts the distinct colors across both color vectors.
std::size_t countDistinctColors(const std::vector<std::uint64_t>& operand_colors,
                                const std::vector<std::uint64_t>& op_colors) {
    std::vector<std::uint64_t> colors;
    colors.reserve(operand_colors.size() + op_colors.size());
    colors.insert(colors.end(), operand_colors.begin(), operand_colors.end());
    colors.insert(colors.end(), op_colors.begin(), op_colors.end());
    std::sort(colors.begin(), colors.end());
    const auto unique_end = std::unique(colors.begin(), colors.end());
    return static_cast<std::size_t>(unique_end - colors.begin());
}

std::vector<std::uint64_t> makeOperandSignature(const std::vector<PatternOperandNode>& operand_nodes,
                                                PatternOperandId operand_id,
                                                const std::vector<std::uint64_t>& operand_colors,
                                                const std::vector<std::uint64_t>& op_colors) {
    const PatternOperandNode& node = operand_nodes[operand_id.getIndex()];

    std::vector<std::pair<std::uint64_t, std::uint64_t>> consumers;
    consumers.reserve(node.getConsumers().size());
    for (const PatternOperationInputPortId consumer : node.getConsumers()) {
        consumers.emplace_back(static_cast<std::uint64_t>(consumer.getPortIndex()),
                               op_colors[consumer.getOpId().getIndex()]);
    }
    std::sort(consumers.begin(), consumers.end());

    std::vector<std::uint64_t> tokens;
    tokens.reserve(8 + consumers.size() * 2);
    tokens.push_back(static_cast<std::uint64_t>(SignatureTag::kOperand));
    tokens.push_back(operand_colors[operand_id.getIndex()]);
    tokens.push_back(static_cast<std::uint64_t>(node.getKind()));
    tokens.push_back(static_cast<std::uint64_t>(SignatureTag::kProducer));
    tokens.push_back(node.getProducer().has_value() ? 1 : 0);
    if (node.getProducer().has_value()) {
        tokens.push_back(static_cast<std::uint64_t>(node.getProducer()->getPortIndex()));
        tokens.push_back(op_colors[node.getProducer()->getOpId().getIndex()]);
    }
    tokens.push_back(static_cast<std::uint64_t>(SignatureTag::kConsumers));
    tokens.push_back(consumers.size());
    for (const auto& consumer : consumers) {
        tokens.push_back(consumer.first);
        tokens.push_back(consumer.second);
    }
    return tokens;
}

std::vector<std::uint64_t> makeOpSignature(const std::vector<PatternOperationNode>& op_nodes, PatternOperationId op_id,
                                           const std::vector<std::uint64_t>& operand_colors,
                                           const std::vector<std::uint64_t>& op_colors) {
    const PatternOperationNode& node = op_nodes[op_id.getIndex()];

    // Input and output port order is significant, so port colors are encoded
    // in port order rather than sorted.
    std::vector<std::uint64_t> tokens;
    tokens.reserve(7 + node.getInputs().size() + node.getOutputs().size());
    tokens.push_back(static_cast<std::uint64_t>(SignatureTag::kOp));
    tokens.push_back(op_colors[op_id.getIndex()]);
    tokens.push_back(static_cast<std::uint64_t>(node.getKind()));
    tokens.push_back(static_cast<std::uint64_t>(SignatureTag::kInputs));
    tokens.push_back(node.getInputs().size());
    for (const PatternOperandId input : node.getInputs()) { tokens.push_back(operand_colors[input.getIndex()]); }
    tokens.push_back(static_cast<std::uint64_t>(SignatureTag::kOutputs));
    tokens.push_back(node.getOutputs().size());
    for (const PatternOperandId output : node.getOutputs()) { tokens.push_back(operand_colors[output.getIndex()]); }
    return tokens;
}

}  // namespace

// ----------------------------------------------------------------- Builder

Pattern PatternBuilder::buildPattern() const {
    // --- validate operand IDs and derive the reverse topology ---
    const std::size_t num_operands = operand_kinds_.size();
    std::vector<std::optional<PatternOperationOutputPortId>> producers(num_operands);
    std::vector<std::vector<PatternOperationInputPortId>> consumers(num_operands);

    for (std::size_t op_index = 0; op_index < op_nodes_.size(); ++op_index) {
        const PatternOperationNode& op_node = op_nodes_[op_index];

        for (std::size_t port_index = 0; port_index < op_node.getInputs().size(); ++port_index) {
            const std::uint64_t operand_index = op_node.getInputs()[port_index].getIndex();
            if (operand_index >= num_operands) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Operation %zu input port %zu references out-of-range operand " PRIu64
                                " for %zu operands",
                                op_index, port_index, operand_index, num_operands);
            }
            for (std::size_t earlier_port = 0; earlier_port < port_index; ++earlier_port) {
                if (op_node.getInputs()[earlier_port].getIndex() == operand_index) {
                    throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "Operation %zu repeats operand " PRIu64 " across input ports", op_index,
                                    operand_index);
                }
            }
            consumers[operand_index].emplace_back(PatternOperationId{op_index}, port_index);
        }

        for (std::size_t port_index = 0; port_index < op_node.getOutputs().size(); ++port_index) {
            const std::uint64_t operand_index = op_node.getOutputs()[port_index].getIndex();
            if (operand_index >= num_operands) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Operation %zu output port %zu references out-of-range operand " PRIu64
                                " for %zu operands",
                                op_index, port_index, operand_index, num_operands);
            }
            for (const PatternOperandId input : op_node.getInputs()) {
                if (input.getIndex() == operand_index) {
                    throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "Operation %zu output port %zu reuses operand " PRIu64 " from an input port",
                                    op_index, port_index, operand_index);
                }
            }
            for (std::size_t earlier_port = 0; earlier_port < port_index; ++earlier_port) {
                if (op_node.getOutputs()[earlier_port].getIndex() == operand_index) {
                    throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "Operation %zu repeats operand " PRIu64 " across output ports", op_index,
                                    operand_index);
                }
            }
            if (producers[operand_index].has_value()) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Operation %zu output port %zu reuses operand " PRIu64 " that already has a producer",
                                op_index, port_index, operand_index);
            }
            producers[operand_index].emplace(PatternOperationId{op_index}, port_index);
        }
    }

    // --- reject dependency cycles among operations ---
    // Op A precedes op B when B consumes an operand A produces; because
    // every operand has at most one producer, cycles in the operand graph
    // are exactly cycles in this op graph. Kahn's algorithm peels the
    // ops layer by layer; whatever cannot be peeled sits on a cycle.
    std::vector<std::vector<std::size_t>> successor_op_indices(op_nodes_.size());
    std::vector<std::size_t> remaining_in_degree(op_nodes_.size(), 0);
    for (std::size_t operand_index = 0; operand_index < num_operands; ++operand_index) {
        if (!producers[operand_index].has_value()) { continue; }
        const std::size_t producer_op_index = producers[operand_index]->getOpId().getIndex();
        for (const PatternOperationInputPortId consumer : consumers[operand_index]) {
            successor_op_indices[producer_op_index].push_back(consumer.getOpId().getIndex());
            ++remaining_in_degree[consumer.getOpId().getIndex()];
        }
    }

    std::vector<std::size_t> peelable_op_indices;
    for (std::size_t op_index = 0; op_index < op_nodes_.size(); ++op_index) {
        if (remaining_in_degree[op_index] == 0) { peelable_op_indices.push_back(op_index); }
    }
    std::size_t num_peeled_ops = 0;
    while (!peelable_op_indices.empty()) {
        const std::size_t op_index = peelable_op_indices.back();
        peelable_op_indices.pop_back();
        ++num_peeled_ops;
        for (const std::size_t successor_op_index : successor_op_indices[op_index]) {
            if (--remaining_in_degree[successor_op_index] == 0) { peelable_op_indices.push_back(successor_op_index); }
        }
    }
    if (num_peeled_ops != op_nodes_.size()) {
        for (std::size_t op_index = 0; op_index < op_nodes_.size(); ++op_index) {
            if (remaining_in_degree[op_index] == 0) { continue; }
            for (const PatternOperandId input : op_nodes_[op_index].getInputs()) {
                const auto& producer = producers[input.getIndex()];
                if (producer.has_value() && remaining_in_degree[producer->getOpId().getIndex()] != 0) {
                    throw Exception(
                        FTRAIN_STATUS_INVALID_ARGUMENT,
                        "Operation %zu input port participates in a dependency cycle through operand " PRIu64, op_index,
                        input.getIndex());
                }
            }
        }
        // Every unpeeled operation has an incoming edge from another unpeeled
        // producer, so the scan above always names one; this guard turns a
        // future logic break into a loud failure instead of a cyclic Pattern.
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "Cycle detection peeled %zu of %zu operations but named no edge",
                        num_peeled_ops, op_nodes_.size());
    }

    std::vector<PatternOperandNode> operand_nodes;
    operand_nodes.reserve(num_operands);
    for (std::size_t operand_index = 0; operand_index < num_operands; ++operand_index) {
        operand_nodes.emplace_back(operand_kinds_[operand_index], std::move(producers[operand_index]),
                                   std::move(consumers[operand_index]));
    }

    // --- copy the operation nodes ---
    std::vector<PatternOperationNode> op_nodes = op_nodes_;

    // --- derive the structural coloring and the key ---
    // Refine vertex colors until the distinct-color count is stable.
    std::vector<std::uint64_t> operand_colors(operand_nodes.size());
    std::vector<std::uint64_t> op_colors(op_nodes.size());
    std::size_t num_colors = 0;
    while (true) {
        std::vector<VertexSignature> signatures;
        signatures.reserve(operand_nodes.size() + op_nodes.size());
        for (std::size_t operand_index = 0; operand_index < operand_nodes.size(); ++operand_index) {
            signatures.push_back(VertexSignature{
                true, operand_index,
                makeOperandSignature(operand_nodes, PatternOperandId{operand_index}, operand_colors, op_colors)});
        }
        for (std::size_t op_index = 0; op_index < op_nodes.size(); ++op_index) {
            signatures.push_back(VertexSignature{
                false, op_index, makeOpSignature(op_nodes, PatternOperationId{op_index}, operand_colors, op_colors)});
        }

        std::vector<std::uint64_t> refined_operand_colors(operand_nodes.size());
        std::vector<std::uint64_t> refined_op_colors(op_nodes.size());
        assignSignatureColors(signatures, refined_operand_colors, refined_op_colors);
        const std::size_t refined_num_colors = countDistinctColors(refined_operand_colors, refined_op_colors);
        operand_colors                       = std::move(refined_operand_colors);
        op_colors                            = std::move(refined_op_colors);
        // The distinct count can never exceed the vertex total, so a full
        // split needs no confirming round.
        if (refined_num_colors == num_colors || refined_num_colors == operand_nodes.size() + op_nodes.size()) { break; }
        num_colors = refined_num_colors;
    }

    std::vector<std::uint64_t> tokens;
    tokens.reserve(operand_colors.size() + op_colors.size());
    for (const std::uint64_t color : operand_colors) { tokens.push_back(color); }
    for (const std::uint64_t color : op_colors) { tokens.push_back(color); }
    std::sort(tokens.begin(), tokens.end());

    return Pattern(std::move(operand_nodes), std::move(op_nodes), PatternKey(std::move(tokens)),
                   std::move(operand_colors), std::move(op_colors));
}

Pattern::Pattern(std::vector<PatternOperandNode>&& operand_nodes, std::vector<PatternOperationNode>&& op_nodes,
                 PatternKey&& key, std::vector<std::uint64_t>&& operand_colors,
                 std::vector<std::uint64_t>&& op_colors) noexcept
    : operand_nodes_(std::move(operand_nodes)), op_nodes_(std::move(op_nodes)), key_(std::move(key)),
      operand_colors_(std::move(operand_colors)), op_colors_(std::move(op_colors)) {}

const PatternOperandNode& Pattern::getOperandNode(PatternOperandId operand_id) const {
    const std::uint64_t index = operand_id.getIndex();
    if (index >= operand_nodes_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "PatternOperandId index " PRIu64 " is out of range for %zu operand nodes", index,
                        operand_nodes_.size());
    }
    return operand_nodes_[index];
}

const PatternOperationNode& Pattern::getOpNode(PatternOperationId op_id) const {
    const std::uint64_t index = op_id.getIndex();
    if (index >= op_nodes_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "PatternOperationId index " PRIu64 " is out of range for %zu operation nodes", index,
                        op_nodes_.size());
    }
    return op_nodes_[index];
}

PatternKey::PatternKey(std::vector<std::uint64_t>&& tokens) noexcept
    : tokens_(std::move(tokens)), hash_(hashPatternKeyTokens(tokens_)) {}

PatternKey::PatternKey(PatternKey&& other) noexcept : tokens_(std::move(other.tokens_)), hash_(other.hash_) {
    other.hash_ = hashPatternKeyTokens(other.tokens_);
}

PatternKey& PatternKey::operator=(PatternKey&& other) noexcept {
    if (this == &other) { return *this; }

    tokens_     = std::move(other.tokens_);
    hash_       = other.hash_;
    other.hash_ = hashPatternKeyTokens(other.tokens_);
    return *this;
}

}  // namespace ftrain
