#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <set>
#include <type_traits>
#include <utility>

#include "flash_train/error.hpp"
#include "flash_train/pattern.hpp"

namespace ftrain {

// ---------------------------------------------------------------- Builder

PatternOperandId PatternBuilder::appendOperand(OperandKind kind) {
    const std::size_t index = operand_kinds_.size();
    operand_kinds_.push_back(kind);
    return PatternOperandId{index};
}

PatternOperationId PatternBuilder::appendOperation(OperationKind kind, std::vector<PatternOperandId>&& inputs,
                                                   std::vector<PatternOperandId>&& outputs) {
    const std::size_t op_index = op_nodes_.size();
    op_nodes_.emplace_back(kind, std::move(inputs), std::move(outputs));
    return PatternOperationId{op_index};
}

OperandKind PatternBuilder::getOperandKind(PatternOperandId operand_id) const {
    const std::uint64_t index = operand_id.getIndex();
    if (index >= operand_kinds_.size()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "PatternOperandId index " PRIu64 " is out of range for %zu operands", index,
                        operand_kinds_.size());
    }
    return operand_kinds_[index];
}

// ------------------------------------------------------------ Pattern

// Validates the builder's stored operand IDs once and derives the reverse
// topology. Violations name the offending operation and port.
void Pattern::validateAndDeriveTopology(const PatternBuilder& pattern, std::vector<PatternOperandNode>& operand_nodes,
                                        const std::vector<PatternOperationNode>& op_nodes) {
    operand_nodes.reserve(pattern.getNumOperands());
    for (std::size_t operand_index = 0; operand_index < pattern.getNumOperands(); ++operand_index) {
        operand_nodes.emplace_back(pattern.getOperandKind(PatternOperandId{operand_index}));
    }

    for (std::size_t op_index = 0; op_index < op_nodes.size(); ++op_index) {
        const PatternOperationNode& op_node = op_nodes[op_index];

        std::set<std::uint64_t> seen_inputs;
        for (std::size_t port_index = 0; port_index < op_node.getInputs().size(); ++port_index) {
            const std::uint64_t operand_index = op_node.getInputs()[port_index].getIndex();
            if (operand_index >= operand_nodes.size()) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Operation %zu input port %zu references out-of-range operand " PRIu64
                                " for %zu operands",
                                op_index, port_index, operand_index, operand_nodes.size());
            }
            if (!seen_inputs.insert(operand_index).second) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Operation %zu repeats operand " PRIu64 " across input ports", op_index, operand_index);
            }
            operand_nodes[operand_index].consumers_.emplace_back(PatternOperationId{op_index}, port_index);
        }

        std::set<std::uint64_t> seen_outputs;
        for (std::size_t port_index = 0; port_index < op_node.getOutputs().size(); ++port_index) {
            const std::uint64_t operand_index = op_node.getOutputs()[port_index].getIndex();
            if (operand_index >= operand_nodes.size()) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Operation %zu output port %zu references out-of-range operand " PRIu64
                                " for %zu operands",
                                op_index, port_index, operand_index, operand_nodes.size());
            }
            if (seen_inputs.count(operand_index) != 0) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Operation %zu output port %zu reuses operand " PRIu64 " from an input port", op_index,
                                port_index, operand_index);
            }
            if (!seen_outputs.insert(operand_index).second) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Operation %zu repeats operand " PRIu64 " across output ports", op_index,
                                operand_index);
            }
            if (operand_nodes[operand_index].producer_.has_value()) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                "Operation %zu output port %zu reuses operand " PRIu64 " that already has a producer",
                                op_index, port_index, operand_index);
            }
            operand_nodes[operand_index].producer_.emplace(PatternOperationId{op_index}, port_index);
        }
    }
}

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

// One color per operand and per operation, refined until stable.
struct SignatureColoring {
    std::vector<std::uint64_t> operand_colors;
    std::vector<std::uint64_t> op_colors;
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
void assignSignatureColors(const std::vector<VertexSignature>& signatures, SignatureColoring& coloring) {
    for (const VertexSignature& signature : signatures) {
        const std::uint64_t color = static_cast<std::uint64_t>(hashPatternKeyTokens(signature.tokens));
        if (signature.is_operand) {
            coloring.operand_colors[signature.vertex_index] = color;
        } else {
            coloring.op_colors[signature.vertex_index] = color;
        }
    }
}

// Counts the distinct colors in one coloring.
std::size_t countDistinctColors(const SignatureColoring& coloring) {
    std::vector<std::uint64_t> colors;
    colors.reserve(coloring.operand_colors.size() + coloring.op_colors.size());
    colors.insert(colors.end(), coloring.operand_colors.begin(), coloring.operand_colors.end());
    colors.insert(colors.end(), coloring.op_colors.begin(), coloring.op_colors.end());
    std::sort(colors.begin(), colors.end());
    const auto unique_end = std::unique(colors.begin(), colors.end());
    return static_cast<std::size_t>(unique_end - colors.begin());
}

std::vector<std::uint64_t> makeOperandSignature(const std::vector<PatternOperandNode>& operand_nodes,
                                                PatternOperandId operand_id, const SignatureColoring& coloring) {
    const PatternOperandNode& node = operand_nodes[operand_id.getIndex()];

    std::vector<std::pair<std::uint64_t, std::uint64_t>> consumers;
    consumers.reserve(node.getConsumers().size());
    for (const PatternOperationInputPortId consumer : node.getConsumers()) {
        consumers.emplace_back(static_cast<std::uint64_t>(consumer.getPortIndex()),
                               coloring.op_colors[consumer.getOpId().getIndex()]);
    }
    std::sort(consumers.begin(), consumers.end());

    std::vector<std::uint64_t> tokens;
    tokens.reserve(8 + consumers.size() * 2);
    tokens.push_back(static_cast<std::uint64_t>(SignatureTag::kOperand));
    tokens.push_back(coloring.operand_colors[operand_id.getIndex()]);
    tokens.push_back(static_cast<std::uint64_t>(node.getKind()));
    tokens.push_back(static_cast<std::uint64_t>(SignatureTag::kProducer));
    tokens.push_back(node.getProducer().has_value() ? 1 : 0);
    if (node.getProducer().has_value()) {
        tokens.push_back(static_cast<std::uint64_t>(node.getProducer()->getPortIndex()));
        tokens.push_back(coloring.op_colors[node.getProducer()->getOpId().getIndex()]);
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
                                           const SignatureColoring& coloring) {
    const PatternOperationNode& node = op_nodes[op_id.getIndex()];

    // Input and output port order is significant, so port colors are encoded
    // in port order rather than sorted.
    std::vector<std::uint64_t> tokens;
    tokens.reserve(7 + node.getInputs().size() + node.getOutputs().size());
    tokens.push_back(static_cast<std::uint64_t>(SignatureTag::kOp));
    tokens.push_back(coloring.op_colors[op_id.getIndex()]);
    tokens.push_back(static_cast<std::uint64_t>(node.getKind()));
    tokens.push_back(static_cast<std::uint64_t>(SignatureTag::kInputs));
    tokens.push_back(node.getInputs().size());
    for (const PatternOperandId input : node.getInputs()) {
        tokens.push_back(coloring.operand_colors[input.getIndex()]);
    }
    tokens.push_back(static_cast<std::uint64_t>(SignatureTag::kOutputs));
    tokens.push_back(node.getOutputs().size());
    for (const PatternOperandId output : node.getOutputs()) {
        tokens.push_back(coloring.operand_colors[output.getIndex()]);
    }
    return tokens;
}

// Refines vertex colors until the distinct-color count is stable.
SignatureColoring computeSignatureColoring(const std::vector<PatternOperandNode>& operand_nodes,
                                           const std::vector<PatternOperationNode>& op_nodes) {
    SignatureColoring coloring{std::vector<std::uint64_t>(operand_nodes.size()),
                               std::vector<std::uint64_t>(op_nodes.size())};
    std::size_t num_colors = 0;
    while (true) {
        std::vector<VertexSignature> signatures;
        signatures.reserve(operand_nodes.size() + op_nodes.size());
        for (std::size_t operand_index = 0; operand_index < operand_nodes.size(); ++operand_index) {
            signatures.push_back(VertexSignature{
                true, operand_index, makeOperandSignature(operand_nodes, PatternOperandId{operand_index}, coloring)});
        }
        for (std::size_t op_index = 0; op_index < op_nodes.size(); ++op_index) {
            signatures.push_back(
                VertexSignature{false, op_index, makeOpSignature(op_nodes, PatternOperationId{op_index}, coloring)});
        }

        SignatureColoring refined{std::vector<std::uint64_t>(operand_nodes.size()),
                                  std::vector<std::uint64_t>(op_nodes.size())};
        assignSignatureColors(signatures, refined);
        const std::size_t refined_num_colors = countDistinctColors(refined);
        coloring                             = std::move(refined);
        if (refined_num_colors == num_colors) { return coloring; }
        num_colors = refined_num_colors;
    }
}

}  // namespace

Pattern PatternBuilder::buildPattern() const { return Pattern(Pattern::makeParts(*this)); }

Pattern::Pattern(Parts&& parts) noexcept
    : operand_nodes_(std::move(parts.operand_nodes)), op_nodes_(std::move(parts.op_nodes)), key_(std::move(parts.key)),
      operand_colors_(std::move(parts.operand_colors)), op_colors_(std::move(parts.op_colors)) {}

Pattern::Parts Pattern::makeParts(const PatternBuilder& pattern) {
    std::vector<PatternOperandNode> operand_nodes;
    std::vector<PatternOperationNode> op_nodes;
    validateAndDeriveTopology(pattern, operand_nodes, pattern.op_nodes_);
    op_nodes.reserve(pattern.getNumOps());
    for (const PatternOperationNode& op_node : pattern.op_nodes_) {
        op_nodes.emplace_back(op_node.getKind(), std::vector<PatternOperandId>(op_node.getInputs()),
                              std::vector<PatternOperandId>(op_node.getOutputs()));
    }

    SignatureColoring coloring = computeSignatureColoring(operand_nodes, op_nodes);
    std::vector<std::uint64_t> tokens;
    tokens.reserve(coloring.operand_colors.size() + coloring.op_colors.size());
    for (const std::uint64_t color : coloring.operand_colors) { tokens.push_back(color); }
    for (const std::uint64_t color : coloring.op_colors) { tokens.push_back(color); }
    std::sort(tokens.begin(), tokens.end());

    return Parts{std::move(operand_nodes), std::move(op_nodes), PatternKey(std::move(tokens)),
                 std::move(coloring.operand_colors), std::move(coloring.op_colors)};
}

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
