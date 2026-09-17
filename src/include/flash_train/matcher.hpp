#ifndef FTRAIN_MATCHER_HPP_
#define FTRAIN_MATCHER_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/pattern.hpp"

namespace ftrain {

class Matcher;

// A structural fingerprint of a Pattern. Patterns built from the same
// operands and operations produce equal keys regardless of the order in which
// the pieces were added. The key does not reference the source Pattern.
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
    friend class Matcher;

    explicit PatternKey(std::vector<std::uint64_t>&& tokens) noexcept;

    std::vector<std::uint64_t> tokens_;
    std::size_t hash_;
};

class PatternKeyHasher final {
  public:
    std::size_t operator()(const PatternKey& key) const noexcept { return key.getHash(); }
};

// One Pattern's key plus a canonical numbering of its operands and
// operations. Structurally identical Patterns get the same numbering, so one
// canonical index refers to the same logical role in each of them. The
// index/ID accessor pairs are inverse bijections; out-of-range arguments
// throw Exception with FTRAIN_STATUS_INVALID_ARGUMENT. The object does not
// reference the source Pattern.
class PatternCanonicalization final {
  public:
    const PatternKey& getKey() const noexcept { return key_; }

    std::uint64_t getNumOperands() const noexcept { return canonical_operand_indices_by_pattern_operand_.size(); }

    std::uint64_t getNumOps() const noexcept { return canonical_op_indices_by_pattern_op_.size(); }

    // Returns the canonical index assigned to pattern_operand_id. An
    // out-of-range ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    std::size_t getCanonicalOperandIndex(OperandId pattern_operand_id) const;

    // Returns the Pattern operand ID assigned to canonical_operand_index. An
    // out-of-range index throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    OperandId getOperandId(std::size_t canonical_operand_index) const;

    // Returns the canonical index assigned to pattern_op_id. An out-of-range
    // ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    std::size_t getCanonicalOpIndex(OperationId pattern_op_id) const;

    // Returns the Pattern operation ID assigned to canonical_op_index. An
    // out-of-range index throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    OperationId getOperationId(std::size_t canonical_op_index) const;

  private:
    friend class Matcher;

    PatternCanonicalization(PatternKey&& key, std::vector<std::size_t>&& canonical_operand_indices_by_pattern_operand,
                            std::vector<OperandId>&& pattern_operand_ids_by_canonical_operand,
                            std::vector<std::size_t>&& canonical_op_indices_by_pattern_op,
                            std::vector<OperationId>&& pattern_op_ids_by_canonical_op) noexcept
        : key_(std::move(key)),
          canonical_operand_indices_by_pattern_operand_(std::move(canonical_operand_indices_by_pattern_operand)),
          pattern_operand_ids_by_canonical_operand_(std::move(pattern_operand_ids_by_canonical_operand)),
          canonical_op_indices_by_pattern_op_(std::move(canonical_op_indices_by_pattern_op)),
          pattern_op_ids_by_canonical_op_(std::move(pattern_op_ids_by_canonical_op)) {}

    PatternKey key_;
    std::vector<std::size_t> canonical_operand_indices_by_pattern_operand_;
    std::vector<OperandId> pattern_operand_ids_by_canonical_operand_;
    std::vector<std::size_t> canonical_op_indices_by_pattern_op_;
    std::vector<OperationId> pattern_op_ids_by_canonical_op_;
};

// One whole-structure match from Matcher::match(): for each user Pattern
// operand (operation) ID, the corresponding supported Pattern operand
// (operation) ID. Meaningful only with the Patterns passed to match(); the
// object retains neither. Out-of-range IDs throw Exception with
// FTRAIN_STATUS_INVALID_ARGUMENT.
class MatchResult final {
  public:
    std::size_t getNumMappedOperands() const noexcept { return supported_operand_ids_by_user_operand_.size(); }

    std::size_t getNumMappedOps() const noexcept { return supported_op_ids_by_user_op_.size(); }

    // Returns the supported Pattern operand assigned to user_operand_id. An
    // out-of-range ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    OperandId getSupportedOperandId(OperandId user_operand_id) const;

    // Returns the supported Pattern operation assigned to user_op_id. An
    // out-of-range ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    OperationId getSupportedOpId(OperationId user_op_id) const;

  private:
    friend class Matcher;

    MatchResult(std::vector<OperandId>&& supported_operand_ids_by_user_operand,
                std::vector<OperationId>&& supported_op_ids_by_user_op) noexcept
        : supported_operand_ids_by_user_operand_(std::move(supported_operand_ids_by_user_operand)),
          supported_op_ids_by_user_op_(std::move(supported_op_ids_by_user_op)) {}

    std::vector<OperandId> supported_operand_ids_by_user_operand_;
    std::vector<OperationId> supported_op_ids_by_user_op_;
};

class Matcher final {
  public:
    Matcher() = delete;

    // Computes pattern's PatternCanonicalization. An empty Pattern is valid.
    // The result owns its data and does not reference pattern. Allocation
    // failure throws std::bad_alloc.
    static PatternCanonicalization canonicalize(const Pattern& pattern);

    // Returns a complete one-to-one mapping from user_pattern's operands and
    // operations onto supported_pattern's, or std::nullopt when the structures
    // are not exactly equivalent. Insertion order and consumer-list order are
    // ignored; input and output port order is significant. Allocation failure
    // throws std::bad_alloc.
    static std::optional<MatchResult> match(const Pattern& user_pattern, const Pattern& supported_pattern);
};

}  // namespace ftrain

#endif
