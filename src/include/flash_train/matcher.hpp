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

// One whole-structure match: for each user PatternBuilder operand (operation) ID,
// the corresponding supported PatternBuilder operand (operation) ID. Meaningful only
// with the Patterns passed to the producing call; the object retains neither.
// Out-of-range IDs throw Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
class MatchResult final {
  public:
    std::size_t getNumMappedOperands() const noexcept { return supported_operand_ids_by_user_operand_.size(); }

    std::size_t getNumMappedOps() const noexcept { return supported_op_ids_by_user_op_.size(); }

    // Returns the supported PatternBuilder operand assigned to user_operand_id. An
    // out-of-range ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    PatternOperandId getSupportedOperandId(PatternOperandId user_operand_id) const;

    // Returns the supported PatternBuilder operation assigned to user_op_id. An
    // out-of-range ID throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    PatternOperationId getSupportedOpId(PatternOperationId user_op_id) const;

  private:
    friend class Matcher;

    MatchResult(std::vector<PatternOperandId>&& supported_operand_ids_by_user_operand,
                std::vector<PatternOperationId>&& supported_op_ids_by_user_op) noexcept
        : supported_operand_ids_by_user_operand_(std::move(supported_operand_ids_by_user_operand)),
          supported_op_ids_by_user_op_(std::move(supported_op_ids_by_user_op)) {}

    std::vector<PatternOperandId> supported_operand_ids_by_user_operand_;
    std::vector<PatternOperationId> supported_op_ids_by_user_op_;
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

    static std::optional<MatchResult> match(const Pattern& user_pattern, const Pattern& supported_pattern);
};

}  // namespace ftrain

#endif
