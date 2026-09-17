#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/matcher.hpp"
#include "flash_train/op_definitions.hpp"
#include "flash_train/pattern.hpp"

namespace ftrain {
namespace {

template<typename Function>
void expectInvalidArgument(Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "Matcher result accepted an out-of-range identifier";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

void expectCompleteValidMapping(const Pattern& user_pattern, const Pattern& supported_pattern,
                                const MatchResult& result) {
    ASSERT_EQ(result.getNumMappedOperands(), user_pattern.getNumOperands());
    ASSERT_EQ(result.getNumMappedOps(), user_pattern.getNumOps());
    ASSERT_EQ(user_pattern.getNumOperands(), supported_pattern.getNumOperands());
    ASSERT_EQ(user_pattern.getNumOps(), supported_pattern.getNumOps());

    std::vector<bool> seen_supported_operands(supported_pattern.getNumOperands(), false);
    for (std::size_t user_operand_index = 0; user_operand_index < user_pattern.getNumOperands(); ++user_operand_index) {
        const OperandId user_operand_id{user_operand_index};
        const OperandId supported_operand_id = result.getSupportedOperandId(user_operand_id);
        ASSERT_LT(supported_operand_id.getIndex(), supported_pattern.getNumOperands());
        EXPECT_FALSE(seen_supported_operands[supported_operand_id.getIndex()]);
        seen_supported_operands[supported_operand_id.getIndex()] = true;
        EXPECT_EQ(user_pattern.getOperandNode(user_operand_id).getKind(),
                  supported_pattern.getOperandNode(supported_operand_id).getKind());
    }
    EXPECT_TRUE(
        std::all_of(seen_supported_operands.begin(), seen_supported_operands.end(), [](bool seen) { return seen; }));

    std::vector<bool> seen_supported_ops(supported_pattern.getNumOps(), false);
    for (std::size_t user_op_index = 0; user_op_index < user_pattern.getNumOps(); ++user_op_index) {
        const OperationId user_op_id{user_op_index};
        const OperationId supported_op_id = result.getSupportedOpId(user_op_id);
        ASSERT_LT(supported_op_id.getIndex(), supported_pattern.getNumOps());
        EXPECT_FALSE(seen_supported_ops[supported_op_id.getIndex()]);
        seen_supported_ops[supported_op_id.getIndex()] = true;

        const PatternOpNode& user_op_node      = user_pattern.getOpNode(user_op_id);
        const PatternOpNode& supported_op_node = supported_pattern.getOpNode(supported_op_id);
        ASSERT_EQ(user_op_node.getKind(), supported_op_node.getKind());
        ASSERT_EQ(user_op_node.getInputs().size(), supported_op_node.getInputs().size());
        ASSERT_EQ(user_op_node.getOutputs().size(), supported_op_node.getOutputs().size());

        for (std::size_t port_index = 0; port_index < user_op_node.getInputs().size(); ++port_index) {
            EXPECT_EQ(result.getSupportedOperandId(user_op_node.getInputs()[port_index]),
                      supported_op_node.getInputs()[port_index]);
        }
        for (std::size_t port_index = 0; port_index < user_op_node.getOutputs().size(); ++port_index) {
            EXPECT_EQ(result.getSupportedOperandId(user_op_node.getOutputs()[port_index]),
                      supported_op_node.getOutputs()[port_index]);
        }
    }
    EXPECT_TRUE(std::all_of(seen_supported_ops.begin(), seen_supported_ops.end(), [](bool seen) { return seen; }));
}

void expectCanonicalCompositionIsExact(const Pattern& user_pattern, const Pattern& supported_pattern,
                                       const PatternCanonicalization& user_canonicalization,
                                       const PatternCanonicalization& supported_canonicalization) {
    ASSERT_EQ(user_canonicalization.getKey(), supported_canonicalization.getKey());
    ASSERT_EQ(user_canonicalization.getNumOperands(), user_pattern.getNumOperands());
    ASSERT_EQ(user_canonicalization.getNumOps(), user_pattern.getNumOps());
    ASSERT_EQ(user_pattern.getNumOperands(), supported_pattern.getNumOperands());
    ASSERT_EQ(user_pattern.getNumOps(), supported_pattern.getNumOps());

    std::vector<OperandId> supported_operand_ids_by_user_operand;
    supported_operand_ids_by_user_operand.reserve(user_pattern.getNumOperands());
    std::vector<bool> seen_supported_operands(supported_pattern.getNumOperands(), false);
    for (std::size_t user_operand_index = 0; user_operand_index < user_pattern.getNumOperands(); ++user_operand_index) {
        const OperandId user_operand_id{user_operand_index};
        const std::size_t canonical_index = user_canonicalization.getCanonicalOperandIndex(user_operand_id);
        ASSERT_LT(canonical_index, user_pattern.getNumOperands());
        EXPECT_EQ(user_canonicalization.getOperandId(canonical_index), user_operand_id);

        const OperandId supported_operand_id = supported_canonicalization.getOperandId(canonical_index);
        ASSERT_LT(supported_operand_id.getIndex(), supported_pattern.getNumOperands());
        EXPECT_EQ(supported_canonicalization.getCanonicalOperandIndex(supported_operand_id), canonical_index);
        EXPECT_FALSE(seen_supported_operands[supported_operand_id.getIndex()]);
        seen_supported_operands[supported_operand_id.getIndex()] = true;
        EXPECT_EQ(user_pattern.getOperandNode(user_operand_id).getKind(),
                  supported_pattern.getOperandNode(supported_operand_id).getKind());
        supported_operand_ids_by_user_operand.push_back(supported_operand_id);
    }
    EXPECT_TRUE(
        std::all_of(seen_supported_operands.begin(), seen_supported_operands.end(), [](bool seen) { return seen; }));

    std::vector<OperationId> supported_op_ids_by_user_op;
    supported_op_ids_by_user_op.reserve(user_pattern.getNumOps());
    std::vector<bool> seen_supported_ops(supported_pattern.getNumOps(), false);
    for (std::size_t user_op_index = 0; user_op_index < user_pattern.getNumOps(); ++user_op_index) {
        const OperationId user_op_id{user_op_index};
        const std::size_t canonical_index = user_canonicalization.getCanonicalOpIndex(user_op_id);
        ASSERT_LT(canonical_index, user_pattern.getNumOps());
        EXPECT_EQ(user_canonicalization.getOperationId(canonical_index), user_op_id);

        const OperationId supported_op_id = supported_canonicalization.getOperationId(canonical_index);
        ASSERT_LT(supported_op_id.getIndex(), supported_pattern.getNumOps());
        EXPECT_EQ(supported_canonicalization.getCanonicalOpIndex(supported_op_id), canonical_index);
        EXPECT_FALSE(seen_supported_ops[supported_op_id.getIndex()]);
        seen_supported_ops[supported_op_id.getIndex()] = true;
        supported_op_ids_by_user_op.push_back(supported_op_id);
    }
    EXPECT_TRUE(std::all_of(seen_supported_ops.begin(), seen_supported_ops.end(), [](bool seen) { return seen; }));

    for (std::size_t user_op_index = 0; user_op_index < user_pattern.getNumOps(); ++user_op_index) {
        const PatternOpNode& user_op_node = user_pattern.getOpNode(OperationId{user_op_index});
        const PatternOpNode& supported_op_node =
            supported_pattern.getOpNode(supported_op_ids_by_user_op[user_op_index]);
        ASSERT_EQ(user_op_node.getKind(), supported_op_node.getKind());
        ASSERT_EQ(user_op_node.getInputs().size(), supported_op_node.getInputs().size());
        ASSERT_EQ(user_op_node.getOutputs().size(), supported_op_node.getOutputs().size());

        for (std::size_t port_index = 0; port_index < user_op_node.getInputs().size(); ++port_index) {
            EXPECT_EQ(supported_operand_ids_by_user_operand[user_op_node.getInputs()[port_index].getIndex()],
                      supported_op_node.getInputs()[port_index]);
        }
        for (std::size_t port_index = 0; port_index < user_op_node.getOutputs().size(); ++port_index) {
            EXPECT_EQ(supported_operand_ids_by_user_operand[user_op_node.getOutputs()[port_index].getIndex()],
                      supported_op_node.getOutputs()[port_index]);
        }
    }

    const std::optional<MatchResult> match_result = Matcher::match(user_pattern, supported_pattern);
    ASSERT_TRUE(match_result.has_value());
    expectCompleteValidMapping(user_pattern, supported_pattern, *match_result);
}

OperationId addGemm(Pattern& pattern, FTrainTensorId a, FTrainTensorId b, FTrainTensorId d) {
    const FTrainTensorId c     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId alpha = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId beta  = Operand<OperandKind::kTensor>::addToPattern(pattern);
    return OperationId{Operation<OperationKind::kGemm>::addToPattern(pattern, a, b, c, d, alpha, beta).opaque};
}

void addGemmCycle(Pattern& pattern, std::size_t cycle_length) {
    std::vector<FTrainTensorId> states;
    std::vector<FTrainTensorId> scales;
    states.reserve(cycle_length);
    scales.reserve(cycle_length);
    for (std::size_t index = 0; index < cycle_length; ++index) {
        states.push_back(Operand<OperandKind::kTensor>::addToPattern(pattern));
        scales.push_back(Operand<OperandKind::kTensor>::addToPattern(pattern));
    }
    for (std::size_t index = 0; index < cycle_length; ++index) {
        addGemm(pattern, states[index], scales[index], states[(index + 1) % cycle_length]);
    }
}

static_assert(!std::is_constructible_v<Matcher>);
static_assert(std::is_same_v<decltype(Matcher::canonicalize(std::declval<const Pattern&>())), PatternCanonicalization>);
static_assert(std::is_same_v<decltype(Matcher::match(std::declval<const Pattern&>(), std::declval<const Pattern&>())),
                             std::optional<MatchResult>>);
static_assert(noexcept(std::declval<const PatternCanonicalization&>().getKey()));
static_assert(std::is_nothrow_move_constructible_v<PatternKey>);
static_assert(std::is_nothrow_move_assignable_v<PatternKey>);
static_assert(noexcept(std::declval<const PatternCanonicalization&>().getNumOperands()));
static_assert(noexcept(std::declval<const PatternCanonicalization&>().getNumOps()));
static_assert(noexcept(std::declval<const MatchResult&>().getNumMappedOperands()));
static_assert(noexcept(std::declval<const MatchResult&>().getNumMappedOps()));
static_assert(
    std::is_same_v<decltype(std::declval<const MatchResult&>().getSupportedOperandId(OperandId{0})), OperandId>);
static_assert(
    std::is_same_v<decltype(std::declval<const MatchResult&>().getSupportedOpId(OperationId{0})), OperationId>);

TEST(MatcherTest, MapsRolesAcrossDifferentAdditionOrders) {
    Pattern supported_pattern;
    const FTrainTensorId supported_input        = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_scale  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_intermediate = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_scale = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_output       = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const OperationId supported_first_op =
        addGemm(supported_pattern, supported_input, supported_first_scale, supported_intermediate);
    const OperationId supported_second_op =
        addGemm(supported_pattern, supported_intermediate, supported_second_scale, supported_output);

    Pattern user_pattern;
    const FTrainTensorId user_output       = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_scale = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_intermediate = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_scale  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_input        = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const OperationId user_second_op       = addGemm(user_pattern, user_intermediate, user_second_scale, user_output);
    const OperationId user_first_op        = addGemm(user_pattern, user_input, user_first_scale, user_intermediate);

    const std::optional<MatchResult> result = Matcher::match(user_pattern, supported_pattern);

    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(user_pattern, supported_pattern, *result);
    EXPECT_EQ(result->getSupportedOperandId(OperandId{user_input.opaque}), OperandId{supported_input.opaque});
    EXPECT_EQ(result->getSupportedOperandId(OperandId{user_first_scale.opaque}),
              OperandId{supported_first_scale.opaque});
    EXPECT_EQ(result->getSupportedOperandId(OperandId{user_intermediate.opaque}),
              OperandId{supported_intermediate.opaque});
    EXPECT_EQ(result->getSupportedOperandId(OperandId{user_second_scale.opaque}),
              OperandId{supported_second_scale.opaque});
    EXPECT_EQ(result->getSupportedOperandId(OperandId{user_output.opaque}), OperandId{supported_output.opaque});
    EXPECT_EQ(result->getSupportedOpId(user_first_op), supported_first_op);
    EXPECT_EQ(result->getSupportedOpId(user_second_op), supported_second_op);
}

TEST(MatchResultTest, ReportsMappingSizesAndRejectsOutOfRangeUserIds) {
    Pattern supported_pattern;
    const FTrainTensorId supported_input  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_scale  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_output = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const OperationId supported_op = addGemm(supported_pattern, supported_input, supported_scale, supported_output);

    Pattern user_pattern;
    const FTrainTensorId user_input  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_scale  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_output = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const OperationId user_op        = addGemm(user_pattern, user_input, user_scale, user_output);

    const std::optional<MatchResult> result = Matcher::match(user_pattern, supported_pattern);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->getNumMappedOperands(), 6);
    EXPECT_EQ(result->getNumMappedOps(), 1);
    EXPECT_EQ(result->getSupportedOperandId(OperandId{user_input.opaque}), OperandId{supported_input.opaque});
    EXPECT_EQ(result->getSupportedOperandId(OperandId{user_scale.opaque}), OperandId{supported_scale.opaque});
    EXPECT_EQ(result->getSupportedOperandId(OperandId{user_output.opaque}), OperandId{supported_output.opaque});
    EXPECT_EQ(result->getSupportedOpId(user_op), supported_op);
    expectInvalidArgument([&] { static_cast<void>(result->getSupportedOperandId(OperandId{6})); });
    expectInvalidArgument([&] { static_cast<void>(result->getSupportedOpId(OperationId{1})); });
}

TEST(MatcherTest, MatchesAllDisconnectedComponentsAndRejectsMissingOrExtraComponents) {
    Pattern supported_pattern;
    const FTrainTensorId supported_first_input   = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_scale   = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_output  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_input  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_scale  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_output = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    addGemm(supported_pattern, supported_first_input, supported_first_scale, supported_first_output);
    addGemm(supported_pattern, supported_second_input, supported_second_scale, supported_second_output);

    Pattern user_pattern;
    const FTrainTensorId user_second_output = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_scale  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_input  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_output  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_scale   = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_input   = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    addGemm(user_pattern, user_second_input, user_second_scale, user_second_output);
    addGemm(user_pattern, user_first_input, user_first_scale, user_first_output);

    const std::optional<MatchResult> result = Matcher::match(user_pattern, supported_pattern);
    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(user_pattern, supported_pattern, *result);

    Pattern missing_component_user_pattern;
    const FTrainTensorId missing_input  = Operand<OperandKind::kTensor>::addToPattern(missing_component_user_pattern);
    const FTrainTensorId missing_scale  = Operand<OperandKind::kTensor>::addToPattern(missing_component_user_pattern);
    const FTrainTensorId missing_output = Operand<OperandKind::kTensor>::addToPattern(missing_component_user_pattern);
    addGemm(missing_component_user_pattern, missing_input, missing_scale, missing_output);
    EXPECT_FALSE(Matcher::match(missing_component_user_pattern, supported_pattern).has_value());

    const FTrainTensorId extra_input  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId extra_scale  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId extra_output = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    addGemm(user_pattern, extra_input, extra_scale, extra_output);
    EXPECT_FALSE(Matcher::match(user_pattern, supported_pattern).has_value());
}

TEST(MatcherTest, DistinguishesDisconnectedComponentsFromConnectedTopologyWithEqualCounts) {
    Pattern supported_pattern;
    const FTrainTensorId first_input   = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId first_scale   = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId first_output  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId second_input  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId second_scale  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId second_output = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    addGemm(supported_pattern, first_input, first_scale, first_output);
    addGemm(supported_pattern, second_input, second_scale, second_output);

    Pattern user_pattern;
    const FTrainTensorId chain_input        = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId chain_intermediate = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId chain_output       = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId chain_first_scale  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId chain_second_scale = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    static_cast<void>(Operand<OperandKind::kTensor>::addToPattern(user_pattern));
    addGemm(user_pattern, chain_intermediate, chain_second_scale, chain_output);
    addGemm(user_pattern, chain_input, chain_first_scale, chain_intermediate);

    EXPECT_FALSE(Matcher::match(user_pattern, supported_pattern).has_value());
}

TEST(MatcherTest, RejectsOperandAndOperationCountMismatchesIndependently) {
    Pattern supported_pattern;
    static_cast<void>(Operand<OperandKind::kTensor>::addToPattern(supported_pattern));
    static_cast<void>(Operand<OperandKind::kTensor>::addToPattern(supported_pattern));

    Pattern user_pattern_with_extra_operand;
    static_cast<void>(Operand<OperandKind::kTensor>::addToPattern(user_pattern_with_extra_operand));
    static_cast<void>(Operand<OperandKind::kTensor>::addToPattern(user_pattern_with_extra_operand));
    static_cast<void>(Operand<OperandKind::kTensor>::addToPattern(user_pattern_with_extra_operand));
    EXPECT_FALSE(Matcher::match(user_pattern_with_extra_operand, supported_pattern).has_value());

    Pattern user_pattern_with_extra_op;
    const FTrainTensorId input  = Operand<OperandKind::kTensor>::addToPattern(user_pattern_with_extra_op);
    const FTrainTensorId scale  = Operand<OperandKind::kTensor>::addToPattern(user_pattern_with_extra_op);
    const FTrainTensorId output = Operand<OperandKind::kTensor>::addToPattern(user_pattern_with_extra_op);
    addGemm(user_pattern_with_extra_op, input, scale, output);
    EXPECT_FALSE(Matcher::match(user_pattern_with_extra_op, supported_pattern).has_value());
}

TEST(MatcherTest, RequiresFanOutTopologyInBothDirections) {
    Pattern fan_out_supported_pattern;
    const FTrainTensorId shared_input  = Operand<OperandKind::kTensor>::addToPattern(fan_out_supported_pattern);
    const FTrainTensorId first_scale   = Operand<OperandKind::kTensor>::addToPattern(fan_out_supported_pattern);
    const FTrainTensorId first_output  = Operand<OperandKind::kTensor>::addToPattern(fan_out_supported_pattern);
    const FTrainTensorId second_scale  = Operand<OperandKind::kTensor>::addToPattern(fan_out_supported_pattern);
    const FTrainTensorId second_output = Operand<OperandKind::kTensor>::addToPattern(fan_out_supported_pattern);
    static_cast<void>(Operand<OperandKind::kTensor>::addToPattern(fan_out_supported_pattern));
    addGemm(fan_out_supported_pattern, shared_input, first_scale, first_output);
    addGemm(fan_out_supported_pattern, shared_input, second_scale, second_output);

    Pattern independent_user_pattern;
    const FTrainTensorId first_input_user   = Operand<OperandKind::kTensor>::addToPattern(independent_user_pattern);
    const FTrainTensorId first_scale_user   = Operand<OperandKind::kTensor>::addToPattern(independent_user_pattern);
    const FTrainTensorId first_output_user  = Operand<OperandKind::kTensor>::addToPattern(independent_user_pattern);
    const FTrainTensorId second_input_user  = Operand<OperandKind::kTensor>::addToPattern(independent_user_pattern);
    const FTrainTensorId second_scale_user  = Operand<OperandKind::kTensor>::addToPattern(independent_user_pattern);
    const FTrainTensorId second_output_user = Operand<OperandKind::kTensor>::addToPattern(independent_user_pattern);
    addGemm(independent_user_pattern, first_input_user, first_scale_user, first_output_user);
    addGemm(independent_user_pattern, second_input_user, second_scale_user, second_output_user);
    EXPECT_FALSE(Matcher::match(independent_user_pattern, fan_out_supported_pattern).has_value());

    Pattern fan_out_user_pattern;
    const FTrainTensorId shared_input_user          = Operand<OperandKind::kTensor>::addToPattern(fan_out_user_pattern);
    const FTrainTensorId first_scale_user_pattern   = Operand<OperandKind::kTensor>::addToPattern(fan_out_user_pattern);
    const FTrainTensorId first_output_user_pattern  = Operand<OperandKind::kTensor>::addToPattern(fan_out_user_pattern);
    const FTrainTensorId second_scale_user_pattern  = Operand<OperandKind::kTensor>::addToPattern(fan_out_user_pattern);
    const FTrainTensorId second_output_user_pattern = Operand<OperandKind::kTensor>::addToPattern(fan_out_user_pattern);
    static_cast<void>(Operand<OperandKind::kTensor>::addToPattern(fan_out_user_pattern));
    addGemm(fan_out_user_pattern, shared_input_user, second_scale_user_pattern, second_output_user_pattern);
    addGemm(fan_out_user_pattern, shared_input_user, first_scale_user_pattern, first_output_user_pattern);

    const std::optional<MatchResult> fan_out_result = Matcher::match(fan_out_user_pattern, fan_out_supported_pattern);
    ASSERT_TRUE(fan_out_result.has_value());
    expectCompleteValidMapping(fan_out_user_pattern, fan_out_supported_pattern, *fan_out_result);
}

TEST(MatcherTest, PreservesOrderedOperationPorts) {
    Pattern supported_pattern;
    const FTrainTensorId supported_source       = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_source_scale = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_intermediate = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_other        = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_output       = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    addGemm(supported_pattern, supported_source, supported_source_scale, supported_intermediate);
    addGemm(supported_pattern, supported_intermediate, supported_other, supported_output);

    Pattern user_pattern;
    const FTrainTensorId user_source       = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_source_scale = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_intermediate = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_other        = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_output       = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    addGemm(user_pattern, user_source, user_source_scale, user_intermediate);
    addGemm(user_pattern, user_other, user_intermediate, user_output);

    EXPECT_FALSE(Matcher::match(user_pattern, supported_pattern).has_value());
}

TEST(MatcherTest, IgnoresConsumerInsertionOrder) {
    Pattern supported_pattern;
    const FTrainTensorId supported_shared        = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_scale   = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_output  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_scale  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_output = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    addGemm(supported_pattern, supported_shared, supported_first_scale, supported_first_output);
    addGemm(supported_pattern, supported_shared, supported_second_scale, supported_second_output);

    Pattern user_pattern;
    const FTrainTensorId user_second_output = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_scale  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_shared        = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_scale   = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_output  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    addGemm(user_pattern, user_shared, user_second_scale, user_second_output);
    addGemm(user_pattern, user_shared, user_first_scale, user_first_output);

    const std::optional<MatchResult> result = Matcher::match(user_pattern, supported_pattern);
    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(user_pattern, supported_pattern, *result);
}

TEST(MatcherTest, ReturnsAnyCompleteValidMappingForSymmetricComponents) {
    Pattern supported_pattern;
    const FTrainTensorId first_input   = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId first_scale   = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId first_output  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId second_input  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId second_scale  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId second_output = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    addGemm(supported_pattern, first_input, first_scale, first_output);
    addGemm(supported_pattern, second_input, second_scale, second_output);

    Pattern user_pattern;
    const FTrainTensorId second_output_user = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId second_scale_user  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId second_input_user  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId first_output_user  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId first_scale_user   = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId first_input_user   = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    addGemm(user_pattern, second_input_user, second_scale_user, second_output_user);
    addGemm(user_pattern, first_input_user, first_scale_user, first_output_user);

    const std::optional<MatchResult> result = Matcher::match(user_pattern, supported_pattern);
    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(user_pattern, supported_pattern, *result);
}

TEST(MatcherTest, BacktracksFromACompatibleLocalCandidateToFindTheCompleteMapping) {
    Pattern supported_pattern;
    std::vector<FTrainTensorId> short_supported_operands;
    std::vector<FTrainTensorId> short_supported_scales;
    for (std::size_t index = 0; index < 3; ++index) {
        short_supported_operands.push_back(Operand<OperandKind::kTensor>::addToPattern(supported_pattern));
        if (index + 1 < 3) {
            short_supported_scales.push_back(Operand<OperandKind::kTensor>::addToPattern(supported_pattern));
        }
    }
    for (std::size_t index = 0; index + 1 < short_supported_operands.size(); ++index) {
        addGemm(supported_pattern, short_supported_operands[index], short_supported_scales[index],
                short_supported_operands[index + 1]);
    }

    std::vector<FTrainTensorId> long_supported_operands;
    std::vector<FTrainTensorId> long_supported_scales;
    for (std::size_t index = 0; index < 5; ++index) {
        long_supported_operands.push_back(Operand<OperandKind::kTensor>::addToPattern(supported_pattern));
        if (index + 1 < 5) {
            long_supported_scales.push_back(Operand<OperandKind::kTensor>::addToPattern(supported_pattern));
        }
    }
    for (std::size_t index = 0; index + 1 < long_supported_operands.size(); ++index) {
        addGemm(supported_pattern, long_supported_operands[index], long_supported_scales[index],
                long_supported_operands[index + 1]);
    }

    Pattern user_pattern;
    std::vector<FTrainTensorId> long_user_operands;
    std::vector<FTrainTensorId> long_user_scales;
    for (std::size_t index = 0; index < 5; ++index) {
        long_user_operands.push_back(Operand<OperandKind::kTensor>::addToPattern(user_pattern));
        if (index + 1 < 5) { long_user_scales.push_back(Operand<OperandKind::kTensor>::addToPattern(user_pattern)); }
    }
    for (std::size_t index = 0; index + 1 < long_user_operands.size(); ++index) {
        addGemm(user_pattern, long_user_operands[index], long_user_scales[index], long_user_operands[index + 1]);
    }

    std::vector<FTrainTensorId> short_user_operands;
    std::vector<FTrainTensorId> short_user_scales;
    for (std::size_t index = 0; index < 3; ++index) {
        short_user_operands.push_back(Operand<OperandKind::kTensor>::addToPattern(user_pattern));
        if (index + 1 < 3) { short_user_scales.push_back(Operand<OperandKind::kTensor>::addToPattern(user_pattern)); }
    }
    for (std::size_t index = 0; index + 1 < short_user_operands.size(); ++index) {
        addGemm(user_pattern, short_user_operands[index], short_user_scales[index], short_user_operands[index + 1]);
    }

    const std::optional<MatchResult> result = Matcher::match(user_pattern, supported_pattern);

    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(user_pattern, supported_pattern, *result);
    EXPECT_EQ(result->getSupportedOperandId(OperandId{short_user_operands.front().opaque}),
              OperandId{short_supported_operands.front().opaque});
    EXPECT_EQ(result->getSupportedOperandId(OperandId{long_user_operands.front().opaque}),
              OperandId{long_supported_operands.front().opaque});
}

TEST(MatcherTest, RejectsOperationAndIsolatedOperandKindMismatches) {
    Pattern supported_pattern;
    const FTrainTensorId supported_input  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_scale  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_output = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    addGemm(supported_pattern, supported_input, supported_scale, supported_output);

    Pattern different_op_pattern;
    const FTrainGroupedTensorId different_a = Operand<OperandKind::kGroupedTensor>::addToPattern(different_op_pattern);
    const FTrainGroupedTensorId different_b = Operand<OperandKind::kGroupedTensor>::addToPattern(different_op_pattern);
    const FTrainGroupedTensorId different_c = Operand<OperandKind::kGroupedTensor>::addToPattern(different_op_pattern);
    const FTrainTensorId different_alpha    = Operand<OperandKind::kTensor>::addToPattern(different_op_pattern);
    const FTrainTensorId different_beta     = Operand<OperandKind::kTensor>::addToPattern(different_op_pattern);
    const FTrainGroupedTensorId different_output =
        Operand<OperandKind::kGroupedTensor>::addToPattern(different_op_pattern);
    static_cast<void>(Operation<OperationKind::kGroupedABCDGemm>::addToPattern(
        different_op_pattern, different_a, different_b, different_c, different_output, different_alpha,
        different_beta));
    EXPECT_FALSE(Matcher::match(different_op_pattern, supported_pattern).has_value());

    Pattern isolated_list_supported_pattern;
    const FTrainTensorId isolated_supported_input =
        Operand<OperandKind::kTensor>::addToPattern(isolated_list_supported_pattern);
    const FTrainTensorId isolated_supported_scale =
        Operand<OperandKind::kTensor>::addToPattern(isolated_list_supported_pattern);
    const FTrainTensorId isolated_supported_output =
        Operand<OperandKind::kTensor>::addToPattern(isolated_list_supported_pattern);
    static_cast<void>(Operand<OperandKind::kTensorList>::addToPattern(isolated_list_supported_pattern));
    addGemm(isolated_list_supported_pattern, isolated_supported_input, isolated_supported_scale,
            isolated_supported_output);

    Pattern isolated_grouped_user_pattern;
    const FTrainTensorId isolated_user_input =
        Operand<OperandKind::kTensor>::addToPattern(isolated_grouped_user_pattern);
    const FTrainTensorId isolated_user_scale =
        Operand<OperandKind::kTensor>::addToPattern(isolated_grouped_user_pattern);
    const FTrainTensorId isolated_user_output =
        Operand<OperandKind::kTensor>::addToPattern(isolated_grouped_user_pattern);
    static_cast<void>(Operand<OperandKind::kGroupedTensor>::addToPattern(isolated_grouped_user_pattern));
    addGemm(isolated_grouped_user_pattern, isolated_user_input, isolated_user_scale, isolated_user_output);
    EXPECT_FALSE(Matcher::match(isolated_grouped_user_pattern, isolated_list_supported_pattern).has_value());

    Pattern isolated_list_user_pattern;
    const FTrainTensorId list_user_input  = Operand<OperandKind::kTensor>::addToPattern(isolated_list_user_pattern);
    const FTrainTensorId list_user_scale  = Operand<OperandKind::kTensor>::addToPattern(isolated_list_user_pattern);
    const FTrainTensorId list_user_output = Operand<OperandKind::kTensor>::addToPattern(isolated_list_user_pattern);
    static_cast<void>(Operand<OperandKind::kTensorList>::addToPattern(isolated_list_user_pattern));
    addGemm(isolated_list_user_pattern, list_user_input, list_user_scale, list_user_output);
    const std::optional<MatchResult> result =
        Matcher::match(isolated_list_user_pattern, isolated_list_supported_pattern);
    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(isolated_list_user_pattern, isolated_list_supported_pattern, *result);
}

TEST(PatternCanonicalizationTest, CanonicalizesEmptyPatternDeterministicallyAndChecksAccessorBounds) {
    const Pattern pattern;

    const PatternCanonicalization first  = Matcher::canonicalize(pattern);
    const PatternCanonicalization second = Matcher::canonicalize(pattern);

    EXPECT_EQ(first.getKey(), second.getKey());
    EXPECT_EQ(first.getNumOperands(), 0);
    EXPECT_EQ(first.getNumOps(), 0);
    expectInvalidArgument([&] { static_cast<void>(first.getCanonicalOperandIndex(OperandId{0})); });
    expectInvalidArgument([&] { static_cast<void>(first.getOperandId(0)); });
    expectInvalidArgument([&] { static_cast<void>(first.getCanonicalOpIndex(OperationId{0})); });
    expectInvalidArgument([&] { static_cast<void>(first.getOperationId(0)); });
}

TEST(PatternCanonicalizationTest, PreservesPatternKeyHashInvariantAcrossMoves) {
    Pattern tensor_pattern;
    static_cast<void>(Operand<OperandKind::kTensor>::addToPattern(tensor_pattern));
    Pattern list_pattern;
    static_cast<void>(Operand<OperandKind::kTensorList>::addToPattern(list_pattern));

    PatternKey tensor_key                = Matcher::canonicalize(tensor_pattern).getKey();
    PatternKey list_key                  = Matcher::canonicalize(list_pattern).getKey();
    const PatternKey expected_tensor_key = tensor_key;
    const PatternKey expected_list_key   = list_key;

    PatternKey moved_tensor_key(std::move(tensor_key));
    PatternKey moved_list_key = expected_tensor_key;
    moved_list_key            = std::move(list_key);

    EXPECT_EQ(moved_tensor_key, expected_tensor_key);
    EXPECT_EQ(moved_list_key, expected_list_key);
    if (tensor_key == list_key) { EXPECT_EQ(tensor_key.getHash(), list_key.getHash()); }
}

TEST(PatternCanonicalizationTest, ComposesRolesAcrossDifferentOperandAndOperationAdditionOrders) {
    Pattern supported_pattern;
    const FTrainTensorId supported_input        = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_scale  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_intermediate = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_scale = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_output       = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const OperationId supported_first_op =
        addGemm(supported_pattern, supported_input, supported_first_scale, supported_intermediate);
    const OperationId supported_second_op =
        addGemm(supported_pattern, supported_intermediate, supported_second_scale, supported_output);

    Pattern user_pattern;
    const FTrainTensorId user_output       = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_scale = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_intermediate = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_scale  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_input        = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const OperationId user_second_op       = addGemm(user_pattern, user_intermediate, user_second_scale, user_output);
    const OperationId user_first_op        = addGemm(user_pattern, user_input, user_first_scale, user_intermediate);

    const PatternCanonicalization supported = Matcher::canonicalize(supported_pattern);
    const PatternCanonicalization user      = Matcher::canonicalize(user_pattern);

    expectCanonicalCompositionIsExact(user_pattern, supported_pattern, user, supported);
    const auto get_supported_operand_id = [&](OperandId user_operand_id) {
        return supported.getOperandId(user.getCanonicalOperandIndex(user_operand_id));
    };
    const auto get_supported_op_id = [&](OperationId user_op_id) {
        return supported.getOperationId(user.getCanonicalOpIndex(user_op_id));
    };
    EXPECT_EQ(get_supported_operand_id(OperandId{user_input.opaque}), OperandId{supported_input.opaque});
    EXPECT_EQ(get_supported_operand_id(OperandId{user_first_scale.opaque}), OperandId{supported_first_scale.opaque});
    EXPECT_EQ(get_supported_operand_id(OperandId{user_intermediate.opaque}), OperandId{supported_intermediate.opaque});
    EXPECT_EQ(get_supported_operand_id(OperandId{user_second_scale.opaque}), OperandId{supported_second_scale.opaque});
    EXPECT_EQ(get_supported_operand_id(OperandId{user_output.opaque}), OperandId{supported_output.opaque});
    EXPECT_EQ(get_supported_op_id(user_first_op), supported_first_op);
    EXPECT_EQ(get_supported_op_id(user_second_op), supported_second_op);
}

TEST(PatternCanonicalizationTest, IgnoresConsumerInsertionOrder) {
    Pattern supported_pattern;
    const FTrainTensorId supported_shared        = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_other   = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_output  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_other  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_output = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    addGemm(supported_pattern, supported_shared, supported_first_other, supported_first_output);
    addGemm(supported_pattern, supported_second_other, supported_shared, supported_second_output);

    Pattern user_pattern;
    const FTrainTensorId user_second_other  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_output = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_other   = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_output  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_shared        = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    addGemm(user_pattern, user_second_other, user_shared, user_second_output);
    addGemm(user_pattern, user_shared, user_first_other, user_first_output);

    const PatternCanonicalization supported = Matcher::canonicalize(supported_pattern);
    const PatternCanonicalization user      = Matcher::canonicalize(user_pattern);

    expectCanonicalCompositionIsExact(user_pattern, supported_pattern, user, supported);
}

TEST(PatternCanonicalizationTest, CanonicalizesSymmetricFanOut) {
    Pattern supported_pattern;
    const FTrainTensorId supported_shared        = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_scale   = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_output  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_scale  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_output = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    addGemm(supported_pattern, supported_shared, supported_first_scale, supported_first_output);
    addGemm(supported_pattern, supported_shared, supported_second_scale, supported_second_output);

    Pattern user_pattern;
    const FTrainTensorId user_second_output = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_scale  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_shared        = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_output  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_scale   = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    addGemm(user_pattern, user_shared, user_second_scale, user_second_output);
    addGemm(user_pattern, user_shared, user_first_scale, user_first_output);

    const PatternCanonicalization supported = Matcher::canonicalize(supported_pattern);
    const PatternCanonicalization user      = Matcher::canonicalize(user_pattern);

    expectCanonicalCompositionIsExact(user_pattern, supported_pattern, user, supported);
}

TEST(PatternCanonicalizationTest, CanonicalizesSymmetricDisconnectedComponents) {
    Pattern supported_pattern;
    for (std::size_t component = 0; component < 2; ++component) {
        const FTrainTensorId input  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
        const FTrainTensorId scale  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
        const FTrainTensorId output = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
        addGemm(supported_pattern, input, scale, output);
    }

    Pattern user_pattern;
    std::array<FTrainTensorId, 2> inputs{Operand<OperandKind::kTensor>::addToPattern(user_pattern),
                                         Operand<OperandKind::kTensor>::addToPattern(user_pattern)};
    std::array<FTrainTensorId, 2> scales{Operand<OperandKind::kTensor>::addToPattern(user_pattern),
                                         Operand<OperandKind::kTensor>::addToPattern(user_pattern)};
    std::array<FTrainTensorId, 2> outputs{Operand<OperandKind::kTensor>::addToPattern(user_pattern),
                                          Operand<OperandKind::kTensor>::addToPattern(user_pattern)};
    for (std::size_t component = 2; component != 0; --component) {
        const std::size_t index = component - 1;
        addGemm(user_pattern, inputs[index], scales[index], outputs[index]);
    }

    const PatternCanonicalization supported = Matcher::canonicalize(supported_pattern);
    const PatternCanonicalization user      = Matcher::canonicalize(user_pattern);

    expectCanonicalCompositionIsExact(user_pattern, supported_pattern, user, supported);
}

TEST(PatternCanonicalizationTest, PreservesOrderedInputPorts) {
    Pattern ordered_inputs;
    const FTrainTensorId input_source       = Operand<OperandKind::kTensor>::addToPattern(ordered_inputs);
    const FTrainTensorId input_source_scale = Operand<OperandKind::kTensor>::addToPattern(ordered_inputs);
    const FTrainTensorId input_intermediate = Operand<OperandKind::kTensor>::addToPattern(ordered_inputs);
    const FTrainTensorId input_other        = Operand<OperandKind::kTensor>::addToPattern(ordered_inputs);
    const FTrainTensorId input_output       = Operand<OperandKind::kTensor>::addToPattern(ordered_inputs);
    addGemm(ordered_inputs, input_source, input_source_scale, input_intermediate);
    addGemm(ordered_inputs, input_intermediate, input_other, input_output);

    Pattern swapped_inputs;
    const FTrainTensorId swapped_source       = Operand<OperandKind::kTensor>::addToPattern(swapped_inputs);
    const FTrainTensorId swapped_source_scale = Operand<OperandKind::kTensor>::addToPattern(swapped_inputs);
    const FTrainTensorId swapped_intermediate = Operand<OperandKind::kTensor>::addToPattern(swapped_inputs);
    const FTrainTensorId swapped_other        = Operand<OperandKind::kTensor>::addToPattern(swapped_inputs);
    const FTrainTensorId swapped_output       = Operand<OperandKind::kTensor>::addToPattern(swapped_inputs);
    addGemm(swapped_inputs, swapped_source, swapped_source_scale, swapped_intermediate);
    addGemm(swapped_inputs, swapped_other, swapped_intermediate, swapped_output);

    EXPECT_NE(Matcher::canonicalize(ordered_inputs).getKey(), Matcher::canonicalize(swapped_inputs).getKey());
    EXPECT_FALSE(Matcher::match(ordered_inputs, swapped_inputs).has_value());
}

TEST(PatternCanonicalizationTest, DistinguishesKindsAndGloballyDifferentTopology) {
    Pattern tensor_pattern;
    static_cast<void>(Operand<OperandKind::kTensor>::addToPattern(tensor_pattern));
    Pattern tensor_list_pattern;
    static_cast<void>(Operand<OperandKind::kTensorList>::addToPattern(tensor_list_pattern));

    EXPECT_NE(Matcher::canonicalize(tensor_pattern).getKey(), Matcher::canonicalize(tensor_list_pattern).getKey());
    EXPECT_FALSE(Matcher::match(tensor_pattern, tensor_list_pattern).has_value());

    Pattern six_cycle;
    addGemmCycle(six_cycle, 6);
    Pattern two_three_cycles;
    addGemmCycle(two_three_cycles, 3);
    addGemmCycle(two_three_cycles, 3);

    EXPECT_NE(Matcher::canonicalize(six_cycle).getKey(), Matcher::canonicalize(two_three_cycles).getKey());
    EXPECT_FALSE(Matcher::match(six_cycle, two_three_cycles).has_value());
}

}  // namespace
}  // namespace ftrain
