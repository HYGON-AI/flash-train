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
#include "flash_train/operation/operation.hpp"
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
        const PatternOperandId user_operand_id{user_operand_index};
        const PatternOperandId supported_operand_id = result.getSupportedOperandId(user_operand_id);
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
        const PatternOperationId user_op_id{user_op_index};
        const PatternOperationId supported_op_id = result.getSupportedOpId(user_op_id);
        ASSERT_LT(supported_op_id.getIndex(), supported_pattern.getNumOps());
        EXPECT_FALSE(seen_supported_ops[supported_op_id.getIndex()]);
        seen_supported_ops[supported_op_id.getIndex()] = true;

        const PatternOperationNode& user_op_node      = user_pattern.getOpNode(user_op_id);
        const PatternOperationNode& supported_op_node = supported_pattern.getOpNode(supported_op_id);
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

void expectSignatureCorrespondence(const PatternBuilder& user_builder, const PatternBuilder& supported_builder) {
    const Pattern user_pattern      = user_builder.buildPattern();
    const Pattern supported_pattern = supported_builder.buildPattern();

    ASSERT_EQ(user_pattern.getKey(), supported_pattern.getKey());
    ASSERT_EQ(user_pattern.getNumOperands(), user_builder.getNumOperands());
    ASSERT_EQ(user_pattern.getNumOps(), user_builder.getNumOps());

    const std::optional<MatchResult> fast = Matcher::matchBySignature(user_pattern, supported_pattern);
    if (fast.has_value()) {
        expectCompleteValidMapping(user_pattern, supported_pattern, *fast);
        return;
    }
    // Symmetric structures whose index-order pairing does not verify fall
    // back to the exact search, exactly as production matching does.
    const std::optional<MatchResult> exact = Matcher::match(user_pattern, supported_pattern);
    ASSERT_TRUE(exact.has_value());
    expectCompleteValidMapping(user_pattern, supported_pattern, *exact);
}

PatternOperationId addGemm(PatternBuilder& pattern, FTrainTensorId a, FTrainTensorId b, FTrainTensorId d) {
    const FTrainTensorId c     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId alpha = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId beta  = pattern.addOperand<OperandKind::kTensor>();
    return PatternOperationId{pattern.addOperation<OperationKind::kGemm>({a, b, c, d, alpha, beta}).opaque};
}

void addGemmCycle(PatternBuilder& pattern, std::size_t cycle_length) {
    std::vector<FTrainTensorId> states;
    std::vector<FTrainTensorId> scales;
    states.reserve(cycle_length);
    scales.reserve(cycle_length);
    for (std::size_t index = 0; index < cycle_length; ++index) {
        states.push_back(pattern.addOperand<OperandKind::kTensor>());
        scales.push_back(pattern.addOperand<OperandKind::kTensor>());
    }
    for (std::size_t index = 0; index < cycle_length; ++index) {
        addGemm(pattern, states[index], scales[index], states[(index + 1) % cycle_length]);
    }
}

static_assert(!std::is_constructible_v<Matcher>);
static_assert(std::is_same_v<decltype(std::declval<const PatternBuilder&>().buildPattern()), Pattern>);
static_assert(std::is_same_v<decltype(Matcher::match(std::declval<const Pattern&>(), std::declval<const Pattern&>())),
                             std::optional<MatchResult>>);
static_assert(noexcept(std::declval<const Pattern&>().getKey()));
static_assert(std::is_nothrow_move_constructible_v<PatternKey>);
static_assert(std::is_nothrow_move_assignable_v<PatternKey>);
static_assert(noexcept(std::declval<const Pattern&>().getNumOperands()));
static_assert(noexcept(std::declval<const Pattern&>().getNumOps()));
static_assert(noexcept(std::declval<const MatchResult&>().getNumMappedOperands()));
static_assert(noexcept(std::declval<const MatchResult&>().getNumMappedOps()));
static_assert(std::is_same_v<decltype(std::declval<const MatchResult&>().getSupportedOperandId(PatternOperandId{0})),
                             PatternOperandId>);
static_assert(std::is_same_v<decltype(std::declval<const MatchResult&>().getSupportedOpId(PatternOperationId{0})),
                             PatternOperationId>);

TEST(MatcherTest, MapsRolesAcrossDifferentAdditionOrders) {
    PatternBuilder supported_pattern;
    const FTrainTensorId supported_input        = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_scale  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_intermediate = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_scale = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_output       = supported_pattern.addOperand<OperandKind::kTensor>();
    const PatternOperationId supported_first_op =
        addGemm(supported_pattern, supported_input, supported_first_scale, supported_intermediate);
    const PatternOperationId supported_second_op =
        addGemm(supported_pattern, supported_intermediate, supported_second_scale, supported_output);

    PatternBuilder user_pattern;
    const FTrainTensorId user_output        = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_scale  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_intermediate  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_scale   = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_input         = user_pattern.addOperand<OperandKind::kTensor>();
    const PatternOperationId user_second_op = addGemm(user_pattern, user_intermediate, user_second_scale, user_output);
    const PatternOperationId user_first_op  = addGemm(user_pattern, user_input, user_first_scale, user_intermediate);

    const std::optional<MatchResult> result =
        Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern());

    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(user_pattern.buildPattern(), supported_pattern.buildPattern(), *result);
    EXPECT_EQ(result->getSupportedOperandId(PatternOperandId{user_input.opaque}),
              PatternOperandId{supported_input.opaque});
    EXPECT_EQ(result->getSupportedOperandId(PatternOperandId{user_first_scale.opaque}),
              PatternOperandId{supported_first_scale.opaque});
    EXPECT_EQ(result->getSupportedOperandId(PatternOperandId{user_intermediate.opaque}),
              PatternOperandId{supported_intermediate.opaque});
    EXPECT_EQ(result->getSupportedOperandId(PatternOperandId{user_second_scale.opaque}),
              PatternOperandId{supported_second_scale.opaque});
    EXPECT_EQ(result->getSupportedOperandId(PatternOperandId{user_output.opaque}),
              PatternOperandId{supported_output.opaque});
    EXPECT_EQ(result->getSupportedOpId(user_first_op), supported_first_op);
    EXPECT_EQ(result->getSupportedOpId(user_second_op), supported_second_op);
}

TEST(MatchResultTest, ReportsMappingSizesAndRejectsOutOfRangeUserIds) {
    PatternBuilder supported_pattern;
    const FTrainTensorId supported_input  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_scale  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_output = supported_pattern.addOperand<OperandKind::kTensor>();
    const PatternOperationId supported_op =
        addGemm(supported_pattern, supported_input, supported_scale, supported_output);

    PatternBuilder user_pattern;
    const FTrainTensorId user_input  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_scale  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_output = user_pattern.addOperand<OperandKind::kTensor>();
    const PatternOperationId user_op = addGemm(user_pattern, user_input, user_scale, user_output);

    const std::optional<MatchResult> result =
        Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern());

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->getNumMappedOperands(), 6);
    EXPECT_EQ(result->getNumMappedOps(), 1);
    EXPECT_EQ(result->getSupportedOperandId(PatternOperandId{user_input.opaque}),
              PatternOperandId{supported_input.opaque});
    EXPECT_EQ(result->getSupportedOperandId(PatternOperandId{user_scale.opaque}),
              PatternOperandId{supported_scale.opaque});
    EXPECT_EQ(result->getSupportedOperandId(PatternOperandId{user_output.opaque}),
              PatternOperandId{supported_output.opaque});
    EXPECT_EQ(result->getSupportedOpId(user_op), supported_op);
    expectInvalidArgument([&] { static_cast<void>(result->getSupportedOperandId(PatternOperandId{6})); });
    expectInvalidArgument([&] { static_cast<void>(result->getSupportedOpId(PatternOperationId{1})); });
}

TEST(MatcherTest, MatchesAllDisconnectedComponentsAndRejectsMissingOrExtraComponents) {
    PatternBuilder supported_pattern;
    const FTrainTensorId supported_first_input   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_scale   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_output  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_input  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_scale  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_output = supported_pattern.addOperand<OperandKind::kTensor>();
    addGemm(supported_pattern, supported_first_input, supported_first_scale, supported_first_output);
    addGemm(supported_pattern, supported_second_input, supported_second_scale, supported_second_output);

    PatternBuilder user_pattern;
    const FTrainTensorId user_second_output = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_scale  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_input  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_output  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_scale   = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_input   = user_pattern.addOperand<OperandKind::kTensor>();
    addGemm(user_pattern, user_second_input, user_second_scale, user_second_output);
    addGemm(user_pattern, user_first_input, user_first_scale, user_first_output);

    const std::optional<MatchResult> result =
        Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern());
    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(user_pattern.buildPattern(), supported_pattern.buildPattern(), *result);

    PatternBuilder missing_component_user_pattern;
    const FTrainTensorId missing_input  = missing_component_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId missing_scale  = missing_component_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId missing_output = missing_component_user_pattern.addOperand<OperandKind::kTensor>();
    addGemm(missing_component_user_pattern, missing_input, missing_scale, missing_output);
    EXPECT_FALSE(
        Matcher::match(missing_component_user_pattern.buildPattern(), supported_pattern.buildPattern()).has_value());

    const FTrainTensorId extra_input  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId extra_scale  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId extra_output = user_pattern.addOperand<OperandKind::kTensor>();
    addGemm(user_pattern, extra_input, extra_scale, extra_output);
    EXPECT_FALSE(Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern()).has_value());
}

TEST(MatcherTest, DistinguishesDisconnectedComponentsFromConnectedTopologyWithEqualCounts) {
    PatternBuilder supported_pattern;
    const FTrainTensorId first_input   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_scale   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_output  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_input  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_scale  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_output = supported_pattern.addOperand<OperandKind::kTensor>();
    addGemm(supported_pattern, first_input, first_scale, first_output);
    addGemm(supported_pattern, second_input, second_scale, second_output);

    PatternBuilder user_pattern;
    const FTrainTensorId chain_input        = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId chain_intermediate = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId chain_output       = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId chain_first_scale  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId chain_second_scale = user_pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(user_pattern.addOperand<OperandKind::kTensor>());
    addGemm(user_pattern, chain_intermediate, chain_second_scale, chain_output);
    addGemm(user_pattern, chain_input, chain_first_scale, chain_intermediate);

    EXPECT_FALSE(Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern()).has_value());
}

TEST(MatcherTest, RejectsOperandAndOperationCountMismatchesIndependently) {
    PatternBuilder supported_pattern;
    static_cast<void>(supported_pattern.addOperand<OperandKind::kTensor>());
    static_cast<void>(supported_pattern.addOperand<OperandKind::kTensor>());

    PatternBuilder user_pattern_with_extra_operand;
    static_cast<void>(user_pattern_with_extra_operand.addOperand<OperandKind::kTensor>());
    static_cast<void>(user_pattern_with_extra_operand.addOperand<OperandKind::kTensor>());
    static_cast<void>(user_pattern_with_extra_operand.addOperand<OperandKind::kTensor>());
    EXPECT_FALSE(
        Matcher::match(user_pattern_with_extra_operand.buildPattern(), supported_pattern.buildPattern()).has_value());

    PatternBuilder user_pattern_with_extra_op;
    const FTrainTensorId input  = user_pattern_with_extra_op.addOperand<OperandKind::kTensor>();
    const FTrainTensorId scale  = user_pattern_with_extra_op.addOperand<OperandKind::kTensor>();
    const FTrainTensorId output = user_pattern_with_extra_op.addOperand<OperandKind::kTensor>();
    addGemm(user_pattern_with_extra_op, input, scale, output);
    EXPECT_FALSE(
        Matcher::match(user_pattern_with_extra_op.buildPattern(), supported_pattern.buildPattern()).has_value());
}

TEST(MatcherTest, RequiresFanOutTopologyInBothDirections) {
    PatternBuilder fan_out_supported_pattern;
    const FTrainTensorId shared_input  = fan_out_supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_scale   = fan_out_supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_output  = fan_out_supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_scale  = fan_out_supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_output = fan_out_supported_pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(fan_out_supported_pattern.addOperand<OperandKind::kTensor>());
    addGemm(fan_out_supported_pattern, shared_input, first_scale, first_output);
    addGemm(fan_out_supported_pattern, shared_input, second_scale, second_output);

    PatternBuilder independent_user_pattern;
    const FTrainTensorId first_input_user   = independent_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_scale_user   = independent_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_output_user  = independent_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_input_user  = independent_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_scale_user  = independent_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_output_user = independent_user_pattern.addOperand<OperandKind::kTensor>();
    addGemm(independent_user_pattern, first_input_user, first_scale_user, first_output_user);
    addGemm(independent_user_pattern, second_input_user, second_scale_user, second_output_user);
    EXPECT_FALSE(
        Matcher::match(independent_user_pattern.buildPattern(), fan_out_supported_pattern.buildPattern()).has_value());

    PatternBuilder fan_out_user_pattern;
    const FTrainTensorId shared_input_user          = fan_out_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_scale_user_pattern   = fan_out_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_output_user_pattern  = fan_out_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_scale_user_pattern  = fan_out_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_output_user_pattern = fan_out_user_pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(fan_out_user_pattern.addOperand<OperandKind::kTensor>());
    addGemm(fan_out_user_pattern, shared_input_user, second_scale_user_pattern, second_output_user_pattern);
    addGemm(fan_out_user_pattern, shared_input_user, first_scale_user_pattern, first_output_user_pattern);

    const std::optional<MatchResult> fan_out_result =
        Matcher::match(fan_out_user_pattern.buildPattern(), fan_out_supported_pattern.buildPattern());
    ASSERT_TRUE(fan_out_result.has_value());
    expectCompleteValidMapping(fan_out_user_pattern.buildPattern(), fan_out_supported_pattern.buildPattern(),
                               *fan_out_result);
}

TEST(MatcherTest, PreservesOrderedOperationPorts) {
    PatternBuilder supported_pattern;
    const FTrainTensorId supported_source       = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_source_scale = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_intermediate = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_other        = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_output       = supported_pattern.addOperand<OperandKind::kTensor>();
    addGemm(supported_pattern, supported_source, supported_source_scale, supported_intermediate);
    addGemm(supported_pattern, supported_intermediate, supported_other, supported_output);

    PatternBuilder user_pattern;
    const FTrainTensorId user_source       = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_source_scale = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_intermediate = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_other        = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_output       = user_pattern.addOperand<OperandKind::kTensor>();
    addGemm(user_pattern, user_source, user_source_scale, user_intermediate);
    addGemm(user_pattern, user_other, user_intermediate, user_output);

    EXPECT_FALSE(Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern()).has_value());
}

TEST(MatcherTest, IgnoresConsumerInsertionOrder) {
    PatternBuilder supported_pattern;
    const FTrainTensorId supported_shared        = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_scale   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_output  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_scale  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_output = supported_pattern.addOperand<OperandKind::kTensor>();
    addGemm(supported_pattern, supported_shared, supported_first_scale, supported_first_output);
    addGemm(supported_pattern, supported_shared, supported_second_scale, supported_second_output);

    PatternBuilder user_pattern;
    const FTrainTensorId user_second_output = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_scale  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_shared        = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_scale   = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_output  = user_pattern.addOperand<OperandKind::kTensor>();
    addGemm(user_pattern, user_shared, user_second_scale, user_second_output);
    addGemm(user_pattern, user_shared, user_first_scale, user_first_output);

    const std::optional<MatchResult> result =
        Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern());
    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(user_pattern.buildPattern(), supported_pattern.buildPattern(), *result);
}

TEST(MatcherTest, ReturnsAnyCompleteValidMappingForSymmetricComponents) {
    PatternBuilder supported_pattern;
    const FTrainTensorId first_input   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_scale   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_output  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_input  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_scale  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_output = supported_pattern.addOperand<OperandKind::kTensor>();
    addGemm(supported_pattern, first_input, first_scale, first_output);
    addGemm(supported_pattern, second_input, second_scale, second_output);

    PatternBuilder user_pattern;
    const FTrainTensorId second_output_user = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_scale_user  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_input_user  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_output_user  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_scale_user   = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_input_user   = user_pattern.addOperand<OperandKind::kTensor>();
    addGemm(user_pattern, second_input_user, second_scale_user, second_output_user);
    addGemm(user_pattern, first_input_user, first_scale_user, first_output_user);

    const std::optional<MatchResult> result =
        Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern());
    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(user_pattern.buildPattern(), supported_pattern.buildPattern(), *result);
}

TEST(MatcherTest, BacktracksFromACompatibleLocalCandidateToFindTheCompleteMapping) {
    PatternBuilder supported_pattern;
    std::vector<FTrainTensorId> short_supported_operands;
    std::vector<FTrainTensorId> short_supported_scales;
    for (std::size_t index = 0; index < 3; ++index) {
        short_supported_operands.push_back(supported_pattern.addOperand<OperandKind::kTensor>());
        if (index + 1 < 3) { short_supported_scales.push_back(supported_pattern.addOperand<OperandKind::kTensor>()); }
    }
    for (std::size_t index = 0; index + 1 < short_supported_operands.size(); ++index) {
        addGemm(supported_pattern, short_supported_operands[index], short_supported_scales[index],
                short_supported_operands[index + 1]);
    }

    std::vector<FTrainTensorId> long_supported_operands;
    std::vector<FTrainTensorId> long_supported_scales;
    for (std::size_t index = 0; index < 5; ++index) {
        long_supported_operands.push_back(supported_pattern.addOperand<OperandKind::kTensor>());
        if (index + 1 < 5) { long_supported_scales.push_back(supported_pattern.addOperand<OperandKind::kTensor>()); }
    }
    for (std::size_t index = 0; index + 1 < long_supported_operands.size(); ++index) {
        addGemm(supported_pattern, long_supported_operands[index], long_supported_scales[index],
                long_supported_operands[index + 1]);
    }

    PatternBuilder user_pattern;
    std::vector<FTrainTensorId> long_user_operands;
    std::vector<FTrainTensorId> long_user_scales;
    for (std::size_t index = 0; index < 5; ++index) {
        long_user_operands.push_back(user_pattern.addOperand<OperandKind::kTensor>());
        if (index + 1 < 5) { long_user_scales.push_back(user_pattern.addOperand<OperandKind::kTensor>()); }
    }
    for (std::size_t index = 0; index + 1 < long_user_operands.size(); ++index) {
        addGemm(user_pattern, long_user_operands[index], long_user_scales[index], long_user_operands[index + 1]);
    }

    std::vector<FTrainTensorId> short_user_operands;
    std::vector<FTrainTensorId> short_user_scales;
    for (std::size_t index = 0; index < 3; ++index) {
        short_user_operands.push_back(user_pattern.addOperand<OperandKind::kTensor>());
        if (index + 1 < 3) { short_user_scales.push_back(user_pattern.addOperand<OperandKind::kTensor>()); }
    }
    for (std::size_t index = 0; index + 1 < short_user_operands.size(); ++index) {
        addGemm(user_pattern, short_user_operands[index], short_user_scales[index], short_user_operands[index + 1]);
    }

    const std::optional<MatchResult> result =
        Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern());

    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(user_pattern.buildPattern(), supported_pattern.buildPattern(), *result);
    EXPECT_EQ(result->getSupportedOperandId(PatternOperandId{short_user_operands.front().opaque}),
              PatternOperandId{short_supported_operands.front().opaque});
    EXPECT_EQ(result->getSupportedOperandId(PatternOperandId{long_user_operands.front().opaque}),
              PatternOperandId{long_supported_operands.front().opaque});
}

TEST(MatcherTest, RejectsOperationAndIsolatedOperandKindMismatches) {
    PatternBuilder supported_pattern;
    const FTrainTensorId supported_input  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_scale  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_output = supported_pattern.addOperand<OperandKind::kTensor>();
    addGemm(supported_pattern, supported_input, supported_scale, supported_output);

    PatternBuilder different_op_pattern;
    const FTrainGroupedTensorId different_a      = different_op_pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId different_b      = different_op_pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId different_c      = different_op_pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainTensorId different_alpha         = different_op_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId different_beta          = different_op_pattern.addOperand<OperandKind::kTensor>();
    const FTrainGroupedTensorId different_output = different_op_pattern.addOperand<OperandKind::kGroupedTensor>();
    static_cast<void>(different_op_pattern.addOperation<OperationKind::kGroupedABCDGemm>(
        {different_a, different_b, different_c, different_output, different_alpha, different_beta}));
    EXPECT_FALSE(Matcher::match(different_op_pattern.buildPattern(), supported_pattern.buildPattern()).has_value());

    PatternBuilder isolated_list_supported_pattern;
    const FTrainTensorId isolated_supported_input  = isolated_list_supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId isolated_supported_scale  = isolated_list_supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId isolated_supported_output = isolated_list_supported_pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(isolated_list_supported_pattern.addOperand<OperandKind::kTensorList>());
    addGemm(isolated_list_supported_pattern, isolated_supported_input, isolated_supported_scale,
            isolated_supported_output);

    PatternBuilder isolated_grouped_user_pattern;
    const FTrainTensorId isolated_user_input  = isolated_grouped_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId isolated_user_scale  = isolated_grouped_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId isolated_user_output = isolated_grouped_user_pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(isolated_grouped_user_pattern.addOperand<OperandKind::kGroupedTensor>());
    addGemm(isolated_grouped_user_pattern, isolated_user_input, isolated_user_scale, isolated_user_output);
    EXPECT_FALSE(
        Matcher::match(isolated_grouped_user_pattern.buildPattern(), isolated_list_supported_pattern.buildPattern())
            .has_value());

    PatternBuilder isolated_list_user_pattern;
    const FTrainTensorId list_user_input  = isolated_list_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId list_user_scale  = isolated_list_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId list_user_output = isolated_list_user_pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(isolated_list_user_pattern.addOperand<OperandKind::kTensorList>());
    addGemm(isolated_list_user_pattern, list_user_input, list_user_scale, list_user_output);
    const std::optional<MatchResult> result =
        Matcher::match(isolated_list_user_pattern.buildPattern(), isolated_list_supported_pattern.buildPattern());
    ASSERT_TRUE(result.has_value());
    expectCompleteValidMapping(isolated_list_user_pattern.buildPattern(),
                               isolated_list_supported_pattern.buildPattern(), *result);
}

TEST(PatternStructuralColoringTest, SignsEmptyPatternDeterministically) {
    const PatternBuilder pattern;
    const Pattern first  = pattern.buildPattern();
    const Pattern second = pattern.buildPattern();

    EXPECT_EQ(first.getKey(), second.getKey());
    EXPECT_EQ(first.getNumOperands(), 0);
    EXPECT_EQ(first.getNumOps(), 0);
    EXPECT_TRUE(first.getOperandColors().empty());
    EXPECT_TRUE(first.getOpColors().empty());
}

TEST(PatternStructuralColoringTest, PreservesPatternKeyHashInvariantAcrossMoves) {
    PatternBuilder tensor_pattern;
    static_cast<void>(tensor_pattern.addOperand<OperandKind::kTensor>());
    PatternBuilder list_pattern;
    static_cast<void>(list_pattern.addOperand<OperandKind::kTensorList>());

    PatternKey tensor_key                = tensor_pattern.buildPattern().getKey();
    PatternKey list_key                  = list_pattern.buildPattern().getKey();
    const PatternKey expected_tensor_key = tensor_key;
    const PatternKey expected_list_key   = list_key;

    PatternKey moved_tensor_key(std::move(tensor_key));
    PatternKey moved_list_key = expected_tensor_key;
    moved_list_key            = std::move(list_key);

    EXPECT_EQ(moved_tensor_key, expected_tensor_key);
    EXPECT_EQ(moved_list_key, expected_list_key);
    if (tensor_key == list_key) { EXPECT_EQ(tensor_key.getHash(), list_key.getHash()); }
}

TEST(PatternStructuralColoringTest, ComposesRolesAcrossDifferentOperandAndOperationAdditionOrders) {
    PatternBuilder supported_pattern;
    const FTrainTensorId supported_input        = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_scale  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_intermediate = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_scale = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_output       = supported_pattern.addOperand<OperandKind::kTensor>();
    const PatternOperationId supported_first_op =
        addGemm(supported_pattern, supported_input, supported_first_scale, supported_intermediate);
    const PatternOperationId supported_second_op =
        addGemm(supported_pattern, supported_intermediate, supported_second_scale, supported_output);

    PatternBuilder user_pattern;
    const FTrainTensorId user_output        = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_scale  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_intermediate  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_scale   = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_input         = user_pattern.addOperand<OperandKind::kTensor>();
    const PatternOperationId user_second_op = addGemm(user_pattern, user_intermediate, user_second_scale, user_output);
    const PatternOperationId user_first_op  = addGemm(user_pattern, user_input, user_first_scale, user_intermediate);

    expectSignatureCorrespondence(user_pattern, supported_pattern);
    const std::optional<MatchResult> result =
        Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern());
    ASSERT_TRUE(result.has_value());
    const auto& mapped = *result;
    EXPECT_EQ(mapped.getSupportedOperandId(PatternOperandId{user_input.opaque}),
              PatternOperandId{supported_input.opaque});
    EXPECT_EQ(mapped.getSupportedOperandId(PatternOperandId{user_first_scale.opaque}),
              PatternOperandId{supported_first_scale.opaque});
    EXPECT_EQ(mapped.getSupportedOperandId(PatternOperandId{user_intermediate.opaque}),
              PatternOperandId{supported_intermediate.opaque});
    EXPECT_EQ(mapped.getSupportedOperandId(PatternOperandId{user_second_scale.opaque}),
              PatternOperandId{supported_second_scale.opaque});
    EXPECT_EQ(mapped.getSupportedOperandId(PatternOperandId{user_output.opaque}),
              PatternOperandId{supported_output.opaque});
    EXPECT_EQ(mapped.getSupportedOpId(user_first_op), supported_first_op);
    EXPECT_EQ(mapped.getSupportedOpId(user_second_op), supported_second_op);
}

TEST(PatternStructuralColoringTest, IgnoresConsumerInsertionOrder) {
    PatternBuilder supported_pattern;
    const FTrainTensorId supported_shared        = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_other   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_output  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_other  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_output = supported_pattern.addOperand<OperandKind::kTensor>();
    addGemm(supported_pattern, supported_shared, supported_first_other, supported_first_output);
    addGemm(supported_pattern, supported_second_other, supported_shared, supported_second_output);

    PatternBuilder user_pattern;
    const FTrainTensorId user_second_other  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_output = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_other   = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_output  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_shared        = user_pattern.addOperand<OperandKind::kTensor>();
    addGemm(user_pattern, user_second_other, user_shared, user_second_output);
    addGemm(user_pattern, user_shared, user_first_other, user_first_output);

    expectSignatureCorrespondence(user_pattern, supported_pattern);
}

TEST(PatternStructuralColoringTest, CanonicalizesSymmetricFanOut) {
    PatternBuilder supported_pattern;
    const FTrainTensorId supported_shared        = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_scale   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_output  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_scale  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_output = supported_pattern.addOperand<OperandKind::kTensor>();
    addGemm(supported_pattern, supported_shared, supported_first_scale, supported_first_output);
    addGemm(supported_pattern, supported_shared, supported_second_scale, supported_second_output);

    PatternBuilder user_pattern;
    const FTrainTensorId user_second_output = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_scale  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_shared        = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_output  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_scale   = user_pattern.addOperand<OperandKind::kTensor>();
    addGemm(user_pattern, user_shared, user_second_scale, user_second_output);
    addGemm(user_pattern, user_shared, user_first_scale, user_first_output);

    expectSignatureCorrespondence(user_pattern, supported_pattern);
}

TEST(PatternStructuralColoringTest, CanonicalizesSymmetricDisconnectedComponents) {
    PatternBuilder supported_pattern;
    for (std::size_t component = 0; component < 2; ++component) {
        const FTrainTensorId input  = supported_pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId scale  = supported_pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId output = supported_pattern.addOperand<OperandKind::kTensor>();
        addGemm(supported_pattern, input, scale, output);
    }

    PatternBuilder user_pattern;
    std::array<FTrainTensorId, 2> inputs{user_pattern.addOperand<OperandKind::kTensor>(),
                                         user_pattern.addOperand<OperandKind::kTensor>()};
    std::array<FTrainTensorId, 2> scales{user_pattern.addOperand<OperandKind::kTensor>(),
                                         user_pattern.addOperand<OperandKind::kTensor>()};
    std::array<FTrainTensorId, 2> outputs{user_pattern.addOperand<OperandKind::kTensor>(),
                                          user_pattern.addOperand<OperandKind::kTensor>()};
    for (std::size_t component = 2; component != 0; --component) {
        const std::size_t index = component - 1;
        addGemm(user_pattern, inputs[index], scales[index], outputs[index]);
    }

    expectSignatureCorrespondence(user_pattern, supported_pattern);
}

TEST(PatternStructuralColoringTest, PreservesOrderedInputPorts) {
    PatternBuilder ordered_inputs;
    const FTrainTensorId input_source       = ordered_inputs.addOperand<OperandKind::kTensor>();
    const FTrainTensorId input_source_scale = ordered_inputs.addOperand<OperandKind::kTensor>();
    const FTrainTensorId input_intermediate = ordered_inputs.addOperand<OperandKind::kTensor>();
    const FTrainTensorId input_other        = ordered_inputs.addOperand<OperandKind::kTensor>();
    const FTrainTensorId input_output       = ordered_inputs.addOperand<OperandKind::kTensor>();
    addGemm(ordered_inputs, input_source, input_source_scale, input_intermediate);
    addGemm(ordered_inputs, input_intermediate, input_other, input_output);

    PatternBuilder swapped_inputs;
    const FTrainTensorId swapped_source       = swapped_inputs.addOperand<OperandKind::kTensor>();
    const FTrainTensorId swapped_source_scale = swapped_inputs.addOperand<OperandKind::kTensor>();
    const FTrainTensorId swapped_intermediate = swapped_inputs.addOperand<OperandKind::kTensor>();
    const FTrainTensorId swapped_other        = swapped_inputs.addOperand<OperandKind::kTensor>();
    const FTrainTensorId swapped_output       = swapped_inputs.addOperand<OperandKind::kTensor>();
    addGemm(swapped_inputs, swapped_source, swapped_source_scale, swapped_intermediate);
    addGemm(swapped_inputs, swapped_other, swapped_intermediate, swapped_output);

    EXPECT_NE(ordered_inputs.buildPattern().getKey(), swapped_inputs.buildPattern().getKey());
    EXPECT_FALSE(Matcher::match(ordered_inputs.buildPattern(), swapped_inputs.buildPattern()).has_value());
}

TEST(PatternStructuralColoringTest, DistinguishesKindsAndGloballyDifferentTopology) {
    PatternBuilder tensor_pattern;
    static_cast<void>(tensor_pattern.addOperand<OperandKind::kTensor>());
    PatternBuilder tensor_list_pattern;
    static_cast<void>(tensor_list_pattern.addOperand<OperandKind::kTensorList>());

    EXPECT_NE(tensor_pattern.buildPattern().getKey(), tensor_list_pattern.buildPattern().getKey());
    EXPECT_FALSE(Matcher::match(tensor_pattern.buildPattern(), tensor_list_pattern.buildPattern()).has_value());

    PatternBuilder six_cycle;
    addGemmCycle(six_cycle, 6);
    PatternBuilder two_three_cycles;
    addGemmCycle(two_three_cycles, 3);
    addGemmCycle(two_three_cycles, 3);

    // The classic refinement-indistinguishable pair: keys are equal, so
    // discrimination falls to the exact matcher, which correctly rejects.
    const Pattern six_pattern         = six_cycle.buildPattern();
    const Pattern three_three_pattern = two_three_cycles.buildPattern();

    EXPECT_EQ(six_pattern.getKey(), three_three_pattern.getKey());
    EXPECT_FALSE(Matcher::matchBySignature(six_cycle.buildPattern(), two_three_cycles.buildPattern()).has_value());
    EXPECT_FALSE(Matcher::match(six_cycle.buildPattern(), two_three_cycles.buildPattern()).has_value());
}

}  // namespace
}  // namespace ftrain
