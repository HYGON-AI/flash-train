#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"

namespace ftrain {
namespace {

template<typename Function>
void expectInvalidArgument(Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "PatternBuilder accepted an invalid identifier or operation";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

template<typename Left, typename Right>
constexpr bool kMutuallyNonConvertible = !std::is_convertible_v<Left, Right> && !std::is_convertible_v<Right, Left>;

template<typename Left, typename Right>
constexpr bool kMutuallyNonConstructible =
    !std::is_constructible_v<Left, Right> && !std::is_constructible_v<Right, Left>;

template<typename Id>
constexpr bool hasExpectedIdInterface() noexcept {
    constexpr Id first{3};
    constexpr Id same{3};
    constexpr Id other{4};

    return first.getIndex() == 3 && first == same && first != other && noexcept(Id{std::uint64_t{0}}) &&
           noexcept(first.getIndex()) && noexcept(first == same) && noexcept(first != other);
}

template<typename PortId>
constexpr bool hasExpectedPortIdInterface() noexcept {
    constexpr PortId first{PatternOperationId{3}, 4};
    constexpr PortId same{PatternOperationId{3}, 4};
    constexpr PortId other_op{PatternOperationId{5}, 4};
    constexpr PortId other_port{PatternOperationId{3}, 6};

    return first.getOpId() == PatternOperationId{3} && first.getPortIndex() == 4 && first == same &&
           first != other_op && first != other_port && noexcept(PortId{PatternOperationId{0}, std::size_t{0}}) &&
           noexcept(first.getOpId()) && noexcept(first.getPortIndex()) && noexcept(first == same) &&
           noexcept(first != same);
}

static_assert(hasExpectedIdInterface<PatternOperandId>());
static_assert(hasExpectedIdInterface<PatternOperationId>());
static_assert(hasExpectedPortIdInterface<PatternOperationInputPortId>());
static_assert(hasExpectedPortIdInterface<PatternOperationOutputPortId>());

static_assert(!std::is_same_v<PatternOperationInputPortId, PatternOperationOutputPortId>);

static_assert(std::is_constructible_v<PatternOperandId, std::size_t>);
static_assert(std::is_constructible_v<PatternOperationId, std::size_t>);
static_assert(!std::is_convertible_v<std::size_t, PatternOperandId>);
static_assert(!std::is_convertible_v<std::size_t, PatternOperationId>);
static_assert(!std::is_convertible_v<PatternOperandId, std::size_t>);
static_assert(!std::is_convertible_v<PatternOperationId, std::size_t>);

static_assert(kMutuallyNonConvertible<PatternOperandId, PatternOperationId>);
static_assert(kMutuallyNonConstructible<PatternOperandId, PatternOperationId>);

static_assert(noexcept(std::declval<const PatternBuilder&>().getNumOperands()));
static_assert(std::is_same_v<decltype(std::declval<const Pattern&>().getOperandNode(PatternOperandId{0})),
                             const PatternOperandNode&>);
static_assert(noexcept(std::declval<const PatternBuilder&>().getNumOps()));
static_assert(std::is_same_v<decltype(std::declval<const Pattern&>().getOpNode(PatternOperationId{0})),
                             const PatternOperationNode&>);

static_assert(std::is_nothrow_constructible_v<PatternOperandNode, OperandKind>);
static_assert(noexcept(std::declval<const PatternOperandNode&>().getKind()));
static_assert(noexcept(std::declval<const PatternOperandNode&>().getProducer()));
static_assert(noexcept(std::declval<const PatternOperandNode&>().getConsumers()));

static_assert(std::is_nothrow_constructible_v<PatternOperationNode, OperationKind, std::vector<PatternOperandId>&&,
                                              std::vector<PatternOperandId>&&>);
static_assert(noexcept(std::declval<const PatternOperationNode&>().getKind()));
static_assert(noexcept(std::declval<const PatternOperationNode&>().getInputs()));
static_assert(noexcept(std::declval<const PatternOperationNode&>().getOutputs()));

TEST(PatternIdTest, PreservesIndexAndComparesWithinPattern) {
    EXPECT_EQ(PatternOperandId{13}.getIndex(), 13);
    EXPECT_EQ(PatternOperationId{14}.getIndex(), 14);

    EXPECT_EQ(PatternOperandId{13}, PatternOperandId{13});
    EXPECT_NE(PatternOperationId{14}, PatternOperationId{15});
}

TEST(PatternPortIdTest, PreservesOperationAndSemanticPortPosition) {
    const PatternOperationInputPortId input{PatternOperationId{13}, 4};
    const PatternOperationOutputPortId output{PatternOperationId{14}, 5};

    EXPECT_EQ(input.getOpId(), PatternOperationId{13});
    EXPECT_EQ(input.getPortIndex(), 4);
    EXPECT_EQ(output.getOpId(), PatternOperationId{14});
    EXPECT_EQ(output.getPortIndex(), 5);

    EXPECT_EQ(input, PatternOperationInputPortId(PatternOperationId{13}, 4));
    EXPECT_NE(output, PatternOperationOutputPortId(PatternOperationId{14}, 6));
}

TEST(PatternTest, StartsWithoutOperandsOrOperations) {
    const PatternBuilder pattern;

    EXPECT_EQ(pattern.getNumOperands(), 0);
    EXPECT_EQ(pattern.getNumOps(), 0);
}

TEST(PatternTest, AddsOperandsInOrderAndStartsThemDisconnected) {
    PatternBuilder pattern;
    const FTrainTensorId tensor_id                = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorListId tensor_list_id       = pattern.addOperand<OperandKind::kTensorList>();
    const FTrainGroupedTensorId grouped_tensor_id = pattern.addOperand<OperandKind::kGroupedTensor>();

    EXPECT_EQ(tensor_id.opaque, 0U);
    EXPECT_EQ(tensor_list_id.opaque, 1U);
    EXPECT_EQ(grouped_tensor_id.opaque, 2U);
    EXPECT_EQ(pattern.getNumOperands(), 3);

    const Pattern prepared                        = pattern.buildPattern();
    const PatternOperandNode& tensor_node         = prepared.getOperandNode(PatternOperandId{tensor_id.opaque});
    const PatternOperandNode& tensor_list_node    = prepared.getOperandNode(PatternOperandId{tensor_list_id.opaque});
    const PatternOperandNode& grouped_tensor_node = prepared.getOperandNode(PatternOperandId{grouped_tensor_id.opaque});
    EXPECT_EQ(tensor_node.getKind(), OperandKind::kTensor);
    EXPECT_EQ(tensor_list_node.getKind(), OperandKind::kTensorList);
    EXPECT_EQ(grouped_tensor_node.getKind(), OperandKind::kGroupedTensor);
    EXPECT_FALSE(tensor_node.getProducer().has_value());
    EXPECT_TRUE(tensor_node.getConsumers().empty());
    EXPECT_FALSE(tensor_list_node.getProducer().has_value());
    EXPECT_TRUE(tensor_list_node.getConsumers().empty());
    EXPECT_FALSE(grouped_tensor_node.getProducer().has_value());
    EXPECT_TRUE(grouped_tensor_node.getConsumers().empty());
}

TEST(PatternTest, AddsTypedOperandsWithConcreteIds) {
    PatternBuilder pattern;
    const FTrainTensorId tensor          = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorListId tensor_list = pattern.addOperand<OperandKind::kTensorList>();
    const FTrainGroupedTensorId grouped  = pattern.addOperand<OperandKind::kGroupedTensor>();

    EXPECT_EQ(tensor.opaque, 0U);
    EXPECT_EQ(tensor_list.opaque, 1U);
    EXPECT_EQ(grouped.opaque, 2U);
    EXPECT_EQ(pattern.getNumOperands(), 3);

    const Pattern prepared = pattern.buildPattern();
    EXPECT_EQ(prepared.getOperandNode(PatternOperandId{tensor.opaque}).getKind(), OperandKind::kTensor);
    EXPECT_EQ(prepared.getOperandNode(PatternOperandId{tensor_list.opaque}).getKind(), OperandKind::kTensorList);
    EXPECT_EQ(prepared.getOperandNode(PatternOperandId{grouped.opaque}).getKind(), OperandKind::kGroupedTensor);
}

TEST(PatternTest, AddsTypedOperationWithConcreteIdPorts) {
    PatternBuilder pattern;
    const FTrainTensorListId a    = pattern.addOperand<OperandKind::kTensorList>();
    const FTrainGroupedTensorId b = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId c = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId d = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainTensorId alpha    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId beta     = pattern.addOperand<OperandKind::kTensor>();

    const PatternOperationId op_id =
        PatternOperationId{pattern.addOperation<OperationKind::kGroupedBCDGemm>({a, b, c, d, alpha, beta}).opaque};

    EXPECT_EQ(op_id.getIndex(), 0U);
    EXPECT_EQ(pattern.getNumOperands(), 6);
    EXPECT_EQ(pattern.getNumOps(), 1);
    const Pattern prepared           = pattern.buildPattern();
    const PatternOperationNode& node = prepared.getOpNode(op_id);
    EXPECT_EQ(node.getKind(), OperationKind::kGroupedBCDGemm);
    EXPECT_EQ(node.getInputs(),
              (std::vector<PatternOperandId>{PatternOperandId{0}, PatternOperandId{1}, PatternOperandId{2},
                                             PatternOperandId{4}, PatternOperandId{5}}));
    EXPECT_EQ(node.getOutputs(), (std::vector<PatternOperandId>{PatternOperandId{3}}));
}

TEST(PatternTest, AddsOperationWithOrderedPortsAndBidirectionalTopology) {
    PatternBuilder pattern;
    const FTrainTensorListId a    = pattern.addOperand<OperandKind::kTensorList>();
    const FTrainGroupedTensorId b = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId c = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainTensorId alpha    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId beta     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainGroupedTensorId d = pattern.addOperand<OperandKind::kGroupedTensor>();
    const PatternOperationId op_id =
        (PatternOperationId{pattern.addOperation<OperationKind::kGroupedBCDGemm>({a, b, c, d, alpha, beta}).opaque});

    EXPECT_EQ(op_id, PatternOperationId{0});
    EXPECT_EQ(pattern.getNumOps(), 1);
    const Pattern prepared              = pattern.buildPattern();
    const PatternOperationNode& op_node = prepared.getOpNode(op_id);
    EXPECT_EQ(op_node.getKind(), OperationKind::kGroupedBCDGemm);
    EXPECT_EQ(op_node.getInputs(),
              (std::vector<PatternOperandId>{PatternOperandId{a.opaque}, PatternOperandId{b.opaque},
                                             PatternOperandId{c.opaque}, PatternOperandId{alpha.opaque},
                                             PatternOperandId{beta.opaque}}));
    EXPECT_EQ(op_node.getOutputs(), (std::vector<PatternOperandId>{PatternOperandId{d.opaque}}));

    for (std::size_t port_index = 0; port_index < op_node.getInputs().size(); ++port_index) {
        const auto& consumers = prepared.getOperandNode(op_node.getInputs()[port_index]).getConsumers();
        ASSERT_EQ(consumers.size(), 1);
        EXPECT_EQ(consumers[0], PatternOperationInputPortId(op_id, port_index));
    }
    ASSERT_TRUE(prepared.getOperandNode(PatternOperandId{d.opaque}).getProducer().has_value());
    EXPECT_EQ(*prepared.getOperandNode(PatternOperandId{d.opaque}).getProducer(),
              PatternOperationOutputPortId(op_id, 0));
}

TEST(PatternTest, AllowsDownstreamOperationBeforeUpstreamProducer) {
    PatternBuilder pattern;
    const FTrainTensorId input            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId intermediate     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId output           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId downstream_b     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId downstream_c     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId downstream_alpha = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId downstream_beta  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId upstream_b       = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId upstream_c       = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId upstream_alpha   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId upstream_beta    = pattern.addOperand<OperandKind::kTensor>();

    const PatternOperationId downstream =
        PatternOperationId{pattern
                               .addOperation<OperationKind::kGemm>({intermediate, downstream_b, downstream_c, output,
                                                                    downstream_alpha, downstream_beta})
                               .opaque};
    const PatternOperationId upstream =
        PatternOperationId{pattern
                               .addOperation<OperationKind::kGemm>(
                                   {input, upstream_b, upstream_c, intermediate, upstream_alpha, upstream_beta})
                               .opaque};

    EXPECT_EQ(downstream, PatternOperationId{0});
    EXPECT_EQ(upstream, PatternOperationId{1});
    EXPECT_EQ(pattern.getNumOps(), 2);
    const Pattern prepared                      = pattern.buildPattern();
    const PatternOperandNode& intermediate_node = prepared.getOperandNode(PatternOperandId{intermediate.opaque});
    ASSERT_TRUE(intermediate_node.getProducer().has_value());
    EXPECT_EQ(*intermediate_node.getProducer(), PatternOperationOutputPortId(upstream, 0));
    ASSERT_EQ(intermediate_node.getConsumers().size(), 1);
    EXPECT_EQ(intermediate_node.getConsumers()[0], PatternOperationInputPortId(downstream, 0));
}

TEST(PatternTest, RejectsRepeatedInputsWithoutMutation) {
    PatternBuilder pattern;
    const FTrainTensorId input  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId c      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId alpha  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId beta   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId output = pattern.addOperand<OperandKind::kTensor>();

    static_cast<void>(pattern.addOperation<OperationKind::kGemm>({input, input, c, output, alpha, beta}));
    expectInvalidArgument([&] { static_cast<void>(pattern.buildPattern()); });
}

TEST(PatternTest, AllowsOneInputToFeedMultipleOperations) {
    PatternBuilder pattern;
    const FTrainTensorId input         = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_output  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_output = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_b       = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_c       = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_alpha   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_beta    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_b      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_c      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_alpha  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_beta   = pattern.addOperand<OperandKind::kTensor>();

    const PatternOperationId first_op  = (PatternOperationId{
        pattern.addOperation<OperationKind::kGemm>({input, first_b, first_c, first_output, first_alpha, first_beta})
            .opaque});
    const PatternOperationId second_op = (PatternOperationId{
        pattern
            .addOperation<OperationKind::kGemm>({input, second_b, second_c, second_output, second_alpha, second_beta})
            .opaque});

    const Pattern prepared               = pattern.buildPattern();
    const PatternOperandNode& input_node = prepared.getOperandNode(PatternOperandId{input.opaque});
    ASSERT_EQ(input_node.getConsumers().size(), 2);
    EXPECT_NE(std::find(input_node.getConsumers().begin(), input_node.getConsumers().end(),
                        PatternOperationInputPortId(first_op, 0)),
              input_node.getConsumers().end());
    EXPECT_NE(std::find(input_node.getConsumers().begin(), input_node.getConsumers().end(),
                        PatternOperationInputPortId(second_op, 0)),
              input_node.getConsumers().end());
}

TEST(PatternTest, RejectsInputOutputOperandReuseWithoutMutation) {
    PatternBuilder pattern;
    const FTrainTensorId first_input = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId b           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId alpha       = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId beta        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId reused      = pattern.addOperand<OperandKind::kTensor>();

    static_cast<void>(pattern.addOperation<OperationKind::kGemm>({first_input, b, reused, reused, alpha, beta}));
    expectInvalidArgument([&] { static_cast<void>(pattern.buildPattern()); });
}

TEST(PatternTest, RejectsOutOfRangeOperationOperandIdsWithoutMutation) {
    PatternBuilder pattern;
    const FTrainTensorId a                     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId b                     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId c                     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId alpha                 = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId beta                  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId output                = pattern.addOperand<OperandKind::kTensor>();
    const FTrainGroupedTensorId grouped_tensor = pattern.addOperand<OperandKind::kGroupedTensor>();

    static_cast<void>(pattern.addOperation<OperationKind::kGemm>({a, b, FTrainTensorId{99}, output, alpha, beta}));
    expectInvalidArgument([&] { static_cast<void>(pattern.buildPattern()); });

    PatternBuilder valid_pattern;
    const FTrainTensorId va      = valid_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId vb      = valid_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId vc      = valid_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId valpha  = valid_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId vbeta   = valid_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId voutput = valid_pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(
        valid_pattern.addOperation<OperationKind::kGemm>({va, vb, vc, valpha, vbeta, FTrainTensorId{99}}));
    expectInvalidArgument([&] { static_cast<void>(valid_pattern.buildPattern()); });
}

TEST(PatternTest, RejectsExistingProducersWithoutCorruptingTopology) {
    PatternBuilder pattern;
    const FTrainTensorId input          = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_input   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId output         = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId producer_b     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId producer_c     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId producer_alpha = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId producer_beta  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_c       = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_alpha   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_beta    = pattern.addOperand<OperandKind::kTensor>();

    const PatternOperationId producer = (PatternOperationId{
        pattern
            .addOperation<OperationKind::kGemm>({input, producer_b, producer_c, output, producer_alpha, producer_beta})
            .opaque});
    static_cast<void>(
        pattern.addOperation<OperationKind::kGemm>({second_input, input, second_c, output, second_alpha, second_beta}));
    expectInvalidArgument([&] { static_cast<void>(pattern.buildPattern()); });

    // The producer assertion now holds on the first op only, via a fresh
    // builder prepared before the conflicting second op is appended.
    PatternBuilder single_pattern;
    const FTrainTensorId s_input             = single_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId s_b                 = single_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId s_c                 = single_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId s_alpha             = single_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId s_beta              = single_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId s_output            = single_pattern.addOperand<OperandKind::kTensor>();
    const PatternOperationId single_producer = PatternOperationId{
        single_pattern.addOperation<OperationKind::kGemm>({s_input, s_b, s_c, s_output, s_alpha, s_beta}).opaque};
    const Pattern prepared = single_pattern.buildPattern();
    ASSERT_TRUE(prepared.getOperandNode(PatternOperandId{s_output.opaque}).getProducer().has_value());
    EXPECT_EQ(*prepared.getOperandNode(PatternOperandId{s_output.opaque}).getProducer(),
              PatternOperationOutputPortId(single_producer, 0));
}

TEST(PatternTest, RejectsOutOfRangePatternIds) {
    const Pattern prepared = PatternBuilder{}.buildPattern();

    expectInvalidArgument([&] { static_cast<void>(prepared.getOperandNode(PatternOperandId{0})); });
    expectInvalidArgument([&] { static_cast<void>(prepared.getOpNode(PatternOperationId{0})); });
}

TEST(PatternOperandNodeTest, StartsDisconnectedAndPreservesKind) {
    const PatternOperandNode tensor_node(OperandKind::kTensor);
    const PatternOperandNode tensor_list_node(OperandKind::kTensorList);
    const PatternOperandNode grouped_tensor_node(OperandKind::kGroupedTensor);

    EXPECT_EQ(tensor_node.getKind(), OperandKind::kTensor);
    EXPECT_FALSE(tensor_node.getProducer().has_value());
    EXPECT_TRUE(tensor_node.getConsumers().empty());

    EXPECT_EQ(tensor_list_node.getKind(), OperandKind::kTensorList);
    EXPECT_FALSE(tensor_list_node.getProducer().has_value());
    EXPECT_TRUE(tensor_list_node.getConsumers().empty());

    EXPECT_EQ(grouped_tensor_node.getKind(), OperandKind::kGroupedTensor);
    EXPECT_FALSE(grouped_tensor_node.getProducer().has_value());
    EXPECT_TRUE(grouped_tensor_node.getConsumers().empty());
}

TEST(PatternOpNodeTest, PreservesKindAndSemanticPortOrder) {
    std::vector<PatternOperandId> inputs{PatternOperandId{5}, PatternOperandId{2}, PatternOperandId{8},
                                         PatternOperandId{4}, PatternOperandId{9}};
    std::vector<PatternOperandId> outputs{PatternOperandId{7}};

    const PatternOperationNode node(OperationKind::kGroupedBCDGemm, std::move(inputs), std::move(outputs));

    ASSERT_EQ(node.getInputs().size(), 5);
    EXPECT_EQ(node.getInputs()[0], PatternOperandId{5});
    EXPECT_EQ(node.getInputs()[1], PatternOperandId{2});
    EXPECT_EQ(node.getInputs()[2], PatternOperandId{8});
    EXPECT_EQ(node.getInputs()[3], PatternOperandId{4});
    EXPECT_EQ(node.getInputs()[4], PatternOperandId{9});

    ASSERT_EQ(node.getOutputs().size(), 1);
    EXPECT_EQ(node.getOutputs()[0], PatternOperandId{7});

    EXPECT_EQ(node.getKind(), OperationKind::kGroupedBCDGemm);
}

}  // namespace
}  // namespace ftrain
