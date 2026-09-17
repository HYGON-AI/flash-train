#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/op_definitions.hpp"
#include "flash_train/pattern.hpp"

namespace ftrain {
namespace {

template<typename Function>
void expectInvalidArgument(Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "Pattern accepted an invalid identifier or operation";
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
    constexpr PortId first{OperationId{3}, 4};
    constexpr PortId same{OperationId{3}, 4};
    constexpr PortId other_op{OperationId{5}, 4};
    constexpr PortId other_port{OperationId{3}, 6};

    return first.getOpId() == OperationId{3} && first.getPortIndex() == 4 && first == same && first != other_op &&
           first != other_port && noexcept(PortId{OperationId{0}, std::size_t{0}}) && noexcept(first.getOpId()) &&
           noexcept(first.getPortIndex()) && noexcept(first == same) && noexcept(first != same);
}

static_assert(hasExpectedIdInterface<OperandId>());
static_assert(hasExpectedIdInterface<OperationId>());
static_assert(hasExpectedPortIdInterface<OperationInputPortId>());
static_assert(hasExpectedPortIdInterface<OperationOutputPortId>());

static_assert(!std::is_same_v<OperationInputPortId, OperationOutputPortId>);

static_assert(std::is_constructible_v<OperandId, std::size_t>);
static_assert(std::is_constructible_v<OperationId, std::size_t>);
static_assert(!std::is_convertible_v<std::size_t, OperandId>);
static_assert(!std::is_convertible_v<std::size_t, OperationId>);
static_assert(!std::is_convertible_v<OperandId, std::size_t>);
static_assert(!std::is_convertible_v<OperationId, std::size_t>);

static_assert(kMutuallyNonConvertible<OperandId, OperationId>);
static_assert(kMutuallyNonConstructible<OperandId, OperationId>);

static_assert(noexcept(std::declval<const Pattern&>().getNumOperands()));
static_assert(
    std::is_same_v<decltype(std::declval<const Pattern&>().getOperandNode(OperandId{0})), const PatternOperandNode&>);
static_assert(noexcept(std::declval<const Pattern&>().getNumOps()));
static_assert(std::is_same_v<decltype(std::declval<const Pattern&>().getOpNode(OperationId{0})), const PatternOpNode&>);

static_assert(std::is_nothrow_constructible_v<PatternOperandNode, OperandKind>);
static_assert(noexcept(std::declval<const PatternOperandNode&>().getKind()));
static_assert(noexcept(std::declval<const PatternOperandNode&>().getProducer()));
static_assert(noexcept(std::declval<const PatternOperandNode&>().getConsumers()));

static_assert(
    std::is_nothrow_constructible_v<PatternOpNode, OperationKind, std::vector<OperandId>&&, std::vector<OperandId>&&>);
static_assert(noexcept(std::declval<const PatternOpNode&>().getKind()));
static_assert(noexcept(std::declval<const PatternOpNode&>().getInputs()));
static_assert(noexcept(std::declval<const PatternOpNode&>().getOutputs()));

TEST(PatternIdTest, PreservesIndexAndComparesWithinPattern) {
    EXPECT_EQ(OperandId{13}.getIndex(), 13);
    EXPECT_EQ(OperationId{14}.getIndex(), 14);

    EXPECT_EQ(OperandId{13}, OperandId{13});
    EXPECT_NE(OperationId{14}, OperationId{15});
}

TEST(PatternPortIdTest, PreservesOperationAndSemanticPortPosition) {
    const OperationInputPortId input{OperationId{13}, 4};
    const OperationOutputPortId output{OperationId{14}, 5};

    EXPECT_EQ(input.getOpId(), OperationId{13});
    EXPECT_EQ(input.getPortIndex(), 4);
    EXPECT_EQ(output.getOpId(), OperationId{14});
    EXPECT_EQ(output.getPortIndex(), 5);

    EXPECT_EQ(input, OperationInputPortId(OperationId{13}, 4));
    EXPECT_NE(output, OperationOutputPortId(OperationId{14}, 6));
}

TEST(PatternTest, StartsWithoutOperandsOrOperations) {
    const Pattern pattern;

    EXPECT_EQ(pattern.getNumOperands(), 0);
    EXPECT_EQ(pattern.getNumOps(), 0);
}

TEST(PatternTest, AddsOperandsInOrderAndStartsThemDisconnected) {
    Pattern pattern;
    const FTrainTensorId tensor_id                = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorListId tensor_list_id       = Operand<OperandKind::kTensorList>::addToPattern(pattern);
    const FTrainGroupedTensorId grouped_tensor_id = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);

    EXPECT_EQ(tensor_id.opaque, 0U);
    EXPECT_EQ(tensor_list_id.opaque, 1U);
    EXPECT_EQ(grouped_tensor_id.opaque, 2U);
    EXPECT_EQ(pattern.getNumOperands(), 3);

    const PatternOperandNode& tensor_node         = pattern.getOperandNode(OperandId{tensor_id.opaque});
    const PatternOperandNode& tensor_list_node    = pattern.getOperandNode(OperandId{tensor_list_id.opaque});
    const PatternOperandNode& grouped_tensor_node = pattern.getOperandNode(OperandId{grouped_tensor_id.opaque});
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
    Pattern pattern;
    const FTrainTensorId tensor          = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorListId tensor_list = Operand<OperandKind::kTensorList>::addToPattern(pattern);
    const FTrainGroupedTensorId grouped  = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);

    EXPECT_EQ(tensor.opaque, 0U);
    EXPECT_EQ(tensor_list.opaque, 1U);
    EXPECT_EQ(grouped.opaque, 2U);
    EXPECT_EQ(pattern.getNumOperands(), 3);

    EXPECT_EQ(pattern.getOperandNode(OperandId{tensor.opaque}).getKind(), OperandKind::kTensor);
    EXPECT_EQ(pattern.getOperandNode(OperandId{tensor_list.opaque}).getKind(), OperandKind::kTensorList);
    EXPECT_EQ(pattern.getOperandNode(OperandId{grouped.opaque}).getKind(), OperandKind::kGroupedTensor);
}

TEST(PatternTest, AddsTypedOperationWithConcreteIdPorts) {
    Pattern pattern;
    const FTrainTensorListId a    = Operand<OperandKind::kTensorList>::addToPattern(pattern);
    const FTrainGroupedTensorId b = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
    const FTrainGroupedTensorId c = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
    const FTrainGroupedTensorId d = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
    const FTrainTensorId alpha    = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId beta     = Operand<OperandKind::kTensor>::addToPattern(pattern);

    const OperationId op_id =
        OperationId{Operation<OperationKind::kGroupedBCDGemm>::addToPattern(pattern, a, b, c, d, alpha, beta).opaque};

    EXPECT_EQ(op_id.getIndex(), 0U);
    EXPECT_EQ(pattern.getNumOperands(), 6);
    EXPECT_EQ(pattern.getNumOps(), 1);
    const PatternOpNode& node = pattern.getOpNode(op_id);
    EXPECT_EQ(node.getKind(), OperationKind::kGroupedBCDGemm);
    EXPECT_EQ(node.getInputs(),
              (std::vector<OperandId>{OperandId{0}, OperandId{1}, OperandId{2}, OperandId{4}, OperandId{5}}));
    EXPECT_EQ(node.getOutputs(), (std::vector<OperandId>{OperandId{3}}));
}

TEST(PatternTest, AddsOperationWithOrderedPortsAndBidirectionalTopology) {
    Pattern pattern;
    const FTrainTensorListId a    = Operand<OperandKind::kTensorList>::addToPattern(pattern);
    const FTrainGroupedTensorId b = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
    const FTrainGroupedTensorId c = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
    const FTrainTensorId alpha    = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId beta     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainGroupedTensorId d = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
    const OperationId op_id =
        (OperationId{Operation<OperationKind::kGroupedBCDGemm>::addToPattern(pattern, a, b, c, d, alpha, beta).opaque});

    EXPECT_EQ(op_id, OperationId{0});
    EXPECT_EQ(pattern.getNumOps(), 1);
    const PatternOpNode& op_node = pattern.getOpNode(op_id);
    EXPECT_EQ(op_node.getKind(), OperationKind::kGroupedBCDGemm);
    EXPECT_EQ(op_node.getInputs(),
              (std::vector<OperandId>{OperandId{a.opaque}, OperandId{b.opaque}, OperandId{c.opaque},
                                      OperandId{alpha.opaque}, OperandId{beta.opaque}}));
    EXPECT_EQ(op_node.getOutputs(), (std::vector<OperandId>{OperandId{d.opaque}}));

    for (std::size_t port_index = 0; port_index < op_node.getInputs().size(); ++port_index) {
        const auto& consumers = pattern.getOperandNode(op_node.getInputs()[port_index]).getConsumers();
        ASSERT_EQ(consumers.size(), 1);
        EXPECT_EQ(consumers[0], OperationInputPortId(op_id, port_index));
    }
    ASSERT_TRUE(pattern.getOperandNode(OperandId{d.opaque}).getProducer().has_value());
    EXPECT_EQ(*pattern.getOperandNode(OperandId{d.opaque}).getProducer(), OperationOutputPortId(op_id, 0));
}

TEST(PatternTest, AllowsDownstreamOperationBeforeUpstreamProducer) {
    Pattern pattern;
    const FTrainTensorId input            = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId intermediate     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId output           = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId downstream_b     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId downstream_c     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId downstream_alpha = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId downstream_beta  = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId upstream_b       = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId upstream_c       = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId upstream_alpha   = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId upstream_beta    = Operand<OperandKind::kTensor>::addToPattern(pattern);

    const OperationId downstream =
        OperationId{Operation<OperationKind::kGemm>::addToPattern(pattern, intermediate, downstream_b, downstream_c,
                                                                  output, downstream_alpha, downstream_beta)
                        .opaque};
    const OperationId upstream =
        OperationId{Operation<OperationKind::kGemm>::addToPattern(pattern, input, upstream_b, upstream_c, intermediate,
                                                                  upstream_alpha, upstream_beta)
                        .opaque};

    EXPECT_EQ(downstream, OperationId{0});
    EXPECT_EQ(upstream, OperationId{1});
    EXPECT_EQ(pattern.getNumOps(), 2);
    const PatternOperandNode& intermediate_node = pattern.getOperandNode(OperandId{intermediate.opaque});
    ASSERT_TRUE(intermediate_node.getProducer().has_value());
    EXPECT_EQ(*intermediate_node.getProducer(), OperationOutputPortId(upstream, 0));
    ASSERT_EQ(intermediate_node.getConsumers().size(), 1);
    EXPECT_EQ(intermediate_node.getConsumers()[0], OperationInputPortId(downstream, 0));
}

TEST(PatternTest, RejectsRepeatedInputsWithoutMutation) {
    Pattern pattern;
    const FTrainTensorId input  = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId c      = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId alpha  = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId beta   = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId output = Operand<OperandKind::kTensor>::addToPattern(pattern);

    expectInvalidArgument([&] {
        static_cast<void>(OperationId{
            Operation<OperationKind::kGemm>::addToPattern(pattern, input, input, c, output, alpha, beta).opaque});
    });

    EXPECT_EQ(pattern.getNumOps(), 0);
    EXPECT_TRUE(pattern.getOperandNode(OperandId{input.opaque}).getConsumers().empty());
    EXPECT_FALSE(pattern.getOperandNode(OperandId{output.opaque}).getProducer().has_value());
}

TEST(PatternTest, AllowsOneInputToFeedMultipleOperations) {
    Pattern pattern;
    const FTrainTensorId input         = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId first_output  = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId second_output = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId first_b       = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId first_c       = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId first_alpha   = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId first_beta    = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId second_b      = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId second_c      = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId second_alpha  = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId second_beta   = Operand<OperandKind::kTensor>::addToPattern(pattern);

    const OperationId first_op =
        (OperationId{Operation<OperationKind::kGemm>::addToPattern(pattern, input, first_b, first_c, first_output,
                                                                   first_alpha, first_beta)
                         .opaque});
    const OperationId second_op =
        (OperationId{Operation<OperationKind::kGemm>::addToPattern(pattern, input, second_b, second_c, second_output,
                                                                   second_alpha, second_beta)
                         .opaque});

    const PatternOperandNode& input_node = pattern.getOperandNode(OperandId{input.opaque});
    ASSERT_EQ(input_node.getConsumers().size(), 2);
    EXPECT_NE(std::find(input_node.getConsumers().begin(), input_node.getConsumers().end(),
                        OperationInputPortId(first_op, 0)),
              input_node.getConsumers().end());
    EXPECT_NE(std::find(input_node.getConsumers().begin(), input_node.getConsumers().end(),
                        OperationInputPortId(second_op, 0)),
              input_node.getConsumers().end());
}

TEST(PatternTest, RejectsInputOutputOperandReuseWithoutMutation) {
    Pattern pattern;
    const FTrainTensorId first_input = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId b           = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId alpha       = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId beta        = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId reused      = Operand<OperandKind::kTensor>::addToPattern(pattern);

    expectInvalidArgument([&] {
        static_cast<void>(OperationId{
            Operation<OperationKind::kGemm>::addToPattern(pattern, first_input, b, reused, reused, alpha, beta)
                .opaque});
    });

    EXPECT_EQ(pattern.getNumOps(), 0);
    EXPECT_FALSE(pattern.getOperandNode(OperandId{first_input.opaque}).getProducer().has_value());
    EXPECT_TRUE(pattern.getOperandNode(OperandId{first_input.opaque}).getConsumers().empty());
    EXPECT_FALSE(pattern.getOperandNode(OperandId{reused.opaque}).getProducer().has_value());
    EXPECT_TRUE(pattern.getOperandNode(OperandId{reused.opaque}).getConsumers().empty());
    EXPECT_FALSE(pattern.getOperandNode(OperandId{b.opaque}).getProducer().has_value());
    EXPECT_TRUE(pattern.getOperandNode(OperandId{b.opaque}).getConsumers().empty());
}

TEST(PatternTest, RejectsOutOfRangeOperationOperandIdsWithoutMutation) {
    Pattern pattern;
    const FTrainTensorId a                     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId b                     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId c                     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId alpha                 = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId beta                  = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId output                = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainGroupedTensorId grouped_tensor = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);

    expectInvalidArgument([&] {
        static_cast<void>(
            Operation<OperationKind::kGemm>::addToPattern(pattern, a, b, FTrainTensorId{99}, output, alpha, beta));
    });
    expectInvalidArgument([&] {
        static_cast<void>(
            Operation<OperationKind::kGemm>::addToPattern(pattern, a, b, c, FTrainTensorId{99}, alpha, beta));
    });

    EXPECT_EQ(pattern.getNumOps(), 0);
    EXPECT_TRUE(pattern.getOperandNode(OperandId{a.opaque}).getConsumers().empty());
    EXPECT_FALSE(pattern.getOperandNode(OperandId{output.opaque}).getProducer().has_value());
    EXPECT_TRUE(pattern.getOperandNode(OperandId{grouped_tensor.opaque}).getConsumers().empty());
    EXPECT_FALSE(pattern.getOperandNode(OperandId{grouped_tensor.opaque}).getProducer().has_value());
}

TEST(PatternTest, RejectsExistingProducersWithoutCorruptingTopology) {
    Pattern pattern;
    const FTrainTensorId input          = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId second_input   = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId output         = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId producer_b     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId producer_c     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId producer_alpha = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId producer_beta  = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId second_c       = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId second_alpha   = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId second_beta    = Operand<OperandKind::kTensor>::addToPattern(pattern);

    const OperationId producer =
        (OperationId{Operation<OperationKind::kGemm>::addToPattern(pattern, input, producer_b, producer_c, output,
                                                                   producer_alpha, producer_beta)
                         .opaque});
    expectInvalidArgument([&] {
        static_cast<void>(OperationId{Operation<OperationKind::kGemm>::addToPattern(
                                          pattern, second_input, input, second_c, output, second_alpha, second_beta)
                                          .opaque});
    });

    EXPECT_EQ(pattern.getNumOps(), 1);
    ASSERT_TRUE(pattern.getOperandNode(OperandId{output.opaque}).getProducer().has_value());
    EXPECT_EQ(*pattern.getOperandNode(OperandId{output.opaque}).getProducer(), OperationOutputPortId(producer, 0));
    EXPECT_TRUE(pattern.getOperandNode(OperandId{second_input.opaque}).getConsumers().empty());
}

TEST(PatternTest, RejectsOutOfRangePatternIds) {
    const Pattern pattern;

    expectInvalidArgument([&] { static_cast<void>(pattern.getOperandNode(OperandId{0})); });
    expectInvalidArgument([&] { static_cast<void>(pattern.getOpNode(OperationId{0})); });
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
    std::vector<OperandId> inputs{OperandId{5}, OperandId{2}, OperandId{8}, OperandId{4}, OperandId{9}};
    std::vector<OperandId> outputs{OperandId{7}};

    const PatternOpNode node(OperationKind::kGroupedBCDGemm, std::move(inputs), std::move(outputs));

    ASSERT_EQ(node.getInputs().size(), 5);
    EXPECT_EQ(node.getInputs()[0], OperandId{5});
    EXPECT_EQ(node.getInputs()[1], OperandId{2});
    EXPECT_EQ(node.getInputs()[2], OperandId{8});
    EXPECT_EQ(node.getInputs()[3], OperandId{4});
    EXPECT_EQ(node.getInputs()[4], OperandId{9});

    ASSERT_EQ(node.getOutputs().size(), 1);
    EXPECT_EQ(node.getOutputs()[0], OperandId{7});

    EXPECT_EQ(node.getKind(), OperationKind::kGroupedBCDGemm);
}

}  // namespace
}  // namespace ftrain
