#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {
namespace {

template<typename Function>
void expectUnsupported(Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "Matching accepted structurally different patterns";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_UNSUPPORTED);
    }
}

template<typename Function>
void expectInvalidArgument(Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "Matching accepted an out-of-range identifier";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

// Marker values let one Args round trip reveal the whole role mapping: user
// operand i is stored as a Tensor with extent i + 1 or a TensorList with
// i + 1 storage views, and user op i as GemmAttributes with compute type
// i + 1; decoding every supported slot reports the user index each one
// received.
Tensor makeMarkerTensor(std::size_t operand_index) {
    static std::int32_t memory;
    const std::int64_t dims[]{static_cast<std::int64_t>(operand_index + 1)};
    const FTrainStorageView description{
        &memory, dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    return Tensor(StorageView(description));
}

TensorList makeMarkerTensorList(std::size_t operand_index) {
    static std::int32_t memory;
    const std::int64_t dims[]{1};
    const FTrainStorageView description{
        &memory, dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    std::vector<StorageView> storage_views(operand_index + 1, StorageView(description));
    return TensorList(std::move(storage_views));
}

// The user-to-supported role correspondence, reconstructed through the
// public surface: one Ops, one Args, and marker values decoded from every
// supported slot.
struct RoleImages {
    std::vector<PatternOperandId> supported_operand_by_user;
    std::vector<PatternOperationId> supported_op_by_user;
};

std::optional<std::size_t> decodeOperandUserIndex(const OperandValue& operand) {
    if (const Tensor* tensor = std::get_if<Tensor>(&operand)) {
        return static_cast<std::size_t>(tensor->getStorageView().getDims()[0]) - 1;
    }
    if (const TensorList* tensor_list = std::get_if<TensorList>(&operand)) {
        return tensor_list->getStorageViews().size() - 1;
    }
    return std::nullopt;
}

void matchRoles(const Pattern& user_pattern, const Pattern& supported_pattern, RoleImages& images) {
    const Ops ops(user_pattern, supported_pattern);
    Args args = ops.makeArgs();
    images.supported_operand_by_user =
        std::vector<PatternOperandId>(user_pattern.getNumOperands(), PatternOperandId{0});
    images.supported_op_by_user = std::vector<PatternOperationId>(user_pattern.getNumOps(), PatternOperationId{0});

    for (std::size_t user_operand_index = 0; user_operand_index < user_pattern.getNumOperands(); ++user_operand_index) {
        const PatternOperandId user_operand_id{user_operand_index};
        switch (user_pattern.getOperandNode(user_operand_id).getKind()) {
            case OperandKind::kTensor:
                args.setOperand<OperandKind::kTensor>(user_operand_id, makeMarkerTensor(user_operand_index));
                break;
            case OperandKind::kTensorList:
                args.setOperand<OperandKind::kTensorList>(user_operand_id, makeMarkerTensorList(user_operand_index));
                break;
            case OperandKind::kGroupedTensor:
                ADD_FAILURE() << "No marker representation for GroupedTensor operands";
                break;
        }
    }
    for (std::size_t user_op_index = 0; user_op_index < user_pattern.getNumOps(); ++user_op_index) {
        // Every successfully matched operation in this suite is a kGemm.
        ASSERT_EQ(user_pattern.getOpNode(PatternOperationId{user_op_index}).getKind(), OperationKind::kGemm);
        args.setOperation<OperationKind::kGemm>(PatternOperationId{user_op_index},
                                                GemmAttributes(static_cast<FTrainNumericType>(user_op_index + 1)));
    }
    ASSERT_TRUE(args.isComplete());

    std::vector<bool> seen_operands(user_pattern.getNumOperands(), false);
    for (std::size_t supported_operand_index = 0; supported_operand_index < supported_pattern.getNumOperands();
         ++supported_operand_index) {
        const std::optional<OperandValue>& operand = args.getOperand(PatternOperandId{supported_operand_index});
        ASSERT_TRUE(operand.has_value());
        const std::optional<std::size_t> user_operand_index = decodeOperandUserIndex(*operand);
        ASSERT_TRUE(user_operand_index.has_value());
        ASSERT_LT(*user_operand_index, seen_operands.size());
        ASSERT_FALSE(seen_operands[*user_operand_index]);
        seen_operands[*user_operand_index]                    = true;
        images.supported_operand_by_user[*user_operand_index] = PatternOperandId{supported_operand_index};
    }
    EXPECT_TRUE(std::all_of(seen_operands.begin(), seen_operands.end(), [](bool seen) { return seen; }));

    std::vector<bool> seen_ops(user_pattern.getNumOps(), false);
    for (std::size_t supported_op_index = 0; supported_op_index < supported_pattern.getNumOps(); ++supported_op_index) {
        const std::optional<OperationValue>& attributes = args.getOpArgument(PatternOperationId{supported_op_index});
        ASSERT_TRUE(attributes.has_value());
        const std::size_t user_op_index =
            static_cast<std::size_t>(std::get<GemmAttributes>(*attributes).getComputeType()) - 1;
        ASSERT_LT(user_op_index, seen_ops.size());
        ASSERT_FALSE(seen_ops[user_op_index]);
        seen_ops[user_op_index]                    = true;
        images.supported_op_by_user[user_op_index] = PatternOperationId{supported_op_index};
    }
    EXPECT_TRUE(std::all_of(seen_ops.begin(), seen_ops.end(), [](bool seen) { return seen; }));
}

void expectCompleteValidMapping(const Pattern& user_pattern, const Pattern& supported_pattern,
                                const RoleImages& images) {
    ASSERT_EQ(images.supported_operand_by_user.size(), user_pattern.getNumOperands());
    ASSERT_EQ(images.supported_op_by_user.size(), user_pattern.getNumOps());
    ASSERT_EQ(user_pattern.getNumOperands(), supported_pattern.getNumOperands());
    ASSERT_EQ(user_pattern.getNumOps(), supported_pattern.getNumOps());

    std::vector<bool> seen_supported_operands(supported_pattern.getNumOperands(), false);
    for (std::size_t user_operand_index = 0; user_operand_index < user_pattern.getNumOperands(); ++user_operand_index) {
        const PatternOperandId user_operand_id{user_operand_index};
        const PatternOperandId supported_operand_id = images.supported_operand_by_user[user_operand_index];
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
        const PatternOperationId supported_op_id = images.supported_op_by_user[user_op_index];
        ASSERT_LT(supported_op_id.getIndex(), supported_pattern.getNumOps());
        EXPECT_FALSE(seen_supported_ops[supported_op_id.getIndex()]);
        seen_supported_ops[supported_op_id.getIndex()] = true;

        const PatternOperationNode& user_op_node      = user_pattern.getOpNode(user_op_id);
        const PatternOperationNode& supported_op_node = supported_pattern.getOpNode(supported_op_id);
        ASSERT_EQ(user_op_node.getKind(), supported_op_node.getKind());
        ASSERT_EQ(user_op_node.getInputs().size(), supported_op_node.getInputs().size());
        ASSERT_EQ(user_op_node.getOutputs().size(), supported_op_node.getOutputs().size());

        for (std::size_t port_index = 0; port_index < user_op_node.getInputs().size(); ++port_index) {
            EXPECT_EQ(images.supported_operand_by_user[user_op_node.getInputs()[port_index].getIndex()],
                      supported_op_node.getInputs()[port_index]);
        }
        for (std::size_t port_index = 0; port_index < user_op_node.getOutputs().size(); ++port_index) {
            EXPECT_EQ(images.supported_operand_by_user[user_op_node.getOutputs()[port_index].getIndex()],
                      supported_op_node.getOutputs()[port_index]);
        }
    }
    EXPECT_TRUE(std::all_of(seen_supported_ops.begin(), seen_supported_ops.end(), [](bool seen) { return seen; }));
}

void expectSignatureCorrespondence(const PatternBuilder& user_builder, const PatternBuilder& supported_builder) {
    const Pattern user_pattern      = user_builder.buildPattern();
    const Pattern supported_pattern = supported_builder.buildPattern();

    ASSERT_EQ(user_pattern.getKey(), supported_pattern.getKey());
    ASSERT_EQ(user_pattern.getNumOperands(), user_builder.buildPattern().getNumOperands());
    ASSERT_EQ(user_pattern.getNumOps(), user_builder.buildPattern().getNumOps());

    // The Ops constructor pairs equal-key structures directly and falls back
    // to the exact search when that pairing does not verify.
    RoleImages images;
    matchRoles(user_pattern, supported_pattern, images);
    expectCompleteValidMapping(user_pattern, supported_pattern, images);
}

PatternOperationId addGemm(PatternBuilder& pattern, FTrainTensorId a, FTrainTensorId b, FTrainTensorId d) {
    const FTrainTensorId c     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId alpha = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId beta  = pattern.addOperand<OperandKind::kTensor>();
    return PatternOperationId{pattern.addOperation<OperationKind::kGemm>(a, b, c, d, alpha, beta).opaque};
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

static_assert(std::is_same_v<decltype(std::declval<const PatternBuilder&>().buildPattern()), Pattern>);
static_assert(noexcept(std::declval<const Pattern&>().getKey()));
static_assert(std::is_nothrow_move_constructible_v<PatternKey>);
static_assert(std::is_nothrow_move_assignable_v<PatternKey>);
static_assert(noexcept(std::declval<const Pattern&>().getNumOperands()));
static_assert(noexcept(std::declval<const Pattern&>().getNumOps()));

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

    RoleImages result;
    matchRoles(user_pattern.buildPattern(), supported_pattern.buildPattern(), result);

    expectCompleteValidMapping(user_pattern.buildPattern(), supported_pattern.buildPattern(), result);
    EXPECT_EQ(result.supported_operand_by_user[user_input.opaque], PatternOperandId{supported_input.opaque});
    EXPECT_EQ(result.supported_operand_by_user[user_first_scale.opaque],
              PatternOperandId{supported_first_scale.opaque});
    EXPECT_EQ(result.supported_operand_by_user[user_intermediate.opaque],
              PatternOperandId{supported_intermediate.opaque});
    EXPECT_EQ(result.supported_operand_by_user[user_second_scale.opaque],
              PatternOperandId{supported_second_scale.opaque});
    EXPECT_EQ(result.supported_operand_by_user[user_output.opaque], PatternOperandId{supported_output.opaque});
    EXPECT_EQ(result.supported_op_by_user[user_first_op.getIndex()], supported_first_op);
    EXPECT_EQ(result.supported_op_by_user[user_second_op.getIndex()], supported_second_op);
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

    RoleImages result;
    matchRoles(user_pattern.buildPattern(), supported_pattern.buildPattern(), result);
    expectCompleteValidMapping(user_pattern.buildPattern(), supported_pattern.buildPattern(), result);

    PatternBuilder missing_component_user_pattern;
    const FTrainTensorId missing_input  = missing_component_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId missing_scale  = missing_component_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId missing_output = missing_component_user_pattern.addOperand<OperandKind::kTensor>();
    addGemm(missing_component_user_pattern, missing_input, missing_scale, missing_output);
    expectUnsupported([&] {
        static_cast<void>(Ops(missing_component_user_pattern.buildPattern(), supported_pattern.buildPattern()));
    });

    const FTrainTensorId extra_input  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId extra_scale  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId extra_output = user_pattern.addOperand<OperandKind::kTensor>();
    addGemm(user_pattern, extra_input, extra_scale, extra_output);
    expectUnsupported([&] { static_cast<void>(Ops(user_pattern.buildPattern(), supported_pattern.buildPattern())); });
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

    expectUnsupported([&] { static_cast<void>(Ops(user_pattern.buildPattern(), supported_pattern.buildPattern())); });
}

TEST(MatcherTest, RejectsOperandAndOperationCountMismatchesIndependently) {
    PatternBuilder supported_pattern;
    static_cast<void>(supported_pattern.addOperand<OperandKind::kTensor>());
    static_cast<void>(supported_pattern.addOperand<OperandKind::kTensor>());

    PatternBuilder user_pattern_with_extra_operand;
    static_cast<void>(user_pattern_with_extra_operand.addOperand<OperandKind::kTensor>());
    static_cast<void>(user_pattern_with_extra_operand.addOperand<OperandKind::kTensor>());
    static_cast<void>(user_pattern_with_extra_operand.addOperand<OperandKind::kTensor>());
    expectUnsupported([&] {
        static_cast<void>(Ops(user_pattern_with_extra_operand.buildPattern(), supported_pattern.buildPattern()));
    });

    PatternBuilder user_pattern_with_extra_op;
    const FTrainTensorId input  = user_pattern_with_extra_op.addOperand<OperandKind::kTensor>();
    const FTrainTensorId scale  = user_pattern_with_extra_op.addOperand<OperandKind::kTensor>();
    const FTrainTensorId output = user_pattern_with_extra_op.addOperand<OperandKind::kTensor>();
    addGemm(user_pattern_with_extra_op, input, scale, output);
    expectUnsupported(
        [&] { static_cast<void>(Ops(user_pattern_with_extra_op.buildPattern(), supported_pattern.buildPattern())); });
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
    expectUnsupported([&] {
        static_cast<void>(Ops(independent_user_pattern.buildPattern(), fan_out_supported_pattern.buildPattern()));
    });

    PatternBuilder fan_out_user_pattern;
    const FTrainTensorId shared_input_user          = fan_out_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_scale_user_pattern   = fan_out_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_output_user_pattern  = fan_out_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_scale_user_pattern  = fan_out_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_output_user_pattern = fan_out_user_pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(fan_out_user_pattern.addOperand<OperandKind::kTensor>());
    addGemm(fan_out_user_pattern, shared_input_user, second_scale_user_pattern, second_output_user_pattern);
    addGemm(fan_out_user_pattern, shared_input_user, first_scale_user_pattern, first_output_user_pattern);

    RoleImages fan_out_result;
    matchRoles(fan_out_user_pattern.buildPattern(), fan_out_supported_pattern.buildPattern(), fan_out_result);
    expectCompleteValidMapping(fan_out_user_pattern.buildPattern(), fan_out_supported_pattern.buildPattern(),
                               fan_out_result);
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

    expectUnsupported([&] { static_cast<void>(Ops(user_pattern.buildPattern(), supported_pattern.buildPattern())); });
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

    RoleImages result;
    matchRoles(user_pattern.buildPattern(), supported_pattern.buildPattern(), result);
    expectCompleteValidMapping(user_pattern.buildPattern(), supported_pattern.buildPattern(), result);
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

    RoleImages result;
    matchRoles(user_pattern.buildPattern(), supported_pattern.buildPattern(), result);
    expectCompleteValidMapping(user_pattern.buildPattern(), supported_pattern.buildPattern(), result);
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

    RoleImages result;
    matchRoles(user_pattern.buildPattern(), supported_pattern.buildPattern(), result);

    expectCompleteValidMapping(user_pattern.buildPattern(), supported_pattern.buildPattern(), result);
    EXPECT_EQ(result.supported_operand_by_user[short_user_operands.front().opaque],
              PatternOperandId{short_supported_operands.front().opaque});
    EXPECT_EQ(result.supported_operand_by_user[long_user_operands.front().opaque],
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
        different_a, different_b, different_c, different_output, different_alpha, different_beta));
    expectUnsupported(
        [&] { static_cast<void>(Ops(different_op_pattern.buildPattern(), supported_pattern.buildPattern())); });

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
    expectUnsupported([&] {
        static_cast<void>(
            Ops(isolated_grouped_user_pattern.buildPattern(), isolated_list_supported_pattern.buildPattern()));
    });

    PatternBuilder isolated_list_user_pattern;
    const FTrainTensorId list_user_input  = isolated_list_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId list_user_scale  = isolated_list_user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId list_user_output = isolated_list_user_pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(isolated_list_user_pattern.addOperand<OperandKind::kTensorList>());
    addGemm(isolated_list_user_pattern, list_user_input, list_user_scale, list_user_output);
    RoleImages result;
    matchRoles(isolated_list_user_pattern.buildPattern(), isolated_list_supported_pattern.buildPattern(), result);
    expectCompleteValidMapping(isolated_list_user_pattern.buildPattern(),
                               isolated_list_supported_pattern.buildPattern(), result);
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
    RoleImages result;
    matchRoles(user_pattern.buildPattern(), supported_pattern.buildPattern(), result);
    EXPECT_EQ(result.supported_operand_by_user[user_input.opaque], PatternOperandId{supported_input.opaque});
    EXPECT_EQ(result.supported_operand_by_user[user_first_scale.opaque],
              PatternOperandId{supported_first_scale.opaque});
    EXPECT_EQ(result.supported_operand_by_user[user_intermediate.opaque],
              PatternOperandId{supported_intermediate.opaque});
    EXPECT_EQ(result.supported_operand_by_user[user_second_scale.opaque],
              PatternOperandId{supported_second_scale.opaque});
    EXPECT_EQ(result.supported_operand_by_user[user_output.opaque], PatternOperandId{supported_output.opaque});
    EXPECT_EQ(result.supported_op_by_user[user_first_op.getIndex()], supported_first_op);
    EXPECT_EQ(result.supported_op_by_user[user_second_op.getIndex()], supported_second_op);
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
    expectUnsupported([&] { static_cast<void>(Ops(ordered_inputs.buildPattern(), swapped_inputs.buildPattern())); });
}

TEST(PatternStructuralColoringTest, DistinguishesKindsAndGloballyDifferentTopology) {
    PatternBuilder tensor_pattern;
    static_cast<void>(tensor_pattern.addOperand<OperandKind::kTensor>());
    PatternBuilder tensor_list_pattern;
    static_cast<void>(tensor_list_pattern.addOperand<OperandKind::kTensorList>());

    EXPECT_NE(tensor_pattern.buildPattern().getKey(), tensor_list_pattern.buildPattern().getKey());
    expectUnsupported(
        [&] { static_cast<void>(Ops(tensor_pattern.buildPattern(), tensor_list_pattern.buildPattern())); });

    PatternBuilder six_cycle;
    addGemmCycle(six_cycle, 6);
    PatternBuilder two_three_cycles;
    addGemmCycle(two_three_cycles, 3);
    addGemmCycle(two_three_cycles, 3);

    // Cycles were the classic refinement-indistinguishable pair: a six
    // cycle and two three cycles produced equal keys, so discrimination
    // fell to the exact matcher. buildPattern now rejects cyclic wiring
    // outright, so those topologies never reach coloring or matching.
    expectInvalidArgument([&] { static_cast<void>(six_cycle.buildPattern()); });
    expectInvalidArgument([&] { static_cast<void>(two_three_cycles.buildPattern()); });
}

}  // namespace
}  // namespace ftrain
