#include <cstddef>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/op_definitions.hpp"

namespace ftrain {
namespace {

static_assert(std::is_same_v<Operand<OperandKind::kTensor>::Id, FTrainTensorId>);
static_assert(std::is_same_v<Operand<OperandKind::kTensorList>::Id, FTrainTensorListId>);
static_assert(std::is_same_v<Operand<OperandKind::kGroupedTensor>::Id, FTrainGroupedTensorId>);
static_assert(std::is_same_v<Operation<OperationKind::kGemm>::Id, FTrainGemmOpId>);
static_assert(std::is_same_v<Operation<OperationKind::kGroupedABCDGemm>::Id, FTrainGroupedABCDGemmOpId>);
static_assert(std::is_same_v<Operation<OperationKind::kGroupedBCDGemm>::Id, FTrainGroupedBCDGemmOpId>);
static_assert(std::is_same_v<Operation<OperationKind::kGroupedABGemm>::Id, FTrainGroupedABGemmOpId>);

TEST(OperandTest, MakeIdAddsOperandOfItsKind) {
    Pattern pattern;
    const FTrainTensorId tensor          = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorListId tensor_list = Operand<OperandKind::kTensorList>::addToPattern(pattern);
    const FTrainGroupedTensorId grouped  = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);

    EXPECT_EQ(tensor.opaque, 0U);
    EXPECT_EQ(tensor_list.opaque, 1U);
    EXPECT_EQ(grouped.opaque, 2U);
    EXPECT_EQ(pattern.getNumOperands(), 3U);
    EXPECT_EQ(pattern.getOperandNode(OperandId{tensor.opaque}).getKind(), OperandKind::kTensor);
    EXPECT_EQ(pattern.getOperandNode(OperandId{tensor_list.opaque}).getKind(), OperandKind::kTensorList);
    EXPECT_EQ(pattern.getOperandNode(OperandId{grouped.opaque}).getKind(), OperandKind::kGroupedTensor);
}

TEST(OperationTest, MakeIdAddsGemmWithTypedPorts) {
    Pattern pattern;
    using Tensor           = Operand<OperandKind::kTensor>;
    const Tensor::Id a     = Tensor::addToPattern(pattern);
    const Tensor::Id b     = Tensor::addToPattern(pattern);
    const Tensor::Id c     = Tensor::addToPattern(pattern);
    const Tensor::Id d     = Tensor::addToPattern(pattern);
    const Tensor::Id alpha = Tensor::addToPattern(pattern);
    const Tensor::Id beta  = Tensor::addToPattern(pattern);

    const FTrainGemmOpId op = Operation<OperationKind::kGemm>::addToPattern(pattern, a, b, c, d, alpha, beta);

    EXPECT_EQ(op.opaque, 0U);
    ASSERT_EQ(pattern.getNumOps(), 1U);
    const PatternOpNode& node = pattern.getOpNode(OperationId{op.opaque});
    EXPECT_EQ(node.getKind(), OperationKind::kGemm);
    EXPECT_EQ(node.getInputs(), (std::vector<OperandId>{OperandId{a.opaque}, OperandId{b.opaque}, OperandId{c.opaque},
                                                        OperandId{alpha.opaque}, OperandId{beta.opaque}}));
    EXPECT_EQ(node.getOutputs(), (std::vector<OperandId>{OperandId{d.opaque}}));
}

TEST(OperationTest, MakeIdAddsGroupedBCDGemmWithHeterogeneousPorts) {
    Pattern pattern;
    using Tensor              = Operand<OperandKind::kTensor>;
    using TensorList          = Operand<OperandKind::kTensorList>;
    using GroupedTensor       = Operand<OperandKind::kGroupedTensor>;
    const TensorList::Id a    = TensorList::addToPattern(pattern);
    const GroupedTensor::Id b = GroupedTensor::addToPattern(pattern);
    const GroupedTensor::Id c = GroupedTensor::addToPattern(pattern);
    const GroupedTensor::Id d = GroupedTensor::addToPattern(pattern);
    const Tensor::Id alpha    = Tensor::addToPattern(pattern);
    const Tensor::Id beta     = Tensor::addToPattern(pattern);

    const FTrainGroupedBCDGemmOpId op =
        Operation<OperationKind::kGroupedBCDGemm>::addToPattern(pattern, a, b, c, d, alpha, beta);

    EXPECT_EQ(op.opaque, 0U);
    const PatternOpNode& node = pattern.getOpNode(OperationId{op.opaque});
    EXPECT_EQ(node.getKind(), OperationKind::kGroupedBCDGemm);
    EXPECT_EQ(node.getInputs(), (std::vector<OperandId>{OperandId{a.opaque}, OperandId{b.opaque}, OperandId{c.opaque},
                                                        OperandId{alpha.opaque}, OperandId{beta.opaque}}));
    EXPECT_EQ(node.getOutputs(), (std::vector<OperandId>{OperandId{d.opaque}}));
}

}  // namespace
}  // namespace ftrain
