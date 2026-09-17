// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"

namespace ftrain {
namespace {

static_assert(std::is_same_v<OperandTraits<OperandKind::kTensor>::Id, FTrainTensorId>);
static_assert(std::is_same_v<OperandTraits<OperandKind::kTensorList>::Id, FTrainTensorListId>);
static_assert(std::is_same_v<OperandTraits<OperandKind::kGroupedTensor>::Id, FTrainGroupedTensorId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kGemm>::Id, FTrainGemmOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kGroupedABCDGemm>::Id, FTrainGroupedABCDGemmOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kGroupedBCDGemm>::Id, FTrainGroupedBCDGemmOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kGroupedABGemm>::Id, FTrainGroupedABGemmOpId>);

TEST(OperandTest, MakeIdAddsOperandOfItsKind) {
    PatternBuilder pattern;
    const FTrainTensorId tensor          = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorListId tensor_list = pattern.addOperand<OperandKind::kTensorList>();
    const FTrainGroupedTensorId grouped  = pattern.addOperand<OperandKind::kGroupedTensor>();

    EXPECT_EQ(tensor.opaque, 0U);
    EXPECT_EQ(tensor_list.opaque, 1U);
    EXPECT_EQ(grouped.opaque, 2U);
    EXPECT_EQ(pattern.buildPattern().getNumOperands(), 3U);
    EXPECT_EQ(pattern.buildPattern().getOperandNode(PatternOperandId{tensor.opaque}).getKind(), OperandKind::kTensor);
    EXPECT_EQ(pattern.buildPattern().getOperandNode(PatternOperandId{tensor_list.opaque}).getKind(),
              OperandKind::kTensorList);
    EXPECT_EQ(pattern.buildPattern().getOperandNode(PatternOperandId{grouped.opaque}).getKind(),
              OperandKind::kGroupedTensor);
}

TEST(OperationTest, MakeIdAddsGemmWithTypedPorts) {
    PatternBuilder pattern;
    using Tensor           = OperandTraits<OperandKind::kTensor>;
    const Tensor::Id a     = pattern.addOperand<OperandKind::kTensor>();
    const Tensor::Id b     = pattern.addOperand<OperandKind::kTensor>();
    const Tensor::Id c     = pattern.addOperand<OperandKind::kTensor>();
    const Tensor::Id d     = pattern.addOperand<OperandKind::kTensor>();
    const Tensor::Id alpha = pattern.addOperand<OperandKind::kTensor>();
    const Tensor::Id beta  = pattern.addOperand<OperandKind::kTensor>();

    const FTrainGemmOpId op = pattern.addOperation<OperationKind::kGemm>(a, b, c, d, alpha, beta);

    EXPECT_EQ(op.opaque, 0U);
    ASSERT_EQ(pattern.buildPattern().getNumOps(), 1U);
    const Pattern prepared           = pattern.buildPattern();
    const PatternOperationNode& node = prepared.getOpNode(PatternOperationId{op.opaque});
    EXPECT_EQ(node.getKind(), OperationKind::kGemm);
    EXPECT_EQ(node.getInputs(), (std::vector<PatternOperandId>{
                                    PatternOperandId{a.opaque}, PatternOperandId{b.opaque}, PatternOperandId{c.opaque},
                                    PatternOperandId{alpha.opaque}, PatternOperandId{beta.opaque}}));
    EXPECT_EQ(node.getOutputs(), (std::vector<PatternOperandId>{PatternOperandId{d.opaque}}));
}

TEST(OperationTest, MakeIdAddsGroupedBCDGemmWithHeterogeneousPorts) {
    PatternBuilder pattern;
    using Tensor              = OperandTraits<OperandKind::kTensor>;
    using TensorList          = OperandTraits<OperandKind::kTensorList>;
    using GroupedTensor       = OperandTraits<OperandKind::kGroupedTensor>;
    const TensorList::Id a    = pattern.addOperand<OperandKind::kTensorList>();
    const GroupedTensor::Id b = pattern.addOperand<OperandKind::kGroupedTensor>();
    const GroupedTensor::Id c = pattern.addOperand<OperandKind::kGroupedTensor>();
    const GroupedTensor::Id d = pattern.addOperand<OperandKind::kGroupedTensor>();
    const Tensor::Id alpha    = pattern.addOperand<OperandKind::kTensor>();
    const Tensor::Id beta     = pattern.addOperand<OperandKind::kTensor>();

    const FTrainGroupedBCDGemmOpId op = pattern.addOperation<OperationKind::kGroupedBCDGemm>(a, b, c, d, alpha, beta);

    EXPECT_EQ(op.opaque, 0U);
    const Pattern prepared           = pattern.buildPattern();
    const PatternOperationNode& node = prepared.getOpNode(PatternOperationId{op.opaque});
    EXPECT_EQ(node.getKind(), OperationKind::kGroupedBCDGemm);
    EXPECT_EQ(node.getInputs(), (std::vector<PatternOperandId>{
                                    PatternOperandId{a.opaque}, PatternOperandId{b.opaque}, PatternOperandId{c.opaque},
                                    PatternOperandId{alpha.opaque}, PatternOperandId{beta.opaque}}));
    EXPECT_EQ(node.getOutputs(), (std::vector<PatternOperandId>{PatternOperandId{d.opaque}}));
}

}  // namespace
}  // namespace ftrain
