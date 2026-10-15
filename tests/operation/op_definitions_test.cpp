// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

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

static_assert(std::is_same_v<OperationTraits<OperationKind::kCompactify>::Id, FTrainCompactifyOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kCSACompressorBwd>::Id, FTrainCSACompressorBwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kCSACompressorFwd>::Id, FTrainCSACompressorFwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kDenseAttnScore>::Id, FTrainDenseAttnScoreOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kDenseAttnScoreThd>::Id, FTrainDenseAttnScoreThdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kDenseIndexerScore>::Id, FTrainDenseIndexerScoreOpId>);
static_assert(
    std::is_same_v<OperationTraits<OperationKind::kDenseIndexerScoreThd>::Id, FTrainDenseIndexerScoreThdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kHAggregateBwd>::Id, FTrainHAggregateBwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kHAggregateFwd>::Id, FTrainHAggregateFwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kHPostBdaBwd>::Id, FTrainHPostBdaBwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kHPostBdaFwd>::Id, FTrainHPostBdaFwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kIndexerBwdDense>::Id, FTrainIndexerBwdDenseOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kIndexerBwdDenseThd>::Id, FTrainIndexerBwdDenseThdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kIndexerBwdSparse>::Id, FTrainIndexerBwdSparseOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kIndexerFwd>::Id, FTrainIndexerFwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kIndexerFwdThd>::Id, FTrainIndexerFwdThdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kIndexerTopK>::Id, FTrainIndexerTopKOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kIndexerTopKThd>::Id, FTrainIndexerTopKThdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kProjRmsBwd>::Id, FTrainProjRmsBwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kProjRmsComputeHBwd>::Id, FTrainProjRmsComputeHBwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kProjRmsComputeHFwd>::Id, FTrainProjRmsComputeHFwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kProjRmsFwd>::Id, FTrainProjRmsFwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kSinkhornBwd>::Id, FTrainSinkhornBwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kSinkhornFwd>::Id, FTrainSinkhornFwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kSparseAttnBwd>::Id, FTrainSparseAttnBwdOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kSparseAttnFwd>::Id, FTrainSparseAttnFwdOpId>);
static_assert(
    std::is_same_v<OperationTraits<OperationKind::kSparseAttnFwdIndexer>::Id, FTrainSparseAttnFwdIndexerOpId>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kSparseAttnTarget>::Id, FTrainSparseAttnTargetOpId>);
static_assert(
    std::is_same_v<OperationTraits<OperationKind::kSparseIndexerPredict>::Id, FTrainSparseIndexerPredictOpId>);

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
