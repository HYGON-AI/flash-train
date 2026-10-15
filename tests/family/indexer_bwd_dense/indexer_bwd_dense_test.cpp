// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/indexer_bwd_dense/family.hpp"
#include "flash_train/family/indexer_bwd_dense/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeIndexerBwdDensePattern() {
    PatternBuilder pattern;
    const FTrainTensorId ratio            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_loss        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId loss_coeff       = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId sm_scale         = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId dk               = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId dw               = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId dq               = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_index_score = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId index_lse        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId index_score      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId attn_l1norm      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId attn_score       = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId index_k          = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId weights          = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId index_q          = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(
        PatternOperationId{pattern
                               .addOperation<OperationKind::kIndexerBwdDense>(
                                   index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse,
                                   grad_index_score, dq, dw, dk, sm_scale, loss_coeff, grad_loss, ratio)
                               .opaque});
    return pattern;
}

TEST(IndexerBwdDenseOpsEngineTest, IndexerBwdDenseEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<IndexerBwdDenseFamily, IndexerBwdDenseRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 15U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kIndexerBwdDense);
    const PatternBuilder user_pattern = makeIndexerBwdDensePattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(IndexerBwdDenseOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeIndexerBwdDensePattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(IndexerBwdDenseOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = IndexerBwdDenseFamily::makeRecords();
    ASSERT_EQ(records.size(), 4U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("Bf16IndexerBwdDense"), 1U);
    EXPECT_EQ(names.count("Fp16IndexerBwdDense"), 1U);
    EXPECT_EQ(names.count("Fp32IndexerBwdDense"), 1U);
    EXPECT_EQ(names.count("MixedFp32IndexerBwdDense"), 1U);
}

}  // namespace
}  // namespace ftrain
