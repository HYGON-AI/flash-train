// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/dense_attn_score/family.hpp"
#include "flash_train/family/dense_attn_score/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeDenseAttnScorePattern() {
    PatternBuilder pattern;
    const FTrainTensorId softmax_scale = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId ratio         = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId denom         = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId out           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId lse           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId k             = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId q             = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{
        pattern.addOperation<OperationKind::kDenseAttnScore>(q, k, lse, out, denom, ratio, softmax_scale).opaque});
    return pattern;
}

TEST(DenseAttnScoreOpsEngineTest, DenseAttnScoreEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<DenseAttnScoreFamily, DenseAttnScoreRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 7U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kDenseAttnScore);
    const PatternBuilder user_pattern = makeDenseAttnScorePattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(DenseAttnScoreOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeDenseAttnScorePattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(DenseAttnScoreOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = DenseAttnScoreFamily::makeRecords();
    ASSERT_EQ(records.size(), 4U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("Bf16DenseAttnScore"), 1U);
    EXPECT_EQ(names.count("Fp16DenseAttnScore"), 1U);
    EXPECT_EQ(names.count("Fp32DenseAttnScore"), 1U);
    EXPECT_EQ(names.count("MixedFp32DenseAttnScore"), 1U);
}

}  // namespace
}  // namespace ftrain
