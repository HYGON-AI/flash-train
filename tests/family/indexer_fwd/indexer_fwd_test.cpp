// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/indexer_fwd/family.hpp"
#include "flash_train/family/indexer_fwd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeIndexerFwdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId sm_scale = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId ratio    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId scores   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId w        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId k        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId q        = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(
        PatternOperationId{pattern.addOperation<OperationKind::kIndexerFwd>(q, k, w, scores, ratio, sm_scale).opaque});
    return pattern;
}

TEST(IndexerFwdOpsEngineTest, IndexerFwdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<IndexerFwdFamily, IndexerFwdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 6U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kIndexerFwd);
    const PatternBuilder user_pattern = makeIndexerFwdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(IndexerFwdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeIndexerFwdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(IndexerFwdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = IndexerFwdFamily::makeRecords();
    ASSERT_EQ(records.size(), 4U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("Bf16IndexerFwd"), 1U);
    EXPECT_EQ(names.count("Fp16IndexerFwd"), 1U);
    EXPECT_EQ(names.count("Fp32IndexerFwd"), 1U);
    EXPECT_EQ(names.count("MixedFp32IndexerFwd"), 1U);
}

}  // namespace
}  // namespace ftrain
