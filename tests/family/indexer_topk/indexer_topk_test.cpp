// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/indexer_topk/family.hpp"
#include "flash_train/family/indexer_topk/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeIndexerTopKPattern() {
    PatternBuilder pattern;
    const FTrainTensorId order_values = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId next_n       = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId top_k        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId out_values   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId out_indices  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId seq_lens     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId input_values = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(
        PatternOperationId{pattern
                               .addOperation<OperationKind::kIndexerTopK>(input_values, seq_lens, out_indices,
                                                                          out_values, top_k, next_n, order_values)
                               .opaque});
    return pattern;
}

TEST(IndexerTopKOpsEngineTest, IndexerTopKEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<IndexerTopKFamily, IndexerTopKRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 7U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kIndexerTopK);
    const PatternBuilder user_pattern = makeIndexerTopKPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(IndexerTopKOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeIndexerTopKPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(IndexerTopKOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = IndexerTopKFamily::makeRecords();
    ASSERT_EQ(records.size(), 1U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("IndexerTopK"), 1U);
}

}  // namespace
}  // namespace ftrain
