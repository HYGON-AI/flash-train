// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/indexer_topk_thd/family.hpp"
#include "flash_train/family/indexer_topk_thd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeIndexerTopKThdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId order_values     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId top_k            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId ratio            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId out_values       = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId out_indices      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId q_causal_offsets = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId cu_seqlens_kv    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId cu_seqlens_q     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId input_values     = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{
        pattern
            .addOperation<OperationKind::kIndexerTopKThd>(input_values, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets,
                                                          out_indices, out_values, ratio, top_k, order_values)
            .opaque});
    return pattern;
}

TEST(IndexerTopKThdOpsEngineTest, IndexerTopKThdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<IndexerTopKThdFamily, IndexerTopKThdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 9U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kIndexerTopKThd);
    const PatternBuilder user_pattern = makeIndexerTopKThdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(IndexerTopKThdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeIndexerTopKThdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(IndexerTopKThdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = IndexerTopKThdFamily::makeRecords();
    ASSERT_EQ(records.size(), 1U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("IndexerTopKThd"), 1U);
}

}  // namespace
}  // namespace ftrain
