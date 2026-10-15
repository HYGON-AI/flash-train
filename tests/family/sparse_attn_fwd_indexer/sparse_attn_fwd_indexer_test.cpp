// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/sparse_attn_fwd_indexer/family.hpp"
#include "flash_train/family/sparse_attn_fwd_indexer/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeSparseAttnFwdIndexerPattern() {
    PatternBuilder pattern;
    const FTrainTensorId indexer_topk  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId softmax_scale = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId lse_indexer   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId lse           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId max_logits    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId out           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId topk_length   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId topk_indices  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId attn_sink     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId kv            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId q             = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{pattern
                                             .addOperation<OperationKind::kSparseAttnFwdIndexer>(
                                                 q, kv, attn_sink, topk_indices, topk_length, out, max_logits, lse,
                                                 lse_indexer, softmax_scale, indexer_topk)
                                             .opaque});
    return pattern;
}

TEST(SparseAttnFwdIndexerOpsEngineTest, SparseAttnFwdIndexerEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<SparseAttnFwdIndexerFamily, SparseAttnFwdIndexerRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 11U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kSparseAttnFwdIndexer);
    const PatternBuilder user_pattern = makeSparseAttnFwdIndexerPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(SparseAttnFwdIndexerOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeSparseAttnFwdIndexerPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(SparseAttnFwdIndexerOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = SparseAttnFwdIndexerFamily::makeRecords();
    ASSERT_EQ(records.size(), 4U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("Bf16SparseAttnFwdIndexer"), 1U);
    EXPECT_EQ(names.count("Fp16SparseAttnFwdIndexer"), 1U);
    EXPECT_EQ(names.count("Fp32SparseAttnFwdIndexer"), 1U);
    EXPECT_EQ(names.count("MixedFp32SparseAttnFwdIndexer"), 1U);
}

}  // namespace
}  // namespace ftrain
