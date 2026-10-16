// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/dense_attn_score_thd/family.hpp"
#include "flash_train/family/dense_attn_score_thd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeDenseAttnScoreThdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId max_seqlen_kv    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId softmax_scale    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId ratio            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId denom            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId out              = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId q_causal_offsets = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId cu_seqlens_kv    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId cu_seqlens_q     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId lse              = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId k                = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId q                = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{
        pattern
            .addOperation<OperationKind::kDenseAttnScoreThd>(q, k, lse, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets,
                                                             out, denom, ratio, softmax_scale, max_seqlen_kv)
            .opaque});
    return pattern;
}

TEST(DenseAttnScoreThdOpsEngineTest, DenseAttnScoreThdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<DenseAttnScoreThdFamily, DenseAttnScoreThdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 11U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kDenseAttnScoreThd);
    const PatternBuilder user_pattern = makeDenseAttnScoreThdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(DenseAttnScoreThdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeDenseAttnScoreThdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(DenseAttnScoreThdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = DenseAttnScoreThdFamily::makeRecords();
    ASSERT_EQ(records.size(), 4U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("Bf16DenseAttnScoreThd"), 1U);
    EXPECT_EQ(names.count("Fp16DenseAttnScoreThd"), 1U);
    EXPECT_EQ(names.count("Fp32DenseAttnScoreThd"), 1U);
    EXPECT_EQ(names.count("MixedFp32DenseAttnScoreThd"), 1U);
}

}  // namespace
}  // namespace ftrain
