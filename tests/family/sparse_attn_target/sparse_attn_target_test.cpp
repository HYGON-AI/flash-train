// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/sparse_attn_target/family.hpp"
#include "flash_train/family/sparse_attn_target/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeSparseAttnTargetPattern() {
    PatternBuilder pattern;
    const FTrainTensorId topk_indices_global = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId softmax_scale       = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId target              = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId topk_indices        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId lse                 = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId k                   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId q                   = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{pattern
                                             .addOperation<OperationKind::kSparseAttnTarget>(
                                                 q, k, lse, topk_indices, target, softmax_scale, topk_indices_global)
                                             .opaque});
    return pattern;
}

TEST(SparseAttnTargetOpsEngineTest, SparseAttnTargetEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<SparseAttnTargetFamily, SparseAttnTargetRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 7U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kSparseAttnTarget);
    const PatternBuilder user_pattern = makeSparseAttnTargetPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(SparseAttnTargetOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeSparseAttnTargetPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(SparseAttnTargetOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = SparseAttnTargetFamily::makeRecords();
    ASSERT_FALSE(records.empty());
    std::set<std::string> names;
    std::string registered_primitives;
    for (const auto& record : records) {
        ASSERT_NE(record, nullptr);
        ASSERT_NE(record->getName(), nullptr);
        EXPECT_STRNE(record->getName(), "");
        EXPECT_TRUE(names.insert(record->getName()).second);
        const auto clone = record->clone();
        ASSERT_NE(clone, nullptr);
        EXPECT_NE(clone.get(), record.get());
        EXPECT_STREQ(clone->getName(), record->getName());
        registered_primitives += std::string(record->getName()) + "\n";
    }
    RecordProperty("family", "sparse_attn_target");
    RecordProperty("registered_primitives", registered_primitives);
    EXPECT_EQ(names.count("Bf16SparseAttnTarget"), 1U);
    EXPECT_EQ(names.count("Fp16SparseAttnTarget"), 1U);
    EXPECT_EQ(names.count("Fp32SparseAttnTarget"), 1U);
    EXPECT_EQ(names.count("MixedFp32SparseAttnTarget"), 1U);
}

}  // namespace
}  // namespace ftrain
