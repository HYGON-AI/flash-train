// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/sparse_attn_fwd/family.hpp"
#include "flash_train/family/sparse_attn_fwd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeSparseAttnFwdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId softmax_scale = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId lse           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId out           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId topk_length   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId topk_indices  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId attn_sink     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId kv            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId q             = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{pattern
                                             .addOperation<OperationKind::kSparseAttnFwd>(
                                                 q, kv, attn_sink, topk_indices, topk_length, out, lse, softmax_scale)
                                             .opaque});
    return pattern;
}

TEST(SparseAttnFwdOpsEngineTest, SparseAttnFwdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<SparseAttnFwdFamily, SparseAttnFwdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 8U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kSparseAttnFwd);
    const PatternBuilder user_pattern = makeSparseAttnFwdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(SparseAttnFwdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeSparseAttnFwdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(SparseAttnFwdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = SparseAttnFwdFamily::makeRecords();
    ASSERT_EQ(records.size(), 4U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("Bf16SparseAttnFwd"), 1U);
    EXPECT_EQ(names.count("Fp16SparseAttnFwd"), 1U);
    EXPECT_EQ(names.count("Fp32SparseAttnFwd"), 1U);
    EXPECT_EQ(names.count("MixedFp32SparseAttnFwd"), 1U);
}

}  // namespace
}  // namespace ftrain
