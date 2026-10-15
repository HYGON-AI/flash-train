// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/sinkhorn_fwd/family.hpp"
#include "flash_train/family/sinkhorn_fwd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeSinkhornFwdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId eps            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId num_iterations = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId m_init         = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId output         = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId logits         = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{
        pattern.addOperation<OperationKind::kSinkhornFwd>(logits, output, m_init, num_iterations, eps).opaque});
    return pattern;
}

TEST(SinkhornFwdOpsEngineTest, SinkhornFwdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<SinkhornFwdFamily, SinkhornFwdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 5U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kSinkhornFwd);
    const PatternBuilder user_pattern = makeSinkhornFwdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(SinkhornFwdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeSinkhornFwdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(SinkhornFwdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = SinkhornFwdFamily::makeRecords();
    ASSERT_EQ(records.size(), 3U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("Bf16SinkhornFwd"), 1U);
    EXPECT_EQ(names.count("Fp16SinkhornFwd"), 1U);
    EXPECT_EQ(names.count("Fp32SinkhornFwd"), 1U);
}

}  // namespace
}  // namespace ftrain
