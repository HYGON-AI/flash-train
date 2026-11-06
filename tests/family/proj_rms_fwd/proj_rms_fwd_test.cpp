// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/proj_rms_fwd/family.hpp"
#include "flash_train/family/proj_rms_fwd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeProjRmsFwdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId eps    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId r      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId proj   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId weight = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId x      = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(
        PatternOperationId{pattern.addOperation<OperationKind::kProjRmsFwd>(x, weight, proj, r, eps).opaque});
    return pattern;
}

TEST(ProjRmsFwdOpsEngineTest, ProjRmsFwdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<ProjRmsFwdFamily, ProjRmsFwdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 5U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kProjRmsFwd);
    const PatternBuilder user_pattern = makeProjRmsFwdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(ProjRmsFwdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeProjRmsFwdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(ProjRmsFwdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = ProjRmsFwdFamily::makeRecords();
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
    RecordProperty("family", "proj_rms_fwd");
    RecordProperty("registered_primitives", registered_primitives);
    EXPECT_EQ(names.count("MixedFp32ProjRmsFwd"), 1U);
    EXPECT_EQ(names.count("Bf16ProjRmsFwd"), 1U);
    EXPECT_EQ(names.count("Fp16ProjRmsFwd"), 1U);
    EXPECT_EQ(names.count("Fp32ProjRmsFwd"), 1U);
}

}  // namespace
}  // namespace ftrain
