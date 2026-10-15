// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/h_post_bda_fwd/family.hpp"
#include "flash_train/family/h_post_bda_fwd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeHPostBdaFwdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId output   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId bias     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId x        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId h_post   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId residual = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId h_res    = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{
        pattern.addOperation<OperationKind::kHPostBdaFwd>(h_res, residual, h_post, x, bias, output).opaque});
    return pattern;
}

TEST(HPostBdaFwdOpsEngineTest, HPostBdaFwdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<HPostBdaFwdFamily, HPostBdaFwdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 6U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kHPostBdaFwd);
    const PatternBuilder user_pattern = makeHPostBdaFwdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(HPostBdaFwdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeHPostBdaFwdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(HPostBdaFwdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = HPostBdaFwdFamily::makeRecords();
    ASSERT_EQ(records.size(), 3U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("Bf16HPostBdaFwd"), 1U);
    EXPECT_EQ(names.count("Fp16HPostBdaFwd"), 1U);
    EXPECT_EQ(names.count("Fp32HPostBdaFwd"), 1U);
}

}  // namespace
}  // namespace ftrain
