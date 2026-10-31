// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/h_post_bda_bwd/family.hpp"
#include "flash_train/family/h_post_bda_bwd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeHPostBdaBwdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId grad_bias     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_h_post   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_h_res    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_x        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_residual = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId h_post        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId h_res         = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId bias          = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId x             = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId residual      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_output   = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{
        pattern
            .addOperation<OperationKind::kHPostBdaBwd>(grad_output, residual, x, bias, h_res, h_post, grad_residual,
                                                       grad_x, grad_h_res, grad_h_post, grad_bias)
            .opaque});
    return pattern;
}

TEST(HPostBdaBwdOpsEngineTest, HPostBdaBwdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<HPostBdaBwdFamily, HPostBdaBwdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 11U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kHPostBdaBwd);
    const PatternBuilder user_pattern = makeHPostBdaBwdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(HPostBdaBwdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeHPostBdaBwdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(HPostBdaBwdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = HPostBdaBwdFamily::makeRecords();
    ASSERT_EQ(records.size(), 3U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("Bf16HPostBdaBwd"), 1U);
    EXPECT_EQ(names.count("Fp16HPostBdaBwd"), 1U);
    EXPECT_EQ(names.count("Fp32HPostBdaBwd"), 1U);
}

}  // namespace
}  // namespace ftrain
