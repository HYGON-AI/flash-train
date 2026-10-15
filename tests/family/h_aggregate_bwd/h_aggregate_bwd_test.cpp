// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/h_aggregate_bwd/family.hpp"
#include "flash_train/family/h_aggregate_bwd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeHAggregateBwdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId grad_h_pre      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_x          = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId h_pre           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId x               = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_aggregated = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{
        pattern.addOperation<OperationKind::kHAggregateBwd>(grad_aggregated, x, h_pre, grad_x, grad_h_pre).opaque});
    return pattern;
}

TEST(HAggregateBwdOpsEngineTest, HAggregateBwdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<HAggregateBwdFamily, HAggregateBwdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 5U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kHAggregateBwd);
    const PatternBuilder user_pattern = makeHAggregateBwdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(HAggregateBwdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeHAggregateBwdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(HAggregateBwdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = HAggregateBwdFamily::makeRecords();
    ASSERT_EQ(records.size(), 2U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("MixedFp32HAggregateBwd"), 1U);
    EXPECT_EQ(names.count("Fp32HAggregateBwd"), 1U);
}

}  // namespace
}  // namespace ftrain
