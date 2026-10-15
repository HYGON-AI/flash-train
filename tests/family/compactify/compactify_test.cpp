// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/compactify/family.hpp"
#include "flash_train/family/compactify/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeCompactifyPattern() {
    PatternBuilder pattern;
    const FTrainTensorId topk_length = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId output      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId indices     = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(
        PatternOperationId{pattern.addOperation<OperationKind::kCompactify>(indices, output, topk_length).opaque});
    return pattern;
}

TEST(CompactifyOpsEngineTest, CompactifyEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<CompactifyFamily, CompactifyRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 3U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kCompactify);
    const PatternBuilder user_pattern = makeCompactifyPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(CompactifyOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeCompactifyPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(CompactifyOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = CompactifyFamily::makeRecords();
    ASSERT_EQ(records.size(), 1U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("Compactify"), 1U);
}

}  // namespace
}  // namespace ftrain
