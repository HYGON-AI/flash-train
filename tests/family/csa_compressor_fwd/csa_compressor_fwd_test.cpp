// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/csa_compressor_fwd/family.hpp"
#include "flash_train/family/csa_compressor_fwd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeCSACompressorFwdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId total_comp      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId coff            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId head_dim        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId ratio           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId compressed      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId cu_seqlens_comp = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId cu_seqlens      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId ape             = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId score           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId kv              = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{
        pattern
            .addOperation<OperationKind::kCSACompressorFwd>(kv, score, ape, cu_seqlens, cu_seqlens_comp, compressed,
                                                            ratio, head_dim, coff, total_comp)
            .opaque});
    return pattern;
}

TEST(CSACompressorFwdOpsEngineTest, CSACompressorFwdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<CSACompressorFwdFamily, CSACompressorFwdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 10U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kCSACompressorFwd);
    const PatternBuilder user_pattern = makeCSACompressorFwdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(CSACompressorFwdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeCSACompressorFwdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(CSACompressorFwdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = CSACompressorFwdFamily::makeRecords();
    ASSERT_EQ(records.size(), 1U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("CSACompressorFwd"), 1U);
}

}  // namespace
}  // namespace ftrain
