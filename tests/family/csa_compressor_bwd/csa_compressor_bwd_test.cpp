// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/csa_compressor_bwd/family.hpp"
#include "flash_train/family/csa_compressor_bwd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeCSACompressorBwdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId coff            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId head_dim        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId ratio           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_ape        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_score      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_kv         = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grad_compressed = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId cu_seqlens_comp = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId cu_seqlens      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId ape             = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId score           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId kv              = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{pattern
                                             .addOperation<OperationKind::kCSACompressorBwd>(
                                                 kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_compressed, grad_kv,
                                                 grad_score, grad_ape, ratio, head_dim, coff)
                                             .opaque});
    return pattern;
}

TEST(CSACompressorBwdOpsEngineTest, CSACompressorBwdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<CSACompressorBwdFamily, CSACompressorBwdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 12U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kCSACompressorBwd);
    const PatternBuilder user_pattern = makeCSACompressorBwdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(CSACompressorBwdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeCSACompressorBwdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(CSACompressorBwdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = CSACompressorBwdFamily::makeRecords();
    ASSERT_EQ(records.size(), 1U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("CSACompressorBwd"), 1U);
}

}  // namespace
}  // namespace ftrain
