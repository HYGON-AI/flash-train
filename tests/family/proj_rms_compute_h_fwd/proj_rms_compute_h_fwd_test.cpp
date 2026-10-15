// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/proj_rms_compute_h_fwd/family.hpp"
#include "flash_train/family/proj_rms_compute_h_fwd/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeProjRmsComputeHFwdPattern() {
    PatternBuilder pattern;
    const FTrainTensorId compute_h_eps = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId eps           = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId n             = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId proj          = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId r             = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId h_res         = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId h_post        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId h_pre         = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId bias          = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId alpha_res     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId alpha_post    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId alpha_pre     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId weight        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId x             = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{
        pattern
            .addOperation<OperationKind::kProjRmsComputeHFwd>(x, weight, alpha_pre, alpha_post, alpha_res, bias, h_pre,
                                                              h_post, h_res, r, proj, n, eps, compute_h_eps)
            .opaque});
    return pattern;
}

TEST(ProjRmsComputeHFwdOpsEngineTest, ProjRmsComputeHFwdEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<ProjRmsComputeHFwdFamily, ProjRmsComputeHFwdRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 14U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kProjRmsComputeHFwd);
    const PatternBuilder user_pattern = makeProjRmsComputeHFwdPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(ProjRmsComputeHFwdOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeProjRmsComputeHFwdPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(ProjRmsComputeHFwdOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = ProjRmsComputeHFwdFamily::makeRecords();
    ASSERT_EQ(records.size(), 4U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("MixedFp32ProjRmsComputeHFwd"), 1U);
    EXPECT_EQ(names.count("Bf16ProjRmsComputeHFwd"), 1U);
    EXPECT_EQ(names.count("Fp16ProjRmsComputeHFwd"), 1U);
    EXPECT_EQ(names.count("Fp32ProjRmsComputeHFwd"), 1U);
}

}  // namespace
}  // namespace ftrain
