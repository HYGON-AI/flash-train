// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/family/sparse_indexer_predict/family.hpp"
#include "flash_train/family/sparse_indexer_predict/finder.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"

namespace ftrain {
namespace {

PatternBuilder makeSparseIndexerPredictPattern() {
    PatternBuilder pattern;
    const FTrainTensorId topk_indices_global = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId sm_scale            = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId predict             = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId topk_indices        = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId w                   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId k                   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId q                   = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{pattern
                                             .addOperation<OperationKind::kSparseIndexerPredict>(
                                                 q, k, w, topk_indices, predict, sm_scale, topk_indices_global)
                                             .opaque});
    return pattern;
}

TEST(SparseIndexerPredictOpsEngineTest, SparseIndexerPredictEngineDefinesTheExactSupportedPattern) {
    const OpsEngine<SparseIndexerPredictFamily, SparseIndexerPredictRegistrationOrderFinder> engine;
    const Pattern& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 7U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kSparseIndexerPredict);
    const PatternBuilder user_pattern = makeSparseIndexerPredictPattern();
    EXPECT_EQ(supported.getKey(), user_pattern.buildPattern().getKey());
}

TEST(SparseIndexerPredictOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern                      = makeSparseIndexerPredictPattern();
    const PatternKey key                              = pattern.buildPattern().getKey();
    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), key);
}

TEST(SparseIndexerPredictOpsEngineTest, RegistersEveryNamedImplementationExactlyOnce) {
    const auto records = SparseIndexerPredictFamily::makeRecords();
    ASSERT_EQ(records.size(), 4U);
    std::set<std::string> names;
    for (const auto& record : records) { EXPECT_TRUE(names.insert(record->getName()).second); }
    EXPECT_EQ(names.count("Bf16SparseIndexerPredict"), 1U);
    EXPECT_EQ(names.count("Fp16SparseIndexerPredict"), 1U);
    EXPECT_EQ(names.count("Fp32SparseIndexerPredict"), 1U);
    EXPECT_EQ(names.count("MixedFp32SparseIndexerPredict"), 1U);
}

}  // namespace
}  // namespace ftrain
