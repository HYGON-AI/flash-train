#include <array>
#include <cstddef>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/gemm.hpp"
#include "flash_train/matcher.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/ops_engine.hpp"
#include "flash_train/registry.hpp"
#include "flash_train/pattern.hpp"

namespace ftrain {
namespace {

PatternBuilder makeGemmPattern() {
    PatternBuilder pattern;
    const FTrainTensorId d     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId beta  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId alpha = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId c     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId b     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId a     = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(PatternOperationId{pattern.addOperation<OperationKind::kGemm>({a, b, c, d, alpha, beta}).opaque});
    return pattern;
}

TEST(GemmOpsEngineTest, FactoryDefinesTheExactSupportedPattern) {
    const std::shared_ptr<OpsEngineBase> engine = makeGemmOpsEngine();
    ASSERT_NE(engine, nullptr);

    const Pattern& supported = engine->getSupportedPattern();
    ASSERT_EQ(supported.getNumOperands(), 6U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < supported.getNumOperands(); ++operand_index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{operand_index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kGemm);

    const PatternBuilder user_pattern = makeGemmPattern();
    EXPECT_EQ(engine->getPatternKey(), user_pattern.buildPattern().getKey());
}

TEST(GemmOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const PatternBuilder pattern = makeGemmPattern();
    const PatternKey key         = pattern.buildPattern().getKey();

    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPatternKey(), key);
}

}  // namespace
}  // namespace ftrain
