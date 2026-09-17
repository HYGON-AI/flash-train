#include <array>
#include <cstddef>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/gemm.hpp"
#include "flash_train/matcher.hpp"
#include "flash_train/op_definitions.hpp"
#include "flash_train/op_schema.hpp"
#include "flash_train/ops_engine.hpp"
#include "flash_train/pattern.hpp"

namespace ftrain {
namespace {

Pattern makeGemmPattern() {
    Pattern pattern;
    const FTrainTensorId d     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId beta  = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId alpha = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId c     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId b     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId a     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    static_cast<void>(
        OperationId{Operation<OperationKind::kGemm>::addToPattern(pattern, a, b, c, d, alpha, beta).opaque});
    return pattern;
}

TEST(GemmOpsEngineTest, FactoryDefinesTheExactSupportedSchema) {
    const std::shared_ptr<OpsEngineBase> engine = makeGemmOpsEngine();
    ASSERT_NE(engine, nullptr);

    const SupportedSchema& schema = engine->getSupportedSchema();
    ASSERT_EQ(schema.getNumOperands(), 6U);
    ASSERT_EQ(schema.getNumOps(), 1U);
    for (std::size_t operand_index = 0; operand_index < schema.getNumOperands(); ++operand_index) {
        EXPECT_EQ(schema.getOperandKind(OperandId{operand_index}), OperandKind::kTensor);
    }
    EXPECT_EQ(schema.getOperationKind(OperationId{0}), OperationKind::kGemm);

    const Pattern user_pattern = makeGemmPattern();
    EXPECT_EQ(engine->getPatternKey(), Matcher::canonicalize(user_pattern).getKey());
}

TEST(GemmOpsEngineTest, BuiltinEngineIsRegisteredBeforeHandleIsObserved) {
    const Pattern pattern = makeGemmPattern();
    const PatternKey key  = Matcher::canonicalize(pattern).getKey();

    const std::shared_ptr<const OpsEngineBase> engine = getGlobalHandle().findOpsEngine(key);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPatternKey(), key);
}

}  // namespace
}  // namespace ftrain
