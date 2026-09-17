#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/api.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"

namespace ftrain {
namespace {

template<OperandKind Kind>
auto addRole(FTrainPattern pattern) {
    return pattern->builder.addOperand<Kind>();
}

void expectOperation(FTrainPattern pattern, std::size_t index, OperationKind kind,
                     std::initializer_list<std::uint64_t> inputs, std::initializer_list<std::uint64_t> outputs) {
    const Pattern prepared(pattern->builder.buildPattern());
    const PatternOperationNode& node = prepared.getOpNode(PatternOperationId{index});
    EXPECT_EQ(node.getKind(), kind);

    std::vector<std::size_t> actual_inputs;
    actual_inputs.reserve(node.getInputs().size());
    for (const PatternOperandId input : node.getInputs()) { actual_inputs.push_back(input.getIndex()); }

    std::vector<std::size_t> actual_outputs;
    actual_outputs.reserve(node.getOutputs().size());
    for (const PatternOperandId output : node.getOutputs()) { actual_outputs.push_back(output.getIndex()); }

    EXPECT_EQ(actual_inputs, (std::vector<std::size_t>(inputs.begin(), inputs.end())));
    EXPECT_EQ(actual_outputs, (std::vector<std::size_t>(outputs.begin(), outputs.end())));
}

TEST(PatternApiTest, CreatesAndDestroysAnEmptyPattern) {
    FTrainPattern pattern = nullptr;

    EXPECT_EQ(ftrainPatternCreate(&pattern), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(pattern, nullptr);
    EXPECT_EQ(pattern->builder.getNumOperands(), 0);
    EXPECT_EQ(pattern->builder.getNumOps(), 0);

    EXPECT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS);
}

TEST(PatternApiTest, AddsOperandKindsInPatternOrder) {
    FTrainPattern pattern = nullptr;
    ASSERT_EQ(ftrainPatternCreate(&pattern), FTRAIN_STATUS_SUCCESS);

    FTrainTensorId tensor{std::numeric_limits<std::uint64_t>::max()};
    FTrainTensorListId tensor_list{std::numeric_limits<std::uint64_t>::max()};
    FTrainGroupedTensorId grouped_tensor{std::numeric_limits<std::uint64_t>::max()};

    EXPECT_EQ(ftrainPatternAddTensor(pattern, &tensor), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPatternAddTensorList(pattern, &tensor_list), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPatternAddGroupedTensor(pattern, &grouped_tensor), FTRAIN_STATUS_SUCCESS);

    EXPECT_EQ(tensor.opaque, 0);
    EXPECT_EQ(tensor_list.opaque, 1);
    EXPECT_EQ(grouped_tensor.opaque, 2);
    ASSERT_EQ(pattern->builder.getNumOperands(), 3);
    EXPECT_EQ(pattern->builder.buildPattern().getOperandNode(PatternOperandId{0}).getKind(), OperandKind::kTensor);
    EXPECT_EQ(pattern->builder.buildPattern().getOperandNode(PatternOperandId{1}).getKind(), OperandKind::kTensorList);
    EXPECT_EQ(pattern->builder.buildPattern().getOperandNode(PatternOperandId{2}).getKind(),
              OperandKind::kGroupedTensor);

    EXPECT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS);
}

TEST(PatternApiTest, AddsEveryDeclaredOperationWithOrderedPorts) {
    FTrainPattern pattern = nullptr;
    ASSERT_EQ(ftrainPatternCreate(&pattern), FTRAIN_STATUS_SUCCESS);
    std::size_t expected_op_index = 0;

    {
        const auto a     = addRole<OperandKind::kTensor>(pattern);
        const auto b     = addRole<OperandKind::kTensor>(pattern);
        const auto c     = addRole<OperandKind::kTensor>(pattern);
        const auto d     = addRole<OperandKind::kTensor>(pattern);
        const auto alpha = addRole<OperandKind::kTensor>(pattern);
        const auto beta  = addRole<OperandKind::kTensor>(pattern);
        FTrainGemmOpId op{std::numeric_limits<std::uint64_t>::max()};

        ASSERT_EQ(ftrainPatternAddGemm(pattern, &op, a, b, c, d, alpha, beta), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op.opaque, expected_op_index);
        expectOperation(pattern, expected_op_index++, OperationKind::kGemm,
                        {a.opaque, b.opaque, c.opaque, alpha.opaque, beta.opaque}, {d.opaque});
    }
    {
        const auto a     = addRole<OperandKind::kGroupedTensor>(pattern);
        const auto b     = addRole<OperandKind::kGroupedTensor>(pattern);
        const auto c     = addRole<OperandKind::kGroupedTensor>(pattern);
        const auto d     = addRole<OperandKind::kGroupedTensor>(pattern);
        const auto alpha = addRole<OperandKind::kTensor>(pattern);
        const auto beta  = addRole<OperandKind::kTensor>(pattern);
        FTrainGroupedABCDGemmOpId op{std::numeric_limits<std::uint64_t>::max()};

        ASSERT_EQ(ftrainPatternAddGroupedABCDGemm(pattern, &op, a, b, c, d, alpha, beta), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op.opaque, expected_op_index);
        expectOperation(pattern, expected_op_index++, OperationKind::kGroupedABCDGemm,
                        {a.opaque, b.opaque, c.opaque, alpha.opaque, beta.opaque}, {d.opaque});
    }
    {
        const auto a     = addRole<OperandKind::kTensorList>(pattern);
        const auto b     = addRole<OperandKind::kGroupedTensor>(pattern);
        const auto c     = addRole<OperandKind::kGroupedTensor>(pattern);
        const auto d     = addRole<OperandKind::kGroupedTensor>(pattern);
        const auto alpha = addRole<OperandKind::kTensor>(pattern);
        const auto beta  = addRole<OperandKind::kTensor>(pattern);
        FTrainGroupedBCDGemmOpId op{std::numeric_limits<std::uint64_t>::max()};

        ASSERT_EQ(ftrainPatternAddGroupedBCDGemm(pattern, &op, a, b, c, d, alpha, beta), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op.opaque, expected_op_index);
        expectOperation(pattern, expected_op_index++, OperationKind::kGroupedBCDGemm,
                        {a.opaque, b.opaque, c.opaque, alpha.opaque, beta.opaque}, {d.opaque});
    }
    {
        const auto a     = addRole<OperandKind::kGroupedTensor>(pattern);
        const auto b     = addRole<OperandKind::kGroupedTensor>(pattern);
        const auto c     = addRole<OperandKind::kTensorList>(pattern);
        const auto d     = addRole<OperandKind::kTensorList>(pattern);
        const auto alpha = addRole<OperandKind::kTensor>(pattern);
        const auto beta  = addRole<OperandKind::kTensor>(pattern);
        FTrainGroupedABGemmOpId op{std::numeric_limits<std::uint64_t>::max()};

        ASSERT_EQ(ftrainPatternAddGroupedABGemm(pattern, &op, a, b, c, d, alpha, beta), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op.opaque, expected_op_index);
        expectOperation(pattern, expected_op_index++, OperationKind::kGroupedABGemm,
                        {a.opaque, b.opaque, c.opaque, alpha.opaque, beta.opaque}, {d.opaque});
    }

    EXPECT_EQ(pattern->builder.getNumOps(), 4);
    EXPECT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS);
}

TEST(PatternApiTest, RejectsNullInputsWithoutMutatingOutputsOrPattern) {
    FTrainPattern pattern = nullptr;
    ASSERT_EQ(ftrainPatternCreate(&pattern), FTRAIN_STATUS_SUCCESS);

    FTrainTensorId tensor{73};
    EXPECT_EQ(ftrainPatternAddTensor(nullptr, &tensor), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(tensor.opaque, 73);
    EXPECT_EQ(pattern->builder.getNumOperands(), 0);

    EXPECT_EQ(ftrainPatternAddTensor(pattern, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pattern->builder.getNumOperands(), 0);
    EXPECT_EQ(ftrainPatternCreate(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPatternDestroy(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS);
}

TEST(PatternApiTest, AddsWithoutValidationAndRejectsOnlyNullOutputs) {
    FTrainPattern pattern = nullptr;
    ASSERT_EQ(ftrainPatternCreate(&pattern), FTRAIN_STATUS_SUCCESS);
    FTrainTensorId a;
    FTrainTensorId b;
    FTrainTensorId c;
    FTrainTensorId d;
    FTrainTensorId alpha;
    FTrainTensorId beta;
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &a), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &b), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &c), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &d), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &alpha), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &beta), FTRAIN_STATUS_SUCCESS);

    // Duplicate inputs are appended without validation; the conflicting
    // structure is rejected when the Pattern is built.
    FTrainGemmOpId op{83};
    EXPECT_EQ(ftrainPatternAddGemm(pattern, &op, a, a, c, d, alpha, beta), FTRAIN_STATUS_SUCCESS);
    EXPECT_NE(op.opaque, 83);
    EXPECT_EQ(pattern->builder.getNumOps(), 1);

    EXPECT_EQ(ftrainPatternAddGemm(pattern, nullptr, a, b, c, d, alpha, beta), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(pattern->builder.getNumOps(), 1);

    FTrainGemmOpId null_pattern_op{97};
    EXPECT_EQ(ftrainPatternAddGemm(nullptr, &null_pattern_op, a, b, c, d, alpha, beta), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(null_pattern_op.opaque, 97);

    EXPECT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS);
}

TEST(PatternApiTest, OpsCreateRejectsConflictingBuilderContents) {
    FTrainPattern pattern = nullptr;
    ASSERT_EQ(ftrainPatternCreate(&pattern), FTRAIN_STATUS_SUCCESS);
    FTrainTensorId a;
    FTrainTensorId b;
    FTrainTensorId c;
    FTrainTensorId d;
    FTrainTensorId alpha;
    FTrainTensorId beta;
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &a), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &b), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &c), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &d), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &alpha), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternAddTensor(pattern, &beta), FTRAIN_STATUS_SUCCESS);

    FTrainGemmOpId op{0};
    EXPECT_EQ(ftrainPatternAddGemm(pattern, &op, a, a, c, d, alpha, beta), FTRAIN_STATUS_SUCCESS);

    FTrainOps ops = nullptr;
    EXPECT_EQ(ftrainOpsCreate(&ops, pattern), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ops, nullptr);
    EXPECT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS);
}

}  // namespace
}  // namespace ftrain
