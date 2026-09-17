#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {
namespace {

template<typename Function>
void expectInvalidArgument(Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "Runtime interface accepted invalid input";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

Ops makeOps(const PatternBuilder& user_pattern, const PatternBuilder& supported_pattern) {
    return Ops(user_pattern.buildPattern(), supported_pattern.buildPattern());
}

Tensor makeTensor(void* memory, std::int64_t extent = 1) {
    const std::int64_t dims[]{extent};
    const FTrainStorageView description{
        memory, dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    return Tensor(StorageView(description));
}

TensorList makeTensorList(void* memory) {
    const std::int64_t dims[]{1};
    const FTrainStorageView description{
        memory, dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    std::vector<StorageView> storage_views;
    storage_views.emplace_back(description);
    return TensorList(std::move(storage_views));
}

GroupedTensor makeGroupedTensor(void* memory) {
    const std::int64_t dims[]{1};
    const FTrainStorageView description{
        memory, dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    StorageView data(description);
    std::optional<StorageView> offsets;
    std::vector<StorageView> dim_sizes;
    std::vector<StorageView> strides;
    return GroupedTensor(0, std::move(data), std::move(offsets), std::move(dim_sizes), std::move(strides));
}

template<OperationKind Kind>
PatternOperationId addStandaloneOp(PatternBuilder& pattern) {
    if constexpr (Kind == OperationKind::kGemm) {
        const FTrainTensorId a     = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId b     = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId c     = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId d     = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId alpha = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId beta  = pattern.addOperand<OperandKind::kTensor>();
        return PatternOperationId{pattern.addOperation<Kind>(a, b, c, d, alpha, beta).opaque};
    } else if constexpr (Kind == OperationKind::kGroupedABCDGemm) {
        const FTrainGroupedTensorId a = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId b = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId c = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId d = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainTensorId alpha    = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId beta     = pattern.addOperand<OperandKind::kTensor>();
        return PatternOperationId{pattern.addOperation<Kind>(a, b, c, d, alpha, beta).opaque};
    } else if constexpr (Kind == OperationKind::kGroupedBCDGemm) {
        const FTrainTensorListId a    = pattern.addOperand<OperandKind::kTensorList>();
        const FTrainGroupedTensorId b = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId c = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId d = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainTensorId alpha    = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId beta     = pattern.addOperand<OperandKind::kTensor>();
        return PatternOperationId{pattern.addOperation<Kind>(a, b, c, d, alpha, beta).opaque};
    } else if constexpr (Kind == OperationKind::kGroupedABGemm) {
        const FTrainGroupedTensorId a = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId b = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainTensorListId c    = pattern.addOperand<OperandKind::kTensorList>();
        const FTrainTensorListId d    = pattern.addOperand<OperandKind::kTensorList>();
        const FTrainTensorId alpha    = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId beta     = pattern.addOperand<OperandKind::kTensor>();
        return PatternOperationId{pattern.addOperation<Kind>(a, b, c, d, alpha, beta).opaque};
    } else {
        static_assert(Kind != Kind, "addStandaloneOp covers every declared OperationKind");
    }
}

static_assert(std::is_same_v<Operand, std::variant<Tensor, TensorList, GroupedTensor>>);
static_assert(std::is_same_v<OperandTraits<OperandKind::kTensor>::Type, Tensor>);
static_assert(std::is_same_v<OperandTraits<OperandKind::kTensorList>::Type, TensorList>);
static_assert(std::is_same_v<OperandTraits<OperandKind::kGroupedTensor>::Type, GroupedTensor>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kGemm>::Type, GemmAttributes>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kGroupedABCDGemm>::Type, GroupedABCDGemmAttributes>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kGroupedBCDGemm>::Type, GroupedBCDGemmAttributes>);
static_assert(std::is_same_v<OperationTraits<OperationKind::kGroupedABGemm>::Type, GroupedABGemmAttributes>);
static_assert(noexcept(std::declval<const Ops&>().getPatternKey()));
static_assert(noexcept(std::declval<const Ops&>().getRoleMapping()));
static_assert(noexcept(std::declval<const Args&>().getPatternKey()));
static_assert(noexcept(std::declval<const Args&>().getNumOperands()));
static_assert(noexcept(std::declval<const Args&>().getNumOps()));
static_assert(noexcept(std::declval<const Args&>().isComplete()));

TEST(RoleMappingTest, ComposesExactRolesAcrossDifferentAdditionOrders) {
    PatternBuilder supported_pattern;
    const FTrainTensorId supported_input         = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_b       = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_c       = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_alpha   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_beta    = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_intermediate  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_b      = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_c      = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_alpha  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_beta   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_output        = supported_pattern.addOperand<OperandKind::kTensor>();
    const PatternOperationId supported_first_op  = (PatternOperationId{
        supported_pattern
            .addOperation<OperationKind::kGemm>(supported_input, supported_first_b, supported_first_c,
                                                supported_intermediate, supported_first_alpha, supported_first_beta)
            .opaque});
    const PatternOperationId supported_second_op = PatternOperationId{
        supported_pattern
            .addOperation<OperationKind::kGemm>(supported_intermediate, supported_second_b, supported_second_c,
                                                supported_output, supported_second_alpha, supported_second_beta)
            .opaque};

    PatternBuilder user_pattern;
    const FTrainTensorId user_output       = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_beta  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_alpha = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_c     = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_b     = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_intermediate = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_beta   = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_alpha  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_c      = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_b      = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_input        = user_pattern.addOperand<OperandKind::kTensor>();
    const PatternOperationId user_second_op =
        PatternOperationId{user_pattern
                               .addOperation<OperationKind::kGemm>(user_intermediate, user_second_b, user_second_c,
                                                                   user_output, user_second_alpha, user_second_beta)
                               .opaque};
    const PatternOperationId user_first_op =
        PatternOperationId{user_pattern
                               .addOperation<OperationKind::kGemm>(user_input, user_first_b, user_first_c,
                                                                   user_intermediate, user_first_alpha, user_first_beta)
                               .opaque};

    const RoleMapping mapping = RoleMapping::fromMatchResult(
        std::move(*Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern())));

    EXPECT_EQ(mapping.getNumOperands(), 11);
    EXPECT_EQ(mapping.getNumOps(), 2);
    EXPECT_EQ(mapping.getSupportedOperandId(PatternOperandId{user_input.opaque}),
              PatternOperandId{supported_input.opaque});
    EXPECT_EQ(mapping.getSupportedOperandId(PatternOperandId{user_first_b.opaque}),
              PatternOperandId{supported_first_b.opaque});
    EXPECT_EQ(mapping.getSupportedOperandId(PatternOperandId{user_intermediate.opaque}),
              PatternOperandId{supported_intermediate.opaque});
    EXPECT_EQ(mapping.getSupportedOperandId(PatternOperandId{user_second_alpha.opaque}),
              PatternOperandId{supported_second_alpha.opaque});
    EXPECT_EQ(mapping.getSupportedOperandId(PatternOperandId{user_output.opaque}),
              PatternOperandId{supported_output.opaque});
    EXPECT_EQ(mapping.getSupportedOpId(user_first_op), supported_first_op);
    EXPECT_EQ(mapping.getSupportedOpId(user_second_op), supported_second_op);
    expectInvalidArgument([&] { static_cast<void>(mapping.getSupportedOperandId(PatternOperandId{11})); });
    expectInvalidArgument([&] { static_cast<void>(mapping.getSupportedOpId(PatternOperationId{2})); });
}

TEST(RoleMappingTest, RejectsDifferentCanonicalPatternKeys) {
    PatternBuilder gemm_pattern;
    const FTrainTensorId gemm_a     = gemm_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId gemm_b     = gemm_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId gemm_c     = gemm_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId gemm_alpha = gemm_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId gemm_beta  = gemm_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId gemm_d     = gemm_pattern.addOperand<OperandKind::kTensor>();
    PatternOperationId{
        gemm_pattern.addOperation<OperationKind::kGemm>(gemm_a, gemm_b, gemm_c, gemm_d, gemm_alpha, gemm_beta).opaque};

    PatternBuilder grouped_pattern;
    const FTrainGroupedTensorId grouped_a = grouped_pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId grouped_b = grouped_pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId grouped_c = grouped_pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainTensorId grouped_alpha    = grouped_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grouped_beta     = grouped_pattern.addOperand<OperandKind::kTensor>();
    const FTrainGroupedTensorId grouped_d = grouped_pattern.addOperand<OperandKind::kGroupedTensor>();
    PatternOperationId{grouped_pattern
                           .addOperation<OperationKind::kGroupedABCDGemm>(grouped_a, grouped_b, grouped_c, grouped_d,
                                                                          grouped_alpha, grouped_beta)
                           .opaque};

    EXPECT_NE(gemm_pattern.buildPattern().getKey(), grouped_pattern.buildPattern().getKey());
    EXPECT_FALSE(Matcher::match(gemm_pattern.buildPattern(), grouped_pattern.buildPattern()).has_value());
}

TEST(OpsTest, OwnsPatternKeyAndRoleMappingSnapshots) {
    PatternBuilder pattern;
    const FTrainTensorId operand = pattern.addOperand<OperandKind::kTensor>();
    const Ops ops(pattern.buildPattern(), pattern.buildPattern());

    EXPECT_EQ(ops.getPatternKey(), pattern.buildPattern().getKey());
    EXPECT_EQ(ops.getRoleMapping().getSupportedOperandId(PatternOperandId{operand.opaque}),
              PatternOperandId{operand.opaque});
}

TEST(ArgsTest, InitializesSupportedSlotsAndMapsUserSettersOnlyOnce) {
    PatternBuilder supported_pattern;
    const FTrainTensorId supported_input         = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_b       = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_c       = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_alpha   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_first_beta    = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_intermediate  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_b      = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_c      = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_alpha  = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_second_beta   = supported_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId supported_output        = supported_pattern.addOperand<OperandKind::kTensor>();
    const PatternOperationId supported_first_op  = (PatternOperationId{
        supported_pattern
            .addOperation<OperationKind::kGemm>(supported_input, supported_first_b, supported_first_c,
                                                supported_intermediate, supported_first_alpha, supported_first_beta)
            .opaque});
    const PatternOperationId supported_second_op = PatternOperationId{
        supported_pattern
            .addOperation<OperationKind::kGemm>(supported_intermediate, supported_second_b, supported_second_c,
                                                supported_output, supported_second_alpha, supported_second_beta)
            .opaque};

    PatternBuilder user_pattern;
    const FTrainTensorId user_output       = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_beta  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_alpha = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_c     = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_second_b     = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_intermediate = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_beta   = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_alpha  = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_c      = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_first_b      = user_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId user_input        = user_pattern.addOperand<OperandKind::kTensor>();
    const PatternOperationId user_second_op =
        PatternOperationId{user_pattern
                               .addOperation<OperationKind::kGemm>(user_intermediate, user_second_b, user_second_c,
                                                                   user_output, user_second_alpha, user_second_beta)
                               .opaque};
    const PatternOperationId user_first_op =
        PatternOperationId{user_pattern
                               .addOperation<OperationKind::kGemm>(user_input, user_first_b, user_first_c,
                                                                   user_intermediate, user_first_alpha, user_first_beta)
                               .opaque};

    const Ops ops = makeOps(user_pattern, supported_pattern);
    Args args     = ops.makeArgs();
    int memories[12]{};

    EXPECT_EQ(args.getPatternKey(), ops.getPatternKey());
    EXPECT_EQ(args.getNumOperands(), 11);
    EXPECT_EQ(args.getNumOps(), 2);
    EXPECT_FALSE(args.isComplete());

    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_input.opaque}, makeTensor(&memories[0]));
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_first_b.opaque}, makeTensor(&memories[1]));
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_first_c.opaque}, makeTensor(&memories[2]));
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_first_alpha.opaque}, makeTensor(&memories[3]));
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_first_beta.opaque}, makeTensor(&memories[4]));
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_intermediate.opaque}, makeTensor(&memories[5]));
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_second_b.opaque}, makeTensor(&memories[6]));
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_second_c.opaque}, makeTensor(&memories[7]));
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_second_alpha.opaque}, makeTensor(&memories[8]));
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_second_beta.opaque}, makeTensor(&memories[9]));
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_output.opaque}, makeTensor(&memories[10]));
    args.setOperation<OperationKind::kGemm>(user_first_op, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP32));
    EXPECT_FALSE(args.isComplete());
    args.setOperation<OperationKind::kGemm>(user_second_op, GemmAttributes(FTRAIN_NUMERIC_TYPE_BF16));

    EXPECT_TRUE(args.isComplete());
    EXPECT_EQ(std::get<Tensor>(*args.getOperand(PatternOperandId{supported_input.opaque})).getStorageView().getMemory(),
              &memories[0]);
    EXPECT_EQ(
        std::get<Tensor>(*args.getOperand(PatternOperandId{supported_output.opaque})).getStorageView().getMemory(),
        &memories[10]);
    EXPECT_EQ(std::get<GemmAttributes>(*args.getOpArgument(supported_first_op)).getComputeType(),
              FTRAIN_NUMERIC_TYPE_FP32);
    EXPECT_EQ(std::get<GemmAttributes>(*args.getOpArgument(supported_second_op)).getComputeType(),
              FTRAIN_NUMERIC_TYPE_BF16);

    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_output.opaque}, makeTensor(&memories[11], 8));
    args.setOperation<OperationKind::kGemm>(user_first_op, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP16));

    EXPECT_EQ(
        std::get<Tensor>(*args.getOperand(PatternOperandId{supported_output.opaque})).getStorageView().getMemory(),
        &memories[11]);
    EXPECT_EQ(std::get<Tensor>(*args.getOperand(PatternOperandId{supported_output.opaque})).getStorageView().getDims(),
              (std::vector<std::int64_t>{8}));
    EXPECT_EQ(std::get<GemmAttributes>(*args.getOpArgument(supported_first_op)).getComputeType(),
              FTRAIN_NUMERIC_TYPE_FP16);
    expectInvalidArgument([&] { static_cast<void>(args.getOperand(PatternOperandId{11})); });
    expectInvalidArgument([&] { static_cast<void>(args.getOpArgument(PatternOperationId{2})); });
}

TEST(ArgsTest, SupportsAllOperandFamiliesAndRequiresEverySlot) {
    PatternBuilder pattern;
    const FTrainTensorId tensor          = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorListId tensor_list = pattern.addOperand<OperandKind::kTensorList>();
    const FTrainGroupedTensorId grouped  = pattern.addOperand<OperandKind::kGroupedTensor>();
    const Ops ops                        = makeOps(pattern, pattern);
    Args args                            = ops.makeArgs();
    int memories[3]{};

    args.setOperand<OperandKind::kTensor>(PatternOperandId{tensor.opaque}, makeTensor(&memories[0]));
    args.setOperand<OperandKind::kTensorList>(PatternOperandId{tensor_list.opaque}, makeTensorList(&memories[1]));
    EXPECT_FALSE(args.isComplete());
    args.setOperand<OperandKind::kGroupedTensor>(PatternOperandId{grouped.opaque}, makeGroupedTensor(&memories[2]));

    EXPECT_TRUE(args.isComplete());
    EXPECT_TRUE(std::holds_alternative<Tensor>(*args.getOperand(PatternOperandId{tensor.opaque})));
    EXPECT_TRUE(std::holds_alternative<TensorList>(*args.getOperand(PatternOperandId{tensor_list.opaque})));
    EXPECT_TRUE(std::holds_alternative<GroupedTensor>(*args.getOperand(PatternOperandId{grouped.opaque})));
    expectInvalidArgument(
        [&] { args.setOperand<OperandKind::kTensor>(PatternOperandId{tensor_list.opaque}, makeTensor(&memories[0])); });
    expectInvalidArgument(
        [&] { args.setOperand<OperandKind::kTensor>(PatternOperandId{3}, makeTensor(&memories[0])); });
    EXPECT_TRUE(args.isComplete());
    EXPECT_EQ(
        std::get<TensorList>(*args.getOperand(PatternOperandId{tensor_list.opaque})).getStorageViews()[0].getMemory(),
        &memories[1]);
}

TEST(ArgsTest, SetOperationValidatesKindsAndStoresAllAttributeFamilies) {
    PatternBuilder pattern;
    const PatternOperationId gemm              = addStandaloneOp<OperationKind::kGemm>(pattern);
    const PatternOperationId grouped_abcd_gemm = addStandaloneOp<OperationKind::kGroupedABCDGemm>(pattern);
    const PatternOperationId grouped_bcd_gemm  = addStandaloneOp<OperationKind::kGroupedBCDGemm>(pattern);
    const PatternOperationId grouped_ab_gemm   = addStandaloneOp<OperationKind::kGroupedABGemm>(pattern);
    const Ops ops                              = makeOps(pattern, pattern);
    Args args                                  = ops.makeArgs();

    expectInvalidArgument(
        [&] { args.setOperation<OperationKind::kGemm>(grouped_abcd_gemm, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP32)); });
    EXPECT_FALSE(args.getOpArgument(grouped_abcd_gemm).has_value());
    expectInvalidArgument([&] {
        args.setOperation<OperationKind::kGroupedABCDGemm>(gemm, GroupedABCDGemmAttributes(FTRAIN_NUMERIC_TYPE_FP32));
    });
    EXPECT_FALSE(args.getOpArgument(gemm).has_value());

    args.setOperation<OperationKind::kGemm>(gemm, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP32));
    args.setOperation<OperationKind::kGroupedABCDGemm>(grouped_abcd_gemm,
                                                       GroupedABCDGemmAttributes(FTRAIN_NUMERIC_TYPE_FP32));
    args.setOperation<OperationKind::kGroupedBCDGemm>(grouped_bcd_gemm,
                                                      GroupedBCDGemmAttributes(FTRAIN_NUMERIC_TYPE_FP16));
    args.setOperation<OperationKind::kGroupedABGemm>(grouped_ab_gemm,
                                                     GroupedABGemmAttributes(FTRAIN_NUMERIC_TYPE_BF16));

    EXPECT_TRUE(std::holds_alternative<GemmAttributes>(*args.getOpArgument(gemm)));
    EXPECT_TRUE(std::holds_alternative<GroupedABCDGemmAttributes>(*args.getOpArgument(grouped_abcd_gemm)));
    EXPECT_TRUE(std::holds_alternative<GroupedBCDGemmAttributes>(*args.getOpArgument(grouped_bcd_gemm)));
    EXPECT_TRUE(std::holds_alternative<GroupedABGemmAttributes>(*args.getOpArgument(grouped_ab_gemm)));
    EXPECT_EQ(std::get<GemmAttributes>(*args.getOpArgument(gemm)).getComputeType(), FTRAIN_NUMERIC_TYPE_FP32);
    EXPECT_EQ(std::get<GroupedBCDGemmAttributes>(*args.getOpArgument(grouped_bcd_gemm)).getComputeType(),
              FTRAIN_NUMERIC_TYPE_FP16);
    EXPECT_EQ(std::get<GroupedABGemmAttributes>(*args.getOpArgument(grouped_ab_gemm)).getComputeType(),
              FTRAIN_NUMERIC_TYPE_BF16);
    EXPECT_FALSE(args.isComplete());
}

TEST(ArgsTest, EmptyOpsIsComplete) {
    PatternBuilder empty_pattern;
    const Ops empty_ops = makeOps(empty_pattern, empty_pattern);

    EXPECT_TRUE(empty_ops.makeArgs().isComplete());
}

}  // namespace
}  // namespace ftrain
