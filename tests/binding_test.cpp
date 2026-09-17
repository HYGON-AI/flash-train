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
#include "flash_train/matcher.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/binding.hpp"
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

template<typename Function>
void expectInternalError(Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "Runtime interface accepted an invalid internal state";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INTERNAL_ERROR);
    }
}

Ops makeOps(const PatternBuilder& user_pattern, const PatternBuilder& supported_pattern) {
    std::optional<MatchResult> match_result =
        Matcher::matchBySignature(user_pattern.buildPattern(), supported_pattern.buildPattern());
    if (!match_result.has_value()) {
        match_result = Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern());
    }
    RoleMapping role_mapping = RoleMapping::fromMatchResult(*match_result);
    return Ops(user_pattern.buildPattern().getKey(), std::move(role_mapping));
}

TensorStorage makeTensorStorage(void* memory, std::int64_t extent = 1) {
    const std::int64_t dims[]{extent};
    const FTrainStorageView description{
        memory, dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    StorageView storage_view(description);
    return TensorStorage(std::move(storage_view));
}

TensorListStorage makeTensorListStorage(void* memory) {
    const std::int64_t dims[]{1};
    const FTrainStorageView description{
        memory, dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    std::vector<StorageView> storage_views;
    storage_views.emplace_back(description);
    return TensorListStorage(std::move(storage_views));
}

GroupedTensorStorage makeGroupedTensorStorage(void* memory) {
    const std::int64_t dims[]{1};
    const FTrainStorageView description{
        memory, dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    StorageView data(description);
    std::optional<StorageView> offsets;
    std::vector<StorageView> dim_sizes;
    std::vector<StorageView> strides;
    return GroupedTensorStorage(0, std::move(data), std::move(offsets), std::move(dim_sizes), std::move(strides));
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
        return PatternOperationId{pattern.addOperation<Kind>({a, b, c, d, alpha, beta}).opaque};
    } else if constexpr (Kind == OperationKind::kGroupedABCDGemm) {
        const FTrainGroupedTensorId a = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId b = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId c = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId d = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainTensorId alpha    = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId beta     = pattern.addOperand<OperandKind::kTensor>();
        return PatternOperationId{pattern.addOperation<Kind>({a, b, c, d, alpha, beta}).opaque};
    } else if constexpr (Kind == OperationKind::kGroupedBCDGemm) {
        const FTrainTensorListId a    = pattern.addOperand<OperandKind::kTensorList>();
        const FTrainGroupedTensorId b = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId c = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId d = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainTensorId alpha    = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId beta     = pattern.addOperand<OperandKind::kTensor>();
        return PatternOperationId{pattern.addOperation<Kind>({a, b, c, d, alpha, beta}).opaque};
    } else if constexpr (Kind == OperationKind::kGroupedABGemm) {
        const FTrainGroupedTensorId a = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainGroupedTensorId b = pattern.addOperand<OperandKind::kGroupedTensor>();
        const FTrainTensorListId c    = pattern.addOperand<OperandKind::kTensorList>();
        const FTrainTensorListId d    = pattern.addOperand<OperandKind::kTensorList>();
        const FTrainTensorId alpha    = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId beta     = pattern.addOperand<OperandKind::kTensor>();
        return PatternOperationId{pattern.addOperation<Kind>({a, b, c, d, alpha, beta}).opaque};
    } else {
        static_assert(Kind != Kind, "addStandaloneOp covers every declared OperationKind");
    }
}

static_assert(std::is_same_v<OpsOperand, std::variant<Tensor, TensorList, GroupedTensor>>);
static_assert(std::is_same_v<Operand<OperandKind::kTensor>::Type, Tensor>);
static_assert(std::is_same_v<Operand<OperandKind::kTensorList>::Type, TensorList>);
static_assert(std::is_same_v<Operand<OperandKind::kGroupedTensor>::Type, GroupedTensor>);
static_assert(std::is_same_v<Operation<OperationKind::kGemm>::Type, GemmAttributes>);
static_assert(std::is_same_v<Operation<OperationKind::kGroupedABCDGemm>::Type, GroupedABCDGemmAttributes>);
static_assert(std::is_same_v<Operation<OperationKind::kGroupedBCDGemm>::Type, GroupedBCDGemmAttributes>);
static_assert(std::is_same_v<Operation<OperationKind::kGroupedABGemm>::Type, GroupedABGemmAttributes>);
static_assert(OpsOperandTraits<Tensor>::kKind == OperandKind::kTensor);
static_assert(OpsOperandTraits<TensorList>::kKind == OperandKind::kTensorList);
static_assert(OpsOperandTraits<GroupedTensor>::kKind == OperandKind::kGroupedTensor);
static_assert(noexcept(std::declval<const Ops&>().getPatternKey()));
static_assert(noexcept(std::declval<const Ops&>().getRoleMapping()));
static_assert(noexcept(std::declval<const Args&>().getPatternKey()));
static_assert(noexcept(std::declval<const Args&>().getNumOperands()));
static_assert(noexcept(std::declval<const Args&>().getNumOps()));
static_assert(noexcept(std::declval<const Args&>().isComplete()));

TEST(RoleMappingTest, ComposesExactBidirectionalRolesAcrossDifferentAdditionOrders) {
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
            .addOperation<OperationKind::kGemm>({supported_input, supported_first_b, supported_first_c,
                                                  supported_intermediate, supported_first_alpha, supported_first_beta})
            .opaque});
    const PatternOperationId supported_second_op = PatternOperationId{
        supported_pattern
            .addOperation<OperationKind::kGemm>({supported_intermediate, supported_second_b, supported_second_c,
                                                 supported_output, supported_second_alpha, supported_second_beta})
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
                               .addOperation<OperationKind::kGemm>({user_intermediate, user_second_b, user_second_c,
                                                                    user_output, user_second_alpha, user_second_beta})
                               .opaque};
    const PatternOperationId user_first_op = PatternOperationId{
        user_pattern
            .addOperation<OperationKind::kGemm>(
                {user_input, user_first_b, user_first_c, user_intermediate, user_first_alpha, user_first_beta})
            .opaque};

    const RoleMapping mapping =
        RoleMapping::fromMatchResult(*Matcher::match(user_pattern.buildPattern(), supported_pattern.buildPattern()));

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
    EXPECT_EQ(mapping.getUserOperandId(PatternOperandId{supported_input.opaque}), PatternOperandId{user_input.opaque});
    EXPECT_EQ(mapping.getUserOperandId(PatternOperandId{supported_output.opaque}),
              PatternOperandId{user_output.opaque});
    EXPECT_EQ(mapping.getSupportedOpId(user_first_op), supported_first_op);
    EXPECT_EQ(mapping.getSupportedOpId(user_second_op), supported_second_op);
    EXPECT_EQ(mapping.getUserOpId(supported_first_op), user_first_op);
    EXPECT_EQ(mapping.getUserOpId(supported_second_op), user_second_op);
    expectInvalidArgument([&] { static_cast<void>(mapping.getSupportedOperandId(PatternOperandId{11})); });
    expectInvalidArgument([&] { static_cast<void>(mapping.getUserOperandId(PatternOperandId{11})); });
    expectInvalidArgument([&] { static_cast<void>(mapping.getSupportedOpId(PatternOperationId{2})); });
    expectInvalidArgument([&] { static_cast<void>(mapping.getUserOpId(PatternOperationId{2})); });
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
        gemm_pattern.addOperation<OperationKind::kGemm>({gemm_a, gemm_b, gemm_c, gemm_d, gemm_alpha, gemm_beta})
            .opaque};

    PatternBuilder grouped_pattern;
    const FTrainGroupedTensorId grouped_a = grouped_pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId grouped_b = grouped_pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId grouped_c = grouped_pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainTensorId grouped_alpha    = grouped_pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId grouped_beta     = grouped_pattern.addOperand<OperandKind::kTensor>();
    const FTrainGroupedTensorId grouped_d = grouped_pattern.addOperand<OperandKind::kGroupedTensor>();
    PatternOperationId{grouped_pattern
                           .addOperation<OperationKind::kGroupedABCDGemm>(
                               {grouped_a, grouped_b, grouped_c, grouped_d, grouped_alpha, grouped_beta})
                           .opaque};

    EXPECT_NE(gemm_pattern.buildPattern().getKey(), grouped_pattern.buildPattern().getKey());
    EXPECT_FALSE(Matcher::match(gemm_pattern.buildPattern(), grouped_pattern.buildPattern()).has_value());
}

TEST(OpsTest, OwnsPatternKeyAndRoleMappingSnapshots) {
    PatternBuilder pattern;
    const FTrainTensorId operand = pattern.addOperand<OperandKind::kTensor>();
    RoleMapping mapping = RoleMapping::fromMatchResult(*Matcher::match(pattern.buildPattern(), pattern.buildPattern()));

    const Ops ops(pattern.buildPattern().getKey(), std::move(mapping));

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
            .addOperation<OperationKind::kGemm>({supported_input, supported_first_b, supported_first_c,
                                                  supported_intermediate, supported_first_alpha, supported_first_beta})
            .opaque});
    const PatternOperationId supported_second_op = PatternOperationId{
        supported_pattern
            .addOperation<OperationKind::kGemm>({supported_intermediate, supported_second_b, supported_second_c,
                                                 supported_output, supported_second_alpha, supported_second_beta})
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
                               .addOperation<OperationKind::kGemm>({user_intermediate, user_second_b, user_second_c,
                                                                    user_output, user_second_alpha, user_second_beta})
                               .opaque};
    const PatternOperationId user_first_op = PatternOperationId{
        user_pattern
            .addOperation<OperationKind::kGemm>(
                {user_input, user_first_b, user_first_c, user_intermediate, user_first_alpha, user_first_beta})
            .opaque};

    const Ops ops        = makeOps(user_pattern, supported_pattern);
    const Pattern schema = supported_pattern.buildPattern();
    Args args(ops, schema);
    int memories[12]{};

    EXPECT_EQ(args.getPatternKey(), ops.getPatternKey());
    EXPECT_EQ(args.getNumOperands(), 11);
    EXPECT_EQ(args.getNumOps(), 2);
    EXPECT_FALSE(args.isComplete());

    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_input.opaque}, Tensor{makeTensorStorage(&memories[0])});
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_first_b.opaque},
                                          Tensor{makeTensorStorage(&memories[1])});
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_first_c.opaque},
                                          Tensor{makeTensorStorage(&memories[2])});
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_first_alpha.opaque},
                                          Tensor{makeTensorStorage(&memories[3])});
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_first_beta.opaque},
                                          Tensor{makeTensorStorage(&memories[4])});
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_intermediate.opaque},
                                          Tensor{makeTensorStorage(&memories[5])});
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_second_b.opaque},
                                          Tensor{makeTensorStorage(&memories[6])});
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_second_c.opaque},
                                          Tensor{makeTensorStorage(&memories[7])});
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_second_alpha.opaque},
                                          Tensor{makeTensorStorage(&memories[8])});
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_second_beta.opaque},
                                          Tensor{makeTensorStorage(&memories[9])});
    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_output.opaque},
                                          Tensor{makeTensorStorage(&memories[10])});
    args.setOperation<OperationKind::kGemm>(user_first_op, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP32));
    EXPECT_FALSE(args.isComplete());
    args.setOperation<OperationKind::kGemm>(user_second_op, GemmAttributes(FTRAIN_NUMERIC_TYPE_BF16));

    EXPECT_TRUE(args.isComplete());
    EXPECT_EQ(std::get<Tensor>(args.getOperand(PatternOperandId{supported_input.opaque}))
                  .getStorage()
                  .getStorageView()
                  .getMemory(),
              &memories[0]);
    EXPECT_EQ(std::get<Tensor>(args.getOperand(PatternOperandId{supported_output.opaque}))
                  .getStorage()
                  .getStorageView()
                  .getMemory(),
              &memories[10]);
    EXPECT_EQ(args.getOpArgument(supported_first_op).getKind(), OperationKind::kGemm);
    EXPECT_EQ(std::get<GemmAttributes>(args.getOpArgument(supported_first_op).getAttributes()).getComputeType(),
              FTRAIN_NUMERIC_TYPE_FP32);
    EXPECT_EQ(std::get<GemmAttributes>(args.getOpArgument(supported_second_op).getAttributes()).getComputeType(),
              FTRAIN_NUMERIC_TYPE_BF16);

    args.setOperand<OperandKind::kTensor>(PatternOperandId{user_output.opaque},
                                          Tensor{makeTensorStorage(&memories[11], 8)});
    args.setOperation<OperationKind::kGemm>(user_first_op, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP16));

    EXPECT_EQ(std::get<Tensor>(args.getOperand(PatternOperandId{supported_output.opaque}))
                  .getStorage()
                  .getStorageView()
                  .getMemory(),
              &memories[11]);
    EXPECT_EQ(std::get<Tensor>(args.getOperand(PatternOperandId{supported_output.opaque}))
                  .getStorage()
                  .getStorageView()
                  .getDims(),
              (std::vector<std::int64_t>{8}));
    EXPECT_EQ(std::get<GemmAttributes>(args.getOpArgument(supported_first_op).getAttributes()).getComputeType(),
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
    const Pattern schema                 = pattern.buildPattern();
    Args args(ops, schema);
    int memories[3]{};

    args.setOperand<OperandKind::kTensor>(PatternOperandId{tensor.opaque}, Tensor{makeTensorStorage(&memories[0])});
    args.setOperand<OperandKind::kTensorList>(PatternOperandId{tensor_list.opaque},
                                              TensorList{makeTensorListStorage(&memories[1])});
    EXPECT_FALSE(args.isComplete());
    args.setOperand<OperandKind::kGroupedTensor>(PatternOperandId{grouped.opaque},
                                                 GroupedTensor{makeGroupedTensorStorage(&memories[2])});

    EXPECT_TRUE(args.isComplete());
    EXPECT_TRUE(std::holds_alternative<Tensor>(args.getOperand(PatternOperandId{tensor.opaque})));
    EXPECT_TRUE(std::holds_alternative<TensorList>(args.getOperand(PatternOperandId{tensor_list.opaque})));
    EXPECT_TRUE(std::holds_alternative<GroupedTensor>(args.getOperand(PatternOperandId{grouped.opaque})));
    expectInvalidArgument([&] {
        args.setOperand<OperandKind::kTensor>(PatternOperandId{tensor_list.opaque},
                                              Tensor{makeTensorStorage(&memories[0])});
    });
    expectInvalidArgument(
        [&] { args.setOperand<OperandKind::kTensor>(PatternOperandId{3}, Tensor{makeTensorStorage(&memories[0])}); });
    expectInvalidArgument([&] { args.setOperand<OperandKind::kTensor>(PatternOperandId{tensor.opaque}, Tensor{}); });
    EXPECT_TRUE(args.isComplete());
    EXPECT_EQ(std::get<TensorList>(args.getOperand(PatternOperandId{tensor_list.opaque}))
                  .getStorage()
                  .getStorageViews()[0]
                  .getMemory(),
              &memories[1]);
}

TEST(ArgsTest, SetOperationValidatesKindsAndStoresAllAttributeFamilies) {
    PatternBuilder pattern;
    const PatternOperationId gemm              = addStandaloneOp<OperationKind::kGemm>(pattern);
    const PatternOperationId grouped_abcd_gemm = addStandaloneOp<OperationKind::kGroupedABCDGemm>(pattern);
    const PatternOperationId grouped_bcd_gemm  = addStandaloneOp<OperationKind::kGroupedBCDGemm>(pattern);
    const PatternOperationId grouped_ab_gemm   = addStandaloneOp<OperationKind::kGroupedABGemm>(pattern);
    const Ops ops                              = makeOps(pattern, pattern);
    const Pattern schema                       = pattern.buildPattern();
    Args args(ops, schema);

    expectInvalidArgument(
        [&] { args.setOperation<OperationKind::kGemm>(grouped_abcd_gemm, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP32)); });
    EXPECT_FALSE(args.getOpArgument(grouped_abcd_gemm).isSet());
    expectInvalidArgument([&] {
        args.setOperation<OperationKind::kGroupedABCDGemm>(gemm, GroupedABCDGemmAttributes(FTRAIN_NUMERIC_TYPE_FP32));
    });
    EXPECT_FALSE(args.getOpArgument(gemm).isSet());

    args.setOperation<OperationKind::kGemm>(gemm, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP32));
    args.setOperation<OperationKind::kGroupedABCDGemm>(grouped_abcd_gemm,
                                                       GroupedABCDGemmAttributes(FTRAIN_NUMERIC_TYPE_FP32));
    args.setOperation<OperationKind::kGroupedBCDGemm>(grouped_bcd_gemm,
                                                      GroupedBCDGemmAttributes(FTRAIN_NUMERIC_TYPE_FP16));
    args.setOperation<OperationKind::kGroupedABGemm>(grouped_ab_gemm,
                                                     GroupedABGemmAttributes(FTRAIN_NUMERIC_TYPE_BF16));

    EXPECT_TRUE(std::holds_alternative<GemmAttributes>(args.getOpArgument(gemm).getAttributes()));
    EXPECT_TRUE(
        std::holds_alternative<GroupedABCDGemmAttributes>(args.getOpArgument(grouped_abcd_gemm).getAttributes()));
    EXPECT_TRUE(std::holds_alternative<GroupedBCDGemmAttributes>(args.getOpArgument(grouped_bcd_gemm).getAttributes()));
    EXPECT_TRUE(std::holds_alternative<GroupedABGemmAttributes>(args.getOpArgument(grouped_ab_gemm).getAttributes()));
    EXPECT_EQ(std::get<GemmAttributes>(args.getOpArgument(gemm).getAttributes()).getComputeType(),
              FTRAIN_NUMERIC_TYPE_FP32);
    EXPECT_EQ(std::get<GroupedBCDGemmAttributes>(args.getOpArgument(grouped_bcd_gemm).getAttributes()).getComputeType(),
              FTRAIN_NUMERIC_TYPE_FP16);
    EXPECT_EQ(std::get<GroupedABGemmAttributes>(args.getOpArgument(grouped_ab_gemm).getAttributes()).getComputeType(),
              FTRAIN_NUMERIC_TYPE_BF16);
    EXPECT_FALSE(args.isComplete());
}

TEST(ArgsTest, EmptyOpsIsCompleteAndMismatchedSchemaIsRejected) {
    PatternBuilder empty_pattern;
    const Ops empty_ops        = makeOps(empty_pattern, empty_pattern);
    const Pattern empty_schema = empty_pattern.buildPattern();
    const Args empty_args(empty_ops, empty_schema);

    EXPECT_TRUE(empty_args.isComplete());

    PatternBuilder nonempty_pattern;
    static_cast<void>(nonempty_pattern.addOperand<OperandKind::kTensor>());
    const Ops nonempty_ops = makeOps(nonempty_pattern, nonempty_pattern);
    expectInvalidArgument([&] { static_cast<void>(Args(nonempty_ops, empty_schema)); });
}

TEST(OpArgumentTest, RejectsAttributesBeforeRoleIsSet) {
    const OpArgument argument(OperationKind::kGemm);

    EXPECT_EQ(argument.getKind(), OperationKind::kGemm);
    EXPECT_FALSE(argument.isSet());
    expectInvalidArgument([&] { static_cast<void>(argument.getAttributes()); });
}

}  // namespace
}  // namespace ftrain
