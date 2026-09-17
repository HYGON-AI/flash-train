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
#include "flash_train/op_definitions.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/runtime.hpp"
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

Ops makeOps(const Pattern& user_pattern, const Pattern& supported_pattern) {
    const PatternCanonicalization user_canonicalization      = Matcher::canonicalize(user_pattern);
    const PatternCanonicalization supported_canonicalization = Matcher::canonicalize(supported_pattern);
    RoleMapping role_mapping = RoleMapping::compose(user_canonicalization, supported_canonicalization);
    return Ops(user_canonicalization.getKey(), std::move(role_mapping));
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
OperationId addStandaloneOp(Pattern& pattern) {
    if constexpr (Kind == OperationKind::kGemm) {
        const FTrainTensorId a     = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId b     = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId c     = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId d     = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId alpha = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId beta  = Operand<OperandKind::kTensor>::addToPattern(pattern);
        return OperationId{Operation<Kind>::addToPattern(pattern, a, b, c, d, alpha, beta).opaque};
    } else if constexpr (Kind == OperationKind::kGroupedABCDGemm) {
        const FTrainGroupedTensorId a = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
        const FTrainGroupedTensorId b = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
        const FTrainGroupedTensorId c = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
        const FTrainGroupedTensorId d = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
        const FTrainTensorId alpha    = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId beta     = Operand<OperandKind::kTensor>::addToPattern(pattern);
        return OperationId{Operation<Kind>::addToPattern(pattern, a, b, c, d, alpha, beta).opaque};
    } else if constexpr (Kind == OperationKind::kGroupedBCDGemm) {
        const FTrainTensorListId a    = Operand<OperandKind::kTensorList>::addToPattern(pattern);
        const FTrainGroupedTensorId b = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
        const FTrainGroupedTensorId c = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
        const FTrainGroupedTensorId d = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
        const FTrainTensorId alpha    = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId beta     = Operand<OperandKind::kTensor>::addToPattern(pattern);
        return OperationId{Operation<Kind>::addToPattern(pattern, a, b, c, d, alpha, beta).opaque};
    } else if constexpr (Kind == OperationKind::kGroupedABGemm) {
        const FTrainGroupedTensorId a = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
        const FTrainGroupedTensorId b = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
        const FTrainTensorListId c    = Operand<OperandKind::kTensorList>::addToPattern(pattern);
        const FTrainTensorListId d    = Operand<OperandKind::kTensorList>::addToPattern(pattern);
        const FTrainTensorId alpha    = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId beta     = Operand<OperandKind::kTensor>::addToPattern(pattern);
        return OperationId{Operation<Kind>::addToPattern(pattern, a, b, c, d, alpha, beta).opaque};
    } else {
        static_assert(Kind != Kind, "addStandaloneOp covers every declared OperationKind");
    }
}

static_assert(std::is_same_v<OpsOperand, std::variant<Tensor, TensorList, GroupedTensor>>);
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
    Pattern supported_pattern;
    const FTrainTensorId supported_input        = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_b      = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_c      = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_alpha  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_beta   = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_intermediate = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_b     = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_c     = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_alpha = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_beta  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_output       = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const OperationId supported_first_op =
        (OperationId{Operation<OperationKind::kGemm>::addToPattern(
                         supported_pattern, supported_input, supported_first_b, supported_first_c,
                         supported_intermediate, supported_first_alpha, supported_first_beta)
                         .opaque});
    const OperationId supported_second_op =
        OperationId{Operation<OperationKind::kGemm>::addToPattern(
                        supported_pattern, supported_intermediate, supported_second_b, supported_second_c,
                        supported_output, supported_second_alpha, supported_second_beta)
                        .opaque};

    Pattern user_pattern;
    const FTrainTensorId user_output       = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_beta  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_alpha = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_c     = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_b     = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_intermediate = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_beta   = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_alpha  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_c      = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_b      = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_input        = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const OperationId user_second_op       = OperationId{
        Operation<OperationKind::kGemm>::addToPattern(user_pattern, user_intermediate, user_second_b, user_second_c,
                                                      user_output, user_second_alpha, user_second_beta)
            .opaque};
    const OperationId user_first_op =
        OperationId{Operation<OperationKind::kGemm>::addToPattern(user_pattern, user_input, user_first_b, user_first_c,
                                                                  user_intermediate, user_first_alpha, user_first_beta)
                        .opaque};

    const PatternCanonicalization user_canonicalization      = Matcher::canonicalize(user_pattern);
    const PatternCanonicalization supported_canonicalization = Matcher::canonicalize(supported_pattern);
    const RoleMapping mapping = RoleMapping::compose(user_canonicalization, supported_canonicalization);

    EXPECT_EQ(mapping.getNumOperands(), 11);
    EXPECT_EQ(mapping.getNumOps(), 2);
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{user_input.opaque}), OperandId{supported_input.opaque});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{user_first_b.opaque}), OperandId{supported_first_b.opaque});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{user_intermediate.opaque}),
              OperandId{supported_intermediate.opaque});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{user_second_alpha.opaque}),
              OperandId{supported_second_alpha.opaque});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{user_output.opaque}), OperandId{supported_output.opaque});
    EXPECT_EQ(mapping.getUserOperandId(OperandId{supported_input.opaque}), OperandId{user_input.opaque});
    EXPECT_EQ(mapping.getUserOperandId(OperandId{supported_output.opaque}), OperandId{user_output.opaque});
    EXPECT_EQ(mapping.getSupportedOpId(user_first_op), supported_first_op);
    EXPECT_EQ(mapping.getSupportedOpId(user_second_op), supported_second_op);
    EXPECT_EQ(mapping.getUserOpId(supported_first_op), user_first_op);
    EXPECT_EQ(mapping.getUserOpId(supported_second_op), user_second_op);
    expectInvalidArgument([&] { static_cast<void>(mapping.getSupportedOperandId(OperandId{11})); });
    expectInvalidArgument([&] { static_cast<void>(mapping.getUserOperandId(OperandId{11})); });
    expectInvalidArgument([&] { static_cast<void>(mapping.getSupportedOpId(OperationId{2})); });
    expectInvalidArgument([&] { static_cast<void>(mapping.getUserOpId(OperationId{2})); });
}

TEST(RoleMappingTest, RejectsDifferentCanonicalPatternKeys) {
    Pattern gemm_pattern;
    const FTrainTensorId gemm_a     = Operand<OperandKind::kTensor>::addToPattern(gemm_pattern);
    const FTrainTensorId gemm_b     = Operand<OperandKind::kTensor>::addToPattern(gemm_pattern);
    const FTrainTensorId gemm_c     = Operand<OperandKind::kTensor>::addToPattern(gemm_pattern);
    const FTrainTensorId gemm_alpha = Operand<OperandKind::kTensor>::addToPattern(gemm_pattern);
    const FTrainTensorId gemm_beta  = Operand<OperandKind::kTensor>::addToPattern(gemm_pattern);
    const FTrainTensorId gemm_d     = Operand<OperandKind::kTensor>::addToPattern(gemm_pattern);
    OperationId{Operation<OperationKind::kGemm>::addToPattern(gemm_pattern, gemm_a, gemm_b, gemm_c, gemm_d, gemm_alpha,
                                                              gemm_beta)
                    .opaque};

    Pattern grouped_pattern;
    const FTrainGroupedTensorId grouped_a = Operand<OperandKind::kGroupedTensor>::addToPattern(grouped_pattern);
    const FTrainGroupedTensorId grouped_b = Operand<OperandKind::kGroupedTensor>::addToPattern(grouped_pattern);
    const FTrainGroupedTensorId grouped_c = Operand<OperandKind::kGroupedTensor>::addToPattern(grouped_pattern);
    const FTrainTensorId grouped_alpha    = Operand<OperandKind::kTensor>::addToPattern(grouped_pattern);
    const FTrainTensorId grouped_beta     = Operand<OperandKind::kTensor>::addToPattern(grouped_pattern);
    const FTrainGroupedTensorId grouped_d = Operand<OperandKind::kGroupedTensor>::addToPattern(grouped_pattern);
    OperationId{Operation<OperationKind::kGroupedABCDGemm>::addToPattern(
                    grouped_pattern, grouped_a, grouped_b, grouped_c, grouped_d, grouped_alpha, grouped_beta)
                    .opaque};

    const PatternCanonicalization gemm_canonicalization    = Matcher::canonicalize(gemm_pattern);
    const PatternCanonicalization grouped_canonicalization = Matcher::canonicalize(grouped_pattern);

    expectInvalidArgument(
        [&] { static_cast<void>(RoleMapping::compose(gemm_canonicalization, grouped_canonicalization)); });
}

TEST(SupportedSchemaTest, OwnsKindsInSupportedRoleOrderAndChecksBounds) {
    Pattern pattern;
    const FTrainTensorId tensor          = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorListId tensor_list = Operand<OperandKind::kTensorList>::addToPattern(pattern);
    const FTrainGroupedTensorId grouped  = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
    const FTrainTensorId a               = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId b               = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId c               = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId alpha           = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId beta            = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId d               = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const OperationId gemm =
        (OperationId{Operation<OperationKind::kGemm>::addToPattern(pattern, a, b, c, d, alpha, beta).opaque});
    static_cast<void>(tensor_list);
    static_cast<void>(grouped);

    const SupportedSchema schema(pattern);

    EXPECT_EQ(schema.getNumOperands(), 9);
    EXPECT_EQ(schema.getNumOps(), 1);
    EXPECT_EQ(schema.getOperandKind(OperandId{0}), OperandKind::kTensor);
    EXPECT_EQ(schema.getOperandKind(OperandId{1}), OperandKind::kTensorList);
    EXPECT_EQ(schema.getOperandKind(OperandId{2}), OperandKind::kGroupedTensor);
    EXPECT_EQ(schema.getOperationKind(gemm), OperationKind::kGemm);
    expectInvalidArgument([&] { static_cast<void>(schema.getOperandKind(OperandId{9})); });
    expectInvalidArgument([&] { static_cast<void>(schema.getOperationKind(OperationId{1})); });
}

TEST(OpsTest, OwnsPatternKeyAndRoleMappingSnapshots) {
    Pattern pattern;
    const FTrainTensorId operand                   = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const PatternCanonicalization canonicalization = Matcher::canonicalize(pattern);
    RoleMapping mapping                            = RoleMapping::compose(canonicalization, canonicalization);

    const Ops ops(canonicalization.getKey(), std::move(mapping));

    EXPECT_EQ(ops.getPatternKey(), canonicalization.getKey());
    EXPECT_EQ(ops.getRoleMapping().getSupportedOperandId(OperandId{operand.opaque}), OperandId{operand.opaque});
}

TEST(ArgsTest, InitializesSupportedSlotsAndMapsUserSettersOnlyOnce) {
    Pattern supported_pattern;
    const FTrainTensorId supported_input        = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_b      = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_c      = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_alpha  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_first_beta   = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_intermediate = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_b     = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_c     = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_alpha = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_second_beta  = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const FTrainTensorId supported_output       = Operand<OperandKind::kTensor>::addToPattern(supported_pattern);
    const OperationId supported_first_op =
        (OperationId{Operation<OperationKind::kGemm>::addToPattern(
                         supported_pattern, supported_input, supported_first_b, supported_first_c,
                         supported_intermediate, supported_first_alpha, supported_first_beta)
                         .opaque});
    const OperationId supported_second_op =
        OperationId{Operation<OperationKind::kGemm>::addToPattern(
                        supported_pattern, supported_intermediate, supported_second_b, supported_second_c,
                        supported_output, supported_second_alpha, supported_second_beta)
                        .opaque};

    Pattern user_pattern;
    const FTrainTensorId user_output       = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_beta  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_alpha = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_c     = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_second_b     = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_intermediate = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_beta   = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_alpha  = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_c      = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_first_b      = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const FTrainTensorId user_input        = Operand<OperandKind::kTensor>::addToPattern(user_pattern);
    const OperationId user_second_op       = OperationId{
        Operation<OperationKind::kGemm>::addToPattern(user_pattern, user_intermediate, user_second_b, user_second_c,
                                                      user_output, user_second_alpha, user_second_beta)
            .opaque};
    const OperationId user_first_op =
        OperationId{Operation<OperationKind::kGemm>::addToPattern(user_pattern, user_input, user_first_b, user_first_c,
                                                                  user_intermediate, user_first_alpha, user_first_beta)
                        .opaque};

    const Ops ops = makeOps(user_pattern, supported_pattern);
    const SupportedSchema schema(supported_pattern);
    Args args(ops, schema);
    int memories[12]{};

    EXPECT_EQ(args.getPatternKey(), ops.getPatternKey());
    EXPECT_EQ(args.getNumOperands(), 11);
    EXPECT_EQ(args.getNumOps(), 2);
    EXPECT_FALSE(args.isComplete());

    args.setTensor(OperandId{user_input.opaque}, makeTensorStorage(&memories[0]));
    args.setTensor(OperandId{user_first_b.opaque}, makeTensorStorage(&memories[1]));
    args.setTensor(OperandId{user_first_c.opaque}, makeTensorStorage(&memories[2]));
    args.setTensor(OperandId{user_first_alpha.opaque}, makeTensorStorage(&memories[3]));
    args.setTensor(OperandId{user_first_beta.opaque}, makeTensorStorage(&memories[4]));
    args.setTensor(OperandId{user_intermediate.opaque}, makeTensorStorage(&memories[5]));
    args.setTensor(OperandId{user_second_b.opaque}, makeTensorStorage(&memories[6]));
    args.setTensor(OperandId{user_second_c.opaque}, makeTensorStorage(&memories[7]));
    args.setTensor(OperandId{user_second_alpha.opaque}, makeTensorStorage(&memories[8]));
    args.setTensor(OperandId{user_second_beta.opaque}, makeTensorStorage(&memories[9]));
    args.setTensor(OperandId{user_output.opaque}, makeTensorStorage(&memories[10]));
    args.setGemm(user_first_op, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP32));
    EXPECT_FALSE(args.isComplete());
    args.setGemm(user_second_op, GemmAttributes(FTRAIN_NUMERIC_TYPE_BF16));

    EXPECT_TRUE(args.isComplete());
    EXPECT_EQ(
        std::get<Tensor>(args.getOperand(OperandId{supported_input.opaque})).getStorage().getStorageView().getMemory(),
        &memories[0]);
    EXPECT_EQ(
        std::get<Tensor>(args.getOperand(OperandId{supported_output.opaque})).getStorage().getStorageView().getMemory(),
        &memories[10]);
    EXPECT_EQ(args.getOpArgument(supported_first_op).getKind(), OperationKind::kGemm);
    EXPECT_EQ(std::get<GemmAttributes>(args.getOpArgument(supported_first_op).getAttributes()).getComputeType(),
              FTRAIN_NUMERIC_TYPE_FP32);
    EXPECT_EQ(std::get<GemmAttributes>(args.getOpArgument(supported_second_op).getAttributes()).getComputeType(),
              FTRAIN_NUMERIC_TYPE_BF16);

    args.setTensor(OperandId{user_output.opaque}, makeTensorStorage(&memories[11], 8));
    args.setGemm(user_first_op, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP16));

    EXPECT_EQ(
        std::get<Tensor>(args.getOperand(OperandId{supported_output.opaque})).getStorage().getStorageView().getMemory(),
        &memories[11]);
    EXPECT_EQ(
        std::get<Tensor>(args.getOperand(OperandId{supported_output.opaque})).getStorage().getStorageView().getDims(),
        (std::vector<std::int64_t>{8}));
    EXPECT_EQ(std::get<GemmAttributes>(args.getOpArgument(supported_first_op).getAttributes()).getComputeType(),
              FTRAIN_NUMERIC_TYPE_FP16);
    expectInvalidArgument([&] { static_cast<void>(args.getOperand(OperandId{11})); });
    expectInvalidArgument([&] { static_cast<void>(args.getOpArgument(OperationId{2})); });
}

TEST(ArgsTest, SupportsAllOperandFamiliesAndRequiresEverySlot) {
    Pattern pattern;
    const FTrainTensorId tensor          = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorListId tensor_list = Operand<OperandKind::kTensorList>::addToPattern(pattern);
    const FTrainGroupedTensorId grouped  = Operand<OperandKind::kGroupedTensor>::addToPattern(pattern);
    const Ops ops                        = makeOps(pattern, pattern);
    const SupportedSchema schema(pattern);
    Args args(ops, schema);
    int memories[3]{};

    args.setTensor(OperandId{tensor.opaque}, makeTensorStorage(&memories[0]));
    args.setTensorList(OperandId{tensor_list.opaque}, makeTensorListStorage(&memories[1]));
    EXPECT_FALSE(args.isComplete());
    args.setGroupedTensor(OperandId{grouped.opaque}, makeGroupedTensorStorage(&memories[2]));

    EXPECT_TRUE(args.isComplete());
    EXPECT_TRUE(std::holds_alternative<Tensor>(args.getOperand(OperandId{tensor.opaque})));
    EXPECT_TRUE(std::holds_alternative<TensorList>(args.getOperand(OperandId{tensor_list.opaque})));
    EXPECT_TRUE(std::holds_alternative<GroupedTensor>(args.getOperand(OperandId{grouped.opaque})));
    expectInvalidArgument([&] { args.setTensor(OperandId{tensor_list.opaque}, makeTensorStorage(&memories[0])); });
    expectInvalidArgument([&] { args.setTensor(OperandId{3}, makeTensorStorage(&memories[0])); });
    EXPECT_TRUE(args.isComplete());
    EXPECT_EQ(std::get<TensorList>(args.getOperand(OperandId{tensor_list.opaque}))
                  .getStorage()
                  .getStorageViews()[0]
                  .getMemory(),
              &memories[1]);
}

TEST(ArgsTest, TypedOpSettersValidateKindsAndStoreAllAttributeFamilies) {
    Pattern pattern;
    const OperationId gemm              = addStandaloneOp<OperationKind::kGemm>(pattern);
    const OperationId grouped_abcd_gemm = addStandaloneOp<OperationKind::kGroupedABCDGemm>(pattern);
    const OperationId grouped_bcd_gemm  = addStandaloneOp<OperationKind::kGroupedBCDGemm>(pattern);
    const OperationId grouped_ab_gemm   = addStandaloneOp<OperationKind::kGroupedABGemm>(pattern);
    const Ops ops                       = makeOps(pattern, pattern);
    const SupportedSchema schema(pattern);
    Args args(ops, schema);

    expectInvalidArgument([&] { args.setGemm(grouped_abcd_gemm, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP32)); });
    EXPECT_FALSE(args.getOpArgument(grouped_abcd_gemm).isSet());
    expectInvalidArgument([&] { args.setGroupedABCDGemm(gemm, GroupedABCDGemmAttributes(FTRAIN_NUMERIC_TYPE_FP32)); });
    EXPECT_FALSE(args.getOpArgument(gemm).isSet());

    args.setGemm(gemm, GemmAttributes(FTRAIN_NUMERIC_TYPE_FP32));
    args.setGroupedABCDGemm(grouped_abcd_gemm, GroupedABCDGemmAttributes(FTRAIN_NUMERIC_TYPE_FP32));
    args.setGroupedBCDGemm(grouped_bcd_gemm, GroupedBCDGemmAttributes(FTRAIN_NUMERIC_TYPE_FP16));
    args.setGroupedABGemm(grouped_ab_gemm, GroupedABGemmAttributes(FTRAIN_NUMERIC_TYPE_BF16));

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
    Pattern empty_pattern;
    const Ops empty_ops = makeOps(empty_pattern, empty_pattern);
    const SupportedSchema empty_schema(empty_pattern);
    const Args empty_args(empty_ops, empty_schema);

    EXPECT_TRUE(empty_args.isComplete());

    Pattern nonempty_pattern;
    static_cast<void>(Operand<OperandKind::kTensor>::addToPattern(nonempty_pattern));
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
