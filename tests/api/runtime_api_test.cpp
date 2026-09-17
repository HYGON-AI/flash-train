#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/api_handles.hpp"
#include "flash_train/op_definitions.hpp"
#include "flash_train/op_schema.hpp"
#include "flash_train/ops_engine.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/primitive.hpp"
#include "flash_train/runtime.hpp"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {
namespace {

struct MockState {
    std::atomic<int> applicable_calls{0};
    std::atomic<int> create_calls{0};
    std::atomic<int> finder_calls{0};
    int execute_calls{0};
    int executed_marker{0};
    void* workspace{nullptr};
    std::uint64_t workspace_bytes{0};
    FTrainStream stream{nullptr};
};

struct MockProblem {
    StorageView first;
};

class MockPrimitive final : public Primitive<MockProblem> {
  public:
    explicit MockPrimitive(std::shared_ptr<MockState> state) : state_(std::move(state)) {}

    const char* getName() const noexcept override { return "MockPrimitive"; }

    std::unique_ptr<PrimitiveBase> clone() const override { return std::make_unique<MockPrimitive>(*this); }

    Result isApplicable(const MockProblem&, const SelectionContext& context) const override {
        ++state_->applicable_calls;
        if (context.getMaxWorkspaceBytes() < 32) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "workspace limit is below 32 bytes");
        }
        return Result{};
    }

    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return 32; }

    void configure(const MockProblem& problem) override {
        ++state_->create_calls;
        const auto& dims = problem.first.getDims();
        marker_          = dims.empty() ? 0 : static_cast<int>(dims[0]);
    }

  private:
    void executeImpl(void* workspace, std::uint64_t workspace_bytes, FTrainStream stream) override {
        ++state_->execute_calls;
        state_->executed_marker = marker_;
        state_->workspace       = workspace;
        state_->workspace_bytes = workspace_bytes;
        state_->stream          = stream;
    }

    std::shared_ptr<MockState> state_;
    int marker_ = 0;
};

class MockFinder final : public Finder {
  public:
    MockFinder(std::shared_ptr<const PrimitiveBase> record, std::shared_ptr<MockState> state)
        : record_(std::move(record)), state_(std::move(state)) {}

    bool isEnabled(const Args&, const SelectionContext&) const override { return true; }

    PrimitiveList findCandidates(const PrimitiveList&, const Args&, const SelectionContext&) const override {
        ++state_->finder_calls;
        return PrimitiveList{record_};
    }

    void sortCandidates(const Args&, const SelectionContext&, PrimitiveList&) const override {}

  private:
    std::shared_ptr<const PrimitiveBase> record_;
    std::shared_ptr<MockState> state_;
};

class MockOpsEngine final : public OpsEngine<MockProblem> {
  public:
    MockOpsEngine(const Pattern& pattern, PrimitiveList records, std::vector<std::shared_ptr<const Finder>> finders)
        : OpsEngine(pattern, std::move(records), std::move(finders)) {}

  private:
    MockProblem makeProblem(const Args& args) const override {
        return MockProblem{std::get<Tensor>(args.getOperand(OperandId{0})).getStorage().getStorageView()};
    }

    // The mock's single record has the same selection and applicability for all
    // complete Args. Device and workspace limit are encoded by OpsEngine.
    std::vector<std::uint64_t> makeSelectionTokens(const MockProblem&, const SelectionContext&) const override {
        return {};
    }
};

struct SimpleRegistration {
    SimpleRegistration() {
        Pattern pattern;
        // Supported operand order: first_a, first_b, first_c, first_alpha,
        // first_beta, first_d, second_b, second_c, second_alpha, second_beta,
        // second_d. The first Gemm produces first_d; the second Gemm consumes
        // first_d and produces second_d.
        const FTrainTensorId first_a      = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId first_b      = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId first_c      = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId first_alpha  = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId first_beta   = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId first_d      = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId second_b     = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId second_c     = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId second_alpha = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId second_beta  = Operand<OperandKind::kTensor>::addToPattern(pattern);
        const FTrainTensorId second_d     = Operand<OperandKind::kTensor>::addToPattern(pattern);
        static_cast<void>(Operation<OperationKind::kGemm>::addToPattern(pattern, first_a, first_b, first_c, first_d,
                                                                        first_alpha, first_beta));
        static_cast<void>(Operation<OperationKind::kGemm>::addToPattern(pattern, first_d, second_b, second_c, second_d,
                                                                        second_alpha, second_beta));

        state  = std::make_shared<MockState>();
        record = std::make_shared<MockPrimitive>(state);
        finder = std::make_shared<MockFinder>(record, state);
        engine = std::make_shared<MockOpsEngine>(pattern, PrimitiveList{record},
                                                 std::vector<std::shared_ptr<const Finder>>{finder});
        getGlobalHandle().registerOpsEngine(engine);
    }

    std::shared_ptr<MockState> state;
    std::shared_ptr<MockPrimitive> record;
    std::shared_ptr<MockFinder> finder;
    std::shared_ptr<MockOpsEngine> engine;
};

SimpleRegistration& getSimpleRegistration() {
    static SimpleRegistration registration;
    return registration;
}

FTrainTensorId addTensor(FTrainPattern pattern) {
    FTrainTensorId id{};
    EXPECT_EQ(ftrainPatternAddTensor(pattern, &id), FTRAIN_STATUS_SUCCESS);
    return id;
}

FTrainTensorListId addTensorList(FTrainPattern pattern) {
    FTrainTensorListId id{};
    EXPECT_EQ(ftrainPatternAddTensorList(pattern, &id), FTRAIN_STATUS_SUCCESS);
    return id;
}

FTrainGroupedTensorId addGroupedTensor(FTrainPattern pattern) {
    FTrainGroupedTensorId id{};
    EXPECT_EQ(ftrainPatternAddGroupedTensor(pattern, &id), FTRAIN_STATUS_SUCCESS);
    return id;
}

struct SimpleIds {
    FTrainTensorId first_a;
    FTrainTensorId first_b;
    FTrainTensorId first_c;
    FTrainTensorId first_alpha;
    FTrainTensorId first_beta;
    FTrainTensorId first_d;
    FTrainTensorId second_b;
    FTrainTensorId second_c;
    FTrainTensorId second_alpha;
    FTrainTensorId second_beta;
    FTrainTensorId second_d;
    FTrainGemmOpId first;
    FTrainGemmOpId second;
};

struct SimplePatternHandle {
    FTrainPattern pattern{nullptr};
    SimpleIds ids{};
};

SimplePatternHandle makeSimpleUserPattern() {
    SimplePatternHandle result;
    EXPECT_EQ(ftrainPatternCreate(&result.pattern), FTRAIN_STATUS_SUCCESS);

    // User order is deliberately the reverse of the supported Pattern order.
    result.ids.second_d     = addTensor(result.pattern);
    result.ids.second_beta  = addTensor(result.pattern);
    result.ids.second_alpha = addTensor(result.pattern);
    result.ids.second_c     = addTensor(result.pattern);
    result.ids.second_b     = addTensor(result.pattern);
    result.ids.first_d      = addTensor(result.pattern);
    result.ids.first_beta   = addTensor(result.pattern);
    result.ids.first_alpha  = addTensor(result.pattern);
    result.ids.first_c      = addTensor(result.pattern);
    result.ids.first_b      = addTensor(result.pattern);
    result.ids.first_a      = addTensor(result.pattern);
    EXPECT_EQ(
        ftrainPatternAddGemm(result.pattern, &result.ids.second, result.ids.first_d, result.ids.second_b,
                             result.ids.second_c, result.ids.second_d, result.ids.second_alpha, result.ids.second_beta),
        FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(
        ftrainPatternAddGemm(result.pattern, &result.ids.first, result.ids.first_a, result.ids.first_b,
                             result.ids.first_c, result.ids.first_d, result.ids.first_alpha, result.ids.first_beta),
        FTRAIN_STATUS_SUCCESS);
    return result;
}

struct ComplexIds {
    FTrainTensorId gemm_a;
    FTrainTensorId gemm_b;
    FTrainTensorId gemm_c;
    FTrainTensorId gemm_d;
    FTrainTensorId gemm_alpha;
    FTrainTensorId gemm_beta;
    FTrainGemmOpId gemm;

    FTrainGroupedTensorId grouped_abcd_gemm_a;
    FTrainGroupedTensorId grouped_abcd_gemm_b;
    FTrainGroupedTensorId grouped_abcd_gemm_c;
    FTrainGroupedTensorId grouped_abcd_gemm_d;
    FTrainTensorId grouped_abcd_gemm_alpha;
    FTrainTensorId grouped_abcd_gemm_beta;
    FTrainGroupedABCDGemmOpId grouped_abcd_gemm;

    FTrainTensorListId grouped_bcd_gemm_a;
    FTrainGroupedTensorId grouped_bcd_gemm_b;
    FTrainGroupedTensorId grouped_bcd_gemm_c;
    FTrainGroupedTensorId grouped_bcd_gemm_d;
    FTrainTensorId grouped_bcd_gemm_alpha;
    FTrainTensorId grouped_bcd_gemm_beta;
    FTrainGroupedBCDGemmOpId grouped_bcd_gemm;

    FTrainGroupedTensorId grouped_ab_gemm_a;
    FTrainGroupedTensorId grouped_ab_gemm_b;
    FTrainTensorListId grouped_ab_gemm_c;
    FTrainTensorListId grouped_ab_gemm_d;
    FTrainTensorId grouped_ab_gemm_alpha;
    FTrainTensorId grouped_ab_gemm_beta;
    FTrainGroupedABGemmOpId grouped_ab_gemm;
};

struct ComplexPatternHandle {
    FTrainPattern pattern{nullptr};
    ComplexIds ids{};
};

ComplexPatternHandle makeComplexPattern() {
    ComplexPatternHandle result;
    EXPECT_EQ(ftrainPatternCreate(&result.pattern), FTRAIN_STATUS_SUCCESS);
    ComplexIds& ids = result.ids;

    ids.gemm_a     = addTensor(result.pattern);
    ids.gemm_b     = addTensor(result.pattern);
    ids.gemm_c     = addTensor(result.pattern);
    ids.gemm_d     = addTensor(result.pattern);
    ids.gemm_alpha = addTensor(result.pattern);
    ids.gemm_beta  = addTensor(result.pattern);
    EXPECT_EQ(ftrainPatternAddGemm(result.pattern, &ids.gemm, ids.gemm_a, ids.gemm_b, ids.gemm_c, ids.gemm_d,
                                   ids.gemm_alpha, ids.gemm_beta),
              FTRAIN_STATUS_SUCCESS);

    ids.grouped_abcd_gemm_a     = addGroupedTensor(result.pattern);
    ids.grouped_abcd_gemm_b     = addGroupedTensor(result.pattern);
    ids.grouped_abcd_gemm_c     = addGroupedTensor(result.pattern);
    ids.grouped_abcd_gemm_d     = addGroupedTensor(result.pattern);
    ids.grouped_abcd_gemm_alpha = addTensor(result.pattern);
    ids.grouped_abcd_gemm_beta  = addTensor(result.pattern);
    EXPECT_EQ(ftrainPatternAddGroupedABCDGemm(result.pattern, &ids.grouped_abcd_gemm, ids.grouped_abcd_gemm_a,
                                              ids.grouped_abcd_gemm_b, ids.grouped_abcd_gemm_c, ids.grouped_abcd_gemm_d,
                                              ids.grouped_abcd_gemm_alpha, ids.grouped_abcd_gemm_beta),
              FTRAIN_STATUS_SUCCESS);

    ids.grouped_bcd_gemm_a     = addTensorList(result.pattern);
    ids.grouped_bcd_gemm_b     = addGroupedTensor(result.pattern);
    ids.grouped_bcd_gemm_c     = addGroupedTensor(result.pattern);
    ids.grouped_bcd_gemm_d     = addGroupedTensor(result.pattern);
    ids.grouped_bcd_gemm_alpha = addTensor(result.pattern);
    ids.grouped_bcd_gemm_beta  = addTensor(result.pattern);
    EXPECT_EQ(ftrainPatternAddGroupedBCDGemm(result.pattern, &ids.grouped_bcd_gemm, ids.grouped_bcd_gemm_a,
                                             ids.grouped_bcd_gemm_b, ids.grouped_bcd_gemm_c, ids.grouped_bcd_gemm_d,
                                             ids.grouped_bcd_gemm_alpha, ids.grouped_bcd_gemm_beta),
              FTRAIN_STATUS_SUCCESS);

    ids.grouped_ab_gemm_a     = addGroupedTensor(result.pattern);
    ids.grouped_ab_gemm_b     = addGroupedTensor(result.pattern);
    ids.grouped_ab_gemm_c     = addTensorList(result.pattern);
    ids.grouped_ab_gemm_d     = addTensorList(result.pattern);
    ids.grouped_ab_gemm_alpha = addTensor(result.pattern);
    ids.grouped_ab_gemm_beta  = addTensor(result.pattern);
    EXPECT_EQ(ftrainPatternAddGroupedABGemm(result.pattern, &ids.grouped_ab_gemm, ids.grouped_ab_gemm_a,
                                            ids.grouped_ab_gemm_b, ids.grouped_ab_gemm_c, ids.grouped_ab_gemm_d,
                                            ids.grouped_ab_gemm_alpha, ids.grouped_ab_gemm_beta),
              FTRAIN_STATUS_SUCCESS);
    return result;
}

void ensureComplexOpsEngine(FTrainPattern pattern) {
    static const bool registered = [pattern] {
        auto engine = std::make_shared<MockOpsEngine>(pattern->pattern, PrimitiveList{},
                                                      std::vector<std::shared_ptr<const Finder>>{});
        getGlobalHandle().registerOpsEngine(std::move(engine));
        return true;
    }();
    static_cast<void>(registered);
}

struct ComplexOpsHandle {
    FTrainOps ops{nullptr};
    ComplexIds ids{};
};

ComplexOpsHandle makeComplexOps() {
    ComplexPatternHandle pattern = makeComplexPattern();
    ensureComplexOpsEngine(pattern.pattern);

    ComplexOpsHandle result;
    result.ids = pattern.ids;
    EXPECT_EQ(ftrainOpsCreate(&result.ops, pattern.pattern), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPatternDestroy(pattern.pattern), FTRAIN_STATUS_SUCCESS);
    return result;
}

FTrainStorageView makeContinuousView(void* memory, const std::int64_t* dims, std::uint8_t num_dims,
                                     FTrainNumericType numeric_type = FTRAIN_NUMERIC_TYPE_FP32,
                                     bool is_host_memory            = false) {
    FTrainStorageView view{};
    view.memory         = memory;
    view.dims           = dims;
    view.strides        = nullptr;
    view.num_dims       = num_dims;
    view.numeric_type   = numeric_type;
    view.index_type     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    view.is_host_memory = is_host_memory;
    return view;
}

template<typename Id>
OperandId getSupportedOperandId(FTrainOps ops, Id id) {
    return ops->ops.getRoleMapping().getSupportedOperandId(OperandId{id.opaque});
}

template<typename Id>
OperationId getSupportedOpId(FTrainOps ops, Id id) {
    return ops->ops.getRoleMapping().getSupportedOpId(OperationId{id.opaque});
}

void setCompleteSimpleArgs(FTrainArgs args, const SimpleIds& ids) {
    static std::uint32_t memory[11]{};
    const std::int64_t dims[11][1]{{101}, {102}, {103}, {104}, {105}, {106}, {107}, {108}, {109}, {110}, {111}};
    const FTrainTensorId tensors[11]{ids.first_a,      ids.first_b,     ids.first_c,  ids.first_alpha,
                                     ids.first_beta,   ids.first_d,     ids.second_b, ids.second_c,
                                     ids.second_alpha, ids.second_beta, ids.second_d};
    for (std::size_t index = 0; index < 11; ++index) {
        EXPECT_EQ(ftrainArgsSetTensor(args, tensors[index], makeContinuousView(&memory[index], dims[index], 1)),
                  FTRAIN_STATUS_SUCCESS);
    }

    EXPECT_EQ(ftrainArgsSetGemm(args, ids.first, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsSetGemm(args, ids.second, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_STATUS_SUCCESS);
}

TEST(RuntimeOpsApiTest, MatchesExactPatternAndPreservesUserToSupportedMappingAfterPatternDestruction) {
    static_cast<void>(getSimpleRegistration());
    SimplePatternHandle user = makeSimpleUserPattern();
    FTrainOps ops            = nullptr;

    ASSERT_EQ(ftrainOpsCreate(&ops, user.pattern), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(ops, nullptr);
    const RoleMapping& mapping = ops->ops.getRoleMapping();
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{static_cast<std::size_t>(user.ids.first_a.opaque)}),
              OperandId{0});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{static_cast<std::size_t>(user.ids.first_b.opaque)}),
              OperandId{1});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{static_cast<std::size_t>(user.ids.first_c.opaque)}),
              OperandId{2});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{static_cast<std::size_t>(user.ids.first_alpha.opaque)}),
              OperandId{3});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{static_cast<std::size_t>(user.ids.first_beta.opaque)}),
              OperandId{4});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{static_cast<std::size_t>(user.ids.first_d.opaque)}),
              OperandId{5});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{static_cast<std::size_t>(user.ids.second_b.opaque)}),
              OperandId{6});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{static_cast<std::size_t>(user.ids.second_c.opaque)}),
              OperandId{7});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{static_cast<std::size_t>(user.ids.second_alpha.opaque)}),
              OperandId{8});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{static_cast<std::size_t>(user.ids.second_beta.opaque)}),
              OperandId{9});
    EXPECT_EQ(mapping.getSupportedOperandId(OperandId{static_cast<std::size_t>(user.ids.second_d.opaque)}),
              OperandId{10});
    EXPECT_EQ(mapping.getSupportedOpId(OperationId{user.ids.first.opaque}), OperationId{0});
    EXPECT_EQ(mapping.getSupportedOpId(OperationId{user.ids.second.opaque}), OperationId{1});

    ASSERT_EQ(ftrainPatternDestroy(user.pattern), FTRAIN_STATUS_SUCCESS);
    FTrainArgs args = nullptr;
    ASSERT_EQ(ftrainArgsCreate(&args, ops), FTRAIN_STATUS_SUCCESS);
    setCompleteSimpleArgs(args, user.ids);
    EXPECT_TRUE(args->args.isComplete());

    EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainOpsDestroy(ops), FTRAIN_STATUS_SUCCESS);
}

TEST(RuntimeOpsApiTest, ReportsUnsupportedWithoutChangingOpsOutputAndChecksNullBoundaries) {
    static_cast<void>(getSimpleRegistration());
    SimplePatternHandle supported = makeSimpleUserPattern();
    FTrainOps existing_ops        = nullptr;
    ASSERT_EQ(ftrainOpsCreate(&existing_ops, supported.pattern), FTRAIN_STATUS_SUCCESS);

    FTrainPattern unsupported = nullptr;
    ASSERT_EQ(ftrainPatternCreate(&unsupported), FTRAIN_STATUS_SUCCESS);
    const FTrainTensorId isolated = addTensor(unsupported);
    static_cast<void>(isolated);

    FTrainOps output = existing_ops;
    EXPECT_EQ(ftrainOpsCreate(&output, unsupported), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(output, existing_ops);
    EXPECT_EQ(ftrainOpsCreate(nullptr, unsupported), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainOpsDestroy(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(ftrainPatternDestroy(unsupported), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPatternDestroy(supported.pattern), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainOpsDestroy(existing_ops), FTRAIN_STATUS_SUCCESS);
}

TEST(RuntimeArgsApiTest, CopiesReplacesAndValidatesAllOperandStorageFamilies) {
    ComplexOpsHandle complex = makeComplexOps();
    ASSERT_NE(complex.ops, nullptr);
    FTrainArgs args = nullptr;
    ASSERT_EQ(ftrainArgsCreate(&args, complex.ops), FTRAIN_STATUS_SUCCESS);

    std::uint32_t tensor_memory = 0;
    std::int64_t tensor_dims[2]{2, 3};
    ASSERT_EQ(ftrainArgsSetTensor(args, complex.ids.gemm_a, makeContinuousView(&tensor_memory, tensor_dims, 2)),
              FTRAIN_STATUS_SUCCESS);
    tensor_dims[0]              = 99;
    const OperandId tensor_slot = getSupportedOperandId(complex.ops, complex.ids.gemm_a);
    const Tensor& tensor        = std::get<Tensor>(args->args.getOperand(tensor_slot));
    EXPECT_EQ(tensor.getStorage().getStorageView().getDims(), (std::vector<std::int64_t>{2, 3}));

    const std::int64_t replacement_dims[1]{7};
    ASSERT_EQ(ftrainArgsSetTensor(args, complex.ids.gemm_a, makeContinuousView(&tensor_memory, replacement_dims, 1)),
              FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(tensor.getStorage().getStorageView().getDims(), (std::vector<std::int64_t>{7}));
    FTrainStorageView invalid_tensor = makeContinuousView(&tensor_memory, replacement_dims, 1);
    invalid_tensor.numeric_type      = FTRAIN_NUMERIC_TYPE_COUNT;
    EXPECT_EQ(ftrainArgsSetTensor(args, complex.ids.gemm_a, invalid_tensor), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(tensor.getStorage().getStorageView().getDims(), (std::vector<std::int64_t>{7}));

    std::uint32_t list_memory[2]{};
    std::int64_t list_dims[2]{4, 5};
    FTrainStorageView list_views[2]{makeContinuousView(&list_memory[0], &list_dims[0], 1),
                                    makeContinuousView(&list_memory[1], &list_dims[1], 1)};
    ASSERT_EQ(ftrainArgsSetTensorList(args, complex.ids.grouped_bcd_gemm_a, list_views, 2), FTRAIN_STATUS_SUCCESS);
    list_dims[0]                  = 88;
    const OperandId list_slot     = getSupportedOperandId(complex.ops, complex.ids.grouped_bcd_gemm_a);
    const TensorList& tensor_list = std::get<TensorList>(args->args.getOperand(list_slot));
    ASSERT_EQ(tensor_list.getStorage().getStorageViews().size(), 2);
    EXPECT_EQ(tensor_list.getStorage().getStorageViews()[0].getDims(), (std::vector<std::int64_t>{4}));
    EXPECT_EQ(ftrainArgsSetTensorList(args, complex.ids.grouped_bcd_gemm_a, nullptr, 1),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(tensor_list.getStorage().getStorageViews().size(), 2);
    EXPECT_EQ(ftrainArgsSetTensorList(args, complex.ids.grouped_bcd_gemm_a, nullptr, 0), FTRAIN_STATUS_SUCCESS);
    EXPECT_TRUE(tensor_list.getStorage().getStorageViews().empty());

    std::uint32_t grouped_data_memory[64]{};
    std::int64_t grouped_data_dims[2]{8, 8};
    std::int64_t metadata_shape[1]{2};
    std::int64_t offsets_memory[2]{0, 16};
    std::int64_t dim0_memory[2]{2, 4};
    std::int64_t dim1_memory[2]{8, 8};
    std::int64_t stride0_memory[2]{8, 8};
    std::int64_t stride1_memory[2]{1, 1};
    const FTrainStorageView grouped_data = makeContinuousView(grouped_data_memory, grouped_data_dims, 2);
    const FTrainStorageView offsets =
        makeContinuousView(offsets_memory, metadata_shape, 1, FTRAIN_NUMERIC_TYPE_INT64, true);
    FTrainStorageView dim_sizes[2]{makeContinuousView(dim0_memory, metadata_shape, 1, FTRAIN_NUMERIC_TYPE_INT64, true),
                                   makeContinuousView(dim1_memory, metadata_shape, 1, FTRAIN_NUMERIC_TYPE_INT64, true)};
    FTrainStorageView strides[2]{
        makeContinuousView(stride0_memory, metadata_shape, 1, FTRAIN_NUMERIC_TYPE_INT64, true),
        makeContinuousView(stride1_memory, metadata_shape, 1, FTRAIN_NUMERIC_TYPE_INT64, true)};
    ASSERT_EQ(ftrainArgsSetGroupedTensor(args, complex.ids.grouped_abcd_gemm_a, 2, grouped_data, &offsets, dim_sizes,
                                         strides),
              FTRAIN_STATUS_SUCCESS);
    grouped_data_dims[0]         = 77;
    metadata_shape[0]            = 77;
    const OperandId grouped_slot = getSupportedOperandId(complex.ops, complex.ids.grouped_abcd_gemm_a);
    const GroupedTensor& grouped = std::get<GroupedTensor>(args->args.getOperand(grouped_slot));
    EXPECT_EQ(grouped.getStorage().getNumGroups(), 2);
    EXPECT_EQ(grouped.getStorage().getData().getDims(), (std::vector<std::int64_t>{8, 8}));
    ASSERT_TRUE(grouped.getStorage().getOffsets().has_value());
    EXPECT_EQ(grouped.getStorage().getOffsets()->getDims(), (std::vector<std::int64_t>{2}));
    ASSERT_EQ(grouped.getStorage().getDimSizes().size(), 2);
    ASSERT_EQ(grouped.getStorage().getStrides().size(), 2);
    EXPECT_EQ(grouped.getStorage().getDimSizes()[0].getDims(), (std::vector<std::int64_t>{2}));
    EXPECT_EQ(grouped.getStorage().getStrides()[1].getDims(), (std::vector<std::int64_t>{2}));

    std::int64_t replacement_grouped_dims[1]{1};
    const FTrainStorageView replacement_grouped = makeContinuousView(grouped_data_memory, replacement_grouped_dims, 1);
    ASSERT_EQ(ftrainArgsSetGroupedTensor(args, complex.ids.grouped_abcd_gemm_a, 1, replacement_grouped, nullptr,
                                         nullptr, nullptr),
              FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(grouped.getStorage().getNumGroups(), 1);
    std::int64_t scalar_offset_memory = 0;
    const FTrainStorageView invalid_offsets =
        makeContinuousView(&scalar_offset_memory, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, true);
    EXPECT_EQ(ftrainArgsSetGroupedTensor(args, complex.ids.grouped_abcd_gemm_a, 1, replacement_grouped,
                                         &invalid_offsets, nullptr, nullptr),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(grouped.getStorage().getNumGroups(), 1);

    FTrainArgs unchanged = args;
    EXPECT_EQ(ftrainArgsCreate(&unchanged, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(unchanged, args);
    EXPECT_EQ(ftrainArgsCreate(nullptr, complex.ops), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainArgsSetTensor(nullptr, complex.ids.gemm_a, makeContinuousView(&tensor_memory, replacement_dims, 1)),
              FTRAIN_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainOpsDestroy(complex.ops), FTRAIN_STATUS_SUCCESS);
}

TEST(RuntimeArgsApiTest, SetsAllOperationFamiliesAndPreservesPreviousOperandsOnValidationFailure) {
    ComplexOpsHandle complex = makeComplexOps();
    ASSERT_NE(complex.ops, nullptr);
    FTrainArgs args = nullptr;
    ASSERT_EQ(ftrainArgsCreate(&args, complex.ops), FTRAIN_STATUS_SUCCESS);

    EXPECT_EQ(ftrainArgsSetGroupedABCDGemm(args, complex.ids.grouped_abcd_gemm, FTRAIN_NUMERIC_TYPE_FP32),
              FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsSetGemm(args, complex.ids.gemm, FTRAIN_NUMERIC_TYPE_FP64), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsSetGroupedBCDGemm(args, complex.ids.grouped_bcd_gemm, FTRAIN_NUMERIC_TYPE_FP16),
              FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsSetGroupedABGemm(args, complex.ids.grouped_ab_gemm, FTRAIN_NUMERIC_TYPE_BF16),
              FTRAIN_STATUS_SUCCESS);

    const auto& grouped_abcd_gemm_attributes = std::get<GroupedABCDGemmAttributes>(
        args->args.getOpArgument(getSupportedOpId(complex.ops, complex.ids.grouped_abcd_gemm)).getAttributes());
    const auto& gemm_attributes = std::get<GemmAttributes>(
        args->args.getOpArgument(getSupportedOpId(complex.ops, complex.ids.gemm)).getAttributes());
    const auto& grouped_bcd_attributes = std::get<GroupedBCDGemmAttributes>(
        args->args.getOpArgument(getSupportedOpId(complex.ops, complex.ids.grouped_bcd_gemm)).getAttributes());
    const auto& grouped_ab_attributes = std::get<GroupedABGemmAttributes>(
        args->args.getOpArgument(getSupportedOpId(complex.ops, complex.ids.grouped_ab_gemm)).getAttributes());
    EXPECT_EQ(grouped_abcd_gemm_attributes.getComputeType(), FTRAIN_NUMERIC_TYPE_FP32);
    EXPECT_EQ(gemm_attributes.getComputeType(), FTRAIN_NUMERIC_TYPE_FP64);
    EXPECT_EQ(grouped_bcd_attributes.getComputeType(), FTRAIN_NUMERIC_TYPE_FP16);
    EXPECT_EQ(grouped_ab_attributes.getComputeType(), FTRAIN_NUMERIC_TYPE_BF16);

    EXPECT_EQ(ftrainArgsSetGemm(args, complex.ids.gemm, FTRAIN_NUMERIC_TYPE_INVALID), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainArgsSetGemm(args, complex.ids.gemm, FTRAIN_NUMERIC_TYPE_COUNT), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainArgsSetGroupedABGemm(args, complex.ids.grouped_ab_gemm, FTRAIN_NUMERIC_TYPE_INVALID),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainArgsSetGroupedABGemm(args, complex.ids.grouped_ab_gemm, FTRAIN_NUMERIC_TYPE_COUNT),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(gemm_attributes.getComputeType(), FTRAIN_NUMERIC_TYPE_FP64);
    EXPECT_EQ(grouped_ab_attributes.getComputeType(), FTRAIN_NUMERIC_TYPE_BF16);
    EXPECT_EQ(ftrainArgsSetGemm(nullptr, complex.ids.gemm, FTRAIN_NUMERIC_TYPE_FP32), FTRAIN_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsDestroy(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainOpsDestroy(complex.ops), FTRAIN_STATUS_SUCCESS);
}

TEST(RuntimePlanApiTest, ValidatesBindsCachesExecutesAndOutlivesOpsAndArgs) {
    SimpleRegistration& registration = getSimpleRegistration();
    SimplePatternHandle user         = makeSimpleUserPattern();
    FTrainOps ops                    = nullptr;
    ASSERT_EQ(ftrainOpsCreate(&ops, user.pattern), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternDestroy(user.pattern), FTRAIN_STATUS_SUCCESS);
    FTrainArgs args = nullptr;
    ASSERT_EQ(ftrainArgsCreate(&args, ops), FTRAIN_STATUS_SUCCESS);

    FTrainPlan plan = nullptr;
    EXPECT_EQ(ftrainPlanCreate(&plan, ops, args, 64), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(plan, nullptr);

    ComplexOpsHandle complex = makeComplexOps();
    FTrainArgs complex_args  = nullptr;
    ASSERT_EQ(ftrainArgsCreate(&complex_args, complex.ops), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanCreate(&plan, ops, complex_args, 64), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(plan, nullptr);
    EXPECT_EQ(ftrainArgsDestroy(complex_args), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainOpsDestroy(complex.ops), FTRAIN_STATUS_SUCCESS);

    setCompleteSimpleArgs(args, user.ids);
    EXPECT_EQ(ftrainPlanCreate(&plan, ops, args, 31), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(plan, nullptr);
    static std::atomic<std::uint64_t> next_workspace_limit{1024};
    const std::uint64_t workspace_limit = next_workspace_limit.fetch_add(1, std::memory_order_relaxed);
    const int applicable_before         = registration.state->applicable_calls.load();
    const int create_before             = registration.state->create_calls.load();
    const int finder_before             = registration.state->finder_calls.load();
    ASSERT_EQ(ftrainPlanCreate(&plan, ops, args, workspace_limit), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(plan, nullptr);
    ASSERT_EQ(plan->primitives.size(), 1);
    FTrainPlan cached_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&cached_plan, ops, args, workspace_limit), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(cached_plan, nullptr);
    EXPECT_NE(cached_plan, plan);
    EXPECT_EQ(registration.state->applicable_calls.load() - applicable_before, 1);
    EXPECT_EQ(registration.state->finder_calls.load() - finder_before, 1);
    EXPECT_EQ(registration.state->create_calls.load() - create_before, 2);

    std::uint64_t workspace_bytes = 777;
    EXPECT_EQ(ftrainPlanGetRequiredWs(nullptr, &workspace_bytes), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(workspace_bytes, 777);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPlanGetRequiredWs(plan, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPlanGetRequiredWs(plan, &workspace_bytes), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(workspace_bytes, 32);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_SUCCESS);

    std::uint64_t workspace[4]{};
    const int execute_before = registration.state->execute_calls;
    EXPECT_EQ(ftrainPlanExecute(plan, workspace, 31, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(registration.state->execute_calls, execute_before);
    EXPECT_EQ(ftrainPlanCreate(nullptr, ops, args, 64), FTRAIN_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainOpsDestroy(ops), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanExecute(plan, workspace, sizeof(workspace), nullptr), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(registration.state->execute_calls, execute_before + 1);
    EXPECT_EQ(registration.state->executed_marker, 101);
    EXPECT_EQ(registration.state->workspace, workspace);
    EXPECT_EQ(registration.state->workspace_bytes, sizeof(workspace));
    EXPECT_EQ(registration.state->stream, nullptr);

    EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanDestroy(cached_plan), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanExecute(nullptr, nullptr, 0, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPlanDestroy(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
}

TEST(RuntimePlanApiTest, RejectsExecutionOnAnotherCurrentDeviceWithoutDispatch) {
    SimpleRegistration& registration = getSimpleRegistration();
    SimplePatternHandle user         = makeSimpleUserPattern();
    FTrainOps ops                    = nullptr;
    ASSERT_EQ(ftrainOpsCreate(&ops, user.pattern), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPatternDestroy(user.pattern), FTRAIN_STATUS_SUCCESS);
    FTrainArgs args = nullptr;
    ASSERT_EQ(ftrainArgsCreate(&args, ops), FTRAIN_STATUS_SUCCESS);
    setCompleteSimpleArgs(args, user.ids);

    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, ops, args, 64), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(plan, nullptr);
    plan->device_id = static_cast<FTrainDeviceId>(getCurrentDeviceId() + 1);

    const int execute_before = registration.state->execute_calls;
    std::uint64_t workspace[4]{};
    EXPECT_EQ(ftrainPlanExecute(plan, workspace, sizeof(workspace), nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(registration.state->execute_calls, execute_before);
    EXPECT_NE(strstr(ftrainGetLastMessage(), "calling thread uses device"), nullptr);

    EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainOpsDestroy(ops), FTRAIN_STATUS_SUCCESS);
}

}  // namespace
}  // namespace ftrain
