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

#include "flash_train/api.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/engine/base.hpp"
#include "flash_train/handle.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/primitive/base.hpp"
#include "flash_train/ops_args.hpp"
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

    // The mock's single record has the same selection and applicability for
    // all complete Args. Device and workspace limit are encoded by OpsEngine.
    std::vector<std::uint64_t> getProblemKey() const { return {}; }
};

class MockPrimitive final : public Primitive<MockProblem> {
  public:
    explicit MockPrimitive(std::shared_ptr<MockState> state) : state_(std::move(state)) {}

    const char* getName() const noexcept override { return "MockPrimitive"; }

    std::unique_ptr<PrimitiveBase> clone() const override { return std::make_unique<MockPrimitive>(*this); }

    Result isApplicable(const MockProblem&, const Constraints& constraints) const override {
        ++state_->applicable_calls;
        if (constraints.getMaxWorkspaceBytes() < 32) {
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
    void executeImpl(const Resources& resources) override {
        ++state_->execute_calls;
        state_->executed_marker = marker_;
        state_->workspace       = resources.getWorkspace();
        state_->workspace_bytes = resources.getWorkspaceBytes();
        state_->stream          = resources.getStream();
    }

    std::shared_ptr<MockState> state_;
    int marker_ = 0;
};

// Finder policy for the mock engines: always enabled, offers every
// registered record, and counts its calls into the registration's state.
using MockRecords = std::vector<std::shared_ptr<const Primitive<MockProblem>>>;

struct MockFinderPolicy {
    static const char* getName() { return "Mock"; }

    static inline std::shared_ptr<MockState> state;

    static bool isEnabled(const Args&, const Constraints&) { return true; }

    static MockRecords findCandidates(const MockRecords& records, const Args&, const Constraints&) {
        ++state->finder_calls;
        return records;
    }

    static void sortCandidates(const Args&, const Constraints&, MockRecords&) {}
};

// Family policies for the mock engines: the simple family carries the plan
// tests' single record, and the complex family only needs a supported
// schema for Ops matching. ftrainOpsCreate verifies each family's wiring
// against the C-API Patterns the tests build.
struct MockFamily {
    using Problem = MockProblem;

    static inline MockRecords records;

    static Pattern makePattern();

    static MockRecords makeRecords() { return records; }

    static MockProblem makeProblem(const Args& args) {
        return MockProblem{std::get<Tensor>(*args.getOperand(PatternOperandId{0})).getStorageView()};
    }
};

struct ComplexMockFamily {
    using Problem = MockProblem;

    static Pattern makePattern();

    static MockRecords makeRecords() { return {}; }

    static MockProblem makeProblem(const Args& args) {
        return MockProblem{std::get<Tensor>(*args.getOperand(PatternOperandId{0})).getStorageView()};
    }
};

using SimpleOpsEngine  = OpsEngine<MockFamily, MockFinderPolicy>;
using ComplexOpsEngine = OpsEngine<ComplexMockFamily>;

// Supported operand order: first_a, first_b, first_c, first_alpha,
// first_beta, first_d, second_b, second_c, second_alpha, second_beta,
// second_d. The first Gemm produces first_d; the second Gemm consumes
// first_d and produces second_d.
Pattern MockFamily::makePattern() {
    PatternBuilder pattern;
    const FTrainTensorId first_a      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_b      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_c      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_alpha  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_beta   = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId first_d      = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_b     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_c     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_alpha = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_beta  = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId second_d     = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(
        pattern.addOperation<OperationKind::kGemm>(first_a, first_b, first_c, first_d, first_alpha, first_beta));
    static_cast<void>(
        pattern.addOperation<OperationKind::kGemm>(first_d, second_b, second_c, second_d, second_alpha, second_beta));
    return pattern.buildPattern();
}

struct SimpleRegistration {
    SimpleRegistration() {
        state                   = std::make_shared<MockState>();
        record                  = std::make_shared<MockPrimitive>(state);
        MockFamily::records     = MockRecords{record};
        MockFinderPolicy::state = state;
        engine                  = std::make_shared<SimpleOpsEngine>();
        getGlobalHandle().registerOpsEngine(engine);
    }

    std::shared_ptr<MockState> state;
    std::shared_ptr<MockPrimitive> record;
    std::shared_ptr<SimpleOpsEngine> engine;
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

    // User order is deliberately the reverse of the supported PatternBuilder order.
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

// Must stay isomorphic to makeComplexPattern's C-API wiring below;
// ftrainOpsCreate verifies the match at runtime.
Pattern ComplexMockFamily::makePattern() {
    PatternBuilder pattern;
    const FTrainTensorId gemm_a     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId gemm_b     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId gemm_c     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId gemm_d     = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId gemm_alpha = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId gemm_beta  = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(
        pattern.addOperation<OperationKind::kGemm>(gemm_a, gemm_b, gemm_c, gemm_d, gemm_alpha, gemm_beta));

    const FTrainGroupedTensorId abcd_a = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId abcd_b = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId abcd_c = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId abcd_d = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainTensorId abcd_alpha    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId abcd_beta     = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(
        pattern.addOperation<OperationKind::kGroupedABCDGemm>(abcd_a, abcd_b, abcd_c, abcd_d, abcd_alpha, abcd_beta));

    const FTrainTensorListId bcd_a    = pattern.addOperand<OperandKind::kTensorList>();
    const FTrainGroupedTensorId bcd_b = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId bcd_c = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId bcd_d = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainTensorId bcd_alpha    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId bcd_beta     = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(
        pattern.addOperation<OperationKind::kGroupedBCDGemm>(bcd_a, bcd_b, bcd_c, bcd_d, bcd_alpha, bcd_beta));

    const FTrainGroupedTensorId ab_a = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainGroupedTensorId ab_b = pattern.addOperand<OperandKind::kGroupedTensor>();
    const FTrainTensorListId ab_c    = pattern.addOperand<OperandKind::kTensorList>();
    const FTrainTensorListId ab_d    = pattern.addOperand<OperandKind::kTensorList>();
    const FTrainTensorId ab_alpha    = pattern.addOperand<OperandKind::kTensor>();
    const FTrainTensorId ab_beta     = pattern.addOperand<OperandKind::kTensor>();
    static_cast<void>(pattern.addOperation<OperationKind::kGroupedABGemm>(ab_a, ab_b, ab_c, ab_d, ab_alpha, ab_beta));
    return pattern.buildPattern();
}

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

void ensureComplexOpsEngine(FTrainPattern) {
    static const bool registered = [] {
        getGlobalHandle().registerOpsEngine(std::make_shared<ComplexOpsEngine>());
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

    ASSERT_EQ(ftrainPatternDestroy(user.pattern), FTRAIN_STATUS_SUCCESS);
    FTrainArgs args = nullptr;
    ASSERT_EQ(ftrainArgsCreate(&args, ops), FTRAIN_STATUS_SUCCESS);
    setCompleteSimpleArgs(args, user.ids);
    EXPECT_TRUE(args->args.isComplete());

    // setCompleteSimpleArgs stores dims 101..111 in user order first_a..second_d
    // and the supported Pattern was built in that same order, so supported slot
    // i must hold dims 101 + i for the mapping to be the identity the engine
    // registered.
    for (std::size_t slot = 0; slot < 11; ++slot) {
        const Tensor& slot_tensor = std::get<Tensor>(*args->args.getOperand(PatternOperandId{slot}));
        EXPECT_EQ(slot_tensor.getStorageView().getDims(),
                  (std::vector<std::int64_t>{static_cast<std::int64_t>(101 + slot)}));
    }
    // Distinct compute types reveal which supported slot each user operation
    // mapped to.
    EXPECT_EQ(ftrainArgsSetGemm(args, user.ids.first, FTRAIN_NUMERIC_TYPE_FP64), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainArgsSetGemm(args, user.ids.second, FTRAIN_NUMERIC_TYPE_BF16), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(std::get<GemmAttributes>(*args->args.getOpArgument(PatternOperationId{0})).getComputeType(),
              FTRAIN_NUMERIC_TYPE_FP64);
    EXPECT_EQ(std::get<GemmAttributes>(*args->args.getOpArgument(PatternOperationId{1})).getComputeType(),
              FTRAIN_NUMERIC_TYPE_BF16);

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
    tensor_dims[0] = 99;
    // The complex Ops matched the identical supported Pattern, so supported
    // slots equal the user IDs.
    const PatternOperandId tensor_slot = PatternOperandId{complex.ids.gemm_a.opaque};
    const Tensor& tensor               = std::get<Tensor>(*args->args.getOperand(tensor_slot));
    EXPECT_EQ(tensor.getStorageView().getDims(), (std::vector<std::int64_t>{2, 3}));

    const std::int64_t replacement_dims[1]{7};
    ASSERT_EQ(ftrainArgsSetTensor(args, complex.ids.gemm_a, makeContinuousView(&tensor_memory, replacement_dims, 1)),
              FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(tensor.getStorageView().getDims(), (std::vector<std::int64_t>{7}));
    FTrainStorageView invalid_tensor = makeContinuousView(&tensor_memory, replacement_dims, 1);
    invalid_tensor.numeric_type      = FTRAIN_NUMERIC_TYPE_COUNT;
    EXPECT_EQ(ftrainArgsSetTensor(args, complex.ids.gemm_a, invalid_tensor), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(tensor.getStorageView().getDims(), (std::vector<std::int64_t>{7}));

    std::uint32_t list_memory[2]{};
    std::int64_t list_dims[2]{4, 5};
    FTrainStorageView list_views[2]{makeContinuousView(&list_memory[0], &list_dims[0], 1),
                                    makeContinuousView(&list_memory[1], &list_dims[1], 1)};
    ASSERT_EQ(ftrainArgsSetTensorList(args, complex.ids.grouped_bcd_gemm_a, list_views, 2), FTRAIN_STATUS_SUCCESS);
    list_dims[0]                     = 88;
    const PatternOperandId list_slot = PatternOperandId{complex.ids.grouped_bcd_gemm_a.opaque};
    const TensorList& tensor_list    = std::get<TensorList>(*args->args.getOperand(list_slot));
    ASSERT_EQ(tensor_list.getStorageViews().size(), 2);
    EXPECT_EQ(tensor_list.getStorageViews()[0].getDims(), (std::vector<std::int64_t>{4}));
    EXPECT_EQ(ftrainArgsSetTensorList(args, complex.ids.grouped_bcd_gemm_a, nullptr, 1),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(tensor_list.getStorageViews().size(), 2);
    EXPECT_EQ(ftrainArgsSetTensorList(args, complex.ids.grouped_bcd_gemm_a, nullptr, 0), FTRAIN_STATUS_SUCCESS);
    EXPECT_TRUE(tensor_list.getStorageViews().empty());

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
    grouped_data_dims[0]                = 77;
    metadata_shape[0]                   = 77;
    const PatternOperandId grouped_slot = PatternOperandId{complex.ids.grouped_abcd_gemm_a.opaque};
    const GroupedTensor& grouped        = std::get<GroupedTensor>(*args->args.getOperand(grouped_slot));
    EXPECT_EQ(grouped.getNumGroups(), 2);
    EXPECT_EQ(grouped.getData().getDims(), (std::vector<std::int64_t>{8, 8}));
    ASSERT_TRUE(grouped.getOffsets().has_value());
    EXPECT_EQ(grouped.getOffsets()->getDims(), (std::vector<std::int64_t>{2}));
    ASSERT_EQ(grouped.getDimSizes().size(), 2);
    ASSERT_EQ(grouped.getStrides().size(), 2);
    EXPECT_EQ(grouped.getDimSizes()[0].getDims(), (std::vector<std::int64_t>{2}));
    EXPECT_EQ(grouped.getStrides()[1].getDims(), (std::vector<std::int64_t>{2}));

    std::int64_t replacement_grouped_dims[1]{1};
    const FTrainStorageView replacement_grouped = makeContinuousView(grouped_data_memory, replacement_grouped_dims, 1);
    ASSERT_EQ(ftrainArgsSetGroupedTensor(args, complex.ids.grouped_abcd_gemm_a, 1, replacement_grouped, nullptr,
                                         nullptr, nullptr),
              FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(grouped.getNumGroups(), 1);
    std::int64_t scalar_offset_memory = 0;
    const FTrainStorageView invalid_offsets =
        makeContinuousView(&scalar_offset_memory, nullptr, 0, FTRAIN_NUMERIC_TYPE_INT64, true);
    EXPECT_EQ(ftrainArgsSetGroupedTensor(args, complex.ids.grouped_abcd_gemm_a, 1, replacement_grouped,
                                         &invalid_offsets, nullptr, nullptr),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(grouped.getNumGroups(), 1);

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
        *args->args.getOpArgument(PatternOperationId{complex.ids.grouped_abcd_gemm.opaque}));
    const auto& gemm_attributes =
        std::get<GemmAttributes>(*args->args.getOpArgument(PatternOperationId{complex.ids.gemm.opaque}));
    const auto& grouped_bcd_attributes = std::get<GroupedBCDGemmAttributes>(
        *args->args.getOpArgument(PatternOperationId{complex.ids.grouped_bcd_gemm.opaque}));
    const auto& grouped_ab_attributes = std::get<GroupedABGemmAttributes>(
        *args->args.getOpArgument(PatternOperationId{complex.ids.grouped_ab_gemm.opaque}));
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
    EXPECT_EQ(ftrainPlanCreate(&plan, args, 64), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(plan, nullptr);

    ComplexOpsHandle complex = makeComplexOps();
    FTrainArgs complex_args  = nullptr;
    ASSERT_EQ(ftrainArgsCreate(&complex_args, complex.ops), FTRAIN_STATUS_SUCCESS);
    // The complex Args are never filled in this test, so validation rejects
    // them before the engine can answer; plan creation now locates the
    // engine through the Args' own topology.
    EXPECT_EQ(ftrainPlanCreate(&plan, complex_args, 64), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(plan, nullptr);
    EXPECT_EQ(ftrainArgsDestroy(complex_args), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainOpsDestroy(complex.ops), FTRAIN_STATUS_SUCCESS);

    setCompleteSimpleArgs(args, user.ids);
    EXPECT_EQ(ftrainPlanCreate(&plan, args, 31), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(plan, nullptr);
    static std::atomic<std::uint64_t> next_workspace_limit{1024};
    const std::uint64_t workspace_limit = next_workspace_limit.fetch_add(1, std::memory_order_relaxed);
    const int applicable_before         = registration.state->applicable_calls.load();
    const int create_before             = registration.state->create_calls.load();
    const int finder_before             = registration.state->finder_calls.load();
    ASSERT_EQ(ftrainPlanCreate(&plan, args, workspace_limit), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(plan, nullptr);
    std::uint64_t num_primitives = 0;
    EXPECT_EQ(ftrainPlanGetNumPrimitives(nullptr, &num_primitives), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPlanGetNumPrimitives(plan, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPlanGetNumPrimitives(plan, &num_primitives), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(num_primitives, 1);
    FTrainPlan cached_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&cached_plan, args, workspace_limit), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(cached_plan, nullptr);
    EXPECT_NE(cached_plan, plan);
    EXPECT_EQ(registration.state->applicable_calls.load() - applicable_before, 1);
    EXPECT_EQ(registration.state->finder_calls.load() - finder_before, 1);
    EXPECT_EQ(registration.state->create_calls.load() - create_before, 2);

    std::uint64_t workspace_bytes = 777;
    EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(nullptr, 0, &workspace_bytes),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(workspace_bytes, 777);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 1, &workspace_bytes), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &workspace_bytes), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(workspace_bytes, 32);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_SUCCESS);

    std::uint64_t workspace[4]{};
    const int execute_before = registration.state->execute_calls;
    EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, 31, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(registration.state->execute_calls, execute_before);
    EXPECT_EQ(ftrainPlanCreate(nullptr, args, 64), FTRAIN_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainOpsDestroy(ops), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, workspace, sizeof(workspace), nullptr), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(registration.state->execute_calls, execute_before + 1);
    EXPECT_EQ(registration.state->executed_marker, 101);
    EXPECT_EQ(registration.state->workspace, workspace);
    EXPECT_EQ(registration.state->workspace_bytes, sizeof(workspace));
    EXPECT_EQ(registration.state->stream, nullptr);

    EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanDestroy(cached_plan), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainPlanExecutePrimitive(nullptr, 0, nullptr, 0, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
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

    // The mismatched device is injected through the internal Plan constructor;
    // a C-API-created plan always carries the creating thread's device.
    std::vector<std::unique_ptr<PrimitiveBase>> mismatched_primitives;
    mismatched_primitives.push_back(std::make_unique<MockPrimitive>(registration.state));
    FTrainPlanStruct mismatched_plan{
        ftrain::Plan{std::move(mismatched_primitives), static_cast<FTrainDeviceId>(getCurrentDeviceId() + 1)}
    };

    const int execute_before = registration.state->execute_calls;
    std::uint64_t workspace[4]{};
    EXPECT_EQ(ftrainPlanExecutePrimitive(&mismatched_plan, 0, workspace, sizeof(workspace), nullptr),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(registration.state->execute_calls, execute_before);
    EXPECT_NE(strstr(ftrainGetLastMessage(), "calling thread uses device"), nullptr);

    EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainOpsDestroy(ops), FTRAIN_STATUS_SUCCESS);
}

}  // namespace
}  // namespace ftrain
