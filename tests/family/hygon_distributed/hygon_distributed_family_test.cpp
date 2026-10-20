// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <flash_train/engine.hpp>
#include <flash_train/family/ag_group_gemm/family.hpp>
#include <flash_train/family/moe_reduce_rs/family.hpp>
#include <flash_train/ops_args.hpp>
#include <flash_train/operation/operation.hpp>
#include <flash_train/pattern.hpp>
#include <flash_train/session/distributed.hpp>
#include <flash_train/primitive/hygon/distributed/ag_group_gemm.hpp>

namespace ftrain {
namespace {

class FakeAGGroupGemmSession final : public AGGroupGemmSession {
  public:
    explicit FakeAGGroupGemmSession(AGGroupGemmSessionInfo info) : info_(std::move(info)) {}

    const AGGroupGemmSessionInfo& info() const noexcept override { return info_; }
    void execute(const AGGroupGemmProblem&, FTrainStream) override {
        throw std::logic_error("selection tests must not execute a kernel");
    }

  private:
    AGGroupGemmSessionInfo info_;
};

class FakeMoEReduceRSSession final : public MoEReduceRSSession {
  public:
    explicit FakeMoEReduceRSSession(MoEReduceRSSessionInfo info) : info_(std::move(info)) {}

    const MoEReduceRSSessionInfo& info() const noexcept override { return info_; }
    void execute(const MoEReduceRSProblem&, FTrainStream) override {
        throw std::logic_error("selection tests must not execute a kernel");
    }

  private:
    MoEReduceRSSessionInfo info_;
};

struct TestTensor {
    alignas(16) static unsigned char storage[512];

    static Tensor make(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides,
                       FTrainNumericType type) {
        FTrainStorageView view{};
        view.memory       = storage;
        view.dims        = dims.data();
        view.strides     = strides.data();
        view.num_dims    = static_cast<std::uint8_t>(dims.size());
        view.numeric_type = type;
        view.index_type  = FTRAIN_INDEX_TYPE_INVALID;
        // Problems require device memory.  These selection-only tests never
        // dereference the fake storage pointer.
        view.is_host_memory = false;
        return Tensor(StorageView(view));
    }
};

alignas(16) unsigned char TestTensor::storage[512];

AGGroupGemmSessionInfo agInfo(std::uint64_t id, FTrainAGRoute route) {
    AGGroupGemmSessionInfo info;
    info.session_id = id;
    info.route = route;
    info.world_size = 8;
    info.rank = 0;
    info.nnodes = 1;
    info.full_m = route == FTRAIN_AG_ROUTE_NATIVE_NK_A2A ? 32 : 384;
    info.n_dim = 128;
    info.k_dim = 64;
    info.num_experts = 8;
    info.topk = route == FTRAIN_AG_ROUTE_NATIVE_NK_A2A ? 6 : 1;
    info.rank_m_sizes = route == FTRAIN_AG_ROUTE_NATIVE_NK_A2A
                             ? std::vector<int64_t>{4, 4, 4, 4, 4, 4, 4, 4}
                             : std::vector<int64_t>{384, 384, 384, 384, 384, 384, 384, 384};
    info.input_type = FTRAIN_NUMERIC_TYPE_BF16;
    info.output_type = FTRAIN_NUMERIC_TYPE_BF16;
    return info;
}

MoEReduceRSSessionInfo moeInfo(std::uint64_t id, FTrainMoERSRoute route,
                               FTrainMoERSGemmVariant variant = FTRAIN_MOERS_GEMM_NK_DIRECT) {
    MoEReduceRSSessionInfo info;
    info.session_id = id;
    info.route = route;
    info.gemm_variant = variant;
    info.world_size = 8;
    info.rank = 0;
    info.nnodes = 1;
    info.max_ntokens = 32;
    info.n_dim = 128;
    info.k_dim = 64;
    info.num_experts = 8;
    info.topk = 1;
    info.rank_m_sizes = {4, 4, 4, 4, 4, 4, 4, 4};
    info.input_type = FTRAIN_NUMERIC_TYPE_BF16;
    info.n_chunks = 0;
    return info;
}

struct TestTensorSet {
    Tensor input{TestTensor::make({32, 64}, {64, 1}, FTRAIN_NUMERIC_TYPE_BF16)};
    Tensor weight{TestTensor::make({8, 64, 128}, {8192, 128, 1}, FTRAIN_NUMERIC_TYPE_BF16)};
    Tensor topk_ids{TestTensor::make({32, 6}, {6, 1}, FTRAIN_NUMERIC_TYPE_INT32)};
    Tensor output{TestTensor::make({192, 128}, {128, 1}, FTRAIN_NUMERIC_TYPE_BF16)};
    Tensor moe_input{TestTensor::make({32, 64}, {64, 1}, FTRAIN_NUMERIC_TYPE_BF16)};
    Tensor moe_weight{TestTensor::make({8, 128, 64}, {8192, 64, 1}, FTRAIN_NUMERIC_TYPE_BF16)};
    Tensor chosen_experts{TestTensor::make({32, 1}, {1, 1}, FTRAIN_NUMERIC_TYPE_INT32)};
    Tensor expert_weight{TestTensor::make({32, 1}, {1, 1}, FTRAIN_NUMERIC_TYPE_BF16)};
    Tensor moe_output{TestTensor::make({4, 128}, {128, 1}, FTRAIN_NUMERIC_TYPE_BF16)};
};

ftrain::AGGroupGemmProblem agProblem(const TestTensorSet& tensors, std::uint64_t id, FTrainAGRoute route) {
    return AGGroupGemmProblem(tensors.input, tensors.weight, tensors.topk_ids, tensors.output,
                              AGGroupGemmAttributes(route, id));
}

ftrain::MoEReduceRSProblem moeProblem(const TestTensorSet& tensors, std::uint64_t id, FTrainMoERSRoute route,
                                      FTrainMoERSGemmVariant variant = FTRAIN_MOERS_GEMM_NK_DIRECT) {
    return MoEReduceRSProblem(tensors.moe_input, tensors.moe_weight, tensors.chosen_experts,
                              tensors.expert_weight, tensors.moe_output,
                              MoEReduceRSAttributes(route, variant, id));
}

TestTensorSet routeTensors(FTrainAGRoute route) {
    TestTensorSet tensors;
    if (route != FTRAIN_AG_ROUTE_NATIVE_NK_A2A) {
        tensors.topk_ids = TestTensor::make({384, 1}, {1, 1}, FTRAIN_NUMERIC_TYPE_INT32);
        tensors.output = TestTensor::make({384, 128}, {128, 1}, FTRAIN_NUMERIC_TYPE_BF16);
    }
    return tensors;
}

Args makeAGArgs(const TestTensorSet& tensors, std::uint64_t session_id, FTrainAGRoute route, std::vector<int64_t> topk_dims = {32, 6}, std::vector<int64_t> output_dims = {192, 128}) {
    PatternBuilder user;
    const auto input = user.addOperand<OperandKind::kTensor>();
    const auto weight = user.addOperand<OperandKind::kTensor>();
    const auto topk_ids = user.addOperand<OperandKind::kTensor>();
    const auto output = user.addOperand<OperandKind::kTensor>();
    const auto op = user.addOperation<OperationKind::kAGGroupGemm>(input, weight, topk_ids, output);
    Ops ops(user.buildPattern(), AGGroupGemmFamily::makePattern());
    Args args = ops.makeArgs();
    args.setOperand<OperandKind::kTensor>(
        OperandTraits<OperandKind::kTensor>::createPatternOperandId(input), Tensor(tensors.input));
    args.setOperand<OperandKind::kTensor>(
        OperandTraits<OperandKind::kTensor>::createPatternOperandId(weight), Tensor(tensors.weight));
    const std::vector<int64_t>& actual_topk_dims = topk_dims;
    const std::vector<int64_t>& actual_output_dims = output_dims;
    args.setOperand<OperandKind::kTensor>(
        OperandTraits<OperandKind::kTensor>::createPatternOperandId(topk_ids),
        TestTensor::make(actual_topk_dims, {actual_topk_dims[1], 1}, FTRAIN_NUMERIC_TYPE_INT32));
    args.setOperand<OperandKind::kTensor>(
        OperandTraits<OperandKind::kTensor>::createPatternOperandId(output),
        TestTensor::make(actual_output_dims, {actual_output_dims[1], 1}, FTRAIN_NUMERIC_TYPE_BF16));
    args.setOperation<OperationKind::kAGGroupGemm>(
        OperationTraits<OperationKind::kAGGroupGemm>::createPatternOperationId(op),
        AGGroupGemmAttributes(route, session_id));
    return args;
}

Args makeMoEArgs(const TestTensorSet& tensors, std::uint64_t session_id, FTrainMoERSRoute route,
                 FTrainMoERSGemmVariant variant = FTRAIN_MOERS_GEMM_NK_DIRECT) {
    PatternBuilder user;
    const auto input = user.addOperand<OperandKind::kTensor>();
    const auto weight = user.addOperand<OperandKind::kTensor>();
    const auto chosen_experts = user.addOperand<OperandKind::kTensor>();
    const auto expert_weight = user.addOperand<OperandKind::kTensor>();
    const auto output = user.addOperand<OperandKind::kTensor>();
    const auto op =
        user.addOperation<OperationKind::kMoEReduceRS>(input, weight, chosen_experts, expert_weight, output);
    Ops ops(user.buildPattern(), MoEReduceRSFamily::makePattern());
    Args args = ops.makeArgs();
    args.setOperand<OperandKind::kTensor>(
        OperandTraits<OperandKind::kTensor>::createPatternOperandId(input), Tensor(tensors.moe_input));
    args.setOperand<OperandKind::kTensor>(
        OperandTraits<OperandKind::kTensor>::createPatternOperandId(weight), Tensor(tensors.moe_weight));
    args.setOperand<OperandKind::kTensor>(
        OperandTraits<OperandKind::kTensor>::createPatternOperandId(chosen_experts), Tensor(tensors.chosen_experts));
    args.setOperand<OperandKind::kTensor>(
        OperandTraits<OperandKind::kTensor>::createPatternOperandId(expert_weight), Tensor(tensors.expert_weight));
    args.setOperand<OperandKind::kTensor>(
        OperandTraits<OperandKind::kTensor>::createPatternOperandId(output), Tensor(tensors.moe_output));
    args.setOperation<OperationKind::kMoEReduceRS>(
        OperationTraits<OperationKind::kMoEReduceRS>::createPatternOperationId(op),
        MoEReduceRSAttributes(route, variant, session_id));
    return args;
}

std::vector<std::string> primitiveNames(const std::vector<std::unique_ptr<PrimitiveBase>>& primitives) {
    std::vector<std::string> names;
    names.reserve(primitives.size());
    for (const auto& primitive : primitives) { names.emplace_back(primitive->getName()); }
    return names;
}

const Constraints kConstraints{0};

TEST(HygonAGGroupGemmFamilyTest, DefinesExactPattern) {
    const OpsEngine<AGGroupGemmFamily, AGGroupGemmRouteFinder> engine;
    const auto& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 4U);
    ASSERT_EQ(supported.getNumOps(), 1U);
    for (std::size_t index = 0; index < supported.getNumOperands(); ++index) {
        EXPECT_EQ(supported.getOperandNode(PatternOperandId{index}).getKind(), OperandKind::kTensor);
    }
    EXPECT_EQ(supported.getOpNode(PatternOperationId{0}).getKind(), OperationKind::kAGGroupGemm);
}

TEST(HygonAGGroupGemmFamilyTest, ProblemKeyCoversRouteSessionAndTensors) {
    TestTensorSet tensors;
    const auto first = agProblem(tensors, 1, FTRAIN_AG_ROUTE_NATIVE_NK_A2A);
    EXPECT_EQ(first.getProblemKey(), agProblem(tensors, 1, FTRAIN_AG_ROUTE_NATIVE_NK_A2A).getProblemKey());
    EXPECT_NE(first.getProblemKey(), agProblem(tensors, 2, FTRAIN_AG_ROUTE_NATIVE_NK_A2A).getProblemKey());
    EXPECT_NE(first.getProblemKey(), agProblem(tensors, 1, FTRAIN_AG_ROUTE_NATIVE_KN_A2A).getProblemKey());

    TestTensorSet changed;
    changed.output = TestTensor::make({384, 128}, {128, 1}, FTRAIN_NUMERIC_TYPE_BF16);
    changed.topk_ids = TestTensor::make({48, 6}, {6, 1}, FTRAIN_NUMERIC_TYPE_INT32);
    EXPECT_NE(first.getProblemKey(), agProblem(changed, 1, FTRAIN_AG_ROUTE_NATIVE_NK_A2A).getProblemKey());
}

TEST(HygonAGGroupGemmFamilyTest, FinderSelectsRouteOwnedPrimitive) {
    const std::pair<FTrainAGRoute, std::string> cases[] = {
        {FTRAIN_AG_ROUTE_NATIVE_NK_A2A, "HygonAGGroupGemmNativeNkA2a"},
        {FTRAIN_AG_ROUTE_NATIVE_KN_A2A, "HygonAGGroupGemmNativeKnA2a"},
        {FTRAIN_AG_ROUTE_NATIVE_KN_RING_FIXED, "HygonAGGroupGemmNativeKnRingFixed"},
        {FTRAIN_AG_ROUTE_NATIVE_KN_RING, "HygonAGGroupGemmNativeKnRing"},
        {FTRAIN_AG_ROUTE_NATIVE_KN_RING_READY_MASK, "HygonAGGroupGemmNativeKnRingReadyMask"},
    };
    const TestTensorSet tensors;
    for (const auto& [route, expected] : cases) {
        const auto problem = agProblem(tensors, 1, route);
        EXPECT_EQ(AGGroupGemmRouteFinder::findCandidates(problem, kConstraints),
                  (std::vector<std::string>{expected}));
    }
}

TEST(HygonAGGroupGemmFamilyTest, EngineSelectsRouteOwnedPrimitive) {
    const std::pair<FTrainAGRoute, std::string> cases[] = {
        {FTRAIN_AG_ROUTE_NATIVE_NK_A2A, "HygonAGGroupGemmNativeNkA2a"},
        {FTRAIN_AG_ROUTE_NATIVE_KN_A2A, "HygonAGGroupGemmNativeKnA2a"},
        {FTRAIN_AG_ROUTE_NATIVE_KN_RING_FIXED, "HygonAGGroupGemmNativeKnRingFixed"},
        {FTRAIN_AG_ROUTE_NATIVE_KN_RING, "HygonAGGroupGemmNativeKnRing"},
        {FTRAIN_AG_ROUTE_NATIVE_KN_RING_READY_MASK, "HygonAGGroupGemmNativeKnRingReadyMask"},
    };
    const OpsEngine<AGGroupGemmFamily, AGGroupGemmRouteFinder> engine;
    for (const auto& [route, expected] : cases) {
        const TestTensorSet tensors = routeTensors(route);
        const auto session_id = route + 100;
        DistributedSessionRegistry::get().registerAGGroupGemm(
            session_id, std::make_shared<FakeAGGroupGemmSession>(agInfo(session_id, route)));
        const auto problem = agProblem(tensors, session_id, route);
        const std::vector<int64_t> topk_dims =
            route == FTRAIN_AG_ROUTE_NATIVE_NK_A2A ? std::vector<int64_t>{32, 6} : std::vector<int64_t>{384, 1};
        const std::vector<int64_t> output_dims =
            route == FTRAIN_AG_ROUTE_NATIVE_NK_A2A ? std::vector<int64_t>{192, 128} : std::vector<int64_t>{384, 128};
        const auto args = makeAGArgs(tensors, session_id, route, topk_dims, output_dims);
        const auto primitives = engine.createPrimitives(args, kConstraints);
        ASSERT_EQ(primitiveNames(primitives), (std::vector<std::string>{expected}));
        DistributedSessionRegistry::get().eraseAGGroupGemm(session_id);
    }
}

TEST(HygonAGGroupGemmFamilyTest, RejectsMissingOrMismatchedSession) {
    const TestTensorSet tensors;
    const OpsEngine<AGGroupGemmFamily, AGGroupGemmRouteFinder> engine;
    const auto problem = agProblem(tensors, 999, FTRAIN_AG_ROUTE_NATIVE_NK_A2A);
    const auto missing_args = makeAGArgs(tensors, 999, FTRAIN_AG_ROUTE_NATIVE_NK_A2A);
    EXPECT_THROW(static_cast<void>(engine.createPrimitives(missing_args, kConstraints)), Exception);

    constexpr auto session_id = 1000;
    DistributedSessionRegistry::get().registerAGGroupGemm(
        session_id, std::make_shared<FakeAGGroupGemmSession>(
                        agInfo(session_id, FTRAIN_AG_ROUTE_NATIVE_KN_A2A)));
    const auto mismatched = makeAGArgs(tensors, session_id, FTRAIN_AG_ROUTE_NATIVE_NK_A2A);
    EXPECT_THROW(static_cast<void>(engine.createPrimitives(mismatched, kConstraints)), Exception);
    DistributedSessionRegistry::get().eraseAGGroupGemm(session_id);
}

TEST(HygonAGGroupGemmFamilyTest, ExpandedRoutesRequireGlobalOutputRows) {
    const std::vector<FTrainAGRoute> routes = {
        FTRAIN_AG_ROUTE_NATIVE_KN_A2A,
        FTRAIN_AG_ROUTE_NATIVE_KN_RING_FIXED,
        FTRAIN_AG_ROUTE_NATIVE_KN_RING,
        FTRAIN_AG_ROUTE_NATIVE_KN_RING_READY_MASK,
    };
    for (const auto route : routes) {
        const auto session_id = route + 3000;
        DistributedSessionRegistry::get().registerAGGroupGemm(
            session_id, std::make_shared<FakeAGGroupGemmSession>(agInfo(session_id, route)));
        const TestTensorSet tensors = routeTensors(route);
        const auto problem = agProblem(tensors, session_id, route);
        HygonAGGroupGemmPrimitive primitive{route};
        EXPECT_TRUE(primitive.isApplicable(problem, kConstraints).isSuccess());
        DistributedSessionRegistry::get().eraseAGGroupGemm(session_id);
    }
}

TEST(HygonMoEReduceRSFamilyTest, DefinesExactPatternAndSelectsRouteOwnedPrimitive) {
    constexpr auto session_id = 2000;
    const auto route = FTRAIN_MOERS_ROUTE_A2A_RAGGED_SIGNAL;
    DistributedSessionRegistry::get().registerMoEReduceRS(
        session_id, std::make_shared<FakeMoEReduceRSSession>(moeInfo(session_id, route)));

    const OpsEngine<MoEReduceRSFamily, MoEReduceRSRouteFinder> engine;
    const auto& supported = engine.getPattern();
    ASSERT_EQ(supported.getNumOperands(), 5U);
    ASSERT_EQ(supported.getNumOps(), 1U);

    const TestTensorSet tensors;
    const auto problem = moeProblem(tensors, session_id, route);
    EXPECT_EQ(MoEReduceRSRouteFinder::findCandidates(problem, kConstraints),
              (std::vector<std::string>{"HygonMoEReduceRSRaggedSignal"}));
    const auto args = makeMoEArgs(tensors, session_id, route);
    const auto primitives = engine.createPrimitives(args, kConstraints);
    ASSERT_EQ(primitiveNames(primitives), (std::vector<std::string>{"HygonMoEReduceRSRaggedSignal"}));

    TestTensorSet kn_tensors;
    kn_tensors.moe_weight = TestTensor::make({8, 64, 128}, {8192, 128, 1}, FTRAIN_NUMERIC_TYPE_BF16);
    const auto wrong_variant =
        moeProblem(kn_tensors, session_id, route, FTRAIN_MOERS_GEMM_KN_DIRECT);
    const auto wrong_args = makeMoEArgs(kn_tensors, session_id, route, FTRAIN_MOERS_GEMM_KN_DIRECT);
    EXPECT_THROW(static_cast<void>(engine.createPrimitives(wrong_args, kConstraints)), Exception);
    DistributedSessionRegistry::get().eraseMoEReduceRS(session_id);
}

}  // namespace

}  // namespace ftrain
