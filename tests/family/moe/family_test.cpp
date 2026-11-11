#include <array>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/engine.hpp"
#include "flash_train/family/moe/family.hpp"
#include "flash_train/handle.hpp"

#include "../../support/moe_session.hpp"

namespace ftrain {
namespace {

using MoeFamilyTest = ftrain_test::MoeSessionTest;

template<typename Family>
void expectRegisteredPattern(OperationKind kind) {
    const auto& roles = Family::makeRoles();
    EXPECT_EQ(roles.pattern.getNumOperands(), Family::num_ports);
    ASSERT_EQ(roles.pattern.getNumOps(), 1);
    const auto& node = roles.pattern.getOpNode(PatternOperationId{roles.operation.opaque});
    EXPECT_EQ(node.getKind(), kind);
    ASSERT_EQ(node.getInputs().size(), Family::num_ports - 1);
    for (size_t index = 0; index + 1 < Family::num_ports; ++index) {
        EXPECT_EQ(node.getInputs()[index], PatternOperandId{roles.ports[index].opaque});
    }
    EXPECT_EQ(node.getOutputs(), (std::vector<PatternOperandId>{PatternOperandId{roles.ports.back().opaque}}));
    const auto engine = getGlobalHandle().findOpsEngine(roles.pattern.getKey());
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->getPattern().getKey(), roles.pattern.getKey());
}





TEST_F(MoeFamilyTest, ProblemKeysDistinguishRoutesShapeAndRoutingDtype) {
    auto original_session = session(makeMoeConfig(*context, ag_config));
    const auto original   = MoeProblem(original_session, bindings(original_session)).getProblemKey();
    EXPECT_NE(original,
              MoeProblem(original_session, bindings(original_session, FTRAIN_NUMERIC_TYPE_INT64)).getProblemKey());
    ag_config.route = "ag_nn_ring";
    auto changed    = session(makeMoeConfig(*context, ag_config));
    EXPECT_NE(original, MoeProblem(changed, bindings(changed)).getProblemKey());
    ag_config.route  = "ag_nn_ring_chunk";
    ag_config.n_dim *= 2;
    changed          = session(makeMoeConfig(*context, ag_config));
    EXPECT_NE(original, MoeProblem(changed, bindings(changed)).getProblemKey());
}

TEST_F(MoeFamilyTest, ProblemKeysContainRouteShapeAndPartitionButNotAddressesOrVersions) {
    auto operation = session(makeMoeConfig(*context, ag_config));
    auto input     = tensor(0, {16, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    auto weight    = tensor(1, {8, 1024, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    auto ids       = tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32);
    auto output    = tensor(3, {128, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    auto problem   = [&](FTrainMoeTensor changed_weight) {
        return MoeProblem(
            operation, {MoeTensor(input), MoeTensor(changed_weight), MoeTensor(ids), MoeTensor(output), std::nullopt});
    };
    const auto original   = problem(weight).getProblemKey();
    weight.storage.memory = buffers[4].data();
    weight.version        = 123;
    EXPECT_EQ(original, problem(weight).getProblemKey());
    std::array<int64_t, 8> sizes{10, 12, 14, 16, 18, 20, 22, 16};
    operation->setRankMSizes(128, sizes.data(), sizes.size());
    dimensions[0][0] = 10;
    EXPECT_NE(original, problem(weight).getProblemKey());
}

TEST_F(MoeFamilyTest, StagedFamilySelectsOneRouteAndRetainsSessionAfterSourceHandlesDie) {
    auto operation = session(makeMoeConfig(*context, ag_config));
    std::vector<std::unique_ptr<PrimitiveBase>> primitives;
    {
        const auto& roles = AGGroupGemmFamily::makeRoles();
        Ops ops(roles.pattern, roles.pattern);
        auto args = ops.makeArgs();
        const std::array<FTrainMoeTensor, 4> tensors{
            tensor(0, {16, 1024}, FTRAIN_NUMERIC_TYPE_BF16), tensor(1, {8, 1024, 256}, FTRAIN_NUMERIC_TYPE_BF16),
            tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32), tensor(3, {128, 256}, FTRAIN_NUMERIC_TYPE_BF16)};
        for (size_t index = 0; index < tensors.size(); ++index) {
            args.setOperand<OperandKind::kTensor>(PatternOperandId{roles.ports[index].opaque},
                                                  Tensor(StorageView(tensors[index].storage)));
        }
        AGGroupGemmAttributes attributes{operation};
        attributes.tensors[1].version = 7;
        attributes.tensors[2].version = 11;
        args.setOperation<OperationKind::kAGGroupGemm>(PatternOperationId{roles.operation.opaque},
                                                       std::move(attributes));
        OpsEngine<AGGroupGemmFamily, ExplicitMoeRouteFinder> engine;
        primitives = engine.createPrimitives(args, Constraints(0));
    }
    ASSERT_EQ(primitives.size(), 1);
    EXPECT_STREQ(primitives[0]->getName(), "ag_nn_ring_chunk");
    operation.reset();
    primitives[0]->execute(Resources(nullptr, 0, nullptr));
    EXPECT_EQ(state->executions, 1);
    EXPECT_EQ(state->weight_version, 7);
    EXPECT_EQ(state->routing_version, 11);
    EXPECT_EQ(state->destructions, 0);
    primitives.clear();
    EXPECT_EQ(state->destructions, 1);
}

}
}
