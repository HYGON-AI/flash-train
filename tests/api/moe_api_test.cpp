#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/api.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/family/moe/family.hpp"
#include "flash_train/flash_train.h"
#include "moe_staged_fixture.hpp"

namespace ftrain {
namespace {

using ftrain_test::BackendState;
using ftrain_test::MoeApiTest;
using ftrain_test::noopGather;

TEST_F(MoeApiTest, DefaultsAreExplicitNNRoutes) {
    EXPECT_STREQ(ag_config.route, "ag_nn_ring_chunk");
    EXPECT_STREQ(rs_config.route, "rs_nn");
    EXPECT_EQ(ag_config.topk, 1);
    EXPECT_EQ(rs_config.topk, 1);
    EXPECT_EQ(ftrainAGGroupGemmConfigInit(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainMoEReduceRSConfigInit(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
}

TEST_F(MoeApiTest, ContextValidationDoesNotOverwriteFailedOutputs) {
    auto sentinel = reinterpret_cast<FTrainCollectiveContext>(uintptr_t{8});
    auto result   = sentinel;
    EXPECT_EQ(ftrainCollectiveContextCreate(&result, 8, 8, 0, noopGather, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(result, sentinel);
    EXPECT_EQ(ftrainCollectiveContextCreate(&result, 0, 8, 0, nullptr, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(result, sentinel);
    EXPECT_EQ(ftrainCollectiveContextCreate(nullptr, 0, 8, 0, noopGather, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainCollectiveContextCreate(&result, 0, 8, 0, noopGather, nullptr), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainCollectiveContextDestroy(result), FTRAIN_STATUS_SUCCESS);
}

TEST_F(MoeApiTest, AGConvenienceForwardsExternalBuffersStreamIdentitiesAndVersions) {
    FTrainAGGroupGemmStruct operation{session(makeMoeConfig(*context, ag_config))};
    auto input      = tensor(0, {16, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    auto weight     = tensor(1, {8, 1024, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    auto ids        = tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32);
    auto output     = tensor(3, {128, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    weight.version  = 4;
    ids.version     = 9;
    input.identity  = buffers[4].data();
    input.version   = 3;
    weight.identity = buffers[0].data();
    ids.identity    = buffers[1].data();
    output.identity = buffers[2].data();
    output.version  = 5;
    auto stream     = reinterpret_cast<FTrainStream>(uintptr_t{16});
    ASSERT_EQ(ftrainAGGroupGemmExecute(&operation, input, weight, ids, output, stream), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(state->executions, 1);
    EXPECT_EQ(state->weight_version, 4);
    EXPECT_EQ(state->routing_version, 9);
    EXPECT_EQ(state->stream, stream);
    EXPECT_EQ(state->output, output.storage.memory);
    ASSERT_TRUE(state->bindings.has_value());
    EXPECT_EQ(state->bindings->input.identity, input.identity);
    EXPECT_EQ(state->bindings->input.version, input.version);
    EXPECT_EQ(state->bindings->weight.identity, weight.identity);
    EXPECT_EQ(state->bindings->routing.identity, ids.identity);
    EXPECT_EQ(state->bindings->output.identity, output.identity);
    EXPECT_EQ(state->bindings->output.version, output.version);
    EXPECT_EQ(state->synchronizations, 0);
    output.storage.is_host_memory = true;
    EXPECT_EQ(ftrainAGGroupGemmExecute(&operation, input, weight, ids, output, stream), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(state->executions, 1);
    EXPECT_NE(std::string(ftrainGetLastMessage()).find("output"), std::string::npos);
}

TEST_F(MoeApiTest, RSRequiresProbabilitiesAndAcceptsRawNTWeights) {
    rs_config.route = "rs_nt";
    rs_config.topk  = 2;
    FTrainMoEReduceRSStruct operation{session(makeMoeConfig(*context, rs_config))};
    auto input             = tensor(0, {256, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    auto weight            = tensor(1, {8, 256, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    auto ids               = tensor(2, {128, 2}, FTRAIN_NUMERIC_TYPE_INT64);
    auto output            = tensor(3, {16, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    auto probabilities     = tensor(4, {128, 2}, FTRAIN_NUMERIC_TYPE_BF16);
    input.version          = 3;
    weight.identity        = buffers[0].data();
    weight.version         = 4;
    ids.identity           = buffers[1].data();
    ids.version            = 5;
    probabilities.identity = buffers[2].data();
    probabilities.version  = 6;
    output.version         = 7;
    auto stream            = reinterpret_cast<FTrainStream>(uintptr_t{16});
    ASSERT_EQ(ftrainMoEReduceRSExecute(&operation, input, weight, ids, probabilities, output, stream),
              FTRAIN_STATUS_SUCCESS);
    ASSERT_TRUE(state->bindings.has_value());
    ASSERT_TRUE(state->bindings->expert_weight.has_value());
    EXPECT_EQ(state->bindings->input.identity, input.storage.memory);
    EXPECT_EQ(state->bindings->input.version, input.version);
    EXPECT_EQ(state->bindings->weight.identity, weight.identity);
    EXPECT_EQ(state->bindings->weight.version, weight.version);
    EXPECT_EQ(state->bindings->routing.identity, ids.identity);
    EXPECT_EQ(state->bindings->routing.version, ids.version);
    EXPECT_EQ(state->bindings->expert_weight->identity, probabilities.identity);
    EXPECT_EQ(state->bindings->expert_weight->version, probabilities.version);
    EXPECT_EQ(state->bindings->output.version, output.version);
    EXPECT_EQ(state->stream, stream);
    EXPECT_EQ(state->synchronizations, 0);
    probabilities.storage.numeric_type = FTRAIN_NUMERIC_TYPE_FP32;
    EXPECT_EQ(ftrainMoEReduceRSExecute(&operation, input, weight, ids, probabilities, output, nullptr),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(state->executions, 1);
}

TEST_F(MoeApiTest, ConvenienceEntriesSelectEveryCanonicalRouteWithRawWeights) {
    for (const auto* route : {"ag_nn_a2a", "ag_nn_ring", "ag_nn_ring_chunk", "ag_nt_a2a", "ag_nt_a2a_chunk",
                              "ag_nt_ring_chunk", "ag_nt_a2a_chunk_32k", "ag_nt_ring_chunk_32k"}) {
        SCOPED_TRACE(route);
        ag_config.route = route;
        ag_config.k_dim = std::string(route).find("32k") == std::string::npos ? 1024 : 32768;
        auto config     = makeMoeConfig(*context, ag_config);
        const bool nt   = config.isNT();
        FTrainAGGroupGemmStruct operation{session(std::move(config))};
        auto input  = tensor(0, {16, ag_config.k_dim}, FTRAIN_NUMERIC_TYPE_BF16);
        auto weight = tensor(1, {8, nt ? 256 : ag_config.k_dim, nt ? ag_config.k_dim : 256}, FTRAIN_NUMERIC_TYPE_BF16);
        auto ids    = tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32);
        auto output = tensor(3, {128, 256}, FTRAIN_NUMERIC_TYPE_BF16);
        ASSERT_EQ(ftrainAGGroupGemmExecute(&operation, input, weight, ids, output, nullptr), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(state->bindings->weight.storage.getMemory(), weight.storage.memory);
    }
    for (const auto* route : {"rs_nn", "rs_nt"}) {
        SCOPED_TRACE(route);
        rs_config.route = route;
        auto config     = makeMoeConfig(*context, rs_config);
        const bool nt   = config.isNT();
        FTrainMoEReduceRSStruct operation{session(std::move(config))};
        auto input         = tensor(0, {128, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
        auto weight        = tensor(1, {8, nt ? 256 : 1024, nt ? 1024 : 256}, FTRAIN_NUMERIC_TYPE_BF16);
        auto ids           = tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32);
        auto output        = tensor(3, {16, 256}, FTRAIN_NUMERIC_TYPE_BF16);
        auto probabilities = tensor(4, {128, 1}, FTRAIN_NUMERIC_TYPE_BF16);
        ASSERT_EQ(ftrainMoEReduceRSExecute(&operation, input, weight, ids, probabilities, output, nullptr),
                  FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(state->bindings->weight.storage.getMemory(), weight.storage.memory);
    }
    EXPECT_EQ(state->executions, 10);
    EXPECT_EQ(state->synchronizations, 0);
}

TEST_F(MoeApiTest, ConvenienceEntriesRejectOtherDevicesThroughPlanSelection) {
    context->device = getCurrentDeviceId() + 1;
    FTrainAGGroupGemmStruct ag{session(makeMoeConfig(*context, ag_config))};
    auto input  = tensor(0, {16, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    auto weight = tensor(1, {8, 1024, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    auto ids    = tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32);
    auto output = tensor(3, {128, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    EXPECT_EQ(ftrainAGGroupGemmExecute(&ag, input, weight, ids, output, nullptr), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_NE(std::string(ftrainGetLastMessage()).find("different HIP device"), std::string::npos);
    FTrainMoEReduceRSStruct rs{session(makeMoeConfig(*context, rs_config))};
    input              = tensor(0, {128, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    output             = tensor(3, {16, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    auto probabilities = tensor(4, {128, 1}, FTRAIN_NUMERIC_TYPE_BF16);
    EXPECT_EQ(ftrainMoEReduceRSExecute(&rs, input, weight, ids, probabilities, output, nullptr),
              FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_NE(std::string(ftrainGetLastMessage()).find("different HIP device"), std::string::npos);
    EXPECT_EQ(state->executions, 0);
}

TEST_F(MoeApiTest, AGConvenienceRebindsBuffersVersionsAndRaggedMWithoutSynchronizing) {
    FTrainAGGroupGemmStruct operation{session(makeMoeConfig(*context, ag_config))};
    auto input  = tensor(0, {16, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    auto weight = tensor(1, {8, 1024, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    auto ids    = tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32);
    auto output = tensor(3, {128, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    ASSERT_EQ(ftrainAGGroupGemmExecute(&operation, input, weight, ids, output, nullptr), FTRAIN_STATUS_SUCCESS);
    const std::array<int64_t, 8> sizes{10, 12, 14, 16, 18, 20, 22, 24};
    ASSERT_EQ(ftrainAGGroupGemmSetRankMSizes(&operation, 136, sizes.data(), sizes.size()), FTRAIN_STATUS_SUCCESS);
    input                 = tensor(0, {10, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    ids                   = tensor(2, {136, 1}, FTRAIN_NUMERIC_TYPE_INT64);
    output                = tensor(3, {136, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    output.storage.memory = buffers[4].data();
    output.identity       = nullptr;
    weight.version        = 7;
    ids.version           = 11;
    ASSERT_EQ(ftrainAGGroupGemmExecute(&operation, input, weight, ids, output, nullptr), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(state->executions, 2);
    EXPECT_EQ(state->updates, 1);
    EXPECT_EQ(state->weight_version, 7);
    EXPECT_EQ(state->routing_version, 11);
    EXPECT_EQ(state->output, buffers[4].data());
    EXPECT_EQ(state->bindings->output.identity, buffers[4].data());
    EXPECT_EQ(state->bindings->input.storage.getDims()[0], 10);
    EXPECT_EQ(state->synchronizations, 0);
}

TEST_F(MoeApiTest, ConvenienceEntriesPreserveExecutionErrorsThroughPlanCleanup) {
    state->failure_status = FTRAIN_STATUS_INTERNAL_ERROR;
    FTrainAGGroupGemmStruct ag{session(makeMoeConfig(*context, ag_config))};
    auto input  = tensor(0, {16, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    auto weight = tensor(1, {8, 1024, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    auto ids    = tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32);
    auto output = tensor(3, {128, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    EXPECT_EQ(ftrainAGGroupGemmExecute(&ag, input, weight, ids, output, nullptr), FTRAIN_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(ftrainGetLastMessage(), "mock MoE execution failed: 50%");
    FTrainMoEReduceRSStruct rs{session(makeMoeConfig(*context, rs_config))};
    input              = tensor(0, {128, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    output             = tensor(3, {16, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    auto probabilities = tensor(4, {128, 1}, FTRAIN_NUMERIC_TYPE_BF16);
    EXPECT_EQ(ftrainMoEReduceRSExecute(&rs, input, weight, ids, probabilities, output, nullptr),
              FTRAIN_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_INTERNAL_ERROR);
    EXPECT_STREQ(ftrainGetLastMessage(), "mock MoE execution failed: 50%");
    EXPECT_EQ(state->executions, 0);
    EXPECT_EQ(state->synchronizations, 0);
    state->failure_status = FTRAIN_STATUS_SUCCESS;
    EXPECT_EQ(ftrainMoEReduceRSExecute(&rs, input, weight, ids, probabilities, output, nullptr), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(ftrainGetLastStatus(), FTRAIN_STATUS_SUCCESS);
    EXPECT_STREQ(ftrainGetLastMessage(), "");
}

TEST_F(MoeApiTest, PublicStagedAGAndRSRetainSessionsAndVersionedBindings) {
    for (const bool is_ag : {true, false}) {
        SCOPED_TRACE(is_ag ? "AG" : "RS");
        state = std::make_shared<BackendState>();
        FTrainAGGroupGemmStruct ag;
        FTrainMoEReduceRSStruct rs;
        if (is_ag) {
            ag.session = session(makeMoeConfig(*context, ag_config));
        } else {
            rs.session = session(makeMoeConfig(*context, rs_config));
        }
        std::vector<FTrainMoeTensor> tensors{tensor(0, {is_ag ? 16 : 128, 1024}, FTRAIN_NUMERIC_TYPE_BF16),
                                             tensor(1, {8, 1024, 256}, FTRAIN_NUMERIC_TYPE_BF16),
                                             tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32)};
        if (!is_ag) { tensors.push_back(tensor(3, {128, 1}, FTRAIN_NUMERIC_TYPE_BF16)); }
        tensors.push_back(tensor(4, {is_ag ? 128 : 16, 256}, FTRAIN_NUMERIC_TYPE_BF16));
        tensors[1].version = 13;
        tensors[2].version = 17;
        auto plan          = makePlan(is_ag ? &ag : nullptr, is_ag ? nullptr : &rs, tensors);
        ASSERT_NE(plan, nullptr);
        destroySources();
        ag.session.reset();
        rs.session.reset();
        uint64_t count     = 0;
        const char* name   = nullptr;
        uint64_t workspace = 1;
        ASSERT_EQ(ftrainPlanGetNumPrimitives(plan, &count), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPlanGetPrimitiveName(plan, 0, &name), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &workspace), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(count, 1);
        EXPECT_EQ(workspace, 0);
        EXPECT_STREQ(name, is_ag ? "ag_nn_ring_chunk" : "rs_nn");
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, nullptr), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(state->executions, 1);
        EXPECT_EQ(state->weight_version, 13);
        EXPECT_EQ(state->routing_version, 17);
        EXPECT_EQ(state->bindings->weight.identity, tensors[1].storage.memory);
        EXPECT_EQ(state->bindings->routing.identity, tensors[2].storage.memory);
        EXPECT_EQ(state->destructions, 0);
        ASSERT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
        plans.pop_back();
        EXPECT_EQ(state->destructions, 1);
    }
}

TEST_F(MoeApiTest, StagedPlansKeepIndependentAddressAndVersionSnapshots) {
    FTrainAGGroupGemmStruct ag{session(makeMoeConfig(*context, ag_config))};
    std::vector<FTrainMoeTensor> tensors{
        tensor(0, {16, 1024}, FTRAIN_NUMERIC_TYPE_BF16), tensor(1, {8, 1024, 256}, FTRAIN_NUMERIC_TYPE_BF16),
        tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32), tensor(3, {128, 256}, FTRAIN_NUMERIC_TYPE_BF16)};
    tensors[1].version = 2;
    tensors[2].version = 3;
    const auto first   = makePlan(&ag, nullptr, tensors);
    ASSERT_NE(first, nullptr);
    tensors[1].version        = 7;
    tensors[2].version        = 9;
    tensors[3].storage.memory = buffers[4].data();
    const auto second         = makePlan(&ag, nullptr, tensors);
    ASSERT_NE(second, nullptr);
    destroySources();
    for (auto plan : {second, first, second}) {
        ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, nullptr), FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(state->weight_version, plan == first ? 2 : 7);
        EXPECT_EQ(state->routing_version, plan == first ? 3 : 9);
        EXPECT_EQ(state->output, plan == first ? buffers[3].data() : buffers[4].data());
    }
    EXPECT_EQ(state->executions, 3);
    EXPECT_EQ(state->synchronizations, 0);
}

TEST_F(MoeApiTest, StagedSettersRejectWrongKindsAndPreserveExistingAttributes) {
    FTrainAGGroupGemmStruct ag{session(makeMoeConfig(*context, ag_config))};
    FTrainMoEReduceRSStruct rs{session(makeMoeConfig(*context, rs_config))};
    for (const bool is_ag : {true, false}) {
        auto operation    = is_ag ? ag.session : rs.session;
        const auto values = bindings(operation);
        auto publicTensor = [](const MoeTensor& value) {
            const auto& storage = value.storage;
            return FTrainMoeTensor{
                {storage.getMemory(), storage.getDims().data(), nullptr, static_cast<uint8_t>(storage.getDims().size()),
                 storage.getNumericType(), storage.getIndexType(), storage.isHostMemory()},
                value.identity,
                value.version
            };
        };
        std::vector<FTrainMoeTensor> tensors{publicTensor(values.input), publicTensor(values.weight),
                                             publicTensor(values.routing)};
        if (!is_ag) { tensors.push_back(publicTensor(*values.expert_weight)); }
        tensors.push_back(publicTensor(values.output));
        const auto plan = makePlan(is_ag ? &ag : nullptr, is_ag ? nullptr : &rs, tensors);
        ASSERT_NE(plan, nullptr);
        const auto args = arguments.back();
        if (is_ag) {
            EXPECT_EQ(ftrainArgsSetAGGroupGemm(args, {0}, &ag, 21, 22), FTRAIN_STATUS_SUCCESS);
            EXPECT_EQ(ftrainArgsSetAGGroupGemm(args, {0}, nullptr, 31, 32), FTRAIN_STATUS_INVALID_ARGUMENT);
            EXPECT_EQ(ftrainArgsSetAGGroupGemm(nullptr, {0}, &ag, 31, 32), FTRAIN_STATUS_INVALID_ARGUMENT);
            EXPECT_EQ(ftrainArgsSetMoEReduceRS(args, {0}, &rs, 31, 32), FTRAIN_STATUS_INVALID_ARGUMENT);
        } else {
            EXPECT_EQ(ftrainArgsSetMoEReduceRS(args, {0}, &rs, 21, 22), FTRAIN_STATUS_SUCCESS);
            EXPECT_EQ(ftrainArgsSetMoEReduceRS(args, {0}, nullptr, 31, 32), FTRAIN_STATUS_INVALID_ARGUMENT);
            EXPECT_EQ(ftrainArgsSetMoEReduceRS(nullptr, {0}, &rs, 31, 32), FTRAIN_STATUS_INVALID_ARGUMENT);
            EXPECT_EQ(ftrainArgsSetAGGroupGemm(args, {0}, &ag, 31, 32), FTRAIN_STATUS_INVALID_ARGUMENT);
        }
        const auto& attributes = *args->args.getOpArgument(PatternOperationId{0});
        std::visit(
            [&](const auto& value) {
                using Value = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Value, AGGroupGemmAttributes> ||
                              std::is_same_v<Value, MoEReduceRSAttributes>) {
                    EXPECT_EQ(value.session, operation);
                    EXPECT_EQ(value.tensors[1].version, 21);
                    EXPECT_EQ(value.tensors[2].version, 22);
                } else {
                    ADD_FAILURE() << "MoE attribute family changed after a rejected setter";
                }
            },
            attributes);
        EXPECT_EQ(state->executions, 0);
    }
}

TEST_F(MoeApiTest, DisabledBackendReportsUnsupportedWithoutChangingOutput) {
    bool available = false;
    ASSERT_EQ(ftrainMoeIsAvailable(&available), FTRAIN_STATUS_SUCCESS);
    if (available) { GTEST_SKIP() << "Requires a build without the MoE backend"; }
    FTrainCollectiveContextStruct group{context};
    auto sentinel = reinterpret_cast<FTrainAGGroupGemm>(uintptr_t{8});
    auto result   = sentinel;
    EXPECT_EQ(ftrainAGGroupGemmCreate(&result, &group, &ag_config, nullptr), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_EQ(result, sentinel);
}

TEST_F(MoeApiTest, NullHandlesReportInvalidArguments) {
    const FTrainMoeTensor empty{};
    EXPECT_EQ(ftrainAGGroupGemmExecute(nullptr, empty, empty, empty, empty, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainMoEReduceRSExecute(nullptr, empty, empty, empty, empty, empty, nullptr),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    FTrainAGGroupGemmStruct ag{};
    FTrainMoEReduceRSStruct rs{};
    EXPECT_EQ(ftrainAGGroupGemmExecute(&ag, empty, empty, empty, empty, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainMoEReduceRSExecute(&rs, empty, empty, empty, empty, empty, nullptr),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainAGGroupGemmSynchronize(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainMoEReduceRSSynchronize(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainAGGroupGemmDestroy(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainMoEReduceRSDestroy(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainCollectiveContextDestroy(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainMoeIsAvailable(nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
}

}
}
