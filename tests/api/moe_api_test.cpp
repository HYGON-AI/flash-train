#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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
    FTrainCollectiveContextStruct operation{context, {session(makeMoeConfig(*context, ag_config))}};
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
    ASSERT_EQ(ftrainAGGroupGemm(&operation, &ag_config, input, weight, ids, output, stream), FTRAIN_STATUS_SUCCESS);
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
    EXPECT_EQ(ftrainAGGroupGemm(&operation, &ag_config, input, weight, ids, output, stream), FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(state->executions, 1);
    EXPECT_NE(std::string(ftrainGetLastMessage()).find("output"), std::string::npos);
}







TEST_F(MoeApiTest, AGConvenienceRebindsBuffersVersionsAndRaggedMWithoutSynchronizing) {
    FTrainCollectiveContextStruct operation{context, {session(makeMoeConfig(*context, ag_config))}};
    auto input  = tensor(0, {16, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    auto weight = tensor(1, {8, 1024, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    auto ids    = tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32);
    auto output = tensor(3, {128, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    ASSERT_EQ(ftrainAGGroupGemm(&operation, &ag_config, input, weight, ids, output, nullptr), FTRAIN_STATUS_SUCCESS);
    const std::array<int64_t, 8> sizes{10, 12, 14, 16, 18, 20, 22, 24};
    ag_config.m_dim = 136;
    ag_config.rank_m_sizes = sizes.data();
    ag_config.num_rank_m_sizes = sizes.size();
    input                 = tensor(0, {10, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    ids                   = tensor(2, {136, 1}, FTRAIN_NUMERIC_TYPE_INT64);
    output                = tensor(3, {136, 256}, FTRAIN_NUMERIC_TYPE_BF16);
    output.storage.memory = buffers[4].data();
    output.identity       = nullptr;
    weight.version        = 7;
    ids.version           = 11;
    ASSERT_EQ(ftrainAGGroupGemm(&operation, &ag_config, input, weight, ids, output, nullptr), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(state->executions, 2);
    EXPECT_EQ(state->updates, 1);
    EXPECT_EQ(state->weight_version, 7);
    EXPECT_EQ(state->routing_version, 11);
    EXPECT_EQ(state->output, buffers[4].data());
    EXPECT_EQ(state->bindings->output.identity, buffers[4].data());
    EXPECT_EQ(state->bindings->input.storage.getDims()[0], 10);
    EXPECT_EQ(state->synchronizations, 0);
}





TEST_F(MoeApiTest, StagedPlansKeepIndependentAddressAndVersionSnapshots) {
    FTrainCollectiveContextStruct ag{context, {session(makeMoeConfig(*context, ag_config))}};
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



TEST_F(MoeApiTest, DisabledBackendReportsUnsupportedWithoutChangingOutput) {
    bool available = false;
    ASSERT_EQ(ftrainMoeIsAvailable(&available), FTRAIN_STATUS_SUCCESS);
    if (available) { GTEST_SKIP() << "Requires a build without the MoE backend"; }
    FTrainCollectiveContextStruct group{context};
    const FTrainMoeTensor empty{};
    EXPECT_EQ(ftrainAGGroupGemm(&group, &ag_config, empty, empty, empty, empty, nullptr), FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_TRUE(group.sessions.empty());
    EXPECT_EQ(state->executions, 0);
}





TEST_F(MoeApiTest, FailedDynamicConfigurationPreservesCachedSession) {
    auto operation = session(makeMoeConfig(*context, ag_config));
    FTrainCollectiveContextStruct group{context, {operation}};
    const std::array<int64_t, 8> sizes{10, 12, 14, 16, 18, 20, 22, 24};
    ag_config.m_dim = 136;
    ag_config.rank_m_sizes = sizes.data();
    ag_config.num_rank_m_sizes = sizes.size();
    state->update_failure = true;
    EXPECT_THROW(group.getSession(makeMoeConfig(*context, ag_config)), Exception);
    EXPECT_EQ(operation->getConfig().m_dim, 128);
    EXPECT_EQ(group.sessions.size(), 1);
    state->update_failure = false;
    EXPECT_EQ(group.getSession(makeMoeConfig(*context, ag_config)), operation);
    EXPECT_EQ(operation->getConfig().m_dim, 136);
    EXPECT_EQ(state->updates, 1);
    EXPECT_EQ(group.getSession(makeMoeConfig(*context, ag_config)), operation);
    EXPECT_EQ(state->updates, 1);
}



TEST_F(MoeApiTest, PlanRetainsSessionAfterPublicContextDestruction) {
    auto group = new FTrainCollectiveContextStruct{context, {session(makeMoeConfig(*context, ag_config))}};
    std::vector<FTrainMoeTensor> tensors{
        tensor(0, {16, 1024}, FTRAIN_NUMERIC_TYPE_BF16), tensor(1, {8, 1024, 256}, FTRAIN_NUMERIC_TYPE_BF16),
        tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_INT32), tensor(3, {128, 256}, FTRAIN_NUMERIC_TYPE_BF16)};
    auto plan = makePlan(group, nullptr, tensors);
    ASSERT_NE(plan, nullptr);
    destroySources();
    ASSERT_EQ(ftrainCollectiveContextDestroy(group), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(state->synchronizations, 1);
    EXPECT_EQ(state->destructions, 0);
    ASSERT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, nullptr), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(state->executions, 1);
    ASSERT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS);
    plans.pop_back();
    EXPECT_EQ(state->destructions, 1);
}

TEST_F(MoeApiTest, ConcurrentContextConfigurationIsRejected) {
    FTrainCollectiveContextStruct group{context, {session(makeMoeConfig(*context, ag_config))}};
    std::lock_guard<std::mutex> lock(group.mutex);
    std::thread caller([&] {
        EXPECT_THROW(group.getSession(makeMoeConfig(*context, ag_config)), Exception);
        EXPECT_EQ(ftrainCollectiveContextSynchronize(&group), FTRAIN_STATUS_INVALID_ARGUMENT);
    });
    caller.join();
    EXPECT_EQ(group.sessions.size(), 1);
    EXPECT_EQ(state->updates, 0);
    EXPECT_EQ(state->synchronizations, 0);
}

}
}
