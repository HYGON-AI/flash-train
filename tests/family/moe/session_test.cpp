#include <array>
#include <limits>

#include "../../support/moe_session.hpp"

namespace ftrain {
namespace {

using ftrain_test::MoeSessionTest;

TEST_F(MoeSessionTest, RejectsUnknownAndLegacyRouteNames) {
    ag_config.route = "native_kn_ring";
    EXPECT_THROW(makeMoeConfig(*context, ag_config), Exception);
    rs_config.route = "nk_direct";
    EXPECT_THROW(makeMoeConfig(*context, rs_config), Exception);
}

TEST_F(MoeSessionTest, AcceptsAllCanonicalRoutesWithTheirLayouts) {
    for (const auto* route : {"ag_nn_a2a", "ag_nn_ring", "ag_nn_ring_chunk", "ag_nt_a2a", "ag_nt_a2a_chunk",
                              "ag_nt_ring_chunk", "ag_nt_a2a_chunk_32k", "ag_nt_ring_chunk_32k"}) {
        ag_config.route = route;
        ag_config.k_dim = std::string(route).find("32k") == std::string::npos ? 1024 : 32768;
        auto config     = makeMoeConfig(*context, ag_config);
        EXPECT_EQ(config.isNT(), std::string(route).rfind("ag_nt_", 0) == 0);
    }
    for (const auto* route : {"rs_nn", "rs_nt"}) {
        rs_config.route = route;
        EXPECT_EQ(makeMoeConfig(*context, rs_config).isNT(), std::string(route) == "rs_nt");
    }
}

TEST_F(MoeSessionTest, LargeRoutesCheckKNotM) {
    ag_config.route      = "ag_nt_a2a_chunk_32k";
    ag_config.m_dim      = 32768;
    ag_config.capacity_m = 32768;
    EXPECT_THROW(makeMoeConfig(*context, ag_config), Exception);
    ag_config.m_dim = 128;
    ag_config.k_dim = 32768;
    EXPECT_NO_THROW(makeMoeConfig(*context, ag_config));
}

TEST_F(MoeSessionTest, ValidatesPartitionsAndCapacityBeforeBackendCalls) {
    auto operation = session(makeMoeConfig(*context, ag_config));
    std::array<int64_t, 8> sizes{10, 12, 14, 16, 18, 20, 22, 24};
    EXPECT_NO_THROW(operation->setRankMSizes(136, sizes.data(), sizes.size()));
    EXPECT_EQ(state->updates, 1);
    EXPECT_EQ(operation->getConfig().m_dim, 136);
    EXPECT_THROW(operation->setRankMSizes(135, sizes.data(), sizes.size()), Exception);
    EXPECT_THROW(operation->setRankMSizes(257, nullptr, 0), Exception);
    EXPECT_THROW(operation->setRankMSizes(128, sizes.data(), 7), Exception);
    EXPECT_EQ(operation->getConfig().m_dim, 136);
    EXPECT_EQ(state->updates, 1);
}

TEST_F(MoeSessionTest, OriginalChunkRejectsUnequalPartitionsButLargeChunkAcceptsThem) {
    std::array<int64_t, 8> sizes{10, 12, 14, 16, 18, 20, 22, 24};
    ag_config.route            = "ag_nt_a2a_chunk";
    ag_config.m_dim            = 136;
    ag_config.rank_m_sizes     = sizes.data();
    ag_config.num_rank_m_sizes = sizes.size();
    EXPECT_THROW(makeMoeConfig(*context, ag_config), Exception);
    ag_config.route = "ag_nt_a2a_chunk_32k";
    ag_config.k_dim = 32768;
    EXPECT_NO_THROW(makeMoeConfig(*context, ag_config));
}

TEST_F(MoeSessionTest, BorrowedTensorPreservesIdentityAndVersion) {
    auto value     = tensor(0, {16, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
    value.version  = 17;
    value.identity = buffers[1].data();
    const MoeTensor borrowed(value);
    EXPECT_EQ(borrowed.storage.getMemory(), buffers[0].data());
    EXPECT_EQ(borrowed.identity, buffers[1].data());
    EXPECT_EQ(borrowed.version, 17);
}

TEST_F(MoeSessionTest, RejectsInvalidDimensionsAlignmentAndTopology) {
    auto invalid  = ag_config;
    invalid.m_dim = 0;
    EXPECT_THROW(makeMoeConfig(*context, invalid), Exception);
    invalid            = ag_config;
    invalid.capacity_m = 64;
    EXPECT_THROW(makeMoeConfig(*context, invalid), Exception);
    invalid       = ag_config;
    invalid.n_dim = 255;
    EXPECT_THROW(makeMoeConfig(*context, invalid), Exception);
    invalid       = ag_config;
    invalid.k_dim = 1023;
    EXPECT_THROW(makeMoeConfig(*context, invalid), Exception);
    invalid      = ag_config;
    invalid.topk = 9;
    EXPECT_THROW(makeMoeConfig(*context, invalid), Exception);
    invalid      = ag_config;
    invalid.topk = 2;
    EXPECT_THROW(makeMoeConfig(*context, invalid), Exception);
    invalid.route = "ag_nn_a2a";
    EXPECT_NO_THROW(makeMoeConfig(*context, invalid));
    rs_config.n_chunks = -1;
    EXPECT_THROW(makeMoeConfig(*context, rs_config), Exception);
    auto smaller  = std::make_shared<CollectiveContext>(0, 4, 0, ftrain_test::noopGather, nullptr);
    invalid       = ag_config;
    invalid.route = "ag_nt_ring_chunk";
    EXPECT_THROW(makeMoeConfig(*smaller, invalid), Exception);
}

TEST_F(MoeSessionTest, FailedPartitionUpdatesAreTransactional) {
    auto operation      = session(makeMoeConfig(*context, ag_config));
    const auto original = operation->getConfig().rank_m_sizes;
    std::array<int64_t, 8> sizes{12, 14, 16, 18, 20, 22, 24, 26};
    state->update_failure = true;
    EXPECT_THROW(operation->setRankMSizes(152, sizes.data(), sizes.size()), Exception);
    EXPECT_EQ(operation->getConfig().m_dim, 128);
    EXPECT_EQ(operation->getConfig().rank_m_sizes, original);
    state->update_failure = false;
    sizes[0]              = 0;
    EXPECT_THROW(operation->setRankMSizes(140, sizes.data(), sizes.size()), Exception);
    sizes[0] = -1;
    EXPECT_THROW(operation->setRankMSizes(139, sizes.data(), sizes.size()), Exception);
    EXPECT_EQ(state->updates, 0);
}

TEST_F(MoeSessionTest, ValidatesBothLayoutsRoutingTypesAndRequiredProbabilities) {
    for (const bool is_ag : {true, false}) {
        for (const bool is_nt : {false, true}) {
            ag_config.route = is_nt ? "ag_nt_ring_chunk" : "ag_nn_ring_chunk";
            rs_config.route = is_nt ? "rs_nt" : "rs_nn";
            auto operation  = session(is_ag ? makeMoeConfig(*context, ag_config) : makeMoeConfig(*context, rs_config));
            for (auto type : {FTRAIN_NUMERIC_TYPE_INT32, FTRAIN_NUMERIC_TYPE_INT64}) {
                SCOPED_TRACE(operation->getConfig().route);
                auto valid = bindings(operation, type);
                EXPECT_NO_THROW(operation->validateBindings(valid));
                auto invalid           = valid;
                invalid.weight.storage = StorageView(tensor(1, {8, 256}, FTRAIN_NUMERIC_TYPE_BF16).storage);
                EXPECT_THROW(operation->validateBindings(invalid), Exception);
                valid                       = bindings(operation, type);
                invalid                     = valid;
                auto host                   = tensor(0, {is_ag ? 16 : 128, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
                host.storage.is_host_memory = true;
                invalid.input               = MoeTensor(host);
                EXPECT_THROW(operation->validateBindings(invalid), Exception);
                valid           = bindings(operation, type);
                invalid         = valid;
                invalid.routing = MoeTensor(tensor(2, {128, 1}, FTRAIN_NUMERIC_TYPE_FP32));
                EXPECT_THROW(operation->validateBindings(invalid), Exception);
                valid                     = bindings(operation, type);
                invalid                   = valid;
                auto misaligned           = tensor(0, {is_ag ? 16 : 128, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
                misaligned.storage.memory = reinterpret_cast<char*>(buffers[0].data()) + 1;
                invalid.input             = MoeTensor(misaligned);
                EXPECT_THROW(operation->validateBindings(invalid), Exception);
                valid        = bindings(operation, type);
                invalid      = valid;
                auto strided = tensor(0, {is_ag ? 16 : 128, 1024}, FTRAIN_NUMERIC_TYPE_BF16);
                const std::array<int64_t, 2> strides{2048, 1};
                strided.storage.index_type = FTRAIN_INDEX_TYPE_INVALID;
                strided.storage.strides    = strides.data();
                invalid.input              = MoeTensor(strided);
                EXPECT_THROW(operation->validateBindings(invalid), Exception);
                valid   = bindings(operation, type);
                invalid = valid;
                invalid.expert_weight.reset();
                if (!is_ag) { EXPECT_THROW(operation->validateBindings(invalid), Exception); }
            }
        }
    }
}

}
}
