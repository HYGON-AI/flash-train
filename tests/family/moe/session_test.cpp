#include <array>
#include <limits>

#include "../../support/moe_session.hpp"

namespace ftrain {
namespace {

using ftrain_test::MoeSessionTest;





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



}
}
