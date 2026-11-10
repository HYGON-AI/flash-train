#ifndef FTRAIN_TESTS_MOE_SESSION_HPP_
#define FTRAIN_TESTS_MOE_SESSION_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/error.hpp"
#include "flash_train/family/moe/session.hpp"

namespace ftrain_test {

using ftrain::CollectiveContext;
using ftrain::Exception;
using ftrain::MoeBackend;
using ftrain::MoeBindings;
using ftrain::MoeConfig;
using ftrain::MoeKind;
using ftrain::MoeSession;
using ftrain::MoeTensor;

inline FTrainStatus noopGather(void*, const void*, void*, uint64_t) { return FTRAIN_STATUS_SUCCESS; }

struct BackendState final {
    bool update_failure         = false;
    int executions              = 0;
    int updates                 = 0;
    int synchronizations        = 0;
    int destructions            = 0;
    uint64_t weight_version     = 0;
    uint64_t routing_version    = 0;
    void* output                = nullptr;
    FTrainStream stream         = nullptr;
    FTrainStatus failure_status = FTRAIN_STATUS_SUCCESS;
    std::optional<MoeBindings> bindings;
};

class MockMoeBackend final : public MoeBackend {
  public:
    explicit MockMoeBackend(std::shared_ptr<BackendState> state) : state_(std::move(state)) {}
    ~MockMoeBackend() override { ++state_->destructions; }

    void execute(const MoeBindings& bindings, FTrainStream stream) override {
        if (state_->failure_status != FTRAIN_STATUS_SUCCESS) {
            throw Exception(state_->failure_status, "mock MoE execution failed: 50%%");
        }
        ++state_->executions;
        state_->weight_version  = bindings.weight.version;
        state_->routing_version = bindings.routing.version;
        state_->output          = bindings.output.storage.getMemory();
        state_->stream          = stream;
        state_->bindings        = bindings;
    }

    void setRankMSizes(int32_t, const std::vector<int64_t>&) override {
        if (state_->update_failure) { throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "mock partition update failed"); }
        ++state_->updates;
    }
    void synchronize() override { ++state_->synchronizations; }

  private:
    std::shared_ptr<BackendState> state_;
};

class MoeSessionTest : public ::testing::Test {
  protected:
    void SetUp() override {
        context = std::make_shared<CollectiveContext>(0, 8, 0, noopGather, nullptr);
        ASSERT_EQ(ftrainAGGroupGemmConfigInit(&ag_config), FTRAIN_STATUS_SUCCESS);
        ag_config.m_dim       = 128;
        ag_config.capacity_m  = 256;
        ag_config.n_dim       = 256;
        ag_config.k_dim       = 1024;
        ag_config.num_experts = 8;
        ASSERT_EQ(ftrainMoEReduceRSConfigInit(&rs_config), FTRAIN_STATUS_SUCCESS);
        rs_config.m_dim       = 128;
        rs_config.capacity_m  = 256;
        rs_config.n_dim       = 256;
        rs_config.k_dim       = 1024;
        rs_config.num_experts = 8;
        state                 = std::make_shared<BackendState>();
    }

    std::shared_ptr<MoeSession> session(MoeConfig config) {
        return std::make_shared<MoeSession>(context, std::move(config), std::make_unique<MockMoeBackend>(state));
    }

    FTrainMoeTensor tensor(size_t slot, std::vector<int64_t> dims, FTrainNumericType type) {
        dimensions[slot] = std::move(dims);
        return {
            {buffers[slot].data(), dimensions[slot].data(), nullptr, static_cast<uint8_t>(dimensions[slot].size()),
             type, FTRAIN_INDEX_TYPE_CONTINUOUS, false},
            buffers[slot].data(),
            0
        };
    }

    MoeBindings bindings(const std::shared_ptr<MoeSession>& operation,
                         FTrainNumericType id_type = FTRAIN_NUMERIC_TYPE_INT32) {
        const auto& config = operation->getConfig();
        const bool is_ag   = config.kind == MoeKind::kAG;
        const auto local_m = config.rank_m_sizes[operation->getContext().rank];
        return {MoeTensor(tensor(0, {is_ag ? local_m : int64_t(config.m_dim) * config.topk, config.k_dim},
                                 FTRAIN_NUMERIC_TYPE_BF16)),
                MoeTensor(tensor(1,
                                 {config.num_experts, config.isNT() ? config.n_dim : config.k_dim,
                                  config.isNT() ? config.k_dim : config.n_dim},
                                 FTRAIN_NUMERIC_TYPE_BF16)),
                MoeTensor(tensor(2, {config.m_dim, config.topk}, id_type)),
                MoeTensor(tensor(3, {is_ag ? int64_t(config.m_dim) * config.topk : local_m, config.n_dim},
                                 FTRAIN_NUMERIC_TYPE_BF16)),
                is_ag ? std::nullopt
                      : std::optional<MoeTensor>(
                            MoeTensor(tensor(4, {config.m_dim, config.topk}, FTRAIN_NUMERIC_TYPE_BF16)))};
    }

    std::shared_ptr<CollectiveContext> context;
    std::shared_ptr<BackendState> state;
    FTrainAGGroupGemmConfig ag_config{};
    FTrainMoEReduceRSConfig rs_config{};
    alignas(8) std::array<std::array<uint64_t, 4>, 5> buffers{};
    std::array<std::vector<int64_t>, 5> dimensions;
};

}

#endif
