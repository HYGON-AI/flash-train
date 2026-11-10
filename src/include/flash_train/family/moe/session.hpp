#ifndef FTRAIN_FAMILY_MOE_SESSION_HPP_
#define FTRAIN_FAMILY_MOE_SESSION_HPP_

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "flash_train/flash_train.h"
#include "flash_train/storage_view.hpp"

namespace ftrain {

enum class MoeKind { kAG, kRS };

struct CollectiveContext final {
    CollectiveContext(int32_t rank, int32_t world_size, FTrainDeviceId device, FTrainByteAllGather allgather,
                      void* user_data);

    int32_t rank;
    int32_t world_size;
    FTrainDeviceId device;
    FTrainByteAllGather allgather;
    void* user_data;
};

struct MoeConfig final {
    MoeKind kind;
    int32_t m_dim;
    int32_t capacity_m;
    int32_t n_dim;
    int32_t k_dim;
    int32_t num_experts;
    int32_t topk;
    std::vector<int64_t> rank_m_sizes;
    std::string route;
    int32_t n_chunks;

    bool isNT() const noexcept;
};

MoeConfig makeMoeConfig(const CollectiveContext& context, const FTrainAGGroupGemmConfig& config);
MoeConfig makeMoeConfig(const CollectiveContext& context, const FTrainMoEReduceRSConfig& config);
std::vector<int64_t> validateMoePartition(const CollectiveContext& context, const MoeConfig& config, int32_t m_dim,
                                         const int64_t* rank_m_sizes, uint32_t count);

struct MoeTensor final {
    explicit MoeTensor(const FTrainMoeTensor& tensor);

    StorageView storage;
    const void* identity;
    uint64_t version;
};

struct MoeBindings final {
    MoeTensor input;
    MoeTensor weight;
    MoeTensor routing;
    MoeTensor output;
    std::optional<MoeTensor> expert_weight;
};

class MoeBackend {
  public:
    virtual ~MoeBackend() = default;
    virtual void execute(const MoeBindings& bindings, FTrainStream stream) = 0;
    virtual void setRankMSizes(int32_t m_dim, const std::vector<int64_t>& sizes) = 0;
    virtual void synchronize() = 0;
};

class MoeSession final {
  public:
    MoeSession(std::shared_ptr<CollectiveContext> context, MoeConfig config, std::unique_ptr<MoeBackend> backend);
    void execute(MoeBindings bindings, FTrainStream stream);
    void setRankMSizes(int32_t m_dim, const int64_t* sizes, uint32_t count);
    void synchronize();
    void validateBindings(const MoeBindings& bindings) const;

    const MoeConfig& getConfig() const noexcept { return config_; }
    const CollectiveContext& getContext() const noexcept { return *context_; }

  private:
    std::shared_ptr<CollectiveContext> context_;
    MoeConfig config_;
    std::unique_ptr<MoeBackend> backend_;
    std::mutex mutex_;
};

}

struct FTrainCollectiveContextStruct final {
    std::shared_ptr<ftrain::CollectiveContext> context;
};

struct FTrainAGGroupGemmStruct final {
    std::shared_ptr<ftrain::MoeSession> session;
};

struct FTrainMoEReduceRSStruct final {
    std::shared_ptr<ftrain::MoeSession> session;
};

#endif
