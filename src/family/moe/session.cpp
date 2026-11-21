#include "flash_train/family/moe/session.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

#include "flash_train/error.hpp"

namespace ftrain {
namespace {

constexpr std::array<const char*, 8> ag_routes{
    "ag_nn_a2a", "ag_nn_ring", "ag_nn_ring_chunk", "ag_nt_a2a", "ag_nt_a2a_chunk", "ag_nt_ring_chunk",
    "ag_nt_a2a_chunk_32k", "ag_nt_ring_chunk_32k"};

void validateConfig(const CollectiveContext& context, MoeConfig& config, const int64_t* sizes, uint32_t count) {
    if (config.capacity_m == 0) { config.capacity_m = config.m_dim; }
    if (config.m_dim <= 0 || config.capacity_m < config.m_dim || config.n_dim <= 0 || config.k_dim <= 0 ||
        config.num_experts <= 0 || config.topk <= 0 || config.topk > config.num_experts || config.n_chunks < 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Invalid MoE dimensions, capacity, topk or n_chunks");
    }
    if (config.capacity_m > std::numeric_limits<int32_t>::max() / config.topk) {
        throw Exception(FTRAIN_STATUS_OVERFLOW, "MoE capacity_m * topk exceeds int32 range");
    }
    if (config.kind == MoeKind::kAG) {
        if (std::find(ag_routes.begin(), ag_routes.end(), config.route) == ag_routes.end()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Unknown AG route: %s", config.route.c_str());
        }
        if (config.n_dim % 256 != 0 || config.k_dim % 64 != 0) {
            throw Exception(FTRAIN_STATUS_UNSUPPORTED, "AG native routes require N_local%%256=0 and K%%64=0");
        }
        if (config.route != "ag_nn_a2a" && config.topk != 1) {
            throw Exception(FTRAIN_STATUS_UNSUPPORTED, "This AG route requires expanded input with topk=1");
        }
        if (config.route.find("_32k") != std::string::npos && config.k_dim != 32768) {
            throw Exception(FTRAIN_STATUS_UNSUPPORTED, "AG _32k routes require K=32768, not M=32768");
        }
        if (config.route.rfind("ag_nt_", 0) == 0 && config.route != "ag_nt_a2a" && context.world_size != 8) {
            throw Exception(FTRAIN_STATUS_UNSUPPORTED, "AG NT chunk routes require ETP8");
        }
        if (config.route != "ag_nn_a2a" && context.world_size != 2 && context.world_size != 4 &&
            context.world_size != 8) {
            throw Exception(FTRAIN_STATUS_UNSUPPORTED, "AG native routes require ETP2/4/8");
        }
    } else if (config.route != "rs_nn" && config.route != "rs_nt") {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Unknown RS route: %s", config.route.c_str());
    }
    if (uint64_t(config.num_experts) >
        std::numeric_limits<uint64_t>::max() / 2 / uint64_t(config.n_dim) / uint64_t(config.k_dim)) {
        throw Exception(FTRAIN_STATUS_OVERFLOW, "MoE weight storage size overflows");
    }
    config.rank_m_sizes = validateMoePartition(context, config, config.m_dim, sizes, count);
}

void validateTensor(const MoeTensor& tensor, const std::vector<int64_t>& dims, FTrainNumericType type,
                    const char* name) {
    const auto& storage = tensor.storage;
    const auto alignment = type == FTRAIN_NUMERIC_TYPE_INT64 ? 8U : type == FTRAIN_NUMERIC_TYPE_INT32 ? 4U : 2U;
    if (storage.getMemory() == nullptr || storage.isHostMemory() ||
        storage.getIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS || storage.getNumericType() != type ||
        storage.getDims() != dims || reinterpret_cast<uintptr_t>(storage.getMemory()) % alignment != 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoE %s has an invalid pointer, shape, dtype or layout", name);
    }
}

std::unique_lock<std::mutex> lockSession(std::mutex& mutex) {
    std::unique_lock<std::mutex> lock(mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Concurrent access to one MoE session is not supported");
    }
    return lock;
}

}

CollectiveContext::CollectiveContext(int32_t rank_value, int32_t world_size_value, FTrainDeviceId device_value,
                                     FTrainByteAllGather allgather_value, void* user_data_value)
    : rank(rank_value), world_size(world_size_value), device(device_value), allgather(allgather_value),
      user_data(user_data_value) {
    if (world_size < 2 || world_size > 8 || rank < 0 || rank >= world_size || device < 0 || allgather == nullptr) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Invalid single-node collective context");
    }
}

bool MoeConfig::isNT() const noexcept { return route.rfind("ag_nt_", 0) == 0 || route == "rs_nt"; }

MoeConfig makeMoeConfig(const CollectiveContext& context, const FTrainAGGroupGemmConfig& config) {
    MoeConfig result{MoeKind::kAG, config.m_dim, config.capacity_m, config.n_dim, config.k_dim, config.num_experts, config.topk,
                     {}, config.route == nullptr ? "ag_nn_ring_chunk" : config.route, 0};
    validateConfig(context, result, config.rank_m_sizes, config.num_rank_m_sizes);
    return result;
}

MoeConfig makeMoeConfig(const CollectiveContext& context, const FTrainMoEReduceRSConfig& config) {
    MoeConfig result{MoeKind::kRS, config.m_dim, config.capacity_m, config.n_dim, config.k_dim, config.num_experts, config.topk,
                     {}, config.route == nullptr ? "rs_nn" : config.route, config.n_chunks};
    validateConfig(context, result, config.rank_m_sizes, config.num_rank_m_sizes);
    return result;
}

std::vector<int64_t> validateMoePartition(const CollectiveContext& context, const MoeConfig& config, int32_t m_dim,
                                         const int64_t* sizes, uint32_t count) {
    if (m_dim <= 0 || m_dim > config.capacity_m || (sizes == nullptr) != (count == 0) ||
        (count != 0 && count != uint32_t(context.world_size))) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Invalid MoE M or rank_m_sizes count");
    }
    if (count == 0 && m_dim % context.world_size != 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Uniform MoE M must be divisible by world_size");
    }
    std::vector<int64_t> result(context.world_size, m_dim / context.world_size);
    if (count != 0) { result.assign(sizes, sizes + count); }
    int64_t total = 0;
    for (const auto size : result) {
        if (size <= 0 || size > m_dim || total > m_dim - size) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoE rank sizes must be positive and sum to M");
        }
        total += size;
    }
    if (total != m_dim) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoE rank sizes must sum to M");
    }
    if (config.route == "ag_nt_a2a_chunk" &&
        (!std::all_of(result.begin(), result.end(), [&](int64_t size) { return size == result.front(); }) ||
         result.front() < 8)) {
        throw Exception(FTRAIN_STATUS_UNSUPPORTED, "ag_nt_a2a_chunk requires equal rank sizes of at least 8 rows");
    }
    return result;
}

MoeTensor::MoeTensor(const FTrainMoeTensor& tensor)
    : storage(tensor.storage), identity(tensor.identity == nullptr ? tensor.storage.memory : tensor.identity),
      version(tensor.version) {}

MoeSession::MoeSession(std::shared_ptr<CollectiveContext> context, MoeConfig config,
                       std::unique_ptr<MoeBackend> backend)
    : context_(std::move(context)), config_(std::move(config)), backend_(std::move(backend)) {
    if (context_ == nullptr || backend_ == nullptr) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoE session requires context and backend");
    }
}

void MoeSession::execute(MoeBindings bindings, FTrainStream stream) {
    auto lock = lockSession(mutex_);
    validateBindings(bindings);
    backend_->execute(bindings, stream);
}

void MoeSession::validateBindings(const MoeBindings& bindings) const {
    const auto local_m = config_.rank_m_sizes[context_->rank];
    validateTensor(bindings.input, {config_.kind == MoeKind::kAG ? local_m : int64_t(config_.m_dim) * config_.topk,
                                    config_.k_dim}, FTRAIN_NUMERIC_TYPE_BF16, "input");
    validateTensor(bindings.weight,
                   {config_.num_experts, config_.isNT() ? config_.n_dim : config_.k_dim,
                    config_.isNT() ? config_.k_dim : config_.n_dim}, FTRAIN_NUMERIC_TYPE_BF16, "weight");
    const auto id_type = bindings.routing.storage.getNumericType();
    if (id_type != FTRAIN_NUMERIC_TYPE_INT32 && id_type != FTRAIN_NUMERIC_TYPE_INT64) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoE routing IDs must be int32 or int64");
    }
    validateTensor(bindings.routing, {config_.m_dim, config_.topk}, id_type, "routing");
    validateTensor(bindings.output, {config_.kind == MoeKind::kAG ? int64_t(config_.m_dim) * config_.topk : local_m,
                                     config_.n_dim}, FTRAIN_NUMERIC_TYPE_BF16, "output");
    if (config_.kind == MoeKind::kRS) {
        if (!bindings.expert_weight.has_value()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "RS requires expert probabilities");
        }
        validateTensor(*bindings.expert_weight, {config_.m_dim, config_.topk}, FTRAIN_NUMERIC_TYPE_BF16,
                       "expert_weight");
    }
}

void MoeSession::setRankMSizes(int32_t m_dim, const int64_t* sizes, uint32_t count) {
    auto lock = lockSession(mutex_);
    auto partition = validateMoePartition(*context_, config_, m_dim, sizes, count);
    backend_->setRankMSizes(m_dim, partition);
    config_.m_dim = m_dim;
    config_.rank_m_sizes = std::move(partition);
}

void MoeSession::synchronize() {
    auto lock = lockSession(mutex_);
    backend_->synchronize();
}

}
