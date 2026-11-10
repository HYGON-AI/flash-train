#include "flash_train/primitive/hygon/moe/backend.hpp"

#include <cstring>
#include <mutex>
#include <utility>

#include "flash_train/error.hpp"

#if FTRAIN_HAS_MOE
#include <moe_core/ag_group_gemm.h>
#include <moe_core/moe_reduce_rs.h>
#endif

namespace ftrain {

bool isMoeBackendEnabled() noexcept { return FTRAIN_HAS_MOE != 0; }

#if FTRAIN_HAS_MOE
namespace {

namespace core = ftrain::moe::core;
using ftrain::moe::ScalarType;

std::mutex& moeDispatchMutex() {
    static std::mutex mutex;
    return mutex;
}

void checkHip(hipError_t status) {
    if (status != hipSuccess) {
        throw Exception(status == hipErrorOutOfMemory ? FTRAIN_STATUS_OUT_OF_MEMORY : FTRAIN_STATUS_INTERNAL_ERROR,
                        "MoE HIP error: %s", hipGetErrorString(status));
    }
}

void checkDevice(FTrainDeviceId device) {
    int current_device = -1;
    checkHip(hipGetDevice(&current_device));
    if (current_device != device) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoE session device %d differs from current device %d",
                        device, current_device);
    }
}

core::Tensor borrowTensor(const MoeTensor& tensor, const std::shared_ptr<core::Context>& context) {
    const auto type = tensor.storage.getNumericType();
    const auto dtype = type == FTRAIN_NUMERIC_TYPE_INT32 ? ScalarType::Int :
                       type == FTRAIN_NUMERIC_TYPE_INT64 ? ScalarType::Long : ScalarType::BFloat16;
    return core::Tensor::from_blob(tensor.storage.getMemory(), tensor.storage.getDims(),
                                     context->tensor_options().dtype(dtype), {}, tensor.identity, tensor.version);
}

class FTrainMoeBackend final : public MoeBackend {
  public:
    FTrainMoeBackend(std::shared_ptr<CollectiveContext> context, const MoeConfig& config, FTrainStream stream)
        : context_(std::move(context)) {
        checkDevice(context_->device);
        hipDeviceProp_t properties{};
        checkHip(hipGetDeviceProperties(&properties, context_->device));
        if (std::strncmp(properties.gcnArchName, "gfx936", 6) != 0) {
            throw Exception(FTRAIN_STATUS_UNSUPPORTED, "Hygon MoE backend currently supports gfx936 only");
        }
        std::vector<int64_t> signature{int64_t(config.kind), context_->world_size, config.m_dim, config.capacity_m,
                                       config.n_dim, config.k_dim, config.num_experts, config.topk, config.n_chunks};
        signature.insert(signature.end(), config.rank_m_sizes.begin(), config.rank_m_sizes.end());
        signature.resize(32, 0);
        std::memcpy(signature.data() + 20, config.route.data(), config.route.size());
        std::vector<int64_t> signatures(signature.size() * context_->world_size);
        const auto bootstrap_status = context_->allgather(context_->user_data, signature.data(), signatures.data(),
                                                          signature.size() * sizeof(int64_t));
        if (bootstrap_status != FTRAIN_STATUS_SUCCESS) {
            throw Exception(bootstrap_status, "MoE configuration allgather failed");
        }
        for (int peer = 0; peer < context_->world_size; ++peer) {
            if (std::memcmp(signature.data(), signatures.data() + peer * signature.size(),
                            signature.size() * sizeof(int64_t)) != 0) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoE configuration differs between ranks");
            }
        }
        execution_context_ = std::make_shared<core::Context>(
            context_->rank, context_->world_size, context_->device,
            [context = context_](const void* send, void* receive, size_t bytes) {
                const auto status = context->allgather(context->user_data, send, receive, bytes);
                if (status != FTRAIN_STATUS_SUCCESS) {
                    throw Exception(status, "MoE byte-allgather bootstrap failed");
                }
            });
        execution_context_->set_stream(stream);
        std::lock_guard<std::mutex> lock(moeDispatchMutex());
        if (config.kind == MoeKind::kAG) {
            ag_ = std::make_unique<core::AGGroupGemm>(execution_context_, 1, config.m_dim, config.n_dim, config.k_dim,
                                                       config.num_experts, config.topk, ScalarType::BFloat16,
                                                       ScalarType::BFloat16, false, config.rank_m_sizes,
                                                       config.capacity_m, config.route);
        } else {
            rs_ = std::make_unique<core::MoEReduceRS>(execution_context_, 1, config.capacity_m, config.n_dim, config.k_dim,
                                                       config.num_experts, config.topk, ScalarType::BFloat16,
                                                       config.n_chunks, "a2a_ragged_signal", config.route);
            rs_->set_rank_m_sizes(config.m_dim, config.rank_m_sizes);
        }
        checkHip(hipEventCreateWithFlags(&completion_, hipEventDisableTiming));
    }

    ~FTrainMoeBackend() override {
        if (completion_ != nullptr) {
            if (pending_) { static_cast<void>(hipEventSynchronize(completion_)); }
            static_cast<void>(hipEventDestroy(completion_));
        }
    }

    void execute(const MoeBindings& bindings, FTrainStream stream) override {
        checkDevice(context_->device);
        auto input = borrowTensor(bindings.input, execution_context_);
        auto weight = borrowTensor(bindings.weight, execution_context_);
        auto routing = borrowTensor(bindings.routing, execution_context_);
        auto output = borrowTensor(bindings.output, execution_context_);
        std::lock_guard<std::mutex> lock(moeDispatchMutex());
        if (pending_ && stream != last_stream_) { checkHip(hipStreamWaitEvent(stream, completion_, 0)); }
        execution_context_->set_stream(stream);
        last_stream_ = stream;
        try {
            if (ag_ != nullptr) {
                ag_->forward(std::move(input), std::move(weight), std::move(routing), std::move(output));
            } else {
                rs_->run_a2a(std::move(input), std::move(weight), std::move(routing),
                              borrowTensor(*bindings.expert_weight, execution_context_), std::move(output));
            }
            checkHip(hipEventRecord(completion_, stream));
            pending_ = true;
        } catch (...) {
            static_cast<void>(hipDeviceSynchronize());
            throw;
        }
    }

    void setRankMSizes(int32_t m_dim, const std::vector<int64_t>& sizes) override {
        synchronize();
        std::lock_guard<std::mutex> lock(moeDispatchMutex());
        std::vector<int64_t> signature{m_dim};
        signature.insert(signature.end(), sizes.begin(), sizes.end());
        std::vector<int64_t> signatures(signature.size() * context_->world_size);
        execution_context_->allgather(signature.data(), signatures.data(), signature.size() * sizeof(int64_t));
        for (int peer = 0; peer < context_->world_size; ++peer) {
            if (std::memcmp(signature.data(), signatures.data() + peer * signature.size(),
                            signature.size() * sizeof(int64_t)) != 0) {
                throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoE rank partitions differ between ranks");
            }
        }
        if (ag_ != nullptr) {
            ag_->set_rank_m_sizes(m_dim, sizes);
        } else {
            rs_->set_rank_m_sizes(m_dim, sizes);
        }
        execution_context_->barrier();
    }

    void synchronize() override {
        checkDevice(context_->device);
        if (pending_) {
            checkHip(hipEventSynchronize(completion_));
            pending_ = false;
        }
    }

  private:
    std::shared_ptr<CollectiveContext> context_;
    std::shared_ptr<core::Context> execution_context_;
    std::unique_ptr<core::AGGroupGemm> ag_;
    std::unique_ptr<core::MoEReduceRS> rs_;
    hipEvent_t completion_ = nullptr;
    FTrainStream last_stream_ = nullptr;
    bool pending_ = false;
};

}
#endif

std::unique_ptr<MoeBackend> makeMoeBackend(std::shared_ptr<CollectiveContext> context,
                                             const MoeConfig& config, FTrainStream stream) {
#if FTRAIN_HAS_MOE
    return std::make_unique<FTrainMoeBackend>(std::move(context), config, stream);
#else
    static_cast<void>(context);
    static_cast<void>(config);
    static_cast<void>(stream);
    throw Exception(FTRAIN_STATUS_UNSUPPORTED, "Build flash-train with FTRAIN_ENABLE_MOE=ON");
#endif
}

}
